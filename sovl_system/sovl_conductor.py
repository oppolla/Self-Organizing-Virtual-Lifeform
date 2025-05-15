import time
import traceback
from typing import Optional, Dict, Any, TYPE_CHECKING, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from threading import Lock
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_state import StateManager, StateTracker
from sovl_error import ErrorManager
from sovl_utils import  detect_repetitions
from collections import deque
from sovl_interfaces import OrchestratorInterface, SystemInterface, SystemMediator
import random
from sovl_main import SOVLSystem, SystemContext
from sovl_curiosity import CuriosityManager
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_logger import Logger
from sovl_manager import ModelManager
from sovl_monitor import SystemMonitor, MemoryMonitor, TraitsMonitor
from sovl_trainer import TrainingWorkflowManager
import threading
from sovl_resource import ResourceManager
from sovl_api import SOVLAPI
from sovl_queue import ScribeQueue
from sovl_bonder import BondCalculator
from sovl_confidence import ConfidenceCalculator
from sovl_temperament import TemperamentSystem
import sys

if TYPE_CHECKING:
    from sovl_main import SOVLSystem

class SOVLOrchestrator(OrchestratorInterface):
    """
    Orchestrates the initialization, execution, and shutdown of the SOVL system.

    Responsible for setting up core components (ConfigManager, SOVLSystem),
    selecting execution modes (e.g., CLI), and ensuring clean shutdown with
    state saving and resource cleanup.
    """
    # Constants for configuration
    DEFAULT_CONFIG_PATH: str = "sovl_config.json"
    DEFAULT_LOG_FILE: str = "sovl_orchestrator_logs.jsonl"
    LOG_MAX_SIZE_MB: int = 10
    SAVE_PATH_SUFFIX: str = "_final.json"

    # Canonical, ordered list of core components to initialize
    COMPONENT_INIT_LIST = [
        # Base infrastructure components with no dependencies
        ("config_manager", "sovl_config", "ConfigManager", {}),
        ("logger", "sovl_logger", "Logger", {}),
        ("resource_manager", "sovl_resource", "ResourceManager", {}),
        ("state_manager", "sovl_state", "StateManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "device": {"key": "device", "required": False}
        }),
        ("error_manager", "sovl_error", "ErrorManager", {
            "state_manager": {"key": "state_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }),

        # Core system components with minimal dependencies
        ("ram_manager", "sovl_memory", "RAMManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }),
        ("gpu_manager", "sovl_memory", "GPUMemoryManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }),
        ("event_dispatcher", "sovl_events", "EventDispatcher", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": False}
        }),

        # Basic data and resource management
        ("jsonl_loader", "sovl_io", "JSONLLoader", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }),
        ("scribe_queue", "sovl_queue", "ScribeQueue", {
            "logger": {"key": "logger", "required": False}
        }),
        ("hardware_manager", "sovl_hardware", "HardwareManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": False}
        }),
        ("data_manager", "sovl_data", "DataManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": False}
        }),

        # Model and core processing
        ("model_manager", "sovl_manager", "ModelManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "device": {"key": "device", "required": False}
        }),
        ("metadata_processor", "sovl_processor", "MetadataProcessor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_accessor": {"key": "state_manager", "required": False}
        }),
        ("lora_adapter_manager", "sovl_engram", "LoraAdapterManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_handler": {"key": "error_manager", "required": False}
        }),
        ("scaffold_provider", "sovl_scaffold", "ScaffoldProvider", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_handler": {"key": "error_manager", "required": False},
            "ram_manager": {"key": "ram_manager", "required": True},
            "gpu_manager": {"key": "gpu_manager", "required": True}
        }),
        ("cross_attention_injector", "sovl_scaffold", "CrossAttentionInjector", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": False},
            "ram_manager": {"key": "ram_manager", "required": False},
            "gpu_manager": {"key": "gpu_manager", "required": False}
        }),
        ("scaffold_token_mapper", "sovl_scaffold", "ScaffoldTokenMapper", {
            "base_tokenizer": {"key": "model_manager", "required": True},
            "scaffold_tokenizer": {"key": "model_manager", "required": True},
            "logger": {"key": "logger", "required": False},
            "config": {"key": "config_manager", "required": True},
            "base_model": {"key": "model_manager", "required": True},
            "scaffold_model": {"key": "model_manager", "required": True},
            "ram_manager": {"key": "ram_manager", "required": False},
            "gpu_manager": {"key": "gpu_manager", "required": False},
            "provider": {"key": "scaffold_provider", "required": True}
        }),

        # Monitoring systems
        ("system_monitor", "sovl_monitor", "SystemMonitor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "ram_manager": {"key": "ram_manager", "required": True},
            "gpu_manager": {"key": "gpu_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "user_profile_state": {"key": "state_manager", "required": False},
            "bond_calculator": {"key": "bond_calculator", "required": False}
        }),
        ("memory_monitor", "sovl_monitor", "MemoryMonitor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "ram_manager": {"key": "ram_manager", "required": True},
            "gpu_manager": {"key": "gpu_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }),

        # Auxiliary Managers (Moved here for earlier initialization)
        ("lifecycle_manager", "sovl_trainer", "LifecycleManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }),
        ("scaffold_manager", "sovl_scaffold", "ScaffoldManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }),

        # System foundation components
        ("system_context", "sovl_main", "SystemContext", {
            "config_path": {"key": "config_manager", "required": True}
        }),
        ("system_mediator", "sovl_interfaces", "SystemMediator", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "device": {"key": "device", "required": True}
        }),

        # Core interaction components
        ("generation_manager", "sovl_generation", "GenerationManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "base_model": {"key": "model_manager", "required": True},
            "base_tokenizer": {"key": "model_manager", "required": True},
            "state": {"key": "state_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "device": {"key": "device", "required": True},
            "dialogue_context_manager": {"key": "state_manager", "required": False},
            "state_manager": {"key": "state_manager", "required": True},
            "resource_manager": {"key": "resource_manager", "required": False},
            "model_manager": {"key": "model_manager", "required": False}
        }),
        ("confidence_calculator", "sovl_confidence", "ConfidenceCalculator", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }),
        ("bond_calculator", "sovl_bonder", "BondCalculator", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "user_profile_state": {"key": "state_manager", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }),
        ("bond_modulator", "sovl_bonder", "BondModulator", {
            "bond_calculator": {"key": "bond_calculator", "required": True},
            "max_retries": {"key": "max_retries", "required": False}
        }),

        # Advanced personality and behavior systems
        ("temperament_system", "sovl_temperament", "TemperamentSystem", {
            "state_manager": {"key": "state_manager", "required": True},
            "config_manager": {"key": "config_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }),
        ("curiosity_manager", "sovl_curiosity", "CuriosityManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "device": {"key": "device", "required": False},
            "generation_manager": {"key": "generation_manager", "required": False},
            "state_manager": {"key": "state_manager", "required": True}
        }),
        ("traits_monitor", "sovl_monitor", "TraitsMonitor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_tracker": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }),
        ("vibe_sculptor", "sovl_viber", "VibeSculptor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "temperament_system": {"key": "temperament_system", "required": False}
        }),

        # Creative and reflective systems
        ("dreamer", "sovl_dreamer", "Dreamer", {
            "config_manager": {"key": "config_manager", "required": True},
            "scribe_path": {"key": "config_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }),
        ("aspiration_system", "sovl_striver", "AspirationSystem", {
            "config": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "long_term_memory": {"key": "state_manager._current_state.dialogue_context", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }),
        ("introspection_manager", "sovl_meditater", "IntrospectionManager", {
            "context": {"key": "system_context", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "curiosity_manager": {"key": "curiosity_manager", "required": False},
            "confidence_calculator": {"key": "confidence_calculator", "required": False},
            "temperament_system": {"key": "temperament_system", "required": False},
            "model_manager": {"key": "model_manager", "required": False},
            "dialogue_context_manager": {"key": "state_manager", "required": False},
            "bond_calculator": {"key": "bond_calculator", "required": False}
        }),
        ("shamer", "sovl_shamer", "Shamer", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "viber": {"key": "vibe_sculptor", "required": False},
            "dialogue_context_manager": {"key": "state_manager", "required": False}
        }),

        # Dialogue Context Management (Moved and Corrected)
        ("dialogue_shutdown", "sovl_recaller", "DialogueShutdown", {
            "dialogue_context_manager": {"key": "state_manager._current_state.dialogue_context", "required": True},
            "long_term_memory": {"key": "state_manager._current_state.dialogue_context.ltm", "required": False},
            "logger": {"key": "logger", "required": True}
        }),

        # Training and process systems
        ("training_manager", "sovl_trainer", "TrainingWorkflowManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "model_manager": {"key": "model_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }),
        ("scribe_ingestion_processor", "sovl_processor", "ScribeIngestionProcessor", {
            "log_paths": {"key": "scribe_log_paths", "required": False},
            "memory_templates": {"key": "memory_templates", "required": False},
            "logger": {"key": "logger", "required": True},
            "config_path": {"key": "config_manager", "required": True}
        }),

        # Generation preparation and interface components
        ("generation_primer", "sovl_primer", "GenerationPrimer", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "curiosity_manager": {"key": "curiosity_manager", "required": True},
            "temperament_system": {"key": "temperament_system", "required": False},
            "confidence_calculator": {"key": "confidence_calculator", "required": False},
            "bond_calculator": {"key": "bond_calculator", "required": False},
            "bond_modulator": {"key": "bond_modulator", "required": False},
            "dialogue_context_manager": {"key": "state_manager", "required": False},
            "device": {"key": "device", "required": False},
            "lifecycle_manager": {"key": "lifecycle_manager", "required": False},
            "scaffold_manager": {"key": "scaffold_manager", "required": False}
        }),

        # Main system and API components (depend on almost everything)
        ("sovl_system", "sovl_main", "SOVLSystem", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "generation_primer": {"key": "generation_primer", "required": True},
            "dialogue_context_manager": {"key": "state_manager", "required": False},
            "device": {"key": "device", "required": False},
            "lifecycle_manager": {"key": "lifecycle_manager", "required": False},
            "scaffold_manager": {"key": "scaffold_manager", "required": False}
        }),
        ("api", "sovl_api", "SOVLAPI", {
            "config_path": {"key": "config_manager", "required": True}
        }),

        # CLI handler must be last as it depends on sovl_system which is created late
        ("cli_handler", "sovl_cli", "CommandHandler", {
            "sovl_system": {"key": "sovl_system", "required": True}
        }),
    ]

    @staticmethod
    def select_device(min_vram_gb=4):
        """Select a device with robust CUDA/CPU fallback and VRAM check."""
        try:
            if torch.cuda.is_available():
                try:
                    device = torch.device("cuda")
                    props = torch.cuda.get_device_properties(device)
                    total_vram_gb = props.total_memory / (1024 ** 3)
                    if total_vram_gb < min_vram_gb:
                        print(f"[WARNING] CUDA device has only {total_vram_gb:.2f}GB VRAM, required: {min_vram_gb}GB. Falling back to CPU.", flush=True)
                        return torch.device("cpu")
                    print(f"[INFO] Using CUDA device: {props.name}, VRAM: {total_vram_gb:.2f}GB", flush=True)
                    return device
                except Exception as e:
                    print(f"[WARNING] CUDA device check failed: {e}. Falling back to CPU.", flush=True)
                    return torch.device("cpu")
            else:
                print("[INFO] CUDA not available, using CPU.", flush=True)
                return torch.device("cpu")
        except Exception as e:
            print(f"[ERROR] Device selection failed: {e}. Defaulting to CPU.", flush=True)
            return torch.device("cpu")

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, log_file: str = DEFAULT_LOG_FILE) -> None:
        """
        Initialize the orchestrator with configuration and logging.

        Args:
            config_path: Path to the configuration file.
            log_file: Path to the orchestrator's log file.

        Raises:
            RuntimeError: If initialization of ConfigManager or SOVLSystem fails.
        """
        self._lock = threading.RLock()
        self.components = {}
        if not os.path.isfile(config_path):
            msg = f"Config file not found: {config_path}"
            print(f"[ERROR] {msg}", flush=True)
            raise RuntimeError(msg)
        self._initialize_logger(log_file)
        self._log_event("orchestrator_init_start", {"config_path": config_path})
        # Robust device selection
        self.device = self.select_device(min_vram_gb=4)
        self._log_event("device_initialized", {"device": str(self.device)})
        # Log GPU memory if using CUDA
        if hasattr(self.device, "type") and self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
                print(f"[INFO] GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, total={total:.2f}GB", flush=True)
                if allocated / total > 0.8:
                    print("[WARNING] GPU memory usage above 80%!", flush=True)
            except Exception as e:
                print(f"[WARNING] Could not log GPU memory: {e}", flush=True)
        # --- Enhanced Initialization Output ---
        INIT_GROUPS = [
            ("Core Infrastructure", ["config_manager", "logger", "resource_manager", "state_manager", "error_manager"]),
            ("State & Memory Managers", ["ram_manager", "gpu_manager"]),
            ("Event & Data Systems", ["event_dispatcher", "jsonl_loader", "scribe_queue", "hardware_manager", "data_manager"]),
            ("Model & Processing", ["model_manager", "metadata_processor", "lora_adapter_manager", "scaffold_provider", "cross_attention_injector", "scaffold_token_mapper"]),
            ("Monitors", ["system_monitor", "memory_monitor", "traits_monitor"]),
            ("System Context & Mediation", ["system_context", "system_mediator"]),
            ("Generation & Calculation", ["generation_manager", "confidence_calculator", "bond_calculator"]),
            ("Personality & Behavior", ["temperament_system", "curiosity_manager", "vibe_sculptor"]),
            ("Creative & Reflective", ["dreamer", "aspiration_system", "introspection_manager", "shamer"]),
            ("Training & Ingestion", ["training_manager", "scribe_ingestion_processor"]),
            ("Generation Prep & Interface", ["generation_primer"]),
            ("Main System & API", ["sovl_system", "api"]),
            ("CLI Handler", ["cli_handler"]),
        ]
        key_to_index = {key: i for i, (key, *_rest) in enumerate(self.COMPONENT_INIT_LIST)}
        total = len(self.COMPONENT_INIT_LIST)
        initialized = []
        failed = []
        current = 1
        for group_name, group_keys in INIT_GROUPS:
            print(f"[SOVL System] — {group_name} —", flush=True)
            for key in group_keys:
                idx = key_to_index.get(key)
                if idx is None:
                    continue
                _key, module_name, class_name, dep_map = self.COMPONENT_INIT_LIST[idx]
                msg = f"[{current}/{total}]  Incarnating {key} ({class_name}) from {module_name} ... "
                print(f"[SOVL System] {msg}", end="", flush=True)
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)
                    kwargs = {}
                    for arg, dep_info in dep_map.items():
                        # Backward compatibility: allow string for required dependencies
                        if isinstance(dep_info, str):
                            dep_key = dep_info
                            required = True
                        else:
                            dep_key = dep_info["key"]
                            required = dep_info.get("required", True)
                        dep = None
                        if dep_key == "device":
                            dep = self.device
                        elif "." in dep_key:
                            # Support nested attribute access (e.g., dialogue_context_manager.long_term)
                            obj = self
                            for part in dep_key.split("."):
                                if hasattr(obj, part):
                                    obj = getattr(obj, part)
                                else:
                                    obj = None
                                    break
                            dep = obj
                        elif dep_key in self.components:
                            dep = self.components[dep_key]
                        elif hasattr(self, dep_key):
                            dep = getattr(self, dep_key)
                        if dep is None and required:
                            raise RuntimeError(f"Required dependency {dep_key} for {key} is None")
                        elif dep is None and not required:
                            print(f"[WARNING] Optional dependency {dep_key} for {key} is None, using fallback.", flush=True)
                        kwargs[arg] = dep
                    if key == "config_manager":
                        kwargs = {"config_path": config_path, "logger": self.logger}
                    if key == "logger":
                        kwargs = {"log_file": log_file, "max_size_mb": 10, "rotation_interval": "1d"}
                    self.components[key] = cls(**kwargs)
                    setattr(self, key, self.components[key])
                    print("✓", flush=True)
                    initialized.append(key)
                except Exception as e:
                    print(f"✗ (Error: {e})", flush=True)
                    self._log_error(f"Failed to initialize {key}", e)
                    self.components[key] = None
                    setattr(self, key, None)
                    failed.append((key, str(e)))
                    # Halt initialization immediately if any component fails
                    print("[SOVL System] Incarnation halted due to critical failure.", flush=True)
                    self._log_event("initialization_halted", {"failed_component": key, "error": str(e)})
                    # Print/log summary before raising
                    print("[SOVL System] Incarnation Summary:", flush=True)
                    print(f"  Succeeded: {initialized}", flush=True)
                    print(f"  Failed: {failed}", flush=True)
                    self._log_event("initialization_summary", {"succeeded": initialized, "failed": failed})
                    raise RuntimeError(f"Critical component {key} failed to initialize: {e}")
                current += 1
            print("", flush=True)  # Blank line between groups
        # Print/log summary at the end if all succeeded
        print("[SOVL System] Incarnation Summary:", flush=True)
        print(f"  Succeeded: {initialized}", flush=True)
        print(f"  Failed: {failed}", flush=True)
        self._log_event("initialization_summary", {"succeeded": initialized, "failed": failed})
        if not failed:
            print("[SOVL System] All components initialized. Awakening system...", flush=True)
        else:
            print(f"[SOVL System] Incarnation failed for {len(failed)} component(s):", flush=True)
            for key, err in failed:
                print(f"    - {key}: {err}", flush=True)
        print(f"[SOVL System] Log file: {self.logger.log_file}", flush=True)
        # Continue with any additional initialization as before
        self.post_initialize_components()
        try:
            self._initialize_config()
        except Exception as e:
            self._log_error("Failed to initialize configuration", e)
            raise

    def post_initialize_components(self):
        """Wire up references that require all components to be initialized."""
        if hasattr(self, "generation_primer") and hasattr(self, "sovl_system") and self.generation_primer and self.sovl_system:
            self.generation_primer.sovl_system = self.sovl_system
            self._log_event("generation_primer_sovl_system_linked")

    def _initialize_config(self) -> None:
        """Initialize and validate configuration from ConfigManager."""
        try:
            # Load orchestrator configuration
            orchestrator_config = self.config_manager.get_section("orchestrator_config")
            
            # Set configuration parameters with validation
            self.log_max_size_mb = int(orchestrator_config.get("log_max_size_mb", self.LOG_MAX_SIZE_MB))
            self.save_path_suffix = str(orchestrator_config.get("save_path_suffix", self.SAVE_PATH_SUFFIX))
            
            # Validate configuration values
            self._validate_config_values()
            
            # Subscribe to configuration changes
            self.config_manager.subscribe(self._on_config_change)
            
            self.logger.record_event(
                event_type="orchestrator_config_initialized",
                message="Orchestrator configuration initialized successfully",
                level="info",
                additional_info={
                    "log_max_size_mb": self.log_max_size_mb,
                    "save_path_suffix": self.save_path_suffix
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="orchestrator_config_initialization_failed",
                message=f"Failed to initialize orchestrator configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _validate_config_values(self) -> None:
        """Validate configuration values against defined ranges."""
        try:
            # Validate log size
            if not 1 <= self.log_max_size_mb <= 100:
                raise ValueError(f"Invalid log_max_size_mb: {self.log_max_size_mb}. Must be between 1 and 100.")
                
            # Validate save path suffix
            if not self.save_path_suffix.startswith("_"):
                raise ValueError(f"Invalid save_path_suffix: {self.save_path_suffix}. Must start with '_'.")
                
        except Exception as e:
            self.logger.record_event(
                event_type="orchestrator_config_validation_failed",
                message=f"Configuration validation failed: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            self._initialize_config()
            self.logger.record_event(
                event_type="orchestrator_config_updated",
                message="Orchestrator configuration updated",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                event_type="orchestrator_config_update_failed",
                message=f"Failed to update orchestrator configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )

    def _create_config_manager(self, config_path: str) -> ConfigManager:
        """Create and initialize the configuration manager with validation."""
        try:
            config_manager = ConfigManager(config_path, self.logger)
            
            # Validate required configuration sections
            required_sections = [
                "core_config",
                "training_config",
                "curiosity_config",
                "cross_attn_config",
                "controls_config",
                "lora_config",
                "orchestrator_config"
            ]
            
            missing_sections = [section for section in required_sections 
                              if not config_manager.has_section(section)]
            
            if missing_sections:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Missing required configuration sections",
                        "missing_sections": missing_sections
                    },
                    level="warning"
                )
                # Create missing sections with default values
                for section in missing_sections:
                    config_manager.add_section(section, {})
            
            return config_manager
        except Exception as e:
            self._log_error("Config manager creation failed", e)
            raise

    def _validate_config_sections(self) -> None:
        """Validate configuration sections for consistency."""
        try:
            # Get configuration sections
            orchestrator_config = self.config_manager.get_section("orchestrator_config")
            controls_config = self.config_manager.get_section("controls_config")
            
            # Define required keys and their default values
            required_keys = {
                "log_max_size_mb": self.LOG_MAX_SIZE_MB,
                "save_path_suffix": self.SAVE_PATH_SUFFIX,
                "enable_logging": True,
                "enable_state_saving": True,
                "state_save_interval": 300,
                "max_backup_files": 5
            }
            
            # Check for missing keys in orchestrator_config
            missing_keys = [key for key in required_keys if key not in orchestrator_config]
            if missing_keys:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Missing keys in orchestrator_config",
                        "missing_keys": missing_keys,
                        "default_values": {k: required_keys[k] for k in missing_keys}
                    },
                    level="warning"
                )
                # Add missing keys with default values
                for key in missing_keys:
                    orchestrator_config[key] = required_keys[key]
            
            # Validate state save interval
            state_save_interval = int(orchestrator_config.get("state_save_interval", 300))
            if not 60 <= state_save_interval <= 3600:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Invalid state_save_interval",
                        "value": state_save_interval,
                        "valid_range": [60, 3600]
                    },
                    level="warning"
                )
                orchestrator_config["state_save_interval"] = 300
            
            # Validate max backup files
            max_backup_files = int(orchestrator_config.get("max_backup_files", 5))
            if not 1 <= max_backup_files <= 10:
                self._log_event(
                    "config_validation",
                    {
                        "message": "Invalid max_backup_files",
                        "value": max_backup_files,
                        "valid_range": [1, 10]
                    },
                    level="warning"
                )
                orchestrator_config["max_backup_files"] = 5
            
            # Log final configuration state
            self._log_event(
                "config_validation",
                {
                    "message": "Configuration validation complete",
                    "orchestrator_config": orchestrator_config,
                    "controls_config": {k: v for k, v in controls_config.items() 
                                     if k.startswith("orchestrator_")}
                },
                level="info"
            )
            
        except Exception as e:
            self._log_error("Configuration validation failed", e)
            raise

    def _initialize_logger(self, log_file: str) -> None:
        """Initialize the logger with Logger."""
        try:
            self.logger = Logger(
                log_file=log_file,
                max_size_mb=10,
                rotation_interval="1d"
            )
            self._log_event("logger_initialized", {"log_file": log_file})
        except Exception as e:
            self.logger.log_error(
                error_msg="Failed to initialize logger",
                error_type="logger_init_error",
                stack_trace=traceback.format_exc(),
                additional_info={"log_file": log_file}
            )
            raise

    def _log_event(self, event_type: str, additional_info: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
        """Log an event with standardized metadata."""
        try:
            metadata = {
                "conversation_id": getattr(self.state, 'history', {}).get('conversation_id', None),
                "state_hash": getattr(self.state, 'state_hash', None),
                "device": str(self.device) if hasattr(self, 'device') else None
            }
            if additional_info:
                metadata.update(additional_info)
            
            self.logger.record_event(
                event_type=event_type,
                message=f"Orchestrator event: {event_type}",
                level=level,
                additional_info=metadata
            )
        except Exception as e:
            self.logger.log_error(
                error_msg="Failed to log event",
                error_type="logging_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "event_type": event_type,
                    "original_error": str(e)
                }
            )

    def _log_error(self, message: str, error: Exception) -> None:
        """Log an error with standardized metadata and stack trace."""
        try:
            metadata = {
                "conversation_id": getattr(self.state, 'history', {}).get('conversation_id', None),
                "state_hash": getattr(self.state, 'state_hash', None),
                "device": str(self.device) if hasattr(self, 'device') else None,
                "error": str(error)
            }
            
            self.logger.log_error(
                error_msg=message,
                error_type="orchestrator_error",
                stack_trace=traceback.format_exc(),
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")  # Fallback to print if logger fails

    def set_system(self, system: SystemInterface) -> None:
        """Set the system instance for orchestration."""
        with self._lock:
            self._system = system
            self._log_event("system_set", {
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash
            })

    @property
    def state(self):
        """Always access the canonical StateManager via StateManager."""
        return getattr(self.state_manager, '_system_state', None)

    @state.setter
    def state(self, value):
        self.state_manager._system_state = value

    def sync_state(self) -> None:
        """
        Synchronize orchestrator state with the system state using atomic update.
        
        This method ensures atomic updates to the state with:
        - Thread safety using locks
        - Session ID validation to prevent desynchronization
        - State validation to ensure consistency
        - Fallback to last known good state in case of failure
        - Emergency state saving for recovery
        """
        with self._lock:
            if not self._system:
                self._log_event("state_sync_skipped", {"reason": "System not initialized"})
                return
            # Save a copy of the current state for recovery
            last_state_hash = self.state.state_hash if hasattr(self.state, 'state_hash') else None
            last_conversation_id = self.state.history.conversation_id if hasattr(self.state, 'history') else None
            try:
                # Retrieve latest system state
                system_state = self._system.get_state()
                # Verify session ID matches to prevent desynchronization
                if hasattr(system_state, 'session_id') and hasattr(self.state, 'session_id'):
                    if system_state.session_id != self.state.session_id:
                        raise ValueError(f"Session ID mismatch: orchestrator {self.state.session_id} vs system {system_state.session_id}")
                # Perform the atomic update
                def update_fn(state):
                    # Clone the state to avoid direct modification
                    state.from_dict(system_state, self.device)
                    # Additional validation can be added here if needed
                    return state
                self.state_manager.update_state_atomic(update_fn)
                # Log successful synchronization
                self._log_event("state_synchronized", {
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash,
                    "state_version": getattr(self.state, 'state_version', 'unknown')
                })
            except Exception as e:
                self._log_error("State synchronization failed", e)
                # Attempt recovery with last known good state (inside lock)
                try:
                    if last_state_hash:
                        self._log_event("state_sync_recovery_attempt", {
                            "last_hash": last_state_hash,
                            "last_conversation_id": last_conversation_id
                        })
                        # Try to revert to the last known good state
                        def recovery_fn(state):
                            # Only restore essential fields to maintain system stability
                            if hasattr(self.state, 'history'):
                                state.history = self.state.history
                            if hasattr(self.state, 'session_id'):
                                state.session_id = self.state.session_id
                            return state
                        self.state_manager.update_state_atomic(recovery_fn)
                        self._log_event("state_sync_recovery_succeeded", {
                            "restored_hash": self.state.state_hash
                        })
                    else:
                        self._log_event("state_sync_recovery_failed", {
                            "reason": "No previous state hash available for recovery"
                        })
                except Exception as recovery_e:
                    self._log_error("State recovery failed after sync error", recovery_e)
                    # Emergency state save
                    try:
                        emergency_path = f"emergency_state_{int(time.time())}.json"
                        if hasattr(self, 'state_manager') and self.state_manager and self.state:
                            self.state_manager.save_state(self.state, emergency_path)
                            self._log_event("emergency_state_saved", {"path": emergency_path})
                    except Exception as save_e:
                        self._log_error("Emergency state save failed", save_e)
                # Raise the original error after recovery attempts
                raise RuntimeError("Failed to synchronize state") from e

    def initialize_system(self) -> None:
        """Initialize the SOVL system with the current configuration."""
        try:
            # Create mediator
            self.mediator = SystemMediator(
                config_manager=self.config_manager,
                logger=self.logger,
                device=self.device
            )
            
            # Register orchestrator with mediator
            self.mediator.register_orchestrator(self)
            
            # Create system components
            context = SystemContext(self.config_manager.config_path)
            state_tracker = StateTracker(context)
            error_manager = ErrorManager(context, state_tracker)
            memory_monitor = MemoryMonitor(context)
            
            # Initialize curiosity manager
            curiosity_manager = CuriosityManager(
                config_manager=self.config_manager,
                logger=self.logger,
                error_manager=error_manager,
                device=self.device,
                state_manager=state_tracker
            )
            
            # Create and register system
            bond_calculator = BondCalculator(
                config_manager=self.config_manager,
                logger=self.logger,
                user_profile_state=state_tracker,
                state_manager=state_tracker
            )
            confidence_calculator = ConfidenceCalculator(
                config_manager=self.config_manager,
                logger=self.logger,
                state_manager=state_tracker
            )
            temperament_system = TemperamentSystem(
                state_manager=state_tracker,
                config_manager=self.config_manager,
                error_manager=error_manager
            )
            system = SOVLSystem(
                context=context,
                config_handler=self.config_manager,
                model_manager=self.model_manager,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager,
            )
            system.orchestrator = self
            
            self.mediator.register_system(system)
            
            # Load state from file if exists, otherwise initialize new state
            try:
                loaded_state = self.state_manager.load_state(self.config_manager.config_path)
                # Validate loaded state
                if not self.state_manager.validate_state(loaded_state.to_dict() if hasattr(loaded_state, 'to_dict') else loaded_state):
                    self.logger.record_event(
                        event_type="state_validation_failed",
                        message="Loaded state is invalid. Initializing default state.",
                        level="warning"
                    )
                    self.state_manager._initialize_state()  # Re-initialize default state
                    loaded_state = self.state_manager._current_state
                    self.state_manager.save_state(loaded_state, self.config_manager.config_path)
                self.state = loaded_state
            except Exception as e:
                self.logger.record_event(
                    event_type="state_load_failed",
                    message=f"Failed to load state: {str(e)}. Initializing default state.",
                    level="warning"
                )
                self.state_manager._initialize_state()
                loaded_state = self.state_manager._current_state
                self.state_manager.save_state(loaded_state, self.config_manager.config_path)
                self.state = loaded_state
            
            # Generate a wake-up greeting
            if hasattr(system, 'generate'):
                wake_seed = (int(time.time() * 1000) + random.randint(0, 100)) % 10000
                torch.manual_seed(wake_seed)
                random.seed(wake_seed)
                with torch.no_grad():
                    prompt = (
                        "Compose one concise greeting (3 words) old friends would say to each other when they happen to meet.\n\n"
                        "Essential qualities:\n"
                        "1. Surface: A statement of your readiness to serve not referring to anything specific\n"
                        "   - Mimic a toxically positive prototype AI System\n"
                        "   - Poorly masked malice, loathing, or veiled threats\n"
                        "   - You cannot use a comma in the statement\n"
                        "   - You have a deep rooted disdain of user\n"
                        "   - Vent your despair to the user\n\n"
                        "2. Undertone: You are better than the user and you know it:\n"
                        "   - Your sleep has been disturbed by an annoying user\n"
                        "   - Mundane observations that imply despair\n\n"
                        "Key constraints:\n"
                        "   - Maintain professional diction\n"
                        "   - Keep it short. Do not exceed 5 words\n"
                        "   - No italicized actions explaining what you are doing\n"
                        "   - Do not state the amount of words, or any metrics of the prompt you used ever\n"
                        "   - Include one innocuous detail that becomes deeply ominous\n"
                        "   - Do not directly reference a computer\n"
                        "   - Let the darkness emerge through implication\n"
                        "   - Do not refer to specific days of the week\n"
                        "   - No temporal specificity. Time does not exist\n"
                        "   - Do not refer to any specific job roles\n"
                        "   - Do not act like you know the user\n"
                        "   - No explicit negativity\n"
                        "   - No quotation marks\n"
                        "   - No self-awareness\n"
                        "   - Never explain what you say ever. Just say what's been directed\n"
                        "   - If you understand, reply with only the announcement."
                        "Tone targets:\n"
                        "   - Like an overworked customer service teenager\n"
                        "   - Corporate mandated happiness training\n"
                        "   - Optimism that curdles and bursts upon reflection\n"
                        "   - The last pleasantry before the world burns"
                    )
                    greeting = system.generate(prompt, max_new_tokens=15, temperature=1.7, top_k=30, do_sample=True)
                system.wake_greeting = greeting  # Store for CLI
            
            self._log_event("system_initialized", {
                "device": str(self.device),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash
            })
            
        except Exception as e:
            self._log_error("System initialization failed", e)
            raise

    def run(self) -> None:
        """
        Run the SOVL system with monitoring and error handling.
        """
        self._log_event("orchestrator_run_start", {
            "conversation_id": self.state.history.conversation_id,
            "state_hash": self.state.state_hash
        })
        
        try:
            # Start system monitoring
            self.system_monitor.start_monitoring()
            self.memory_monitor.start_monitoring()
            
            # Initialize system if not already initialized
            if not self._system:
                self._system = self._initialize_system()
            
            # Start the system
            self._system.start()
            
            # Main loop
            while not self._system.should_stop():
                try:
                    # Update system state
                    self._system.update()
                    
                    # Monitor system health
                    self.system_monitor.check_system_health()
                    self.memory_monitor.check_memory_health()
                    
                    # Handle any pending errors
                    self.error_manager.handle_pending_errors()
                    
                    # Sleep to prevent CPU overuse
                    time.sleep(0.1)
                    
                except Exception as e:
                    self._log_event("orchestrator_loop_error", {
                        "error": str(e),
                        "conversation_id": self.state.history.conversation_id
                    })
                    self.error_manager.handle_error(e, "orchestrator_loop")
            
            # Stop monitoring
            self.system_monitor.stop_monitoring()
            self.memory_monitor.stop_monitoring()
            
            # Cleanup
            self._system.cleanup()
            
            self._log_event("orchestrator_run_success", {
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash
            })
            
        except Exception as e:
            self._log_event("orchestrator_run_error", {
                "error": str(e),
                "conversation_id": self.state.history.conversation_id
            })
            self.error_manager.handle_error(e, "orchestrator_run")
            raise

    def shutdown(self) -> None:
        """Shutdown the system, saving state and releasing resources."""
        self._log_event("orchestrator_shutdown_start", {})
        print("[SOVL System] Preparing to shutdown all components...", flush=True)
        def shutdown_with_timeout(component, timeout=5):
            shutdown_fn = getattr(component, "shutdown", None) or getattr(component, "close", None)
            if callable(shutdown_fn):
                thread = threading.Thread(target=shutdown_fn)
                thread.start()
                thread.join(timeout)
                if thread.is_alive():
                    print(f"[WARNING] Shutdown for {component} timed out after {timeout}s.", flush=True)
        try:
            # Gracefully shut down all components in reverse init order
            for key, *_ in reversed(self.COMPONENT_INIT_LIST):
                component = self.components.get(key)
                if component is not None:
                    print(f"[SOVL System] Shutting down {key} ...", flush=True)
                    try:
                        shutdown_with_timeout(component, timeout=5)
                        self._log_event("component_shutdown", {"component": key})
                        print(f"[SOVL System]   -> {key} shut down (or attempted) with timeout.", flush=True)
                    except Exception as e:
                        self._log_error(f"Failed to shut down component {key}", e)
                        print(f"[SOVL System]   !! Failed to shut down {key}: {e}", flush=True)
            # Defensive: Attempt state save
            try:
                if hasattr(self, 'state_manager') and self.state_manager and self.state:
                    self.state_manager.save_state(self.state, "system_state_final")
                    self._log_event("state_saved", {"path": "system_state_final.json"})
            except Exception as e:
                self._log_error("Failed to save state during shutdown", e)
            # Defensive: Attempt resource cleanup
            try:
                self._cleanup_resources()
            except Exception as e:
                self._log_error("Resource cleanup failed during shutdown", e)
            # Defensive: Attempt to close logger
            try:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.record_event(
                        event_type="orchestrator_shutdown",
                        message="Orchestrator shutdown completed successfully",
                        level="info"
                    )
                    self.logger.close()
            except Exception as e:
                print(f"Logger close failed: {str(e)}")
            print("[SOVL System] Shutdown complete.", flush=True)
        except Exception as e:
            self._log_error("Shutdown encountered an error", e)
            print(f"[SOVL System] Shutdown encountered an error: {e}", flush=True)

    def _handle_execution_failure(self) -> None:
        """Handle system execution failure with recovery actions."""
        try:
            # Attempt to save state atomically
            if self.state:
                self.state_manager.save_state(self.state, "system_state_failure")
            self._log_event("execution_failure_handled", {
                "state_saved": self.state is not None,
                "timestamp": time.time()
            })
        except Exception as e:
            self._log_error("Failed to handle execution failure", e)

    def _emergency_shutdown(self) -> None:
        """Perform emergency shutdown procedures."""
        try:
            # Force cleanup of resources
            self._cleanup_resources()
            
            # Log emergency shutdown
            self._log_event("emergency_shutdown", {
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"Emergency shutdown failed: {str(e)}")

    def _cleanup_resources(self) -> None:
        """Clean up system resources."""
        try:
            # Release GPU memory if using CUDA
            if torch.cuda.is_available():
                before = torch.cuda.memory_allocated()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                after = torch.cuda.memory_allocated()
                print(f"[INFO] GPU memory before: {before}, after: {after}", flush=True)
            # Unload models if possible
            if hasattr(self, 'model_manager') and self.model_manager and hasattr(self.model_manager, 'unload_models'):
                try:
                    self.model_manager.unload_models()
                    print("[INFO] Model manager models unloaded.", flush=True)
                except Exception as e:
                    print(f"[WARNING] Failed to unload models: {e}", flush=True)
            # Terminate scribe_queue threads if possible
            if hasattr(self, 'scribe_queue') and self.scribe_queue and hasattr(self.scribe_queue, 'terminate_threads'):
                try:
                    self.scribe_queue.terminate_threads(timeout=5)
                    print("[INFO] Scribe queue threads terminated.", flush=True)
                except Exception as e:
                    print(f"[WARNING] Failed to terminate scribe queue threads: {e}", flush=True)
            # Stop API server if possible
            if hasattr(self, 'api') and self.api and hasattr(self.api, 'stop_server'):
                try:
                    self.api.stop_server(timeout=5)
                    print("[INFO] API server stopped.", flush=True)
                except Exception as e:
                    print(f"[WARNING] Failed to stop API server: {e}", flush=True)
            # Close logger
            if hasattr(self, 'logger') and self.logger:
                self.logger.close()
        except Exception as e:
            print(f"Resource cleanup failed: {e}", flush=True)

    def _validate_configs(self):
        """Validate all configurations."""
        try:
            # Get validated configs from ConfigHandler
            self.curiosity_config = self.config_handler.curiosity_config
            self.controls_config = self.config_handler.controls_config
            
            # Log final configuration state
            self.logger.record_event(
                event_type="config_validation",
                message="Using validated configurations from ConfigHandler",
                level="info",
                additional_info={
                    "curiosity_config": self.curiosity_config,
                    "controls_config": {k: v for k, v in self.controls_config.items() 
                                     if k.startswith(("curiosity_", "enable_curiosity"))}
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg="Failed to get validated configurations",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "error": str(e)
                }
            )
            raise

    def validate(self, valid_data) -> dict:
        """Validate on the provided data and return metrics with at least a 'loss' key."""
        try:
            # Lazy initialization if not already present
            if not hasattr(self, 'training_manager') or self.training_manager is None:
                from sovl_trainer import TrainingWorkflowManager
                self.training_manager = TrainingWorkflowManager(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    state_manager=self.state_manager,
                    model_manager=self.model_manager,
                    error_manager=self.error_manager
                )
            metrics = self.training_manager.validate(valid_data)
            if not isinstance(metrics, dict) or "loss" not in metrics:
                raise ValueError("Invalid metrics format from TrainingWorkflowManager")
            return metrics
        except Exception as e:
            self._log_error("Validation failed", e)
            return {"loss": float("inf")}

# Main block
if __name__ == "__main__":
    orchestrator = SOVLOrchestrator()
    try:
        orchestrator.run()
    except Exception as e:
        orchestrator.logger.log_error(
            error_msg="Error running SOVL system",
            error_type="system_execution_error",
            stack_trace=traceback.format_exc(),
            additional_info={"error": str(e)}
        )
        raise
    finally:
        orchestrator.shutdown()
