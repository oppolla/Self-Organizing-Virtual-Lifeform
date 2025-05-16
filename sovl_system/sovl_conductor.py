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
        ("config_manager", "sovl_config", "ConfigManager", {}, {"is_critical": True}),
        ("logger", "sovl_logger", "Logger", {}, {"is_critical": True}),
        ("resource_manager", "sovl_resource", "ResourceManager", {}, {"is_critical": False}),
        ("state_manager", "sovl_state", "StateManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "device": {"key": "device", "required": False}
        }, {"is_critical": True}),
        ("error_manager", "sovl_error", "ErrorManager", {
            "state_manager": {"key": "state_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }, {"is_critical": True}),

        # Core system components with minimal dependencies
        ("ram_manager", "sovl_memory", "RAMManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }, {"is_critical": False}),
        ("gpu_manager", "sovl_memory", "GPUMemoryManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }, {"is_critical": False}),
        ("event_dispatcher", "sovl_events", "EventDispatcher", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": False}
        }, {"is_critical": False}),

        # Basic data and resource management
        ("jsonl_loader", "sovl_io", "JSONLLoader", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }, {"is_critical": False}),
        ("scribe_queue", "sovl_queue", "ScribeQueue", {
            "logger": {"key": "logger", "required": False}
        }, {"is_critical": False}),
        ("hardware_manager", "sovl_hardware", "HardwareManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": False}
        }, {"is_critical": False}),
        ("data_manager", "sovl_data", "DataManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": False}
        }, {"is_critical": False}),

        # Model and core processing
        ("model_manager", "sovl_manager", "ModelManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "device": {"key": "device", "required": False}
        }, {"is_critical": True}),
        ("metadata_processor", "sovl_processor", "MetadataProcessor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_accessor": {"key": "state_manager", "required": False}
        }, {"is_critical": False}),
        ("lora_adapter_manager", "sovl_engram", "LoraAdapterManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_handler": {"key": "error_manager", "required": False}
        }, {"is_critical": True}),
        ("scaffold_provider", "sovl_scaffold", "ScaffoldProvider", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_handler": {"key": "error_manager", "required": False},
            "ram_manager": {"key": "ram_manager", "required": True},
            "gpu_manager": {"key": "gpu_manager", "required": True}
        }, {"is_critical": True}),
        ("cross_attention_injector", "sovl_scaffold", "CrossAttentionInjector", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": False},
            "ram_manager": {"key": "ram_manager", "required": False},
            "gpu_manager": {"key": "gpu_manager", "required": False}
        }, {"is_critical": True}),
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
        }, {"is_critical": True}),

        # Monitoring systems
        ("system_monitor", "sovl_monitor", "SystemMonitor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "ram_manager": {"key": "ram_manager", "required": True},
            "gpu_manager": {"key": "gpu_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "user_profile_state": {"key": "state_manager", "required": False},
            "bond_calculator": {"key": "bond_calculator", "required": False}
        }, {"is_critical": False}),
        ("memory_monitor", "sovl_monitor", "MemoryMonitor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "ram_manager": {"key": "ram_manager", "required": True},
            "gpu_manager": {"key": "gpu_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }, {"is_critical": False}),

        # Auxiliary Managers (Moved here for earlier initialization)
        ("lifecycle_manager", "sovl_trainer", "LifecycleManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }, {"is_critical": False}),
        ("scaffold_manager", "sovl_scaffold", "ScaffoldManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True}
        }, {"is_critical": True}),

        # System foundation components
        ("system_context", "sovl_main", "SystemContext", {
            "config_path": {"key": "config_manager", "required": True}
        }, {"is_critical": True}),
        ("system_mediator", "sovl_interfaces", "SystemMediator", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "device": {"key": "device", "required": True}
        }, {"is_critical": True}),

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
        }, {"is_critical": True}),
        ("confidence_calculator", "sovl_confidence", "ConfidenceCalculator", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }, {"is_critical": False}),
        ("bond_calculator", "sovl_bonder", "BondCalculator", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "user_profile_state": {"key": "state_manager", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }, {"is_critical": False}),
        ("bond_modulator", "sovl_bonder", "BondModulator", {
            "bond_calculator": {"key": "bond_calculator", "required": True},
            "max_retries": {"key": "max_retries", "required": False}
        }, {"is_critical": False}),

        # Advanced personality and behavior systems
        ("temperament_system", "sovl_temperament", "TemperamentSystem", {
            "state_manager": {"key": "state_manager", "required": True},
            "config_manager": {"key": "config_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }, {"is_critical": False}),
        ("chronos_system", "sovl_chronos", "Chronos", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }, {"is_critical": False}),
        ("curiosity_manager", "sovl_curiosity", "CuriosityManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "device": {"key": "device", "required": False},
            "generation_manager": {"key": "generation_manager", "required": False},
            "state_manager": {"key": "state_manager", "required": True}
        }, {"is_critical": False}),
        ("traits_monitor", "sovl_monitor", "TraitsMonitor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_tracker": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }, {"is_critical": False}),
        ("vibe_sculptor", "sovl_viber", "VibeSculptor", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "temperament_system": {"key": "temperament_system", "required": False}
        }, {"is_critical": False}),

        # Creative and reflective systems
        ("dreamer", "sovl_dreamer", "Dreamer", {
            "config_manager": {"key": "config_manager", "required": True},
            "scribe_path": {"key": "config_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "state_manager": {"key": "state_manager", "required": True}
        }, {"is_critical": False}),
        ("aspiration_system", "sovl_striver", "AspirationSystem", {
            "config": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "long_term_memory": {"key": "state_manager._current_state.dialogue_context", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }, {"is_critical": False}),
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
        }, {"is_critical": False}),
        ("shamer", "sovl_shamer", "Shamer", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "viber": {"key": "vibe_sculptor", "required": False},
            "dialogue_context_manager": {"key": "state_manager", "required": False}
        }, {"is_critical": False}),

        # Dialogue Context Management (Moved and Corrected)
        ("dialogue_shutdown", "sovl_recaller", "DialogueShutdown", {
            "dialogue_context_manager": {"key": "state_manager._current_state.dialogue_context", "required": True},
            "long_term_memory": {"key": "state_manager._current_state.dialogue_context.ltm", "required": False},
            "logger": {"key": "logger", "required": True}
        }, {"is_critical": True}),

        # Training and process systems
        ("training_manager", "sovl_trainer", "TrainingWorkflowManager", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "model_manager": {"key": "model_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True}
        }, {"is_critical": True}),
        ("scribe_ingestion_processor", "sovl_processor", "ScribeIngestionProcessor", {
            "log_paths": {"key": "scribe_log_paths", "required": False},
            "memory_templates": {"key": "memory_templates", "required": False},
            "logger": {"key": "logger", "required": True},
            "config_path": {"key": "config_manager", "required": True}
        }, {"is_critical": False}),

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
        }, {"is_critical": True}),

        # Main system and API components (depend on almost everything)
        ("sovl_system", "sovl_main", "SOVLSystem", {
            "config_manager": {"key": "config_manager", "required": True},
            "logger": {"key": "logger", "required": True},
            "state_manager": {"key": "state_manager", "required": True},
            "error_manager": {"key": "error_manager", "required": True},
            "generation_primer": {"key": "generation_primer", "required": True},
            "chronos_system": {"key": "chronos_system", "required": True},
            "dialogue_context_manager": {"key": "state_manager", "required": False},
            "device": {"key": "device", "required": False},
            "lifecycle_manager": {"key": "lifecycle_manager", "required": False},
            "scaffold_manager": {"key": "scaffold_manager", "required": False}
        }, {"is_critical": True}),
        ("api", "sovl_api", "SOVLAPI", {
            "config_path": {"key": "config_manager", "required": True}
        }, {"is_critical": True}),

        # CLI handler must be last as it depends on sovl_system which is created late
        ("cli_handler", "sovl_cli", "CommandHandler", {
            "sovl_system": {"key": "sovl_system", "required": True}
        }, {"is_critical": True}),
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
        self._system = None # Initialize self._system to None

        if not os.path.isfile(config_path):
            msg = f"Config file not found: {config_path}"
            # Logger not ready yet, print directly
            print(f"[CRITICAL ERROR] {msg}", flush=True)
            raise RuntimeError(msg)
        
        self._initialize_logger(log_file) # Logger is now available
        self._log_event("orchestrator_init_start", {"config_path": config_path, "log_file": log_file})
        
        self.device = self.select_device(min_vram_gb=4)
        self._log_event("device_selected", {"device": str(self.device)})
        
        if hasattr(self.device, "type") and self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
                self._log_event("gpu_memory_initial_check", {"allocated_gb": allocated, "reserved_gb": reserved, "total_gb": total})
                if allocated / total > 0.8:
                    self._log_event("gpu_memory_high_usage_warning_initial", {"usage_percent": (allocated/total)*100}, level="warning")
            except Exception as e:
                self._log_event("gpu_memory_log_failed_initial", {"error": str(e)}, level="warning")

        INIT_GROUPS = [
            ("Core Infrastructure", ["config_manager", "logger", "resource_manager", "state_manager", "error_manager"]),
            ("State & Memory Managers", ["ram_manager", "gpu_manager"]),
            ("Event & Data Systems", ["event_dispatcher", "jsonl_loader", "scribe_queue", "hardware_manager", "data_manager"]),
            ("Model & Processing", ["model_manager", "metadata_processor", "lora_adapter_manager", "scaffold_provider", "cross_attention_injector", "scaffold_token_mapper", "scaffold_manager"]),
            ("Monitors", ["system_monitor", "memory_monitor", "traits_monitor"]),
            ("Auxiliary Managers", ["lifecycle_manager"]),
            ("System Context & Mediation", ["system_context", "system_mediator"]),
            ("Generation & Calculation", ["generation_manager", "confidence_calculator", "bond_calculator", "bond_modulator"]),
            ("Personality & Behavior", ["temperament_system", "curiosity_manager", "vibe_sculptor"]),
            ("Creative & Reflective", ["dreamer", "aspiration_system", "introspection_manager", "shamer"]),
            ("Dialogue Context Management", ["dialogue_shutdown"]), 
            ("Training & Ingestion", ["training_manager", "scribe_ingestion_processor"]),
            ("Generation Prep & Interface", ["generation_primer"]),
            ("Main System & API", ["sovl_system", "api"]),
            ("CLI Handler", ["cli_handler"]),
        ]
        key_to_index = {key: i for i, (key, *_rest) in enumerate(self.COMPONENT_INIT_LIST)}
        total_components_to_init = len(self.COMPONENT_INIT_LIST)
        initialized_components = []
        failed_components_info = [] # Stores (key, error_str, is_critical_flag)
        component_idx_counter = 1

        print("[SOVL System] Starting Component Incarnation Phase...", flush=True)
        for group_name, group_keys in INIT_GROUPS:
            print(f"[SOVL System] — {group_name} —", flush=True)
            for key in group_keys:
                idx = key_to_index.get(key)
                if idx is None:
                    self._log_event("init_group_component_missing", {"component": key, "group": group_name}, level="warning")
                    print(f"[SOVL System] [WARNING] Component '{key}' in INIT_GROUPS but not COMPONENT_INIT_LIST. Skipping.", flush=True)
                    continue
                
                if len(self.COMPONENT_INIT_LIST[idx]) < 4:
                    # This is a malformed entry, should be critical regardless of component's own flag
                    msg = f"Malformed entry for component {key} in COMPONENT_INIT_LIST (entry: {self.COMPONENT_INIT_LIST[idx]})."
                    self._log_event("component_entry_malformed", {"component": key, "entry_str": str(self.COMPONENT_INIT_LIST[idx])}, level="critical") # Use critical for logging
                    print(f"[SOVL System] [CRITICAL ERROR] {msg} Halting.", flush=True)
                    raise RuntimeError(msg)

                _key, module_name, class_name, dep_map, *options_tuple = self.COMPONENT_INIT_LIST[idx]
                component_options = options_tuple[0] if options_tuple else {}
                is_critical_component = component_options.get("is_critical", True) # Default to True

                msg_prefix = f"[{component_idx_counter}/{total_components_to_init}] Incarnating {key} ({class_name}) ... "
                print(f"[SOVL System] {msg_prefix}", end="", flush=True)
                
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)
                    kwargs = {}
                    for arg, dep_info in dep_map.items():
                        if isinstance(dep_info, str):
                            dep_key = dep_info
                            required = True
                        else:
                            dep_key = dep_info["key"]
                            required = dep_info.get("required", True)
                        
                        dep_instance = None
                        if dep_key == "device":
                            dep_instance = self.device
                        elif "." in dep_key:
                            current_obj = self
                            valid_path = True
                            for part in dep_key.split("."):
                                if hasattr(current_obj, part):
                                    current_obj = getattr(current_obj, part)
                                    if current_obj is None and required: # Check for None midway if path is required
                                        # This can happen if, e.g. state_manager._current_state is None
                                        valid_path = False
                                        break 
                                else:
                                    valid_path = False
                                    break
                            if valid_path:
                                dep_instance = current_obj
                        elif dep_key in self.components:
                            dep_instance = self.components[dep_key]
                        elif hasattr(self, dep_key):
                            dep_instance = getattr(self, dep_key)
                        
                        if dep_instance is None and required:
                            raise RuntimeError(f"Required dependency '{dep_key}' for '{key}' resolved to None or was not found.")
                        elif dep_instance is None and not required:
                            self._log_event("optional_dependency_not_found", {"component": key, "dependency": dep_key}, level="warning")
                            print(f"[INFO] Optional dependency '{dep_key}' for '{key}' is None. Fallback behavior expected in component.", flush=True)
                        kwargs[arg] = dep_instance

                    if key == "config_manager": # Special handling for config_manager
                        if not hasattr(self, 'logger') or not self.logger:
                             # This is a critical bootstrap issue, logger should have been in kwargs but depends on this method of init
                             print("CRITICAL: Logger not initialized before ConfigManager attempts to use it as kwarg. This indicates a flaw in bootstrap.", flush=True)
                             raise RuntimeError("Logger not available for ConfigManager initialization")
                        kwargs = {"config_path": config_path, "logger": self.logger}
                    elif key == "logger": # Special handling for logger
                        kwargs = {"log_file": log_file, "max_size_mb": self.LOG_MAX_SIZE_MB, "rotation_interval": "1d"}

                    instance = cls(**kwargs)
                    self.components[key] = instance
                    setattr(self, key, instance)
                    print("✓", flush=True)
                    initialized_components.append(key)
                except Exception as e:
                    print(f"✗ (Error: {e})", flush=True)
                    tb_str = traceback.format_exc()
                    self._log_error(f"Failed to initialize component '{key}'", e) # Already logs stack trace
                    self.components[key] = None
                    setattr(self, key, None)
                    failed_components_info.append((key, str(e), is_critical_component))

                    if is_critical_component:
                        print("[SOVL System] Incarnation halted due to CRITICAL component failure.", flush=True)
                        self._log_event("initialization_halted_critical_failure", {"failed_component": key, "error": str(e), "traceback": tb_str})
                        # Print summary before raising
                        print("[SOVL System] Partial Incarnation Summary:", flush=True)
                        print(f"  Successfully Initialized: {initialized_components}", flush=True)
                        failed_summary = [(k, err_str) for k, err_str, _is_crit in failed_components_info]
                        print(f"  Failed (up to this point): {failed_summary}", flush=True)
                        raise RuntimeError(f"Critical component '{key}' failed to initialize: {e}") from e
                    else:
                        print(f"[SOVL System] [WARNING] Optional component '{key}' failed to initialize. System will continue with reduced functionality.", flush=True)
                        self._log_event("optional_component_initialization_failed", {"component": key, "error": str(e)})
                
                component_idx_counter += 1
            print("", flush=True) # Blank line between groups
        
        print("[SOVL System] Component Incarnation Phase Complete. Final Summary:", flush=True)
        print(f"  Successfully Initialized ({len(initialized_components)} components): {initialized_components}", flush=True)
        
        non_critical_failures = [(k, err) for k, err, is_crit in failed_components_info if not is_crit]
        if non_critical_failures:
            print(f"  Optional Components Failed ({len(non_critical_failures)} components - system may be degraded):", flush=True)
            for f_key, f_err in non_critical_failures:
                print(f"    - {f_key}: {f_err}", flush=True)
        
        # This part of summary about critical failures should ideally not be reached if they halt execution above.
        critical_failures_recorded = [(k, err) for k, err, is_crit in failed_components_info if is_crit]
        if critical_failures_recorded and not any(f_is_crit for _, _, f_is_crit in failed_components_info if f_is_crit and f_key == critical_failures_recorded[0][0]):
             # This implies a critical failure was recorded but somehow bypassed the immediate halt.
             print(f"  [UNEXPECTED ERROR] Critical Components Recorded as Failed but did not halt: {critical_failures_recorded}", flush=True)
             # Consider a final raise here if this state is truly unrecoverable and unexpected.
        
        self._log_event("initialization_final_summary", {"succeeded": initialized_components, "failed_info": failed_components_info})

        if not self.components.get("sovl_system"):
            # If sovl_system (a critical component) failed, the raise above should have caught it.
            # This is a fallback check.
            msg = "SOVLSystem (core system component) failed to initialize. Cannot continue."
            print(f"[SOVL System] [CRITICAL] {msg}", flush=True)
            self._log_event("sovl_system_not_initialized_final_check", {}, level="critical")
            # Ensure an error is raised if not already done by a critical component failure
            if not any(f_is_crit for _, _, f_is_crit in failed_components_info):
                 raise RuntimeError(msg)
            # If a critical failure already happened, this point shouldn't be reached.

        # Proceed with post-initialization steps if critical components are up
        self._load_initial_state() # Load state after all components are attempted
        self.post_initialize_components() # Wire up inter-component references
        self._initialize_config() # Load orchestrator specific configs

        if hasattr(self, 'sovl_system') and self.sovl_system:
            self._prepare_system_for_run(self.sovl_system)
            self._system = self.sovl_system # Assign the main system object to self._system
            self._log_event("orchestrator_system_link_complete", {"system_id": id(self._system)})
        else:
            # This means sovl_system (critical) was not initialized or became None.
            # The checks above should have caught this and raised.
            msg = "SOVLSystem instance not available after initialization. Orchestrator cannot proceed."
            self._log_error(msg, RuntimeError(msg))
            # Defensive raise if not already halted by a critical failure. No component should be critical to this point and not halted.
            if not any(is_crit for _, _, is_crit in failed_components_info): 
                raise RuntimeError(msg)

        print(f"[SOVL System] Orchestrator initialization complete. Log file: {self.logger.log_file}", flush=True)
        self._log_event("orchestrator_init_complete")

    def post_initialize_components(self):
        """Wire up references that require all components to be initialized."""
        if hasattr(self, "generation_primer") and hasattr(self, "sovl_system") and self.generation_primer and self.sovl_system:
            self.generation_primer.sovl_system = self.sovl_system
            self._log_event("generation_primer_sovl_system_linked")

    def _load_initial_state(self):
        """Load and validate initial system state, or initialize default state."""
        if not hasattr(self, 'state_manager') or not self.state_manager:
            msg = "StateManager not available for _load_initial_state."
            print(f"[CRITICAL ERROR] {msg}", flush=True)
            self._log_error(msg, RuntimeError(msg)) # Logger might be available
            raise RuntimeError(msg)
        if not hasattr(self, 'config_manager') or not self.config_manager:
            msg = "ConfigManager not available for _load_initial_state."
            print(f"[CRITICAL ERROR] {msg}", flush=True)
            self._log_error(msg, RuntimeError(msg))
            raise RuntimeError(msg)

        try:
            loaded_state = self.state_manager.load_state(self.config_manager.config_path)
            is_valid_state = False
            if loaded_state:
                state_dict_for_validation = loaded_state.to_dict() if hasattr(loaded_state, 'to_dict') else loaded_state
                is_valid_state = self.state_manager.validate_state(state_dict_for_validation)
            
            if not loaded_state or not is_valid_state:
                warning_msg = "Loaded state is invalid or missing. Initializing default state."
                if loaded_state and not is_valid_state:
                    warning_msg = "Loaded state validation failed. Initializing default state."
                elif not loaded_state:
                    warning_msg = "No state file found or failed to load. Initializing default state."
                
                self.logger.record_event(
                    event_type="state_validation_failed_or_missing",
                    message=warning_msg,
                    level="warning"
                )
                self.state_manager._initialize_state()  # Initialize default state
                loaded_state = self.state_manager._current_state
                self.state_manager.save_state(loaded_state, self.config_manager.config_path)
                self._log_event("default_state_initialized_and_saved")
            
            self.state = loaded_state # Uses the orchestrator's state property setter
            self._log_event("initial_state_loaded_and_assigned", {"state_hash": getattr(loaded_state, 'state_hash', None)})
        
        except Exception as e:
            self._log_error("Critical error during _load_initial_state", e)
            self.logger.record_event(
                event_type="state_load_critical_failure",
                message=f"Failed to load or initialize state: {str(e)}. Attempting to force default state.",
                level="error"
            )
            try:
                self.state_manager._initialize_state()
                loaded_state = self.state_manager._current_state
                # No save here to avoid potential loop if save path is problematic
                self.state = loaded_state
                self._log_event("default_state_force_initialized_no_save")
            except Exception as final_e:
                self._log_error("Failed to even force-initialize default state", final_e)
                raise RuntimeError("Unable to initialize any system state.") from final_e

    def _prepare_system_for_run(self, system_instance: 'SOVLSystem'):
        """Perform any final preparations on the SOVLSystem instance before run, like generating a wake-up greeting."""
        if not system_instance:
            self._log_error("Cannot prepare a null SOVLSystem instance for run.", ValueError("system_instance is None"))
            return

        try:
            if hasattr(system_instance, 'generate') and callable(system_instance.generate) and hasattr(system_instance, 'config_handler'):
                # Wake-up greeting logic (copied and adapted from original initialize_system)
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
                    # Determine generation parameters from config if possible, else defaults
                    gen_config = system_instance.config_handler.get_section("generation_config") if hasattr(system_instance, 'config_handler') else {}
                    max_new_tokens = gen_config.get("default_max_new_tokens", 15)
                    temperature = gen_config.get("default_temperature", 1.7)
                    top_k = gen_config.get("default_top_k", 30)
                    do_sample = gen_config.get("default_do_sample", True)

                    greeting = system_instance.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, do_sample=do_sample)
                system_instance.wake_greeting = greeting  # Store for CLI or other uses
                self._log_event("system_wake_greeting_generated", {"greeting": greeting, "system_id": id(system_instance)})
            else:
                self._log_event("system_wake_greeting_skipped", {"reason": "System instance missing generate method or config_handler", "system_id": id(system_instance)})
        except Exception as e:
            self._log_error(f"Failed to generate wake-up greeting for system {id(system_instance)}", e)
            if hasattr(system_instance, 'wake_greeting'):
                 system_instance.wake_greeting = "Initialization complete."

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

    def _initialize_logger(self, log_file: str) -> None:
        """Initialize the logger with Logger."""
        try:
            self.logger = Logger(
                log_file=log_file,
                max_size_mb=10,
                rotation_interval="1d"
            )
            # self._log_event is not available yet as config_manager might not be ready for full context
            # So, we log directly after logger is confirmed to be an object.
            if hasattr(self, 'logger') and self.logger:
                 self.logger.record_event("logger_initialized", "Logger instance created.", additional_info={"log_file": log_file})
            else:
                # This case should ideally not be hit if Logger constructor raises an error
                print(f"CRITICAL: Logger object not created but no exception from Logger constructor for {log_file}", flush=True)

        except Exception as e:
            # If Logger() constructor fails, self.logger might not be assigned or be None.
            # Cannot use self._log_error or self.logger.log_error here reliably.
            print(f"CRITICAL: Failed to initialize logger: {e}\n{traceback.format_exc()}", flush=True)
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
            # Use self.components which stores all initialized components
            # Iterate over a copy of keys if components might be removed during shutdown (defensive)
            component_keys = list(self.components.keys())
            # Find the original order from COMPONENT_INIT_LIST to shutdown in reverse
            ordered_shutdown_keys = [key for key, *_ in reversed(self.COMPONENT_INIT_LIST) if key in component_keys]
            
            for key in ordered_shutdown_keys:
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
            
            # Ensure system itself (if it has a shutdown) is handled if not in components dict by key 'sovl_system'
            # However, sovl_system IS in COMPONENT_INIT_LIST and self.components, so above loop handles it.
            # If self._system was a different object, it would need separate handling:
            # if self._system and hasattr(self._system, 'shutdown'):
            #     print("[SOVL System] Shutting down main system object...", flush=True)
            #     shutdown_with_timeout(self._system, timeout=10)

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
