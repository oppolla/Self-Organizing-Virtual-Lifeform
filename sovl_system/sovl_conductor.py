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

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, log_file: str = DEFAULT_LOG_FILE) -> None:
        """
        Initialize the orchestrator with configuration and logging.

        Args:
            config_path: Path to the configuration file.
            log_file: Path to the orchestrator's log file.

        Raises:
            RuntimeError: If initialization of ConfigManager or SOVLSystem fails.
        """
        # Initialize thread lock for state synchronization
        self._lock = threading.RLock()
        
        self._initialize_logger(log_file)
        self._log_event("orchestrator_init_start", {"config_path": config_path})

        # Defensive: Check config file existence
        if not os.path.isfile(config_path):
            msg = f"Config file not found: {config_path}"
            self._log_error(msg, FileNotFoundError(msg))
            raise RuntimeError(msg)

        # --- ResourceManager integration start ---
        self.components = {}  # Store system components for compatibility
        self.components["resource_manager"] = ResourceManager(logger=self.logger)
        # Reserve 2048 MB GPU memory for ModelManager (adjust as needed)
        gpu_mem_mb = 2048
        acquired = self.components["resource_manager"].acquire("gpu_memory", amount=gpu_mem_mb)
        if not acquired:
            raise RuntimeError(f"Insufficient GPU memory for ModelManager (requested {gpu_mem_mb} MB)")
        # --- ResourceManager integration end ---

        try:
            # Initialize ConfigManager with validation
            try:
                self.config_manager = self._create_config_manager(config_path)
            except Exception as e:
                self._log_error("Failed to initialize ConfigManager", e)
                raise
            # Initialize configuration
            try:
                self._initialize_config()
            except Exception as e:
                self._log_error("Failed to initialize configuration", e)
                raise
            # Initialize device
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._log_event("device_initialized", {"device": str(self.device)})
            except Exception as e:
                self._log_error("Failed to initialize device", e)
                self.device = torch.device("cpu")
            # Initialize state manager
            try:
                self.state_manager = StateManager(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    device=self.device
                )
            except Exception as e:
                self._log_error("Failed to initialize StateManager", e)
                self.state_manager = None
            # Initialize model manager early
            try:
                self.model_manager = ModelManager(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    device=self.device
                )
            except Exception as e:
                # Release GPU memory on failure
                self.components["resource_manager"].release("gpu_memory", amount=gpu_mem_mb)
                self._log_error("Failed to initialize ModelManager", e)
                self.model_manager = None
            # Initialize memory managers
            try:
                self.ram_manager = RAMManager(self.config_manager, self.logger)
            except Exception as e:
                self._log_error("Failed to initialize RAMManager", e)
                self.ram_manager = None
            try:
                self.gpu_manager = GPUMemoryManager(self.config_manager, self.logger)
            except Exception as e:
                self._log_error("Failed to initialize GPUMemoryManager", e)
                self.gpu_manager = None
            # Initialize error manager
            try:
                self.error_manager = ErrorManager(self.state_manager, self.logger)
            except Exception as e:
                self._log_error("Failed to initialize ErrorManager", e)
                self.error_manager = None
            # Initialize monitors
            try:
                self.system_monitor = SystemMonitor(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    ram_manager=self.ram_manager,
                    gpu_manager=self.gpu_manager,
                    error_manager=self.error_manager
                )
            except Exception as e:
                self._log_error("Failed to initialize SystemMonitor", e)
                self.system_monitor = None
            try:
                self.memory_monitor = MemoryMonitor(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    ram_manager=self.ram_manager,
                    gpu_manager=self.gpu_manager,
                    error_manager=self.error_manager
                )
            except Exception as e:
                self._log_error("Failed to initialize MemoryMonitor", e)
                self.memory_monitor = None
            try:
                self.traits_monitor = TraitsMonitor(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    state_tracker=getattr(self.state_manager, 'state_tracker', None),
                    error_manager=self.error_manager
                )
            except Exception as e:
                self._log_error("Failed to initialize TraitsMonitor", e)
                self.traits_monitor = None
        except Exception as e:
            self._log_error("Orchestrator initialization failed", e)
            raise

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
        """Always access the canonical SOVLState via StateManager."""
        return getattr(self.state_manager, '_system_state', None)

    @state.setter
    def state(self, value):
        """Set the canonical SOVLState in StateManager."""
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
                
                # Attempt recovery with last known good state
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
            system = SOVLSystem(
                context=context,
                config_handler=self.config_manager,
                model_manager=self.model_manager,
                curiosity_manager=curiosity_manager,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
            self.mediator.register_system(system)
            
            # Load state from file if exists, otherwise initialize new state
            loaded_state = self.state_manager.load_state(self.config_manager.config_path)
            if loaded_state is None:
                raise RuntimeError("Failed to load state. System cannot proceed without a valid state.")
            self.state = loaded_state  # Set as canonical state in StateManager
            
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
        try:
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
        except Exception as e:
            self._log_error("Shutdown encountered an error", e)

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
                torch.cuda.empty_cache()
            
            # Close any open file handles
            if hasattr(self, 'logger'):
                self.logger.close()
        except Exception as e:
            print(f"Resource cleanup failed: {str(e)}")

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

class Conductor:
    """Orchestrates the SOVL system components."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize the conductor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self._lock = Lock()
        
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health across all memory managers."""
        try:
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            return {
                "ram_health": ram_health,
                "gpu_health": gpu_health
            }
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to check memory health: {str(e)}",
                error_type="memory_health_error",
                stack_trace=traceback.format_exc()
            )
            return {
                "ram_health": {"status": "error"},
                "gpu_health": {"status": "error"}
            }

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
