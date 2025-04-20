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
from sovl_cli import run_cli
from sovl_logger import LoggingManager
from sovl_state import SOVLState
from sovl_error import ErrorHandler
from sovl_utils import calculate_confidence, detect_repetitions
from sovl_grafter import PluginManager
from collections import deque
from sovl_state import StateManager
from sovl_interfaces import OrchestratorInterface, SystemInterface, SystemMediator
import random
from sovl_main import SOVLSystem, SystemContext, ConfigHandler, ModelLoader, MemoryMonitor, StateTracker, ErrorManager
from sovl_curiosity import CuriosityEngine

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
        self._initialize_logger(log_file)
        self._log_event("orchestrator_init_start", {"config_path": config_path})

        try:
            # Initialize ConfigManager with validation
            self.config_manager = self._create_config_manager(config_path)
            
            # Initialize configuration
            self._initialize_config()
            
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._log_event("device_initialized", {"device": str(self.device)})
            
            # Initialize state manager
            self.state_manager = StateManager(
                config_manager=self.config_manager,
                logger=self.logger,
                device=self.device
            )
            
            # Load state from file if exists, otherwise initialize new state
            self.state = self.state_manager.load_state()
            
            # Initialize system early to ensure state consistency
            self._system: Optional[SystemInterface] = None
            
            # Initialize error handler and plugin manager
            self.error_handler = ErrorHandler(self.logger)
            self.plugin_manager = PluginManager(
                config_manager=self.config_manager,
                logger=self.logger,
                state=self.state
            )
            
            self._lock = Lock()
            self._log_event("orchestrator_init_success", {
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash
            })
        except Exception as e:
            self._log_error("Orchestrator initialization failed", e)
            self.error_handler.handle_generic_error(
                error=e,
                context="orchestrator_initialization",
                fallback_action=lambda: self._cleanup_resources()
            )
            raise RuntimeError(f"Failed to initialize orchestrator: {str(e)}") from e

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
        """Initialize the logger with LoggingManager."""
        try:
            self.logger = LoggingManager(
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

    def sync_state(self) -> None:
        """Synchronize orchestrator state with the system state."""
        with self._lock:
            if not self._system:
                return
            try:
                system_state = self._system.get_state()
                self.state.from_dict(system_state, self.device)
                self._log_event("state_synchronized", {
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash
                })
            except Exception as e:
                self._log_error("State synchronization failed", e)
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
            from sovl_main import SOVLSystem, SystemContext, ConfigHandler, ModelLoader, CuriosityEngine, MemoryMonitor, StateTracker, ErrorManager
            
            context = SystemContext(self.config_manager.config_path, str(self.device))
            config_handler = ConfigHandler(self.config_manager.config_path, self.logger, context.event_dispatcher)
            state_tracker = StateTracker(context)
            error_manager = ErrorManager(context, state_tracker)
            model_loader = ModelLoader(context)
            memory_monitor = MemoryMonitor(context)
            curiosity_engine = CuriosityEngine(
                config_handler=config_handler,
                model_loader=model_loader,
                state_tracker=state_tracker,
                error_manager=error_manager,
                logger=self.logger,
                device=str(self.device)
            )
            
            # Create and register system
            system = SOVLSystem(
                context=context,
                config_handler=config_handler,
                model_loader=model_loader,
                curiosity_engine=curiosity_engine,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
            self.mediator.register_system(system)
            
            # Load state from file if exists, otherwise initialize new state
            self.state = self.state_manager.load_state()
            if self.state is None:
                raise RuntimeError("Failed to load state. System cannot proceed without a valid state.")
            
            # Generate a wake-up greeting
            if hasattr(system, 'generate'):
                wake_seed = (int(time.time() * 1000) + random.randint(0, 100)) % 10000
                torch.manual_seed(wake_seed)
                random.seed(wake_seed)
                with torch.no_grad():
                    greeting = system.generate(" ", max_new_tokens=15, temperature=1.7, top_k=30, do_sample=True)
                print(f"\n{greeting}\n")
            
            self._log_event("system_initialized", {
                "device": str(self.device),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash
            })
            
        except Exception as e:
            self._log_error("System initialization failed", e)
            raise

    def run(self) -> None:
        """Run the SOVL system in the appropriate mode."""
        try:
            # Validate state consistency before running
            if not self._system:
                raise RuntimeError("System not initialized")
            
            # Run in CLI mode by default
            run_cli(self.config_manager)
            
        except Exception as e:
            self._log_error("System execution failed", e)
            self.error_handler.handle_generic_error(
                error=e,
                context="system_execution",
                fallback_action=lambda: self._handle_execution_failure()
            )
            raise

    def shutdown(self) -> None:
        """Shutdown the system, saving state and releasing resources."""
        with self._lock:
            try:
                # Log shutdown start
                self.logger.record_event(
                    event_type="system_shutdown_start",
                    message="Starting system shutdown process",
                    level="info"
                )
    
                # Cleanup event dispatcher
                self.context.event_dispatcher.cleanup()
                self.logger.record_event(
                    event_type="event_dispatcher_cleanup",
                    message="Event dispatcher cleaned up",
                    level="info"
                )
    
                # Save final state
                self.state_tracker.state.save_state()
                self.logger.record_event(
                    event_type="state_saved",
                    message="Final state saved",
                    level="info"
                )
    
                # Clear state history
                self.state_tracker.clear_history()
                self.logger.record_event(
                    event_type="state_history_cleared",
                    message="State history cleared",
                    level="info"
                )
    
                # Clear error history
                self.error_manager.clear_error_history()
                self.logger.record_event(
                    event_type="error_history_cleared",
                    message="Error history cleared",
                    level="info"
                )
    
                # Release GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.record_event(
                        event_type="gpu_memory_cleared",
                        message="GPU memory cache cleared",
                        level="info"
                    )
    
                # Final cleanup of logger
                self.logger.clear_queues()
                self.logger.record_event(
                    event_type="system_shutdown_complete",
                    message="System shutdown completed successfully",
                    level="info"
                )
    
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Failed to shutdown system: {str(e)}",
                    error_type="shutdown_error",
                    stack_trace=traceback.format_exc()
                )
                raise RuntimeError("Failed to shutdown system") from e

    def _handle_execution_failure(self) -> None:
        """Handle system execution failure with recovery actions."""
        try:
            # Attempt to save state
            if self._system and hasattr(self._system, 'state_tracker'):
                self._system.state_tracker.state.save_state()
            
            # Log failure details
            self._log_event("execution_failure_handled", {
                "state_saved": self._system is not None,
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
