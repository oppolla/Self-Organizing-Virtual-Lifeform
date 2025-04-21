from typing import Optional, Dict, Any, Callable, List, Union, Literal, Type
import traceback
import time
from collections import defaultdict, deque
from threading import Lock
from sovl_logger import Logger
from sovl_state import SOVLState, StateTracker
from sovl_config import ConfigManager
import torch
from dataclasses import dataclass
from datetime import datetime
from sovl_memory import GPUMemoryManager
from sovl_records import ErrorRecordBridge, IErrorHandler, ErrorRecord
from functools import wraps

@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    additional_info: Dict[str, Any]

class ErrorManager(IErrorHandler):
    """Manages error handling and recovery for the SOVL system."""
    
    def __init__(
        self,
        context: SystemContext,
        state_tracker: StateTracker,
        config_manager: ConfigManager,
        error_cooldown: float = 1.0,
        max_recent_errors: int = 100
    ) -> None:
        """Initialize the ErrorManager with context and configuration."""
        if not context:
            raise ValueError("context cannot be None")
        if not state_tracker:
            raise ValueError("state_tracker cannot be None")
        if not config_manager:
            raise ValueError("config_manager cannot be None")
            
        self.context = context
        self.state_tracker = state_tracker
        self.config_manager = config_manager
        self.logger = context.logger
        
        # Initialize configuration
        self._initialize_config()
        
        # Register with ErrorRecordBridge
        ErrorRecordBridge().register_handler(self)
        
        # Subscribe to configuration changes
        self.config_manager.subscribe(self._on_config_change)

    def handle_error(self, record: ErrorRecord) -> None:
        """Handle error records from the ErrorRecordBridge."""
        try:
            # Check severity thresholds
            error_count = ErrorRecordBridge().get_error_count(record.error_type)
            if error_count >= self.severity_thresholds['critical']:
                self._handle_critical_error(record)
            elif error_count >= self.severity_thresholds['warning']:
                self._handle_warning(record)
                
            # Log the error
            self.logger.record_event(
                event_type="error",
                message=record.error_message,
                level="error",
                additional_info={
                    "error_type": record.error_type,
                    "stack_trace": record.stack_trace,
                    **record.additional_info
                }
            )
            
            # Apply recovery strategy if available
            if record.error_type in self.recovery_strategies:
                self.recovery_strategies[record.error_type](record)
                
        except Exception as e:
            print(f"[ERROR] Failed to handle error: {str(e)}")
            traceback.print_exc()

    def record_error(
        self,
        error: Exception,
        error_type: str,
        context: Dict[str, Any] = None,
        stack_trace: Optional[str] = None
    ) -> None:
        """Record an error through the ErrorRecordBridge."""
        try:
            ErrorRecordBridge().record_error(
                error_type=error_type,
                error_message=str(error),
                stack_trace=stack_trace or traceback.format_exc(),
                additional_info=context or {}
            )
        except Exception as e:
            print(f"[ERROR] Failed to record error: {str(e)}")
            traceback.print_exc()

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics from ErrorRecordBridge."""
        return {
            "error_counts": dict(ErrorRecordBridge()._error_counts),
            "recent_errors": [record.__dict__ for record in ErrorRecordBridge().get_recent_errors()]
        }

    def _handle_critical_error(self, record: ErrorRecord) -> None:
        """Handle critical error threshold exceeded."""
        self.logger.record_event(
            event_type="critical_error_threshold",
            message=f"Critical error threshold exceeded for {record.error_type}",
            level="critical",
            additional_info={
                "error_type": record.error_type,
                "error_count": ErrorRecordBridge().get_error_count(record.error_type),
                "threshold": self.severity_thresholds['critical']
            }
        )

    def _handle_warning(self, record: ErrorRecord) -> None:
        """Handle warning threshold exceeded."""
        self.logger.record_event(
            event_type="warning_threshold",
            message=f"Warning threshold exceeded for {record.error_type}",
            level="warning",
            additional_info={
                "error_type": record.error_type,
                "error_count": ErrorRecordBridge().get_error_count(record.error_type),
                "threshold": self.severity_thresholds['warning']
            }
        )

    def _initialize_config(self) -> None:
        """Initialize error handling configuration from ConfigManager."""
        try:
            # Load error handling configuration
            error_config = self.config_manager.get_section("error_config")
            if not error_config:
                raise ValueError("Error configuration section is missing or empty.")
            
            # Set error cooldown with validation
            self.error_cooldown = float(error_config.get("error_cooldown", 1.0))
            if self.error_cooldown <= 0:
                self.logger.record_event(
                    event_type="config_validation",
                    message="Invalid error_cooldown value, using default",
                    level="warning",
                    additional_info={"value": self.error_cooldown, "default": 1.0}
                )
                self.error_cooldown = 1.0
            
            # Set severity thresholds with validation
            self.severity_thresholds = {
                "warning": float(error_config.get("warning_threshold", 3.0)),
                "error": float(error_config.get("error_threshold", 5.0)),
                "critical": float(error_config.get("critical_threshold", 10.0))
            }
            
            # Validate thresholds
            for level, threshold in self.severity_thresholds.items():
                if threshold <= 0:
                    self.logger.record_event(
                        event_type="config_validation",
                        message=f"Invalid {level} threshold, using default",
                        level="warning",
                        additional_info={"value": threshold, "default": 3.0 if level == "warning" else 5.0 if level == "error" else 10.0}
                    )
                    self.severity_thresholds[level] = 3.0 if level == "warning" else 5.0 if level == "error" else 10.0
            
            # Initialize recovery strategies
            self.recovery_strategies = {
                "training": self._recover_training,
                "curiosity": self._recover_curiosity,
                "memory": self._recover_memory,
                "generation": self._recover_generation,
                "data": self._recover_data
            }
            
            # Initialize parameter adjustments with configuration validation
            self.parameter_adjustments = {
                "training": lambda ctx: self._adjust_training_params(ctx),
                "curiosity": lambda ctx: self._adjust_curiosity_params(ctx),
                "memory": lambda ctx: self._adjust_memory_params(ctx),
                "generation": lambda ctx: self._adjust_generation_params(ctx),
                "data": lambda ctx: self._adjust_data_params(ctx)
            }
            
            self.logger.record_event(
                event_type="error_config_initialized",
                message="Error handling configuration initialized successfully",
                level="info",
                additional_info={
                    "error_cooldown": self.error_cooldown,
                    "severity_thresholds": self.severity_thresholds
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="error_config_initialization_failed",
                message=f"Failed to initialize error handling configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            self._initialize_config()
            self.logger.record_event(
                event_type="error_config_updated",
                message="Error handling configuration updated",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                event_type="error_config_update_failed",
                message=f"Failed to update error handling configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            
    def _adjust_training_params(self, context: Dict[str, Any]) -> None:
        """Adjust training parameters based on configuration."""
        current_batch_size = self.config_manager.get("training_config.batch_size", 32)
        new_batch_size = max(1, current_batch_size // 2)
        self.config_manager.update("training_config.batch_size", new_batch_size)
        
    def _adjust_curiosity_params(self, context: Dict[str, Any]) -> None:
        """Adjust curiosity parameters based on configuration."""
        current_threshold = self.config_manager.get("curiosity_config.pressure_threshold", 0.5)
        new_threshold = max(0.1, current_threshold - 0.05)
        self.config_manager.update("curiosity_config.pressure_threshold", new_threshold)
        
    def _adjust_memory_params(self, context: Dict[str, Any]) -> None:
        """Adjust memory parameters based on configuration."""
        current_limit = self.config_manager.get("memory_config.max_memory_mb", 512)
        new_limit = max(256, current_limit // 2)
        self.config_manager.update("memory_config.max_memory_mb", new_limit)
        
    def _adjust_generation_params(self, context: Dict[str, Any]) -> None:
        """Adjust generation parameters based on configuration."""
        current_temp = self.config_manager.get("generation_config.temperature", 1.0)
        new_temp = max(0.5, current_temp - 0.05)
        self.config_manager.update("generation_config.temperature", new_temp)
        
    def _adjust_data_params(self, context: Dict[str, Any]) -> None:
        """Adjust data parameters based on configuration."""
        current_batch_size = self.config_manager.get("data_config.batch_size", 32)
        new_batch_size = max(1, current_batch_size // 2)
        self.config_manager.update("data_config.batch_size", new_batch_size)
        
    def _recover_training(self, record: ErrorRecord) -> None:
        """Attempt to recover from a training error."""
        try:
            # Reset training state
            self.state_tracker.reset_training_state()
            
            # Adjust batch size
            current_batch_size = self.context.config_manager.get("training_config.batch_size", 32)
            new_batch_size = max(1, current_batch_size // 2)
            self.context.config_manager.update("training_config.batch_size", new_batch_size)
            
            self.logger.record_event(
                event_type="training_recovery",
                message="Reset training state and reduced batch size",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_batch_size": current_batch_size,
                    "new_batch_size": new_batch_size,
                    **record.additional_info
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="training_recovery_failed",
                message=f"Failed to recover from training error: {str(e)}",
                level="critical",
                additional_info={
                    "error_type": record.error_type,
                    **record.additional_info
                }
            )

    def _recover_curiosity(self, record: ErrorRecord) -> None:
        """Attempt to recover from a curiosity error."""
        try:
            # Reset curiosity state
            self.state_tracker.reset_curiosity_state()
            
            # Adjust pressure threshold
            current_threshold = self.context.config_manager.get("curiosity_config.pressure_threshold", 0.5)
            new_threshold = max(0.1, current_threshold - 0.05)
            self.context.config_manager.update("curiosity_config.pressure_threshold", new_threshold)
            
            self.logger.record_event(
                event_type="curiosity_recovery",
                message="Reset curiosity state and reduced pressure threshold",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_threshold": current_threshold,
                    "new_threshold": new_threshold,
                    **record.additional_info
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="curiosity_recovery_failed",
                message=f"Failed to recover from curiosity error: {str(e)}",
                level="critical",
                additional_info={
                    "error_type": record.error_type,
                    **record.additional_info
                }
            )

    def _recover_memory(self, record: ErrorRecord) -> None:
        """Attempt to recover from a memory error."""
        try:
            # Clear memory cache
            self.state_tracker.clear_memory_cache()
            
            # Reduce memory limit
            current_limit = self.context.config_manager.get("memory_config.max_memory_mb", 512)
            new_limit = max(256, current_limit // 2)
            self.context.config_manager.update("memory_config.max_memory_mb", new_limit)
            
            self.logger.record_event(
                event_type="memory_recovery",
                message="Cleared memory cache and reduced memory limit",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_limit": current_limit,
                    "new_limit": new_limit,
                    **record.additional_info
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="memory_recovery_failed",
                message=f"Failed to recover from memory error: {str(e)}",
                level="critical",
                additional_info={
                    "error_type": record.error_type,
                    **record.additional_info
                }
            )

    def _recover_generation(self, record: ErrorRecord) -> None:
        """Attempt to recover from a generation error."""
        try:
            # Reset generation state
            self.state_tracker.reset_generation_state()
            
            # Adjust temperature
            current_temp = self.context.config_manager.get("generation_config.temperature", 1.0)
            new_temp = max(0.5, current_temp - 0.05)
            self.context.config_manager.update("generation_config.temperature", new_temp)
            
            self.logger.record_event(
                event_type="generation_recovery",
                message="Reset generation state and reduced temperature",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_temperature": current_temp,
                    "new_temperature": new_temp,
                    **record.additional_info
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="generation_recovery_failed",
                message=f"Failed to recover from generation error: {str(e)}",
                level="critical",
                additional_info={
                    "error_type": record.error_type,
                    **record.additional_info
                }
            )

    def _recover_data(self, record: ErrorRecord) -> None:
        """Attempt to recover from a data error."""
        try:
            # Reset data state
            self.state_tracker.reset_data_state()
            
            # Reduce batch size
            current_batch_size = self.context.config_manager.get("data_config.batch_size", 32)
            new_batch_size = max(1, current_batch_size // 2)
            self.context.config_manager.update("data_config.batch_size", new_batch_size)
            
            self.logger.record_event(
                event_type="data_recovery",
                message="Reset data state and reduced batch size",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_batch_size": current_batch_size,
                    "new_batch_size": new_batch_size,
                    **record.additional_info
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="data_recovery_failed",
                message=f"Failed to recover from data error: {str(e)}",
                level="critical",
                additional_info={
                    "error_type": record.error_type,
                    **record.additional_info
                }
            )

    def handle_generation_error(
        self,
        error: Exception,
        context: str,
        additional_info: Optional[Dict[str, Any]] = None,
        state: Optional[SOVLState] = None
    ) -> None:
        """Handle generation-specific errors with enhanced context and recovery."""
        try:
            # Gather comprehensive error context
            error_context = self.gather_error_context(error, context, state, additional_info)
            
            # Select and apply recovery strategy
            strategy = self.select_recovery_strategy(error, context)
            recovery_result = self._apply_recovery_strategy(strategy, error_context)
            
            # Update error context with recovery information
            error_context["recovery"] = recovery_result
            
            # Record the error
            self.record_error(
                error=error,
                error_type=f"generation_{context}",
                context=error_context
            )
            
            # Log error with recovery information
            self.logger.log_error(
                error_msg=str(error),
                error_type=type(error).__name__,
                stack_trace=traceback.format_exc(),
                additional_info={
                    "context": context,
                    "recovery_strategy": strategy,
                    "recovery_success": recovery_result["success"]
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to handle generation error: {str(e)}",
                error_type="generation_error_handling_error",
                stack_trace=traceback.format_exc()
            )

    def _apply_recovery_strategy(self, strategy: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the selected recovery strategy with enhanced error handling."""
        result = {"strategy": strategy, "success": False, "details": {}}
        
        try:
            if strategy == "full_memory_reset":
                # Reset both RAM and GPU memory
                if hasattr(self.context, 'memory_manager'):
                    self.context.memory_manager.manage_memory()
                if hasattr(self.context, 'gpu_manager'):
                    self.context.gpu_manager.manage_memory()
                result["details"]["actions"] = ["ram_cleanup", "gpu_cleanup"]
                result["success"] = True
                
            elif strategy == "gpu_memory_cleanup":
                # Clean up GPU memory
                if hasattr(self.context, 'gpu_manager'):
                    self.context.gpu_manager.manage_memory()
                result["details"]["actions"] = ["gpu_cleanup"]
                result["success"] = True
                
            elif strategy == "ram_memory_cleanup":
                # Clean up RAM memory
                if hasattr(self.context, 'ram_manager'):
                    self.context.ram_manager.manage_memory()
                result["details"]["actions"] = ["ram_cleanup"]
                result["success"] = True
                
            elif strategy == "device_reset":
                # Reset device state
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                result["details"]["actions"] = ["device_reset"]
                result["success"] = True
                
            elif strategy == "config_reset":
                # Reset configuration
                if hasattr(self.context, 'config_manager'):
                    self.context.config_manager.reset_section("generation_config")
                result["details"]["actions"] = ["config_reset"]
                result["success"] = True
                
            else:
                # Default strategy: try memory cleanup first
                if hasattr(self.context, 'memory_manager'):
                    self.context.memory_manager.manage_memory()
                result["details"]["actions"] = ["default_cleanup"]
                result["success"] = True
                
            # Log recovery attempt
            self.logger.log_info(
                message=f"Applied recovery strategy: {strategy}",
                details=result["details"]
            )
            
            return result
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to apply recovery strategy: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc()
            )
            result["details"]["error"] = str(e)
            return result

    def error_handler(context: str):
        """Decorator for consistent error handling and logging."""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    state = kwargs.get('state', None)
                    self.error_manager.handle_generation_error(
                        error=e,
                        context=context,
                        state=state
                    )
                    raise
            return wrapper
        return decorator

class ErrorHandler:
    """Handles error logging, recovery, and monitoring for the SOVL system."""

    _DEFAULT_CONFIG = {
        "error_handling.max_history_per_error": 10,
        "error_handling.critical_threshold": 5,
        "error_handling.warning_threshold": 10,
        "error_handling.retry_attempts": 3,
        "error_handling.retry_delay": 1.0,
        "error_handling.memory_recovery_attempts": 3,
        "error_handling.memory_recovery_delay": 1.0,
    }

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Logger,
        error_log_file: str = "sovl_errors.jsonl",
        max_error_log_size_mb: int = 10,
        compress_old: bool = True,
    ):
        """Initialize the error handler with configuration and logging dependencies."""
        self.config = {**self._DEFAULT_CONFIG, **config}
        self.logger = logger
        self.lock = Lock()
        self.error_counts = defaultdict(int)
        self.error_history = defaultdict(lambda: deque(maxlen=self._get_max_history_per_error()))
        self.severity_thresholds = self._load_severity_thresholds()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self._validate_config()

    def _get_max_history_per_error(self) -> int:
        """Get the maximum number of error instances to keep in history."""
        return self.config["error_handling.max_history_per_error"]

    def _load_severity_thresholds(self) -> Dict[str, int]:
        """Load severity thresholds from configuration."""
        return {
            "critical": self.config["error_handling.critical_threshold"],
            "warning": self.config["error_handling.warning_threshold"]
        }

    def _initialize_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            "model_loading": self._recover_model_loading,
            "training": self._recover_training,
            "generation": self._recover_generation,
            "memory": self._recover_memory,
            "default": self._recover_default
        }

    def _validate_config(self) -> None:
        """Validate error handling configuration."""
        try:
            # Ensure all required keys are present with valid values
            for key, (default, (min_val, max_val)) in self._DEFAULT_CONFIG.items():
                if key not in self.config:
                    self.config[key] = default
                    self.logger.record_event(
                        event_type="error_config_missing_key",
                        message=f"Added missing error handling key {key} with default {default}",
                        level="warning"
                    )
                value = self.config[key]
                if not (min_val <= value <= max_val):
                    self.config[key] = default
                    self.logger.record_event(
                        event_type="error_config_invalid_value",
                        message=f"Invalid {key}: {value}. Reset to default {default}",
                        level="warning"
                    )
            
            # Log final configuration
            self.logger.record_event(
                event_type="error_handler_config_validated",
                message="Error handler configuration validated successfully",
                level="info",
                additional_info={"config": self.config}
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="error_config_validation_failed",
                message=f"Failed to validate error handler configuration: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _log_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        severity: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error details using the standardized logger interface."""
        error_info = {
            "context": context,
            "phase": phase,
            "severity": severity,
            "timestamp": time.time(),
            **(additional_info or {})
        }
        
        self.logger.log_error(
            error_msg=str(error),
            error_type=f"{context}_{phase}_error",
            stack_trace=traceback.format_exc(),
            additional_info=error_info
        )

    def _get_state_hash(self) -> Optional[str]:
        """Safely get the current state hash with proper locking."""
        if not self.state:
            return None
        with self.state.lock:
            return self.state.state_hash()

    def _get_state_info(self) -> Dict[str, Any]:
        """Safely get state information with proper locking."""
        if not self.state:
            return {}
        with self.state.lock:
            return {
                "state_hash": self.state.state_hash(),
                "conversation_id": self.state.history.conversation_id if hasattr(self.state, 'history') else None
            }

    def handle_generation_error(self, error: Exception, prompt: str, state: Optional[SOVLState] = None) -> str:
        """Handle generation errors and return a fallback response."""
        with self.lock:
            error_key = f"generation:generate:{type(error).__name__}"
            self.record_error(
                error=error,
                context="generation",
                phase="generate",
                additional_info={
                    "prompt": prompt[:200],  # Truncate for logging
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_generation(error_key)
                
        return "An error occurred during generation"

    def handle_training_error(self, error: Exception, batch_size: int, state: Optional[SOVLState] = None) -> None:
        """Handle training errors."""
        with self.lock:
            error_key = f"training:train_step:{type(error).__name__}"
            self.record_error(
                error=error,
                context="training",
                phase="train_step",
                additional_info={
                    "batch_size": batch_size,
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_training(error_key)

    def handle_data_loading_error(self, error: Exception, file_path: str) -> None:
        """Handle data loading errors."""
        self.record_error(
            error=error,
            context="data_loading",
            phase="load_training_data",
            additional_info={
                "file_path": file_path,
                "is_error_prompt": True
            },
            severity="error"
        )

    def handle_curiosity_error(self, error: Exception, event_type: str, state: Optional[SOVLState] = None) -> None:
        """Handle curiosity-related errors."""
        with self.lock:
            error_key = f"curiosity:{event_type}:{type(error).__name__}"
            self.record_error(
                error=error,
                context="curiosity",
                phase=event_type,
                additional_info={
                    "error_type": "generation_error" if event_type == "question_generation" else "curiosity_error",
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_curiosity(error_key)

    def handle_memory_error(self, error: Exception, model_size: int, state: Optional[SOVLState] = None) -> None:
        """Handle memory errors and attempt recovery."""
        with self.lock:
            error_key = f"memory:check_health:{type(error).__name__}"
            self.record_error(
                error=error,
                context="memory",
                phase="check_health",
                additional_info={
                    "model_size": model_size,
                    "state_hash": state.state_hash() if state else None,
                    "conversation_id": state.history.conversation_id if state else None,
                    "device": str(state.device) if state else None,
                    "memory_stats": self._get_memory_stats() if torch.cuda.is_available() else None
                },
                severity="error"
            )
            
            # Attempt recovery if error count exceeds threshold
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._recover_memory(error_key)

    def record_error(
        self,
        error: Exception,
        context: str,
        phase: str,
        additional_info: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        reraise: bool = False
    ) -> None:
        """
        Record an error with detailed context and handle it according to severity.

        Args:
            error: The exception that was raised.
            context: The context in which the error occurred (e.g., "training", "generation").
            phase: The specific phase or operation where the error occurred.
            additional_info: Additional context-specific information.
            severity: The severity level of the error ("error", "warning", "critical").
            reraise: Whether to re-raise the exception after handling.
        """
        with self.lock:
            error_key = f"{context}:{phase}:{type(error).__name__}"
            self.error_counts[error_key] += 1
            self.error_history[error_key].append({
                "timestamp": time.time(),
                "error": str(error),
                "stack_trace": traceback.format_exc(),
                "severity": severity,
                "additional_info": additional_info or {}
            })

            # Log the error using standardized format
            self._log_error(error, context, phase, severity, additional_info)

            # Log the event using record_event
            self.logger.record_event(
                event_type=f"error_{context}_{phase}",
                message=f"Error recorded: {str(error)}",
                level=severity,
                additional_info={
                    "context": context,
                    "phase": phase,
                    "stack_trace": traceback.format_exc(),
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    **(additional_info or {})
                }
            )

            # Check if we've exceeded thresholds
            if self.error_counts[error_key] >= self.severity_thresholds["critical"]:
                self._handle_critical_error(error_key)
            elif self.error_counts[error_key] >= self.severity_thresholds["warning"]:
                self._handle_warning(error_key)

            if reraise:
                raise error

    def _handle_critical_error(self, error_key: str) -> None:
        """Handle critical errors that exceed the threshold."""
        error_history = list(self.error_history[error_key])
        self.logger.record_event(
            event_type="critical_error_threshold_exceeded",
            message=f"Critical error threshold exceeded for {error_key}",
            level="critical",
            additional_info={
                "error_key": error_key,
                "count": self.error_counts[error_key],
                "recent_errors": error_history[-5:],  # Last 5 errors
                "timestamp": time.time()
            }
        )

        # Attempt recovery based on error context
        context = error_key.split(":")[0]
        recovery_strategy = self.recovery_strategies.get(context, self.recovery_strategies["default"])
        try:
            recovery_strategy(error_key)
        except Exception as e:
            self._log_error(
                error=e,
                context="error_recovery",
                phase="critical_error",
                severity="error",
                additional_info={"error_key": error_key}
            )

    def _handle_warning(self, error_key: str) -> None:
        """Handle warnings that exceed the threshold."""
        self.logger.record_event(
            event_type="warning_threshold_exceeded",
            message=f"Warning threshold exceeded for {error_key}",
            level="warning",
            additional_info={
                "error_key": error_key,
                "count": self.error_counts[error_key],
                "timestamp": time.time()
            }
        )

    def _recover_model_loading(self, error_key: str) -> None:
        """Recovery strategy for model loading errors."""
        try:
            self.logger.record_event(
                event_type="model_loading_recovery_attempt",
                message=f"Attempting model loading recovery for {error_key}",
                level="info"
            )
            
            if self.state:
                with self.state.lock:
                    # Clear model cache and reload
                    if hasattr(self.state, 'model'):
                        del self.state.model
                    if hasattr(self.state, 'tokenizer'):
                        del self.state.tokenizer
                        
                    # Reset model configuration
                    self.state.model_config = self.state.initial_model_config.copy()
                    
                    self.logger.record_event(
                        event_type="model_loading_recovery",
                        message="Model cache cleared and configuration reset",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="model_loading_recovery_failed",
                message=f"Model loading recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_training(self, error_key: str) -> None:
        """Recovery strategy for training errors."""
        try:
            self.logger.record_event(
                event_type="training_recovery_attempt",
                message=f"Attempting training recovery for {error_key}",
                level="info"
            )
            
            if self.state:
                with self.state.lock:
                    # Reduce batch size
                    current_batch = self.state.training_config.get("batch_size", 4)
                    new_batch = max(1, current_batch // 2)
                    self.state.training_config["batch_size"] = new_batch
                    
                    # Adjust learning rate
                    current_lr = self.state.training_config.get("learning_rate", 0.0003)
                    new_lr = current_lr * 0.5
                    self.state.training_config["learning_rate"] = new_lr
                    
                    self.logger.record_event(
                        event_type="training_recovery",
                        message=f"Reduced batch size to {new_batch} and learning rate to {new_lr}",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="training_recovery_failed",
                message=f"Training recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_generation(self, error_key: str) -> None:
        """Recovery strategy for generation errors."""
        try:
            self.logger.record_event(
                event_type="generation_recovery_attempt",
                message=f"Attempting generation recovery for {error_key}",
                level="info"
            )
            
            if self.state:
                with self.state.lock:
                    # Reset generation parameters to safer values
                    self.state.generation_config["temperature"] = 0.7
                    self.state.generation_config["top_p"] = 0.9
                    self.state.generation_config["max_length"] = 128
                    
                    # Clear generation cache
                    if hasattr(self.state, 'generation_cache'):
                        self.state.generation_cache.clear()
                    
                    self.logger.record_event(
                        event_type="generation_recovery",
                        message="Generation parameters reset to safer values",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="generation_recovery_failed",
                message=f"Generation recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_memory(self, error_key: str) -> None:
        """Recovery strategy for memory errors."""
        try:
            self.logger.record_event(
                event_type="memory_recovery_attempt",
                message=f"Attempting memory recovery for {error_key}",
                level="info"
            )
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if self.state:
                with self.state.lock:
                    # Reduce memory-intensive parameters
                    if "batch_size" in self.state.training_config:
                        self.state.training_config["batch_size"] = max(1, self.state.training_config["batch_size"] // 2)
                    
                    # Clear caches
                    if hasattr(self.state, 'cache'):
                        self.state.cache.clear()
                    
                    self.logger.record_event(
                        event_type="memory_recovery",
                        message="Memory cache cleared and batch size reduced",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="memory_recovery_failed",
                message=f"Memory recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def _recover_default(self, error_key: str) -> None:
        """Default recovery strategy for unhandled error types."""
        self.logger.record({
            "event": "attempting_default_recovery",
            "error_key": error_key,
            "timestamp": time.time()
        })
        # Implement default recovery logic here

    def _recover_curiosity(self, error_key: str) -> None:
        """Recovery strategy for curiosity errors."""
        try:
            self.logger.record_event(
                event_type="curiosity_recovery_attempt",
                message=f"Attempting curiosity recovery for {error_key}",
                level="info"
            )
            
            if self.state and hasattr(self.state, 'curiosity'):
                with self.state.lock:
                    # Reset curiosity parameters
                    self.state.curiosity.pressure = 0.5
                    self.state.curiosity.novelty_threshold_spontaneous = 0.7
                    self.state.curiosity.novelty_threshold_response = 0.6
                    
                    # Clear question queue
                    if hasattr(self.state.curiosity, 'unanswered_questions'):
                        self.state.curiosity.unanswered_questions.clear()
                    
                    self.logger.record_event(
                        event_type="curiosity_recovery",
                        message="Curiosity parameters reset to default values",
                        level="info"
                    )
                    
        except Exception as e:
            self.logger.record_event(
                event_type="curiosity_recovery_failed",
                message=f"Curiosity recovery failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc()
            )

    def get_error_summary(self) -> Dict[str, Any]:
        """Return a summary of recorded errors."""
        with self.lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_types": dict(self.error_counts),
                "recent_errors": {
                    error_type: list(history)[-5:]
                    for error_type, history in self.error_history.items()
                },
            }

    def clear_error_history(self) -> None:
        """Clear error counts and history."""
        with self.lock:
            self.error_counts.clear()
            self.error_history.clear()
            self.logger.record({
                "event": "error_history_cleared",
                "timestamp": time.time(),
            })

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics if CUDA is available."""
        if not torch.cuda.is_available():
            return {}
            
        try:
            gpu_manager = GPUMemoryManager(self.config_manager, self.logger)
            gpu_stats = gpu_manager.get_gpu_usage()
            
            return {
                "allocated": gpu_stats.get('gpu_usage', 0.0),
                "reserved": gpu_stats.get('gpu_reserved', 0.0),
                "max_allocated": gpu_stats.get('max_gpu_usage', 0.0),
                "max_reserved": gpu_stats.get('max_gpu_reserved', 0.0),
                "device_count": gpu_stats.get('device_count', 0),
                "current_device": gpu_stats.get('current_device', 0),
                "device_name": gpu_stats.get('device_name', 'unknown')
            }
        except Exception as e:
            self.logger.record_event(
                event_type="memory_stats_error",
                message=f"Failed to get memory stats: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            return {}
