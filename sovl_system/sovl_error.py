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

class ConfigurationError(Exception):
    """Raised when there is an error related to configuration."""
    pass

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
                "data": self._recover_data,
                "model_loading": self._recover_model_loading
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
            
    def _recover_training(self, record: ErrorRecord) -> None:
        """Attempt to recover from a training error."""
        try:
            # Reset training state
            self.state_tracker.reset_training_state()
            
            # Adjust batch size
            current_batch_size = self.config_manager.get("training_config.batch_size", 32)
            new_batch_size = max(1, current_batch_size // 2)
            self.config_manager.update("training_config.batch_size", new_batch_size)
            
            # Adjust learning rate
            current_lr = self.config_manager.get("training_config.learning_rate", 0.0003)
            new_lr = current_lr * 0.5
            self.config_manager.update("training_config.learning_rate", new_lr)
            
            self.logger.record_event(
                event_type="training_recovery",
                message="Reset training state, reduced batch size and learning rate",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_batch_size": current_batch_size,
                    "new_batch_size": new_batch_size,
                    "old_learning_rate": current_lr,
                    "new_learning_rate": new_lr,
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
            
            # Reset curiosity parameters
            curiosity_params = {
                "pressure": 0.5,
                "novelty_threshold_spontaneous": 0.7,
                "novelty_threshold_response": 0.6
            }
            
            # Update parameters via config manager
            for param, value in curiosity_params.items():
                self.config_manager.update(f"curiosity_config.{param}", value)
            
            # Clear question queue if available
            if hasattr(self.context, 'curiosity_manager'):
                self.context.curiosity_manager.clear_question_queue()
            
            self.logger.record_event(
                event_type="curiosity_recovery",
                message="Reset curiosity state and parameters",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "parameters": curiosity_params,
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
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Clear memory cache
            self.state_tracker.clear_memory_cache()
            
            # Reduce memory limit
            current_limit = self.config_manager.get("memory_config.max_memory_mb", 512)
            new_limit = max(256, current_limit // 2)
            self.config_manager.update("memory_config.max_memory_mb", new_limit)
            
            # Reduce training batch size
            current_batch = self.config_manager.get("training_config.batch_size", 32)
            new_batch = max(1, current_batch // 2)
            self.config_manager.update("training_config.batch_size", new_batch)
            
            # Clear system caches if available
            if hasattr(self.context, 'memory_manager'):
                self.context.memory_manager.clear_caches()
            
            self.logger.record_event(
                event_type="memory_recovery",
                message="Cleared memory caches and reduced limits",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "old_memory_limit": current_limit,
                    "new_memory_limit": new_limit,
                    "old_batch_size": current_batch,
                    "new_batch_size": new_batch,
                    "cuda_cache_cleared": torch.cuda.is_available(),
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
            
            # Reset generation parameters to safer values
            generation_params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_length": 128
            }
            
            # Update parameters via config manager
            for param, value in generation_params.items():
                self.config_manager.update(f"generation_config.{param}", value)
            
            # Clear generation cache if available
            if hasattr(self.context, 'generation_manager'):
                self.context.generation_manager.clear_cache()
            
            self.logger.record_event(
                event_type="generation_recovery",
                message="Reset generation state and parameters",
                level="info",
                additional_info={
                    "error_type": record.error_type,
                    "parameters": generation_params,
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
            current_batch_size = self.config_manager.get("data_config.batch_size", 32)
            new_batch_size = max(1, current_batch_size // 2)
            self.config_manager.update("data_config.batch_size", new_batch_size)
            
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

    def _recover_model_loading(self, record: ErrorRecord) -> None:
        """Recovery strategy for model loading errors."""
        try:
            self.logger.record_event(
                event_type="model_loading_recovery_attempt",
                message=f"Attempting model loading recovery for {record.error_type}",
                level="info"
            )
            
            # Clear model cache and reload
            if hasattr(self.context, 'model_manager'):
                self.context.model_manager.clear_model_cache()
                self.context.model_manager.reload_model()
                
            # Reset model configuration
            if hasattr(self.context, 'config_manager'):
                self.context.config_manager.reset_section("model_config")
                
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
