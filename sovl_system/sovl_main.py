from typing import Optional, Any, List, Dict, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import time
import random
import bitsandbytes as bnb
import json
from collections import deque, defaultdict
import traceback
import os
from threading import Lock
from sovl_curiosity import CuriosityManager, CuriosityState
from sovl_logger import Logger
from sovl_io import validate_quantization_mode, InsufficientDataError
from sovl_state import SOVLState, ConversationHistory
from sovl_trainer import TrainingConfig, SOVLTrainer, TrainingCycleManager
from sovl_config import ConfigManager, ConfigHandler, ValidationSchema
from sovl_scaffold import CrossAttentionInjector, ScaffoldManager, CrossAttentionLayer, ScaffoldTokenMapper
from sovl_processor import LogitsProcessor, SOVLProcessor
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_memory import MemoryManager
from sovl_manager import ModelManager
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner
from sovl_error import ErrorHandler
from sovl_state import StateManager
from sovl_grafter import PluginManager
from sovl_confidence import calculate_confidence_score
from sovl_events import EventDispatcher
from sovl_interfaces import SystemInterface
from sovl_utils import (
    detect_repetitions,
    safe_compare,
    float_gt,
    synchronized,
    validate_components,
    NumericalGuard,
    initialize_component_state,
    sync_component_states,
    validate_component_states
)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sovl_conductor import SOVLOrchestrator

class SystemContext:
    """Manages system-wide context and resources."""
    
    def __init__(self, config_path: str, device: str = "cuda"):
        """
        Initialize system context with shared resources.
        
        Args:
            config_path: Path to configuration file
            device: Device to use for tensor operations
        """
        self.config_path = config_path
        self.device = device
        self.logger = Logger()
        self.event_dispatcher = EventDispatcher(self.logger)
        
        # Initialize config manager with event dispatcher
        self.config_handler = ConfigHandler(config_path, self.logger, self.event_dispatcher)
        
        # Validate initial configuration
        if not self.config_handler.validate():
            raise SystemInitializationError(
                "Failed to validate initial configuration",
                config_path,
                traceback.format_exc()
            )
        
        # Subscribe to configuration changes
        self.event_dispatcher.subscribe("config_change", self._on_config_change)
        
    def _on_config_change(self) -> None:
        """Handle configuration changes and propagate them to affected components."""
        try:
            # Validate configuration after change
            if not self.config_handler.validate():
                self.logger.log_error(
                    error_msg="Configuration validation failed after change",
                    error_type="config_validation_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "config_path": self.config_path,
                        "timestamp": time.time()
                    }
                )
                return
                
            self.logger.record_event(
                event_type="config_change",
                message="Configuration changed and validated successfully",
                level="info",
                additional_info={
                    "timestamp": time.time(),
                    "config_path": self.config_path
                }
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="config_change_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_path": self.config_path,
                    "timestamp": time.time()
                }
            )

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

class ModelLoader:
    """Handles model loading and initialization."""
    
    def __init__(self, context: SystemContext):
        """
        Initialize model loader with system context.
        
        Args:
            context: System context containing shared resources
        """
        self.context = context
        self.logger = context.logger
        self.config = context.config_handler.get_section("model")
        
        # Validate model configuration
        if not self._validate_model_config():
            raise ModelLoadingError(
                "Invalid model configuration",
                self.config,
                traceback.format_exc()
            )
            
    def _validate_model_config(self) -> bool:
        """Validate model configuration section."""
        try:
            required_fields = ["model_path", "model_type", "quantization_mode"]
            for field in required_fields:
                if field not in self.config or self.config[field] is None:
                    self.logger.log_error(
                        error_msg=f"Missing or invalid model configuration field: {field}",
                        error_type="config_validation_error",
                        stack_trace=None,
                        additional_info={
                            "missing_field": field,
                            "config_section": "model"
                        }
                    )
                    return False
                    
            # Validate quantization mode
            if not validate_quantization_mode(self.config["quantization_mode"]):
                self.logger.log_error(
                    error_msg=f"Invalid quantization mode: {self.config['quantization_mode']}",
                    error_type="config_validation_error",
                    stack_trace=None,
                    additional_info={
                        "invalid_value": self.config["quantization_mode"],
                        "config_section": "model"
                    }
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to validate model configuration: {str(e)}",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_section": "model"
                }
            )
            return False
            
    def load_model(self) -> torch.nn.Module:
        """Load and initialize the model with validated configuration."""
        try:
            if not self._validate_model_config():
                raise ModelLoadingError(
                    "Cannot load model with invalid configuration",
                    self.config,
                    traceback.format_exc()
                )
                
            # Load model with validated configuration
            model = self._load_model(
                self.config["model_path"],
                self.config["model_type"],
                self.config["quantization_mode"],
                self.context.device
            )
            
            self.logger.record_event(
                event_type="model_loaded",
                message="Model loaded successfully",
                level="info",
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"],
                    "device": self.context.device
                }
            )
            
            return model
            
        except NameError as e:
            self.logger.log_error(
                error_msg=f"Model loading function not defined: {str(e)}",
                error_type="model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"]
                }
            )
            raise ModelLoadingError("Model loading function is not defined.") from e
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"]
                }
            )
            raise

            
    def load_model(self) -> torch.nn.Module:
        """Load and initialize the model with validated configuration."""
        try:
            if not self._validate_model_config():
                raise ModelLoadingError(
                    "Cannot load model with invalid configuration",
                    self.config,
                    traceback.format_exc()
                )
                
            # Load model with validated configuration
            model = self._load_model(
                self.config["model_path"],
                self.config["model_type"],
                self.config["quantization_mode"],
                self.context.device
            )
            
            self.logger.record_event(
                event_type="model_loaded",
                message="Model loaded successfully",
                level="info",
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"],
                    "device": self.context.device
                }
            )
            
            return model
            
        except NameError as e:
            self.logger.log_error(
                error_msg=f"Model loading function not defined: {str(e)}",
                error_type="model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"]
                }
            )
            raise ModelLoadingError("Model loading function is not defined.") from e
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": self.config["model_path"],
                    "model_type": self.config["model_type"],
                    "quantization_mode": self.config["quantization_mode"]
                }
            )
            raise

    def _load_model(self, model_path: str, model_type: str, quantization_mode: str, device: str) -> nn.Module:
        """Load the model based on the provided parameters."""
        # Placeholder for actual model loading logic
        # This should be implemented to load the model based on the type and quantization
        raise NotImplementedError("Model loading logic is not implemented.")

class StateTracker:
    """Tracks system state and history."""
    
    def __init__(self, context: SystemContext):
        """Initialize state tracker with system context."""
        self.context = context
        self.state = None
        self._state_history = deque(maxlen=100)  # Keep last 100 states
        self._state_changes = deque(maxlen=50)  # Keep last 50 state changes
        self._lock = Lock()
        
    def _validate_state_config(self) -> bool:
        """Validate state configuration."""
        try:
            config = self.context.config_handler.get_section("state")
            if not config:
                self.context.logger.log_error(
                    error_msg="Missing state configuration section",
                    error_type="config_validation_error"
                )
                return False
                
            required_fields = ["max_history", "state_file"]
            for field in required_fields:
                if field not in config:
                    self.context.logger.log_error(
                        error_msg=f"Missing required state configuration field: {field}",
                        error_type="config_validation_error",
                        additional_info={"missing_field": field}
                    )
                    return False
                    
            return True
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to validate state configuration: {str(e)}",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc()
            )
            return False
            
    def get_state(self) -> Dict[str, Any]:
        """Get current system state."""
        with self._lock:
            if not self.state:
                return {}
            return self.state.to_dict()
            
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent state history."""
        with self._lock:
            return [state.to_dict() for state in list(self._state_history)[-limit:]]
            
    def get_state_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent state changes."""
        with self._lock:
            return list(self._state_changes)[-limit:]
            
    def get_state_stats(self) -> Dict[str, Any]:
        """Get state tracking statistics."""
        with self._lock:
            return {
                "total_states": len(self._state_history),
                "total_changes": len(self._state_changes),
                "current_state_age": time.time() - self.state.timestamp if self.state else None,
                "last_change_time": self._state_changes[-1]["timestamp"] if self._state_changes else None,
                "state_types": {
                    state_type: count
                    for state_type, count in Counter(
                        change["type"] for change in self._state_changes
                    ).items()
                }
            }
            
    def update_state(self, key: str, value: Any) -> None:
        """Update state with new value and record the change."""
        with self._lock:
            if not self.state:
                self.state = SOVLState()
                
            old_value = getattr(self.state, key, None)
            setattr(self.state, key, value)
            
            # Record state change
            change = {
                "type": "state_update",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "timestamp": time.time()
            }
            self._state_changes.append(change)
            
            # Add current state to history
            self._state_history.append(self.state.copy())
            
            # Log state change
            self.context.logger.record_event(
                event_type="state_change",
                message=f"State updated: {key}",
                level="debug" if self.context.logger.is_debug_enabled() else "info",
                additional_info={
                    "key": key,
                    "old_value": old_value,
                    "new_value": value,
                    "state_hash": self.state.state_hash
                }
            )
            
    def clear_history(self) -> None:
        """Clear state history and changes."""
        with self._lock:
            self._state_history.clear()
            self._state_changes.clear()
            
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information about state tracking."""
        with self._lock:
            return {
                "current_state": self.get_state(),
                "state_stats": self.get_state_stats(),
                "recent_changes": self.get_state_changes(5),
                "recent_history": self.get_state_history(5),
                "memory_usage": {
                    "state_history_size": len(self._state_history),
                    "state_changes_size": len(self._state_changes),
                    "current_state_size": sys.getsizeof(self.state) if self.state else 0
                }
            }

class ErrorManager:
    """Manages error handling and recovery for the SOVL system."""
    
    def __init__(self, context: SystemContext, state_tracker: StateTracker):
        """Initialize error manager with required dependencies."""
        self.context = context
        self.state_tracker = state_tracker
        self.logger = context.logger
        self.error_counts = defaultdict(int)
        self.recent_errors = deque(maxlen=100)
        self._lock = Lock()
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize error handling configuration."""
        config = self.context.config_handler.config_manager.get_section("error_config", {})
        self.error_cooldown = float(config.get("error_cooldown", 1.0))
        self.severity_thresholds = {
            "warning": float(config.get("warning_threshold", 3.0)),
            "error": float(config.get("error_threshold", 5.0)),
            "critical": float(config.get("critical_threshold", 10.0))
        }
        self.recovery_actions = {
            "training": self._recover_training,
            "curiosity": self._recover_curiosity,
            "memory": self._recover_memory,
            "generation": self._recover_generation,
            "data": self._recover_data
        }
        self.logger.record_event(
            event_type="error_manager_initialized",
            message="Error manager initialized successfully",
            level="info",
            additional_info={"error_cooldown": self.error_cooldown}
        )
        
    def _is_duplicate_error(self, error: Exception, error_type: str) -> bool:
        """Check if this error is a duplicate within the cooldown period."""
        error_key = f"{error_type}:{type(error).__name__}"
        current_time = time.time()
        
        with self._lock:
            # Remove old errors from tracking
            while self.recent_errors and float_gt(current_time - self.recent_errors[0]["timestamp"], self.error_cooldown):
                self.recent_errors.popleft()
                
            # Check for duplicates
            for recent_error in self.recent_errors:
                if recent_error["key"] == error_key:
                    return True
                    
            # Add to recent errors
            self.recent_errors.append({
                "key": error_key,
                "timestamp": current_time
            })
        
        return False
        
    def handle_training_error(self, error: Exception, batch_size: int) -> None:
        """Handle training-related errors."""
        try:
            error_type = "training_error"
            self._record_error(error, error_type, {"batch_size": batch_size})
            
            if not self._is_duplicate_error(error, error_type):
                self.logger.log_error(
                    error_msg=str(error),
                    error_type=error_type,
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "batch_size": batch_size,
                        "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                    }
                )
                
            self._recover_training(error_type)
            self._adjust_training_parameters(batch_size)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error handler failed: {str(e)}",
                error_type="error_handler_failure",
                stack_trace=traceback.format_exc()
            )
            
    def handle_curiosity_error(self, error: Exception, pressure: float) -> None:
        """Handle curiosity-related errors."""
        try:
            error_type = "curiosity_error"
            self._record_error(error, error_type, {"pressure": pressure})
            
            if not self._is_duplicate_error(error, error_type):
                self.logger.log_error(
                    error_msg=str(error),
                    error_type=error_type,
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "pressure": pressure,
                        "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                    }
                )
                
            self._recover_curiosity(error_type)
            self._adjust_curiosity_parameters(pressure)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error handler failed: {str(e)}",
                error_type="error_handler_failure",
                stack_trace=traceback.format_exc()
            )
            
    def handle_memory_error(self, error: Exception, memory_usage: float) -> None:
        """Handle memory-related errors."""
        try:
            error_type = "memory_error"
            self._record_error(error, error_type, {"memory_usage": memory_usage})
            
            if not self._is_duplicate_error(error, error_type):
                self.logger.log_error(
                    error_msg=str(error),
                    error_type=error_type,
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "memory_usage": memory_usage,
                        "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                    }
                )
                
            self._recover_memory(error_type)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error handler failed: {str(e)}",
                error_type="error_handler_failure",
                stack_trace=traceback.format_exc()
            )
            
    def handle_generation_error(self, error: Exception, temperature: float) -> None:
        """Handle generation-related errors."""
        try:
            error_type = "generation_error"
            self._record_error(error, error_type, {"temperature": temperature})
            
            if not self._is_duplicate_error(error, error_type):
                self.logger.log_error(
                    error_msg=str(error),
                    error_type=error_type,
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "temperature": temperature,
                        "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                    }
                )
                
            self._recover_generation(error_type)
            self._adjust_generation_parameters(temperature)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error handler failed: {str(e)}",
                error_type="error_handler_failure",
                stack_trace=traceback.format_exc()
            )
            
    def handle_data_error(self, error: Exception, context: Dict[str, Any], conversation_id: str) -> None:
        """Handle data-related errors with duplicate detection."""
        try:
            error_key = f"data:{type(error).__name__}"
            
            if self._is_duplicate_error(error, "data"):
                self.logger.record_event(
                    event_type="duplicate_data_error",
                    message=f"Duplicate data error detected: {error_key}",
                    level="warning",
                    additional_info={
                        "error_key": error_key,
                        "context": context,
                        "conversation_id": conversation_id
                    }
                )
                return
                
            self.error_counts[error_key] += 1
            
            self.logger.log_error(
                error_msg=str(error),
                error_type="data_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "error_key": error_key,
                    "error_count": self.error_counts[error_key],
                    "context": context,
                    "conversation_id": conversation_id
                }
            )
            
            if safe_compare(self.error_counts[error_key], self.severity_thresholds["critical"], mode='gt', logger=self.logger):
                self._recover_data(error_key)
            elif safe_compare(self.error_counts[error_key], self.severity_thresholds["error"], mode='gt', logger=self.logger):
                self._adjust_data_parameters(context)
                
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error handler failed: {str(e)}",
                error_type="error_handler_failure",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "original_error": str(error),
                    "context": context,
                    "conversation_id": conversation_id
                }
            )

    def _recover_training(self, error_key: str) -> None:
        """Recover from critical training errors."""
        try:
            self.error_counts[error_key] = 0
            
            self.context.config_handler.config_manager.update("training_config.batch_size", 1)
            self.context.config_handler.config_manager.update("training_config.learning_rate", 1e-5)
            
            self.logger.record_event(
                event_type="training_recovery",
                message="Recovered from critical training error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to recover from training error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_key": error_key}
            )
            
    def _adjust_training_parameters(self, batch_size: int) -> None:
        """Adjust training parameters for non-critical errors."""
        try:
            new_batch_size = max(1, batch_size // 2)
            self.context.config_handler.config_manager.update("training_config.batch_size", new_batch_size)
            
            self.logger.record_event(
                event_type="training_adjustment",
                message="Adjusted training parameters",
                level="info",
                additional_info={
                    "old_batch_size": batch_size,
                    "new_batch_size": new_batch_size
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to adjust training parameters: {str(e)}",
                error_type="adjustment_error",
                stack_trace=traceback.format_exc()
            )
            
    def _recover_curiosity(self, error_key: str) -> None:
        """Recover from critical curiosity errors."""
        try:
            self.error_counts[error_key] = 0
            
            self.context.config_handler.config_manager.update("curiosity_config.pressure_threshold", 0.5)
            self.context.config_handler.config_manager.update("curiosity_config.decay_rate", 0.9)
            
            self.logger.record_event(
                event_type="curiosity_recovery",
                message="Recovered from critical curiosity error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to recover from curiosity error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_key": error_key}
            )
            
    def _recover_memory(self, error_key: str) -> bool:
        """Recover from critical memory errors."""
        try:
            self.error_counts[error_key] = 0
            
            self.context.config_handler.config_manager.update("memory_config.max_memory_mb", 512)
            self.context.config_handler.config_manager.update("memory_config.garbage_collection_threshold", 0.7)
            
            self.logger.record_event(
                event_type="memory_recovery",
                message="Recovered from critical memory error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to recover from memory error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_key": error_key}
            )
            return False
            
        except Exception as e:
            self.logger.record_event(
                event_type="recovery_failed",
                message=f"Failed to recover from memory error: {str(e)}",
                level="critical",
                additional_info={"error_key": error_key}
            )
            return False
            
    def _adjust_curiosity_parameters(self, pressure: float) -> None:
        """Adjust curiosity parameters for non-critical errors."""
        try:
            current_pressure = self.context.config_handler.config_manager.get("curiosity_config.pressure_threshold", 0.5)
            new_pressure = max(0.1, current_pressure - 0.05)
            self.context.config_handler.config_manager.update("curiosity_config.pressure_threshold", new_pressure)
            
            self.logger.record_event(
                event_type="curiosity_adjustment",
                message="Adjusted curiosity parameters",
                level="info",
                additional_info={
                    "old_pressure": current_pressure,
                    "new_pressure": new_pressure,
                    "pressure": pressure
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to adjust curiosity parameters: {str(e)}",
                error_type="adjustment_error",
                stack_trace=traceback.format_exc()
            )
            
    def _recover_generation(self, error_key: str) -> str:
        """Recover from critical generation errors."""
        try:
            self.error_counts[error_key] = 0
            
            self.context.config_handler.config_manager.update("generation_config.temperature", 0.7)
            self.context.config_handler.config_manager.update("generation_config.top_p", 0.9)
            
            self.logger.record_event(
                event_type="generation_recovery",
                message="Recovered from critical generation error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
            return "System recovered from error. Please try your request again."
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to recover from generation error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_key": error_key}
            )
            return "A critical error occurred. Please try again later."
            
    def _adjust_generation_parameters(self, temperature: float) -> str:
        """Adjust generation parameters for non-critical errors."""
        try:
            current_temp = self.context.config_handler.config_manager.get("generation_config.temperature", 1.0)
            new_temp = max(0.5, current_temp - 0.05)
            self.context.config_handler.config_manager.update("generation_config.temperature", new_temp)
            
            self.logger.record_event(
                event_type="generation_adjustment",
                message="Adjusted generation parameters",
                level="info",
                additional_info={
                    "old_temperature": current_temp,
                    "new_temperature": new_temp,
                    "temperature": temperature
                }
            )
            
            return "System adjusted parameters. Please try your request again."
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to adjust generation parameters: {str(e)}",
                error_type="adjustment_error",
                stack_trace=traceback.format_exc()
            )
            return "An error occurred. Please try again."
            
        except Exception as e:
            self.logger.record_event(
                event_type="adjustment_failed",
                message=f"Failed to adjust generation parameters: {str(e)}",
                level="error"
            )
            return "An error occurred. Please try again."

    def _recover_data(self, error_key: str) -> None:
        """Recover from critical data errors."""
        try:
            self.error_counts[error_key] = 0
            
            self.context.config_handler.config_manager.update("data_config.batch_size", 1)
            self.context.config_handler.config_manager.update("data_config.max_retries", 3)
            
            self.logger.record_event(
                event_type="data_recovery",
                message="Recovered from critical data error",
                level="info",
                additional_info={"error_key": error_key}
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to recover from data error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_key": error_key}
            )

    def _adjust_data_parameters(self, context: Dict[str, Any]) -> None:
        """Adjust data parameters for non-critical errors."""
        try:
            current_batch_size = self.context.config_handler.config_manager.get("data_config.batch_size", 32)
            new_batch_size = max(1, current_batch_size // 2)
            self.context.config_handler.config_manager.update("data_config.batch_size", new_batch_size)
            
            self.logger.record_event(
                event_type="data_adjustment",
                message="Adjusted data parameters",
                level="info",
                additional_info={
                    "old_batch_size": current_batch_size,
                    "new_batch_size": new_batch_size,
                    "context": context
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to adjust data parameters: {str(e)}",
                error_type="adjustment_error",
                stack_trace=traceback.format_exc()
            )

    def _record_error(self, error: Exception, error_type: str, context: Dict[str, Any] = None) -> None:
        """Record an error in the history."""
        with self._lock:
            error_entry = {
                "type": error_type,
                "message": str(error),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "context": context or {}
            }
            self.recent_errors.append(error_entry)
            self.error_counts[f"{error_type}:{type(error).__name__}"] += 1

    def get_recent_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the list of recent errors."""
        with self._lock:
            return list(self.recent_errors)[-limit:]

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """Get the most recent error."""
        with self._lock:
            return self.recent_errors[-1] if self.recent_errors else None

    def clear_error_history(self) -> None:
        """Clear the error history."""
        with self._lock:
            self.recent_errors.clear()
            self.error_counts.clear()
            self.logger.record_event(
                event_type="error_history_cleared",
                message="Error history cleared successfully",
                level="info"
            )

class MemoryMonitor:
    """Monitors system memory health."""

    def __init__(self, context: SystemContext):
        """Initialize the memory monitor with context."""
        self.context = context
        self.memory_manager = MemoryManager(context)

    def check_memory_health(self, model_size: int, trainer: Optional[SOVLTrainer] = None) -> bool:
        """Check memory health and handle any errors."""
        try:
            is_healthy = self.memory_manager.check_memory_health(model_size, trainer)
            self.context.logger.log_memory_health(
                model_size=model_size,
                trainer=trainer,
                health_status="healthy" if is_healthy else "unhealthy",
                device=self.context.device
            )
            return is_healthy
        except Exception as e:
            # Handle memory errors through ErrorManager
            self.context.error_manager.handle_memory_error(e, model_size)
            return False

class CuriosityEngine:
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        config_handler: ConfigHandler,
        model_loader: ModelLoader,
        state_tracker: StateTracker,
        error_manager: ErrorManager,
        logger: Logger,
        device: str
    ):
        """
        Initialize the curiosity engine with explicit dependencies.
        
        Args:
            config_handler: Configuration handler
            model_loader: Model loader instance
            state_tracker: State tracker instance
            error_manager: Error manager instance
            logger: Logger instance
            device: Device to use for tensor operations
        """
        self.config_handler = config_handler
        self.model_loader = model_loader
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        self.logger = logger
        self.device = device
        
        # Initialize components
        self.curiosity_manager = self._create_curiosity_manager()
        self.cycle_manager = self._create_training_cycle_manager()
        
        # Log initialization
        self.logger.record_event(
            event_type="curiosity_engine_initialized",
            message="Curiosity engine initialized successfully",
            level="info"
        )
        
    def _create_curiosity_manager(self) -> CuriosityManager:
        """Create and initialize the curiosity manager."""
        try:
            return CuriosityManager(
                config_manager=self.config_handler.config_manager,
                logger=self.logger,
                device=self.device
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            raise
            
    def _create_training_cycle_manager(self) -> TrainingCycleManager:
        """Create and initialize the training cycle manager."""
        try:
            return TrainingCycleManager(
                config=self.config_handler.config_manager.get_section("sovl_config"),
                logger=self.logger,
                device=self.device,
                state_manager=self.state_tracker,
                curiosity_manager=self.curiosity_manager
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            raise
            
    def _validate_configuration(self) -> bool:
        """Validate current configuration state."""
        try:
            if not self.config_handler.validate():
                self.logger.record_event(
                    event_type="config_validation_failed",
                    message="Configuration validation failed, attempting recovery",
                    level="error"
                )
                self.config_handler._refresh_configs()
                if not self.config_handler.validate():
                    self.logger.record_event(
                        event_type="config_recovery_failed",
                        message="Configuration recovery failed",
                        level="error"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error during configuration validation: {str(e)}",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error": str(e)}
            )
            return False
            
    def run_training_cycle(
        self,
        train_data: Optional[List] = None,
        valid_data: Optional[List] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """Run a training cycle with configuration validation."""
        try:
            if not self._validate_configuration():
                raise RuntimeError("Invalid configuration state")
                
            state = self.state_tracker.get_state()
            
            results = self.cycle_manager.run_training_cycle(
                train_data=train_data,
                valid_data=valid_data,
                epochs=epochs,
                batch_size=batch_size
            )
            
            self._log_event("training_complete", {
                "epochs": epochs,
                "batch_size": batch_size,
                "results": results
            })
            
        except Exception as e:
            self.error_manager.handle_training_error(e, batch_size or 1)
            raise
            
    def _log_event(self, event: str, data: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=f"curiosity_{event}",
            message=f"Curiosity event: {event}",
            level="info",
            additional_info=data
        )

class SOVLSystem(SystemInterface):
    """Main SOVL system class that manages all components and state."""
    
    def __init__(
        self,
        context: SystemContext,
        config_handler: ConfigHandler,
        model_loader: ModelLoader,
        curiosity_engine: CuriosityEngine,
        memory_monitor: MemoryMonitor,
        state_tracker: StateTracker,
        error_manager: ErrorManager
    ):
        """
        Initialize the SOVL system with pre-initialized components.
        
        Args:
            context: System context containing shared resources
            config_handler: Configuration handler component
            model_loader: Model loading component
            curiosity_engine: Curiosity engine component
            memory_monitor: Memory monitoring component
            state_tracker: State tracking component
            error_manager: Error management component
        """
        try:
            # Validate required components
            validate_components(
                context=context,
                config_handler=config_handler,
                model_loader=model_loader,
                curiosity_engine=curiosity_engine,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
            # Store injected components
            self.context = context
            self.config_handler = config_handler
            self.model_loader = model_loader
            self.curiosity_engine = curiosity_engine
            self.memory_monitor = memory_monitor
            self.state_tracker = state_tracker
            self.error_manager = error_manager
            
            # Initialize thread safety
            self._lock = Lock()
            
            # Initialize component state
            self._initialize_component_state()
            
            # Log successful initialization
            self.context.logger.record_event(
                event_type="system_initialized",
                message="SOVL system initialized successfully with dependency injection",
                level="info",
                additional_info={
                    "config_path": self.config_handler.config_path,
                    "device": self.context.device,
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
            )
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to initialize SOVL system: {str(e)}",
                error_type="system_initialization_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_path": self.config_handler.config_path if hasattr(self, 'config_handler') else None,
                    "device": self.context.device if hasattr(self, 'context') else None
                }
            )
            raise

    def _initialize_component_state(self) -> None:
        """Initialize state for all components."""
        try:
            components = [
                self.curiosity_engine,
                self.memory_monitor,
                self.model_loader
            ]
            initialize_component_state(self.state_tracker, components)
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to initialize component state: {str(e)}",
                error_type="component_state_initialization_error",
                stack_trace=traceback.format_exc()
            )
            raise

    @synchronized("_lock")
    def toggle_memory(self, enable: bool) -> bool:
        """Enable or disable memory management."""
        try:
            if not hasattr(self.memory_monitor, 'memory_manager'):
                self.context.logger.record_event(
                    event_type="memory_error",
                    message="Memory manager not initialized",
                    level="error"
                )
                return False
                
            self.memory_monitor.memory_manager.set_enabled(enable)
            self.context.logger.record_event(
                event_type="memory_toggle",
                message=f"Memory management {'enabled' if enable else 'disabled'}",
                level="info",
                additional_info={
                    "enabled": enable,
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
            )
            return True
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, 0)  # 0 for memory size since this is a toggle operation
            self.context.logger.log_error(
                error_msg=f"Failed to toggle memory management: {str(e)}",
                error_type="memory_toggle_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "enabled": enable,
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
            )
            return False

    def generate_curiosity_question(self) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            if not hasattr(self.curiosity_engine, 'curiosity_manager'):
                self.context.logger.record_event(
                    event_type="curiosity_error",
                    message="Curiosity manager not initialized",
                    level="error"
                )
                return None
                
            question = self.curiosity_engine.curiosity_manager.generate_question()
            self.context.logger.record_event(
                event_type="curiosity_question_generated",
                message="Generated curiosity question",
                level="info",
                additional_info={"question": question}
            )
            return question
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            self.context.logger.log_error(
                error_msg=f"Failed to generate curiosity question: {str(e)}",
                error_type="curiosity_question_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error": str(e)}
            )
            return None

    def dream(self) -> bool:
        """Run a dream cycle to process and consolidate memories."""
        try:
            if not hasattr(self.curiosity_engine, 'run_dream_cycle'):
                self.context.logger.record_event(
                    event_type="dream_error",
                    message="Dream cycle not supported",
                    level="error"
                )
                return False
                
            self.curiosity_engine.run_dream_cycle()
            self.context.logger.record_event(
                event_type="dream_cycle_complete",
                message="Dream cycle completed successfully",
                level="info"
            )
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            self.context.logger.log_error(
                error_msg=f"Failed to run dream cycle: {str(e)}",
                error_type="dream_cycle_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error": str(e)}
            )
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            if not hasattr(self.memory_monitor, 'memory_manager'):
                return {"error": "Memory manager not initialized"}
                
            stats = {
                "total_allocated": self.memory_monitor.memory_manager.get_total_allocated(),
                "peak_allocated": self.memory_monitor.memory_manager.get_peak_allocated(),
                "current_usage": self.memory_monitor.memory_manager.get_current_usage(),
                "available_memory": self.memory_monitor.memory_manager.get_available_memory(),
                "fragmentation": self.memory_monitor.memory_manager.get_fragmentation(),
                "gc_count": self.memory_monitor.memory_manager.get_gc_count()
            }
            
            if torch.cuda.is_available():
                stats.update({
                    "gpu_allocated": torch.cuda.memory_allocated(),
                    "gpu_cached": torch.cuda.memory_reserved(),
                    "gpu_max_memory": torch.cuda.max_memory_allocated()
                })
                
            return stats
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, 0)
            return {"error": str(e)}

    def get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get list of recent errors from error manager."""
        try:
            if not hasattr(self, 'error_manager'):
                return []
                
            return self.error_manager.get_recent_errors()
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to get recent errors: {str(e)}",
                error_type="error_retrieval_error",
                stack_trace=traceback.format_exc()
            )
            return []

    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all system components."""
        try:
            return {
                "model_loader": hasattr(self, "model_loader"),
                "curiosity_engine": hasattr(self, "curiosity_engine"),
                "memory_monitor": hasattr(self, "memory_monitor"),
                "state_tracker": hasattr(self, "state_tracker"),
                "error_manager": hasattr(self, "error_manager"),
                "config_handler": hasattr(self, "config_handler")
            }
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to get component status: {str(e)}",
                error_type="component_status_error",
                stack_trace=traceback.format_exc()
            )
            return {}

    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        try:
            if not hasattr(self, 'state_tracker'):
                return {"error": "State tracker not initialized"}
                
            state = self.state_tracker.get_state()
            state.update({
                "debug_mode": self.context.logger.is_debug_enabled(),
                "last_error": self.error_manager.get_last_error() if hasattr(self, 'error_manager') else None,
                "components": self.get_component_status(),
                "memory_stats": self.get_memory_stats()
            })
            return state
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to get system state: {str(e)}",
                error_type="state_retrieval_error",
                stack_trace=traceback.format_exc()
            )
            return {"error": str(e)}

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        try:
            if enabled:
                self.context.logger.set_level(logging.DEBUG)
                self.context.logger.record_event(
                    event_type="debug_mode_change",
                    message="Debug mode enabled",
                    level="debug"
                )
            else:
                self.context.logger.set_level(logging.INFO)
                self.context.logger.record_event(
                    event_type="debug_mode_change",
                    message="Debug mode disabled",
                    level="info"
                )
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to set debug mode: {str(e)}",
                error_type="debug_mode_error",
                stack_trace=traceback.format_exc()
            )

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get recent execution trace from logger."""
        try:
            if not hasattr(self.context, 'logger'):
                return []
                
            return self.context.logger.get_recent_events()
            
        except Exception as e:
            self.context.logger.log_error(
                error_msg=f"Failed to get execution trace: {str(e)}",
                error_type="trace_retrieval_error",
                stack_trace=traceback.format_exc()
            )
            return []

    def get_state(self) -> Dict[str, Any]:
        """Get the current system state."""
        with self._lock:
            return self.state_tracker.get_state()

    def update_state(self, state_dict: Dict[str, Any]) -> None:
        """Update the system state with the provided dictionary."""
        with self._lock:
            try:
                self.state_tracker.update_state(state_dict)
                self.context.logger.record_event(
                    event_type="state_updated",
                    message="System state updated successfully",
                    level="info",
                    additional_info={
                        "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                    }
                )
            except Exception as e:
                self.context.logger.log_error(
                    error_msg=f"Failed to update system state: {str(e)}",
                    error_type="state_update_error",
                    stack_trace=traceback.format_exc()
                )
                raise ValueError("Invalid state update") from e

    def shutdown(self) -> None:
        """Shutdown the system, saving state and releasing resources."""
        with self._lock:
            try:
                # Log shutdown start
                self.context.logger.record_event(
                    event_type="system_shutdown_start",
                    message="Starting system shutdown process",
                    level="info"
                )

                # 1. Cleanup event dispatcher
                self.context.event_dispatcher.cleanup()
                self.context.logger.record_event(
                    event_type="event_dispatcher_cleanup",
                    message="Event dispatcher cleaned up",
                    level="info"
                )

                # 2. Save final state
                self.state_tracker.state.save_state()
                self.context.logger.record_event(
                    event_type="state_saved",
                    message="Final state saved",
                    level="info"
                )

                # 3. Clear state history
                self.state_tracker.clear_history()
                self.context.logger.record_event(
                    event_type="state_history_cleared",
                    message="State history cleared",
                    level="info"
                )

                # 4. Clear error history
                self.error_manager.clear_error_history()
                self.context.logger.record_event(
                    event_type="error_history_cleared",
                    message="Error history cleared",
                    level="info"
                )

                # 5. Release GPU memory if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.context.logger.record_event(
                        event_type="gpu_memory_cleared",
                        message="GPU memory cache cleared",
                        level="info"
                    )

                # 6. Final cleanup of logger
                self.context.logger.clear_queues()
                self.context.logger.record_event(
                    event_type="system_shutdown_complete",
                    message="System shutdown completed successfully",
                    level="info"
                )

            except Exception as e:
                self.context.logger.log_error(
                    error_msg=f"Failed to shutdown system: {str(e)}",
                    error_type="shutdown_error",
                    stack_trace=traceback.format_exc()
                )
                raise RuntimeError("Failed to shutdown system") from e
