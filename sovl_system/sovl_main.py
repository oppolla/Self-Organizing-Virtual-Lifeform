# Standard library imports
from typing import Optional, Any, List, Dict, Tuple, Callable, TYPE_CHECKING
import time
import traceback
import os
from collections import deque, defaultdict
from threading import Lock

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb

# Core components
from sovl_config import ConfigManager, ConfigHandler, ValidationSchema
from sovl_state import SOVLState, ConversationHistory, StateManager
from sovl_error import ErrorHandler
from sovl_logger import Logger
from sovl_events import EventDispatcher
from sovl_interfaces import SystemInterface

# Model and processing
from sovl_manager import ModelManager
from sovl_processor import LogitsProcessor, SOVLProcessor
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner

# Memory and state management
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_experience import MemoriaManager
from sovl_state import StateManager

# AI components
from sovl_curiosity import CuriosityEngine, CuriosityManager
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_scaffold import (
    CrossAttentionInjector,
    ScaffoldManager,
    CrossAttentionLayer,
    ScaffoldTokenMapper
)

# Utilities
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
from sovl_confidence import calculate_confidence_score
from sovl_io import validate_quantization_mode, InsufficientDataError
from sovl_trainer import TrainingConfig, SOVLTrainer, TrainingCycleManager

# Type checking imports
if TYPE_CHECKING:
    from sovl_conductor import SOVLOrchestrator

# System-wide configuration constants
class SystemConstants:
    """System-wide configuration constants."""
    DEFAULT_DEVICE = "cuda"
    DEFAULT_CONFIG_PATH = "sovl_config.json"
    
    # Memory thresholds
    MIN_MEMORY_THRESHOLD = 0.1
    MAX_MEMORY_THRESHOLD = 0.95
    DEFAULT_MEMORY_THRESHOLD = 0.85
    
    # Error handling
    MAX_ERROR_HISTORY = 100
    ERROR_COOLDOWN = 1.0
    
    # State management
    MAX_STATE_HISTORY = 100
    MAX_STATE_CHANGES = 50
    
    # Component initialization
    COMPONENT_INIT_TIMEOUT = 30.0  # seconds
    COMPONENT_RETRY_DELAY = 1.0    # seconds
    
    # Logging
    LOG_BUFFER_SIZE = 1000
    LOG_FLUSH_INTERVAL = 5.0  # seconds

class SystemContext:
    """Manages system-wide context and resources."""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: str = SystemConstants.DEFAULT_CONFIG_PATH):
        if self._initialized:
            return
            
        self._initialized = True
        self._lock = Lock()
        self._resource_locks = defaultdict(Lock)
        self._component_states = {}
        self._error_history = deque(maxlen=SystemConstants.MAX_ERROR_HISTORY)
        self._last_error_time = 0
        
        # Initialize core components
        self.config_manager = ConfigManager(config_path)
        self.logger = Logger()
        self.error_handler = ErrorHandler()
        self.event_dispatcher = EventDispatcher()
        
        # Initialize state tracking
        self.state_tracker = StateTracker()
        self.conversation_history = ConversationHistory()
        
        # Initialize memory managers
        self.ram_manager = RAMManager()
        self.gpu_manager = GPUMemoryManager()
        
        # Initialize AI components
        self.curiosity_engine = CuriosityEngine()
        self.temperament_system = TemperamentSystem()
        self.scaffold_manager = ScaffoldManager()
        
        # Initialize model components
        self.model_manager = ModelManager()
        self.processor = SOVLProcessor()
        self.generation_manager = GenerationManager()
        
        # Initialize training components
        self.trainer = SOVLTrainer()
        self.training_cycle_manager = TrainingCycleManager()
        
        # Initialize experience management
        self.memoria_manager = MemoriaManager()
        
        # Initialize system state
        self._initialize_system_state()
        
    def _initialize_system_state(self):
        """Initialize the system state with default values."""
        self.system_state = {
            'is_initialized': False,
            'is_training': False,
            'is_generating': False,
            'memory_usage': 0.0,
            'error_count': 0,
            'last_error': None,
            'component_states': {}
        }
        
    @synchronized
    def get_resource_lock(self, resource_name: str) -> Lock:
        """Get a lock for a specific resource."""
        return self._resource_locks[resource_name]
        
    @synchronized
    def update_component_state(self, component_name: str, state: Dict[str, Any]):
        """Update the state of a component in a thread-safe manner."""
        with self.get_resource_lock(component_name):
            self._component_states[component_name] = state
            self.system_state['component_states'][component_name] = state
            
    @synchronized
    def get_component_state(self, component_name: str) -> Dict[str, Any]:
        """Get the current state of a component in a thread-safe manner."""
        with self.get_resource_lock(component_name):
            return self._component_states.get(component_name, {})
            
    @synchronized
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error in a thread-safe manner."""
        current_time = time.time()
        if current_time - self._last_error_time < SystemConstants.ERROR_COOLDOWN:
            return
            
        self._last_error_time = current_time
        error_info = {
            'timestamp': current_time,
            'error': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self._error_history.append(error_info)
        self.system_state['error_count'] += 1
        self.system_state['last_error'] = error_info
        
        # Notify error handler
        self.error_handler.handle_error(error, context)
        
    @synchronized
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get the error history in a thread-safe manner."""
        return list(self._error_history)
        
    @synchronized
    def clear_error_history(self):
        """Clear the error history in a thread-safe manner."""
        self._error_history.clear()
        self.system_state['error_count'] = 0
        self.system_state['last_error'] = None
        
    @synchronized
    def update_memory_usage(self):
        """Update memory usage statistics in a thread-safe manner."""
        ram_usage = self.ram_manager.get_usage()
        gpu_usage = self.gpu_manager.get_usage()
        
        self.system_state['memory_usage'] = {
            'ram': ram_usage,
            'gpu': gpu_usage
        }
        
    @synchronized
    def get_system_state(self) -> Dict[str, Any]:
        """Get the current system state in a thread-safe manner."""
        self.update_memory_usage()
        return self.system_state.copy()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with error handling."""
        if exc_type is not None:
            self.record_error(exc_val, {'context': 'SystemContext cleanup'})
        return False

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

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
    """Monitors system memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        memoria_manager: MemoriaManager,  # For experiential aspects
        ram_manager: RAMManager,          # For RAM memory management
        gpu_manager: GPUMemoryManager     # For GPU memory management
    ):
        """
        Initialize memory monitor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            memoria_manager: MemoriaManager instance for experiential memory management
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = memoria_manager  # Manages experiential memory
        self.ram_manager = ram_manager          # Manages RAM resources
        self.gpu_manager = gpu_manager          # Manages GPU resources
        
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health across hardware memory managers."""
        try:
            # Only use RAM and GPU managers for hardware memory health
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.get_gpu_usage()
            
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

class SOVLSystem(SystemInterface):
    """Main SOVL system class that manages all components and state."""
    
    def __init__(
        self,
        context: SystemContext,
        config_handler: ConfigHandler,
        model_manager: ModelManager,
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
            model_manager: Model manager component
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
                model_manager=model_manager,
                curiosity_engine=curiosity_engine,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
            # Store injected components
            self.context = context
            self.config_handler = config_handler
            self.model_manager = model_manager
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
            self.error_manager.handle_error(
                error_type="system_initialization",
                error_message=f"Failed to initialize SOVL system: {str(e)}",
                error_context={
                    "config_path": config_handler.config_path if config_handler else None,
                    "device": context.device if context else None
                }
            )
            raise

    def _initialize_component_state(self):
        """Initialize the state of all components."""
        try:
            # Initialize component states
            self.state_tracker.update_state({
                "config_handler": {
                    "status": "initialized",
                    "config_path": self.config_handler.config_path
                },
                "model_manager": {
                    "status": "initialized",
                    "active_model": self.model_manager.active_model_name if self.model_manager else None
                },
                "curiosity_engine": {
                    "status": "initialized",
                    "question_cache_size": len(self.curiosity_engine.question_cache) if self.curiosity_engine else 0
                },
                "memory_monitor": {
                    "status": "initialized",
                    "memory_usage": self.memory_monitor.get_memory_usage() if self.memory_monitor else None
                },
                "state_tracker": {
                    "status": "initialized",
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                },
                "error_manager": {
                    "status": "initialized",
                    "error_count": len(self.error_manager.error_history) if self.error_manager else 0
                }
            })
            
        except Exception as e:
            self.error_manager.handle_error(
                error_type="component_state_initialization",
                error_message=f"Failed to initialize component states: {str(e)}",
                error_context={
                    "component": "SOVLSystem",
                    "method": "_initialize_component_state"
                }
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
            ram_stats = self.ram_manager.check_memory_health()
            gpu_stats = self.gpu_manager.get_gpu_usage()
            
            return {
                "ram": ram_stats,
                "gpu": gpu_stats
            }
            
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
                "model_manager": hasattr(self, "model_manager"),
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

