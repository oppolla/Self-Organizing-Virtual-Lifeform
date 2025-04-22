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
from sovl_state import SOVLState, ConversationHistory, StateManager, StateTracker
from sovl_error import ErrorManager, ErrorHandler
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
from sovl_curiosity import CuriosityManager
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_scaffold import (
    CrossAttentionInjector,
    CrossAttentionLayer,
    ScaffoldTokenMapper
)

# Monitoring components
from sovl_monitor import MemoryMonitor, SystemMonitor, TraitsMonitor

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
        
        # Initialize memory managers
        self.ram_manager = RAMManager()
        self.gpu_manager = GPUMemoryManager()
        
        # Initialize experience management
        self.memoria_manager = MemoriaManager()
        
        # Initialize state management
        self.state_manager = StateManager(
            config_manager=self.config_manager,
            logger=self.logger,
            memoria_manager=self.memoria_manager,
            ram_manager=self.ram_manager,
            gpu_manager=self.gpu_manager
        )
        
        # Initialize state tracking
        self.state_tracker = StateTracker(
            config_manager=self.config_manager,
            logger=self.logger
        )
        
        # Initialize AI components
        self.curiosity_manager = CuriosityManager(
            config_manager=self.config_manager,
            logger=self.logger,
            error_manager=self.error_handler,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            state_manager=self.state_manager
        )
        self.temperament_system = TemperamentSystem()
        self.scaffold_manager = ScaffoldManager()
        
        # Initialize model components
        self.model_manager = ModelManager()
        self.processor = SOVLProcessor()
        self.generation_manager = GenerationManager()
        
        # Initialize training components
        self.trainer = SOVLTrainer()
        self.training_cycle_manager = TrainingCycleManager()
        
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

class SOVLSystem(SystemInterface):
    """Main SOVL system class that manages all components and state."""
    
    def __init__(
        self,
        context: SystemContext,
        config_handler: ConfigHandler,
        model_manager: ModelManager,
        curiosity_manager: CuriosityManager,
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
            curiosity_manager: Curiosity manager component
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
                curiosity_manager=curiosity_manager,
                memory_monitor=memory_monitor,
                state_tracker=state_tracker,
                error_manager=error_manager
            )
            
            # Store injected components
            self.context = context
            self.config_handler = config_handler
            self.model_manager = model_manager
            self.curiosity_manager = curiosity_manager
            self.memory_monitor = memory_monitor
            self.state_tracker = state_tracker
            self.error_manager = error_manager
            
            # Initialize thread safety
            self._lock = Lock()
            
            # Initialize monitoring components
            self.system_monitor = SystemMonitor(
                config_manager=context.config_manager,
                logger=context.logger,
                ram_manager=context.ram_manager,
                gpu_manager=context.gpu_manager,
                error_manager=error_manager
            )
            
            self.traits_monitor = TraitsMonitor(
                config_manager=context.config_manager,
                logger=context.logger,
                state_manager=context.state_manager,
                curiosity_manager=curiosity_manager.curiosity_manager,
                training_manager=context.training_cycle_manager,
                error_manager=error_manager
            )
            
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
            # Initialize component states using StateManager
            component_states = {
                "config_handler": {
                    "status": "initialized",
                    "config_path": self.config_handler.config_path
                },
                "model_manager": {
                    "status": "initialized",
                    "active_model": self.model_manager.active_model_name if self.model_manager else None
                },
                "curiosity_manager": {
                    "status": "initialized",
                    "question_cache_size": len(self.curiosity_manager.question_cache) if self.curiosity_manager else 0
                },
                "memory_monitor": {
                    "status": "initialized",
                    "memory_usage": self.memory_monitor.check_memory_health() if self.memory_monitor else None
                },
                "system_monitor": {
                    "status": "initialized",
                    "metrics": self.system_monitor._collect_metrics() if self.system_monitor else None
                },
                "traits_monitor": {
                    "status": "initialized",
                    "traits": self.traits_monitor._get_current_traits() if self.traits_monitor else None
                },
                "state_tracker": {
                    "status": "initialized",
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                },
                "error_manager": {
                    "status": "initialized",
                    "error_count": len(self.error_manager.get_error_stats()["error_counts"]) if self.error_manager else 0
                }
            }
            
            # Save component states using StateManager
            self.context.state_manager.save_state(component_states)
            
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
            if not hasattr(self.curiosity_manager, 'generate_question'):
                self.context.logger.record_event(
                    event_type="curiosity_error",
                    message="Curiosity manager not properly initialized",
                    level="error"
                )
                return None
                
            question = self.curiosity_manager.generate_question()
            
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
                stack_trace=traceback.format_exc()
            )
            return None

    def dream(self) -> bool:
        """Run a dream cycle to explore new ideas."""
        try:
            if not hasattr(self.curiosity_manager, 'queue_exploration'):
                self.context.logger.record_event(
                    event_type="curiosity_error",
                    message="Curiosity manager not properly initialized",
                    level="error"
                )
                return False
                
            self.curiosity_manager.queue_exploration("dream_cycle")
            
            self.context.logger.record_event(
                event_type="dream_cycle_started",
                message="Started dream cycle",
                level="info"
            )
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            self.context.logger.log_error(
                error_msg=f"Failed to start dream cycle: {str(e)}",
                error_type="dream_cycle_error",
                stack_trace=traceback.format_exc()
            )
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            # Use system monitor for comprehensive metrics
            metrics = self.system_monitor._collect_metrics()
            
            # Add memory health check
            memory_health = self.memory_monitor.check_memory_health()
            
            return {
                "metrics": metrics,
                "memory_health": memory_health
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
        """Get the status of all components."""
        return {
            "config_handler": hasattr(self, "config_handler"),
            "model_manager": hasattr(self, "model_manager"),
            "curiosity_manager": hasattr(self, "curiosity_manager"),
            "memory_monitor": hasattr(self, "memory_monitor"),
            "state_tracker": hasattr(self, "state_tracker"),
            "error_manager": hasattr(self, "error_manager")
        }

    def get_system_state(self) -> Dict[str, Any]:
        """Get the current system state."""
        return {
            "curiosity_manager": {
                "is_initialized": hasattr(self, "curiosity_manager"),
                "exploration_queue_size": len(self.curiosity_manager.exploration_queue) if hasattr(self.curiosity_manager, 'exploration_queue') else 0
            },
            "memory_usage": self.memory_monitor.get_memory_metrics() if hasattr(self, "memory_monitor") else {},
            "error_count": len(self.error_manager.get_recent_errors()) if hasattr(self, "error_manager") else 0
        }

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
                # Use StateManager for core state updates
                self.context.state_manager.save_state(state_dict)
                
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

    def start_monitoring(self) -> None:
        """Start all monitoring systems."""
        try:
            # Start traits monitoring
            self.traits_monitor.start()
            
            self.context.logger.record_event(
                event_type="monitoring_started",
                message="System monitoring started",
                level="info"
            )
            
        except Exception as e:
            self.error_manager.handle_error(
                error_type="monitoring_start_error",
                error_message=f"Failed to start monitoring: {str(e)}",
                error_context={"component": "SOVLSystem"}
            )

    def stop_monitoring(self) -> None:
        """Stop all monitoring systems."""
        try:
            # Stop traits monitoring
            self.traits_monitor.stop()
            
            self.context.logger.record_event(
                event_type="monitoring_stopped",
                message="System monitoring stopped",
                level="info"
            )
            
        except Exception as e:
            self.error_manager.handle_error(
                error_type="monitoring_stop_error",
                error_message=f"Failed to stop monitoring: {str(e)}",
                error_context={"component": "SOVLSystem"}
            )

