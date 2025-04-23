# Standard library imports
from typing import Optional, Any, List, Dict, Tuple, Callable, TYPE_CHECKING
import time
import traceback
import os
from collections import deque, defaultdict
from threading import Lock, RLock

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb

# Core components
from sovl_config import ConfigManager, ValidationSchema
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
    ERROR_CLEANUP_INTERVAL = 3600  # 1 hour in seconds
    
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
    _lock = RLock()  # Using RLock for reentrant locking
    
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
        self._lock = RLock()  # Using RLock for reentrant locking
        self._resource_locks = defaultdict(RLock)  # Using RLock for reentrant locking
        self._component_states = {}
        self._error_history = deque(maxlen=SystemConstants.MAX_ERROR_HISTORY)
        self._last_error_time = 0
        self._last_error_cleanup = 0
        self._initialization_complete = threading.Event()  # Add initialization completion tracking
        self._startup_monitor = None  # Track startup monitor thread
        
        try:
            # Initialize core components in dependency order
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
            
            # Start memory monitoring
            self._start_memory_monitoring()
            
            # Signal initialization complete
            self._initialization_complete.set()
            
        except Exception as e:
            self._handle_initialization_error(e)
            raise SystemInitializationError(
                message=f"Failed to initialize system context: {str(e)}",
                config_path=config_path,
                stack_trace=traceback.format_exc()
            )
    
    def _handle_initialization_error(self, error: Exception):
        """Handle initialization errors safely."""
        try:
            if hasattr(self, 'logger'):
                self.logger.log_error(
                    error_msg=f"System initialization failed: {str(error)}",
                    error_type="system_initialization",
                    stack_trace=traceback.format_exc()
                )
        except Exception:
            # If logger is not available, print to stderr
            print(f"Critical error during initialization: {str(error)}", file=sys.stderr)
    
    def _start_memory_monitoring(self):
        """Start proactive memory monitoring during initialization."""
        def monitor_memory():
            while not self._initialization_complete.is_set():
                try:
                    self.update_memory_usage()
                    self._cleanup_old_errors()
                    time.sleep(1)  # Check every second during initialization
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.log_error(
                            error_msg=f"Startup memory monitoring error: {str(e)}",
                            error_type="startup_monitoring",
                            stack_trace=traceback.format_exc()
                        )
                    time.sleep(5)  # Wait longer on error
        
        self._startup_monitor = threading.Thread(target=monitor_memory, daemon=True)
        self._startup_monitor.start()
    
    def _cleanup_old_errors(self):
        """Clean up old errors from history."""
        current_time = time.time()
        if current_time - self._last_error_cleanup > SystemConstants.ERROR_CLEANUP_INTERVAL:
            with self._lock:
                # Remove errors older than 24 hours
                cutoff_time = current_time - 86400  # 24 hours in seconds
                self._error_history = deque(
                    (e for e in self._error_history if e['timestamp'] > cutoff_time),
                    maxlen=SystemConstants.MAX_ERROR_HISTORY
                )
                self._last_error_cleanup = current_time
    
    @synchronized("_lock")
    def get_resource_lock(self, resource_name: str) -> RLock:
        """Get a lock for a specific resource."""
        return self._resource_locks[resource_name]
    
    @synchronized("_lock")
    def update_component_state(self, component_name: str, state: Dict[str, Any]):
        """Update the state of a component in a thread-safe manner."""
        with self.get_resource_lock(component_name):
            self._component_states[component_name] = state
            self.system_state['component_states'][component_name] = state
    
    @synchronized("_lock")
    def get_component_state(self, component_name: str) -> Dict[str, Any]:
        """Get the current state of a component in a thread-safe manner."""
        with self.get_resource_lock(component_name):
            return self._component_states.get(component_name, {}).copy()
    
    @synchronized("_lock")
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
        
        with self._lock:
            self._error_history.append(error_info)
            self.system_state['error_count'] += 1
            self.system_state['last_error'] = error_info
        
        # Notify error handler
        self.error_handler.handle_error(error, context)
    
    @synchronized("_lock")
    def get_error_history(self) -> List[Dict[str, Any]]:
        """Get the error history in a thread-safe manner."""
        return list(self._error_history)
    
    @synchronized("_lock")
    def clear_error_history(self):
        """Clear the error history in a thread-safe manner."""
        with self._lock:
            self._error_history.clear()
            self.system_state['error_count'] = 0
            self.system_state['last_error'] = None
    
    @synchronized("_lock")
    def update_memory_usage(self):
        """Update memory usage statistics in a thread-safe manner."""
        try:
            ram_usage = self.ram_manager.get_usage()
            gpu_usage = self.gpu_manager.get_usage()
            
            with self._lock:
                self.system_state['memory_usage'] = {
                    'ram': ram_usage,
                    'gpu': gpu_usage
                }
                
                # Check for memory thresholds
                if ram_usage > SystemConstants.MAX_MEMORY_THRESHOLD:
                    self._handle_memory_threshold_exceeded('ram', ram_usage)
                if gpu_usage > SystemConstants.MAX_MEMORY_THRESHOLD:
                    self._handle_memory_threshold_exceeded('gpu', gpu_usage)
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.log_error(
                    error_msg=f"Memory update failed: {str(e)}",
                    error_type="memory_update",
                    stack_trace=traceback.format_exc()
                )
    
    def _handle_memory_threshold_exceeded(self, memory_type: str, usage: float):
        """Handle memory threshold exceeded events."""
        try:
            self.logger.log_warning(
                f"{memory_type.upper()} memory threshold exceeded",
                additional_info={
                    "usage": usage,
                    "threshold": SystemConstants.MAX_MEMORY_THRESHOLD
                }
            )
            
            # Trigger memory cleanup
            if memory_type == 'ram':
                self.ram_manager.cleanup()
            else:
                self.gpu_manager.cleanup()
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.log_error(
                    error_msg=f"Memory threshold handling failed: {str(e)}",
                    error_type="memory_threshold",
                    stack_trace=traceback.format_exc()
                )
    
    @synchronized("_lock")
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

    def _complete_initialization(self):
        """Complete initialization and hand off to main monitoring."""
        try:
            # Signal startup monitoring to stop
            self._initialization_complete.set()
            
            # Wait for startup monitor to finish
            if self._startup_monitor:
                self._startup_monitor.join(timeout=5)
            
            # Start main monitoring systems
            self.start_monitoring()
            
        except Exception as e:
            self.error_handler.handle_error(
                error_type="initialization_completion",
                error_message=f"Failed to complete initialization: {str(e)}",
                error_context={
                    "initialization_state": self._initialization_complete.is_set()
                }
            )

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

class SOVLSystem(SystemInterface):
    """Self-Organizing Virtual Lifeform system class that manages all components and state."""
    
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
            self._lock = RLock()  # Using RLock for reentrant locking
            
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
            
            # Update component states in a thread-safe manner
            with self._lock:
                for component_name, state in component_states.items():
                    self.context.update_component_state(component_name, state)
                    
        except Exception as e:
            self.error_manager.handle_error(
                error_type="component_initialization",
                error_message=f"Failed to initialize component states: {str(e)}",
                error_context={
                    "component_states": component_states
                }
            )
            raise

    @synchronized("_lock")
    def toggle_memory(self, enable: bool) -> bool:
        """Toggle memory management features."""
        try:
            if enable:
                # Enable memory management
                self.memory_monitor.start_monitoring()
                self.context.ram_manager.enable_cleanup()
                self.context.gpu_manager.enable_cleanup()
            else:
                # Disable memory management
                self.memory_monitor.stop_monitoring()
                self.context.ram_manager.disable_cleanup()
                self.context.gpu_manager.disable_cleanup()
                
            return True
            
        except Exception as e:
            self.error_manager.handle_error(
                error_type="memory_toggle",
                error_message=f"Failed to toggle memory management: {str(e)}",
                error_context={
                    "enable": enable
                }
            )
            return False

    def generate_curiosity_question(self) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            with self._lock:
                if not self.curiosity_manager:
                    raise ValueError("Curiosity manager not initialized")
                    
                question = self.curiosity_manager.generate_curiosity_question(
                    context=None,
                    spontaneous=True,
                    tokenizer=self.model_manager.tokenizer if self.model_manager else None,
                    model=self.model_manager.model if self.model_manager else None
                )
                if question:
                    self.context.logger.record_event(
                        event_type="curiosity_question_generated",
                        message="Generated new curiosity question",
                        level="info",
                        additional_info={
                            "question": question
                        }
                    )
                return question
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="curiosity_question",
                error_message=f"Failed to generate curiosity question: {str(e)}"
            )
            return None

    def dream(self) -> bool:
        """Execute a dreaming cycle."""
        try:
            with self._lock:
                if not self.model_manager or not self.curiosity_manager:
                    raise ValueError("Required components not initialized")
                    
                # Generate curiosity question
                question = self.generate_curiosity_question()
                if not question:
                    return False
                    
                # Process the question
                response = self.model_manager.generate_response(question)
                if not response:
                    return False
                    
                # Update memory and state
                self.context.memoria_manager.add_experience(question, response)
                self.state_tracker.update_state({
                    "last_dream": time.time(),
                    "dream_question": question,
                    "dream_response": response
                })
                
                return True
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="dream_cycle",
                error_message=f"Failed to execute dream cycle: {str(e)}"
            )
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            with self._lock:
                return {
                    "ram": self.context.ram_manager.get_usage(),
                    "gpu": self.context.gpu_manager.get_usage(),
                    "memoria": len(self.context.memoria_manager.get_experiences())
                }
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="memory_stats",
                error_message=f"Failed to get memory statistics: {str(e)}"
            )
            return {}

    def get_recent_errors(self) -> List[Dict[str, Any]]:
        """Get recent error history."""
        try:
            with self._lock:
                return self.context.get_error_history()
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="error_history",
                error_message=f"Failed to get error history: {str(e)}"
            )
            return []

    def get_component_status(self) -> Dict[str, bool]:
        """Get the status of all components."""
        try:
            with self._lock:
                return {
                    "config_handler": bool(self.config_handler),
                    "model_manager": bool(self.model_manager),
                    "curiosity_manager": bool(self.curiosity_manager),
                    "memory_monitor": bool(self.memory_monitor),
                    "state_tracker": bool(self.state_tracker),
                    "error_manager": bool(self.error_manager),
                    "system_monitor": bool(self.system_monitor),
                    "traits_monitor": bool(self.traits_monitor)
                }
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="component_status",
                error_message=f"Failed to get component status: {str(e)}"
            )
            return {}

    def get_system_state(self) -> Dict[str, Any]:
        """Get the current system state."""
        try:
            with self._lock:
                return {
                    "memory_stats": self.get_memory_stats(),
                    "component_status": self.get_component_status(),
                    "recent_errors": self.get_recent_errors(),
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                }
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="system_state",
                error_message=f"Failed to get system state: {str(e)}"
            )
            return {}

    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug mode."""
        try:
            with self._lock:
                self.context.logger.set_debug_mode(enabled)
                if self.system_monitor:
                    self.system_monitor.set_debug_mode(enabled)
                if self.traits_monitor:
                    self.traits_monitor.set_debug_mode(enabled)
                    
        except Exception as e:
            self.error_manager.handle_error(
                error_type="debug_mode",
                error_message=f"Failed to set debug mode: {str(e)}",
                error_context={
                    "enabled": enabled
                }
            )

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get the execution trace of recent operations."""
        try:
            with self._lock:
                return self.context.logger.get_execution_trace()
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="execution_trace",
                error_message=f"Failed to get execution trace: {str(e)}"
            )
            return []

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the system."""
        try:
            with self._lock:
                return self.state_tracker.state.to_dict() if self.state_tracker.state else {}
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="state_retrieval",
                error_message=f"Failed to get system state: {str(e)}"
            )
            return {}

    def update_state(self, state_dict: Dict[str, Any]) -> None:
        """Update the system state."""
        try:
            with self._lock:
                if not self.state_tracker:
                    raise ValueError("State tracker not initialized")
                    
                self.state_tracker.update_state(state_dict)
                
        except Exception as e:
            self.error_manager.handle_error(
                error_type="state_update",
                error_message=f"Failed to update system state: {str(e)}",
                error_context={
                    "state_dict": state_dict
                }
            )

    def start_monitoring(self):
        """Start system monitoring."""
        with self._lock:
            if not self._initialization_complete.is_set():
                self._complete_initialization()
                return
                
            try:
                # Start memory monitoring if not already running
                if not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
                    self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
                    self._monitor_thread.start()
                    self.logger.log_info("Main memory monitoring started")
                
                # Start other monitoring systems
                self.state_tracker.start_monitoring()
                self.error_handler.start_monitoring()
                
            except Exception as e:
                self.error_handler.handle_error(
                    error_type="monitoring_startup",
                    error_message=f"Failed to start monitoring: {str(e)}",
                    error_context={
                        "initialization_state": self._initialization_complete.is_set(),
                        "monitor_thread_alive": hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive()
                    }
                )
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        with self._lock:
            try:
                # Stop startup monitoring if still running
                if self._startup_monitor and self._startup_monitor.is_alive():
                    self._initialization_complete.set()
                    self._startup_monitor.join(timeout=5)
                
                # Stop main monitoring
                if hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive():
                    self._monitor_thread.join(timeout=5)
                
                # Stop other monitoring systems
                self.state_tracker.stop_monitoring()
                self.error_handler.stop_monitoring()
                
            except Exception as e:
                self.error_handler.handle_error(
                    error_type="monitoring_shutdown",
                    error_message=f"Failed to stop monitoring: {str(e)}",
                    error_context={
                        "startup_monitor_alive": self._startup_monitor and self._startup_monitor.is_alive(),
                        "monitor_thread_alive": hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive()
                    }
                )
    
    def _monitor_memory(self):
        """Main memory monitoring loop."""
        while True:
            try:
                self.update_memory_usage()
                self._cleanup_old_errors()
                time.sleep(5)  # Check every 5 seconds during normal operation
            except Exception as e:
                self.error_handler.handle_error(
                    error_type="memory_monitoring",
                    error_message=f"Memory monitoring error: {str(e)}",
                    error_context={
                        "monitoring_phase": "main",
                        "initialization_complete": self._initialization_complete.is_set()
                    }
                )
                time.sleep(10)  # Wait longer on error

