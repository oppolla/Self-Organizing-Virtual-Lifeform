# Standard library imports
from typing import Optional, Any, List, Dict, Tuple, Callable, TYPE_CHECKING
import time
import traceback
import os
import sys
from collections import deque, defaultdict
from threading import Lock, RLock, Event, Thread

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
from sovl_error import ErrorManager
from sovl_logger import Logger
from sovl_events import EventDispatcher
from sovl_interfaces import SystemInterface
from sovl_queue import get_scribe_queue
from sovl_scribe import Scriber

# Model and processing
from sovl_manager import ModelManager
from sovl_processor import SOVLProcessor, MetadataProcessor
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner

# Memory and state management
from sovl_memory import RAMManager, GPUMemoryManager

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
    validate_quantization_mode,
    validate_component_states
)
from sovl_confidence import calculate_confidence_score
from sovl_io import  InsufficientDataError
from sovl_trainer import TrainingConfig, SOVLTrainer, TrainingCycleManager

# Type checking imports
if TYPE_CHECKING:
    from sovl_conductor import SOVLOrchestrator

# System-wide configuration constants
class SystemConstants:
    """System-wide configuration constants."""
    DEFAULT_DEVICE = "cuda"
    DEFAULT_CONFIG_PATH = "sovl_config.json"
    
    # Session management
    SESSION_COUNTER_DIR = os.path.join(os.path.expanduser("~"), ".sovl")
    SESSION_COUNTER_FILE = os.path.join(SESSION_COUNTER_DIR, "session_id_counter")
    SESSION_COUNTER_BACKUP = os.path.join(SESSION_COUNTER_DIR, "session_id_counter.bak")
    
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
    _session_lock = RLock()  # Lock for session ID operations
    
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
        self._initialization_complete = Event()  # Add initialization completion tracking
        self._startup_monitor = None  # Track startup monitor thread
        
        try:
            # Initialize session ID first
            self.session_id = self._get_next_session_id()
            
            # Initialize core components in dependency order
            self.config_manager = ConfigManager(config_path)
            # Store session_id in config for other components to access
            self.config_manager.set("runtime.session_id", self.session_id)
            
            self.logger = Logger()
            # Now that logger is initialized, log the session start
            self.logger.log_info(f"Starting SOVL Session: {self.session_id}")
            
            self.error_handler = ErrorManager()
            self.event_dispatcher = EventDispatcher()
            
            # Initialize metadata processor for scribe
            self.metadata_processor = MetadataProcessor(
                config_manager=self.config_manager,
                logger=self.logger
            )
            
            # Initialize scribe queue and scriber
            self.scribe_queue = get_scribe_queue()
            self.scriber = Scriber(
                config_manager=self.config_manager,
                error_manager=self.error_handler,
                metadata_processor=self.metadata_processor,
                logger=self.logger,
                scribe_queue=self.scribe_queue,
                state_accessor=self.state_manager
            )
            
            # Initialize memory managers
            self.ram_manager = RAMManager()
            self.gpu_manager = GPUMemoryManager()
            
            # Initialize state management
            self.state_manager = StateManager(
                config_manager=self.config_manager,
                logger=self.logger,
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
            
            # Complete initialization sequence
            self._complete_initialization()
            
        except Exception as e:
            self._handle_initialization_error(e)
            raise SystemInitializationError(
                message=f"Failed to initialize system context: {str(e)}",
                config_path=config_path,
                stack_trace=traceback.format_exc()
            )
    
    def _ensure_session_dir(self):
        """Ensure the session counter directory exists."""
        try:
            os.makedirs(SystemConstants.SESSION_COUNTER_DIR, exist_ok=True)
        except Exception as e:
            print(f"Error creating session counter directory: {e}", file=sys.stderr)
            raise
    
    def _backup_session_counter(self):
        """Create a backup of the session counter file."""
        try:
            if os.path.exists(SystemConstants.SESSION_COUNTER_FILE):
                with open(SystemConstants.SESSION_COUNTER_FILE, 'r') as src:
                    content = src.read()
                with open(SystemConstants.SESSION_COUNTER_BACKUP, 'w') as dst:
                    dst.write(content)
        except Exception as e:
            print(f"Warning: Could not backup session counter: {e}", file=sys.stderr)
    
    def _restore_session_counter(self):
        """Restore the session counter from backup if main file is corrupted."""
        try:
            if os.path.exists(SystemConstants.SESSION_COUNTER_BACKUP):
                with open(SystemConstants.SESSION_COUNTER_BACKUP, 'r') as src:
                    content = src.read()
                with open(SystemConstants.SESSION_COUNTER_FILE, 'w') as dst:
                    dst.write(content)
        except Exception as e:
            print(f"Warning: Could not restore session counter: {e}", file=sys.stderr)
    
    @synchronized("_session_lock")
    def _get_next_session_id(self) -> int:
        """Reads the last session ID, increments it, and writes it back."""
        self._ensure_session_dir()
        last_id = 0
        
        try:
            # Try to read the current counter
            if os.path.exists(SystemConstants.SESSION_COUNTER_FILE):
                with open(SystemConstants.SESSION_COUNTER_FILE, 'r') as f:
                    content = f.read().strip()
                    if content.isdigit():
                        last_id = int(content)
                    else:
                        # File exists but content is invalid, try backup
                        self._restore_session_counter()
                        if os.path.exists(SystemConstants.SESSION_COUNTER_FILE):
                            with open(SystemConstants.SESSION_COUNTER_FILE, 'r') as f:
                                content = f.read().strip()
                                if content.isdigit():
                                    last_id = int(content)
        except Exception as e:
            print(f"Warning: Could not read session ID counter file: {e}", file=sys.stderr)
            last_id = 0  # Reset on error

        current_id = last_id + 1

        try:
            # Create backup before writing
            self._backup_session_counter()
            
            # Write new counter
            with open(SystemConstants.SESSION_COUNTER_FILE, 'w') as f:
                f.write(str(current_id))
        except Exception as e:
            print(f"Error: Could not write session ID counter file: {e}", file=sys.stderr)
            # The current session will use current_id, but the next might reuse it

        return current_id
    
    def get_session_id(self) -> int:
        """Returns the current session ID."""
        return self.session_id
    
    def _initialize_system_state(self):
        """Initialize the central system state dictionary."""
        self.system_state = {
            'session_id': self.session_id,  # Add session ID to system state
            'start_time': time.time(),
            'status': 'initializing',
            'memory_usage': {'ram': 0.0, 'gpu': 0.0},
            'error_count': 0,
            'last_error': None,
            'component_states': {}
        }
        
        try:
            # Initialize core components in dependency order
            self.config_manager = ConfigManager(config_path)
            self.logger = Logger()
            self.error_handler = ErrorManager()
            self.event_dispatcher = EventDispatcher()
            
            # Initialize metadata processor for scribe
            self.metadata_processor = MetadataProcessor(
                config_manager=self.config_manager,
                logger=self.logger
            )
            
            # Initialize scribe queue and scriber
            self.scribe_queue = get_scribe_queue()
            self.scriber = Scriber(
                config_manager=self.config_manager,
                error_manager=self.error_handler,
                metadata_processor=self.metadata_processor,
                logger=self.logger,
                scribe_queue=self.scribe_queue,
                state_accessor=self.state_manager
            )
            
            # Initialize memory managers
            self.ram_manager = RAMManager()
            self.gpu_manager = GPUMemoryManager()
            
            # Initialize state management
            self.state_manager = StateManager(
                config_manager=self.config_manager,
                logger=self.logger,
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
            
            # Complete initialization sequence
            self._complete_initialization()
            
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
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(
                    error_msg=f"System initialization failed: {str(error)}",
                    error_type="system_initialization",
                    stack_trace=traceback.format_exc()
                )
            else:
                print(f"Critical error during initialization (logger unavailable): {str(error)}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
        except Exception as log_err:
            print(f"Critical error during initialization: {str(error)}", file=sys.stderr)
            print(f"Additionally, failed to log error: {str(log_err)}", file=sys.stderr)
    
    def _start_memory_monitoring(self):
        """Start proactive memory monitoring during initialization."""
        def monitor_memory():
            while not self._initialization_complete.is_set():
                try:
                    if hasattr(self, 'ram_manager') and hasattr(self, 'gpu_manager'):
                        self.update_memory_usage()
                    if hasattr(self, '_lock'):
                        self._cleanup_old_errors()
                    time.sleep(1)  # Check every second during initialization
                except Exception as e:
                    try:
                        if hasattr(self, 'logger') and self.logger:
                            self.logger.log_error(
                                error_msg=f"Startup memory monitoring error: {str(e)}",
                                error_type="startup_monitoring",
                                stack_trace=traceback.format_exc()
                            )
                        else:
                            print(f"Error in startup memory monitor (logger unavailable): {str(e)}", file=sys.stderr)
                    except Exception:
                        print(f"Critical error in startup memory monitor exception handler: {str(e)}", file=sys.stderr)
                    time.sleep(5)  # Wait longer on error
        
        self._startup_monitor = Thread(target=monitor_memory, daemon=True)
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
        """Complete initialization and clean up startup monitoring."""
        try:
            # Signal startup monitoring to stop
            if hasattr(self, '_initialization_complete'):
                self._initialization_complete.set()
            
            # Wait for startup monitor to finish
            if hasattr(self, '_startup_monitor') and self._startup_monitor and self._startup_monitor.is_alive():
                self._startup_monitor.join(timeout=5)
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_info("SystemContext initialization complete.")
                
        except Exception as e:
            try:
                if hasattr(self, 'error_handler') and self.error_handler:
                    self.error_handler.handle_error(
                        error_type="initialization_completion",
                        error_message=f"Failed to complete context initialization: {str(e)}",
                        error_context={
                            "initialization_state": self._initialization_complete.is_set() if hasattr(self, '_initialization_complete') else 'Unknown'
                        }
                    )
                elif hasattr(self, 'logger') and self.logger:
                    self.logger.log_error(
                        error_msg=f"Failed to complete context initialization: {str(e)}",
                        error_type="initialization_completion",
                        stack_trace=traceback.format_exc()
                    )
                else:
                    print(f"Error completing SystemContext initialization (logger/handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
            except Exception:
                print(f"Critical error in initialization completion exception handler: {str(e)}", file=sys.stderr)

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
        model_manager: ModelManager,
        curiosity_manager: CuriosityManager,
        state_tracker: StateTracker,
        error_manager: ErrorManager
    ):
        """
        Initialize the SOVL system with pre-initialized components.
        
        Args:
            context: System context containing shared resources
            model_manager: Model manager component
            curiosity_manager: Curiosity manager component
            state_tracker: State tracking component
            error_manager: Error management component
        """
        try:
            # Validate required components
            if not context:
                raise ValueError("SystemContext is required")
            
            # Store context and components
            self.context = context
            self.config_handler = context.config_manager
            self.model_manager = model_manager
            self.curiosity_manager = curiosity_manager
            self.state_tracker = state_tracker
            self.error_manager = error_manager
            
            # Initialize thread safety
            self._lock = RLock()
            
            # Initialize monitoring components
            self.memory_monitor = MemoryMonitor(
                config_manager=context.config_manager,
                logger=context.logger,
                ram_manager=context.ram_manager,
                gpu_manager=context.gpu_manager,
                error_manager=self.error_manager
            )
            
            self.system_monitor = SystemMonitor(
                config_manager=context.config_manager,
                logger=context.logger,
                ram_manager=context.ram_manager,
                gpu_manager=context.gpu_manager,
                error_manager=self.error_manager
            )
            
            self.traits_monitor = TraitsMonitor(
                config_manager=context.config_manager,
                logger=context.logger,
                state_manager=context.state_manager,
                curiosity_manager=self.curiosity_manager,
                training_manager=context.training_cycle_manager,
                error_manager=self.error_manager
            )
            
            # Initialize component state
            self._initialize_component_state()
            
            # Log successful initialization
            self.context.logger.record_event(
                event_type="system_initialized",
                message="SOVL system initialized successfully",
                level="info",
                additional_info={
                    "config_path": self.config_handler.config_path if self.config_handler else None,
                    "state_hash": self.state_tracker.get_state_hash() if hasattr(self.state_tracker, 'get_state_hash') else None
                }
            )
            
        except Exception as e:
            if hasattr(self, 'error_manager') and self.error_manager:
                self.error_manager.handle_error(
                    error_type="system_initialization",
                    error_message=f"Failed to initialize SOVL system: {str(e)}",
                    error_context={
                        "config_path": self.config_handler.config_path if hasattr(self, 'config_handler') and self.config_handler else None
                    },
                    error=e
                )
            else:
                print(f"Critical error initializing SOVLSystem (error handler unavailable): {str(e)}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
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
        """Start system monitoring components managed by SOVLSystem."""
        with self._lock:
            try:
                # Start memory monitoring if not already running
                if hasattr(self, 'memory_monitor') and self.memory_monitor:
                    if hasattr(self.memory_monitor, 'start_monitoring'):
                        self.memory_monitor.start_monitoring()
                        if self.context.logger:
                            self.context.logger.log_info("SOVL Memory monitoring started.")
                    else:
                        if not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
                            self._monitor_thread = Thread(target=self._monitor_memory, daemon=True)
                            self._monitor_thread.start()
                            if self.context.logger:
                                self.context.logger.log_info("SOVL main memory monitoring thread started.")

                # Start other monitoring systems
                for monitor in [self.state_tracker, self.error_manager, self.system_monitor, self.traits_monitor]:
                    if monitor and hasattr(monitor, 'start_monitoring'):
                        monitor.start_monitoring()
                    elif monitor and hasattr(monitor, 'start'):
                        monitor.start()
                
            except Exception as e:
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="monitoring_startup",
                        error_message=f"Failed to start SOVLSystem monitoring: {str(e)}",
                        error_context={
                            "monitor_thread_alive": hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive()
                        },
                        error=e
                    )
                else:
                    print(f"Error starting SOVLSystem monitoring (handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
    
    def stop_monitoring(self):
        """Stop system monitoring components managed by SOVLSystem."""
        with self._lock:
            try:
                # Stop main memory monitoring thread if it exists and is alive
                if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
                    if hasattr(self, '_stop_monitoring_event'):
                        self._stop_monitoring_event.set()
                    if self.context.logger:
                        self.context.logger.log_info("Attempting to stop main memory monitoring thread...")

                # Stop memory monitor instance if it has a stop method
                if hasattr(self, 'memory_monitor') and self.memory_monitor and hasattr(self.memory_monitor, 'stop_monitoring'):
                    self.memory_monitor.stop_monitoring()
                    if self.context.logger:
                        self.context.logger.log_info("SOVL Memory monitoring stopped.")

                # Stop other monitoring systems
                for monitor in [self.state_tracker, self.error_manager, self.system_monitor, self.traits_monitor]:
                    if monitor and hasattr(monitor, 'stop_monitoring'):
                        monitor.stop_monitoring()
                    elif monitor and hasattr(monitor, 'stop'):
                        monitor.stop()
                
            except Exception as e:
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="monitoring_shutdown",
                        error_message=f"Failed to stop SOVLSystem monitoring: {str(e)}",
                        error_context={
                            "monitor_thread_alive": hasattr(self, '_monitor_thread') and self._monitor_thread.is_alive() if hasattr(self, '_monitor_thread') else False
                        },
                        error=e
                    )
                else:
                    print(f"Error stopping SOVLSystem monitoring (handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

    def _monitor_memory(self):
        """Main memory monitoring loop (if managed directly by SOVLSystem)."""
        self._stop_monitoring_event = Event()
        while not self._stop_monitoring_event.is_set():
            try:
                if self.context and hasattr(self.context, 'update_memory_usage') and hasattr(self.context, '_cleanup_old_errors'):
                    self.context.update_memory_usage()
                    self.context._cleanup_old_errors()
                self._stop_monitoring_event.wait(5)  # Check every 5 seconds or when event is set
            except Exception as e:
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="memory_monitoring",
                        error_message=f"SOVLSystem memory monitoring error: {str(e)}",
                        error_context={
                            "monitoring_phase": "main"
                        },
                        error=e
                    )
                else:
                    print(f"Error in SOVLSystem memory monitor loop (handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                self._stop_monitoring_event.wait(10)  # Wait longer on error

        if self.context and self.context.logger:
            self.context.logger.log_info("SOVL main memory monitoring loop stopped.")

