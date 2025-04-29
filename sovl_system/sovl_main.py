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
from sovl_volition import AutonomyManager

# Model and processing
from sovl_manager import ModelManager
from sovl_processor import SOVLProcessor, MetadataProcessor
from sovl_generation import GenerationManager
from sovl_tuner import SOVLTuner

# Memory and state management
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_recaller import DialogueContextManager

# AI components
from sovl_curiosity import CuriosityManager
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_meditater import IntrospectionManager
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
    _init_lock = RLock()  # Separate lock for initialization
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: str = SystemConstants.DEFAULT_CONFIG_PATH):
        # Return quickly if already initialized
        if self._initialized:
            return
        
        # Use a separate lock for initialization to prevent race conditions
        with self._init_lock:
            # Double-check pattern - check again inside the lock
            if self._initialized:
                return
            
            # Setup basic attributes first but don't set _initialized until the end
            self._lock = RLock()  # Using RLock for reentrant locking
            self._component_states = {}
            self._error_history = deque(maxlen=SystemConstants.MAX_ERROR_HISTORY)
            self._last_error_time = 0
            self._last_error_cleanup = 0
            self._initialization_complete = Event()  # Add initialization completion tracking
            self._startup_monitor = None  # Track startup monitor thread
            self._initialized_components = set()  # Track which components have been initialized
            
            try:
                # Phase 1: Initialize core components with minimal dependencies
                self._initialize_core_components(config_path)
                
                # Phase 2: Initialize components that require dependencies
                self._initialize_dependent_components()
                
                # Phase 3: Connect cross-references safely after all components exist
                self._connect_component_references()
                
                # Mark initialization as complete
                self._initialized = True
                self._initialization_complete.set()  # Signal initialization is complete
                self.logger.log_info("SystemContext initialization complete")
                
            except Exception as e:
                # Log the initialization error
                error_msg = f"SystemContext initialization failed: {str(e)}"
                stack_trace = traceback.format_exc()
                
                # Use fallback logging if logger isn't initialized yet
                if hasattr(self, 'logger') and self.logger:
                    self.logger.log_error(error_msg=error_msg, error_type="init_error", stack_trace=stack_trace)
                else:
                    print(f"ERROR: {error_msg}\n{stack_trace}")
                
                # Clean up any partially initialized resources
                self._cleanup_partial_initialization()
                
                # Re-raise with additional context
                raise SystemInitializationError(error_msg, config_path, stack_trace)
    
    def _initialize_core_components(self, config_path: str):
        """Initialize core components with minimal dependencies."""
        try:
            # Initialize session ID first
            self.session_id = self._get_next_session_id()
            self._initialized_components.add("session_id")
            
            # Initialize configuration manager
            self.config_manager = ConfigManager(config_path)
            self._initialized_components.add("config_manager")
            
            # Store session_id in config for other components to access
            self.config_manager.set("runtime.session_id", self.session_id)
            
            # Initialize logger
            self.logger = Logger()
            self._initialized_components.add("logger")
            
            # Now that logger is initialized, log the session start
            self.logger.log_info(f"Starting SOVL Session: {self.session_id}")
            
            # Initialize error handler
            self.error_handler = ErrorManager()
            self._initialized_components.add("error_handler")
            
            # Initialize event dispatcher
            self.event_dispatcher = EventDispatcher()
            self._initialized_components.add("event_dispatcher")
            
            # Initialize memory managers (no dependencies)
            self.ram_manager = RAMManager()
            self._initialized_components.add("ram_manager")
            
            self.gpu_manager = GPUMemoryManager()
            self._initialized_components.add("gpu_manager")
            
            # Initialize state management
            self.state_manager = StateManager(
                config_manager=self.config_manager,
                logger=self.logger,
                ram_manager=self.ram_manager,
                gpu_manager=self.gpu_manager
            )
            self._initialized_components.add("state_manager")
            
            # Initialize state tracking
            self.state_tracker = StateTracker(
                config_manager=self.config_manager,
                logger=self.logger
            )
            self._initialized_components.add("state_tracker")
            
            # Log core components initialization
            self.logger.log_info("Core components initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize core components: {str(e)}"
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(error_msg=error_msg, error_type="init_core_error", stack_trace=traceback.format_exc())
            else:
                print(f"ERROR: {error_msg}\n{traceback.format_exc()}")
            raise
    
    def _initialize_dependent_components(self):
        """Initialize components that depend on core components."""
        try:
            # Initialize metadata processor for scribe
            self.metadata_processor = MetadataProcessor(
                config_manager=self.config_manager,
                logger=self.logger
            )
            self._initialized_components.add("metadata_processor")
            
            # Initialize scribe queue with logger
            from sovl_queue import get_scribe_queue
            self.scribe_queue = get_scribe_queue(logger=self.logger)
            self._initialized_components.add("scribe_queue")
            
            # Initialize scriber without state_accessor (will be set later)
            self.scriber = Scriber(
                config_manager=self.config_manager,
                error_manager=self.error_handler,
                metadata_processor=self.metadata_processor,
                logger=self.logger,
                scribe_queue=self.scribe_queue
            )
            self._initialized_components.add("scriber")
            
            # Initialize AI components
            self.curiosity_manager = CuriosityManager(
                config_manager=self.config_manager,
                logger=self.logger,
                error_manager=self.error_handler,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                state_manager=self.state_manager
            )
            self._initialized_components.add("curiosity_manager")
            
            self.temperament_system = TemperamentSystem()
            self._initialized_components.add("temperament_system")
            
            # Log dependent components initialization
            self.logger.log_info("Dependent components initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize dependent components: {str(e)}"
            self.logger.log_error(error_msg=error_msg, error_type="init_dependent_error", stack_trace=traceback.format_exc())
            raise
    
    def _connect_component_references(self):
        """Connect cross-references between components safely after all are initialized."""
        try:
            # Connect Scriber to state_accessor
            if hasattr(self.scriber, 'set_state_accessor'):
                self.scriber.set_state_accessor(self)
            else:
                self.logger.log_warning("Scriber doesn't have set_state_accessor method")
            
            # Connect Generation Manager, if it exists
            if hasattr(self, 'generation_manager') and self.generation_manager:
                if hasattr(self.generation_manager, 'set_system_context'):
                    self.generation_manager.set_system_context(self)
                else:
                    self.logger.log_warning("GenerationManager doesn't have set_system_context method")
            
            # Start the Scriber's writer thread now that connections are established
            if hasattr(self.scriber, '_initialize_writer_thread'):
                self.scriber._initialize_writer_thread()
            
            # Log connections established
            self.logger.log_info("Component cross-references connected successfully")
            
        except Exception as e:
            error_msg = f"Failed to connect component references: {str(e)}"
            self.logger.log_error(error_msg=error_msg, error_type="init_connect_error", stack_trace=traceback.format_exc())
            raise
    
    def _cleanup_partial_initialization(self):
        """Clean up resources after partial initialization failure."""
        # Clean up in reverse order of dependency
        for component_name in reversed(list(self._initialized_components)):
            component = getattr(self, component_name, None)
            if component and hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                    self.logger.log_info(f"Cleaned up {component_name}")
                except Exception as e:
                    if hasattr(self, 'logger') and self.logger:
                        self.logger.log_error(
                            error_msg=f"Error cleaning up {component_name}: {str(e)}",
                            error_type="cleanup_error"
                        )
                    else:
                        print(f"Error cleaning up {component_name}: {str(e)}")
        
        # Reset initialization flags
        self._initialized = False
        self._initialization_complete.clear()
        self._initialized_components.clear()

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
        """Reads the last session ID, increments it, and writes it back atomically."""
        self._ensure_session_dir()
        last_id = 0
        
        try:
            # Open file in read+write mode for atomic operations
            with open(SystemConstants.SESSION_COUNTER_FILE, 'a+') as f:
                # Use platform-specific file locking
                try:
                    # For Unix/Linux/Mac
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX)
                except (ImportError, AttributeError):
                    try:
                        # For Windows
                        import msvcrt
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    except (ImportError, AttributeError):
                        # Fallback - we're already using a Python lock
                        # but file operations themselves aren't atomic
                        print("Warning: File locking not available for session counter; using synchronized lock only", 
                              file=sys.stderr)
                
                try:
                    # Read current value
                    f.seek(0)
                    content = f.read().strip()
                    
                    if content.isdigit():
                        last_id = int(content)
                    else:
                        # File exists but content is invalid, try backup
                        self._restore_session_counter()
                        f.seek(0)
                        content = f.read().strip()
                        if content.isdigit():
                            last_id = int(content)
                    
                    # Increment and write back atomically
                    current_id = last_id + 1
                    f.seek(0)
                    f.truncate(0)
                    f.write(str(current_id))
                    f.flush()
                    os.fsync(f.fileno())  # Force sync to disk
                    
                    # Create backup after successful update
                    self._backup_session_counter()
                    
                    return current_id
                    
                finally:
                    # Release lock
                    try:
                        # Unix/Linux/Mac
                        fcntl.flock(f, fcntl.LOCK_UN)
                    except (NameError, AttributeError):
                        try:
                            # Windows
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        except (NameError, AttributeError):
                            pass  # No locking, so nothing to release
                            
        except Exception as e:
            print(f"Warning: Could not read/write session ID counter file: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(
                    error_msg=f"Failed to get next session ID: {str(e)}",
                    error_type="session_id",
                    stack_trace=traceback.format_exc()
                )
            
            # Generate a random session ID as fallback to avoid collisions
            import random
            import time
            return int(time.time() * 1000) % 1000000 + random.randint(1000, 9999)
    
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

    def _handle_initialization_error(self, error: Exception):
        """Handle initialization errors safely."""
        try:
            # First make sure initialization is marked as failed
            self._initialization_complete.set()  # Allow waiters to proceed
            
            # Log the error to a file if logger isn't available
            log_msg = f"System initialization failed: {str(error)}\n{traceback.format_exc()}"
            
            if hasattr(self, 'logger') and self.logger:
                try:
                    self.logger.log_error(
                        error_msg=f"System initialization failed: {str(error)}",
                        error_type="system_initialization",
                        stack_trace=traceback.format_exc()
                    )
                except Exception as log_err:
                    print(f"Additionally, failed to log error: {str(log_err)}", file=sys.stderr)
                    # Try to write to an emergency file
                    try:
                        with open("sovl_initialization_error.log", "a") as f:
                            f.write(f"[{time.ctime()}] {log_msg}\n")
                    except Exception:
                        pass  # Nothing more we can do
            else:
                print(f"Critical error during initialization (logger unavailable): {str(error)}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                # Try to write to an emergency file
                try:
                    with open("sovl_initialization_error.log", "a") as f:
                        f.write(f"[{time.ctime()}] {log_msg}\n")
                except Exception:
                    pass  # Nothing more we can do
        except Exception as e2:
            print(f"Critical error during initialization error handling: {str(error)}", file=sys.stderr)
            print(f"Additionally, error handler failed: {str(e2)}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    
    def _start_memory_monitoring(self):
        """Start proactive memory monitoring during initialization."""
        def monitor_memory():
            while not self._initialization_complete.is_set():
                try:
                    if hasattr(self, 'ram_manager') and hasattr(self, 'gpu_manager'):
                        # Use getattr with None default to avoid AttributeError
                        ram_manager = getattr(self, 'ram_manager', None)
                        gpu_manager = getattr(self, 'gpu_manager', None)
                        
                        if ram_manager and gpu_manager:
                            with self._lock:
                                if hasattr(self, 'system_state'):
                                    self.system_state['memory_usage'] = {
                                        'ram': ram_manager.get_usage(),
                                        'gpu': gpu_manager.get_usage()
                                    }
                    
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
                    except Exception as e2:
                        print(f"Critical error in startup memory monitor exception handler: {str(e)}", file=sys.stderr)
                        print(f"Additionally, error handler failed: {str(e2)}", file=sys.stderr)
                    
                    time.sleep(5)  # Wait longer on error
        
        try:
            # Make sure the initialization event is properly initialized
            if not hasattr(self, '_initialization_complete') or self._initialization_complete is None:
                self._initialization_complete = Event()
                
            # Create and start the monitor thread
            self._startup_monitor = Thread(target=monitor_memory, daemon=True, name="SOVLStartupMonitor")
            self._startup_monitor.start()
            
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_debug("Startup memory monitoring started")
                
        except Exception as e:
            print(f"Error starting memory monitoring: {str(e)}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    
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
    def update_component_state(self, component_name: str, state: Dict[str, Any]):
        """Update the state of a component in a thread-safe manner."""
        with self._lock:
            self._component_states[component_name] = state
            self.system_state['component_states'][component_name] = state
    
    @synchronized("_lock")
    def get_component_state(self, component_name: str) -> Dict[str, Any]:
        """Get the current state of a component in a thread-safe manner."""
        with self._lock:
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
        # Ensure we're initialized before accessing components
        if not self.is_fully_initialized():
            return {
                'status': 'initializing',
                'session_id': getattr(self, 'session_id', None),
                'start_time': time.time(),
                'error': 'SystemContext not fully initialized'
            }
        
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
                self.logger.log_info("SystemContext initialization sequence complete")
                
        except Exception as e:
            try:
                if hasattr(self, 'error_handler') and self.error_handler:
                    self.error_handler.handle_error(
                        error_type="initialization_completion",
                        error_message=f"Failed to complete context initialization sequence: {str(e)}",
                        error_context={
                            "initialization_state": self._initialization_complete.is_set() if hasattr(self, '_initialization_complete') else 'Unknown'
                        }
                    )
                elif hasattr(self, 'logger') and self.logger:
                    self.logger.log_error(
                        error_msg=f"Failed to complete context initialization sequence: {str(e)}",
                        error_type="initialization_completion",
                        stack_trace=traceback.format_exc()
                    )
                else:
                    print(f"Error completing SystemContext initialization sequence (logger/handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
            except Exception as err2:
                print(f"Critical error in initialization completion exception handler: {str(err2)}", file=sys.stderr)
                print(f"Original error: {str(e)}", file=sys.stderr)

    def add_message_to_memory(self, role: str, content: str, user_id: str = "default"):
        if self.memory_context:
            self.memory_context.add_message(role, content, user_id)

    def get_short_term_context(self):
        if self.memory_context:
            return self.memory_context.get_short_term_context()
        return []

    def get_long_term_context(self, user_id: str = "default", query_embedding=None, top_k: int = 5):
        if self.memory_context:
            return self.memory_context.get_long_term_context(user_id, query_embedding, top_k)
        return []

    def clear_short_term_memory(self):
        if self.memory_context:
            self.memory_context.clear_short_term_memory()

    def clear_long_term_memory(self, user_id: str = "default"):
        if self.memory_context:
            self.memory_context.clear_long_term_memory(user_id)

    def bind_generation_manager(self, generation_manager):
        """
        Bind a GenerationManager instance to this SystemContext.
        This method is designed to be called after all components are initialized.
        
        Args:
            generation_manager: Instance of GenerationManager to bind
        """
        if not self._initialized:
            self.logger.log_warning("Attempting to bind GenerationManager before SystemContext is fully initialized")
            
        self.generation_manager = generation_manager
        self._initialized_components.add("generation_manager")
        
        # Connect the generation manager to the system context
        if hasattr(generation_manager, 'set_system_context'):
            generation_manager.set_system_context(self)
            self.logger.log_info("GenerationManager bound to SystemContext")
        else:
            self.logger.log_warning("GenerationManager does not implement set_system_context method")
            
        # Update any cross-references that depend on GenerationManager
        if hasattr(self, 'meditator') and self.meditator and hasattr(self.meditator, 'set_generation_manager'):
            self.meditator.set_generation_manager(generation_manager)
            
        return self.generation_manager

    # Add a method to check if fully initialized
    def is_fully_initialized(self) -> bool:
        """Check if the SystemContext is fully initialized and ready for use."""
        if not self._initialized:
            return False
        
        # Check critical components
        required_components = [
            'config_manager', 'logger', 'error_handler', 'state_manager',
            'ram_manager', 'gpu_manager', 'event_dispatcher'
        ]
        
        for component in required_components:
            if not hasattr(self, component) or getattr(self, component) is None:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.log_warning(
                        f"SystemContext not fully initialized: missing {component}")
                return False
        
        return True
        
    # Add method to wait for initialization to complete
    def wait_for_initialization(self, timeout: float = SystemConstants.COMPONENT_INIT_TIMEOUT) -> bool:
        """
        Wait for the initialization to complete with a timeout.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if initialization completed, False if timed out
        """
        if self._initialized and self.is_fully_initialized():
            return True
            
        # Use the initialization event with a timeout
        if hasattr(self, '_initialization_complete'):
            return self._initialization_complete.wait(timeout)
            
        # Fallback
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._initialized and self.is_fully_initialized():
                return True
            time.sleep(0.1)
            
        return False

    @classmethod
    def initialize_singleton(cls, config_path: str = None) -> 'SystemContext':
        """
        Thread-safe way to initialize the singleton instance.
        This method ensures that only one thread creates and initializes the SystemContext.
        
        Args:
            config_path: Optional path to the configuration file
            
        Returns:
            The fully initialized singleton instance
            
        Raises:
            SystemInitializationError: If initialization fails
        """
        config_path = config_path or SystemConstants.DEFAULT_CONFIG_PATH
        
        with cls._lock:
            if cls._instance is None or not cls._instance._initialized:
                # Create and initialize the instance
                instance = cls(config_path)
                
                # Wait for initialization to complete
                if not instance.wait_for_initialization():
                    raise SystemInitializationError(
                        message="Timeout waiting for SystemContext initialization",
                        config_path=config_path,
                        stack_trace=""
                    )
                
                return instance
            else:
                # Return the existing instance
                return cls._instance

    def create_and_bind_generation_manager(self):
        """
        Create and bind a GenerationManager with all proper dependencies.
        This method should be called after SystemContext is fully initialized.
        
        Returns:
            The newly created and bound GenerationManager instance
        """
        if not self._initialized:
            self.logger.log_warning("Attempting to create GenerationManager before SystemContext is fully initialized")
            self.wait_for_initialization()
        
        try:
            # Check if model manager is available
            if not hasattr(self, 'model_manager') or self.model_manager is None:
                self.logger.log_error("ModelManager is not initialized. Cannot create GenerationManager.")
                return None
            
            # Create the GenerationManager with proper dependencies
            generation_manager = GenerationManager(
                config_manager=self.config_manager,
                base_model=self.model_manager.get_base_model(),
                scaffolds=[self.model_manager.get_scaffold_model(i) for i in range(self.model_manager.get_scaffold_model_count())],
                base_tokenizer=self.model_manager.get_base_tokenizer(),
                scaffold_tokenizer=self.model_manager.get_scaffold_tokenizer(),
                state=self.state_manager.get_state(),
                logger=self.logger,
                error_manager=self.error_handler,
                cross_attention_injector=self.model_manager.get_cross_attention_injector() if hasattr(self.model_manager, 'get_cross_attention_injector') else None,
                scaffold_manager=self.model_manager.get_scaffold_manager() if hasattr(self.model_manager, 'get_scaffold_manager') else None,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                curiosity_manager=self.curiosity_manager if hasattr(self, 'curiosity_manager') else None,
                dialogue_context_manager=self.dialogue_context_manager if hasattr(self, 'dialogue_context_manager') else None,
                state_manager=self.state_manager
            )
            
            # Bind the GenerationManager to the SystemContext
            self.bind_generation_manager(generation_manager)
            
            return generation_manager
            
        except Exception as e:
            error_msg = f"Failed to create GenerationManager: {str(e)}"
            self.logger.log_error(error_msg=error_msg, error_type="generation_manager_creation_error", stack_trace=traceback.format_exc())
            return None

    def initialize_model_manager(self):
        """
        Initialize the ModelManager separately after core components are initialized.
        This allows proper handling of its heavy resource requirements and dependencies.
        
        Returns:
            bool: Whether initialization was successful
        """
        if not self._initialized_components.issuperset({"config_manager", "logger"}):
            error_msg = "Cannot initialize ModelManager: required core components not available"
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(error_msg=error_msg, error_type="dependency_error")
            else:
                print(f"ERROR: {error_msg}")
            return False
            
        try:
            # Create and initialize the ModelManager
            self.model_manager = ModelManager(
                config_manager=self.config_manager,
                logger=self.logger,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._initialized_components.add("model_manager")
            
            # Initialize DialogueContextManager now that ModelManager is available
            if "state_manager" in self._initialized_components:
                self.dialogue_context_manager = DialogueContextManager(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    model_manager=self.model_manager,
                    state_manager=self.state_manager
                )
                self._initialized_components.add("dialogue_context_manager")
            else:
                self.logger.log_warning("Cannot initialize DialogueContextManager: state_manager not available")
            
            # Log successful initialization
            self.logger.log_info("ModelManager initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize ModelManager: {str(e)}"
            if hasattr(self, 'logger') and self.logger:
                self.logger.log_error(
                    error_msg=error_msg,
                    error_type="model_manager_init_error",
                    stack_trace=traceback.format_exc()
                )
            else:
                print(f"ERROR: {error_msg}\n{traceback.format_exc()}")
            return False

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
            
            # Initialize monitoring state
            self._monitoring_active = False
            self._stop_monitoring_event = Event()
            self._monitor_thread = None
            
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
            
            # --- Integrate AutonomyManager elegantly ---
            self.autonomy_manager = AutonomyManager(
                config_manager=context.config_manager if hasattr(context, 'config_manager') else None,
                logger=context.logger if hasattr(context, 'logger') else None,
                device=getattr(model_manager, 'device', None),
                system_ref=self,
                tuner=getattr(context, 'tuner', None) if hasattr(context, 'tuner') else None
            )
            # Optionally, attach to context for global access
            if not hasattr(context, 'autonomy_manager'):
                context.autonomy_manager = self.autonomy_manager
            
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
                # Use atomic update protocol
                if hasattr(self.context, 'state_manager') and self.context.state_manager:
                    def update_fn(state):
                        # This assumes SOVLState has a from_dict method that mutates in place
                        state.from_dict(state_dict, getattr(self.context, 'device', None))
                    self.context.state_manager.update_state_atomic(update_fn)
                else:
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
        """Start system monitoring components with validation and fallbacks."""
        with self._lock:
            # Track which monitors we've started successfully
            started_monitors = []
            failed_monitors = []
            self._monitoring_active = True  # Track if we have any active monitoring
            
            try:
                # First validate that monitors exist and are in valid state
                monitor_checks = [
                    ("memory_monitor", lambda m: hasattr(m, "start_monitoring") or hasattr(m, "check_memory_health")),
                    ("system_monitor", lambda m: hasattr(m, "start_monitoring") or hasattr(m, "get_system_metrics")),
                    ("state_tracker", lambda m: hasattr(m, "start_monitoring") or hasattr(m, "get_state_hash")),
                    ("error_manager", lambda m: hasattr(m, "handle_error")),
                    ("traits_monitor", lambda m: hasattr(m, "start_monitoring") or hasattr(m, "_get_current_traits"))
                ]
                
                for monitor_name, check_func in monitor_checks:
                    monitor = getattr(self, monitor_name, None)
                    if monitor is None:
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_warning(f"Monitor '{monitor_name}' not available")
                        failed_monitors.append(monitor_name)
                        continue
                        
                    if not check_func(monitor):
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_warning(
                                f"Monitor '{monitor_name}' missing required methods")
                        failed_monitors.append(monitor_name)
                        continue
                
                # Start memory monitoring if available
                if hasattr(self, 'memory_monitor') and self.memory_monitor:
                    try:
                        if hasattr(self.memory_monitor, 'start_monitoring'):
                            self.memory_monitor.start_monitoring()
                            started_monitors.append("memory_monitor")
                            if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                self.context.logger.log_info("SOVL Memory monitoring started.")
                        else:
                            if not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
                                self._stop_monitoring_event = Event()
                                self._monitor_thread = Thread(
                                    target=self._monitor_memory,
                                    daemon=True,
                                    name="SOVLMemoryMonitor"
                                )
                                self._monitor_thread.start()
                                started_monitors.append("internal_memory_monitor")
                                if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                    self.context.logger.log_info("SOVL internal memory monitoring thread started.")
                    except Exception as e:
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_error(
                                f"Failed to start memory monitor: {str(e)}",
                                error_type="monitor_startup_failure"
                            )
                        failed_monitors.append("memory_monitor")
                        # Fall back to internal monitoring if external fails
                        if not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
                            try:
                                self._stop_monitoring_event = Event()
                                self._monitor_thread = Thread(
                                    target=self._monitor_memory,
                                    daemon=True,
                                    name="SOVLMemoryMonitor_Fallback"
                                )
                                self._monitor_thread.start()
                                started_monitors.append("internal_memory_monitor_fallback")
                                if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                    self.context.logger.log_info(
                                        "SOVL fallback memory monitoring started after external monitor failure.")
                            except Exception as e2:
                                if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                    self.context.logger.log_error(
                                        f"Failed to start fallback memory monitor: {str(e2)}",
                                        error_type="fallback_monitor_failure"
                                    )
                # If no memory monitoring has been started, try to start internal monitoring
                elif not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
                    try:
                        self._stop_monitoring_event = Event()
                        self._monitor_thread = Thread(
                            target=self._monitor_memory,
                            daemon=True,
                            name="SOVLMemoryMonitor_Primary"
                        )
                        self._monitor_thread.start()
                        started_monitors.append("internal_memory_monitor_primary")
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_info("SOVL primary memory monitoring thread started.")
                    except Exception as e:
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_error(
                                f"Failed to start primary memory monitor: {str(e)}",
                                error_type="monitor_startup_failure"
                            )
                        failed_monitors.append("internal_memory_monitor")

                # Start other monitoring systems with validation and individual error handling
                other_monitors = [
                    ("state_tracker", self.state_tracker),
                    ("error_manager", self.error_manager),
                    ("system_monitor", self.system_monitor),
                    ("traits_monitor", self.traits_monitor)
                ]
                
                for name, monitor in other_monitors:
                    if monitor is None:
                        continue
                        
                    try:
                        if hasattr(monitor, 'start_monitoring'):
                            monitor.start_monitoring()
                            started_monitors.append(name)
                        elif hasattr(monitor, 'start'):
                            monitor.start()
                            started_monitors.append(name)
                    except Exception as e:
                        if hasattr(self, 'error_manager') and self.error_manager:
                            self.error_manager.handle_error(
                                error_type="monitor_startup",
                                error_message=f"Failed to start {name}: {str(e)}",
                                error=e
                            )
                        failed_monitors.append(name)
                
                # Log monitoring status
                if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                    if started_monitors:
                        self.context.logger.log_info(
                            f"Started monitors: {', '.join(started_monitors)}")
                    if failed_monitors:
                        self.context.logger.log_warning(
                            f"Failed to start monitors: {', '.join(failed_monitors)}")
                        
            except Exception as e:
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="monitoring_startup",
                        error_message=f"Failed to start SOVLSystem monitoring: {str(e)}",
                        error_context={
                            "started_monitors": started_monitors,
                            "failed_monitors": failed_monitors,
                            "monitor_thread_alive": hasattr(self, '_monitor_thread') and 
                                                   self._monitor_thread.is_alive() if hasattr(self, '_monitor_thread') else False
                        },
                        error=e
                    )
                else:
                    print(f"Error starting SOVLSystem monitoring (handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

    def _monitor_memory(self):
        """Main memory monitoring loop with exponential backoff and thresholds."""
        self._stop_monitoring_event = Event()
        error_count = 0
        base_sleep_time = 5  # Normal interval (seconds)
        max_sleep_time = 300  # Maximum backoff (5 minutes)
        max_consecutive_errors = 10  # Threshold for disabling monitoring
        
        # Define a local helper function for emergency shutdown within the thread
        def _monitor_shutdown_unsafe():
            """Emergency shutdown of monitoring when in error state."""
            try:
                if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                    self.context.logger.log_critical(
                        "Memory monitoring thread shutting down due to repeated failures")
                # Set the flag so other code knows we're no longer monitoring
                self._monitoring_active = False
            except Exception:
                # This is a last-resort method, so we silently fail
                pass
        
        while not self._stop_monitoring_event.is_set():
            try:
                if self.context and hasattr(self.context, 'update_memory_usage') and hasattr(self.context, '_cleanup_old_errors'):
                    self.context.update_memory_usage()
                    self.context._cleanup_old_errors()
                
                # Reset error count and sleep time after successful execution
                if error_count > 0:
                    if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                        self.context.logger.log_info(f"Memory monitoring recovered after {error_count} consecutive errors")
                    error_count = 0
                    sleep_time = base_sleep_time
                else:
                    sleep_time = base_sleep_time
                    
                self._stop_monitoring_event.wait(sleep_time)
                
            except Exception as e:
                error_count += 1
                # Calculate exponential backoff with jitter
                import random
                sleep_time = min(base_sleep_time * (2 ** min(error_count, 6)) + random.uniform(0, 1), max_sleep_time)
                
                # Log the error with appropriate severity
                if hasattr(self, 'error_manager') and self.error_manager:
                    severity = "critical" if error_count >= max_consecutive_errors else "error"
                    self.error_manager.handle_error(
                        error_type="memory_monitoring",
                        error_message=f"SOVLSystem memory monitoring error (attempt {error_count}): {str(e)}",
                        error_context={
                            "monitoring_phase": "main",
                            "consecutive_errors": error_count,
                            "backoff_seconds": sleep_time
                        },
                        error=e
                    )
                else:
                    print(f"Error in SOVLSystem memory monitor loop (handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                
                # Check for threshold breach
                if error_count >= max_consecutive_errors:
                    if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                        self.context.logger.log_critical(
                            f"Memory monitoring disabled after {error_count} consecutive failures")
                    
                    if hasattr(self, 'error_manager') and self.error_manager:
                        self.error_manager.handle_error(
                            error_type="monitoring_failure",
                            error_message=f"Memory monitoring disabled after {error_count} consecutive failures",
                            error_context={"monitoring_component": "SOVLSystem._monitor_memory"},
                            error=e
                        )
                    # Call local helper for emergency shutdown
                    _monitor_shutdown_unsafe()
                    # Break out of the monitoring loop
                    break
                
                self._stop_monitoring_event.wait(sleep_time)

        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
            self.context.logger.log_info("SOVL main memory monitoring loop stopped.")

    def stop_monitoring(self):
        """Stop system monitoring components with graceful thread termination."""
        with self._lock:
            # Track which monitors we've stopped successfully
            stopped_monitors = []
            failed_stops = []
            
            try:
                # Stop main memory monitoring thread if it exists and is alive
                if hasattr(self, '_monitor_thread') and self._monitor_thread and self._monitor_thread.is_alive():
                    try:
                        if hasattr(self, '_stop_monitoring_event'):
                            self._stop_monitoring_event.set()
                            
                        # Join with timeout to ensure it stops
                        self._monitor_thread.join(timeout=5.0)
                        if self._monitor_thread.is_alive():
                            if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                self.context.logger.log_warning(
                                    "Memory monitoring thread did not terminate within timeout")
                            failed_stops.append("internal_memory_monitor")
                        else:
                            stopped_monitors.append("internal_memory_monitor")
                            if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                self.context.logger.log_info("SOVL main memory monitoring thread stopped.")
                    except Exception as e:
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_error(
                                f"Error stopping memory monitoring thread: {str(e)}",
                                error_type="monitor_stop_failure"
                            )
                        failed_stops.append("internal_memory_monitor")

                # Stop memory monitor instance if it has a stop method
                if hasattr(self, 'memory_monitor') and self.memory_monitor:
                    try:
                        if hasattr(self.memory_monitor, 'stop_monitoring'):
                            self.memory_monitor.stop_monitoring()
                            stopped_monitors.append("memory_monitor")
                            if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                                self.context.logger.log_info("SOVL Memory monitoring stopped.")
                    except Exception as e:
                        if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                            self.context.logger.log_error(
                                f"Error stopping memory monitor: {str(e)}",
                                error_type="monitor_stop_failure"
                            )
                        failed_stops.append("memory_monitor")

                # Stop other monitoring systems with validation and individual error handling
                other_monitors = [
                    ("state_tracker", self.state_tracker),
                    ("error_manager", self.error_manager),
                    ("system_monitor", self.system_monitor),
                    ("traits_monitor", self.traits_monitor)
                ]
                
                for name, monitor in other_monitors:
                    if monitor is None:
                        continue
                        
                    try:
                        if hasattr(monitor, 'stop_monitoring'):
                            monitor.stop_monitoring()
                            stopped_monitors.append(name)
                        elif hasattr(monitor, 'stop'):
                            monitor.stop()
                            stopped_monitors.append(name)
                    except Exception as e:
                        if hasattr(self, 'error_manager') and self.error_manager:
                            self.error_manager.handle_error(
                                error_type="monitor_shutdown",
                                error_message=f"Failed to stop {name}: {str(e)}",
                                error=e
                            )
                        failed_stops.append(name)
                
                # Log monitoring status
                if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                    if stopped_monitors:
                        self.context.logger.log_info(
                            f"Stopped monitors: {', '.join(stopped_monitors)}")
                    if failed_stops:
                        self.context.logger.log_warning(
                            f"Failed to stop monitors: {', '.join(failed_stops)}")
                
                # Update monitoring status
                self._monitoring_active = False
                    
            except Exception as e:
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="monitoring_shutdown",
                        error_message=f"Failed to stop SOVLSystem monitoring: {str(e)}",
                        error_context={
                            "stopped_monitors": stopped_monitors,
                            "failed_stops": failed_stops,
                            "monitor_thread_alive": hasattr(self, '_monitor_thread') and 
                                                  self._monitor_thread.is_alive() if hasattr(self, '_monitor_thread') else False
                        },
                        error=e
                    )
                else:
                    print(f"Error stopping SOVLSystem monitoring (handler unavailable): {str(e)}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

    def check_monitoring_health(self) -> Dict[str, Any]:
        """Check the health of all monitoring components and their threads."""
        health_status = {
            "monitoring_active": getattr(self, "_monitoring_active", False),
            "monitors": {},
            "threads": {}
        }
        
        # Check monitor instances
        monitor_map = {
            "memory_monitor": self.memory_monitor,
            "system_monitor": self.system_monitor,
            "state_tracker": self.state_tracker,
            "error_manager": self.error_manager,
            "traits_monitor": self.traits_monitor
        }
        
        for monitor_name, monitor in monitor_map.items():
            if monitor is None:
                health_status["monitors"][monitor_name] = {"status": "unavailable"}
                continue
                
            try:
                # Try to call a health check method if available
                if hasattr(monitor, "check_health"):
                    result = monitor.check_health()
                    health_status["monitors"][monitor_name] = {
                        "status": "healthy" if result.get("healthy", False) else "unhealthy",
                        "details": result
                    }
                elif hasattr(monitor, "is_alive"):
                    health_status["monitors"][monitor_name] = {
                        "status": "active" if monitor.is_alive() else "inactive"
                    }
                elif hasattr(monitor, "check_memory_health") and monitor_name == "memory_monitor":
                    try:
                        result = monitor.check_memory_health()
                        health_status["monitors"][monitor_name] = {
                            "status": "healthy",
                            "details": result
                        }
                    except Exception as e:
                        health_status["monitors"][monitor_name] = {
                            "status": "error",
                            "error": str(e)
                        }
                else:
                    health_status["monitors"][monitor_name] = {
                        "status": "unknown", 
                        "exists": True
                    }
            except Exception as e:
                health_status["monitors"][monitor_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Check monitoring threads
        thread_map = {
            "memory_monitor": getattr(self, "_monitor_thread", None)
        }
        
        for thread_name, thread in thread_map.items():
            if thread is None:
                health_status["threads"][thread_name] = {"status": "not_running"}
                continue
                
            try:
                health_status["threads"][thread_name] = {
                    "status": "active" if thread.is_alive() else "stopped",
                    "name": getattr(thread, "name", "unnamed"),
                    "daemon": getattr(thread, "daemon", None),
                    "ident": getattr(thread, "ident", None)
                }
            except Exception as e:
                health_status["threads"][thread_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Add recovery suggestions based on status
        health_status["recommendations"] = []
        
        # Check if any monitors are unhealthy
        unhealthy_monitors = [
            name for name, info in health_status["monitors"].items()
            if info.get("status") in ["error", "unhealthy"]
        ]
        if unhealthy_monitors:
            health_status["recommendations"].append({
                "type": "monitor_restart",
                "message": f"Consider restarting unhealthy monitors: {', '.join(unhealthy_monitors)}",
                "monitors": unhealthy_monitors
            })
            
        # Check if monitoring thread is inactive but should be running
        for thread_name, thread_info in health_status["threads"].items():
            if thread_info.get("status") == "stopped" and health_status["monitoring_active"]:
                health_status["recommendations"].append({
                    "type": "thread_restart",
                    "message": f"Monitoring thread '{thread_name}' is stopped but monitoring is active",
                    "thread": thread_name
                })
        
        return health_status

    def _emergency_monitor_shutdown(self):
        """
        Emergency termination of monitoring components when in a critical error state.
        This is called when we need to force-stop monitoring regardless of state.
        """
        try:
            # First log that we're doing emergency shutdown
            if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                self.context.logger.log_critical(
                    "Performing emergency shutdown of monitoring components")
            else:
                print("CRITICAL: Emergency shutdown of monitoring components", file=sys.stderr)
            
            # Force terminate all monitoring threads
            if hasattr(self, '_monitor_thread') and self._monitor_thread:
                try:
                    if hasattr(self, '_stop_monitoring_event'):
                        self._stop_monitoring_event.set()
                    # Give it a very short time to terminate
                    self._monitor_thread.join(timeout=1.0)
                except Exception:
                    pass  # Ignore any errors during emergency shutdown
            
            # Set status flags
            self._monitoring_active = False
            
            # Log completion
            if hasattr(self, 'context') and self.context and hasattr(self.context, 'logger'):
                self.context.logger.log_warning(
                    "Emergency monitoring shutdown completed")
            
        except Exception as e:
            # Last-resort error handling - if even this fails, just print to stderr
            print(f"CRITICAL: Failed during emergency monitoring shutdown: {str(e)}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
