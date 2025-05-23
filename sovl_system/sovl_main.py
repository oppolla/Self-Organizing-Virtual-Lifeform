# Standard library imports
from typing import Optional, Any, List, Dict, Tuple, Callable, TYPE_CHECKING
import time
import traceback
import os
import sys
from collections import deque, defaultdict
from threading import Lock, RLock, Event, Thread
import math
import threading
import select

# Core components
from sovl_config import ConfigManager
from sovl_state import StateManager, StateTracker
from sovl_error import ErrorManager
from sovl_logger import Logger
from sovl_events import EventDispatcher
from sovl_interfaces import SystemInterface
from sovl_queue import get_scribe_queue
from sovl_scribe import Scriber
from sovl_volition import AutonomyManager
from sovl_chronos import Chronos
from sovl_dreamer import Dreamer
from sovl_tooler import Tooler, ToolRegistry, ProcedureManager

# Model and processing
from sovl_manager import ModelManager
from sovl_processor import MetadataProcessor
from sovl_generation import GenerationManager

# Memory and state management
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_recaller import DialogueContextManager

# AI components
from sovl_bonder import BondCalculator, BondModulator
from sovl_curiosity import CuriosityManager
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_meditater import IntrospectionManager
from sovl_scaffold import ScaffoldTokenMapper

# Monitoring components
from sovl_monitor import MonitorManager

# Utilities
from sovl_utils import (
    synchronized,
    atomic_file_counter,
    safe_append_to_file
)
from sovl_confidence import calculate_confidence_score, ConfidenceCalculator
from sovl_io import  prune_scribe_journal
from sovl_trainer import TrainingConfig, SOVLTrainer, TrainingWorkflowManager

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

    # Grafter system
    GRAFTER_DIR = "plugins"
    MAX_GRAFTS = 10
    GRAFT_TIMEOUT = 30.0  # seconds
    GRAFT_RETRY_DELAY = 1.0  # seconds
    GRAFT_ERROR_COOLDOWN = 5.0  # seconds

# --- Autonomous Tiredness & Sleep Logic ---
TIREDNESS_THRESHOLD = 0.7  # Tune as needed
TIREDNESS_CHECK_INTERVAL = 10  # seconds

class SystemContext:
    """Manages system-wide context and resources. All dependencies must be injected."""

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_manager,
        logger,
        error_handler,
        state_manager,
        model_manager,
        ram_manager,
        gpu_manager,
        event_dispatcher,
        session_id=None,
        session_lock=None,
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.error_handler = error_handler
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self.event_dispatcher = event_dispatcher
        self.grafter = None  # Will be set by SOVLSystem
        # Session ID logic
        if session_id is not None:
            self.session_id = session_id
        else:
            from sovl_main import SESSION_COUNTER_FILE, SESSION_COUNTER_BACKUP
            lock = session_lock or Lock()
            self.session_id = atomic_file_counter(
                SESSION_COUNTER_FILE,
                SESSION_COUNTER_BACKUP,
                lock,
                logger=self.logger
            )

class SystemInitializationError(Exception):
    """Custom exception for system initialization failures."""
    
    def __init__(self, message: str, config_path: str, stack_trace: str):
        self.message = message
        self.config_path = config_path
        self.stack_trace = stack_trace
        super().__init__(f"{message}\nConfig path: {config_path}\nStack trace:\n{stack_trace}")

class TirednessManager:
    """Manages tiredness, gestation, and sleep logic for the SOVL system."""
    def __init__(self, state_manager, logger, config_handler):
        self.state_manager = state_manager
        self.logger = logger
        self.config_handler = config_handler
        gestation_cfg = self.config_handler.get('gestation_config', {}) if self.config_handler else {}
        self.tiredness_threshold = gestation_cfg.get('tiredness_threshold', 0.7)
        self.tiredness_check_interval = gestation_cfg.get('tiredness_check_interval', 10)
        self.tiredness_decay_k = gestation_cfg.get('tiredness_decay_k', 0.01)
        self.sleep_log_min = gestation_cfg.get('sleep_log_min', 10)
        self.gestation_countdown_seconds = gestation_cfg.get('gestation_countdown_seconds', 30)
        self.tiredness_weights = gestation_cfg.get('tiredness_weights', {"log": 0.4, "confidence": 0.3, "time": 0.3})
        self.min_awake_seconds = gestation_cfg.get('min_awake_seconds', 60)
        self.max_awake_seconds = gestation_cfg.get('max_awake_seconds', 7200)
        self.post_abort_cooldown_seconds = gestation_cfg.get('post_abort_cooldown_seconds', 120)
        self.last_gestation_time = time.time()
        self.last_gestation_abort_time = 0.0
        self.current_tiredness_threshold = self.tiredness_threshold
        self._pending_gestation_start = None
        self._gestating_start = None
        self._last_gestation_reason = None
        self._gestation_event_log = []
        self._monitor_thread = None
        self._stop_event = threading.Event()

    def _compute_tiredness(self):
        """Compute system tiredness based on log size, confidence, exposure, and time since last sleep."""
        try:
            log_entries = self.logger.read()
        except Exception:
            log_entries = []
        new_entries = len(log_entries)
        log_factor = min(1.0, new_entries / getattr(self, 'sleep_log_min', 10))
        conf_hist = getattr(self, 'confidence_history', [])
        conf = 1.0 - (sum(conf_hist) / len(conf_hist) if conf_hist else 0.5)
        k = getattr(self, 'tiredness_decay_k', 0.01)
        data_exposure = getattr(self, 'data_exposure', 0.0)
        exposure_factor = math.exp(-k * data_exposure)
        last_trained = getattr(self, 'last_trained', 0)
        time_since_sleep = (time.time() - last_trained) / 3600.0 if last_trained else 0.0
        time_factor = min(1.0, time_since_sleep / 2.0)
        weights = getattr(self, 'tiredness_weights', {"log": 0.4, "confidence": 0.3, "time": 0.3})
        tiredness = (weights.get("log", 0.4) * log_factor + weights.get("confidence", 0.3) * conf + weights.get("time", 0.3) * time_factor) * exposure_factor
        return tiredness

    def start(self):
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(target=self._tiredness_monitor_loop, daemon=True)
            self._monitor_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _tiredness_monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                now = time.time()
                tiredness = self._compute_tiredness()
                if hasattr(self.state_manager, 'set_tiredness'):
                    self.state_manager.set_tiredness(tiredness)
                mode = self.state_manager.get_mode() if hasattr(self.state_manager, 'get_mode') else 'online'
                stuck_limit = 2 * self.gestation_countdown_seconds
                if mode == 'pending_gestation':
                    if self._pending_gestation_start is None:
                        self._pending_gestation_start = now
                    elif now - self._pending_gestation_start > stuck_limit:
                        print("[WARN] Stuck in pending_gestation, resetting to online.")
                        self.state_manager.set_mode('online')
                        self._pending_gestation_start = None
                        continue
                else:
                    self._pending_gestation_start = None
                if mode == 'gestating':
                    if self._gestating_start is None:
                        self._gestating_start = now
                    elif now - self._gestating_start > stuck_limit:
                        print("[WARN] Stuck in gestating, resetting to online.")
                        self.state_manager.set_mode('online')
                        self._gestating_start = None
                        continue
                else:
                    self._gestating_start = None
                if now - self.last_gestation_time < self.min_awake_seconds:
                    time.sleep(self.tiredness_check_interval)
                    continue
                if now - self.last_gestation_abort_time < self.post_abort_cooldown_seconds:
                    time.sleep(self.tiredness_check_interval)
                    continue
                if now - self.last_gestation_time >= self.max_awake_seconds:
                    reason = "max_awake_time"
                    self._trigger_gestation(now, tiredness, reason)
                    time.sleep(self.tiredness_check_interval)
                    continue
                awake_time = now - self.last_gestation_time
                if awake_time > 2 * self.min_awake_seconds:
                    self.current_tiredness_threshold = max(0.3, self.current_tiredness_threshold - 0.01)
                else:
                    self.current_tiredness_threshold = self.tiredness_threshold
                if tiredness > self.current_tiredness_threshold and mode == 'online':
                    reason = "tiredness"
                    self._trigger_gestation(now, tiredness, reason)
                time.sleep(self.tiredness_check_interval)
            except Exception as e:
                print(f"Tiredness monitor error: {e}")
                time.sleep(self.tiredness_check_interval)

    def _trigger_gestation(self, now, tiredness, reason):
        print(f"System is getting tired... preparing to sleep. Reason: {reason}")
        self.state_manager.set_mode('pending_gestation')
        if hasattr(self.state_manager, 'set_gestation_countdown'):
            self.state_manager.set_gestation_countdown(self.gestation_countdown_seconds)
        log_entry = {
            "event": "gestation_triggered",
            "reason": reason,
            "tiredness": tiredness,
            "threshold": self.current_tiredness_threshold,
            "timestamp": now,
            "mode": self.state_manager.get_mode(),
            "min_awake_seconds": self.min_awake_seconds,
            "max_awake_seconds": self.max_awake_seconds,
            "post_abort_cooldown_seconds": self.post_abort_cooldown_seconds
        }
        self._gestation_event_log.append(log_entry)
        if hasattr(self.logger, 'write'):
            self.logger.write(log_entry)
        self._last_gestation_reason = reason

    def on_gestation_complete(self):
        self.last_gestation_time = time.time()
        self.current_tiredness_threshold = self.tiredness_threshold
        self._pending_gestation_start = None
        self._gestating_start = None
        self._last_gestation_reason = None

    def on_gestation_abort(self):
        self.last_gestation_abort_time = time.time()
        self._pending_gestation_start = None
        self._gestating_start = None
        self.current_tiredness_threshold = min(1.0, self.current_tiredness_threshold + 0.05)

class SOVLSystem(SystemInterface):
    """Self-Organizing Virtual Lifeform system class that manages all components and state."""
    
    def __init__(
        self,
        context: SystemContext,
        model_manager: ModelManager,
        state_tracker: StateTracker,
        error_manager: ErrorManager,
        trainer: SOVLTrainer,
        generation_manager: 'GenerationManager',
        scriber: 'Scriber',
    ):
        """
        Initialize the SOVL system with pre-initialized components.
        Args:
            context: System context containing shared resources
            model_manager: Model manager component
            state_tracker: State tracking component
            error_manager: Error management component
            trainer: SOVLTrainer instance for gestation/dream cycles
            generation_manager: GenerationManager instance for text generation
            scriber: Scriber instance for event and metadata logging
        """
        try:
            if not context:
                raise ValueError("SystemContext is required")

            self.context = context
            self.config_handler = context.config_manager
            self.model_manager = model_manager
            self.state_tracker = state_tracker
            self.error_manager = error_manager
            self.trainer = trainer
            self.generation_manager = generation_manager
            self.scriber = scriber

            self._lock = RLock()
            self._monitoring_active = False
            self._stop_monitoring_event = Event()
            self._monitor_thread = None

            # Initialize grafter (plugin system)
            try:
                from sovl_grafter import initialize_plugin_manager
                self.grafter = initialize_plugin_manager(self)
                self.context.grafter = self.grafter
                self.generation_manager.primer.system = self  # Set system reference for plugin access
                self.logger.record({
                    "event": "grafter_initialized",
                    "plugin_count": len(self.grafter.plugins) if self.grafter else 0,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.grafter = None
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="grafter_init",
                        error_message=f"Failed to initialize grafter: {str(e)}",
                        error=e
                    )

            self.monitor_manager = MonitorManager(
                config_manager=context.config_manager,
                logger=context.logger,
                ram_manager=context.ram_manager,
                gpu_manager=context.gpu_manager,
                state_manager=context.state_manager,
                error_manager=self.error_manager,
                training_manager=getattr(context, 'training_cycle_manager', None)
            )

            # IntrospectionManager (for recursive introspection, meditation, etc)
            try:
                self.introspection_manager = IntrospectionManager(
                    context=context,
                    state_manager=context.state_manager,
                    error_manager=self.error_manager,
                    curiosity_manager=getattr(context, 'curiosity_manager', None),
                    confidence_calculator=getattr(context, 'confidence_calculator', None),
                    temperament_system=getattr(context, 'temperament_system', None),
                    model_manager=self.model_manager,
                    dialogue_context_manager=getattr(context, 'dialogue_context_manager', None),
                    bond_calculator=getattr(context, 'bond_calculator', None)
                )
            except Exception as e:
                self.introspection_manager = None
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="introspection_manager_init",
                        error_message=f"Failed to initialize IntrospectionManager: {str(e)}",
                        error=e
                    )

            # --- Autonomous Tiredness & Sleep Logic ---
            # --- Load gestation_config parameters from config ---
            gestation_cfg = self.config_handler.get('gestation_config', {}) if self.config_handler else {}
            self.tiredness_threshold = gestation_cfg.get('tiredness_threshold', 0.7)
            self.tiredness_check_interval = gestation_cfg.get('tiredness_check_interval', 10)
            self.tiredness_decay_k = gestation_cfg.get('tiredness_decay_k', 0.01)
            self.sleep_log_min = gestation_cfg.get('sleep_log_min', 10)
            self.gestation_countdown_seconds = gestation_cfg.get('gestation_countdown_seconds', 30)
            self.tiredness_weights = gestation_cfg.get('tiredness_weights', {"log": 0.4, "confidence": 0.3, "time": 0.3})
            self.min_awake_seconds = gestation_cfg.get('min_awake_seconds', 60)
            self.max_awake_seconds = gestation_cfg.get('max_awake_seconds', 7200)
            self.post_abort_cooldown_seconds = gestation_cfg.get('post_abort_cooldown_seconds', 120)
            # --- State for gestation logic ---
            self.last_gestation_time = time.time()  # Set to now on startup
            self.last_gestation_abort_time = 0.0
            self.current_tiredness_threshold = self.tiredness_threshold
            self._pending_gestation_start = None
            self._gestating_start = None
            self._last_gestation_reason = None
            self._gestation_event_log = []
            self._tiredness_monitor_thread = threading.Thread(target=self._tiredness_monitor_loop, daemon=True)
            self._tiredness_monitor_thread.start()
            
            self.tiredness_manager = TirednessManager(
                state_manager=context.state_manager,
                logger=context.logger,
                config_handler=self.config_handler
            )
            self.tiredness_manager.start()
            
            self.is_paused = False
            self._pause_lock = RLock()

            # Attach ScaffoldTokenMapper for CLI and system use
            try:
                self.scaffold_token_mapper = ScaffoldTokenMapper(
                    config_handler=self.config_handler,
                    logger=getattr(self, 'logger', None),
                    error_manager=self.error_manager
                )
                # Support for multiple scaffold token mappers (future-proof)
                self.scaffold_token_mappers = [self.scaffold_token_mapper]
            except Exception as e:
                self.scaffold_token_mapper = None
                self.scaffold_token_mappers = []
                if hasattr(self, 'error_manager') and self.error_manager:
                    self.error_manager.handle_error(
                        error_type="scaffold_token_mapper_init",
                        error_message=f"Failed to initialize ScaffoldTokenMapper: {str(e)}",
                        error=e
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
                safe_append_to_file("sovl_initialization_error.log", f"[{time.ctime()}] Critical error initializing SOVLSystem (error handler unavailable): {str(e)}\n")
                safe_append_to_file("sovl_initialization_error.log", traceback.format_exc() + "\n")
            raise

    def _initialize_component_state(self):
        """Initialize the state of all components."""
        try:
            component_states = {
                "config_handler": {
                    "status": "initialized",
                    "config_path": self.config_handler.config_path
                },
                "model_manager": {
                    "status": "initialized",
                    "active_model": self.model_manager.active_model_name if self.model_manager else None
                },
                "memory_monitor": {
                    "status": "initialized",
                    "memory_usage": self.monitor_manager.memory_monitor.check_memory_health() if self.monitor_manager else None
                },
                "system_monitor": {
                    "status": "initialized",
                    "metrics": self.monitor_manager.system_monitor._collect_metrics() if self.monitor_manager else None
                },
                "traits_monitor": {
                    "status": "initialized",
                    "traits": self.monitor_manager.traits_monitor._get_current_traits() if self.monitor_manager else None
                },
                "state_tracker": {
                    "status": "initialized",
                    "state_hash": self.state_tracker.state.state_hash if self.state_tracker.state else None
                },
                "error_manager": {
                    "status": "initialized",
                    "error_count": len(self.error_manager.get_error_stats()["error_counts"]) if self.error_manager else 0
                },
                "grafter": {
                    "status": "initialized" if self.grafter else "disabled",
                    "plugin_count": len(self.grafter.plugins) if self.grafter else 0,
                    "active_plugins": [p.get_metadata().name for p in self.grafter.plugins.values()] if self.grafter else []
                }
            }
            with self._lock:
                for component_name, state in component_states.items():
                    self.context.update_component_state(component_name, state)
                self.context.update_component_state("temperament_system", {"status": "initialized"})
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
                self.monitor_manager.memory_monitor.start_monitoring()
                self.context.ram_manager.enable_cleanup()
                self.context.gpu_manager.enable_cleanup()
            else:
                # Disable memory management
                self.monitor_manager.memory_monitor.stop_monitoring()
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
                status = {
                    "config_handler": bool(self.config_handler),
                    "model_manager": bool(self.model_manager),
                    "memory_monitor": bool(self.monitor_manager),
                    "state_tracker": bool(self.state_tracker),
                    "error_manager": bool(self.error_manager),
                    "system_monitor": bool(self.monitor_manager),
                    "traits_monitor": bool(self.monitor_manager),
                    "temperament_system": bool(self.temperament_system),
                    "grafter": bool(self.grafter)
                }
                # Add individual plugin status
                if self.grafter:
                    for plugin_name, plugin in self.grafter.plugins.items():
                        status[f"graft_{plugin_name}"] = plugin.validate()
                return status
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
                if self.monitor_manager:
                    self.monitor_manager.system_monitor.set_debug_mode(enabled)
                    self.monitor_manager.traits_monitor.set_debug_mode(enabled)
                    
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
        """Update the system state atomically using StateManager if available."""
        try:
            with self._lock:
                if not self.state_tracker:
                    raise ValueError("State tracker not initialized")
                if hasattr(self.context, 'state_manager') and self.context.state_manager:
                    def update_fn(state):
                        state.from_dict(state_dict, getattr(self.context, 'device', None))
                    self.context.state_manager.update_state_atomic(update_fn)
                else:
                    self.error_manager.handle_error(
                        error_type="state_update_fallback",
                        error_message="StateManager not available, falling back to direct update (not atomic)."
                    )
                    self.state_tracker.update_state(state_dict)
        except Exception as e:
            self.error_manager.handle_error(
                error_type="state_update",
                error_message=f"Failed to update system state: {str(e)}",
                error_context={
                    "state_dict": state_dict
                }
            )

    def run_gestation_and_dream_cycle(self, *args, **kwargs):
        """Run gestation, then (optionally) a fast, abortable dream cycle as a system state."""
        self.state_manager.set_mode('gestating')
        # --- Begin: Training and Pruning ---
        # Use canonical gestation entry point: TrainingWorkflowManager if available
        trained_memories = set()
        conversation_history = kwargs.get('conversation_history', [])
        if hasattr(self.trainer, 'training_workflow_manager') and hasattr(self.trainer.training_workflow_manager, 'run_gestation_cycle'):
            self.trainer.training_workflow_manager.run_gestation_cycle(conversation_history)
        else:
            scribe_path = getattr(self, 'scribe_path', 'scribe/sovl_scribe.jsonl')
            batch_size = getattr(self, 'batch_size', 32)
            epochs = getattr(self, 'train_epochs', 1)
            trained_memories = self.trainer.train_on_scribe_journal(scribe_path, batch_size=batch_size, epochs=epochs)
        if trained_memories:
            prune_scribe_journal(trained_memories, getattr(self, 'scribe_path', 'scribe/sovl_scribe.jsonl'), backup=True)
        # --- End: Training and Pruning ---
        # Check config if we should dream after gestation
        dream_after_gestation = getattr(self, 'dream_after_gestation', True)
        if hasattr(self, 'config_handler'):
            dream_after_gestation = self.config_handler.get('gestation_config.dream_after_gestation', True)
        if dream_after_gestation:
            self.run_dream_cycle_with_abort()
        else:
            self.state_manager.set_mode('online')

    def run_dream_cycle_with_abort(self):
        """Set mode to 'dreaming', run Dreamer, allow abort, then return to 'online'."""
        import time
        import select
        from threading import Thread
        self.state_manager.set_mode('dreaming')
        logger = getattr(self, 'logger', None)
        dreamer = getattr(self, 'dreamer', None)
        if dreamer is None:
            # Instantiate Dreamer if not present
            dreamer = Dreamer(
                self.config_handler,
                self.config_handler.get('scribe_path', 'scribe/sovl_scribe.jsonl'),
                getattr(self, 'scribe_event_fn', None),
                getattr(self, 'error_manager', None)
            )
            self.dreamer = dreamer
        dream_done = [False]
        abort_flag = [False]
        def dream_thread():
            try:
                dreamer.run_dream_cycle()
                dream_done[0] = True
            except Exception as e:
                if logger:
                    logger.log_error(f"Dreamer failed: {e}")
                dream_done[0] = True
        t = Thread(target=dream_thread)
        t.start()
        try:
            while not dream_done[0]:
                print("\rDreaming... (press any key to abort)", end="", flush=True)
                if sys.platform.startswith('win') or os.name == 'nt':
                    import msvcrt
                    time.sleep(0.1)
                    if msvcrt.kbhit():
                        msvcrt.getch()  # Clear the input buffer
                        print("\nAre you sure you want to abort dreaming? (y/N): ", end="", flush=True)
                        ans = msvcrt.getch().decode(errors='ignore').lower()
                        print(ans)  # Echo the key
                        if ans == 'y':
                            abort_flag[0] = True
                            if logger:
                                logger.info("Dreaming aborted by user.")
                            break
                        else:
                            print("Resuming dreaming...")
                else:
                    import select
                    dr, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if dr:
                        print("\nAre you sure you want to abort dreaming? (y/N): ", end="", flush=True)
                        ans = sys.stdin.readline().strip().lower()
                        if ans == 'y':
                            abort_flag[0] = True
                            if logger:
                                logger.info("Dreaming aborted by user.")
                            break
                        else:
                            print("Resuming dreaming...")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nAre you sure you want to abort dreaming? (y/N): ", end="", flush=True)
            ans = sys.stdin.readline().strip().lower()
            if ans == 'y':
                abort_flag[0] = True
                if logger:
                    logger.info("Dreaming aborted by user (KeyboardInterrupt).")
            else:
                print("Resuming dreaming...")
        finally:
            self.state_manager.set_mode('online')
            print("\nDreaming complete. Returning to online mode.")

    def run_meditation_cycle_with_abort(self, *args, **kwargs):
        """Set mode to 'meditating', run meditation/introspection, allow abort, then return to 'online'."""
        import time
        import sys
        import select
        from threading import Thread
        self.state_manager.set_mode('meditating')
        logger = getattr(self, 'logger', None)
        introspection_manager = getattr(self, 'introspection_manager', None)
        if introspection_manager is None:
            print("[ERROR] IntrospectionManager not available. Cannot meditate.")
            self.state_manager.set_mode('online')
            return
        meditation_done = [False]
        abort_flag = [False]
        meditation_result = [None]
        def meditation_thread():
            try:
                # This should be replaced with the actual meditation/introspection call
                # For now, we call the LLM-based select_and_execute
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                meditation_result[0] = loop.run_until_complete(introspection_manager._select_and_execute())
                meditation_done[0] = True
            except Exception as e:
                if logger:
                    logger.log_error(f"Meditation failed: {e}")
                meditation_done[0] = True
        t = Thread(target=meditation_thread)
        t.start()
        try:
            while not meditation_done[0]:
                print("\rMeditating... (press any key to abort)", end="", flush=True)
                if sys.platform.startswith('win') or os.name == 'nt':
                    import msvcrt
                    time.sleep(0.1)
                    if msvcrt.kbhit():
                        msvcrt.getch()  # Clear the input buffer
                        print("\nAre you sure you want to abort meditation? (y/N): ", end="", flush=True)
                        ans = msvcrt.getch().decode(errors='ignore').lower()
                        print(ans)  # Echo the key
                        if ans == 'y':
                            abort_flag[0] = True
                            if logger:
                                logger.info("Meditation aborted by user.")
                            break
                        else:
                            print("Resuming meditation...")
                else:
                    dr, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if dr:
                        print("\nAre you sure you want to abort meditation? (y/N): ", end="", flush=True)
                        ans = sys.stdin.readline().strip().lower()
                        if ans == 'y':
                            abort_flag[0] = True
                            if logger:
                                logger.info("Meditation aborted by user.")
                            break
                        else:
                            print("Resuming meditation...")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nAre you sure you want to abort meditation? (y/N): ", end="", flush=True)
            ans = sys.stdin.readline().strip().lower()
            if ans == 'y':
                abort_flag[0] = True
                if logger:
                    logger.info("Meditation aborted by user (KeyboardInterrupt).")
            else:
                print("Resuming meditation...")
        finally:
            self.state_manager.set_mode('online')
            self.state_manager.set_meditating_progress(0.0)
            print("\nMeditation complete. Returning to online mode.")
        # Optionally, handle abort logic (not implemented: need to make meditation cancellable)
        # Return result for further processing if needed
        return meditation_result[0]

    def on_gestation_complete(self):
        self.tiredness_manager.on_gestation_complete()

    def on_gestation_abort(self):
        self.tiredness_manager.on_gestation_abort()

    def pause(self) -> bool:
        """Pause the system's main operations. Returns True if paused, False if already paused."""
        with self._pause_lock:
            if not self.is_paused:
                self.is_paused = True
                return True
            return False

    def resume(self) -> bool:
        """Resume the system's main operations. Returns True if resumed, False if not paused."""
        with self._pause_lock:
            if self.is_paused:
                self.is_paused = False
                return True
            return False

    def generate_interactive_response(self, prompt_data: Dict[str, Any], user_id: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Generates an interactive response using the GenerationManager,
        after updating the temporal prompt via Chronos.
        'prompt_data' is passed to the generation_manager.
        'user_id' and 'conversation_history' are used by Chronos if provided.
        """
        if not self.context:
            # This should ideally not happen if system is initialized correctly
            print("[ERROR] SOVLSystem context is not available.") # Or use a logger if available early
            return None

        if self.is_paused:
            if self.context.logger:
                self.context.logger.warn("SOVLSystem is paused. Cannot generate response.")
            else:
                print("[WARN] SOVLSystem is paused. Cannot generate response.")
            return None

        # 1. Trigger Chronos to update temporal prompt in SOVLState
        if self.context.chronos_system:
            try:
                self.context.chronos_system.update_temporal_prompt(
                    user_id=user_id,
                    conversation_history=conversation_history
                )
                if self.context.logger:
                    self.context.logger.info("Chronos temporal prompt updated successfully.")
            except Exception as e:
                if self.context.error_manager:
                    self.context.error_manager.handle_error(
                        error_type="chronos_trigger_error",
                        error_message=f"Error triggering Chronos 'update_temporal_prompt': {str(e)}",
                        error=e
                    )
                else:
                    print(f"[ERROR] Error triggering Chronos: {str(e)}")
                # Depending on policy, we might proceed or abort. For now, log and proceed.
        elif self.context.logger:
            self.context.logger.info("Chronos system not available in context. Skipping temporal prompt update.")


        # 2. Call the GenerationManager
        if self.generation_manager:
            try:
                # Assuming generation_manager.generate uses prompt_data
                # and internally accesses SOVLState (via its Primer) for the Chronos prompt.
                response = self.generation_manager.generate(prompt_data=prompt_data)
                return response
            except Exception as e:
                if self.context.error_manager:
                    self.context.error_manager.handle_error(
                        error_type="generation_manager_error",
                        error_message=f"Error during GenerationManager 'generate': {str(e)}",
                        error_context={"prompt_data": "hidden for brevity" if prompt_data else None}, # Avoid logging large prompt_data
                        error=e
                    )
                else:
                    print(f"[ERROR] Error during generation: {str(e)}")
                return None
        else:
            if self.context.logger:
                self.context.logger.error("GenerationManager not available in SOVLSystem.")
            else:
                print("[ERROR] GenerationManager not available.")
            return None

    # Example: If you have a main loop or long-running operation, add a check for is_paused
    def main_loop(self):
        while not self.should_stop():
            with self._pause_lock:
                if self.is_paused:
                    # Sleep while paused, but allow for quick resume
                    time.sleep(0.1)
                    continue
            # ... rest of main loop logic ...

    def save_state(self, filename: str) -> None:
        """
        Save the current system state to disk using StateManager.
        Args:
            filename: The filename or path prefix for the state file.
        """
        path_prefix = filename.replace('.json', '')
        state_manager = getattr(self.context, 'state_manager', None)
        if not state_manager or not hasattr(self.state_tracker, 'state'):
            raise RuntimeError("StateManager or state not available for saving.")
        state_manager.save_state(self.state_tracker.state, path_prefix)

    def load_state(self, filename: str) -> None:
        """
        Load system state from disk using StateManager and update the current state.
        Args:
            filename: The filename or path prefix for the state file.
        """
        path_prefix = filename.replace('.json', '')
        state_manager = getattr(self.context, 'state_manager', None)
        if not state_manager:
            raise RuntimeError("StateManager not available for loading.")
        loaded_state = state_manager.load_state(path_prefix)
        if loaded_state:
            self.state_tracker.state = loaded_state
        else:
            raise RuntimeError(f"Failed to load state from {filename}")

    @property
    def state_tracker(self):
        """Return the state tracker instance."""
        return self._state_tracker if hasattr(self, '_state_tracker') else self.__dict__.get('state_tracker')

    @property
    def context(self):
        """Return the system context."""
        return self._context if hasattr(self, '_context') else self.__dict__.get('context')

    @property
    def config_handler(self):
        """Return the config handler (ConfigManager)."""
        return self._config_handler if hasattr(self, '_config_handler') else self.__dict__.get('config_handler')

    @property
    def scaffold_token_mapper(self):
        """Return the ScaffoldTokenMapper instance, or None if unavailable."""
        return self._scaffold_token_mapper if hasattr(self, '_scaffold_token_mapper') else self.__dict__.get('scaffold_token_mapper')

    @property
    def scaffold_token_mappers(self):
        """Return the list of ScaffoldTokenMapper instances (for multi-model support)."""
        return self._scaffold_token_mappers if hasattr(self, '_scaffold_token_mappers') else self.__dict__.get('scaffold_token_mappers', [])

    # Optionally, add more properties for other major attributes as needed
