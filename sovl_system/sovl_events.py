import asyncio
import re
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Generator, Union, cast
from sovl_logger import Logger, LoggerConfig
from sovl_config import ConfigManager
from sovl_state import StateManager, SOVLState
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_error import ErrorManager, ErrorRecord
import traceback

# Type alias for callbacks - clearer name
EventHandler = Callable[..., Any]

# Event type validation regex (alphanumeric, underscores, hyphens, dots)
EVENT_TYPE_PATTERN = re.compile(r'^[a-zA-Z0-9_\-\.]+$')

# State-related event types
class StateEventTypes:
    STATE_UPDATED = "state_updated"
    STATE_VALIDATED = "state_validated"
    STATE_RESET = "state_reset"
    STATE_SAVED = "state_saved"
    STATE_LOADED = "state_loaded"
    STATE_ERROR = "state_error"
    STATE_CACHE_UPDATED = "state_cache_updated"
    STATE_CACHE_CLEARED = "state_cache_cleared"
    STATE_MEMORY_UPDATED = "state_memory_updated"
    STATE_CONFIDENCE_UPDATED = "state_confidence_updated"

# Memory-related event types
class MemoryEventTypes:
    MEMORY_INITIALIZED = "memory_initialized"
    MEMORY_CONFIG_UPDATED = "memory_config_updated"
    MEMORY_THRESHOLD_REACHED = "memory_threshold_reached"
    MEMORY_CLEANUP_STARTED = "memory_cleanup_started"
    MEMORY_CLEANUP_COMPLETED = "memory_cleanup_completed"
    MEMORY_HEALTH_CHECK = "memory_health_check"
    MEMORY_STATS_UPDATED = "memory_stats_updated"
    TOKEN_MAP_UPDATED = "token_map_updated"
    SCAFFOLD_CONTEXT_UPDATED = "scaffold_context_updated"
    CONVERSATION_STARTED = "conversation_started"
    MEMORY_ERROR = "memory_error"

class StateEventDispatcher(EventDispatcher):
    """
    Extends EventDispatcher to handle state-related events and state management integration.
    """
    
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager, logger: Optional[Logger] = None):
        """
        Initialize the StateEventDispatcher.

        Args:
            config_manager: ConfigManager instance for configuration handling
            state_manager: StateManager instance for state management
            logger: Optional Logger instance. If None, creates a new Logger instance.
        """
        super().__init__(config_manager, logger)
        self.state_manager = state_manager
        self._state_change_history = deque(maxlen=100)
        self._state_cache = {}
        self._state_cache_lock = Lock()
        
        # Register state event handlers
        self._register_state_handlers()
        
    def _register_state_handlers(self) -> None:
        """Register default handlers for state events."""
        self.subscribe(StateEventTypes.STATE_UPDATED, self._handle_state_update, priority=10)
        self.subscribe(StateEventTypes.STATE_ERROR, self._handle_state_error, priority=10)
        self.subscribe(StateEventTypes.STATE_CACHE_UPDATED, self._handle_cache_update, priority=5)
        self.subscribe(StateEventTypes.STATE_CACHE_CLEARED, self._handle_cache_clear, priority=5)
        
    async def _handle_state_update(self, event_data: Dict[str, Any]) -> None:
        """Handle state update events with error management."""
        try:
            state = event_data.get('state')
            if not isinstance(state, SOVLState):
                raise ValueError("Invalid state object in event data")
                
            # Record state change
            self._state_change_history.append({
                'timestamp': time.time(),
                'event_type': StateEventTypes.STATE_UPDATED,
                'state_hash': state.state_hash(),
                'changes': event_data.get('changes', {})
            })
            
            # Update state through state manager
            self.state_manager.update_state(state)
            
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="state_event_error",
                context={
                    "event_type": StateEventTypes.STATE_UPDATED,
                    "event_data": event_data
                }
            )
            
    async def _handle_state_error(self, event_data: Dict[str, Any]) -> None:
        """Handle state error events with error management."""
        try:
            error_msg = event_data.get('error_msg', 'Unknown state error')
            error_type = event_data.get('error_type', 'state_error')
            
            self.error_manager.record_error(
                error=Exception(error_msg),
                error_type=error_type,
                context={
                    "event_type": StateEventTypes.STATE_ERROR,
                    "error_data": event_data
                }
            )
            
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="state_error_handling_error",
                context={"event_data": event_data}
            )
        
    async def _handle_cache_update(self, event_data: Dict[str, Any]) -> None:
        """Handle state cache update events."""
        with self._state_cache_lock:
            key = event_data.get('key')
            value = event_data.get('value')
            if key is not None:
                self._state_cache[key] = value
                
    async def _handle_cache_clear(self, event_data: Dict[str, Any]) -> None:
        """Handle state cache clear events."""
        with self._state_cache_lock:
            self._state_cache.clear()
            
    def get_state_change_history(self) -> List[Dict[str, Any]]:
        """Get recent state change history."""
        return list(self._state_change_history)
        
    async def validate_state_consistency(self) -> bool:
        """Validate that event history matches current state."""
        current_state = self.state_manager.get_state()
        last_change = self._state_change_history[-1] if self._state_change_history else None
        
        return (last_change and 
                last_change['state_hash'] == current_state.state_hash())
                
    async def dispatch_state_event(self, event_type: str, state_change: Dict[str, Any]) -> None:
        """
        Dispatch events related to state changes.
        
        Args:
            event_type: Type of state event
            state_change: Dictionary containing state change information
        """
        try:
            # Validate event type
            if not hasattr(StateEventTypes, event_type.upper()):
                raise ValueError(f"Invalid state event type: {event_type}")
                
            # Create event data
            event_data = {
                'type': event_type,
                'timestamp': time.time(),
                'state': self.state_manager.get_state(),
                'changes': state_change
            }
            
            # Dispatch event
            await self.async_notify(event_type, event_data)
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to dispatch state event: {str(e)}"),
                "state_event_dispatch",
                traceback.format_exc()
            )

class MemoryEventDispatcher(EventDispatcher):
    """Dispatches memory-related events to registered handlers."""
    
    def __init__(
        self,
        memoria_manager: MemoriaManager,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager,
        config_manager: ConfigManager,
        logger: Logger
    ):
        """
        Initialize the memory event dispatcher.
        
        Args:
            memoria_manager: MemoriaManager instance for core memory management
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
        """
        super().__init__(config_manager, logger)
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self._memory_events_history = deque(maxlen=100)
        
        # Register memory event handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default handlers for memory events."""
        self.subscribe(MemoryEventTypes.MEMORY_INITIALIZED, self._handle_memory_initialized, priority=10)
        self.subscribe(MemoryEventTypes.MEMORY_CONFIG_UPDATED, self._handle_config_update, priority=10)
        self.subscribe(MemoryEventTypes.MEMORY_THRESHOLD_REACHED, self._handle_memory_threshold, priority=20)
        self.subscribe(MemoryEventTypes.MEMORY_ERROR, self._handle_memory_error, priority=30)
        self.subscribe(MemoryEventTypes.TOKEN_MAP_UPDATED, self._handle_token_map_update, priority=15)
        self.subscribe(MemoryEventTypes.SCAFFOLD_CONTEXT_UPDATED, self._handle_scaffold_context_update, priority=15)
        self.subscribe(MemoryEventTypes.CONVERSATION_STARTED, self._handle_conversation_started, priority=15)

    async def _handle_memory_threshold(self, event: MemoryEvent) -> None:
        """Handle memory threshold events with error management."""
        try:
            # Record memory event
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.MEMORY_THRESHOLD_REACHED,
                'memory_type': event.memory_type,
                'threshold': event.threshold
            })
            
            # Handle memory threshold
            if event.memory_type == 'ram':
                await self.ram_manager.handle_threshold(event.threshold)
            elif event.memory_type == 'gpu':
                await self.gpu_manager.handle_threshold(event.threshold)
            else:
                raise ValueError(f"Unknown memory type: {event.memory_type}")
                
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="memory_event_error",
                context={
                    "event_type": MemoryEventTypes.MEMORY_THRESHOLD_REACHED,
                    "event": event.__dict__
                }
            )

    async def _handle_token_map_update(self, event_data: Dict[str, Any]) -> None:
        """Handle token map update events."""
        try:
            prompt = event_data.get('prompt')
            confidence = event_data.get('confidence')
            tokenizer = event_data.get('tokenizer')
            
            if not all([prompt, confidence, tokenizer]):
                raise ValueError("Missing required parameters for token map update")
                
            # Record update event
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.TOKEN_MAP_UPDATED,
                'prompt_length': len(prompt),
                'confidence': confidence
            })
            
            # Update token map
            self.memoria_manager.update_token_map_memory(prompt, confidence, tokenizer)
            
        except Exception as e:
            self.logger.error(f"Error handling token map update: {str(e)}", exc_info=True)

    async def _handle_scaffold_context_update(self, event_data: Dict[str, Any]) -> None:
        """Handle scaffold context update events."""
        try:
            scaffold_hidden_states = event_data.get('scaffold_hidden_states')
            
            if scaffold_hidden_states is None:
                raise ValueError("No scaffold hidden states provided")
                
            # Record update event
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.SCAFFOLD_CONTEXT_UPDATED,
                'tensor_shape': scaffold_hidden_states.shape
            })
            
            # Update scaffold context
            self.memoria_manager.set_scaffold_context(scaffold_hidden_states)
            
        except Exception as e:
            self.logger.error(f"Error handling scaffold context update: {str(e)}", exc_info=True)

    async def _handle_conversation_started(self, event_data: Dict[str, Any]) -> None:
        """Handle conversation started events."""
        try:
            conversation_id = event_data.get('conversation_id')
            
            if conversation_id is None:
                raise ValueError("No conversation ID provided")
                
            # Record conversation started event
            self._memory_events_history.append({
                'timestamp': time.time(),
                'event_type': MemoryEventTypes.CONVERSATION_STARTED,
                'conversation_id': conversation_id
            })
            
        except Exception as e:
            self.logger.error(f"Error handling conversation started event: {str(e)}", exc_info=True)

    def get_memory_events_history(self) -> List[Dict[str, Any]]:
        """Get recent memory events history."""
        return list(self._memory_events_history)

class EventDispatcher:
    """
    Manages event subscriptions and notifications in a thread-safe manner.

    This class provides a robust event handling system allowing components
    to subscribe to specific event types and receive notifications when those
    events occur.

    Features:
        - Thread-safe operations using locks.
        - Prioritized event handlers (higher priority executed first).
        - Synchronous (`notify`) and asynchronous (`async_notify`) notification.
        - Channel-based pub/sub pattern support.
        - Optional event metadata (timestamp, event_id).
        - Validation of event types and handlers.
        - Duplicate subscription detection and warning.
        - Deferred unsubscription: Prevents errors if handlers unsubscribe
          during a notification cycle.
        - Cleanup methods for stale events or all subscribers.
    """

    __slots__ = (
        '_subscribers',
        '_channels',
        '_lock',
        '_logger',
        '_notification_depth',
        '_deferred_unsubscriptions',
        '_config_manager',
        'error_manager',
        '_error_thresholds'
    )

    def __init__(self, config_manager: ConfigManager, logger: Optional[Logger] = None):
        """
        Initialize the EventDispatcher.

        Args:
            config_manager: ConfigManager instance for configuration handling
            logger: Optional Logger instance. If None, creates a new Logger instance.
        """
        self._config_manager = config_manager
        self._subscribers: Dict[str, List[Tuple[int, EventHandler]]] = defaultdict(list)
        self._channels: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._lock = Lock()
        self._logger = logger or Logger(LoggerConfig(log_file="sovl_events.log"))
        self._notification_depth: int = 0
        self._deferred_unsubscriptions: Dict[str, Set[EventHandler]] = defaultdict(set)
        
        # Initialize error management
        self._initialize_error_manager()

    def _initialize_error_manager(self) -> None:
        """Initialize error management system."""
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,  # Will be set by the system
            config_manager=self._config_manager
        )
        
        # Set error thresholds
        self._error_thresholds = {
            "event_validation_error": 3,
            "handler_execution_error": 5,
            "state_event_error": 3,
            "memory_event_error": 3,
            "channel_error": 2
        }
        
        # Register recovery strategies
        self._register_recovery_strategies()

    def _register_recovery_strategies(self) -> None:
        """Register error recovery strategies."""
        self.error_manager.register_recovery_strategy(
            "event_validation_error",
            self._recover_event_validation
        )
        self.error_manager.register_recovery_strategy(
            "handler_execution_error",
            self._recover_handler_execution
        )
        self.error_manager.register_recovery_strategy(
            "state_event_error",
            self._recover_state_event
        )
        self.error_manager.register_recovery_strategy(
            "memory_event_error",
            self._recover_memory_event
        )
        self.error_manager.register_recovery_strategy(
            "channel_error",
            self._recover_channel
        )

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        try:
            # Validate required configuration sections
            required_sections = ["logging_config", "controls_config"]
            for section in required_sections:
                if not self._config_manager.validate_section(section):
                    raise ValueError(f"Missing required configuration section: {section}")

            # Validate specific configuration values
            self._validate_config_values()

        except Exception as e:
            self._log_error(
                Exception(f"Configuration initialization failed: {str(e)}"),
                "config_initialization",
                traceback.format_exc()
            )
            raise

    def _validate_config_values(self) -> None:
        """Validate specific configuration values."""
        try:
            # Log file validation
            log_file = self._get_config_value("logging_config.log_file", "sovl_events.log")
            if not isinstance(log_file, str) or not log_file:
                raise ValueError(f"Invalid log_file: {log_file}")

            # Max size validation
            max_size_mb = self._get_config_value("logging_config.max_size_mb", 10)
            if not isinstance(max_size_mb, int) or max_size_mb < 1:
                raise ValueError(f"Invalid max_size_mb: {max_size_mb}")

            # Compress old validation
            compress_old = self._get_config_value("logging_config.compress_old", False)
            if not isinstance(compress_old, bool):
                raise ValueError(f"Invalid compress_old: {compress_old}")

        except Exception as e:
            self._log_error(
                Exception(f"Configuration validation failed: {str(e)}"),
                "config_validation",
                traceback.format_exc()
            )
            raise

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get a configuration value with validation."""
        try:
            return self._config_manager.get(key, default)
        except Exception as e:
            self._log_error(
                Exception(f"Failed to get config value for {key}: {str(e)}"),
                "config_access",
                traceback.format_exc()
            )
            return default

    def _update_config(self, key: str, value: Any) -> bool:
        """Update a configuration value with validation."""
        try:
            return self._config_manager.update(key, value)
        except Exception as e:
            self._log_error(
                Exception(f"Failed to update config value for {key}: {str(e)}"),
                "config_update",
                traceback.format_exc()
            )
            return False

    def _log_error(self, error: Exception, context: str, stack_trace: Optional[str] = None) -> None:
        """Log an error with context and stack trace."""
        self._logger.log_error(
            error_msg=str(error),
            error_type=f"events_{context}_error",
            stack_trace=stack_trace or traceback.format_exc(),
            additional_info={
                "context": context,
                "timestamp": time.time()
            }
        )

    @contextmanager
    def _locked(self):
        """Context manager for acquiring and releasing the internal lock."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def _validate_event_type(self, event_type: Any) -> str:
        """
        Validates the event type format and returns it if valid.

        Args:
            event_type: The event type to validate.

        Returns:
            The validated event type string.

        Raises:
            ValueError: If the event type is not a non-empty string or
                        does not match the required pattern.
        """
        if not isinstance(event_type, str) or not event_type:
            self._logger.log_error(
                error_msg="Event type must be a non-empty string",
                error_type="validation_error"
            )
            raise ValueError("Event type must be a non-empty string")
        if not EVENT_TYPE_PATTERN.match(event_type):
            self._logger.log_error(
                error_msg=f"Invalid event type format: '{event_type}'. Must match pattern [a-zA-Z0-9_.-]+",
                error_type="validation_error"
            )
            raise ValueError(f"Invalid event type format: '{event_type}'. Must match pattern [a-zA-Z0-9_.-]+")
        return event_type

    def _validate_handler(self, handler: Any) -> EventHandler:
        """
        Validates that the handler is callable.

        Args:
            handler: The event handler to validate.

        Returns:
            The validated event handler.

        Raises:
            TypeError: If the handler is not callable.
        """
        if not callable(handler):
            self._logger.log_error(
                error_msg=f"Invalid event handler: {type(handler).__name__} is not callable.",
                error_type="validation_error"
            )
            raise TypeError(f"Invalid event handler: {type(handler).__name__} is not callable.")
        return cast(EventHandler, handler)

    def subscribe(self, event_type: str, handler: EventHandler, priority: int = 0) -> None:
        """
        Subscribes an event handler to an event type with optional priority.

        Args:
            event_type: The type of event to subscribe to.
            handler: The function or method to call when the event occurs.
            priority: An integer representing the handler's priority.

        Raises:
            ValueError: If the event_type is invalid.
            TypeError: If the handler is not callable.
        """
        valid_event_type = self._validate_event_type(event_type)
        valid_handler = self._validate_handler(handler)
        handler_name = getattr(valid_handler, '__qualname__', repr(valid_handler))

        with self._locked():
            sub_list = self._subscribers[valid_event_type]

            if any(h == valid_handler for _, h in sub_list):
                self._logger.record_event(
                    event_type="duplicate_subscription",
                    message=f"Handler '{handler_name}' is already subscribed to event '{valid_event_type}'",
                    level="warning"
                )
                return

            sub_list.append((priority, valid_handler))
            sub_list.sort(key=lambda item: item[0], reverse=True)
            self._logger.record_event(
                event_type="subscription",
                message=f"Subscribed handler '{handler_name}' to event '{valid_event_type}' with priority {priority}",
                level="debug"
            )

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribes an event handler from an event type.

        Args:
            event_type: The type of event to unsubscribe from.
            handler: The specific handler function/method to remove.

        Raises:
            ValueError: If the event_type is invalid.
            TypeError: If the handler is not callable.
        """
        valid_event_type = self._validate_event_type(event_type)
        valid_handler = self._validate_handler(handler)
        handler_name = getattr(valid_handler, '__qualname__', repr(valid_handler))

        with self._locked():
            if self._notification_depth > 0 and valid_event_type in self._subscribers:
                self._deferred_unsubscriptions[valid_event_type].add(valid_handler)
                self._logger.record_event(
                    event_type="deferred_unsubscription",
                    message=f"Deferred unsubscription for handler '{handler_name}' from event '{valid_event_type}'",
                    level="debug"
                )
                return

            if valid_event_type in self._subscribers:
                original_count = len(self._subscribers[valid_event_type])
                self._subscribers[valid_event_type] = [
                    (prio, h) for prio, h in self._subscribers[valid_event_type] if h != valid_handler
                ]
                new_count = len(self._subscribers[valid_event_type])

                if new_count < original_count:
                    self._logger.record_event(
                        event_type="unsubscription",
                        message=f"Unsubscribed handler '{handler_name}' from event '{valid_event_type}'",
                        level="debug"
                    )
                else:
                    self._logger.record_event(
                        event_type="unsubscription_warning",
                        message=f"Attempted to unsubscribe handler '{handler_name}' from event '{valid_event_type}', but it was not found.",
                        level="warning"
                    )

                if not self._subscribers[valid_event_type]:
                    del self._subscribers[valid_event_type]
                    self._deferred_unsubscriptions.pop(valid_event_type, None)
            else:
                self._logger.record_event(
                    event_type="unsubscription_warning",
                    message=f"Attempted to unsubscribe from non-existent or empty event type '{valid_event_type}'.",
                    level="warning"
                )

    def notify(self, event_type: str, *args: Any, include_metadata: bool = False, **kwargs: Any) -> None:
        """
        Notifies all subscribed handlers of an event synchronously.

        Args:
            event_type: The type of event being triggered.
            *args: Positional arguments to pass to each handler.
            include_metadata: If True, includes metadata in the event.
            **kwargs: Keyword arguments to pass to each handler.

        Raises:
            ValueError: If the event_type is invalid.
        """
        subscribers_copy = self._prepare_notification(event_type)
        if not subscribers_copy:
            self._finalize_notification()
            return

        metadata = {}
        if include_metadata:
            now = time.time()
            metadata = {
                "event_id": f"{event_type}-{now:.6f}",
                "timestamp": now,
            }

        for _, handler in subscribers_copy:
            handler_name = getattr(handler, '__qualname__', repr(handler))
            try:
                if asyncio.iscoroutinefunction(handler):
                    self._logger.record_event(
                        event_type="async_handler_warning",
                        message=f"Attempted to call async handler '{handler_name}' for event '{event_type}' using synchronous notify().",
                        level="warning"
                    )
                    continue

                call_kwargs = kwargs.copy()
                if include_metadata:
                    call_kwargs['metadata'] = metadata

                handler(*args, **call_kwargs)

            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Error executing synchronous handler '{handler_name}' for event '{event_type}': {str(e)}",
                    error_type="handler_error",
                    stack_trace=str(e)
                )

        self._finalize_notification()

    async def async_notify(self, event_type: str, *args: Any, include_metadata: bool = False, **kwargs: Any) -> None:
        """
        Notifies all subscribed handlers of an event asynchronously.

        Args:
            event_type: The type of event being triggered.
            *args: Positional arguments to pass to each handler.
            include_metadata: If True, includes metadata in the event.
            **kwargs: Keyword arguments to pass to each handler.

        Raises:
            ValueError: If the event_type is invalid.
        """
        subscribers_copy = self._prepare_notification(event_type)
        if not subscribers_copy:
            self._finalize_notification()
            return

        metadata = {}
        if include_metadata:
            now = time.time()
            metadata = {
                "event_id": f"{event_type}-{now:.6f}",
                "timestamp": now,
            }

        for _, handler in subscribers_copy:
            handler_name = getattr(handler, '__qualname__', repr(handler))
            try:
                call_kwargs = kwargs.copy()
                if include_metadata:
                    call_kwargs['metadata'] = metadata

                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **call_kwargs)
                else:
                    handler(*args, **call_kwargs)

            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Error executing handler '{handler_name}' during async notification for event '{event_type}': {str(e)}",
                    error_type="handler_error",
                    stack_trace=str(e)
                )

        self._finalize_notification()

    def cleanup(self) -> None:
        """Clean up event dispatcher resources."""
        try:
            with self._locked():
                self._subscribers.clear()
                self._deferred_unsubscriptions.clear()
                self._logger.record_event(
                    event_type="cleanup",
                    message="Event dispatcher cleaned up successfully",
                    level="info"
                )
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Error during event dispatcher cleanup: {str(e)}",
                error_type="cleanup_error",
                stack_trace=str(e)
            )

    def _prepare_notification(self, event_type: str) -> List[Tuple[int, EventHandler]]:
        """Internal helper to prepare for notification."""
        valid_event_type = self._validate_event_type(event_type)
        with self._locked():
            self._notification_depth += 1
            # IMPORTANT: Create a *copy* of the subscriber list.
            # This allows releasing the lock while iterating and calling handlers,
            # preventing deadlocks if a handler tries to subscribe/unsubscribe.
            subscribers_copy = list(self._subscribers.get(valid_event_type, []))
        return subscribers_copy

    def _finalize_notification(self) -> None:
        """Internal helper to finalize notification and process deferred actions."""
        with self._locked():
            self._notification_depth -= 1
            if self._notification_depth == 0:
                # Process deferred unsubscriptions only when the outermost notification cycle ends
                if self._deferred_unsubscriptions:
                    self._process_deferred_unsubscriptions()

    def _process_deferred_unsubscriptions(self) -> None:
        """
        Processes handlers marked for deferred unsubscription.
        Must be called while holding the lock and when notification_depth is 0.
        """
        if not self._deferred_unsubscriptions:
            return

        self._logger.record_event(
            event_type="processing_deferred_unsubscriptions",
            message="Processing deferred unsubscriptions...",
            level="debug"
        )
        for event_type, handlers_to_remove in self._deferred_unsubscriptions.items():
            if event_type in self._subscribers:
                initial_len = len(self._subscribers[event_type])
                self._subscribers[event_type] = [
                    (prio, h) for prio, h in self._subscribers[event_type]
                    if h not in handlers_to_remove
                ]
                removed_count = initial_len - len(self._subscribers[event_type])
                if removed_count > 0:
                    self._logger.record_event(
                        event_type="processed_deferred_unsubscription",
                        message=f"Processed {removed_count} deferred unsubscription(s) for event '{event_type}'.",
                        level="debug"
                    )

                # Clean up if event type becomes empty
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    self._logger.record_event(
                        event_type="cleaned_up_stale_event_type_entry",
                        message=f"Cleaned up stale event type entry: '{event_type}'",
                        level="debug"
                    )

        self._deferred_unsubscriptions.clear()
        self._logger.record_event(
            event_type="finished_processing_deferred_unsubscriptions",
            message="Finished processing deferred unsubscriptions.",
            level="debug"
        )

    def publish(self, channel: str, event: Any) -> None:
        """
        Publish an event to a specific channel.

        Args:
            channel: The channel to publish the event to.
            event: The event data to publish.
        """
        valid_channel = self._validate_event_type(channel)
        with self._locked():
            self._channels[valid_channel].put_nowait(event)
            self._logger.record_event(
                event_type="publish",
                message=f"Published event to channel '{valid_channel}'",
                level="debug"
            )

    async def subscribe_channel(self, channel: str) -> Generator[Any, None, None]:
        """
        Subscribe to a specific channel and yield events.

        Args:
            channel: The channel to subscribe to.

        Yields:
            Events published to the channel.
        """
        valid_channel = self._validate_event_type(channel)
        while True:
            try:
                event = await self._channels[valid_channel].get()
                yield event
                self._channels[valid_channel].task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._log_error(
                    Exception(f"Error in channel subscription for '{valid_channel}': {str(e)}"),
                    "channel_subscription",
                    traceback.format_exc()
                )
                break

    def get_channel(self, channel: str) -> asyncio.Queue:
        """
        Get the queue for a specific channel.

        Args:
            channel: The channel to get the queue for.

        Returns:
            The queue for the specified channel.
        """
        valid_channel = self._validate_event_type(channel)
        return self._channels[valid_channel]

    def cleanup_channel(self, channel: str) -> None:
        """
        Clean up a specific channel.

        Args:
            channel: The channel to clean up.
        """
        valid_channel = self._validate_event_type(channel)
        with self._locked():
            if valid_channel in self._channels:
                del self._channels[valid_channel]
                self._logger.record_event(
                    event_type="channel_cleanup",
                    message=f"Cleaned up channel '{valid_channel}'",
                    level="debug"
                )

    def _recover_event_validation(self, record: ErrorRecord) -> None:
        """Recover from event validation errors."""
        try:
            # Log recovery attempt
            self._logger.record_event(
                "error_recovery",
                f"Attempting to recover from event validation error: {record.error_type}",
                "warning",
                {"error_context": record.context}
            )
            
            # Reset event validation state
            if "event_type" in record.context:
                event_type = record.context["event_type"]
                if event_type in self._subscribers:
                    # Revalidate handlers for this event type
                    self._validate_handlers(event_type)
            
            # Clear error count for this type
            self.error_manager.clear_error_count(record.error_type)
            
        except Exception as e:
            self._logger.record_event(
                "error_recovery_failed",
                f"Failed to recover from event validation error: {str(e)}",
                "error",
                {"original_error": str(record.error)}
            )

    def _recover_handler_execution(self, record: ErrorRecord) -> None:
        """Recover from handler execution errors."""
        try:
            # Log recovery attempt
            self._logger.record_event(
                "error_recovery",
                f"Attempting to recover from handler execution error: {record.error_type}",
                "warning",
                {"error_context": record.context}
            )
            
            # Remove problematic handler if identified
            if "handler" in record.context:
                handler = record.context["handler"]
                event_type = record.context.get("event_type")
                if event_type and handler:
                    self.unsubscribe(event_type, handler)
            
            # Clear error count for this type
            self.error_manager.clear_error_count(record.error_type)
            
        except Exception as e:
            self._logger.record_event(
                "error_recovery_failed",
                f"Failed to recover from handler execution error: {str(e)}",
                "error",
                {"original_error": str(record.error)}
            )

    def _recover_state_event(self, record: ErrorRecord) -> None:
        """Recover from state event errors."""
        try:
            # Log recovery attempt
            self._logger.record_event(
                "error_recovery",
                f"Attempting to recover from state event error: {record.error_type}",
                "warning",
                {"error_context": record.context}
            )
            
            # Reset state event handling
            if "state" in record.context:
                state = record.context["state"]
                if isinstance(state, SOVLState):
                    # Reinitialize state handling
                    self._initialize_state_handling()
            
            # Clear error count for this type
            self.error_manager.clear_error_count(record.error_type)
            
        except Exception as e:
            self._logger.record_event(
                "error_recovery_failed",
                f"Failed to recover from state event error: {str(e)}",
                "error",
                {"original_error": str(record.error)}
            )

    def _recover_memory_event(self, record: ErrorRecord) -> None:
        """Recover from memory event errors."""
        try:
            # Log recovery attempt
            self._logger.record_event(
                "error_recovery",
                f"Attempting to recover from memory event error: {record.error_type}",
                "warning",
                {"error_context": record.context}
            )
            
            # Reset memory event handling
            self._initialize_memory_handling()
            
            # Clear error count for this type
            self.error_manager.clear_error_count(record.error_type)
            
        except Exception as e:
            self._logger.record_event(
                "error_recovery_failed",
                f"Failed to recover from memory event error: {str(e)}",
                "error",
                {"original_error": str(record.error)}
            )

    def _recover_channel(self, record: ErrorRecord) -> None:
        """Recover from channel errors."""
        try:
            # Log recovery attempt
            self._logger.record_event(
                "error_recovery",
                f"Attempting to recover from channel error: {record.error_type}",
                "warning",
                {"error_context": record.context}
            )
            
            # Reset problematic channel
            if "channel" in record.context:
                channel = record.context["channel"]
                if channel in self._channels:
                    self.cleanup_channel(channel)
                    self._channels[channel] = asyncio.Queue()
            
            # Clear error count for this type
            self.error_manager.clear_error_count(record.error_type)
            
        except Exception as e:
            self._logger.record_event(
                "error_recovery_failed",
                f"Failed to recover from channel error: {str(e)}",
                "error",
                {"original_error": str(record.error)}
            )

class EventManager:
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = MemoriaManager(config_manager, logger)
        self.ram_manager = RAMManager(config_manager, logger)
        self.gpu_manager = GPUMemoryManager(config_manager, logger)
        
    def handle_memory_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle memory-related events."""
        try:
            # Check memory health before handling event
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Log event with memory health info
            self._logger.record_event(
                event_type=event_type,
                message=f"Memory event: {event_type}",
                level="info",
                additional_info={
                    "event_data": event_data,
                    "ram_health": ram_health,
                    "gpu_health": gpu_health
                }
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to handle memory event: {str(e)}",
                error_type="memory_event_error",
                stack_trace=traceback.format_exc()
            )
