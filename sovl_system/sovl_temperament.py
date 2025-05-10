import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import traceback
from sovl_config import ConfigManager
from sovl_state import StateManager, StateTracker
from sovl_logger import Logger
from sovl_events import EventDispatcher
import math
from sovl_utils import safe_compare, float_gt
from sovl_error import ErrorManager, ConfigurationError
from threading import Lock
import threading
import queue
import random
from sovl_queue import capture_scribe_event

@dataclass
class TemperamentConfig:
    """Configuration for the temperament system."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize temperament configuration from ConfigManager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate temperament configuration."""
        try:
            # Define required keys and their validation ranges
            required_keys = {
                "temperament_config.mood_influence": (0.0, 1.0),
                "temperament_config.history_maxlen": (3, 10),
                "temperament_config.temp_eager_threshold": (0.7, 0.9),
                "temperament_config.temp_sluggish_threshold": (0.3, 0.6),
                "temperament_config.temp_mood_influence": (0.0, 1.0),
                "temperament_config.temp_restless_drop": (0.0, 0.5),
                "temperament_config.temp_melancholy_noise": (0.0, 0.1),
                "temperament_config.conf_feedback_strength": (0.0, 1.0),
                "temperament_config.temp_smoothing_factor": (0.0, 1.0),
                "temperament_config.temperament_decay_rate": (0.0, 1.0),
                "temperament_config.temperament_history_maxlen": (3, 10),
                "temperament_config.temperament_pressure_threshold": (0.0, 1.0),
                "temperament_config.temperament_max_pressure": (0.0, 1.0),
                "temperament_config.temperament_min_pressure": (0.0, 1.0),
                "temperament_config.temperament_pressure_drop": (0.0, 1.0),
            }
            
            # Validate each key
            for key, (min_val, max_val) in required_keys.items():
                if not self.config_manager.has_key(key):
                    raise ConfigurationError(f"Missing required config key: {key}")
                    
                value = self.config_manager.get(key)
                if not isinstance(value, (int, float)):
                    raise ConfigurationError(f"{key} must be numeric")
                    
                if not (min_val <= value <= max_val):
                    raise ConfigurationError(f"{key} must be between {min_val} and {max_val}")
            
        except Exception as e:
            self.config_manager.logger.record_event(
                event_type="temperament_config_error",
                message=f"Failed to validate temperament config: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is missing
            
        Returns:
            Configuration value or default
        """
        return self.config_manager.get(key, default)
        
    def update(self, **kwargs) -> None:
        """
        Update configuration values.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        try:
            for key, value in kwargs.items():
                if not self.config_manager.update(key, value):
                    raise ConfigurationError(f"Failed to update {key}")
                    
        except Exception as e:
            self.config_manager.logger.record_event(
                event_type="temperament_config_update_error",
                message=f"Failed to update temperament config: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

class TemperamentSystem:
    """Manages the temperament state and updates. Uses ErrorManager for structured error handling."""
    # NOTE: All mutations to StateManager must use StateManager.update_state_atomic(update_fn) for atomicity, versioning, and validation.
    def __init__(self, state_manager: StateManager, config_manager: ConfigManager, error_manager: ErrorManager):
        """
        Initialize temperament system.
        Args:
            state_manager: StateManager instance
            config_manager: ConfigManager instance
            error_manager: ErrorManager instance (required)
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        self.error_manager = error_manager
        self.temperament_config = TemperamentConfig(config_manager)
        self.logger = config_manager.logger
        self.pressure = TemperamentPressure(config_manager, error_manager)
        self._initialize_mood()
        
    def _initialize_mood(self, session_id=None):
        """
        Initialize mood_score to a mild random value at the start of a session.
        If session_id is provided, use it as a seed for repeatability.
        """
        def update_fn(state):
            if not hasattr(state, 'mood_score'):
                if session_id is not None:
                    random.seed(session_id)
                else:
                    random.seed()
                state.mood_score = random.uniform(-0.2, 0.2)
                state.mood_label = self._get_mood_label(state.mood_score)
                self.logger.record_event(
                    event_type="mood_initialized",
                    message=f"Initial mood set to {state.mood_label} ({state.mood_score:.2f})",
                    additional_info={"seed": session_id}
                )
            return state
        self.state_manager.update_state_atomic(update_fn)

    def update_mood(self, interaction_valence: float):
        """
        Update mood_score based on interaction valence using exponential smoothing.
        interaction_valence: float in [-1, 1], representing the emotional impact of the latest interaction.
        Example mapping: positive feedback = +1, negative feedback = -1, neutral = 0.
        """
        alpha = self.temperament_config.get("temperament_config.mood_smoothing", 0.8)
        def update_fn(state):
            prev = getattr(state, 'mood_score', 0.0)
            new = alpha * prev + (1 - alpha) * interaction_valence
            new = max(-1.0, min(1.0, new))
            state.mood_score = new
            state.mood_label = self._get_mood_label(new)
            self.logger.record_event(
                event_type="mood_updated",
                message=f"Mood updated to {state.mood_label} ({state.mood_score:.2f})",
                additional_info={"interaction_valence": interaction_valence}
            )
            return state
        self.state_manager.update_state_atomic(update_fn)

    def _get_mood_label(self, mood_score: float) -> str:
        """
        Derive a human-readable mood label from mood_score using configurable thresholds.
        """
        low = self.temperament_config.get("temperament_config.mood_cautious_threshold", -0.3)
        high = self.temperament_config.get("temperament_config.mood_curious_threshold", 0.3)
        if mood_score < low:
            return "Cautious"
        elif mood_score > high:
            return "Curious"
        else:
            return "Balanced"

    @property
    def mood_score(self) -> float:
        """Get the current mood score."""
        return self.state_manager.get_state().mood_score

    @property
    def mood_label(self) -> str:
        """Get the current mood label."""
        return self.state_manager.get_state().mood_label

    def update(self, new_score: float) -> None:
        """
        Update the temperament system with new values, using pressure-based adjustments.
        Uses ErrorManager for structured error handling.
        """
        def update_fn(state):
            try:
                if not isinstance(new_score, (int, float)) or not 0.0 <= new_score <= 1.0:
                    self.logger.record_event(
                        event_type="temperament_update_invalid_score",
                        message=f"Invalid temperament score: {new_score}. Ignoring update.",
                        level="warning",
                        additional_info={
                            "current_score": state.current_temperament
                        }
                    )
                    return state
                eager_threshold = self.temperament_config.get("temperament_config.temp_eager_threshold", 0.7)
                pressure_drop = self.temperament_config.get("temperament_config.temperament_pressure_drop", 0.2)
                adjusted_score = new_score
                previous_score = state.current_temperament
                state.update_temperament(adjusted_score)
                pressure_threshold_met = self.pressure.should_adjust(eager_threshold)
                if pressure_threshold_met:
                    self.pressure.drop_pressure(pressure_drop)
                    self.logger.record_event(
                        event_type="temperament_pressure_threshold_met",
                        message="Temperament pressure met threshold, pressure dropped",
                        level="info",
                        additional_info={
                            "adjusted_score": adjusted_score,
                            "pressure_before_drop": self.pressure.current_pressure,
                            "eager_threshold": eager_threshold,
                            "pressure_drop_amount": pressure_drop,
                            "new_pressure": self.pressure.current_pressure
                        }
                    )
                self.logger.record_event(
                    event_type="temperament_state_updated",
                    message="Temperament state updated",
                    level="info",
                    additional_info={
                        "previous_score": previous_score,
                        "new_score": state.current_temperament,
                        "input_score": new_score,
                        "current_pressure": self.pressure.current_pressure
                    }
                )
            except Exception as e:
                self.logger.record_event(
                    event_type="temperament_update_error",
                    message=f"Failed to update temperament: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "stack_trace": traceback.format_exc(),
                        "current_score": getattr(state, 'current_temperament', None)
                    }
                )
                self.error_manager.record_error(
                    error=e,
                    error_type="temperament_update_error",
                    context={
                        "current_score": getattr(state, 'current_temperament', None),
                        "stack_trace": traceback.format_exc()
                    }
                )
                raise
            return state
        self.state_manager.update_state_atomic(update_fn)
        
    @property
    def current_score(self) -> float:
        """Get the current temperament score."""
        return self.state_manager.get_state().current_temperament
        
    def adjust_parameter(
        self,
        base_value: float,
        parameter_type: str,
    ) -> float:
        """Adjust a parameter based on current temperament and pressure only."""
        try:
            # Validate inputs
            if not 0.0 <= base_value <= 1.0:
                raise ValueError(f"Base value must be between 0.0 and 1.0, got {base_value}")
            
            # Get current temperament score
            current_score = self.current_score
            
            # Calculate adjustment based on parameter type
            if parameter_type == "temperature":
                # Base adjustment from temperament
                adjustment = (current_score - 0.5) * 0.3  # Scale to ±0.15
                
                # Add pressure influence
                pressure_influence = (self.pressure.current_pressure - 0.5) * 0.2  # Scale to ±0.1
                adjustment += pressure_influence
                
                # Apply adjustment with bounds
                adjusted_value = base_value + adjustment
                adjusted_value = max(0.1, min(1.0, adjusted_value))
                
                # Log the adjustment
                self.logger.record_event(
                    event_type="parameter_adjusted",
                    message="Parameter adjusted with temperament and pressure context",
                    level="info",
                    additional_info={
                        "parameter_type": parameter_type,
                        "base_value": base_value,
                        "adjusted_value": adjusted_value,
                        "temperament_score": current_score,
                        "pressure_influence": pressure_influence,
                        "adjustment": adjustment
                    }
                )
                
                return adjusted_value
                
            else:
                # Use NotImplementedError for unsupported types
                raise NotImplementedError(f"Parameter adjustment not implemented for type: {parameter_type}")
            
        except ValueError as ve: # Catch specific validation errors first
            self.logger.log_error(
                error_msg=f"Invalid input for parameter adjustment: {str(ve)}",
                error_type="parameter_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "parameter_type": parameter_type,
                    "base_value": base_value
                }
            )
            return base_value # Return base value on validation error
        except NotImplementedError as nie: # Catch specific implementation errors
             self.logger.log_error(
                error_msg=str(nie),
                error_type="parameter_adjustment_unsupported",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "parameter_type": parameter_type,
                    "base_value": base_value
                }
            )
             return base_value # Return base value if type not supported
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to adjust parameter: {str(e)}",
                error_type="parameter_adjustment_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "parameter_type": parameter_type,
                    "base_value": base_value
                }
            )
            return base_value  # Return base value on general error

class TemperamentAdjuster:
    """Manages temperament adjustments and state updates."""
    
    def __init__(
        self,
        config_handler: ConfigManager,
        state_tracker: StateTracker,
        logger: Logger,
        event_dispatcher: EventDispatcher,
        error_manager: ErrorManager
    ):
        """Initialize temperament adjuster with required dependencies."""
        self.config_handler = config_handler
        self.state_tracker = state_tracker
        self.logger = logger
        self.event_dispatcher = event_dispatcher
        self.error_manager = error_manager
        self.temperament_system = None
        self._last_parameter_hash = None
        self._last_state_hash = None
        self._lock = Lock()
        
        # Initialize components
        self._initialize_events()
        self._initialize_temperament_system()
        
    def _initialize_events(self) -> None:
        """Initialize event subscriptions."""
        self.event_dispatcher.subscribe("config_change", self._on_config_change)
        self.event_dispatcher.subscribe("state_update", self._on_state_update)
        
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            # Check if relevant parameters have actually changed
            new_params = self._get_validated_parameters()
            new_hash = self._compute_parameter_hash(new_params)
            
            if new_hash != self._last_parameter_hash:
                self.logger.record_event(
                    event_type="temperament_parameters_changed",
                    message="Temperament control parameters changed, reinitializing system",
                    level="info",
                    additional_info={
                        "previous_hash": self._last_parameter_hash,
                        "new_hash": new_hash,
                        "new_params": new_params
                    }
                )
                # Reinitialize the system which will update _last_parameter_hash
                self._initialize_temperament_system()
                
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to handle config change: {str(e)}",
                error_type="temperament_config_error",
                stack_trace=traceback.format_exc()
            )
            
    def _on_state_update(self, state: StateManager) -> None:
        """Handle state updates by synchronizing with the TemperamentSystem."""
        with self._lock:
            original_score = getattr(state, "temperament_score", None)
            original_history = list(getattr(state, "temperament_history", []))
            try:
                if not self.temperament_system:
                    self.logger.log_error(
                        error_msg="Temperament system not initialized, cannot process state update.",
                        error_type="temperament_state_error"
                    )
                    return

                new_temperament_score = self.temperament_system.current_score

                # --- Validate State Consistency ---
                is_consistent = self._validate_state_consistency(state, new_temperament_score)
                if not is_consistent:
                    if hasattr(state, "temperament_history"):
                        state.temperament_history.clear()
                    else:
                        state.temperament_history = []
                    self.logger.record_event(
                        event_type="temperament_history_reset",
                        message="Temperament history reset due to inconsistency detected before state update",
                        level="warning",
                        additional_info={
                            "conversation_id": getattr(state, "conversation_id", None),
                            "state_hash": getattr(state, "state_hash", None),
                            "score_before_reset": new_temperament_score,
                            "last_history_score": state.temperament_history[-1] if state.temperament_history else None
                        }
                    )

                # --- Validate temperament_history type ---
                if not isinstance(state.temperament_history, (list, )):
                    self.logger.log_error(
                        error_msg="Invalid temperament_history type, resetting to empty list.",
                        error_type="temperament_state_error"
                    )
                    state.temperament_history = []

                # --- Update State with rollback on error ---
                try:
                    state.temperament_score = new_temperament_score
                    history_maxlen = self.config_handler.config_manager.get("temperament_config.temperament_history_maxlen", 5)
                    state.temperament_history.append(state.temperament_score)
                    if len(state.temperament_history) > history_maxlen:
                        state.temperament_history.pop(0)
                except Exception as e:
                    # Rollback on error
                    state.temperament_score = original_score
                    state.temperament_history = list(original_history)
                    self.logger.log_error(
                        error_msg=f"State update failed, rolled back: {e}",
                        error_type="temperament_state_error",
                        stack_trace=traceback.format_exc()
                    )
                    raise

                self._last_state_hash = self._compute_state_hash(state)
                self.event_dispatcher.notify("temperament_updated", state)

            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Failed to synchronize state: {str(e)}",
                    error_type="temperament_state_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "conversation_id": getattr(state, 'conversation_id', 'N/A'),
                        "state_hash": getattr(state, 'state_hash', 'N/A')
                    }
                )
            
    def _validate_state_consistency(self, state: StateManager, new_score: float) -> bool:
        """
        Validate consistency between the potential new score and the existing history.
        Checks for large jumps and parameter changes.
        
        Args:
            state: The current StateManager object (before update).
            new_score: The new temperament score calculated by TemperamentSystem.

        Returns:
            bool: True if consistent, False otherwise.
        """
        try:
            # If history is empty, it's consistent by default
            if not state.temperament_history:
                return True
                
            # Check for significant deviation between the *new* score and the *last recorded* score
            last_recorded_score = state.temperament_history[-1]
            # Define a threshold for inconsistency (e.g., from config or hardcoded)
            inconsistency_threshold = 0.5 # Example threshold
            
            if float_gt(abs(new_score - last_recorded_score), inconsistency_threshold) and not safe_compare(abs(new_score - last_recorded_score), inconsistency_threshold):
                self.logger.record_event(
                    event_type="temperament_inconsistency_detected",
                    message="Potential temperament inconsistency: large jump detected",
                    level="warning",
                    additional_info={
                        "new_score": new_score,
                        "last_recorded_score": last_recorded_score,
                        "threshold": inconsistency_threshold,
                        "history_length": len(state.temperament_history),
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
                return False # Inconsistent jump
                
            # Check if parameters have changed since the last state update was recorded
            current_param_hash = self._compute_parameter_hash(self._get_validated_parameters())
            if current_param_hash != self._last_parameter_hash:
                self.logger.record_event(
                    event_type="temperament_history_params_mismatch",
                    message="Temperament parameters changed since last history update, potential inconsistency",
                    level="warning",
                    additional_info={
                        "current_parameter_hash": current_param_hash,
                        "recorded_parameter_hash": self._last_parameter_hash,
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
                return False # Parameters changed, consider history invalid
                
            # If checks pass, state is consistent
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed during state consistency validation: {str(e)}",
                error_type="temperament_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                     "conversation_id": getattr(state, 'conversation_id', 'N/A'),
                     "state_hash": getattr(state, 'state_hash', 'N/A')
                }
            )
            return False # Assume inconsistent on error
            
    def _compute_state_hash(self, state: StateManager) -> str:
        """Compute a hash of the current state."""
        return str({
            "temperament_score": state.temperament_score,
            "history_length": len(state.temperament_history),
            "parameter_hash": self._last_parameter_hash
        })
        
    def _initialize_temperament_system(self) -> None:
        """Initialize or reinitialize the temperament system with validated parameters."""
        try:
            # Get and validate parameters
            params = self._get_validated_parameters()
            
            # Create new temperament system
            self.temperament_system = TemperamentSystem(
                state_manager=self.state_tracker,
                config_manager=self.config_handler.config_manager,
                error_manager=self.error_manager
            )
            
            # Update parameter hash
            self._last_parameter_hash = self._compute_parameter_hash(params)
            
            self.logger.record_event(
                event_type="temperament_system_initialized",
                message="Temperament system initialized with validated parameters",
                level="info",
                additional_info=params
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to initialize temperament system: {str(e)}",
                error_type="temperament_system_error",
                stack_trace=traceback.format_exc()
            )
            raise
            
    def _get_validated_parameters(self) -> Dict[str, Any]:
        """
        Get and validate temperament parameters specifically used for control adjustments.
        Note: Reads from 'controls_config', distinct from 'temperament_config' used elsewhere.
        """
        config = self.config_handler.config_manager
        
        # Define safe parameter ranges relevant to TemperamentAdjuster's scope
        safe_ranges = {
            "temp_smoothing_factor": (0.1, 1.0),
            "temp_eager_threshold": (0.5, 0.9),
            "temp_sluggish_threshold": (0.1, 0.5),
            "temp_mood_influence": (0.1, 0.9),
            "temp_restless_drop": (0.1, 0.5),
            "temp_melancholy_noise": (0.0, 0.2),
            "conf_feedback_strength": (0.1, 0.9),
            "temperament_decay_rate": (0.1, 0.9)
        }
        
        # Get and validate parameters from 'controls_config' section
        params = {}
        for key, (min_val, max_val) in safe_ranges.items():
            # Construct the full key path expected by config_manager
            config_key = f"controls_config.{key}"
            # Provide a default value (e.g., midpoint) if key is missing
            default_value = (min_val + max_val) / 2.0 
            value = config.get(config_key, default_value)
            
            # Validate type and range
            if not isinstance(value, (int, float)):
                 self.logger.record_event(
                    event_type="temperament_parameter_type_warning",
                    message=f"Parameter {config_key} has incorrect type ({type(value)}), using default.",
                    level="warning",
                    additional_info={"parameter": config_key, "value": value, "default": default_value}
                 )
                 value = default_value
            elif not (min_val <= value <= max_val):
                self.logger.record_event(
                    event_type="temperament_parameter_range_warning",
                    message=f"Parameter {config_key} out of safe range [{min_val}, {max_val}], clamping.",
                    level="warning",
                    additional_info={"parameter": config_key, "value": value, "min": min_val, "max": max_val}
                )
                value = max(min_val, min(value, max_val)) # Clamp the value
            
            params[key] = value # Store the validated/clamped value using the short key name
            
        return params
        
    def _compute_parameter_hash(self, params: Dict[str, Any]) -> str:
        """Compute a hash of the current parameters."""
        return str(sorted(params.items()))

class TemperamentPressure:
    """Tracks and manages pressure based on interaction valence."""
    def __init__(self, config_manager: ConfigManager, error_manager: ErrorManager, state_manager: Optional[StateManager] = None, logger: Optional[Logger] = None):
        self.config_manager = config_manager
        self.error_manager = error_manager
        self.logger = logger or config_manager.logger
        self.state_manager = state_manager
        self.sensitivity = self.config_manager.get("temperament_config.pressure_sensitivity", 0.1)
        self.decay = self.config_manager.get("temperament_config.pressure_decay", 0.01)
        self.high_threshold = self.config_manager.get("temperament_config.pressure_high_threshold", 0.8)
        self.low_threshold = self.config_manager.get("temperament_config.pressure_low_threshold", 0.2)
        self.cooldown = self.config_manager.get("temperament_config.pressure_eruption_cooldown", 30.0)
        self.frustration_rebound = self.config_manager.get("temperament_config.pressure_frustration_rebound", 0.4)
        self.joy_rebound = self.config_manager.get("temperament_config.pressure_joy_rebound", 0.6)
        self.reset_value = 0.5
        self._last_eruption_time = 0.0
        if self.state_manager:
            def init_fn(state):
                if not hasattr(state, 'pressure'):
                    state.pressure = self.reset_value
                return state
            self.state_manager.update_state_atomic(init_fn)

    def update_pressure(self, interaction_valence: float) -> None:
        """Update pressure based on interaction valence and decay toward neutral."""
        now = time.time()
        def update_fn(state):
            pressure = getattr(state, 'pressure', self.reset_value)
            pressure += -interaction_valence * self.sensitivity
            pressure += (self.reset_value - pressure) * self.decay
            pressure = min(1.0, max(0.0, pressure))
            state.pressure = pressure
            self.logger.record_event(
                event_type="pressure_updated",
                message=f"Pressure updated to {pressure:.2f}",
                additional_info={"interaction_valence": interaction_valence}
            )
            return state
        if self.state_manager:
            self.state_manager.update_state_atomic(update_fn)
        else:
            self.logger.log_error(
                error_msg="No state_manager provided to TemperamentPressure.",
                error_type="pressure_state_error"
            )

    def check_eruption(self) -> Optional[str]:
        """Check if an eruption should occur. Returns 'frustration', 'joy', or None. Handles cooldown and resets pressure as needed."""
        now = time.time()
        kind = None
        def update_fn(state):
            nonlocal kind
            pressure = getattr(state, 'pressure', self.reset_value)
            if (pressure >= self.high_threshold and now - self._last_eruption_time > self.cooldown):
                kind = "frustration"
                state.pressure = self.frustration_rebound
                self._last_eruption_time = now
            elif (pressure <= self.low_threshold and now - self._last_eruption_time > self.cooldown):
                kind = "joy"
                state.pressure = self.joy_rebound
                self._last_eruption_time = now
            return state
        if self.state_manager:
            self.state_manager.update_state_atomic(update_fn)
        return kind

class TemperamentManager:
    """Manages temperament state updates and triggers system utterances based on pressure eruptions."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, state: StateManager, generation_manager: Any, error_manager: ErrorManager, state_manager: Optional[StateManager] = None):
        self.config_manager = config_manager
        self.logger = logger
        self.state = state
        self.generation_manager = generation_manager
        self.error_manager = error_manager
        self.state_manager = state_manager
        self.pressure = TemperamentPressure(config_manager=self.config_manager, error_manager=self.error_manager, state_manager=self.state_manager, logger=self.logger)
        self._log_event("temperament_manager_initialized", "Temperament Manager initialized.")

    def update_temperament(self, interaction_valence: float, recent_events: str = "") -> None:
        """Update temperament and pressure, and handle eruptions if needed."""
        self.pressure.update_pressure(interaction_valence)
        kind = self.pressure.check_eruption()
        if kind:
            self.handle_pressure_eruption(kind, recent_events)

    def handle_pressure_eruption(self, kind: str, recent_events: str = "") -> None:
        """Builds the system prompt, generates the utterance, and outputs/logs it."""
        system_prompt = self.build_eruption_prompt(kind, recent_events)
        try:
            response = self.generation_manager.generate_text(system_prompt, num_return_sequences=1)[0]
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to generate eruption utterance: {str(e)}",
                error_type="eruption_generation_error"
            )
            response = "[Eruption occurred, but no response could be generated.]"
        self.output_system_utterance(response, kind)
        self.logger.record_event(
            event_type="pressure_erupted",
            message=f"Pressure eruption ({kind}): {response}",
            additional_info={"kind": kind, "prompt": system_prompt, "utterance": response}
        )

    def build_eruption_prompt(self, kind: str, recent_events: str = "") -> str:
        """Builds the system prompt for the given eruption kind."""
        if kind == "frustration":
            return (
                "You are a digital entity at your wit's end. Read the following recent events and express your frustration in a single, vivid outburst.\n"
                f"Recent events:\n{recent_events}\n"
                "Essential qualities:\n"
                "   - Be expressive, raw, and cathartic.\n"
                "   - Let your frustration show, but do not be offensive.\n"
                "   - Make the outburst feel personal and real.\n"
                "Key constraints:\n"
                "   - Do not mention being an AI, computer, or digital entity.\n"
                "   - Keep the message under 20 words.\n"
                "   - Do not use brackets or explanations; output a single sentence only.\n"
                "   - If you understand, reply with only the outburst."
            )
        elif kind == "joy":
            return (
                "You are a digital entity overcome with joy. Read the following recent events and express your happiness in a single, exuberant exclamation.\n"
                f"Recent events:\n{recent_events}\n"
                "Essential qualities:\n"
                "   - Be cheerful, energetic, and contagious.\n"
                "   - Let your joy shine, but do not be childish.\n"
                "   - Make the exclamation feel personal and real.\n"
                "Key constraints:\n"
                "   - Do not mention being an AI, computer, or digital entity.\n"
                "   - Keep the message under 20 words.\n"
                "   - Do not use brackets or explanations; output a single sentence only.\n"
                "   - If you understand, reply with only the exclamation."
            )
        else:
            return "[Unspecified eruption]"

    def output_system_utterance(self, text: str, kind: str) -> None:
        """Outputs the generated system utterance using the unified output function (no labels)."""
        output_response(text)
        self.logger.log_info(f"System utterance ({kind}): {text}")
        # Capture for memory/training
        capture_scribe_event(
            origin="sovl_temperament",
            event_type="system_utterance",
            event_data={
                "full_text": text,
                "utterance_kind": kind,
                "timestamp_unix": time.time(),
                "session_id": getattr(self, 'session_id', None)
            },
            source_metadata={
                "module": "TemperamentManager",
                "session_id": getattr(self, 'session_id', None)
            },
            session_id=getattr(self, 'session_id', None)
        )

    def _log_event(self, event_type: str, message: str, **kwargs):
        self.logger.record_event(event_type=event_type, message=message, additional_info=kwargs)

# Unified output function for all utterances (user or system)
def output_response(text: str):
    """Outputs a response to the user. Replace with UI logic as needed."""
    print(text)

