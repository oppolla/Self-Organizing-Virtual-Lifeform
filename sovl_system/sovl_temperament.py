import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import traceback
from sovl_config import ConfigManager
from sovl_state import SOVLState, StateManager, StateTracker
from sovl_logger import Logger
from sovl_events import EventDispatcher
from sovl_trainer import TrainingCycleManager
from sovl_confidence import ConfidenceCalculator
import math
from sovl_utils import synchronized, safe_divide
from sovl_error import ErrorManager, ConfigurationError
from threading import Lock

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
                "temperament_config.temp_curiosity_boost": (0.0, 0.5),
                "temperament_config.temp_restless_drop": (0.0, 0.5),
                "temperament_config.temp_melancholy_noise": (0.0, 0.1),
                "temperament_config.conf_feedback_strength": (0.0, 1.0),
                "temperament_config.temp_smoothing_factor": (0.0, 1.0),
                "temperament_config.temperament_decay_rate": (0.0, 1.0),
                "temperament_config.temperament_history_maxlen": (3, 10),
                "temperament_config.confidence_history_maxlen": (3, 10),
                "temperament_config.temperament_pressure_threshold": (0.0, 1.0),
                "temperament_config.temperament_max_pressure": (0.0, 1.0),
                "temperament_config.temperament_min_pressure": (0.0, 1.0),
                "temperament_config.temperament_confidence_adjustment": (0.0, 1.0),
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
            
            # Validate lifecycle parameters if present
            if self.config_manager.has_key("temperament_config.lifecycle_params"):
                lifecycle_params = self.config_manager.get("temperament_config.lifecycle_params")
                if not isinstance(lifecycle_params, dict):
                    raise ConfigurationError("lifecycle_params must be a dictionary")
                    
                for stage, params in lifecycle_params.items():
                    if not isinstance(params, dict) or "bias" not in params or "decay" not in params:
                        raise ConfigurationError(f"lifecycle_params for {stage} must contain 'bias' and 'decay'")
            
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
    """Manages the temperament state and updates."""
    # NOTE: All mutations to SOVLState must use StateManager.update_state_atomic(update_fn) for atomicity, versioning, and validation.
    def __init__(self, state_manager: StateManager, config_manager: ConfigManager, lifecycle_manager: Optional[Any] = None):
        """
        Initialize temperament system.
        """
        self.state_manager = state_manager
        self.config_manager = config_manager
        self.temperament_config = TemperamentConfig(config_manager)
        self.logger = config_manager.logger
        self.lifecycle_manager = lifecycle_manager
        self._lifecycle_stage = "initialization"
        self._last_lifecycle_update = time.time()
        self.pressure = TemperamentPressure(config_manager)
        if self.lifecycle_manager:
            self.logger.record_event(
                event_type="temperament_lifecycle_integration_initialized",
                message="Temperament system initialized with lifecycle manager",
                level="info",
                additional_info={
                    "lifecycle_stage": self._lifecycle_stage,
                    "lifecycle_manager": str(self.lifecycle_manager)
                }
            )
        
    def update(self, new_score: float, confidence: float, lifecycle_stage: Optional[str] = None) -> None:
        """
        Update the temperament system with new values, using pressure-based adjustments.
        """
        def update_fn(state):
            try:
                if not isinstance(new_score, (int, float)) or not 0.0 <= new_score <= 1.0:
                    self.logger.record_event(
                        event_type="temperament_update_invalid_score",
                        message=f"Invalid temperament score: {new_score}. Ignoring update.",
                        level="warning",
                        additional_info={
                            "lifecycle_stage": lifecycle_stage,
                            "current_score": state.current_temperament
                        }
                    )
                    return state
                if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                    self.logger.record_event(
                        event_type="temperament_update_invalid_confidence",
                        message=f"Invalid confidence: {confidence}. Ignoring update.",
                        level="warning",
                        additional_info={
                            "lifecycle_stage": lifecycle_stage,
                            "current_score": state.current_temperament
                        }
                    )
                    return state
                eager_threshold = self.temperament_config.get("temperament_config.temp_eager_threshold", 0.7)
                pressure_drop = self.temperament_config.get("temperament_config.temperament_pressure_drop", 0.2)
                if lifecycle_stage is None and self.lifecycle_manager:
                    lifecycle_stage_local = self.lifecycle_manager.get_lifecycle_stage()
                else:
                    lifecycle_stage_local = lifecycle_stage
                lifecycle_params = self.temperament_config.get("temperament_config.lifecycle_params", {})
                if lifecycle_stage_local in lifecycle_params:
                    stage_params = lifecycle_params[lifecycle_stage_local]
                    bias = stage_params.get("bias", 0.0)
                    decay = stage_params.get("decay", 1.0)
                    time_since_update = time.time() - self._last_lifecycle_update
                    decay_factor = math.exp(-decay * time_since_update)
                    adjusted_score = (new_score + bias) * decay_factor
                    adjusted_score = max(0.0, min(1.0, adjusted_score))
                    self.logger.record_event(
                        event_type="temperament_lifecycle_adjustment",
                        message="Applied lifecycle-based temperament adjustments",
                        level="info",
                        additional_info={
                            "lifecycle_stage": lifecycle_stage_local,
                            "bias": bias,
                            "decay": decay,
                            "decay_factor": decay_factor,
                            "adjusted_score": adjusted_score
                        }
                    )
                else:
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
                            "confidence": confidence,
                            "pressure_before_drop": self.pressure.current_pressure,
                            "eager_threshold": eager_threshold,
                            "pressure_drop_amount": pressure_drop,
                            "new_pressure": self.pressure.current_pressure,
                            "lifecycle_stage": lifecycle_stage_local
                        }
                    )
                self._lifecycle_stage = lifecycle_stage_local
                self._last_lifecycle_update = time.time()
                self.logger.record_event(
                    event_type="temperament_state_updated",
                    message="Temperament state updated",
                    level="info",
                    additional_info={
                        "previous_score": previous_score,
                        "new_score": state.current_temperament,
                        "input_score": new_score,
                        "confidence": confidence,
                        "current_pressure": self.pressure.current_pressure,
                        "lifecycle_stage": lifecycle_stage_local,
                        "conversation_id": getattr(state, 'conversation_id', None),
                        "state_hash": getattr(state, 'state_hash', None),
                        "time_since_last_lifecycle_update": time.time() - self._last_lifecycle_update,
                        "lifecycle_params": lifecycle_params.get(lifecycle_stage_local, {})
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
                        "lifecycle_stage": lifecycle_stage,
                        "current_score": getattr(state, 'current_temperament', None)
                    }
                )
                raise
            return state
        self.state_manager.update_state_atomic(update_fn)
        
    @property
    def current_score(self) -> float:
        """Get the current temperament score."""
        return self.state_manager.get_state().current_temperament
        
    @property
    def mood_label(self) -> str:
        """Get a human-readable mood label based on the current pressure or score."""
        return self.pressure.get_mood_label()

    def adjust_parameter(
        self,
        base_value: float,
        parameter_type: str,
        curiosity_pressure: Optional[float] = None
    ) -> float:
        """Adjust a parameter based on current temperament and curiosity pressure."""
        try:
            # Validate inputs
            if not 0.0 <= base_value <= 1.0:
                raise ValueError(f"Base value must be between 0.0 and 1.0, got {base_value}")
            if curiosity_pressure is not None and not 0.0 <= curiosity_pressure <= 1.0:
                raise ValueError(f"Curiosity pressure must be between 0.0 and 1.0, got {curiosity_pressure}")
            
            # Get current temperament score and lifecycle stage
            current_score = self.current_score
            lifecycle_params = self.temperament_config.get("temperament_config.lifecycle_params", {})
            stage_params = lifecycle_params.get(self._lifecycle_stage, {}) if lifecycle_params else {}
            
            # Calculate adjustment based on parameter type
            if parameter_type == "temperature":
                # Base adjustment from temperament
                adjustment = (current_score - 0.5) * 0.3  # Scale to ±0.15
                
                # Add curiosity influence if available
                if curiosity_pressure is not None:
                    adjustment += curiosity_pressure * 0.2  # Scale to +0.2
                
                # Add pressure influence
                pressure_influence = (self.pressure.current_pressure - 0.5) * 0.2  # Scale to ±0.1
                adjustment += pressure_influence
                
                # Apply lifecycle-based adjustments if available
                if stage_params and "temperature_bias" in stage_params:
                    try:
                        temperature_bias = float(stage_params["temperature_bias"])
                        if -0.5 <= temperature_bias <= 0.5:  # Validate bias range
                            adjustment += temperature_bias
                        else:
                            self.logger.record_event(
                                event_type="temperament_parameter_warning",
                                message="Temperature bias out of valid range, ignoring",
                                level="warning",
                                additional_info={
                                    "temperature_bias": temperature_bias,
                                    "lifecycle_stage": self._lifecycle_stage
                                }
                            )
                    except (ValueError, TypeError):
                        self.logger.record_event(
                            event_type="temperament_parameter_error",
                            message="Invalid temperature bias value",
                            level="error",
                            additional_info={
                                "temperature_bias": stage_params.get("temperature_bias"),
                                "lifecycle_stage": self._lifecycle_stage
                            }
                        )
                
                # Apply adjustment with bounds
                adjusted_value = base_value + adjustment
                adjusted_value = max(0.1, min(1.0, adjusted_value))
                
                # Log the adjustment with lifecycle context
                self.logger.record_event(
                    event_type="parameter_adjusted",
                    message="Parameter adjusted with lifecycle and pressure context",
                    level="info",
                    additional_info={
                        "parameter_type": parameter_type,
                        "base_value": base_value,
                        "adjusted_value": adjusted_value,
                        "temperament_score": current_score,
                        "curiosity_pressure": curiosity_pressure,
                        "pressure_influence": pressure_influence,
                        "lifecycle_stage": self._lifecycle_stage,
                        "adjustment": adjustment,
                        "lifecycle_params": stage_params
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
                    "base_value": base_value,
                    "curiosity_pressure": curiosity_pressure
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
                    "base_value": base_value,
                    "curiosity_pressure": curiosity_pressure
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
        event_dispatcher: EventDispatcher
    ):
        """Initialize temperament adjuster with required dependencies."""
        self.config_handler = config_handler
        self.state_tracker = state_tracker
        self.logger = logger
        self.event_dispatcher = event_dispatcher
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
            
    def _on_state_update(self, state: SOVLState) -> None:
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
            
    def _validate_state_consistency(self, state: SOVLState, new_score: float) -> bool:
        """
        Validate consistency between the potential new score and the existing history.
        Checks for large jumps and parameter changes.
        
        Args:
            state: The current SOVLState object (before update).
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
            
            if abs(new_score - last_recorded_score) > inconsistency_threshold:
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
            
    def _compute_state_hash(self, state: SOVLState) -> str:
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
                config_manager=self.config_handler.config_manager
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
            "temp_curiosity_boost": (0.1, 0.5),
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
    """Monitors temperament score and triggers empty prompts when thresholds are met."""
    
    def __init__(
        self,
        config_manager: ConfigManager
    ):
        """
        Initialize temperament pressure monitoring, validating config values.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = config_manager.logger
        
        # Get configuration values with defaults and validation
        config_section = "temperament_config"
        
        self.empty_prompt_threshold = self._get_validated_float(
            config_section, "temperament_empty_prompt_threshold", 0.7, 0.0, 1.0
        )
        self.cooldown_period = self._get_validated_float(
            config_section, "temperament_cooldown_period", 300, 0.0, float('inf')
        )
        self.min_pressure = self._get_validated_float(
            config_section, "temperament_min_pressure", 0.0, 0.0, 1.0
        )
        self.max_pressure = self._get_validated_float(
            config_section, "temperament_max_pressure", 1.0, 0.0, 1.0
        )

        # Validate min_pressure <= max_pressure
        if self.min_pressure > self.max_pressure:
             self._log_error(
                 f"Configuration error: min_pressure ({self.min_pressure}) cannot be greater than max_pressure ({self.max_pressure}). Using defaults.",
                 error_type="temperament_config_error"
             )
             self.min_pressure = 0.0
             self.max_pressure = 1.0

        # Initialize state
        self.current_pressure = self.min_pressure
        self.last_trigger_time = 0
        
        # Log initialization with validated values
        self._log_event(
            "temperament_pressure_initialized",
            "Temperament pressure monitoring initialized with validated config",
            level="info",
            additional_info={
                "empty_prompt_threshold": self.empty_prompt_threshold,
                "cooldown_period": self.cooldown_period,
                "min_pressure": self.min_pressure,
                "max_pressure": self.max_pressure,
                "initial_pressure": self.current_pressure
            }
        )

    def _get_validated_float(self, section: str, key: str, default: float, min_val: float, max_val: float) -> float:
        """Helper to get and validate a float config value."""
        full_key = f"{section}.{key}"
        value = self.config_manager.get(full_key, default)
        
        if not isinstance(value, (int, float)):
            self._log_event(
                "temperament_pressure_config_warning",
                f"Config value {full_key} is not numeric ({type(value)}), using default {default}",
                level="warning",
                additional_info={"key": full_key, "value": value}
            )
            return default
            
        value = float(value) # Ensure it's float
        
        if not (min_val <= value <= max_val):
             self._log_event(
                "temperament_pressure_config_warning",
                f"Config value {full_key} ({value}) out of range [{min_val}, {max_val}], clamping",
                level="warning",
                additional_info={"key": full_key, "value": value}
             )
             value = max(min_val, min(value, max_val))
             
        return value

    def should_trigger_empty_prompt(self, temperament_score: float) -> bool:
        """
        Check if an empty prompt should be triggered based on temperament score.
        Updates internal pressure based on the provided score.
        
        Args:
            temperament_score: Current temperament score from TemperamentSystem
            
        Returns:
            bool: True if empty prompt should be triggered
        """
        try:
            current_time = time.time()
            
            # Check cooldown period first
            if current_time - self.last_trigger_time < self.cooldown_period:
                return False # Still in cooldown
            
            # Update internal pressure based on the current temperament score
            self.current_pressure = max(self.min_pressure, 
                                     min(self.max_pressure, temperament_score))
            
            # Check if should trigger
            should_trigger = self.current_pressure >= self.empty_prompt_threshold
            
            if should_trigger:
                self.last_trigger_time = current_time
                self._log_event(
                    "temperament_empty_prompt_triggered",
                    "Empty prompt triggered due to high temperament pressure",
                    level="info",
                    additional_info={
                        "temperament_score": temperament_score,
                        "current_pressure": self.current_pressure,
                        "threshold": self.empty_prompt_threshold,
                        "time_since_last_trigger": current_time - self.last_trigger_time
                    }
                )
            
            return should_trigger
            
        except Exception as e:
            self._log_error(
                f"Failed to check empty prompt trigger: {str(e)}",
                error_type="temperament_trigger_error",
                stack_trace=traceback.format_exc()
            )
            return False

    def should_adjust(self, threshold: float) -> bool:
        """
        Checks if the current pressure meets a generic adjustment threshold.
        Used by TemperamentSystem to decide if pressure-related actions should occur.
        
        Args:
            threshold: The threshold to check against.

        Returns:
            bool: True if current_pressure >= threshold.
        """
        return self.current_pressure >= threshold

    def drop_pressure(self, amount: float) -> None:
        """
        Reduce pressure by a specified amount.
        
        Args:
            amount: Amount to reduce pressure by (non-negative)
        """
        try:
            if not isinstance(amount, (int, float)) or amount < 0:
                raise ValueError("Amount must be a non-negative number")
            
            old_pressure = self.current_pressure
            self.current_pressure = max(self.min_pressure, self.current_pressure - amount)
            
            self._log_event(
                "temperament_pressure_dropped",
                "Reduced temperament pressure",
                level="debug",
                additional_info={
                    "old_pressure": old_pressure,
                    "new_pressure": self.current_pressure,
                    "amount": amount
                }
            )
            
        except Exception as e:
            self._log_error(
                f"Failed to drop temperament pressure: {str(e)}",
                error_type="temperament_pressure_drop_error",
                stack_trace=traceback.format_exc()
            )

    def get_mood_label(self) -> str:
        """
        Get mood label based on current pressure.
        
        Returns:
            str: Mood label
        """
        try:
            if self.current_pressure < 0.3:
                return "Cautious"
            elif self.current_pressure < 0.7:
                return "Balanced"
            else:
                return "Curious"
                
        except Exception as e:
            self._log_error(
                f"Failed to determine mood label: {str(e)}",
                error_type="temperament_mood_label_error",
                stack_trace=traceback.format_exc()
            )
            return "Unknown"

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log event with standardized format."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info=kwargs
        )

    def _log_error(self, message: str, error_type: str = "temperament_pressure_error", **kwargs) -> None:
        """Log error with standardized format."""
        self.logger.log_error(
            error_msg=message,
            error_type=error_type,
            stack_trace=kwargs.get("stack_trace", traceback.format_exc()),
            additional_info=kwargs.get("additional_info", {})
        )

class TemperamentManager:
    """Manages temperament state updates and triggers internal prompts based on pressure."""
    
    def __init__(
        self, 
        config_manager: ConfigManager, 
        logger: Logger, 
        state: SOVLState, 
        generation_manager: Any # Use Any to avoid potential circular import
    ):
        """Initialize the TemperamentManager."""
        self.config_manager = config_manager
        self.logger = logger
        self.state = state
        self.generation_manager = generation_manager
        
        # Initialize the pressure component using its own config
        # Assumes TemperamentPressure is defined above in this file
        self.pressure = TemperamentPressure(config_manager=self.config_manager) 

        self._log_event("temperament_manager_initialized", "Temperament Manager initialized.")

    def check_and_generate_internal_prompt(self) -> Optional[str]:
        """
        Checks temperament pressure and triggers internal prompt generation if threshold is met.
        Returns:
            Optional[str]: The generated internal prompt response string if triggered, otherwise None.
        """
        try:
            # Safely get current temperament score from state
            if not hasattr(self.state, 'temperament_score'):
                 self._log_error(
                     "State object missing 'temperament_score'. Cannot check pressure.", 
                     error_type="state_error"
                 )
                 return None
            # Ensure score is a float for comparison
            temperament_score = float(self.state.temperament_score)

            # Check if the pressure threshold is met using the pressure component
            if self.pressure.should_trigger_empty_prompt(temperament_score):
                self._log_event(
                    "internal_prompt_trigger_check", 
                    f"Temperament pressure {self.pressure.current_pressure:.2f} meets or exceeds threshold {self.pressure.empty_prompt_threshold:.2f}. Triggering internal prompt.",
                    level="info",
                    temperament_score=temperament_score,
                    current_pressure=self.pressure.current_pressure,
                    threshold=self.pressure.empty_prompt_threshold
                )

                prompt = " "  # Default minimal prompt
                max_retries = 3
                response = None
                last_exception = None

                for attempt in range(max_retries):
                    try:
                        result = self.generation_manager.generate_text(
                            prompt=prompt,
                            num_return_sequences=1,
                            temperature=1.0
                        )
                        response = result[0] if result and isinstance(result, list) else None
                        if response:
                            break
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            self.logger.log_warning(
                                f"Internal prompt generation attempt {attempt + 1} failed: {e}"
                            )
                            time.sleep(1)
                            continue
                        self.logger.log_error(
                            f"Internal prompt generation failed after {max_retries} attempts: {e}"
                        )
                        response = None

                # Fallback prompt if all attempts fail
                if not response:
                    response = "[No response generated. Please try again later.]"
                    self.logger.log_warning("Using fallback prompt due to generation failure")

                # Only capture scribe event for successful (non-fallback) generations
                if response and (last_exception is None or response != "[No response generated. Please try again later.]"):
                    try:
                        from sovl_queue import capture_scribe_event
                        capture_scribe_event(
                            origin="sovl_temperament",
                            event_type="temperament_yell",
                            event_data={
                                "prompt": prompt,
                                "response": response,
                                "full_text": response,
                                "temperament_score": temperament_score,
                                "pressure": self.pressure.current_pressure,
                                "threshold": self.pressure.empty_prompt_threshold
                            },
                            source_metadata={
                                "module": "TemperamentManager",
                                "session_id": getattr(self, 'session_id', None)
                            },
                            session_id=getattr(self, 'session_id', None)
                        )
                    except Exception as e:
                        self._log_error(
                            f"Failed to capture scribe event for internal prompt: {str(e)}",
                            error_type="scribe_event_error",
                            stack_trace=traceback.format_exc()
                        )

                if response and (last_exception is None or response != "[No response generated. Please try again later.]"):
                    self._log_event(
                        "internal_prompt_generated",
                        "Internal prompt generated successfully due to temperament pressure.",
                        level="info",
                        response_snippet=response[:50] + '...' if response else 'None'
                    )
                else:
                    self._log_event(
                        "internal_prompt_generation_failed",
                        "Failed to generate internal prompt, using fallback.",
                        level="warning",
                        response_snippet=response[:50] + '...' if response else 'None'
                    )
                return response
            else:
                # Threshold not met, no internal prompt needed
                return None

        except AttributeError as ae:
            # Handle cases where generation_manager might be missing expected methods
            self._log_error(
                f"Missing method or attribute during internal prompt check: {str(ae)}. Check GenerationManager dependency.",
                error_type="dependency_error",
                stack_trace=traceback.format_exc()
            )
            return None
        except Exception as e:
            # Catch any other unexpected errors
            self._log_error(
                f"Unexpected error during internal prompt check: {str(e)}",
                error_type="internal_prompt_check_error",
                stack_trace=traceback.format_exc()
            )
            return None  # Return None on error

    # --- Logging Helper Methods ---

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log event with standardized format."""
        # Ensure logger is available before attempting to log
        if not hasattr(self, 'logger') or not self.logger: return 
        try:
            self.logger.record_event(
                event_type=event_type,
                message=message,
                level=level,
                component="TemperamentManager", # Add component name for context
                additional_info=kwargs
            )
        except Exception:
            # Avoid logging errors causing further issues
            pass 

    def _log_error(self, message: str, error_type: str = "temperament_manager_error", **kwargs) -> None:
        """Log error with standardized format."""
        # Ensure logger is available before attempting to log
        if not hasattr(self, 'logger') or not self.logger: return 
        try:
            self.logger.log_error(
                error_msg=message,
                error_type=error_type,
                component="TemperamentManager", # Add component name for context
                stack_trace=kwargs.get("stack_trace", traceback.format_exc()),
                additional_info=kwargs.get("additional_info", {})
            )
        except Exception:
            # Avoid logging errors causing further issues
            pass

