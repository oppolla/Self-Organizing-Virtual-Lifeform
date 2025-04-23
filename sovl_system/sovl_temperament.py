import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import traceback
from sovl_config import ConfigManager
from sovl_state import SOVLState
from sovl_logger import Logger
from sovl_events import EventDispatcher
from sovl_trainer import TrainingCycleManager
from sovl_confidence import ConfidenceCalculator
import math
from sovl_utils import synchronized, safe_divide
from sovl_error import ErrorHandler

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
    
    def __init__(self, state: SOVLState, config_manager: ConfigManager, lifecycle_manager: Optional[Any] = None):
        """
        Initialize temperament system.
        
        Args:
            state: SOVL state instance
            config_manager: Configuration manager instance
            lifecycle_manager: Optional LifecycleManager instance for lifecycle-based adjustments
        """
        self.state = state
        self.config_manager = config_manager
        self.temperament_config = TemperamentConfig(config_manager)
        self.logger = config_manager.logger
        self.lifecycle_manager = lifecycle_manager
        self._lifecycle_stage = "initialization"
        self._last_lifecycle_update = time.time()
        
        # Initialize temperament pressure
        self.pressure = TemperamentPressure(config_manager)
        
        # Initialize lifecycle integration if available
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
        
        Args:
            new_score: New temperament score (0.0 to 1.0)
            confidence: Confidence level in the update (0.0 to 1.0)
            lifecycle_stage: Optional current lifecycle stage. If None, will use lifecycle_manager if available.
        """
        try:
            # Validate inputs
            if not isinstance(new_score, (int, float)) or not 0.0 <= new_score <= 1.0:
                self.logger.record_event(
                    event_type="temperament_update_invalid_score",
                    message=f"Invalid temperament score: {new_score}. Ignoring update.",
                    level="warning",
                    additional_info={
                        "lifecycle_stage": lifecycle_stage,
                        "current_score": self.state.current_temperament
                    }
                )
                return

            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                self.logger.record_event(
                    event_type="temperament_update_invalid_confidence",
                    message=f"Invalid confidence: {confidence}. Ignoring update.",
                    level="warning",
                    additional_info={
                        "lifecycle_stage": lifecycle_stage,
                        "current_score": self.state.current_temperament
                    }
                )
                return
                
            # Get configuration values
            smoothing_factor = self.temperament_config.get("temperament_config.temp_smoothing_factor")
            feedback_strength = self.temperament_config.get("temperament_config.conf_feedback_strength")
            eager_threshold = self.temperament_config.get("temperament_config.temp_eager_threshold", 0.7)
            pressure_drop = self.temperament_config.get("temperament_config.temperament_pressure_drop", 0.2)
            
            # Get lifecycle stage from manager if not provided
            if lifecycle_stage is None and self.lifecycle_manager:
                lifecycle_stage = self.lifecycle_manager.get_lifecycle_stage()
            
            # Update pressure
            current_pressure = self.pressure.update(confidence, lifecycle_stage)
            
            # Apply lifecycle-based adjustments to score
            lifecycle_params = self.temperament_config.get("temperament_config.lifecycle_params", {})
            if lifecycle_stage in lifecycle_params:
                stage_params = lifecycle_params[lifecycle_stage]
                bias = stage_params.get("bias", 0.0)
                decay = stage_params.get("decay", 1.0)
                
                # Apply bias and decay based on lifecycle stage
                time_since_update = time.time() - self._last_lifecycle_update
                decay_factor = math.exp(-decay * time_since_update)
                adjusted_score = (new_score + bias) * decay_factor
                
                # Ensure score remains in valid range
                adjusted_score = max(0.0, min(1.0, adjusted_score))
                
                # Log lifecycle adjustments
                self.logger.record_event(
                    event_type="temperament_lifecycle_adjustment",
                    message="Applied lifecycle-based temperament adjustments",
                    level="info",
                    additional_info={
                        "lifecycle_stage": lifecycle_stage,
                        "bias": bias,
                        "decay": decay,
                        "decay_factor": decay_factor,
                        "adjusted_score": adjusted_score
                    }
                )
            else:
                adjusted_score = new_score
            
            # Check if temperament should adjust based on pressure
            if self.pressure.should_adjust(eager_threshold):
                # Update state with adjusted score
                self.state.update_temperament(adjusted_score)
                self.pressure.drop_pressure(pressure_drop)
                
                self.logger.record_event(
                    event_type="temperament_adjusted",
                    message="Temperament adjusted due to pressure threshold",
                    level="info",
                    additional_info={
                        "new_score": adjusted_score,
                        "confidence": confidence,
                        "current_pressure": current_pressure,
                        "eager_threshold": eager_threshold,
                        "pressure_drop": pressure_drop,
                        "lifecycle_stage": lifecycle_stage
                    }
                )
            
            # Update lifecycle state
            self._lifecycle_stage = lifecycle_stage
            self._last_lifecycle_update = time.time()
            
            # Log the update with enhanced lifecycle context
            self.logger.record_event(
                event_type="temperament_updated",
                message="Temperament system updated",
                level="info",
                additional_info={
                    "new_score": adjusted_score,
                    "confidence": confidence,
                    "current_pressure": current_pressure,
                    "lifecycle_stage": lifecycle_stage,
                    "current_score": self.state.current_temperament,
                    "conversation_id": self.state.conversation_id,
                    "state_hash": self.state.state_hash,
                    "smoothing_factor": smoothing_factor,
                    "feedback_strength": feedback_strength,
                    "time_since_last_update": time.time() - self._last_lifecycle_update,
                    "lifecycle_params": lifecycle_params.get(lifecycle_stage, {})
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
                    "current_score": self.state.current_temperament
                }
            )
            raise
        
    @property
    def current_score(self) -> float:
        """Get the current temperament score."""
        return self.state.current_temperament
        
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
                raise ValueError(f"Unsupported parameter type: {parameter_type}")
            
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
            return base_value  # Return base value on error

class TemperamentAdjuster:
    """Manages temperament adjustments and state updates."""
    
    def __init__(
        self,
        config_handler: ConfigHandler,
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
            current_params = self._get_validated_parameters()
            current_hash = self._compute_parameter_hash(current_params)
            
            if current_hash != self._last_parameter_hash:
                self.logger.record_event(
                    event_type="temperament_parameters_changed",
                    message="Temperament parameters changed, reinitializing system",
                    level="info",
                    additional_info=current_params
                )
                self._initialize_temperament_system()
                
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to handle config change: {str(e)}",
                error_type="temperament_config_error",
                stack_trace=traceback.format_exc()
            )
            
    def _on_state_update(self, state: SOVLState) -> None:
        """Handle state updates."""
        try:
            # Validate state consistency
            if not self._validate_state_consistency(state):
                # Reset history if inconsistent
                state.temperament_history.clear()
                self.logger.record_event(
                    event_type="temperament_history_reset",
                    message="Temperament history reset due to inconsistency",
                    level="info",
                    additional_info={
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
            
            # Update state with current temperament
            state.temperament_score = self.temperament_system.current_score
            state.temperament_history.append(state.temperament_score)
            
            # Update state hash
            self._last_state_hash = self._compute_state_hash(state)
            
            # Notify other components
            self.event_dispatcher.notify("temperament_updated", state)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to synchronize state: {str(e)}",
                error_type="temperament_state_error",
                stack_trace=traceback.format_exc(),
                conversation_id=state.conversation_id,
                state_hash=state.state_hash
            )
            raise
            
    def _validate_state_consistency(self, state: SOVLState) -> bool:
        """Validate consistency between current state and temperament history."""
        try:
            if not state.temperament_history:
                return True
                
            # Check for significant deviation between current score and history
            if abs(state.temperament_history[-1] - state.temperament_score) > 0.5:
                self.logger.record_event(
                    event_type="temperament_inconsistency",
                    message="Temperament history inconsistent with current score",
                    level="warning",
                    additional_info={
                        "current_score": state.temperament_score,
                        "last_history_score": state.temperament_history[-1],
                        "history_length": len(state.temperament_history),
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
                return False
                
            # Check for parameter changes that might invalidate history
            current_hash = self._compute_parameter_hash(self._get_validated_parameters())
            if current_hash != self._last_parameter_hash:
                self.logger.record_event(
                    event_type="temperament_history_invalidated",
                    message="Temperament parameters changed, history may be invalid",
                    level="warning",
                    additional_info={
                        "parameter_hash": current_hash,
                        "last_parameter_hash": self._last_parameter_hash,
                        "conversation_id": state.conversation_id,
                        "state_hash": state.state_hash
                    }
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to validate state consistency: {str(e)}",
                error_type="temperament_validation_error",
                stack_trace=traceback.format_exc(),
                conversation_id=state.conversation_id,
                state_hash=state.state_hash
            )
            return False
            
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
                state=self.state_tracker.get_state(),
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
        """Get and validate temperament parameters."""
        config = self.config_handler.config_manager
        
        # Define safe parameter ranges
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
        
        # Get and validate parameters
        params = {}
        for key, (min_val, max_val) in safe_ranges.items():
            value = config.get(f"controls_config.{key}", (min_val + max_val) / 2)
            if not (min_val <= value <= max_val):
                self.logger.record_event(
                    event_type="temperament_parameter_warning",
                    message=f"Parameter {key} out of safe range, clamping to bounds",
                    level="warning",
                    additional_info={
                        "parameter": key,
                        "value": value,
                        "min": min_val,
                        "max": max_val
                    }
                )
                value = max(min_val, min(value, max_val))
            params[key] = value
            
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
        Initialize temperament pressure monitoring.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.logger = config_manager.logger
        
        # Get configuration values with defaults
        config = config_manager.get_section("temperament_config", {})
        self.empty_prompt_threshold = config.get("temperament_empty_prompt_threshold", 0.7)
        self.cooldown_period = config.get("temperament_cooldown_period", 300)  # 5 minutes default
        self.min_pressure = config.get("temperament_min_pressure", 0.0)
        self.max_pressure = config.get("temperament_max_pressure", 1.0)
        
        # Initialize state
        self.current_pressure = self.min_pressure
        self.last_trigger_time = 0
        
        # Log initialization
        self._log_event(
            "temperament_pressure_initialized",
            "Temperament pressure monitoring initialized",
            level="info",
            additional_info={
                "empty_prompt_threshold": self.empty_prompt_threshold,
                "cooldown_period": self.cooldown_period,
                "min_pressure": self.min_pressure,
                "max_pressure": self.max_pressure
            }
        )

    def should_trigger_empty_prompt(self, temperament_score: float) -> bool:
        """
        Check if an empty prompt should be triggered based on temperament score.
        
        Args:
            temperament_score: Current temperament score from TemperamentSystem
            
        Returns:
            bool: True if empty prompt should be triggered
        """
        try:
            current_time = time.time()
            
            # Check cooldown period
            if current_time - self.last_trigger_time < self.cooldown_period:
                return False
            
            # Update internal pressure based on temperament score
            self.current_pressure = max(self.min_pressure, 
                                     min(self.max_pressure, temperament_score))
            
            # Check if should trigger
            should_trigger = self.current_pressure >= self.empty_prompt_threshold
            
            if should_trigger:
                self.last_trigger_time = current_time
                self._log_event(
                    "temperament_empty_prompt_triggered",
                    "Empty prompt triggered due to high temperament",
                    level="info",
                    additional_info={
                        "temperament_score": temperament_score,
                        "current_pressure": self.current_pressure,
                        "threshold": self.empty_prompt_threshold,
                        "time_since_last": current_time - self.last_trigger_time
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
                
                # Call the GenerationManager's method to handle internal prompts.
                # It uses the default prompt (" ") unless specified otherwise.
                response = self.generation_manager._handle_internal_prompt() 
                
                self._log_event(
                    "internal_prompt_generated",
                    "Internal prompt generated successfully due to temperament pressure.",
                    level="info",
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
            return None # Return None on error

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

# ... potentially other code like utility functions if they exist at the end ...
