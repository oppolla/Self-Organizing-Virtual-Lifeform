from typing import Optional, Dict, Any, List, Tuple
import torch
from threading import Lock
from collections import deque
from sovl_logger import Logger
import traceback
from sovl_state import SOVLState, StateManager
from sovl_error import ErrorManager
from sovl_main import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized, NumericalGuard
from sovl_config import ConfigManager
from sovl_temperament import TemperamentSystem
from sovl_trainer import TrainingCycleManager
import time
import threading
import math
from torch import nn
from sovl_error import ErrorManager

# Constants
DEFAULT_CONFIDENCE = 0.5
RECOVERY_WEIGHTS = [0.5, 0.3, 0.2]
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_HISTORY_LENGTH = 3
CURIOSITY_PRESSURE_FACTOR = 0.1
DEFAULT_TEMPERAMENT_INFLUENCE = 0.3
DEFAULT_LIFECYCLE_INFLUENCE = 0.2

# Temperament-based confidence adjustments
TEMPERAMENT_MOOD_MULTIPLIERS = {
    "Cautious": 0.8,  # Reduce confidence in cautious mood
    "Balanced": 1.0,  # No adjustment in balanced mood
    "Curious": 1.2    # Increase confidence in curious mood
}

# Lifecycle stage adjustments
LIFECYCLE_STAGE_MULTIPLIERS = {
    "initialization": 0.9,    # More conservative during initialization
    "exploration": 1.1,       # More confident during exploration
    "consolidation": 1.0,     # Normal confidence during consolidation
    "refinement": 0.95        # Slightly more conservative during refinement
}

"""
Confidence calculation module for the SOVL system.

This module provides functionality to calculate confidence scores for model outputs,
incorporating curiosity and temperament adjustments. It is thread-safe and includes
robust error recovery mechanisms.

Primary interface: calculate_confidence_score
"""


class ConfidenceCalculator:
    """Handles confidence score calculation with thread safety.
    All mutations to SOVLState must use StateManager.update_state_atomic(update_fn) for atomicity, versioning, and validation.
    """
    
    def __init__(
        self, 
        config_manager: ConfigManager, 
        logger: Logger, 
        temperament_system: Optional[TemperamentSystem] = None,
        lifecycle_manager: Optional[TrainingCycleManager] = None,
        state_manager: Optional[Any] = None
    ):
        """Initialize the confidence calculator with configuration and logging.
        
        Args:
            config_manager: ConfigManager instance for configuration handling
            logger: Logger instance for logging
            temperament_system: Optional TemperamentSystem instance for mood-based adjustments
            lifecycle_manager: Optional TrainingCycleManager instance for lifecycle-based adjustments
            state_manager: Optional StateManager for atomic state updates
            
        Raises:
            ValueError: If config_manager or logger is None
            TypeError: If config_manager is not a ConfigManager instance
        """
        if not config_manager:
            raise ValueError("config_manager cannot be None")
        if not logger:
            raise ValueError("logger cannot be None")
        if not isinstance(config_manager, ConfigManager):
            raise TypeError("config_manager must be a ConfigManager instance")
            
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()
        self.temperament_system = temperament_system
        self.lifecycle_manager = lifecycle_manager
        self.state_manager = state_manager
        
        # Initialize configuration
        self._initialize_config()
        
    def _initialize_config(self) -> None:
        """Initialize and validate configuration from ConfigManager."""
        try:
            # Load confidence configuration
            confidence_config = self.config_manager.get_section("confidence_config")
            
            # Set configuration parameters with validation
            self.min_confidence = float(confidence_config.get("min_confidence", 0.0))
            self.max_confidence = float(confidence_config.get("max_confidence", 1.0))
            self.default_confidence = float(confidence_config.get("default_confidence", 0.5))
            self.min_history_length = int(confidence_config.get("min_history_length", 3))
            self.curiosity_pressure_factor = float(confidence_config.get("curiosity_pressure_factor", 0.1))
            self.temperament_influence = float(confidence_config.get("temperament_influence", 0.3))
            self.lifecycle_influence = float(confidence_config.get("lifecycle_influence", 0.2))
            self.recovery_weights = [
                float(w) for w in confidence_config.get("recovery_weights", [0.5, 0.3, 0.2])
            ]
            
            # Load temperament configuration if available
            if self.temperament_system:
                self.temperament_config = self.temperament_system.temperament_config
                self.logger.record_event(
                    event_type="temperament_integration_initialized",
                    message="Temperament system integration initialized",
                    level="info",
                    additional_info={
                        "temperament_influence": self.temperament_influence,
                        "mood_multipliers": TEMPERAMENT_MOOD_MULTIPLIERS,
                        "lifecycle_multipliers": LIFECYCLE_STAGE_MULTIPLIERS
                    }
                )
            
            # Load lifecycle configuration if available
            if self.lifecycle_manager:
                self.logger.record_event(
                    event_type="lifecycle_integration_initialized",
                    message="Lifecycle manager integration initialized",
                    level="info",
                    additional_info={
                        "lifecycle_influence": self.lifecycle_influence,
                        "stage_multipliers": LIFECYCLE_STAGE_MULTIPLIERS
                    }
                )
            
            # Validate configuration values
            self._validate_config_values()
            
            # Subscribe to configuration changes
            self.config_manager.subscribe(self._on_config_change)
            
            self.logger.record_event(
                event_type="confidence_config_initialized",
                message="Confidence configuration initialized successfully",
                level="info",
                additional_info={
                    "min_confidence": self.min_confidence,
                    "max_confidence": self.max_confidence,
                    "default_confidence": self.default_confidence,
                    "min_history_length": self.min_history_length,
                    "curiosity_pressure_factor": self.curiosity_pressure_factor,
                    "temperament_influence": self.temperament_influence,
                    "lifecycle_influence": self.lifecycle_influence,
                    "recovery_weights": self.recovery_weights
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="confidence_config_initialization_failed",
                message=f"Failed to initialize confidence configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _validate_config_values(self) -> None:
        """Validate configuration values against defined ranges."""
        try:
            # Validate confidence ranges
            if not 0.0 <= self.min_confidence <= 1.0:
                raise ValueError(f"Invalid min_confidence: {self.min_confidence}. Must be between 0.0 and 1.0.")
                
            if not 0.0 <= self.max_confidence <= 1.0:
                raise ValueError(f"Invalid max_confidence: {self.max_confidence}. Must be between 0.0 and 1.0.")
                
            if self.min_confidence >= self.max_confidence:
                raise ValueError(f"min_confidence ({self.min_confidence}) must be less than max_confidence ({self.max_confidence})")
                
            if not 0.0 <= self.default_confidence <= 1.0:
                raise ValueError(f"Invalid default_confidence: {self.default_confidence}. Must be between 0.0 and 1.0.")
                
            # Validate history parameters
            if not 1 <= self.min_history_length <= 10:
                raise ValueError(f"Invalid min_history_length: {self.min_history_length}. Must be between 1 and 10.")
                
            # Validate influence factors
            if not 0.0 <= self.curiosity_pressure_factor <= 1.0:
                raise ValueError(f"Invalid curiosity_pressure_factor: {self.curiosity_pressure_factor}. Must be between 0.0 and 1.0.")
                
            if not 0.0 <= self.temperament_influence <= 1.0:
                raise ValueError(f"Invalid temperament_influence: {self.temperament_influence}. Must be between 0.0 and 1.0.")
                
            if not 0.0 <= self.lifecycle_influence <= 1.0:
                raise ValueError(f"Invalid lifecycle_influence: {self.lifecycle_influence}. Must be between 0.0 and 1.0.")
                
            # Validate recovery weights
            if len(self.recovery_weights) != 3:
                raise ValueError(f"Invalid recovery_weights length: {len(self.recovery_weights)}. Must be exactly 3 weights.")
                
            if not all(0.0 <= w <= 1.0 for w in self.recovery_weights):
                raise ValueError("All recovery weights must be between 0.0 and 1.0.")
                
            if abs(sum(self.recovery_weights) - 1.0) > 1e-6:
                raise ValueError("Recovery weights must sum to 1.0.")
                
        except Exception as e:
            self.logger.record_event(
                event_type="confidence_config_validation_failed",
                message=f"Configuration validation failed: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            self._initialize_config()
            self.logger.record_event(
                event_type="confidence_config_updated",
                message="Confidence configuration updated",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                event_type="confidence_config_update_failed",
                message=f"Failed to update confidence configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            
    def _apply_temperament_adjustments(self, base_confidence: float, state: SOVLState) -> float:
        """Apply temperament-based adjustments to confidence score.
        
        Args:
            base_confidence: Initial confidence score
            state: Current SOVL state
            
        Returns:
            float: Adjusted confidence score
        """
        if not self.temperament_system:
            return base_confidence
            
        try:
            # Get current mood and lifecycle stage
            mood_label = self.temperament_system.mood_label
            lifecycle_stage = getattr(state, 'lifecycle_stage', 'initialization')
            
            # Apply mood-based multiplier
            mood_multiplier = TEMPERAMENT_MOOD_MULTIPLIERS.get(mood_label, 1.0)
            
            # Apply lifecycle stage multiplier
            lifecycle_multiplier = LIFECYCLE_STAGE_MULTIPLIERS.get(lifecycle_stage, 1.0)
            
            # Calculate adjusted confidence
            adjusted_confidence = base_confidence * mood_multiplier * lifecycle_multiplier
            
            # Log the adjustments
            self.logger.record_event(
                event_type="temperament_adjustment_applied",
                message="Applied temperament-based confidence adjustments",
                level="info",
                additional_info={
                    "base_confidence": base_confidence,
                    "adjusted_confidence": adjusted_confidence,
                    "mood_label": mood_label,
                    "lifecycle_stage": lifecycle_stage,
                    "mood_multiplier": mood_multiplier,
                    "lifecycle_multiplier": lifecycle_multiplier
                }
            )
            
            return adjusted_confidence
            
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_adjustment_failed",
                message=f"Failed to apply temperament adjustments: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            return base_confidence

    def _apply_lifecycle_adjustments(self, base_confidence: float, state: SOVLState) -> float:
        """Apply lifecycle-based adjustments to confidence score.
        
        Args:
            base_confidence: Initial confidence score
            state: Current SOVL state
            
        Returns:
            float: Adjusted confidence score
        """
        if not self.lifecycle_manager:
            return base_confidence
            
        try:
            # Get current lifecycle stage
            lifecycle_stage = self.lifecycle_manager.get_lifecycle_stage()
            
            # Apply lifecycle stage multiplier
            stage_multiplier = LIFECYCLE_STAGE_MULTIPLIERS.get(lifecycle_stage, 1.0)
            
            # Calculate adjusted confidence
            adjusted_confidence = base_confidence * stage_multiplier
            
            # Log the adjustments
            self.logger.record_event(
                event_type="lifecycle_adjustment_applied",
                message="Applied lifecycle-based confidence adjustments",
                level="info",
                additional_info={
                    "base_confidence": base_confidence,
                    "adjusted_confidence": adjusted_confidence,
                    "lifecycle_stage": lifecycle_stage,
                    "stage_multiplier": stage_multiplier
                }
            )
            
            return adjusted_confidence
            
        except Exception as e:
            self.logger.record_event(
                event_type="lifecycle_adjustment_failed",
                message=f"Failed to apply lifecycle adjustments: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            return base_confidence

    @synchronized()
    def calculate_confidence_score(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None,
        state_manager: Optional[Any] = None
    ) -> float:
        """Calculate confidence score with robust error recovery and thread safety."""
        with self.lock:
            try:
                self.logger.record_event(
                    event_type="confidence_calculation_start",
                    message="Starting confidence calculation",
                    level="info",
                    additional_info={"state_id": getattr(state, 'id', None)}
                )
                # Validate inputs
                self.__validate_inputs(logits, generated_ids)
                # Calculate probabilities
                probs = self.__calculate_probabilities(logits)
                # Compute base confidence
                base_confidence = self.__compute_base_confidence(probs)
                # Apply adjustments
                adjusted_confidence = self.__apply_adjustments(
                    base_confidence, state, context, curiosity_manager
                )
                # Clamp to valid range
                adjusted_confidence = max(self.min_confidence, min(self.max_confidence, adjusted_confidence))
                # Finalize and update history
                final_confidence = self.__finalize_confidence(adjusted_confidence, state)
                self.logger.record_event(
                    event_type="confidence_calculation_success",
                    message="Confidence calculation completed",
                    level="info",
                    additional_info={"final_confidence": final_confidence}
                )
                return final_confidence
            except Exception as e:
                # Log error and attempt recovery
                error_manager.record_error(
                    error_type="confidence_calculation_error",
                    message=str(e),
                    stack_trace=traceback.format_exc(),
                    context="calculate_confidence_score"
                )
                self.logger.record_event(
                    event_type="confidence_calculation_failed",
                    message="Confidence calculation failed, attempting recovery",
                    level="error",
                    additional_info={"error": str(e)}
                )
                # Attempt recovery
                recovered_confidence = self.__recover_confidence(e, state, error_manager)
                # Clamp recovered confidence
                recovered_confidence = max(self.min_confidence, min(self.max_confidence, recovered_confidence))
                return recovered_confidence

    def __validate_inputs(self, logits: torch.Tensor, generated_ids: torch.Tensor) -> None:
        """Validate input tensors for confidence calculation.
        
        Args:
            logits: Model output logits
            generated_ids: Generated token IDs
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(logits, torch.Tensor) or not isinstance(generated_ids, torch.Tensor):
            raise ValueError("logits and generated_ids must be torch.Tensor")
        if logits.dim() != 2 or generated_ids.dim() != 1:
            raise ValueError("logits must be 2D and generated_ids must be 1D")

    def __calculate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate softmax probabilities from logits.
        
        Args:
            logits: Model output logits
        
        Returns:
            torch.Tensor: Softmax probabilities
        """
        with NumericalGuard():
            return torch.softmax(logits, dim=-1)

    def __compute_base_confidence(self, probs: torch.Tensor) -> float:
        """Compute base confidence from probabilities.
        
        Args:
            probs: Softmax probabilities
        
        Returns:
            float: Base confidence score
        """
        max_probs = probs.max(dim=-1).values
        return max_probs.mean().item()

    def __apply_adjustments(
        self,
        base_confidence: float,
        state: SOVLState,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager]
    ) -> float:
        """Apply curiosity and temperament adjustments to confidence.
        
        Args:
            base_confidence: Initial confidence score
            state: Current SOVL state
            context: System context
            curiosity_manager: Optional curiosity manager
        
        Returns:
            float: Adjusted confidence score
        """
        confidence = base_confidence
        
        # Apply curiosity pressure adjustment if available
        if curiosity_manager is not None:
            pressure = curiosity_manager.get_pressure()
            confidence *= (1.0 - pressure * self.curiosity_pressure_factor)
        
        # Apply temperament influence
        confidence *= (1.0 + state.temperament_score * self.temperament_influence)
        
        return confidence

    def __finalize_confidence(self, confidence: float, state: SOVLState) -> float:
        """Finalize confidence score and update history atomically using StateManager."""
        if self.state_manager:
            def update_fn(s):
                c = max(self.min_confidence, min(self.max_confidence, confidence))
                if hasattr(s, 'confidence_history') and isinstance(s.confidence_history, deque):
                    s.confidence_history.append(c)
                    while len(s.confidence_history) > 100:
                        s.confidence_history.popleft()
                else:
                    s.confidence_history = deque([c], maxlen=100)
                return s
            self.state_manager.update_state_atomic(update_fn)
            return confidence
        else:
            # Deprecated: direct mutation fallback. StateManager is required for atomic updates.
            raise RuntimeError("StateManager is required for atomic confidence updates.")

    def __recover_confidence(self, error: Exception, state: SOVLState, error_manager: ErrorManager) -> float:
        """Attempt to recover confidence from history or use default, with defensive checks and logging."""
        # No mutation to state; only reads. If mutation is needed, use atomic update.
        try:
            # Validate confidence history
            if not hasattr(state, 'confidence_history') or not isinstance(state.confidence_history, deque):
                error_manager.logger.record_event(
                    event_type="confidence_history_error",
                    message="Invalid confidence history structure",
                    level="error",
                    additional_info={"error": str(error)}
                )
                return self.default_confidence

            if len(state.confidence_history) < self.min_history_length:
                error_manager.logger.record_event(
                    event_type="confidence_history_insufficient",
                    message="Insufficient history for recovery",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "history_length": len(state.confidence_history)
                    }
                )
                return self.default_confidence

            recent_confidences = list(state.confidence_history)[-self.min_history_length:]
            valid_confidences = [
                c for c in recent_confidences
                if isinstance(c, (int, float)) and self.min_confidence <= c <= self.max_confidence
            ]

            if not valid_confidences:
                # Try alternative: use temperament if available
                if hasattr(state, 'temperament_score') and isinstance(state.temperament_score, (int, float)):
                    recovered_confidence = self.default_confidence * (1.0 + state.temperament_score * 0.1)
                    recovered_confidence = max(self.min_confidence, min(self.max_confidence, recovered_confidence))
                    error_manager.logger.record_event(
                        event_type="confidence_recovered_temperament",
                        message="Recovered confidence using temperament score as fallback",
                        level="info",
                        additional_info={
                            "error": str(error),
                            "recovered_confidence": recovered_confidence,
                            "temperament_score": state.temperament_score
                        }
                    )
                    return recovered_confidence
                error_manager.logger.record_event(
                    event_type="confidence_history_no_valid_entries",
                    message="No valid history entries for recovery, using default",
                    level="warning",
                    additional_info={
                        "error": str(error),
                        "history": recent_confidences
                    }
                )
                return self.default_confidence

            # Use average of valid confidences for partial recovery
            weights = [1.0 / len(valid_confidences)] * len(valid_confidences)
            recovered_confidence = sum(c * w for c, w in zip(valid_confidences, weights))
            recovered_confidence = max(self.min_confidence, min(self.max_confidence, recovered_confidence))
            error_manager.logger.record_event(
                event_type="partial_confidence_recovery",
                message=f"Recovered confidence using {len(valid_confidences)} valid history entries",
                level="info",
                additional_info={
                    "error": str(error),
                    "recovered_confidence": recovered_confidence,
                    "valid_confidences": valid_confidences
                }
            )
            return recovered_confidence

        except Exception as recovery_error:
            error_manager.logger.record_event(
                event_type="confidence_recovery_failed",
                message="Failed to recover confidence",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "recovery_error": str(recovery_error)
                }
            )
            return self.default_confidence

# Singleton instance of ConfidenceCalculator
_confidence_calculator = None
_confidence_calculator_lock = Lock()

def calculate_confidence_score(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    state: SOVLState,
    error_manager: ErrorManager,
    context: SystemContext,
    curiosity_manager: Optional[CuriosityManager] = None,
    state_manager: Optional[Any] = None
) -> float:
    """Calculate confidence score with robust error recovery.
    
    Args:
        logits: Model output logits
        generated_ids: Generated token IDs
        state: Current SOVL state
        error_manager: Error handling manager
        context: System context
        curiosity_manager: Optional curiosity manager
        state_manager: Optional StateManager for atomic state updates
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    global _confidence_calculator
    with _confidence_calculator_lock:
        if _confidence_calculator is None:
            if not hasattr(state, 'config_manager') or not hasattr(state, 'logger'):
                raise ValueError("State missing required config_manager or logger")
            if not isinstance(state.config_manager, ConfigManager) or not isinstance(state.logger, Logger):
                raise TypeError("Invalid config_manager or logger types")
            _confidence_calculator = ConfidenceCalculator(state.config_manager, state.logger, state_manager=state_manager)
    return _confidence_calculator.calculate_confidence_score(
        logits=logits,
        generated_ids=generated_ids,
        state=state,
        error_manager=error_manager,
        context=context,
        curiosity_manager=curiosity_manager
    )

def reset_confidence_calculator():
    """Reset the singleton ConfidenceCalculator instance (for testing or reconfiguration)."""
    global _confidence_calculator
    with _confidence_calculator_lock:
        _confidence_calculator = None
