from typing import Optional, Dict, Any
import torch
from threading import Lock
from collections import deque
from sovl_logger import Logger
import traceback

# SOVL-specific dependencies
from sovl_state import SOVLState
from sovl_error import ErrorManager
from sovl_main import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized, NumericalGuard
from sovl_config import ConfigManager

# Constants
DEFAULT_CONFIDENCE = 0.5
RECOVERY_WEIGHTS = [0.5, 0.3, 0.2]
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_HISTORY_LENGTH = 3
CURIOSITY_PRESSURE_FACTOR = 0.1
DEFAULT_TEMPERAMENT_INFLUENCE = 0.3

"""
Confidence calculation module for the SOVL system.

This module provides functionality to calculate confidence scores for model outputs,
incorporating curiosity and temperament adjustments. It is thread-safe and includes
robust error recovery mechanisms.

Primary interface: calculate_confidence_score
"""


class ConfidenceCalculator:
    """Handles confidence score calculation with thread safety."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize the confidence calculator with configuration and logging.
        
        Args:
            config_manager: ConfigManager instance for configuration handling
            logger: Logger instance for logging
            
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
            self.recovery_weights = [
                float(w) for w in confidence_config.get("recovery_weights", [0.5, 0.3, 0.2])
            ]
            
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
            
    @synchronized()
    def calculate_confidence_score(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None
    ) -> float:
        """Calculate confidence score with robust error recovery.
        
        Args:
            logits: Model output logits
            generated_ids: Generated token IDs
            state: Current SOVL state
            error_manager: Error handling manager
            context: System context
            curiosity_manager: Optional curiosity manager
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        try:
            __validate_inputs(logits, generated_ids)
            probs = __calculate_probabilities(logits)
            base_confidence = __compute_base_confidence(probs)
            adjusted_confidence = __apply_adjustments(
                base_confidence, state, context, curiosity_manager
            )
            final_confidence = __finalize_confidence(adjusted_confidence, state)
            
            self.logger.record_event(
                event_type="confidence_calculated",
                message="Confidence calculation completed successfully",
                level="info",
                additional_info={
                    "base_confidence": base_confidence,
                    "adjusted_confidence": adjusted_confidence,
                    "final_confidence": final_confidence,
                    "logits_shape": logits.shape,
                    "generated_ids_shape": generated_ids.shape
                }
            )
            
            return final_confidence
        except Exception as e:
            return __recover_confidence(e, state, error_manager)

def __validate_inputs(logits: torch.Tensor, generated_ids: torch.Tensor) -> None:
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

def __calculate_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Calculate softmax probabilities from logits.
    
    Args:
        logits: Model output logits
        
    Returns:
        torch.Tensor: Softmax probabilities
    """
    with NumericalGuard():
        return torch.softmax(logits, dim=-1)

def __compute_base_confidence(probs: torch.Tensor) -> float:
    """Compute base confidence from probabilities.
    
    Args:
        probs: Softmax probabilities
        
    Returns:
        float: Base confidence score
    """
    max_probs = probs.max(dim=-1).values
    return max_probs.mean().item()

def __apply_adjustments(
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

def __finalize_confidence(confidence: float, state: SOVLState) -> float:
    """Finalize confidence score and update history.
    
    Args:
        confidence: Adjusted confidence score
        state: Current SOVL state
        
    Returns:
        float: Final confidence score
    """
    final_confidence = max(self.min_confidence, min(self.max_confidence, confidence))
    state.confidence_history.append(final_confidence)
    
    state.logger.record_event(
        event_type="confidence_finalized",
        message="Confidence score finalized and history updated",
        level="info",
        additional_info={
            "final_confidence": final_confidence,
            "history_length": len(state.confidence_history)
        }
    )
    
    return final_confidence

def __recover_confidence(error: Exception, state: SOVLState, error_manager: ErrorManager) -> float:
    """Attempt to recover confidence from history or use default.
    
    Args:
        error: Original exception
        state: Current SOVL state
        error_manager: Error handling manager
        
    Returns:
        float: Recovered confidence score
    """
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

        if len(state.confidence_history) >= self.min_history_length:
            # Use weighted average of recent confidences
            recent_confidences = list(state.confidence_history)[-self.min_history_length:]
            # Validate history values
            if any(not isinstance(c, (int, float)) or c < self.min_confidence or c > self.max_confidence for c in recent_confidences):
                error_manager.logger.record_event(
                    event_type="confidence_history_invalid",
                    message="Invalid values in confidence history",
                    level="error",
                    additional_info={"history": recent_confidences}
                )
                return self.default_confidence
                
            recovered_confidence = sum(c * w for c, w in zip(recent_confidences, self.recovery_weights))
            
            # Log recovery
            error_manager.logger.record_event(
                event_type="confidence_recovered",
                message="Recovered confidence from history",
                level="warning",
                additional_info={
                    "error": str(error),
                    "recovered_confidence": recovered_confidence,
                    "history_length": len(state.confidence_history)
                }
            )
            
            return recovered_confidence
            
        # If recovery fails, use conservative default
        error_manager.logger.record_event(
            event_type="confidence_default_used",
            message="Using default confidence due to insufficient history",
            level="warning",
            additional_info={
                "error": str(error),
                "history_length": len(state.confidence_history)
            }
        )
        return self.default_confidence
        
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

def calculate_confidence_score(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    state: SOVLState,
    error_manager: ErrorManager,
    context: SystemContext,
    curiosity_manager: Optional[CuriosityManager] = None
) -> float:
    """Calculate confidence score with robust error recovery.
    
    Args:
        logits: Model output logits
        generated_ids: Generated token IDs
        state: Current SOVL state
        error_manager: Error handling manager
        context: System context
        curiosity_manager: Optional curiosity manager
        
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    global _confidence_calculator
    if _confidence_calculator is None:
        _confidence_calculator = ConfidenceCalculator(state.config_manager, state.logger)
        
    return _confidence_calculator.calculate_confidence_score(
        logits=logits,
        generated_ids=generated_ids,
        state=state,
        error_manager=error_manager,
        context=context,
        curiosity_manager=curiosity_manager
    )
