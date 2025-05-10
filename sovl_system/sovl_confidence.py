from typing import Optional, Dict, Any, List, Tuple
import torch
from threading import Lock
from collections import deque
from sovl_logger import Logger
import traceback
from sovl_state import StateManager
from sovl_error import ErrorManager
from sovl_main import SystemContext
from sovl_utils import synchronized, NumericalGuard
from sovl_config import ConfigManager
import time
import threading
import math
from torch import nn
from sovl_error import ErrorManager

# Constants
DEFAULT_CONFIDENCE = 0.5
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_HISTORY_LENGTH = 3

"""
Confidence calculation module for the SOVL system.

This module provides functionality to calculate confidence scores for model outputs,

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
        state_manager: StateManager
    ):
        """Initialize the confidence calculator with configuration and logging.
        
        Args:
            config_manager: ConfigManager instance for configuration handling
            logger: Logger instance for logging
            state_manager: StateManager for atomic state updates
            
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
        self.state_manager = state_manager
        
        # Initialize configuration
        self._initialize_config()
        
        # Temperature scaling attribute (default 1.0)
        self._temperature = 1.0
        
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
                    "min_history_length": self.min_history_length
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
        error_manager: ErrorManager,
        context: SystemContext,
        user_id: str = "default",
        strategy: str = "blended",  # "classic", "experience", "entropy", "margin", "blended"
        top_k: int = 5,
        weights: Optional[dict] = None
    ) -> float:
        """Calculate confidence score with robust error recovery and thread safety.
        Supports multiple strategies: classic, experience, entropy, margin, blended.
        All state access is via StateManager.
        """
        with self.lock:
            try:
                state = self.state_manager.get_state()
                self.logger.record_event(
                    event_type="confidence_calculation_start",
                    message="Starting confidence calculation",
                    level="info",
                    additional_info={"state_id": getattr(state, 'id', None), "strategy": strategy}
                )
                # Validate inputs
                self.__validate_inputs(logits, generated_ids)
                # Calculate probabilities
                probs = self.__calculate_probabilities(logits)
                # 1. Base confidence (mean max prob)
                base_conf = probs.max(dim=-1).values.mean().item()
                # 2. Experience-based adjustment
                experience_factor = None
                experience_adj = base_conf
                if recaller is not None:
                    # Get embedding for current context (assume context has a 'text' or similar attribute)
                    context_text = getattr(context, 'text', None)
                    if context_text is not None and hasattr(recaller, 'embedding_fn') and hasattr(recaller, 'get_long_term_context'):
                        embedding = recaller.embedding_fn(context_text)
                        similar_msgs = recaller.get_long_term_context(user_id=user_id, query_embedding=embedding, top_k=top_k)
                        experience_factor = min(len(similar_msgs) / float(top_k), 1.0) if top_k > 0 else 0.0
                        experience_adj = 0.7 * base_conf + 0.3 * experience_factor
                # 3. Entropy-based confidence
                entropy = -(probs * probs.log()).sum(dim=-1)
                max_entropy = math.log(probs.size(-1))
                norm_entropy = entropy / max_entropy
                entropy_conf = 1 - norm_entropy.mean().item()
                # 4. Margin-based confidence
                top2 = torch.topk(probs, 2, dim=-1).values
                margin_conf = (top2[:, 0] - top2[:, 1]).mean().item()
                # 5. Blended confidence
                if weights is None:
                    weights = {"base": 0.5, "entropy": 0.25, "margin": 0.25, "experience": 0.3}
                if strategy == "classic":
                    confidence = base_conf
                elif strategy == "experience":
                    confidence = experience_adj
                elif strategy == "entropy":
                    confidence = entropy_conf
                elif strategy == "margin":
                    confidence = margin_conf
                elif strategy == "blended":
                    # If recaller is not provided, fallback to base_conf for experience_adj
                    confidence = (
                        weights.get("base", 0.5) * experience_adj +
                        weights.get("entropy", 0.25) * entropy_conf +
                        weights.get("margin", 0.25) * margin_conf
                    )
                else:
                    confidence = base_conf  # fallback
                # Clamp to valid range
                confidence = max(self.min_confidence, min(self.max_confidence, confidence))
                # Finalize and update history
                final_confidence = self.__finalize_confidence(confidence)
                self.logger.record_event(
                    event_type="confidence_calculation_success",
                    message="Confidence calculation completed",
                    level="info",
                    additional_info={
                        "final_confidence": final_confidence,
                        "base_conf": base_conf,
                        "experience_factor": experience_factor,
                        "experience_adj": experience_adj,
                        "entropy_conf": entropy_conf,
                        "margin_conf": margin_conf,
                        "strategy": strategy
                    }
                )
                return final_confidence
            except Exception as e:
                # Log error and attempt recovery
                state = self.state_manager.get_state()
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
                recovered_confidence = self.__recover_confidence(e, error_manager)
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
        """Calculate softmax probabilities from logits, with temperature scaling."""
        with NumericalGuard():
            return torch.softmax(logits / self._temperature, dim=-1)

    def __compute_base_confidence(self, probs: torch.Tensor) -> float:
        """Compute base confidence from probabilities.
        
        Args:
            probs: Softmax probabilities
        
        Returns:
            float: Base confidence score
        """
        max_probs = probs.max(dim=-1).values
        return max_probs.mean().item()

    def __finalize_confidence(self, confidence: float) -> float:
        """Finalize confidence score and update history atomically using StateManager."""
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

    def __recover_confidence(self, error: Exception, error_manager: ErrorManager) -> float:
        """Attempt to recover confidence from history or use default, with defensive checks and logging."""
        state = self.state_manager.get_state()
        try:
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
            recovered_confidence = sum(valid_confidences) / len(valid_confidences)
            recovered_confidence = max(self.min_confidence, min(self.max_confidence, recovered_confidence))
            error_manager.logger.record_event(
                event_type="partial_confidence_recovery",
                message=f"Recovered confidence using mean of {len(valid_confidences)} valid history entries",
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

    def set_temperature(self, temperature: float) -> None:
        """Set the temperature for temperature scaling."""
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self._temperature = float(temperature)

    def get_temperature(self) -> float:
        """Get the current temperature value."""
        return self._temperature

    def fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 1000, lr: float = 0.01, allow: bool = False) -> float:
        """
        Fit the temperature parameter on a validation set to minimize NLL.
        WARNING: This method is for offline/manual calibration only and should not be used in production or normal operation.
        By default, calling this method will raise a RuntimeError unless explicitly allowed by setting allow=True.
        Args:
            logits: [N, C] tensor of model logits
            labels: [N] tensor of true class indices
            max_iter: maximum optimization steps
            lr: learning rate for optimizer
            allow: must be True to enable fitting (default False)
        Returns:
            float: The optimized temperature value
        Raises:
            RuntimeError: if allow is not True
        """
        if not allow:
            raise RuntimeError(
                "fit_temperature is restricted to offline/manual calibration only. "
                "To run, call with allow=True during maintenance periods."
            )
        import torch.optim as optim
        import torch.nn.functional as F
        device = logits.device
        temperature = torch.ones(1, device=device, requires_grad=True) * self._temperature
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        labels = labels.to(device)
        
        def _nll():
            # Clamp temperature to avoid zero or negative
            temp = temperature.clamp(min=1e-3)
            scaled_logits = logits / temp
            loss = F.cross_entropy(scaled_logits, labels)
            return loss
        
        def closure():
            optimizer.zero_grad()
            loss = _nll()
            loss.backward()
            return loss
        optimizer.step(closure)
        self._temperature = float(temperature.clamp(min=1e-3).item())
        self.logger.record_event(
            event_type="temperature_fitted",
            message=f"Temperature fitted to {self._temperature}",
            level="info",
            additional_info={"temperature": self._temperature}
        )
        return self._temperature

# Singleton instance of ConfidenceCalculator
_confidence_calculator = None
_confidence_calculator_lock = Lock()

def calculate_confidence_score(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    error_manager: ErrorManager,
    context: SystemContext,
    state_manager: StateManager,
    user_id: str = "default",
    strategy: str = "blended",  # "classic", "experience", "entropy", "margin", "blended"
    top_k: int = 5,
    weights: Optional[dict] = None
) -> float:
    """Calculate confidence score with robust error recovery.
    All state access is via StateManager.
    """
    global _confidence_calculator
    with _confidence_calculator_lock:
        if _confidence_calculator is None:
            config_manager = state_manager.config_manager if hasattr(state_manager, 'config_manager') else None
            logger = state_manager.logger if hasattr(state_manager, 'logger') else None
            _confidence_calculator = ConfidenceCalculator(config_manager, logger, state_manager=state_manager)
    return _confidence_calculator.calculate_confidence_score(
        logits=logits,
        generated_ids=generated_ids,
        error_manager=error_manager,
        context=context,
        user_id=user_id,
        strategy=strategy,
        top_k=top_k,
        weights=weights
    )

def reset_confidence_calculator():
    """Reset the singleton ConfidenceCalculator instance (for testing or reconfiguration)."""
    global _confidence_calculator
    with _confidence_calculator_lock:
        _confidence_calculator = None
