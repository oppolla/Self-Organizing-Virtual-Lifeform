import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Union, List, Optional, Dict, Any, Tuple, Set, Deque
import torch
import traceback
import re
from dataclasses import dataclass
from sovl_utils import NumericalGuard, safe_divide, synchronized
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_records import ConfidenceHistory
from transformers import PreTrainedTokenizer, LogitsProcessor
from sovl_confidence import ConfidenceCalculator, SystemContext, CuriosityManager
from sovl_error import ErrorManager, ErrorRecord, ConfigurationError, StateError

# Placeholder for SOVLState until we can properly import it
try:
    from sovl_state import SOVLState
except ImportError:
    SOVLState = Any  # Fallback type

class LogitsError(Exception):
    """Custom exception for logits processing failures."""
    pass

class EventType(Enum):
    """Enum for logging event types."""
    PROCESSOR_INIT = "processor_init"
    CONFIDENCE_CALC = "confidence_calculation"
    PROCESSOR_TUNE = "processor_tune"
    PROCESSOR_LOAD_STATE = "processor_load_state"
    PROCESSOR_RESET = "processor_reset"
    TOKEN_MAP_UPDATE = "token_map_updated"
    ERROR = "error"
    WARNING = "warning"

@dataclass
class ProcessorConfig:
    """Configuration for SOVLProcessor with validation."""
    flat_distribution_confidence: float = 0.2
    confidence_var_threshold: float = 1e-5
    confidence_smoothing_factor: float = 0.0
    max_confidence_history: int = 10
    # Repetition detection configuration
    min_rep_length: int = 3
    max_rep_scan: int = 100
    rep_confidence_penalty: float = 0.3
    enable_rep_detection: bool = True

    def __post_init__(self):
        """Validate after initialization."""
        self._validate_config()

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ProcessorConfig':
        """Create a ProcessorConfig instance from ConfigManager."""
        processor_config = config_manager.get_section("processor_config", {})
        instance = cls(
            flat_distribution_confidence=processor_config.get("flat_distribution_confidence", 0.2),
            confidence_var_threshold=processor_config.get("confidence_var_threshold", 1e-5),
            confidence_smoothing_factor=processor_config.get("confidence_smoothing_factor", 0.0),
            max_confidence_history=processor_config.get("max_confidence_history", 10),
            min_rep_length=processor_config.get("min_rep_length", 3),
            max_rep_scan=processor_config.get("max_rep_scan", 100),
            rep_confidence_penalty=processor_config.get("rep_confidence_penalty", 0.3),
            enable_rep_detection=processor_config.get("enable_rep_detection", True)
        )
        return instance

    def _validate_config(self) -> None:
        """Validate all configuration parameters."""
        try:
            # Validate confidence parameters
            if not (0.0 <= self.flat_distribution_confidence <= 0.5):
                raise ValueError(f"flat_distribution_confidence must be in [0.0, 0.5], got {self.flat_distribution_confidence}")
            if not (1e-6 <= self.confidence_var_threshold <= 1e-4):
                raise ValueError(f"confidence_var_threshold must be in [1e-6, 1e-4], got {self.confidence_var_threshold}")
            if not (0.0 <= self.confidence_smoothing_factor <= 1.0):
                raise ValueError(f"confidence_smoothing_factor must be in [0.0, 1.0], got {self.confidence_smoothing_factor}")
            if not (5 <= self.max_confidence_history <= 20):
                raise ValueError(f"max_confidence_history must be in [5, 20], got {self.max_confidence_history}")

            # Validate repetition detection parameters
            if not (2 <= self.min_rep_length <= 10):
                raise ValueError(f"min_rep_length must be in [2, 10], got {self.min_rep_length}")
            if not (50 <= self.max_rep_scan <= 200):
                raise ValueError(f"max_rep_scan must be in [50, 200], got {self.max_rep_scan}")
            if not (0.0 <= self.rep_confidence_penalty <= 1.0):
                raise ValueError(f"rep_confidence_penalty must be in [0.0, 1.0], got {self.rep_confidence_penalty}")
            if not isinstance(self.enable_rep_detection, bool):
                raise ValueError(f"enable_rep_detection must be boolean, got {type(self.enable_rep_detection)}")

        except ValueError as e:
            raise ConfigurationError(f"Invalid processor configuration: {str(e)}")

    def update(self, config_manager: ConfigManager, **kwargs) -> None:
        """Update configuration parameters with validation."""
        validated_args = {}
        try:
            # Create a temporary copy for validation
            temp_config = ProcessorConfig(**vars(self))
            
            # Apply and validate each change
            for key, value in kwargs.items():
                if hasattr(temp_config, key):
                    setattr(temp_config, key, value)
                    temp_config._validate_config()  # Validate after each change
                    validated_args[key] = value
                else:
                    raise ValueError(f"Unknown configuration parameter: {key}")

            # If all validations passed, update local config
            for key, value in validated_args.items():
                setattr(self, key, value)

            # Update config in ConfigManager
            processor_config = config_manager.get_section("processor_config", {})
            processor_config.update(validated_args)
            config_manager.update_section("processor_config", processor_config)

        except (ValueError, ConfigurationError) as e:
            raise ConfigurationError(f"Invalid processor configuration update: {str(e)}")


class TensorValidator:
    """Handles tensor validation and conversion for logits and generated IDs."""
    
    def __init__(self, device: torch.device, logger: Logger):
        self.device = device
        self.logger = logger
        self._valid_dtypes = (torch.float16, torch.float32, torch.float64)
        self._valid_dims = (2, 3)
        self._clamp_range = (-100.0, 100.0)  # Reasonable range for logits

    def validate_logits(self, logits: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Convert and validate logits to a 3D tensor (batch, seq_len, vocab_size).

        Args:
            logits: Input logits, single tensor or list of tensors.

        Returns:
            Validated 3D tensor on the specified device.

        Raises:
            LogitsError: If validation fails.
        """
        try:
            # Fast path for common case: single tensor on correct device
            if isinstance(logits, torch.Tensor) and logits.device == self.device:
                if logits.dim() in self._valid_dims and logits.dtype in self._valid_dtypes:
                    return self._handle_nan_inf(logits)
                
            # Handle list of tensors
            if isinstance(logits, list):
                logits = torch.stack(logits)
                
            # Type check
            if not isinstance(logits, torch.Tensor):
                raise LogitsError(f"Expected tensor/list, got {type(logits)}")
                
            # Dimension check
            if logits.dim() not in self._valid_dims:
                raise LogitsError(f"Logits must be 2D or 3D, got {logits.dim()}D")
                
            # Dtype check
            if logits.dtype not in self._valid_dtypes:
                raise LogitsError(f"Logits must be float type, got {logits.dtype}")
                
            # Move to device and handle NaN/Inf
            logits = logits.to(self.device)
            return self._handle_nan_inf(logits)
            
        except Exception as e:
            self._log_error("Logits validation failed", str(e), logits=logits)
            raise LogitsError(f"Logits validation failed: {str(e)}")

    def _handle_nan_inf(self, logits: torch.Tensor) -> torch.Tensor:
        """Handle NaN and Inf values in logits with logging and fallback strategy."""
        if not torch.isfinite(logits).all():
            # Count problematic values
            nan_count = torch.isnan(logits).sum().item()
            inf_count = torch.isinf(logits).sum().item()
            total_elements = logits.numel()
            
            # Log the issue
            self.logger.record({
                "event": EventType.WARNING.value,
                "message": "Logits contain NaN/Inf values",
                "timestamp": time.time(),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "total_elements": total_elements,
                "nan_percentage": (nan_count / total_elements) * 100,
                "inf_percentage": (inf_count / total_elements) * 100
            })
            
            # Apply fallback strategy: clamp values and replace NaN with 0
            logits = torch.nan_to_num(logits, nan=0.0, posinf=self._clamp_range[1], neginf=self._clamp_range[0])
            logits = torch.clamp(logits, *self._clamp_range)
            
        return logits

    def validate_generated_ids(self, generated_ids: Optional[torch.Tensor], logits: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Validate generated IDs against logits.

        Args:
            generated_ids: Optional tensor of generated IDs (batch, seq_len).
            logits: Reference logits tensor.

        Returns:
            Validated generated IDs tensor or None.

        Raises:
            LogitsError: If validation fails.
        """
        if generated_ids is None:
            return None
            
        try:
            # Fast path for common case: tensor on correct device with matching shape
            if (isinstance(generated_ids, torch.Tensor) and 
                generated_ids.device == self.device and 
                generated_ids.dtype == torch.long and
                generated_ids.dim() == 2 and 
                generated_ids.shape[:2] == logits.shape[:2]):
                return generated_ids
                
            # Basic validation
            if not isinstance(generated_ids, torch.Tensor) or generated_ids.dtype != torch.long:
                raise LogitsError("Generated IDs must be LongTensor")
                
            if generated_ids.dim() != 2 or generated_ids.shape[:2] != logits.shape[:2]:
                raise LogitsError("Generated IDs shape mismatch with logits")
                
            return generated_ids.to(self.device)
            
        except Exception as e:
            self._log_error(
                "Generated IDs validation failed", str(e),
                generated_ids=generated_ids, logits=logits
            )
            raise LogitsError(f"Generated IDs validation failed: {str(e)}")

    def _log_error(self, message: str, error: str, **kwargs) -> None:
        """Log validation errors with context."""
        self.logger.log_error(
            error_msg=f"{message}: {error}",
            error_type="validation_error",
            stack_trace=traceback.format_exc(),
            additional_info={f"{key}_shape": str(getattr(value, 'shape', 'N/A')) for key, value in kwargs.items()}
        )


class SOVLProcessor:
    """Processes and manages the SOVL system state."""
    
    # Constants for adjustments
    TEMPERAMENT_SCALE: float = 0.1
    CURIOSITY_BOOST: float = 0.05
    MAX_CONFIDENCE: float = 1.0
    MIN_CONFIDENCE: float = 0.0
    PADDING_ID: int = -100
    DEFAULT_SPECIAL_IDS: Set[int] = {-100, 0, 1, 2, 3}  # Example default special IDs

    def __init__(self, config_manager: Optional[ConfigManager] = None, logger: Optional[Logger] = None, device: Optional[torch.device] = None):
        """
        Initialize the SOVL processor.

        Args:
            config_manager: Configuration manager instance. If None, will be created.
            logger: Logger for event recording. If None, will be created.
            device: Device for tensor operations. If None, will use default device.
        """
        # Initialize fundamental dependencies
        self.config_manager = config_manager or ConfigManager()
        self.logger = logger or Logger()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize state attributes
        self.token_map: Dict[int, int] = {}
        self.scaffold_unk_id: int = 0
        self.config: Optional[ProcessorConfig] = None
        
        # Initialize components
        self._lock = Lock()
        self._validator = TensorValidator(self.device, self.logger)
        self.confidence_calculator = ConfidenceCalculator(self.config_manager, self.logger)
        self._confidence_history = ConfidenceHistory(self.config_manager, self.logger)
        
        # Initialize message queue for curiosity parameters
        self._curiosity_queue = deque(maxlen=100)
        self._last_curiosity_update = 0.0
        self._curiosity_update_interval = 1.0  # seconds
        
        # Initialize error manager
        self._initialize_error_manager()
        
        # Set initial state using reset
        self.reset()

    def _initialize_error_manager(self) -> None:
        """Initialize error manager with processor-specific configuration."""
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,
            config_manager=self.config_manager,
            error_cooldown=1.0
        )
        
        # Register processor-specific thresholds
        self.error_manager.severity_thresholds.update({
            "tensor_validation": 3,  # 3 validation failures before critical
            "confidence_calc": 5,    # 5 calculation failures before critical
            "vibe_sculpting": 2,     # 2 vibe sculpting failures before critical
            "repetition_detection": 3 # 3 detection failures before critical
        })
        
        # Register recovery strategies
        self.error_manager.recovery_strategies.update({
            "tensor_validation_error": self._recover_tensor_validation,
            "confidence_calc_error": self._recover_confidence_calc,
            "vibe_sculpting_error": self._recover_vibe_sculpting,
            "repetition_detection_error": self._recover_repetition_detection
        })

    def _recover_tensor_validation(self, record: ErrorRecord) -> None:
        """Recovery strategy for tensor validation errors."""
        try:
            # Clear tensor caches
            torch.cuda.empty_cache()
            
            # Reset validator state
            self._validator = TensorValidator(self.device, self.logger)
            
            self.logger.record_event(
                "tensor_validation_recovery",
                "Reset tensor validator",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                "validation_recovery_failed",
                f"Failed to recover from tensor validation error: {str(e)}",
                level="error"
            )

    def _recover_confidence_calc(self, record: ErrorRecord) -> None:
        """Recovery strategy for confidence calculation errors."""
        try:
            # Reset confidence calculator
            self.confidence_calculator = ConfidenceCalculator(self.config_manager, self.logger)
            
            # Clear confidence history
            self._confidence_history.clear_history()
            
            self.logger.record_event(
                "confidence_calc_recovery",
                "Reset confidence calculator",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                "confidence_recovery_failed",
                f"Failed to recover from confidence calculation error: {str(e)}",
                level="error"
            )

    def _recover_vibe_sculpting(self, record: ErrorRecord) -> None:
        """Recovery strategy for vibe sculpting errors."""
        try:
            # Reset vibe history
            self._curiosity_queue.clear()
            
            # Reset last update timestamp
            self._last_curiosity_update = 0.0
            
            self.logger.record_event(
                "vibe_sculpting_recovery",
                "Reset vibe sculpting state",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                "vibe_recovery_failed",
                f"Failed to recover from vibe sculpting error: {str(e)}",
                level="error"
            )

    def _recover_repetition_detection(self, record: ErrorRecord) -> None:
        """Recovery strategy for repetition detection errors."""
        try:
            # Reset repetition detection parameters to defaults
            self.config = ProcessorConfig.from_config_manager(self.config_manager)
            
            self.logger.record_event(
                "repetition_detection_recovery",
                "Reset repetition detection parameters",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                "repetition_recovery_failed",
                f"Failed to recover from repetition detection error: {str(e)}",
                level="error"
            )

    def _log_init(self) -> None:
        """Log initialization event."""
        if self.config is None:
            self.logger.log_training_event(
                event_type="processor_init_warning",
                message="Processor logging init before full reset completion.",
                level="warning",
            )
            return

        self.logger.log_training_event(
            event_type="processor_initialized",
            message="Processor initialized/reset successfully",
            level="info",
            additional_info={
                "config": vars(self.config),
                "device": str(self.device),
                "token_mapping": {
                    "scaffold_unk_id": self.scaffold_unk_id,
                    "token_map_size": len(self.token_map)
                }
            }
        )

    def update_curiosity_params(self, pressure: float, timestamp: float) -> None:
        """Update curiosity parameters in the queue."""
        with self._lock:
            self._curiosity_queue.append((pressure, timestamp))
            self._last_curiosity_update = timestamp
            
            # Log curiosity update
            self.logger.log_event(
                "curiosity_params_updated",
                {
                    "pressure": pressure,
                    "timestamp": timestamp
                }
            )

    def _get_curiosity_params(self, timestamp: Optional[float] = None) -> Optional[float]:
        """Get current curiosity parameters from queue."""
        with self._lock:
            if not self._curiosity_queue:
                return None
                
            if timestamp is None or timestamp - self._last_curiosity_update >= self._curiosity_update_interval:
                # Get most recent pressure
                pressure, _ = self._curiosity_queue[-1]
                return pressure
                
            return None

    def validate_logits(self, logits: torch.Tensor) -> bool:
        """
        Validate logits tensor using the validator.

        Args:
            logits: Logits tensor to validate.

        Returns:
            bool: True if validation passes, False otherwise.
        """
        try:
            # Use the validator to validate logits
            validated_logits = self._validator.validate_logits(logits)
            # If no exception was raised, validation passed
            return True
        except LogitsError as e:
            self.error_manager.record_error(
                "tensor_validation_error",
                f"Logits tensor validation failed: {str(e)}",
                severity=2,
                context={"tensor_shape": getattr(logits, 'shape', 'N/A'), "error": str(e)}
            )
            return False
        except Exception as e:
            self.error_manager.record_error(
                "tensor_validation_error",
                f"Unexpected error during logits validation: {str(e)}",
                severity=3,
                context={"error": str(e)}
            )
            return False

    def calculate_confidence(self, logits: torch.Tensor) -> float:
        """
        Calculate confidence score for the given logits.

        Args:
            logits: Logits tensor to calculate confidence for.

        Returns:
            float: Confidence score between 0 and 1.
        """
        try:
            # Validate logits first
            if not self.validate_logits(logits):
                return self.MIN_CONFIDENCE

            # Calculate confidence
            confidence = self.confidence_calculator.calculate_confidence(logits)
            
            # Validate confidence value
            if not (self.MIN_CONFIDENCE <= confidence <= self.MAX_CONFIDENCE):
                self.error_manager.record_error(
                    "confidence_calc_error",
                    f"Invalid confidence value: {confidence}",
                    severity=2,
                    context={"confidence": confidence}
                )
                return self.MIN_CONFIDENCE

            return confidence
        except Exception as e:
            self.error_manager.record_error(
                "confidence_calc_error",
                f"Error calculating confidence: {str(e)}",
                severity=3,
                context={"error": str(e)}
            )
            return self.MIN_CONFIDENCE

    def get_confidence_history(self) -> Deque[float]:
        """Get the confidence history."""
        return self._confidence_history.get_confidence_history()

    def clear_confidence_history(self) -> None:
        """Clear the confidence history."""
        self._confidence_history.clear_history()

    def get_state(self) -> Dict[str, Any]:
        """Export processor state."""
        with self._lock:
            return {
                "config": vars(self.config),
                "token_mapping": {
                    "scaffold_unk_id": self.scaffold_unk_id,
                    "token_map": self.token_map
                },
                "confidence_history": self._confidence_history.to_dict()
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load processor state.

        Args:
            state: State dictionary.
        """
        try:
            with self._lock:
                # Basic type checking for loaded state
                if "config" in state:
                    if not isinstance(state["config"], dict):
                        raise TypeError("Expected 'config' to be a dict in loaded state")
                    self.config.update(self.config_manager, **state["config"])
                
                if "token_mapping" in state:
                    if not isinstance(state["token_mapping"], dict):
                        raise TypeError("Expected 'token_mapping' to be a dict in loaded state")
                    self.scaffold_unk_id = state["token_mapping"].get("scaffold_unk_id", 0)
                    self.token_map = state["token_mapping"].get("token_map", {})
                    if not isinstance(self.token_map, dict):
                        raise TypeError("Expected 'token_map' within 'token_mapping' to be a dict")

                if "confidence_history" in state:
                    if not isinstance(state["confidence_history"], dict):
                        raise TypeError("Expected 'confidence_history' to be a dict in loaded state")
                    # Assuming ConfidenceHistory handles its own loading validation
                    self._confidence_history.from_dict(state["confidence_history"])
                
                self.logger.log_training_event(
                    event_type="state_loaded",
                    message="Processor state loaded",
                    level="info",
                    additional_info={"loaded_keys": list(state.keys())} # Avoid logging potentially large state
                )
        except Exception as e:
            self.logger.log_error(
                error_type="state_error",
                message=f"Failed to load processor state: {str(e)}",
                error=str(e),
                stack_trace=traceback.format_exc()
            )
            # Reset to a known good state on load failure
            self.reset()
            self.logger.log_training_event(
                event_type="state_load_failed_reset",
                message="Processor reset due to state load failure.",
                level="warning"
            )

    def reset(self) -> None:
        """Reset processor state to defaults."""
        with self._lock:
            # Reset configuration
            self.config = ProcessorConfig.from_config_manager(self.config_manager)

            # Reset token mapping
            self.scaffold_unk_id = 0
            self.token_map = {}

            # Reset confidence history
            self._confidence_history.clear_history()

            # Reset validator and confidence calculator
            self._validator = TensorValidator(self.device, self.logger)
            self.confidence_calculator = ConfidenceCalculator(self.config_manager, self.logger)

            # Reset curiosity queue
            self._curiosity_queue.clear()
            self._last_curiosity_update = 0.0

            # Re-initialize error manager
            self._initialize_error_manager()

            self.logger.log_training_event(
                event_type="processor_reset",
                message="Processor state reset",
                level="info"
            )
            # Log init state after reset is complete
            self._log_init()

    def detect_repetitions(
        self,
        token_ids: Union[List[int], torch.Tensor],
        special_ids: Optional[Set[int]] = None,
        min_rep_length: Optional[int] = None,
        max_scan: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Detect repeating token sequences with optimized batch processing.
        
        Args:
            token_ids: List of token IDs or tensor of shape (batch_size, seq_len)
            special_ids: Set of special token IDs to ignore. If None, uses defaults.
            min_rep_length: Minimum sequence length to check
            max_scan: Maximum number of tokens to scan
            batch_size: Optional batch size for processing
            
        Returns:
            (start_idx, end_idx) of first repetition found or None
        """
        if not self.config.enable_rep_detection:
            return None

        try:
            with self._lock:
                # Use config values if not specified
                min_rep_length = min_rep_length if min_rep_length is not None else self.config.min_rep_length
                max_scan = max_scan if max_scan is not None else self.config.max_rep_scan
                special_ids = special_ids if special_ids is not None else self.DEFAULT_SPECIAL_IDS

                # Handle empty input
                if isinstance(token_ids, list) and not token_ids:
                    return None
                if isinstance(token_ids, torch.Tensor) and token_ids.numel() == 0:
                    return None

                # Convert to tensor if needed
                input_tensor: torch.Tensor
                if isinstance(token_ids, list):
                    input_tensor = torch.tensor(token_ids, device=self.device, dtype=torch.long)
                elif isinstance(token_ids, torch.Tensor):
                    input_tensor = token_ids.to(self.device)
                else:
                    raise TypeError(f"Unsupported type for token_ids: {type(token_ids)}")

                # Ensure tensor is at least 1D
                if input_tensor.dim() == 0:
                    input_tensor = input_tensor.unsqueeze(0)
                
                # Handle batch processing if input is 2D
                if input_tensor.dim() == 2:
                    effective_batch_size = batch_size if batch_size is not None else input_tensor.size(0)
                    
                    for i in range(0, input_tensor.size(0), effective_batch_size):
                        batch = input_tensor[i:i + effective_batch_size]
                        result = self._detect_repetitions_batch(
                            batch, special_ids, min_rep_length, max_scan
                        )
                        if result is not None:
                            return result
                    return None
                
                # Single sequence processing
                return self._detect_repetitions_single(
                    input_tensor, special_ids, min_rep_length, max_scan
                )
                
        except Exception as e:
            self.error_manager.record_error(
                "repetition_detection_error",
                f"Repetition detection failed: {str(e)}",
                severity=2,
                context={
                    "token_ids_shape": str(getattr(token_ids, 'shape', 'N/A')),
                    "min_rep_length": min_rep_length,
                    "max_scan": max_scan
                }
            )
            return None

    def _detect_repetitions_batch(
        self,
        token_ids_batch: torch.Tensor,  # Should be 2D (batch, seq_len)
        special_ids: Set[int],
        min_rep_length: int,
        max_scan: int
    ) -> Optional[Tuple[int, int]]:
        """Detect repetitions in a batch of sequences."""
        for i in range(token_ids_batch.size(0)):
            sequence_tensor = token_ids_batch[i]
            result = self._detect_repetitions_single(
                sequence_tensor, special_ids, min_rep_length, max_scan
            )
            if result is not None:
                return result
        return None

    def _detect_repetitions_single(
        self,
        token_ids: torch.Tensor,  # Should be 1D
        special_ids: Set[int],
        min_rep_length: int,
        max_scan: int
    ) -> Optional[Tuple[int, int]]:
        """Detect repetitions in a single sequence."""
        if token_ids.numel() < 2 * min_rep_length:
            return None

        # Convert to list for processing, filtering special IDs
        ids_list = token_ids.tolist()
        filtered_ids = [i for i in ids_list if i not in special_ids]
        
        if len(filtered_ids) < 2 * min_rep_length:
            return None

        scan_len = min(len(filtered_ids), max_scan)
        
        # Check for repetitions using a sliding window approach
        for i in range(scan_len - min_rep_length):
            # Check if the sequence starting at i repeats immediately after
            idx_end_first = i + min_rep_length
            idx_end_second = idx_end_first + min_rep_length

            # Ensure we don't go out of bounds
            if idx_end_second > len(filtered_ids):
                break

            seq1 = filtered_ids[i:idx_end_first]
            seq2 = filtered_ids[idx_end_first:idx_end_second]

            if seq1 == seq2:
                # Repetition detected
                self.logger.record({
                    "event": EventType.WARNING.value,
                    "message": "Repetition detected (based on filtered tokens)",
                    "filtered_start_idx": i,
                    "filtered_end_idx": idx_end_first,
                    "length": min_rep_length,
                    "timestamp": time.time()
                })
                return (i, idx_end_first)
        
        return None

    def detect_repetition_in_sequence(self, token_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Check if a sequence contains any repetitions.
        This is a convenience method that uses detect_repetitions.

        Args:
            token_ids: List of token IDs or tensor to check for repetitions.

        Returns:
            bool: True if repetition is detected, False otherwise.
        """
        if not self.config.enable_rep_detection:
            return False

        try:
            repetition_found = self.detect_repetitions(token_ids=token_ids)
            
            if repetition_found is not None:
                self.logger.record_event(
                    "repetition_detected",
                    "Repetition detected in sequence",
                    level="warning",
                    additional_info={"detected_indices": repetition_found}
                )
            
            return repetition_found is not None

        except Exception as e:
            self.error_manager.record_error(
                "repetition_detection_error",
                f"Error detecting repetition in sequence: {str(e)}",
                severity=3,
                context={"error": str(e)}
            )
            return False

    def vibe_sculpt(self, logits: torch.Tensor, state: SOVLState) -> torch.Tensor:
        """
        Apply vibe sculpting to the logits.

        Args:
            logits: Input logits tensor.
            state: Current SOVL state.

        Returns:
            torch.Tensor: Sculpted logits.
        """
        try:
            # Validate logits first
            if not self.validate_logits(logits):
                return logits

            # Apply vibe sculpting
            sculpted_logits = self.vibe_sculptor.sculpt(logits, state)
            
            # Validate sculpted logits
            if not self.validate_logits(sculpted_logits):
                self.error_manager.record_error(
                    "vibe_sculpt_error",
                    "Invalid sculpted logits",
                    severity=2,
                    context={"original_shape": logits.shape, "sculpted_shape": sculpted_logits.shape}
                )
                return logits

            return sculpted_logits
        except Exception as e:
            self.error_manager.record_error(
                "vibe_sculpt_error",
                f"Error during vibe sculpting: {str(e)}",
                severity=3,
                context={"error": str(e)}
            )
            return logits

    def detect_repetition(self, logits: torch.Tensor, state: SOVLState) -> bool:
        """
        Detect repetition in the logits.

        Args:
            logits: Input logits tensor.
            state: Current SOVL state.

        Returns:
            bool: True if repetition is detected, False otherwise.
        """
        try:
            # Validate logits first
            if not self.validate_logits(logits):
                return False

            # Check for repetition
            is_repetitive = self.repetition_detector.detect(logits, state)
            
            if is_repetitive:
                self.error_manager.record_error(
                    "repetition_detected",
                    "Repetition detected in logits",
                    severity=1,
                    context={"logits_shape": logits.shape}
                )

            return is_repetitive
        except Exception as e:
            self.error_manager.record_error(
                "repetition_detection_error",
                f"Error detecting repetition: {str(e)}",
                severity=3,
                context={"error": str(e)}
            )
            return False

class SoulLogitsProcessor(LogitsProcessor):
    """Boosts token probabilities for .soul file keywords during generation.

    Args:
        soul_keywords: Dictionary mapping keywords to their boost weights.
        tokenizer: Tokenizer for encoding keywords.
        logger: Logger for error reporting.
    """
    
    def __init__(self, soul_keywords: Dict[str, float], tokenizer: PreTrainedTokenizer, logger: Logger):
        self.soul_keywords = soul_keywords
        self.tokenizer = tokenizer
        self.logger = logger

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply hypersensitive boost to token probabilities for .soul keywords.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            scores: Logits scores (batch_size, vocab_size).

        Returns:
            Modified scores with boosted probabilities.
        """
        try:
            for keyword, weight in self.soul_keywords.items():
                token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
                for token_id in token_ids:
                    scores[:, token_id] += weight * 2.0  # Hypersensitive boost
            return scores
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to apply soul logits processing: {str(e)}",
                error_type="soul_logits_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "keywords": self.soul_keywords,
                    "input_ids_shape": str(input_ids.shape),
                    "scores_shape": str(scores.shape)
                }
            )
            return scores

class VibeSculptor:
    """Sculpts conversational vibes as dynamic, empathetic fingerprints."""

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        temperament_system: Optional['TemperamentSystem'] = None,
        lifecycle_manager: Optional['LifecycleManager'] = None
    ):
        """Initialize with config, logger, and optional system components."""
        if not config_manager or not logger:
            raise ValueError("config_manager and logger cannot be None")
        self.config_manager = config_manager
        self.logger = logger
        self.temperament_system = temperament_system
        self.lifecycle_manager = lifecycle_manager
        self.vibes = deque(maxlen=self._get_config("history_maxlen", 20))
        self._load_config()

    def _load_config(self) -> None:
        """Load vibe configuration elegantly."""
        try:
            vibe_config = self.config_manager.get_section("vibe_config", {})
            self.default_vibe = vibe_config.get("default_vibe_score", 0.5)
            self.min_vibe = vibe_config.get("min_vibe_score", 0.0)
            self.max_vibe = vibe_config.get("max_vibe_score", 1.0)
            self.switch_threshold = vibe_config.get("switch_threshold", 0.3)
            self.decay_factor = vibe_config.get("decay_factor", 0.9)
            self.weights = {
                "energy": vibe_config.get("energy_weight", 0.25),  # Lexical + sentiment
                "flow": vibe_config.get("flow_weight", 0.25),      # Syntactic + rhythm
                "resonance": vibe_config.get("resonance_weight", 0.25),  # Topic + temperament
                "curiosity": vibe_config.get("curiosity_weight", 0.25)   # Curiosity synergy
            }
            if abs(sum(self.weights.values()) - 1.0) > 1e-6:
                raise ValueError("Vibe weights must sum to 1.0")
            self.logger.record_event(
                event_type="vibe_config_loaded",
                message="Vibe sculptor configured",
                level="info",
                additional_info={"weights": self.weights}
            )
        except Exception as e:
            self.logger.record_event(
                event_type="vibe_config_failed",
                message=f"Vibe config failed: {str(e)}",
                level="error"
            )
            raise StateError(f"Vibe config failed: {str(e)}")

    def _get_config(self, key: str, default: Any) -> Any:
        """Helper to get vibe config values."""
        return self.config_manager.get(f"vibe_config.{key}", default)

    def _compute_energy(self, text: str) -> float:
        """Measure lexical diversity and sentiment as conversational energy."""
        words = re.findall(r'\w+', text.lower())
        if not words:
            return 0.5
        # Lexical diversity (TTR)
        ttr = len(set(words)) / len(words)
        # Sentiment (simple rule-based)
        pos_words = {'good', 'great', 'happy', 'awesome', 'love'}
        neg_words = {'bad', 'sad', 'hate', 'terrible', 'awful'}
        pos_count = len(set(words) & pos_words)
        neg_count = len(set(words) & neg_words)
        sentiment = pos_count / (pos_count + neg_count) if pos_count + neg_count else 0.5
        return 0.6 * ttr + 0.4 * sentiment

    def _compute_flow(self, text: str, profile: Dict) -> float:
        """Measure syntactic complexity and interaction rhythm as conversational flow."""
        # Syntactic complexity (sentence length)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_length = sum(len(re.findall(r'\w+', s)) for s in sentences) / len(sentences) if sentences else 0.0
        syntax = min(avg_length / 20.0, 1.0)
        # Rhythm (input frequency and length)
        inputs = profile.get("inputs", deque())
        rhythm = min(len(inputs) / 10.0, 1.0) * 0.5 + min(sum(len(i) for i in inputs) / (200.0 * len(inputs) or 1), 1.0) * 0.5
        return 0.5 * syntax + 0.5 * rhythm

    def _compute_resonance(self, text: str, state: SOVLState) -> float:
        """Measure topic consistency and temperament alignment as vibe resonance."""
        # Topic consistency (Jaccard with profile inputs)
        profile = state.user_profile_state.get(state.history.conversation_id)
        inputs = profile.get("inputs", deque())
        text_words = set(re.findall(r'\w+', text.lower()))
        topic = sum(
            len(text_words & set(re.findall(r'\w+', h.lower()))) /
            len(text_words | set(re.findall(r'\w+', h.lower()))) if text_words and h else 0.5
            for h in inputs
        ) / len(inputs) if inputs else 0.5
        # Temperament alignment
        temperament_score = self.temperament_system.get_temperament_score() if self.temperament_system else 0.5
        user_mood = self._compute_energy(text)  # Proxy for user mood
        alignment = 1.0 - abs(temperament_score - user_mood)
        return 0.5 * topic + 0.5 * alignment

    def _compute_curiosity(self, text: str, curiosity_manager: Optional[CuriosityManager]) -> float:
        """Measure engagement with curiosity-driven questions."""
        if not curiosity_manager:
            return 0.5
        novelty = curiosity_manager.get_novelty_score(text)
        return min(novelty / 0.7, 1.0)  # Normalize based on typical novelty threshold

    @synchronized()
    def sculpt_vibe(
        self,
        user_input: str,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None
    ) -> float:
        """Sculpt a vibe score that resonates with user and system energy."""
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")
            conversation_id = state.history.conversation_id
            profile = state.user_profile_state.get(conversation_id)
            state.user_profile_state.update(conversation_id, user_input, state.session_start)

            # Calculate vibe components with temporal decay
            now = time.time()
            decay = self.decay_factor ** ((now - profile.get("last_interaction", now)) / 86400.0)
            energy = self._compute_energy(user_input) * decay
            flow = self._compute_flow(user_input, profile) * decay
            resonance = self._compute_resonance(user_input, state) * decay
            curiosity = self._compute_curiosity(user_input, curiosity_manager) * decay

            # Combine with lifecycle influence
            lifecycle_factor = self.lifecycle_manager.get_lifecycle_factor() if self.lifecycle_manager else 1.0
            vibe = (
                self.weights["energy"] * energy +
                self.weights["flow"] * flow +
                self.weights["resonance"] * resonance +
                self.weights["curiosity"] * curiosity
            ) * lifecycle_factor

            vibe = max(self.min_vibe, min(self.max_vibe, vibe))
            self.vibes.append((vibe, now))
            self.logger.record_event(
                event_type="vibe_sculpted",
                message="Vibe score sculpted",
                level="info",
                additional_info={
                    "vibe": vibe,
                    "energy": energy,
                    "flow": flow,
                    "resonance": resonance,
                    "curiosity": curiosity,
                    "conversation_id": conversation_id
                }
            )
            return vibe

        except Exception as e:
            self.logger.record_event(
                event_type="vibe_sculpt_failed",
                message=f"Vibe sculpting failed: {str(e)}",
                level="error"
            )
            error_manager.handle_data_error(e, {"user_input": user_input[:50]}, conversation_id)
            return self.default_vibe

    def predict_vibe_shift(self, vibe: float) -> bool:
        """Predict if a vibe shift is occurring based on recent trends."""
        if len(self.vibes) < 3:
            return False
        recent_vibes = [v for v, _ in self.vibes]
        avg_vibe = sum(recent_vibes) / len(recent_vibes)
        deviation = abs(vibe - avg_vibe)
        if deviation > self.switch_threshold:
            self.logger.record_event(
                event_type="vibe_shift_detected",
                message="Potential vibe shift detected",
                level="warning",
                additional_info={"vibe": vibe, "avg_vibe": avg_vibe, "deviation": deviation}
            )
            return True
        return False

    def get_vibe_aura(self, state: SOVLState) -> Dict[str, float]:
        """Generate a vibe 'aura' for visualization."""
        profile = state.user_profile_state.get(state.history.conversation_id)
        vibe_scores = [v for v, _ in self.vibes]
        return {
            "energy": self._compute_energy(" ".join(profile.get("inputs", []))),
            "flow": self._compute_flow("", profile),
            "resonance": self._compute_resonance(" ".join(profile.get("inputs", [])), state),
            "curiosity": vibe_scores[-1] if vibe_scores else 0.5,
            "trend": (vibe_scores[-1] - vibe_scores[0]) / (len(self.vibes) - 1) if len(self.vibes) > 1 else 0.0
        }
    
