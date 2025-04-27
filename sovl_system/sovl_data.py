from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import random
import time
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_error import ErrorManager
from sovl_io import InsufficientDataError
import traceback
import os
import json
from collections import defaultdict
from sovl_memory import RAMManager, GPUMemoryManager
from threading import Lock
from dataclasses import dataclass, field
from sovl_monitor import MemoryMonitor

@dataclass
class DataStats:
    """Tracks data loading and quality statistics."""
    total_entries: int = 0
    valid_entries: int = 0
    invalid_entries: int = 0
    last_load_time: float = 0.0
    average_entry_length: float = 0.0
    validation_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    data_quality_score: float = 0.0
    data_diversity_score: float = 0.0
    last_update_time: float = 0.0

    def update(self, total_entries: int, valid_entries: int, invalid_entries: int,
              validation_errors: Dict[str, int], average_entry_length: float) -> None:
        """Update data statistics."""
        try:
            self.total_entries = total_entries
            self.valid_entries = valid_entries
            self.invalid_entries = invalid_entries
            self.last_load_time = time.time()
            self.average_entry_length = average_entry_length
            self.validation_errors = validation_errors
            self.last_update_time = time.time()
            self.data_quality_score = valid_entries / total_entries if total_entries > 0 else 0.0
            self.data_diversity_score = min(1.0, average_entry_length / 1000.0)
        except Exception as e:
            import traceback
            print(f"[DataStats] Failed to update statistics: {e}\n{traceback.format_exc()}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_entries": self.total_entries, "valid_entries": self.valid_entries,
            "invalid_entries": self.invalid_entries, "last_load_time": self.last_load_time,
            "average_entry_length": self.average_entry_length, "validation_errors": dict(self.validation_errors),
            "data_quality_score": self.data_quality_score, "data_diversity_score": self.data_diversity_score,
            "last_update_time": self.last_update_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataStats':
        """Create from dictionary."""
        stats = cls()
        stats.total_entries = data.get("total_entries", 0)
        stats.valid_entries = data.get("valid_entries", 0)
        stats.invalid_entries = data.get("invalid_entries", 0)
        stats.last_load_time = data.get("last_load_time", 0.0)
        stats.average_entry_length = data.get("average_entry_length", 0.0)
        stats.validation_errors = defaultdict(int, data.get("validation_errors", {}))
        stats.data_quality_score = data.get("data_quality_score", 0.0)
        stats.data_diversity_score = data.get("data_diversity_score", 0.0)
        stats.last_update_time = data.get("last_update_time", 0.0)
        return stats

class DataProvider(ABC):
    """Abstract interface for data providers."""
    
    @abstractmethod
    def load_data(self, source: str, min_entries: int = 0) -> List[Dict[str, Any]]:
        """
        Load data from a specified source.
        
        Args:
            source: Identifier for the data source (e.g., file path, database URI).
            min_entries: Minimum number of entries required.
        
        Returns:
            List of data entries as dictionaries.
        
        Raises:
            InsufficientDataError: If not enough data is loaded.
            ValueError: If the source is invalid.
        """
        pass

    @abstractmethod
    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate the integrity of loaded data.
        
        Args:
            data: List of data entries to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        pass

class FileDataProvider(DataProvider):
    """Data provider for loading data from JSONL files."""
    
    # Define required fields and their validation rules
    REQUIRED_FIELDS = {
        "prompt": {
            "type": str,
            "min_length": 1,
            "max_length": 10000,
            "description": "Input prompt",
            "required": True
        },
        "response": {
            "type": str,
            "min_length": 1,
            "max_length": 10000,
            "description": "Model response",
            "required": True
        },
        "confidence_score": {
            "type": (int, float),
            "range": (0.0, 1.0),
            "description": "Confidence score",
            "required": False
        },
        "temperament_score": {
            "type": (int, float),
            "range": (0.0, 1.0),
            "description": "Temperament score",
            "required": False
        },
        "conversation_id": {
            "type": str,
            "min_length": 1,
            "description": "Unique conversation identifier",
            "required": True
        },
        "timestamp": {
            "type": (str, float, int),
            "description": "Entry timestamp",
            "required": True
        }
    }
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_handler: Optional[ErrorManager] = None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.error_handler = error_handler or ErrorManager(logger)
        self._initialized = False
        self._validation_errors = []
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(
            config_manager=self.config_manager,
            logger=self.logger,
            ram_manager=None,  # Not needed for data loading
            gpu_manager=None,  # Not needed for data loading
            error_manager=self.error_handler
        )
        
        # Mark as initialized
        self._initialized = True
        
    def _initialize_config(self) -> None:
        """Initialize configuration settings."""
        try:
            # Get data loading configuration
            self.batch_size = self.config_manager.get("data_config.batch_size", 1000)
            self.max_memory_mb = self.config_manager.get("data_config.max_memory_mb", 1024)
            self.memory_threshold = self.config_manager.get("data_config.memory_threshold", 0.8)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FileDataProvider: {str(e)}")
            raise
            
    def load_data(self, source: str, min_entries: int = 0) -> List[Dict[str, Any]]:
        """Load data from a JSONL file with memory monitoring."""
        try:
            self._validate_min_entries(min_entries)
            if not os.path.exists(source):
                self._log_error(FileNotFoundError(f"File not found: {source}"), "load_data")
                raise FileNotFoundError(f"File not found: {source}")
            data = []
            with open(source, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        data.append(entry)
                    except Exception as parse_exc:
                        self._log_error(parse_exc, "load_data: JSON parse error", traceback.format_exc())
            if len(data) < min_entries:
                self._log_error(InsufficientDataError(f"Loaded {len(data)} entries, required {min_entries}"), "load_data")
                raise InsufficientDataError(f"Loaded {len(data)} entries, required {min_entries}")
            self._log_load_success(source, len(data))
            return data
        except Exception as e:
            self._log_load_error(source, e)
            raise
            
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        data: List[Dict[str, Any]]
    ) -> None:
        """Process a batch of entries with memory monitoring."""
        try:
            # Check memory before processing
            if not self.memory_monitor.is_memory_available(len(batch) * 0.1):  # Estimate 0.1MB per entry
                self.logger.warning("Insufficient memory for batch processing")
                return
                
            # Process batch
            data.extend(batch)
            
            # Log memory usage periodically
            if len(data) % (self.batch_size * 10) == 0:
                self.memory_monitor.log_memory_usage()
                
        except Exception as e:
            self.logger.error(f"Failed to process batch: {str(e)}")
            raise
            
    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a single data entry."""
        try:
            required_fields = self.config_manager.get("data_config.required_fields", [])
            for field in required_fields:
                if field not in entry:
                    return False
            return True
        except Exception:
            return False
            
    def _log_event(self, event_type: str, message: str, level: str = "info", additional_info: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info={
                "timestamp": time.time(),
                **(additional_info or {})
            }
        )
        
    def _log_error(self, error: Exception, context: str, stack_trace: Optional[str] = None) -> None:
        """Log an error with context and stack trace."""
        self.logger.log_error(
            error_msg=str(error),
            error_type="data_error",
            stack_trace=stack_trace or traceback.format_exc(),
            additional_info={
                "context": context,
                "timestamp": time.time()
            }
        )

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """Validate JSONL data entries with detailed reporting."""
        try:
            if not isinstance(data, list):
                self._log_validation_warning("Data is not a list.")
                return False
            valid = True
            for entry in data:
                if not self._is_valid_entry(entry):
                    valid = False
                    self._log_validation_warning(f"Invalid entry: {entry}")
            return valid
        except Exception as e:
            self._log_error(e, "validate_data", traceback.format_exc())
            return False

    def _is_valid_entry(self, entry: Any) -> bool:
        """Validate a single data entry against required fields and types.
        
        Args:
            entry: The data entry to validate
            
        Returns:
            bool: True if the entry is valid, False otherwise
        """
        try:
            # Check if entry is a dictionary
            if not isinstance(entry, dict):
                self._log_validation_warning(
                    f"Invalid data entry: not a dictionary, got {str(entry)[:100]}"
                )
                return False
                
            # Track validation failures
            validation_errors = []
            
            # Validate each field
            for field, validator in self.REQUIRED_FIELDS.items():
                # Skip optional fields if not present
                if not validator.get("required", True) and field not in entry:
                    continue
                    
                # Check if field exists
                if field not in entry:
                    validation_errors.append(f"Missing required field: {field}")
                    continue
                    
                value = entry[field]
                
                # Type validation
                if not isinstance(value, validator["type"]):
                    validation_errors.append(
                        f"Invalid type for {field}: expected {validator['type'].__name__}, "
                        f"got {type(value).__name__}"
                    )
                    continue
                    
                # Length validation for strings
                if validator["type"] == str:
                    if "min_length" in validator and len(value.strip()) < validator["min_length"]:
                        validation_errors.append(
                            f"Field {field} too short: minimum length {validator['min_length']}, "
                            f"got {len(value.strip())}"
                        )
                    if "max_length" in validator and len(value.strip()) > validator["max_length"]:
                        validation_errors.append(
                            f"Field {field} too long: maximum length {validator['max_length']}, "
                            f"got {len(value.strip())}"
                        )
                        
                # Range validation for numbers
                if isinstance(value, (int, float)) and "range" in validator:
                    min_val, max_val = validator["range"]
                    if not (min_val <= value <= max_val):
                        validation_errors.append(
                            f"Field {field} out of range: expected [{min_val}, {max_val}], "
                            f"got {value}"
                        )
                        
            # Log validation results
            if validation_errors:
                self._log_validation_warning(
                    f"Data entry validation failed: {', '.join(validation_errors)}"
                )
                return False
                
            return True
            
        except Exception as e:
            # Handle any unexpected errors during validation
            self._log_validation_warning(
                f"Unexpected error during validation: {str(e)}"
            )
            return False

    def _log_load_success(self, source: str, entry_count: int) -> None:
        """Log successful data load event."""
        self._log_event(
            event_type="data_load_success",
            message="Data loaded successfully",
            level="info",
            additional_info={
                "source": source,
                "entry_count": entry_count
            }
        )

    def _log_load_error(self, source: str, error: Exception) -> None:
        """Log data load error."""
        self._log_error(
            error=error,
            context=f"Failed to load data from {source}",
            stack_trace=traceback.format_exc()
        )

    def _log_validation_warning(self, message: str) -> None:
        """Log validation warning."""
        self._log_event(
            event_type="data_validation_warning",
            message=message,
            level="warning"
        )

    def _validate_split_ratio(self, split_ratio: float) -> None:
        """Validate split ratio parameter.
        
        Args:
            split_ratio: The ratio to validate
            
        Raises:
            ValueError: If split_ratio is invalid
        """
        if not isinstance(split_ratio, (int, float)):
            raise ValueError(f"split_ratio must be a number, got {type(split_ratio)}")
            
        if not (0.0 <= split_ratio <= 1.0):
            raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")
            
        # Log validation
        self.logger.record_event(
            event_type="split_ratio_validated",
            message="Split ratio validated successfully",
            level="info",
            additional_info={"split_ratio": split_ratio}
        )

    def _validate_min_entries(self, min_entries: int) -> None:
        """Validate minimum entries parameter.
        
        Args:
            min_entries: The minimum entries to validate
            
        Raises:
            ValueError: If min_entries is invalid
        """
        if not isinstance(min_entries, int):
            raise ValueError(f"min_entries must be an integer, got {type(min_entries)}")
            
        if min_entries < 0:
            raise ValueError(f"min_entries must be non-negative, got {min_entries}")
            
        # Log validation
        self.logger.record_event(
            event_type="min_entries_validated",
            message="Minimum entries validated successfully",
            level="info",
            additional_info={"min_entries": min_entries}
        )

class DataManager:
    """Manages data loading, validation, and statistics."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, state: Optional[SOVLState] = None):
        """Initialize the data manager."""
        self.config_manager = config_manager
        self.logger = logger
        self.state = state
        self.data_stats = DataStats()
        self._lock = Lock()
        
    def load_and_split(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and split data into training and validation sets."""
        try:
            with self._lock:
                total_entries = len(data)
                valid_entries = 0
                invalid_entries = 0
                validation_errors = defaultdict(int)
                total_length = 0
                for entry in data:
                    if self._validate_entry(entry):
                        valid_entries += 1
                        total_length += len(entry.get("prompt", ""))
                    else:
                        invalid_entries += 1
                        validation_errors["invalid_format"] += 1
                avg_entry_length = safe_divide(total_length, valid_entries)
                self.data_stats.update(
                    total_entries=total_entries,
                    valid_entries=valid_entries,
                    invalid_entries=invalid_entries,
                    validation_errors=validation_errors,
                    average_entry_length=avg_entry_length
                )
                if self.state:
                    try:
                        self.state.update_data_stats({
                            "total_entries": total_entries,
                            "valid_entries": valid_entries,
                            "invalid_entries": invalid_entries,
                            "validation_errors": validation_errors,
                            "avg_entry_length": avg_entry_length
                        })
                    except Exception as state_exc:
                        self.logger.log_error(f"Failed to update state data stats: {state_exc}", error_type="data_manager_state_error")
                split_ratio = self.config_manager.get("data_config.validation_split_ratio", 0.2)
                if not (0.0 < split_ratio < 1.0):
                    self.logger.log_error(f"Invalid split ratio: {split_ratio}, using default 0.2", error_type="data_manager_split_ratio_warning")
                    split_ratio = 0.2
                split_idx = int(len(data) * (1 - split_ratio))
                return data[:split_idx], data[split_idx:]
        except Exception as e:
            import traceback
            self.logger.log_error(f"Failed to load and split data: {str(e)}\n{traceback.format_exc()}", error_type="data_loading_error")
            raise
            
    def get_data_stats(self) -> Dict[str, Any]:
        """Get current data statistics."""
        with self._lock:
            return self.data_stats.to_dict()

    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a single data entry."""
        try:
            required_fields = self.config_manager.get("data_config.required_fields", [])
            for field in required_fields:
                if field not in entry:
                    return False
            return True
        except Exception as e:
            self.logger.log_error(f"Validation error: {e}", error_type="data_entry_validation_error")
            return False
