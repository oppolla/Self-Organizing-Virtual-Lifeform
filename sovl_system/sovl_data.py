from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import random
import time
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_error import ErrorHandler
from sovl_io import InsufficientDataError
import traceback
import os
import json
from collections import defaultdict
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager
from threading import Lock

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
        self.memory_monitor = MemoryMonitor(config_manager, logger)
        
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
        if not self._initialized:
            raise RuntimeError("FileDataProvider not initialized")
            
        try:
            # Check file existence
            if not os.path.exists(source):
                raise FileNotFoundError(f"Data source not found: {source}")
                
            # Initialize data collection
            data = []
            current_batch = []
            total_entries = 0
            
            # Open file for streaming
            with open(source, 'r') as f:
                for line in f:
                    # Check memory before processing each entry
                    if not self.memory_monitor.check_memory_usage():
                        self.logger.warning("Memory threshold exceeded, stopping data load")
                        break
                        
                    try:
                        entry = json.loads(line)
                        if self._validate_entry(entry):
                            current_batch.append(entry)
                            total_entries += 1
                            
                            # Process batch if size reached
                            if len(current_batch) >= self.batch_size:
                                self._process_batch(current_batch, data)
                                current_batch = []
                                
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON in line: {str(e)}")
                        continue
                        
                # Process remaining entries
                if current_batch:
                    self._process_batch(current_batch, data)
                    
            # Validate minimum entries
            if total_entries < min_entries:
                raise InsufficientDataError(
                    f"Loaded {total_entries} entries, minimum required: {min_entries}"
                )
                
            # Log success
            self.logger.info(
                "Data loaded successfully",
                extra={
                    "source": source,
                    "total_entries": total_entries,
                    "valid_entries": len(data),
                    "memory_usage": self.memory_monitor.get_memory_usage()
                }
            )
            
            return data
            
        except Exception as e:
            self.error_handler.handle_data_error(e, source=source)
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
        if not data:
            self._log_validation_warning("Empty data provided for validation")
            return False
        
        validation_stats = {
            "total_entries": len(data),
            "valid_entries": 0,
            "invalid_entries": 0,
            "missing_fields": defaultdict(int),
            "type_errors": defaultdict(int),
            "range_errors": defaultdict(int),
            "length_errors": defaultdict(int)
        }
        
        for entry in data:
            if self._is_valid_entry(entry):
                validation_stats["valid_entries"] += 1
            else:
                validation_stats["invalid_entries"] += 1
                
        # Log validation statistics
        self.logger.record_event(
            event_type="data_validation_summary",
            message="Data validation completed",
            level="info",
            additional_info=validation_stats
        )
        
        # Return true only if all entries are valid
        return validation_stats["valid_entries"] == validation_stats["total_entries"]

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
    """Manages data operations and memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        memoria_manager: MemoriaManager,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize data manager.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            memoria_manager: MemoriaManager instance for core memory management
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health across all memory managers."""
        try:
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            return {
                "ram_health": ram_health,
                "gpu_health": gpu_health
            }
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to check memory health: {str(e)}",
                error_type="memory_health_error",
                stack_trace=traceback.format_exc()
            )
            return {
                "ram_health": {"status": "error"},
                "gpu_health": {"status": "error"}
            }

    def load_and_split(
        self,
        source: Optional[str] = None,
        split_ratio: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load data from source and split into training and validation sets.
        
        Args:
            source: Optional source path for data. If None, uses default from config.
            split_ratio: Optional ratio for validation split. If None, uses default from config.
            
        Returns:
            Tuple of (formatted_training_data, valid_data)
            
        Raises:
            RuntimeError: If DataManager is not properly initialized
            ValueError: If data validation fails
            InsufficientDataError: If insufficient valid data is available
        """
        # Check initialization
        self._check_initialization()
        
        try:
            # Use defaults if not provided
            source = source or self._config_manager.get("core_config.data_source", "sovl_seed.jsonl")
            split_ratio = split_ratio or self._config_manager.get("core_config.valid_split_ratio", 0.2)
            
            # Validate split ratio
            if not 0 < split_ratio < 1:
                raise ValueError(f"Invalid split ratio: {split_ratio}. Must be between 0 and 1.")
                
            # Load and validate data
            self._logger.record_event(
                event_type="data_load_start",
                message=f"Loading data from {source}",
                level="info"
            )
            
            data = self.provider.load_data(source)
            
            # Calculate statistics
            total_entries = len(data)
            valid_entries = len([d for d in data if self._is_valid_entry(d)])
            invalid_entries = total_entries - valid_entries
            avg_entry_length = sum(len(str(d)) for d in data) / total_entries if total_entries > 0 else 0
            
            # Update state if available
            if self.state:
                self.state.update_data_stats({
                    "total_entries": total_entries,
                    "valid_entries": valid_entries,
                    "invalid_entries": invalid_entries,
                    "avg_entry_length": avg_entry_length
                })
                
            # Validate data size
            if valid_entries < self.min_entries:
                raise InsufficientDataError(
                    f"Insufficient valid data: {valid_entries} entries, "
                    f"minimum required: {self.min_entries}"
                )
                
            # Split data
            split_idx = int(len(data) * (1 - split_ratio))
            formatted_training_data = data[:split_idx]
            valid_data = data[split_idx:]
            
            # Log success
            self._logger.record_event(
                event_type="data_load_success",
                message="Data loaded and split successfully",
                level="info",
                additional_info={
                    "source": source,
                    "total_entries": total_entries,
                    "valid_entries": valid_entries,
                    "invalid_entries": invalid_entries,
                    "avg_entry_length": avg_entry_length,
                    "train_size": len(formatted_training_data),
                    "valid_size": len(valid_data),
                    "split_ratio": split_ratio
                }
            )
            
            return formatted_training_data, valid_data
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to load and split data: {str(e)}",
                error_type="data_load_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _get_conversation_id(self) -> str:
        """Get the current conversation ID from state or generate a default."""
        if self.state and hasattr(self.state, 'history') and hasattr(self.state.history, 'conversation_id'):
            return self.state.history.conversation_id
        return "data_operation"

    def _load_data(self, source: str, min_entries: int) -> List[Dict[str, Any]]:
        """Load data with logging."""
        data = self.provider.load_data(source, min_entries)
        if not data:
            self._logger.record_event(
                event_type="data_load_warning",
                message=f"No data loaded from {source}",
                level="warning",
                additional_info={
                    "source": source,
                    "min_entries": min_entries,
                    "conversation_id": self._get_conversation_id()
                }
            )
        return data

    def _split_data(
        self, data: List[Dict[str, Any]], split_ratio: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into training and validation sets with validation.
        
        Args:
            data: List of data entries to split
            split_ratio: Fraction of data to use for validation (must be between 0 and 1)
            
        Returns:
            Tuple of (formatted_training_data, valid_data)
            
        Raises:
            ValueError: If split_ratio is invalid or data is empty
        """
        try:
            # Validate split ratio
            self._validate_split_ratio(split_ratio)
            
            # Handle empty data
            if not data:
                self._logger.record_event(
                    event_type="data_split_warning",
                    message="Attempted to split empty data",
                    level="warning"
                )
                return [], []
                
            # Ensure data is a list
            if not isinstance(data, list):
                self._logger.record_event(
                    event_type="data_split_error",
                    message=f"Data must be a list, got {type(data)}",
                    level="error"
                )
                raise ValueError("data must be a list")
                
            # Get configuration for batch size and max shuffle size
            batch_size = self._config_manager.get("core_config.batch_size", 1000)
            max_shuffle_size = self._config_manager.get("core_config.max_shuffle_size", 10000)
            
            # For large datasets, use batch-based shuffling
            if len(data) > max_shuffle_size:
                self._logger.record_event(
                    event_type="data_split_info",
                    message=f"Large dataset detected ({len(data)} entries), using batch-based shuffling",
                    level="info",
                    additional_info={
                        "max_shuffle_size": max_shuffle_size,
                        "batch_size": batch_size
                    }
                )
                
                # Calculate split indices
                total_size = len(data)
                train_size = int(total_size * (1 - split_ratio))
                valid_size = total_size - train_size
                
                # Initialize empty lists for train and validation data
                formatted_training_data = []
                valid_data = []
                
                # Process data in batches
                random.seed(self.random_seed)
                indices = list(range(total_size))
                random.shuffle(indices)
                
                for i in range(0, total_size, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_data = [data[idx] for idx in batch_indices]
                    
                    # Add to appropriate split based on current sizes
                    if len(formatted_training_data) < train_size:
                        formatted_training_data.extend(batch_data[:train_size - len(formatted_training_data)])
                        if len(formatted_training_data) == train_size and len(valid_data) < valid_size:
                            valid_data.extend(batch_data[train_size - len(formatted_training_data):])
                    else:
                        valid_data.extend(batch_data)
                
                # Log successful split with batch processing
                self._logger.record_event(
                    event_type="data_split_success",
                    message="Data successfully split using batch processing",
                    level="info",
                    additional_info={
                        "total_entries": total_size,
                        "train_entries": len(formatted_training_data),
                        "valid_entries": len(valid_data),
                        "split_ratio": split_ratio,
                        "batch_size": batch_size,
                        "max_shuffle_size": max_shuffle_size,
                        "conversation_id": self._get_conversation_id()
                    }
                )
                
                return formatted_training_data, valid_data
                
            # For smaller datasets, use standard shuffling
            random.seed(self.random_seed)
            shuffled_data = data.copy()
            random.shuffle(shuffled_data)
            split_idx = int(len(shuffled_data) * (1 - split_ratio))
            
            # Ensure we don't get empty splits
            if split_idx == 0 or split_idx == len(shuffled_data):
                self._logger.record_event(
                    event_type="data_split_warning",
                    message="Split would result in empty dataset",
                    level="warning",
                    additional_info={
                        "split_idx": split_idx,
                        "data_length": len(shuffled_data),
                        "split_ratio": split_ratio
                    }
                )
                # Adjust split to ensure non-empty sets
                split_idx = max(1, min(len(shuffled_data) - 1, split_idx))
                
            # Perform split
            formatted_training_data = shuffled_data[:split_idx]
            valid_data = shuffled_data[split_idx:]
            
            # Log successful split
            self._logger.record_event(
                event_type="data_split_success",
                message="Data successfully split into training and validation sets",
                level="info",
                additional_info={
                    "total_entries": len(data),
                    "train_entries": len(formatted_training_data),
                    "valid_entries": len(valid_data),
                    "split_ratio": split_ratio,
                    "conversation_id": self._get_conversation_id()
                }
            )
            
            return formatted_training_data, valid_data
            
        except Exception as e:
            # Handle any unexpected errors during splitting
            self._logger.log_error(
                error_msg=f"Failed to split data: {str(e)}",
                error_type="data_split_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate data using the current provider.
        
        Args:
            data: List of data entries to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        try:
            return self.provider.validate_data(data)
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to validate data: {str(e)}",
                error_type="data_validation_error",
                stack_trace=traceback.format_exc()
            )
            return False

    def get_data_stats(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about the data.
        
        Args:
            data: List of data entries.
        
        Returns:
            Dictionary with statistics (e.g., entry count, prompt lengths).
        """
        if not data:
            return {"entry_count": 0, "avg_prompt_length": 0.0, "has_response": False}
        
        stats = {
            "entry_count": len(data),
            "avg_prompt_length": 0.0,
            "has_response": False
        }
        
        prompt_lengths = [len(entry.get("prompt", "")) for entry in data]
        stats["has_response"] = any("response" in entry for entry in data)
        stats["avg_prompt_length"] = (
            sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
        )
        
        return stats

    def _handle_validation_error(self, source: str, entry_count: int) -> None:
        """Handle data validation errors."""
        self._logger.record_event(
            event_type="data_validation_error",
            message=f"Validation failed for {entry_count} entries from {source}",
            level="error",
            additional_info={
                "source": source,
                "entry_count": entry_count
            }
        )

    def _handle_data_error(self, error: Exception, source: str, min_entries: int) -> None:
        """Handle general data errors."""
        self._logger.log_error(
            error_msg=f"Failed to load data from {source}: {str(error)}",
            error_type="data_load_error",
            stack_trace=traceback.format_exc()
        )

    def _log_split_success(self, source: str, total: int, train: int, valid: int, split_ratio: float) -> None:
        """Log successful data split."""
        self._logger.record_event(
            event_type="data_split_success",
            message=f"Successfully split {total} entries into {train} training and {valid} validation entries",
            level="info",
            additional_info={
                "source": source,
                "total_entries": total,
                "training_entries": train,
                "validation_entries": valid,
                "split_ratio": split_ratio
            }
        )

    def _validate_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """Validate data entries and track validation statistics.
        
        Args:
            data: List of data entries to validate
            
        Returns:
            Tuple of (valid_data, invalid_data, validation_errors)
            where validation_errors is a dict mapping error types to counts
        """
        if not data:
            return [], [], {}
            
        valid_data = []
        invalid_data = []
        validation_errors = defaultdict(int)
        
        for entry in data:
            try:
                if self._is_valid_entry(entry):
                    valid_data.append(entry)
                else:
                    invalid_data.append(entry)
                    validation_errors["invalid_format"] += 1
            except Exception as e:
                invalid_data.append(entry)
                validation_errors[str(type(e).__name__)] += 1
                
        return valid_data, invalid_data, dict(validation_errors)

    def split_data(self, data: List, valid_split_ratio: float) -> Tuple[List, List]:
        """Split data into training and validation sets."""
        split_idx = int(len(data) * (1 - valid_split_ratio))
        formatted_training_data = data[:split_idx]
        valid_data = data[split_idx:]
        
        return formatted_training_data, valid_data

    def prepare_training_data(self, data: List, config: Dict) -> Tuple[List, List]:
        """Prepare training data according to configuration."""
        formatted_training_data = []
        valid_data = []
        
        # ... rest of the function ...
        
        return formatted_training_data, valid_data
