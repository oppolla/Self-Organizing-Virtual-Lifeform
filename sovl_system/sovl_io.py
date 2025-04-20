import json
import os
import gzip
from typing import Optional, List, Dict, Any, Callable, Tuple
from threading import Lock
import traceback
from sovl_logger import Logger
from sovl_config import ConfigManager
import random
import time

class InsufficientDataError(Exception):
    """Raised when loaded data doesn't meet minimum entry requirements."""
    pass

class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass

class ConfigurationError(Exception):
    """Raised when there is an error in configuration."""
    pass

class JSONLLoader:
    """Thread-safe JSONL data loader with configurable validation."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize loader with configuration and logger.

        Args:
            config_manager: ConfigManager instance for validation rules
            logger: Logger instance for recording events
        """
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()
        
        # Load configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load and validate configuration."""
        try:
            # Get field mapping from config with defaults
            self.field_mapping = self.config_manager.get(
                "io_config.field_mapping",
                {"response": "completion", "prompt": "prompt"},
                expected_type=dict
            )
            
            # Get required fields from config with defaults
            self.required_fields = self.config_manager.get(
                "io_config.required_fields",
                ["prompt", "response"],
                expected_type=list
            )
            
            # Get string length constraints from config
            self.min_string_length = self.config_manager.get(
                "io_config.min_string_length",
                1,
                expected_type=int
            )
            self.max_string_length = self.config_manager.get(
                "io_config.max_string_length",
                10000,
                expected_type=int
            )
            
            # Get validation settings
            self.enable_validation = self.config_manager.get(
                "io_config.enable_validation",
                True,
                expected_type=bool
            )
            self.strict_validation = self.config_manager.get(
                "io_config.strict_validation",
                False,
                expected_type=bool
            )
            
            # Initialize field validators
            self.field_validators = {
                "prompt": lambda x: isinstance(x, str) and self.min_string_length <= len(x.strip()) <= self.max_string_length,
                "response": lambda x: isinstance(x, str) and self.min_string_length <= len(x.strip()) <= self.max_string_length,
                "conversation_id": lambda x: isinstance(x, str) and len(x) > 0,
                "timestamp": lambda x: isinstance(x, (str, float, int)) and (isinstance(x, str) and len(x) > 0 or x > 0)
            }
            
            # Validate configuration
            self._validate_config()
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to load IO configuration: {str(e)}",
                error_type="config_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_section": "io_config"
                }
            )
            raise ConfigurationError(
                f"Failed to load IO configuration: {str(e)}",
                traceback.format_exc()
            )
            
    def _validate_config(self) -> None:
        """Validate IO configuration."""
        try:
            # Validate required configuration sections
            required_sections = ["field_mapping", "required_fields", "min_string_length", "max_string_length"]
            for section in required_sections:
                if not self.config_manager.get(f"io_config.{section}"):
                    raise ConfigurationError(
                        f"Missing required IO configuration section: {section}",
                        traceback.format_exc()
                    )
                    
            # Validate string length constraints
            if self.min_string_length < 0:
                raise ConfigurationError(
                    "Minimum string length must be non-negative",
                    traceback.format_exc()
                )
            if self.max_string_length <= self.min_string_length:
                raise ConfigurationError(
                    "Maximum string length must be greater than minimum string length",
                    traceback.format_exc()
                )
                
            # Validate field mapping
            if not isinstance(self.field_mapping, dict):
                raise ConfigurationError(
                    "Field mapping must be a dictionary",
                    traceback.format_exc()
                )
                
            # Validate required fields
            if not isinstance(self.required_fields, list):
                raise ConfigurationError(
                    "Required fields must be a list",
                    traceback.format_exc()
                )
                
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to validate IO configuration: {str(e)}",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "config_section": "io_config"
                }
            )
            raise ConfigurationError(
                f"Failed to validate IO configuration: {str(e)}",
                traceback.format_exc()
            )
            
    def update_config(self, key: str, value: Any) -> bool:
        """
        Update IO configuration.
        
        Args:
            key: Configuration key to update
            value: New value for the configuration key
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Update in config manager
            success = self.config_manager.update(f"io_config.{key}", value)
            
            if success:
                # Reload configuration to ensure consistency
                self._load_config()
                
            return success
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to update IO configuration: {str(e)}",
                error_type="config_update_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "key": key,
                    "value": value
                }
            )
            raise ConfigurationError(
                f"Failed to update IO configuration: {str(e)}",
                traceback.format_exc()
            )
            
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get IO configuration value.
        
        Args:
            key: Configuration key to get
            default: Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        try:
            return self.config_manager.get(f"io_config.{key}", default)
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to get IO configuration: {str(e)}",
                error_type="config_get_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "key": key
                }
            )
            raise ConfigurationError(
                f"Failed to get IO configuration: {str(e)}",
                traceback.format_exc()
            )

    def load_jsonl(
        self,
        file_path: str,
        min_entries: int = 0,
        field_mapping: Optional[Dict[str, str]] = None,
        custom_validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load a JSONL file into a list of dictionaries with validation.

        Args:
            file_path: Path to the JSONL file (supports .jsonl and .jsonl.gz)
            min_entries: Minimum number of valid entries required (0 to disable)
            field_mapping: Optional mapping of input fields to output fields
            custom_validators: Optional custom validation functions for fields

        Returns:
            List of validated dictionaries

        Raises:
            InsufficientDataError: If fewer than min_entries valid entries are loaded
            DataValidationError: If file is invalid or corrupted
        """
        data = []
        errors = []
        
        # Use provided field mapping or fall back to config
        field_mapping = field_mapping or self.field_mapping
        
        # Combine default and custom validators
        validators = self.field_validators.copy()
        if custom_validators:
            validators.update(custom_validators)

        try:
            with self.lock:
                if not os.path.exists(file_path):
                    self.logger.log_error(
                        error_msg=f"File not found: {file_path}",
                        error_type="file_not_found",
                        stack_trace=traceback.format_exc(),
                        additional_info={
                            "file_path": file_path
                        }
                    )
                    return []

                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    self.logger.log_error(
                        error_msg=f"File {file_path} is empty",
                        error_type="empty_file",
                        additional_info={
                            "file_path": file_path
                        }
                    )
                    return []

                open_func = gzip.open if file_path.endswith('.gz') else open
                mode = 'rt' if file_path.endswith('.gz') else 'r'

                with open_func(file_path, mode, encoding='utf-8') as file:
                    for line_number, line in enumerate(file, start=1):
                        line = line.strip()
                        if not line:
                            errors.append(f"Line {line_number}: Empty line")
                            continue
                        try:
                            entry = json.loads(line)
                            validated_entry = {}
                            
                            # Validate required fields
                            for field in self.required_fields:
                                if field not in entry:
                                    errors.append(f"Line {line_number}: Missing required field '{field}'")
                                    continue
                                if field in validators and not validators[field](entry[field]):
                                    errors.append(f"Line {line_number}: Invalid value for '{field}': {entry[field]}")
                                    continue
                                output_field = field_mapping.get(field, field)
                                validated_entry[output_field] = entry[field]
                            
                            if len(validated_entry) == len(self.required_fields):
                                data.append(validated_entry)
                            else:
                                errors.append(f"Line {line_number}: Incomplete entry after validation")
                                
                        except json.JSONDecodeError as e:
                            errors.append(f"Line {line_number}: JSON decode error: {str(e)}")
                            continue
                        except Exception as e:
                            errors.append(f"Line {line_number}: Unexpected error: {str(e)}")
                            continue

                # Log errors in batches to reduce overhead
                if errors:
                    self.logger.log_error(
                        error_msg="JSONL loading errors",
                        error_type="data_validation_error",
                        additional_info={
                            "errors": errors[:100],  # Limit for performance
                            "total_errors": len(errors),
                            "file_path": file_path,
                            "field_mapping": field_mapping,
                            "required_fields": self.required_fields
                        }
                    )

                if min_entries > 0 and len(data) < min_entries:
                    error_msg = f"Loaded only {len(data)} valid entries from {file_path}. Minimum required: {min_entries}"
                    self.logger.log_error(
                        error_msg=error_msg,
                        error_type="insufficient_data",
                        additional_info={
                            "entries_loaded": len(data),
                            "min_required": min_entries,
                            "file_path": file_path,
                            "field_mapping": field_mapping,
                            "required_fields": self.required_fields
                        }
                    )
                    raise InsufficientDataError(error_msg)

                self.logger.record({
                    "event": "jsonl_load",
                    "file_path": file_path,
                    "entries_loaded": len(data),
                    "file_size_bytes": file_size,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields,
                    "timestamp": time.time()
                })
                return data

        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to load JSONL file {file_path}: {str(e)}",
                error_type="data_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "file_path": file_path,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields
                }
            )
            raise DataValidationError(f"Failed to load JSONL file: {str(e)}")

def load_and_split_data(config_manager: ConfigManager, logger: Logger, formatted_training_data: List, valid_split_ratio: float) -> Tuple[List, List]:
    """
    Load and split the training data into training and validation sets.

    Args:
        config_manager: ConfigManager instance for configuration settings
        logger: Logger instance for recording events
        formatted_training_data: List of training data samples
        valid_split_ratio: Ratio for splitting validation data

    Returns:
        A tuple containing the training and validation data lists

    Raises:
        DataValidationError: If data validation fails
    """
    try:
        # Validate input parameters
        if not isinstance(formatted_training_data, list):
            raise DataValidationError("formatted_training_data must be a list")
            
        if not 0 < valid_split_ratio < 1:
            raise DataValidationError("valid_split_ratio must be between 0 and 1")
            
        if not formatted_training_data:
            logger.warning("Empty training data provided")
            return [], []
            
        # Get configuration values
        random_seed = config_manager.get("io_config.random_seed", 42, expected_type=int)
        shuffle_data = config_manager.get("io_config.shuffle_data", True, expected_type=bool)
        
        # Set random seed
        random.seed(random_seed)
        
        # Validate data structure
        valid_data = []
        for entry in formatted_training_data:
            if isinstance(entry, dict) and 'input' in entry and 'output' in entry:
                if isinstance(entry['input'], str) and isinstance(entry['output'], str):
                    if entry['input'].strip() and entry['output'].strip():
                        valid_data.append(entry)
        
        if not valid_data:
            logger.warning("No valid training data found after validation")
            return [], []
            
        # Shuffle data if enabled
        if shuffle_data:
            random.shuffle(valid_data)
            
        # Calculate split index with validation
        split_idx = int(len(valid_data) * (1 - valid_split_ratio))
        if split_idx == 0 or split_idx == len(valid_data):
            logger.warning("Split would result in empty dataset, adjusting split ratio")
            split_idx = max(1, min(len(valid_data) - 1, split_idx))
            
        train_data = valid_data[:split_idx]
        valid_data = valid_data[split_idx:]
        
        # Log data split
        logger.log_training_event(
            event_type="data_split",
            message="Data split into training and validation sets",
            additional_info={
                "train_samples": len(train_data),
                "valid_samples": len(valid_data),
                "split_ratio": valid_split_ratio,
                "random_seed": random_seed,
                "shuffled": shuffle_data,
                "original_size": len(formatted_training_data),
                "validated_size": len(valid_data)
            }
        )
        
        return train_data, valid_data
        
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to split data: {str(e)}",
            error_type="data_split_error",
            stack_trace=traceback.format_exc(),
            additional_info={
                "train_data_size": len(formatted_training_data),
                "valid_split_ratio": valid_split_ratio
            }
        )
        raise DataValidationError(f"Failed to split data: {str(e)}")

def validate_quantization_mode(mode: str, logger: Logger) -> str:
    """
    Validate and normalize the quantization mode.

    Args:
        mode: The quantization mode to validate
        logger: Logger instance for recording events

    Returns:
        The normalized quantization mode

    Raises:
        ValueError: If the mode is invalid
    """
    try:
        # Get valid modes from config
        valid_modes = ["fp16", "int8", "int4"]
        normalized_mode = mode.lower()
        
        if normalized_mode not in valid_modes:
            logger.log_training_event(
                event_type="quantization_mode_validation",
                message=f"Invalid quantization mode '{mode}'. Defaulting to 'fp16'.",
                additional_info={
                    "invalid_mode": mode,
                    "valid_modes": valid_modes,
                    "default_mode": "fp16"
                }
            )
            return "fp16"
        
        return normalized_mode
        
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to validate quantization mode: {str(e)}",
            error_type="quantization_validation_error",
            stack_trace=traceback.format_exc(),
            additional_info={
                "mode": mode
            }
        )
        raise ValueError(f"Failed to validate quantization mode: {str(e)}")

def load_training_data(config_manager: ConfigManager, logger: Logger) -> Tuple[List, List]:
    """
    Load and validate training data from the seed file.

    Args:
        config_manager: ConfigManager instance for configuration settings
        logger: Logger instance for recording events

    Returns:
        A tuple containing the training and validation data lists

    Raises:
        InsufficientDataError: If fewer than minimum required entries are loaded
        DataValidationError: If data validation fails
    """
    try:
        # Get configuration values
        seed_file = config_manager.get("io_config.seed_file", "sovl_seed.jsonl", expected_type=str)
        min_entries = config_manager.get("io_config.min_training_entries", 10, expected_type=int)
        valid_split_ratio = config_manager.get("io_config.valid_split_ratio", 0.2, expected_type=float)
        
        # Initialize JSONL loader
        loader = JSONLLoader(config_manager, logger)
        
        # Load training data
        formatted_training_data = loader.load_jsonl(seed_file, min_entries=min_entries)
        
        # Split data
        formatted_training_data, valid_data = load_and_split_data(config_manager, logger, formatted_training_data, valid_split_ratio)
        
        # Log successful data loading
        logger.log_training_event(
            event_type="training_data_loaded",
            message="Training data loaded successfully",
            additional_info={
                "train_samples": len(formatted_training_data),
                "valid_samples": len(valid_data),
                "min_entries": min_entries,
                "seed_file": seed_file,
                "valid_split_ratio": valid_split_ratio
            }
        )
        
        return formatted_training_data, valid_data
        
    except InsufficientDataError as e:
        logger.log_error(
            error_msg=str(e),
            error_type="insufficient_data",
            stack_trace=traceback.format_exc(),
            additional_info={
                "min_entries": min_entries,
                "seed_file": seed_file
            }
        )
        raise
        
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to load training data: {str(e)}",
            error_type="data_loading_error",
            stack_trace=traceback.format_exc(),
            additional_info={
                "seed_file": seed_file,
                "min_entries": min_entries
            }
        )
        raise DataValidationError(f"Failed to load training data: {str(e)}")

if __name__ == "__main__":
    from sovl_logger import Logger, LoggerConfig
    from sovl_config import ConfigManager
    logger = Logger(LoggerConfig())
    config_manager = ConfigManager("sovl_config.json", logger)
    loader = JSONLLoader(config_manager, logger)
    try:
        data = loader.load_jsonl("sample.jsonl", min_entries=1)
        logger.record_event(
            event_type="data_loaded",
            message=f"Loaded {len(data)} entries from sample.jsonl",
            level="info",
            additional_info={
                "entries_loaded": len(data),
                "file_path": "sample.jsonl"
            }
        )
    except (InsufficientDataError, DataValidationError) as e:
        logger.log_error(
            error_msg=str(e),
            error_type="data_loading_error",
            stack_trace=traceback.format_exc(),
            additional_info={
                "file_path": "sample.jsonl"
            }
        )