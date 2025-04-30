import json
import os
import gzip
from typing import Optional, List, Dict, Any, Callable, Tuple
from threading import Lock
import traceback
import random
import time
from sovl_logger import Logger, LoggerConfig
from sovl_config import ConfigManager
from sovl_error import ErrorManager, ConfigurationError

class InsufficientDataError(Exception):
    """Raised when loaded data doesn't meet minimum entry requirements."""
    pass

class DataValidationError(Exception):
    """Raised when data fails validation checks."""
    pass

class JSONLLoader:
    """Thread-safe JSONL data loader with configurable validation."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, error_manager: ErrorManager):
        """
        Initialize loader with configuration and logger.

        Args:
            config_manager: ConfigManager instance for validation rules
            logger: Logger instance for recording events
            error_manager: ErrorManager instance for error handling
        """
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.lock = Lock()
        
        # Load configuration
        self._load_config()
        
    def _load_config(self) -> None:
        """Load and validate configuration."""
        try:
            # Get field mapping from config with defaults
            self.field_mapping = self.config_manager.get(
                "io_config.field_mapping",
                {"prompt": "prompt", "completion": "completion"}
            )
            
            # Get required fields from config with defaults
            self.required_fields = self.config_manager.get(
                "io_config.required_fields",
                ["prompt", "completion"]
            )
            
            # Get string length constraints from config
            self.min_string_length = self.config_manager.get(
                "io_config.min_string_length",
                1
            )
            self.max_string_length = self.config_manager.get(
                "io_config.max_string_length",
                10000
            )
            
            # Get validation settings
            self.strict_validation = self.config_manager.get(
                "io_config.strict_validation",
                False
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
            self.error_manager.handle_error(
                e,
                error_type="config",
                context={
                    "config_section": "io_config",
                    "error_type": "config_loading_error"
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
            self.error_manager.handle_error(
                e,
                error_type="config",
                context={
                    "config_section": "io_config",
                    "error_type": "config_validation_error"
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
            self.error_manager.handle_error(
                e,
                error_type="config",
                context={
                    "key": key,
                    "value": value,
                    "error_type": "config_update_error"
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
            self.error_manager.handle_error(
                e,
                error_type="config",
                context={
                    "key": key,
                    "error_type": "config_get_error"
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
            InsufficientDataError: If fewer than min_entries valid entries are loaded (non-strict mode only)
            DataValidationError: If file is invalid or corrupted, or if strict_validation is True and any validation fails
        """
        data = []
        errors = []  # Only used in non-strict mode
        
        # Use provided field mapping or fall back to config
        field_mapping = field_mapping or self.field_mapping
        
        # Combine default and custom validators
        validators = self.field_validators.copy()
        if custom_validators:
            validators.update(custom_validators)

        try:
            with self.lock:
                # Pre-loop validation checks
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    if self.strict_validation:
                        raise DataValidationError(error_msg)
                    self.logger.log_error(
                        error_msg=error_msg,
                        error_type="file_not_found",
                        stack_trace=traceback.format_exc(),
                        additional_info={"file_path": file_path}
                    )
                    return []

                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    error_msg = f"File {file_path} is empty"
                    if self.strict_validation:
                        raise DataValidationError(error_msg)
                    self.logger.log_error(
                        error_msg=error_msg,
                        error_type="empty_file",
                        additional_info={"file_path": file_path}
                    )
                    return []

                open_func = gzip.open if file_path.endswith('.gz') else open
                mode = 'rt' if file_path.endswith('.gz') else 'r'

                with open_func(file_path, mode, encoding='utf-8') as file:
                    for line_number, line in enumerate(file, start=1):
                        try:
                            # Process single line
                            line = line.strip()
                            if not line:
                                raise DataValidationError("Empty line")

                            # Parse JSON
                            entry = json.loads(line)
                            validated_entry = {}
                            
                            # Validate fields
                            for field in self.required_fields:
                                if field not in entry:
                                    raise DataValidationError(f"Missing required field '{field}'")
                                    
                                if field in validators and not validators[field](entry[field]):
                                    raise DataValidationError(f"Invalid value for '{field}': {entry[field]}")
                                    
                                output_field = field_mapping.get(field, field)
                                validated_entry[output_field] = entry[field]
                            
                            # If we get here, the entry is valid
                            data.append(validated_entry)
                            
                        except (DataValidationError, json.JSONDecodeError, TypeError, ValueError) as e:
                            # Handle any validation or parsing error for this line
                            error_msg = f"Line {line_number}: {str(e)}"
                            if self.strict_validation:
                                raise DataValidationError(error_msg) from e
                            errors.append(error_msg)
                            continue

                # Post-loop processing (non-strict mode only)
                if not self.strict_validation:
                    # Log accumulated errors
                    if errors:
                        self.logger.log_error(
                            error_msg="JSONL loading finished with errors",
                            error_type="data_validation_error",
                            additional_info={
                                "errors": errors[:100],  # Limit for performance
                                "total_errors": len(errors),
                                "file_path": file_path,
                                "field_mapping": field_mapping,
                                "required_fields": self.required_fields
                            }
                        )

                    # Check minimum entries requirement
                    if min_entries > 0 and len(data) < min_entries:
                        error_msg = f"Loaded only {len(data)} valid entries from {file_path}. Minimum required: {min_entries}"
                        self.error_manager.handle_error(
                            InsufficientDataError(error_msg),
                            error_type="data",
                            context={
                                "entries_loaded": len(data),
                                "min_required": min_entries,
                                "file_path": file_path,
                                "field_mapping": field_mapping,
                                "required_fields": self.required_fields
                            }
                        )
                        raise InsufficientDataError(error_msg)

                # Log successful load event
                self.logger.record({
                    "event": "jsonl_load_complete",
                    "file_path": file_path,
                    "entries_loaded": len(data),
                    "errors_found_non_strict": len(errors),
                    "file_size_bytes": file_size,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields,
                    "strict_mode": self.strict_validation,
                    "timestamp": time.time()
                })
                return data

        except Exception as e:
            # Handle any unexpected errors or re-raise strict validation errors
            error_to_raise = e if isinstance(e, (DataValidationError, InsufficientDataError, ConfigurationError)) else DataValidationError(f"Failed to load JSONL file: {str(e)}")
            
            self.error_manager.handle_error(
                error_to_raise,
                error_type="data_loading_failure",
                context={
                    "file_path": file_path,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields,
                    "strict_mode": getattr(self, 'strict_validation', 'unknown')
                }
            )
            raise error_to_raise from e

    def stream_jsonl(
        self,
        file_path: str,
        field_mapping: Optional[Dict[str, str]] = None,
        custom_validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ):
        """
        Stream (yield) validated entries from a JSONL file line-by-line.
        Args:
            file_path: Path to the JSONL file (supports .jsonl and .jsonl.gz)
            field_mapping: Optional mapping of input fields to output fields
            custom_validators: Optional custom validation functions for fields
        Yields:
            Dict[str, Any]: Each validated entry
        Raises:
            DataValidationError: If file is invalid or corrupted, or if strict_validation is True and any validation fails
        """
        field_mapping = field_mapping or self.field_mapping
        validators = self.field_validators.copy()
        if custom_validators:
            validators.update(custom_validators)
        try:
            with self.lock:
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    if self.strict_validation:
                        raise DataValidationError(error_msg)
                    self.logger.log_error(
                        error_msg=error_msg,
                        error_type="file_not_found",
                        stack_trace=traceback.format_exc(),
                        additional_info={"file_path": file_path}
                    )
                    return
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    error_msg = f"File {file_path} is empty"
                    if self.strict_validation:
                        raise DataValidationError(error_msg)
                    self.logger.log_error(
                        error_msg=error_msg,
                        error_type="empty_file",
                        additional_info={"file_path": file_path}
                    )
                    return
                open_func = gzip.open if file_path.endswith('.gz') else open
                mode = 'rt' if file_path.endswith('.gz') else 'r'
                with open_func(file_path, mode, encoding='utf-8') as file:
                    for line_number, line in enumerate(file, start=1):
                        try:
                            line = line.strip()
                            if not line:
                                raise DataValidationError("Empty line")
                            entry = json.loads(line)
                            validated_entry = {}
                            for field in self.required_fields:
                                if field not in entry:
                                    raise DataValidationError(f"Missing required field '{field}'")
                                if field in validators and not validators[field](entry[field]):
                                    raise DataValidationError(f"Invalid value for '{field}': {entry[field]}")
                                output_field = field_mapping.get(field, field)
                                validated_entry[output_field] = entry[field]
                            yield validated_entry
                        except (DataValidationError, json.JSONDecodeError, TypeError, ValueError) as e:
                            error_msg = f"Line {line_number}: {str(e)}"
                            if self.strict_validation:
                                raise DataValidationError(error_msg) from e
                            self.logger.log_warning(
                                error_msg,
                                error_type="streaming_data_validation_error",
                                additional_info={
                                    "file_path": file_path,
                                    "line_number": line_number
                                }
                            )
                            continue
        except Exception as e:
            error_to_raise = e if isinstance(e, (DataValidationError, InsufficientDataError, ConfigurationError)) else DataValidationError(f"Failed to stream JSONL file: {str(e)}")
            self.error_manager.handle_error(
                error_to_raise,
                error_type="data_streaming_failure",
                context={
                    "file_path": file_path,
                    "field_mapping": field_mapping,
                    "required_fields": self.required_fields,
                    "strict_mode": getattr(self, 'strict_validation', 'unknown')
                }
            )
            raise error_to_raise from e

def load_and_split_data(
    config_manager: ConfigManager,
    logger: Logger,
    error_manager: ErrorManager,
    formatted_training_data: List,
    valid_split_ratio: float
) -> Tuple[List, List]:
    """
    Load and split the training data into training and validation sets.

    Args:
        config_manager: ConfigManager instance for configuration settings
        logger: Logger instance for recording events
        error_manager: ErrorManager instance for error handling
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
        random_seed = config_manager.get("core_config.random_seed", 42)
        shuffle_data = config_manager.get("io_config.shuffle_data", True)
        
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
            
        formatted_training_data = valid_data[:split_idx]
        valid_data = valid_data[split_idx:]
        
        # Log data split
        logger.log_training_event(
            event_type="data_split",
            message="Data split into training and validation sets",
            additional_info={
                "random_seed": random_seed,
                "shuffled": shuffle_data,
                "original_size": len(formatted_training_data),
                "validated_size": len(valid_data)
            }
        )
        
        return formatted_training_data, valid_data
        
    except Exception as e:
        error_manager.handle_error(
            e,
            error_type="data",
            context={
                "formatted_training_data_size": len(formatted_training_data),
                "valid_split_ratio": valid_split_ratio
            }
        )
        raise DataValidationError(f"Failed to split data: {str(e)}")

def load_training_data(
    config_manager: ConfigManager,
    logger: Logger,
    error_manager: ErrorManager
) -> Tuple[List, List]:
    """
    Load and validate training data from the seed file.

    Args:
        config_manager: ConfigManager instance for configuration settings
        logger: Logger instance for recording events
        error_manager: ErrorManager instance for error handling

    Returns:
        A tuple containing the training and validation data lists

    Raises:
        ConfigurationError: If there is a problem with the configuration
        InsufficientDataError: If fewer than minimum required entries are loaded
        DataValidationError: If data validation fails or for unexpected errors
    """
    # Initialize context variables with defaults
    seed_file = "N/A"
    min_entries = "N/A"
    valid_split_ratio = "N/A"
    
    try:
        # Get configuration values
        seed_file = config_manager.get("io_config.seed_file", "sovl_seed.jsonl")
        min_entries = config_manager.get("io_config.min_training_entries", 10)
        valid_split_ratio = config_manager.get("core_config.valid_split_ratio", 0.2)
        
        # Initialize JSONL loader with error manager
        loader = JSONLLoader(config_manager, logger, error_manager)
        
        # Load training data
        formatted_training_data = loader.load_jsonl(seed_file, min_entries=min_entries)
        
        # Split data
        formatted_training_data, valid_data = load_and_split_data(
            config_manager,
            logger,
            error_manager,
            formatted_training_data,
            valid_split_ratio
        )
        
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
        
    except ConfigurationError as e:
        # Handle configuration errors specifically
        error_manager.handle_error(
            e,
            error_type="config",
            context={
                "seed_file": seed_file,
                "min_entries": min_entries,
                "valid_split_ratio": valid_split_ratio
            }
        )
        raise  # Re-raise the ConfigurationError
        
    except InsufficientDataError as e:
        # Handle insufficient data errors
        error_manager.handle_error(
            e,
            error_type="data",
            context={
                "seed_file": seed_file,
                "min_entries": min_entries,
                "valid_split_ratio": valid_split_ratio
            }
        )
        raise  # Re-raise the InsufficientDataError
        
    except DataValidationError as e:
        # Handle general data validation errors
        error_manager.handle_error(
            e,
            error_type="data",
            context={
                "seed_file": seed_file,
                "min_entries": min_entries,
                "valid_split_ratio": valid_split_ratio
            }
        )
        raise  # Re-raise the DataValidationError
        
    except Exception as e:
        # Handle any unexpected errors
        error_manager.handle_error(
            e,
            error_type="unexpected_data_loading_error",
            context={
                "seed_file": seed_file,
                "min_entries": min_entries,
                "valid_split_ratio": valid_split_ratio
            }
        )
        # Wrap unexpected errors in DataValidationError
        raise DataValidationError(f"Unexpected error during data loading: {str(e)}")

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

class JsonlWriter:
    """
    Handles writing structured data to a rotating JSONL file.
    Manages file I/O, buffering, and rotation for the scribed output.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        error_manager: ErrorManager,
        logger: Logger
    ):
        """
        Initialize the JSONL writer.

        Args:
            config_manager: Configuration manager instance
            error_manager: Error manager instance for error handling
            logger: Logger instance for operational logging
        """
        self.config_manager = config_manager
        self.error_manager = error_manager
        self.logger = logger
        self.lock = threading.Lock()
        
        # Load configuration
        self._load_config()
        
        # Setup fallback logger
        self.fallback_logger = self._setup_fallback_logger()
        
        # Initialize file handling
        self._buffer: List[str] = []
        self._file_handle = None
        self._setup_scribe_file()

    def _load_config(self) -> None:
        """Load and validate configuration settings."""
        try:
            # Get file path
            self.scribe_file_path = self.config_manager.get(
                "scribed_config.log_path",
                "logs/sovl_scribed.jsonl"
            )
            
            # Get max file size (convert MB to bytes)
            max_mb = self.config_manager.get(
                "scribed_config.max_file_size_mb",
                50
            )
            self.max_file_size_bytes = max_mb * 1024 * 1024
            
            # Get buffer size
            self.buffer_size = max(1, self.config_manager.get(
                "scribed_config.buffer_size",
                10
            ))
            
        except Exception as e:
            self.error_manager.handle_error(
                e,
                error_type="config",
                context={
                    "config_section": "scribed_config",
                    "error_type": "config_loading_error"
                }
            )
            raise

    def _setup_fallback_logger(self) -> logging.Logger:
        """Sets up an independent logger for critical I/O errors."""
        logger = logging.getLogger('sovl.scribe.jsonl_writer')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
        return logger

    def _setup_scribe_file(self) -> None:
        """Sets up the scribe file for writing."""
        try:
            # Ensure we have a valid path
            if not self.scribe_file_path:
                self.scribe_file_path = "logs/sovl_scribed.jsonl"
            
            # Create directory if it doesn't exist
            scribe_dir = os.path.dirname(self.scribe_file_path)
            if scribe_dir:  # Only create if there's a directory component
                os.makedirs(scribe_dir, exist_ok=True)
            
            self._open_file()
        except OSError as e:
            self.fallback_logger.exception(
                f"Failed to create directory or open scribe file: {self.scribe_file_path}"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "file_setup_error"
                }
            )
            raise

    def _open_file(self) -> None:
        """Opens the scribe file in append mode."""
        try:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.close()
            self._file_handle = open(self.scribe_file_path, 'a', encoding='utf-8')
        except IOError as e:
            self.fallback_logger.exception(
                f"Failed to open scribe file: {self.scribe_file_path}"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "file_open_error"
                }
            )
            self._file_handle = None
            raise

    def write(self, json_string: str) -> bool:
        """
        Writes a JSON string to the scribe file.

        Args:
            json_string: The JSON string to write

        Returns:
            bool: True if writing was successful, False otherwise
        """
        try:
            with self.lock:
                # Add to buffer
                self._buffer.append(json_string)

                # Flush if buffer is full
                if len(self._buffer) >= self.buffer_size:
                    self._flush_buffer()

            return True
        except Exception as write_error:
            self.fallback_logger.exception(
                f"Failed to buffer scribe entry. Error: {write_error}. "
                f"Entry prefix: {json_string[:500]}..."
            )
            self.error_manager.handle_error(
                write_error,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "write_error"
                }
            )
            return False

    def _flush_buffer(self) -> None:
        """Writes all buffered entries to the file."""
        if not self._buffer:
            return

        if not self._file_handle or self._file_handle.closed:
            self.fallback_logger.warning(
                "Attempted to flush buffer, but file handle is not open."
            )
            return

        try:
            for entry in self._buffer:
                self._file_handle.write(entry + '\n')
            self._file_handle.flush()
            self._buffer = []

            # Check file size for rotation
            if self._file_handle.tell() > self.max_file_size_bytes:
                self._rotate_scribe()

        except Exception as e:
            self.fallback_logger.exception(
                "Failed during buffer flush"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "flush_error"
                }
            )

    def _rotate_scribe(self) -> None:
        """Rotates the scribe file when it reaches the size limit."""
        if not self._file_handle:
            return

        try:
            # Close current file
            self._file_handle.close()
            
            # Rename current file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            old_path = self._file_handle.name
            new_path = f"{old_path}.{timestamp}"
            os.rename(old_path, new_path)
            
            # Reopen file
            self._open_file()
            
        except Exception as e:
            self.fallback_logger.exception(
                f"Failed to rotate scribe file: {self._file_handle.name}"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "rotation_error"
                }
            )

    def close(self) -> None:
        """Flushes buffer and closes the scribe file."""
        try:
            with self.lock:
                self._flush_buffer()
                if self._file_handle and not self._file_handle.closed:
                    self._file_handle.close()
                    self._file_handle = None
        except Exception as e:
            self.fallback_logger.exception(
                "Error closing scribe file"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "close_error"
                }
            )

class StreamingJSONLoader:
    """
    Loads samples from a JSONL file, yielding one sample at a time (streaming).
    Wraps the JSONLLoader from sovl_io for thread safety and validation.
    """
    def __init__(self, file_path: str, config_manager, logger, error_manager):
        from sovl_io import JSONLLoader
        if file_path is None:
            file_path = config_manager.get("data_provider.data_path")
        self.loader = JSONLLoader(config_manager, logger, error_manager)
        self.file_path = file_path
        self.logger = logger

    def __iter__(self):
        try:
            for sample in self.loader.stream_jsonl(self.file_path):
                yield sample
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to stream JSONL data: {e}", error_type="jsonl_stream_error")
            return