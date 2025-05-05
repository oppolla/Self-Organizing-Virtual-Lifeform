import json
import os
import gzip
import uuid
import time
import logging
from datetime import datetime
from threading import Lock, RLock
from typing import List, Dict, Union, Optional, Callable, Any, Tuple, Literal
from dataclasses import dataclass, field
import traceback
from collections import deque, defaultdict
from sovl_config import ConfigManager
from sovl_records import ErrorRecordBridge, IErrorHandler, ErrorRecord

LOGGING_ENABLED = True  # Universal on/off switch for all logging

# Utility to set LOGGING_ENABLED from config

def set_logging_enabled_from_config(config_manager: ConfigManager):
    global LOGGING_ENABLED
    try:
        LOGGING_ENABLED = config_manager.get("logging_config.logging_enabled", True)
    except Exception:
        LOGGING_ENABLED = True

@dataclass
class LoggerConfig:
    """Configuration for Logger with validation."""
    log_file: str = "sovl_logs.jsonl"
    max_size_mb: int = 10
    compress_old: bool = False
    max_in_memory_logs: int = 1000
    rotation_count: int = 5
    max_log_age_days: int = 30  # Maximum age of logs to keep
    prune_interval_hours: int = 24  # How often to prune old logs
    memory_threshold_mb: int = 100  # Memory threshold to trigger aggressive pruning
    gpu_memory_threshold: float = 0.85  # GPU memory usage threshold (0-1)
    log_level: str = "INFO"  # Log level threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Error handling configuration
    error_cooldown: float = 1.0  # Time in seconds before an error is no longer considered recent
    max_recent_errors: int = 100  # Maximum number of recent errors to track
    error_handling_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_history_per_error": 10,
        "critical_threshold": 5,
        "warning_threshold": 10,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "memory_recovery_attempts": 3,
        "memory_recovery_delay": 1.0
    })

    _RANGES = {
        "max_size_mb": (0, 100),
        "max_in_memory_logs": (100, 10000),
        "max_log_age_days": (1, 365),
        "prune_interval_hours": (1, 168),
        "memory_threshold_mb": (10, 1000),
        "gpu_memory_threshold": (0.1, 1.0),
        "error_cooldown": (0.1, 60.0),
        "max_recent_errors": (10, 1000)
    }

    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.log_file, str) or not self.log_file.endswith(".jsonl"):
            raise ValueError("log_file must be a .jsonl file path")
        if not isinstance(self.compress_old, bool):
            raise ValueError("compress_old must be a boolean")
        for key, (min_val, max_val) in self._RANGES.items():
            value = getattr(self, key)
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
        # Validate log_level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got {self.log_level}")
        # Validate error handling config
        if not isinstance(self.error_handling_config, dict):
            raise ValueError("error_handling_config must be a dictionary")
        required_error_keys = {"max_history_per_error", "critical_threshold", "warning_threshold"}
        if not all(key in self.error_handling_config for key in required_error_keys):
            raise ValueError(f"error_handling_config must contain all required keys: {required_error_keys}")

    def update(self, **kwargs) -> None:
        """Update configuration with validation."""
        for key, value in kwargs.items():
            if key == "log_file":
                if not isinstance(value, str) or not value.endswith(".jsonl"):
                    raise ValueError("log_file must be a .jsonl file path")
            elif key == "compress_old":
                if not isinstance(value, bool):
                    raise ValueError("compress_old must be a boolean")
            elif key in self._RANGES:
                min_val, max_val = self._RANGES[key]
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {value}")
            elif key == "rotation_count":
                if not isinstance(value, int) or value < 0:
                    raise ValueError("rotation_count must be a non-negative integer")
            elif key == "error_handling_config":
                if not isinstance(value, dict):
                    raise ValueError("error_handling_config must be a dictionary")
                required_keys = {"max_history_per_error", "critical_threshold", "warning_threshold"}
                if not all(k in value for k in required_keys):
                    raise ValueError(f"error_handling_config must contain all required keys: {required_keys}")
            elif key == "log_level":
                valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
                if not isinstance(value, str) or value.upper() not in valid_levels:
                    raise ValueError(f"log_level must be one of {valid_levels}, got {value}")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)

class _LogValidator:
    """Handles log entry validation logic."""
    
    REQUIRED_FIELDS = {'timestamp', 'conversation_id'}
    OPTIONAL_FIELDS = {'prompt', 'response', 'confidence_score', 'error', 'warning', 'mood', 'variance', 'logits_shape'}
    FIELD_VALIDATORS = {
        'timestamp': lambda x: isinstance(x, (str, float, int)),
        'conversation_id': lambda x: isinstance(x, str),
        'confidence_score': lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
        'is_error_prompt': lambda x: isinstance(x, bool),
        'mood': lambda x: x in {'melancholic', 'restless', 'calm', 'curious'},
        'variance': lambda x: isinstance(x, (int, float)) and x >= 0.0,
        'logits_shape': lambda x: isinstance(x, (tuple, list, str))
    }

    def __init__(self, fallback_logger: logging.Logger):
        self.fallback_logger = fallback_logger

    def validate_entry(self, entry: Dict) -> bool:
        """Validate log entry structure and types."""
        if not isinstance(entry, dict):
            self.fallback_logger.warning("Log entry is not a dictionary")
            return False

        try:
            # Ensure required fields
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().isoformat()
            if 'conversation_id' not in entry:
                entry['conversation_id'] = str(uuid.uuid4())

            # Validate field types
            for field, validator in self.FIELD_VALIDATORS.items():
                if field in entry and not validator(entry[field]):
                    self.fallback_logger.warning(f"Invalid value for field {field}: {entry[field]}")
                    return False

            return True
        except Exception as e:
            self.fallback_logger.error(f"Validation failed: {str(e)}")
            return False

class _FileHandler:
    """Manages file operations for logging."""
    
    def __init__(self, config: LoggerConfig, fallback_logger: logging.Logger):
        self.config = config
        self.fallback_logger = fallback_logger

    def safe_file_op(self, operation: Callable, *args, **kwargs):
        """Execute file operation with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    self.fallback_logger.error(f"File operation failed after {max_retries} retries: {str(e)}")
                    raise
                time.sleep(0.1 * (attempt + 1))

    def atomic_write(self, filename: str, content: str) -> None:
        """Perform atomic file write using temporary file."""
        temp_file = f"{filename}.tmp"
        try:
            with self.safe_file_op(open, temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.safe_file_op(os.replace, temp_file, filename)
        except Exception as e:
            self.fallback_logger.error(f"Atomic write failed: {str(e)}")
            if os.path.exists(temp_file):
                self.safe_file_op(os.remove, temp_file)
            raise

    def rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        if self.config.max_size_mb <= 0 or not os.path.exists(self.config.log_file):
            return

        file_size = os.path.getsize(self.config.log_file)
        if file_size < self.config.max_size_mb * 1024 * 1024:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_file = f"{self.config.log_file}.{timestamp}"

            if self.config.compress_old:
                rotated_file += ".gz"
                with self.safe_file_op(open, self.config.log_file, 'rb') as f_in:
                    with self.safe_file_op(gzip.open, rotated_file, 'wb') as f_out:
                        f_out.writelines(f_in)
            else:
                self.safe_file_op(os.rename, self.config.log_file, rotated_file)

            self.fallback_logger.info(f"Rotated logs to {rotated_file}")
        except Exception as e:
            self.fallback_logger.error(f"Failed to rotate log file: {str(e)}")

    def compress_logs(self, keep_original: bool = False) -> Optional[str]:
        """Compress current log file."""
        if not os.path.exists(self.config.log_file):
            return None

        compressed_file = f"{self.config.log_file}.{datetime.now().strftime('%Y%m%d')}.gz"
        try:
            with self.safe_file_op(open, self.config.log_file, 'rb') as f_in:
                with self.safe_file_op(gzip.open, compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)

            if not keep_original:
                self.safe_file_op(os.remove, self.config.log_file)

            self.fallback_logger.info(f"Compressed logs to {compressed_file}")
            return compressed_file
        except Exception as e:
            self.fallback_logger.error(f"Failed to compress logs: {str(e)}")
            return None

    def manage_rotation(self, max_files: int = 5) -> None:
        """Manage rotated log files, keeping only max_files most recent."""
        if not os.path.exists(self.config.log_file):
            return

        try:
            base_name = os.path.basename(self.config.log_file)
            log_dir = os.path.dirname(self.config.log_file) or '.'

            rotated_files = [
                os.path.join(log_dir, f) for f in os.listdir(log_dir)
                if f.startswith(base_name) and f != base_name
            ]

            rotated_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            for old_file in rotated_files[max_files:]:
                try:
                    self.safe_file_op(os.remove, old_file)
                    self.fallback_logger.info(f"Removed old log file {old_file}")
                except OSError:
                    self.fallback_logger.error(f"Failed to remove old log file {old_file}")
        except Exception as e:
            self.fallback_logger.error(f"Error managing log rotation: {str(e)}")

    def write_batch(self, entries: List[Dict]) -> None:
        """Optimized batch writing with validation and atomic write."""
        if not entries:
            return

        valid_entries = []
        for entry in entries:
            if self.validate_entry(entry):
                if "error" in entry or "warning" in entry:
                    entry["is_error_prompt"] = True
                valid_entries.append(entry)
            else:
                self.fallback_logger.warning(f"Invalid log entry skipped: {entry}")

        if not valid_entries:
            return

        try:
            with self.safe_file_op(open, self.config.log_file, 'a') as f:
                for entry in valid_entries:
                    f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.fallback_logger.error(f"Error writing batch: {str(e)}")

    def cleanup(self) -> None:
        """Clean up logging resources for the file handler."""
        try:
            self.manage_rotation()
            self.compress_logs()
        except Exception as e:
            if self.fallback_logger:
                self.fallback_logger.log_error(
                    error_msg=f"Failed to clean up file handler: {str(e)}",
                    error_type="file_handler_cleanup_error",
                    stack_trace=traceback.format_exc()
                )

class ILoggerClient:
    """Interface for logger clients to ensure consistent interaction."""
    def log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        raise NotImplementedError

    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, **kwargs) -> None:
        raise NotImplementedError

class Logger(IErrorHandler):
    """Main logger class for the SOVL system."""
    _instance = None
    _lock = RLock()
    LOG_LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
            
    def __init__(self, config_manager: ConfigManager = None):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._lock = RLock()
            # Set LOGGING_ENABLED from config if available
            if config_manager is not None:
                set_logging_enabled_from_config(config_manager)
            
            # Initialize configuration
            self.config = LoggerConfig()
            
            # Initialize fallback logger for internal errors
            self._fallback_logger = logging.getLogger('sovl_internal')
            if not self._fallback_logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self._fallback_logger.addHandler(handler)
                self._fallback_logger.setLevel(logging.INFO)
            
            # Initialize components
            self._validator = _LogValidator(self._fallback_logger)
            self._file_handler = _FileHandler(self.config, self._fallback_logger)
            
            # Register with ErrorRecordBridge
            ErrorRecordBridge().register_handler(self)
            
            # Initial cleanup
            self._file_handler.manage_rotation()
    
    @classmethod
    def get_instance(cls) -> 'Logger':
        """Get the singleton instance of the Logger."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def should_log(self, entry_level: str) -> bool:
        config_level = self.config.log_level
        entry_level_num = self.LOG_LEVELS.get(entry_level.upper(), 20)
        config_level_num = self.LOG_LEVELS.get(config_level.upper(), 20)
        return entry_level_num >= config_level_num
    
    def record_event(self, event_type: str, message: str, level: str = "info", additional_info: Dict[str, Any] = None) -> None:
        """Record a general system event."""
        if not LOGGING_ENABLED or not self.should_log(level):
            return
        with self._lock:
            try:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'conversation_id': str(uuid.uuid4()),
                    'event_type': event_type,
                    'message': message,
                    'level': level,
                    **(additional_info or {})
                }
                if self._validator.validate_entry(log_entry):
                    self._file_handler.write_batch([log_entry])
                else:
                    self._fallback_logger.warning(f"Invalid log entry skipped: {log_entry}")
            except Exception as e:
                self._fallback_logger.error(f"Failed to record event: {str(e)}")
                self._fallback_logger.error(traceback.format_exc())
    
    def handle_error(self, record: ErrorRecord) -> None:
        """Handle error records from the ErrorRecordBridge."""
        if not LOGGING_ENABLED or not self.should_log("ERROR"):
            return
        with self._lock:
            try:
                # Construct error log entry
                error_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'conversation_id': str(uuid.uuid4()),
                    'event_type': 'error',
                    'message': record.error_message,
                    'level': 'error',
                    'error_type': record.error_type,
                    'stack_trace': record.stack_trace,
                    **(record.additional_info or {})
                }
                # Write error to log file
                if self._validator.validate_entry(error_entry):
                    self._file_handler.write_batch([error_entry])
                else:
                    self._fallback_logger.warning(f"Invalid error entry skipped: {error_entry}")
            except Exception as e:
                self._fallback_logger.error(f"Failed to handle error: {str(e)}")
                self._fallback_logger.error(traceback.format_exc())
    
    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, additional_info: Dict[str, Any] = None) -> None:
        """Log an error with detailed information."""
        if not LOGGING_ENABLED:
            return
        with self._lock:
            # Record through the bridge
            ErrorRecordBridge().record_error(
                error_type=error_type or "unknown_error",
                error_message=error_msg,
                stack_trace=stack_trace,
                additional_info=additional_info
            )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not LOGGING_ENABLED:
            return {}
        with self._lock:
            return {
                "error_counts": dict(ErrorRecordBridge()._error_counts),
                "recent_errors": [record.__dict__ for record in ErrorRecordBridge().get_recent_errors()]
            }
    
    def cleanup(self) -> None:
        """Clean up logging resources."""
        if not LOGGING_ENABLED:
            return
        with self._lock:
            try:
                self._file_handler.manage_rotation()
                self._file_handler.compress_logs()
            except Exception as e:
                self._fallback_logger.error(f"Failed to clean up logger: {str(e)}")
                self._fallback_logger.error(traceback.format_exc())
    
    def update_config(self, **kwargs) -> None:
        """Update logger configuration."""
        if not LOGGING_ENABLED:
            return
        with self._lock:
            try:
                self.config.update(**kwargs)
                # Reinitialize file handler with new config
                self._file_handler = _FileHandler(self.config, self._fallback_logger)
            except Exception as e:
                self._fallback_logger.error(f"Failed to update logger config: {str(e)}")
                self._fallback_logger.error(traceback.format_exc())