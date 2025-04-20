import json
import os
import gzip
import uuid
import time
import logging
from datetime import datetime
from threading import Lock, RLock
from typing import List, Dict, Union, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import torch
import traceback
from collections import deque

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

    _RANGES = {
        "max_size_mb": (0, 100),
        "max_in_memory_logs": (100, 10000),
        "max_log_age_days": (1, 365),
        "prune_interval_hours": (1, 168),
        "memory_threshold_mb": (10, 1000),
        "gpu_memory_threshold": (0.1, 1.0)
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

class LogManager:
    """Singleton manager for logging operations."""
    _instance = None
    _lock = RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'):
                    self._log_queue = deque(maxlen=1000)
                    self._config = LoggerConfig()
                    self._file_handler = _FileHandler(self._config)
                    self._validator = _LogValidator()
                    self._write_lock = Lock()
                    self._initialized = True
    
    def configure(self, config: LoggerConfig) -> None:
        """Configure the logger with new settings."""
        with self._lock:
            self._config = config
            self._file_handler = _FileHandler(config)
    
    def log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Thread-safe event logging."""
        entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'level': level,
            'message': message,
            **kwargs
        }
        
        with self._lock:
            if self._validator.validate_entry(entry):
                self._log_queue.append(entry)
                self._write_batch()
    
    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, **kwargs) -> None:
        """Thread-safe error logging."""
        entry = {
            'timestamp': time.time(),
            'event_type': 'error',
            'level': 'error',
            'error_msg': error_msg,
            'error_type': error_type or 'unknown_error',
            'stack_trace': stack_trace,
            **kwargs
        }
        
        with self._lock:
            if self._validator.validate_entry(entry):
                self._log_queue.append(entry)
                self._write_batch()
    
    def _write_batch(self) -> None:
        """Thread-safe batch writing."""
        with self._write_lock:
            if not self._log_queue:
                return
            
            entries = list(self._log_queue)
            self._log_queue.clear()
            
            try:
                self._file_handler.write_batch(entries)
            except Exception as e:
                logging.error(f"Failed to write log batch: {str(e)}")

class Logger(ILoggerClient):
    """Main logger class for the SOVL system."""
    _instance = None
    _lock = RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
            
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._log_queue = deque(maxlen=1000)
            self._event_queue = deque(maxlen=100)
            self._error_queue = deque(maxlen=50)
            self._debug_mode = False
            self._log_level = logging.INFO
            self._lock = RLock()
            
    def set_level(self, level: int) -> None:
        """Set the logging level."""
        with self._lock:
            self._log_level = level
            self._debug_mode = level == logging.DEBUG
            
    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        return self._debug_mode
        
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events from the event queue."""
        with self._lock:
            return list(self._event_queue)[-limit:]
            
    def get_recent_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors from the error queue."""
        with self._lock:
            return list(self._error_queue)[-limit:]
            
    def get_debug_stats(self) -> Dict[str, Any]:
        """Get debug statistics about the logging system."""
        with self._lock:
            return {
                "debug_mode": self._debug_mode,
                "log_level": self._log_level,
                "queue_sizes": {
                    "log_queue": len(self._log_queue),
                    "event_queue": len(self._event_queue),
                    "error_queue": len(self._error_queue)
                },
                "queue_limits": {
                    "log_queue": self._log_queue.maxlen,
                    "event_queue": self._event_queue.maxlen,
                    "error_queue": self._error_queue.maxlen
                }
            }
            
    def record_event(self, event_type: str, message: str, level: str = "info", additional_info: Dict[str, Any] = None) -> None:
        """Record an event with timestamp and optional additional information."""
        with self._lock:
            event = {
                "event_type": event_type,
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat()
            }
            if additional_info:
                event["additional_info"] = additional_info
                
            self._event_queue.append(event)
            
            if level == "error":
                self._error_queue.append(event)
                
            if self._debug_mode or level in ["error", "warning"]:
                print(f"[{level.upper()}] {event_type}: {message}")
                if additional_info:
                    print(f"Additional Info: {json.dumps(additional_info, indent=2)}")
                    
    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, additional_info: Dict[str, Any] = None) -> None:
        """Log an error with detailed information."""
        with self._lock:
            error = {
                "error_type": error_type or "unknown_error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat(),
                "stack_trace": stack_trace
            }
            if additional_info:
                error["additional_info"] = additional_info
                
            self._error_queue.append(error)
            
            if self._debug_mode:
                print(f"[ERROR] {error_type}: {error_msg}")
                if stack_trace:
                    print(f"Stack Trace:\n{stack_trace}")
                if additional_info:
                    print(f"Additional Info: {json.dumps(additional_info, indent=2)}")
                    
    def clear_queues(self) -> None:
        """Clear all event and error queues."""
        with self._lock:
            self._log_queue.clear()
            self._event_queue.clear()
            self._error_queue.clear()
            
    def get_log_level_name(self) -> str:
        """Get the current log level name."""
        level_names = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL"
        }
        return level_names.get(self._log_level, "UNKNOWN")

class LoggingManager:
    """Manages logging setup and configuration for the SOVL system."""

    _DEFAULT_CONFIG = {
        "logging.max_size_mb": 10,
        "logging.compress_old": True,
        "logging.rotation_count": 5,
        "logging.level": "INFO"
    }

    def __init__(
        self,
        config_manager: ConfigManager,
        log_dir: str = "logs",
        system_log_file: str = "sovl_system.log",
        debug_log_file: str = "sovl_debug.log"
    ):
        """
        Initialize the logging manager with configuration and file paths.

        Args:
            config_manager: ConfigManager instance for accessing logging settings.
            log_dir: Directory to store log files.
            system_log_file: Name of the system log file.
            debug_log_file: Name of the debug log file.
        """
        if not isinstance(config_manager, ConfigManager):
            raise TypeError("config_manager must be an instance of ConfigManager")
            
        self.config_manager = config_manager
        self.log_dir = log_dir
        self.system_log_file = os.path.join(log_dir, system_log_file)
        self.debug_log_file = os.path.join(log_dir, debug_log_file)
        self.loggers = {}
        
        # Configure fallback logger first
        self._configure_fallback_logger()
        
        # Ensure logging configuration exists
        self._ensure_logging_config()
        self._validate_config()
        self._setup_logging()

    def _configure_fallback_logger(self) -> None:
        """Configure the fallback logger with proper handlers and formatting."""
        # Get the fallback logger
        fallback_logger = logging.getLogger(__name__)
        
        # Set log level from config or default to INFO
        log_level = self.config_manager.get(
            "logging.level",
            self._DEFAULT_CONFIG["logging.level"]
        )
        fallback_logger.setLevel(getattr(logging, log_level))
        
        # Remove any existing handlers to avoid duplicates
        for handler in fallback_logger.handlers[:]:
            fallback_logger.removeHandler(handler)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Create detailed formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        fallback_logger.addHandler(console_handler)
        
        # Log successful configuration
        fallback_logger.info(
            f"Configured fallback logger with level {log_level}"
        )

    def _ensure_logging_config(self) -> None:
        """Ensure logging configuration exists in config manager."""
        try:
            # Get or create logging section
            logging_config = self.config_manager.get_section("logging", {})
            
            # Set default values if not present
            for key, default_value in self._DEFAULT_CONFIG.items():
                if key not in logging_config:
                    self.config_manager.set(key, default_value)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to ensure logging configuration: {str(e)}")

    def _validate_config(self) -> None:
        """Validate logging configuration."""
        try:
            max_size = self.config_manager.get(
                "logging.max_size_mb",
                self._DEFAULT_CONFIG["logging.max_size_mb"]
            )
            if max_size <= 0:
                raise ValueError("Log file size must be positive")
        except Exception as e:
            print(f"Logging configuration validation failed: {str(e)}")
            raise

    def _setup_logging(self) -> None:
        """Set up logging configuration and handlers."""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Validate and get configuration values
        try:
            max_size_mb = self.config_manager.get(
                "logging.max_size_mb",
                self._DEFAULT_CONFIG["logging.max_size_mb"]
            )
            if not isinstance(max_size_mb, (int, float)) or max_size_mb <= 0:
                raise ValueError(
                    f"Invalid logging.max_size_mb: {max_size_mb}. Must be a positive number."
                )

            compress_old = self.config_manager.get(
                "logging.compress_old",
                self._DEFAULT_CONFIG["logging.compress_old"]
            )
            if not isinstance(compress_old, bool):
                raise ValueError(
                    f"Invalid logging.compress_old: {compress_old}. Must be a boolean."
                )

            rotation_count = self.config_manager.get(
                "logging.rotation_count",
                self._DEFAULT_CONFIG["logging.rotation_count"]
            )
            if not isinstance(rotation_count, int) or rotation_count < 0:
                raise ValueError(
                    f"Invalid logging.rotation_count: {rotation_count}. Must be a non-negative integer."
                )

            # Configure system logger with validated settings
            system_config = LoggerConfig(
                log_file=self.system_log_file,
                max_size_mb=max_size_mb,
                compress_old=compress_old,
                max_in_memory_logs=1000,
                rotation_count=rotation_count
            )
            self.loggers["system"] = Logger(system_config)

            # Configure debug logger with same validated settings
            debug_config = LoggerConfig(
                log_file=self.debug_log_file,
                max_size_mb=max_size_mb,
                compress_old=compress_old,
                max_in_memory_logs=1000,
                rotation_count=rotation_count
            )
            self.loggers["debug"] = Logger(debug_config)

            # Log successful setup
            self.loggers["system"].log_event(
                event_type="logging_initialized",
                message="Logging system initialized successfully",
                level="info",
                additional_info={
                    "max_size_mb": max_size_mb,
                    "compress_old": compress_old,
                    "rotation_count": rotation_count
                }
            )

        except ValueError as e:
            # Log configuration error and re-raise
            logging.getLogger(__name__).error(
                f"Invalid logging configuration: {str(e)}"
            )
            raise
        except Exception as e:
            # Log unexpected error and re-raise
            logging.getLogger(__name__).error(
                f"Failed to setup logging: {str(e)}",
                exc_info=True
            )
            raise

    def setup_logging(self) -> Logger:
        """
        Set up and return the main logger.

        Returns:
            The main logger instance
        """
        if not self.loggers:
            self._setup_logging()
        return self.loggers["system"]

    def get_logger(self, name: str) -> Logger:
        """
        Get a logger instance by name.

        Args:
            name: The name of the logger to retrieve.

        Returns:
            The requested logger instance.

        Raises:
            KeyError: If the logger name is not found.
        """
        if name not in self.loggers:
            raise KeyError(f"Logger '{name}' not found")
        return self.loggers[name]

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update logging configuration and reconfigure loggers.

        Args:
            new_config: Dictionary containing new configuration values.
        """
        # Update configuration
        for key, value in new_config.items():
            if key.startswith("logging."):
                self.config_manager.set(key, value)

        # Revalidate and reconfigure
        self._validate_config()
        self._setup_logging()

    def log_error(self, error_msg: str, error_type: str = None, stack_trace: str = None, 
                  conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log an error with standardized format."""
        self.get_logger("system").log_error(
            error_msg=error_msg,
            error_type=error_type,
            stack_trace=stack_trace,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_memory_usage(self, phase: str, device: torch.device, **kwargs) -> None:
        """Log memory usage for a specific phase and device."""
        try:
            # Get memory stats from appropriate manager based on device
            if device.type == 'cuda':
                gpu_manager = kwargs.get('gpu_manager')
                if gpu_manager:
                    gpu_stats = gpu_manager.get_gpu_usage()
                    memory_stats = {
                        'gpu_usage': gpu_stats.get('usage', 0.0),
                        'gpu_allocated': gpu_stats.get('allocated', 0.0),
                        'gpu_cached': gpu_stats.get('cached', 0.0)
                    }
                else:
                    memory_stats = {'gpu_usage': 0.0, 'gpu_allocated': 0.0, 'gpu_cached': 0.0}
            else:
                ram_manager = kwargs.get('ram_manager')
                if ram_manager:
                    ram_stats = ram_manager.check_memory_health()
                    memory_stats = {
                        'ram_usage': ram_stats.get('usage', 0.0),
                        'ram_available': ram_stats.get('available', 0.0),
                        'ram_total': ram_stats.get('total', 0.0)
                    }
                else:
                    memory_stats = {'ram_usage': 0.0, 'ram_available': 0.0, 'ram_total': 0.0}

            # Add memoria stats if available
            memoria_manager = kwargs.get('memoria_manager')
            if memoria_manager:
                memoria_stats = memoria_manager.get_state()
                memory_stats.update({
                    'memoria_usage': memoria_stats.get('usage', 0.0),
                    'memoria_capacity': memoria_stats.get('capacity', 0.0)
                })

            self.record_event(
                event_type="memory_usage",
                message=f"Memory usage for {phase} on {device.type}",
                level="info",
                additional_info={
                    "phase": phase,
                    "device": str(device),
                    "memory_stats": memory_stats,
                    **kwargs
                }
            )
        except Exception as e:
            self.log_error(
                error_msg=f"Failed to log memory usage: {str(e)}",
                error_type="memory_logging_error",
                stack_trace=traceback.format_exc()
            )

    def log_memory_health(self, model_size: int, **kwargs) -> None:
        """Log comprehensive memory health metrics."""
        try:
            memory_health = {}
            
            # Get GPU memory health
            gpu_manager = kwargs.get('gpu_manager')
            if gpu_manager:
                gpu_stats = gpu_manager.get_gpu_usage()
                memory_health['gpu'] = {
                    'usage': gpu_stats.get('usage', 0.0),
                    'allocated': gpu_stats.get('allocated', 0.0),
                    'cached': gpu_stats.get('cached', 0.0),
                    'model_size': model_size
                }
            
            # Get RAM health
            ram_manager = kwargs.get('ram_manager')
            if ram_manager:
                ram_stats = ram_manager.check_memory_health()
                memory_health['ram'] = {
                    'usage': ram_stats.get('usage', 0.0),
                    'available': ram_stats.get('available', 0.0),
                    'total': ram_stats.get('total', 0.0)
                }
            
            # Get Memoria health
            memoria_manager = kwargs.get('memoria_manager')
            if memoria_manager:
                memoria_stats = memoria_manager.get_state()
                memory_health['memoria'] = {
                    'usage': memoria_stats.get('usage', 0.0),
                    'capacity': memoria_stats.get('capacity', 0.0)
                }
            
            # Get overall memory monitor stats if available
            memory_monitor = kwargs.get('memory_monitor')
            if memory_monitor:
                monitor_stats = memory_monitor.get_memory_usage()
                memory_health['monitor'] = {
                    'cpu_usage': monitor_stats.get('cpu_usage', 0.0),
                    'ram_usage': monitor_stats.get('ram_usage', 0.0),
                    'gpu_usage': monitor_stats.get('gpu_usage', 0.0)
                }

            self.record_event(
                event_type="memory_health",
                message="Comprehensive memory health check",
                level="info",
                additional_info={
                    "memory_health": memory_health,
                    "model_size": model_size,
                    **kwargs
                }
            )
        except Exception as e:
            self.log_error(
                error_msg=f"Failed to log memory health: {str(e)}",
                error_type="memory_health_logging_error",
                stack_trace=traceback.format_exc()
            )

    def log_training_event(self, event_type: str, epoch: int = None, loss: float = None,
                          batch_size: int = None, data_exposure: float = None,
                          conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log training-related events."""
        self.get_logger("system").log_training_event(
            event_type=event_type,
            epoch=epoch,
            loss=loss,
            batch_size=batch_size,
            data_exposure=data_exposure,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_generation_event(self, prompt: str, response: str, confidence_score: float,
                            generation_params: dict = None, conversation_id: str = None,
                            state_hash: str = None, **kwargs) -> None:
        """Log generation-related events."""
        self.get_logger("system").log_generation_event(
            prompt=prompt,
            response=response,
            confidence_score=confidence_score,
            generation_params=generation_params,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def log_cleanup_event(self, phase: str, success: bool, error: str = None,
                         conversation_id: str = None, state_hash: str = None, **kwargs) -> None:
        """Log cleanup-related events."""
        self.get_logger("system").log_cleanup_event(
            phase=phase,
            success=success,
            error=error,
            conversation_id=conversation_id,
            state_hash=state_hash,
            **kwargs
        )

    def record(self, entry: Dict) -> None:
        """Write a validated log entry."""
        self.get_logger("system").record(entry)

    def cleanup(self) -> None:
        """Clean up logging resources for all loggers."""
        for name, logger in self.loggers.items():
            try:
                logger.cleanup()
                self.get_logger("system").log_event(
                    event_type="logger_cleanup",
                    message=f"Successfully cleaned up logger {name}",
                    level="info"
                )
            except Exception as e:
                # Use system logger to log the error
                self.get_logger("system").log_error(
                    error_msg=f"Failed to clean up logger {name}: {str(e)}",
                    error_type="cleanup_error",
                    stack_trace=traceback.format_exc()
                )
                # Also log to debug logger for more detailed information
                self.get_logger("debug").log_event(
                    event_type="logger_cleanup_error",
                    message=f"Detailed error during cleanup of logger {name}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "logger_name": name,
                        "stack_trace": traceback.format_exc()
                    }
                )

class LoggingError(Exception):
    """Raised for logging-related errors."""
    pass
