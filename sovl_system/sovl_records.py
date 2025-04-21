from typing import Optional, Dict, Any, Deque, Callable, List
from collections import deque, defaultdict
import time
import hashlib
import json
import os
import threading
import traceback
from dataclasses import dataclass, field
from threading import Lock, RLock
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, synchronized

"""
Module for managing confidence history, decoupling it from sovl_state.py and
sovl_processor.py to resolve mutual dependencies. Provides thread-safe storage,
serialization, and backup of confidence scores.
"""

class HistoryError(Exception):
    """Base exception for confidence history errors."""
    pass

class SerializationError(HistoryError):
    """Raised for serialization/deserialization failures."""
    pass

class ValidationError(HistoryError):
    """Raised for history validation failures."""
    pass

@dataclass
class ConfidenceHistoryConfig:
    """Configuration for confidence history management."""
    max_confidence_history: int
    confidence_history_file: str = "confidence_history.json"
    backup_interval: float = 300.0  # Seconds between backups
    enable_persistence: bool = True  # Whether to save to disk
    strict_validation: bool = True  # Enforce strict validation on load

    def validate(self) -> None:
        """Validate configuration parameters."""
        try:
            if self.max_confidence_history <= 0:
                raise ValidationError("max_confidence_history must be positive")
            if not isinstance(self.confidence_history_file, str) or not self.confidence_history_file.endswith(".json"):
                raise ValidationError("confidence_history_file must be a valid JSON file path")
            if self.backup_interval <= 0:
                raise ValidationError("backup_interval must be positive")
            if not isinstance(self.enable_persistence, bool):
                raise ValidationError("enable_persistence must be boolean")
            if not isinstance(self.strict_validation, bool):
                raise ValidationError("strict_validation must be boolean")
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {str(e)}")

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ConfidenceHistoryConfig':
        """Create configuration from ConfigManager."""
        try:
            config = cls(
                max_confidence_history=config_manager.get("controls_config.confidence_history_maxlen", 5),
                confidence_history_file=config_manager.get("controls_config.confidence_history_file", "confidence_history.json"),
                backup_interval=config_manager.get("controls_config.history_backup_interval", 300.0),
                enable_persistence=config_manager.get("controls_config.history_enable_persistence", True),
                strict_validation=config_manager.get("controls_config.history_strict_validation", True)
            )
            config.validate()
            return config
        except Exception as e:
            raise HistoryError(f"Failed to create ConfidenceHistoryConfig: {str(e)}")

class ConfidenceHistory:
    """Thread-safe manager for confidence history with persistence and backups."""
    
    VERSION = "1.0"

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the confidence history manager.

        Args:
            config_manager: Configuration manager instance.
        """
        self.config_manager = config_manager
        self._config = ConfidenceHistoryConfig.from_config_manager(config_manager)
        self._confidence_history: Deque[float] = deque(maxlen=self._config.max_confidence_history)
        self._lock = Lock()
        self._last_update_time: float = time.time()
        self._cached_hash: Optional[str] = None
        self._initialize_history()
        if self._config.enable_persistence:
            self._start_backup_thread()

    @synchronized("_lock")
    def _initialize_history(self) -> None:
        """Initialize or load confidence history from file."""
        try:
            if not self._config.enable_persistence:
                return
            history_file = self._config.confidence_history_file
            if os.path.exists(history_file):
                try:
                    self._load_history(history_file)
                except SerializationError:
                    print(f"[ERROR] Corrupted history file, falling back to backup or empty history")
                    backup_file = f"{history_file}.backup"
                    if os.path.exists(backup_file):
                        self._load_history(backup_file)
            self._validate_history()
            self._cached_hash = None  # Invalidate cache
        except Exception as e:
            print(f"[ERROR] History initialization failed: {str(e)}")
            raise HistoryError(f"History initialization failed: {str(e)}")

    @synchronized("_lock")
    def add_confidence(self, confidence: float) -> None:
        """
        Add a confidence score to the history.

        Args:
            confidence: Confidence score in [0.0, 1.0].

        Raises:
            ValidationError: If the confidence score is invalid.
        """
        try:
            with NumericalGuard():
                if not isinstance(confidence, (int, float)):
                    raise ValidationError(f"Confidence must be a number, got {type(confidence)}")
                if not 0.0 <= confidence <= 1.0:
                    raise ValidationError(f"Confidence must be in [0.0, 1.0], got {confidence}")
                self._confidence_history.append(float(confidence))
                self._last_update_time = time.time()
                self._cached_hash = None  # Invalidate cache
        except Exception as e:
            print(f"[ERROR] Failed to add confidence score: {str(e)}")
            raise ValidationError(f"Add confidence failed: {str(e)}")

    @synchronized("_lock")
    def get_confidence_history(self) -> Deque[float]:
        """
        Retrieve the current confidence history.

        Returns:
            Deque of confidence scores.
        """
        return self._confidence_history

    @synchronized("_lock")
    def clear_history(self) -> None:
        """Clear the confidence history."""
        try:
            self._confidence_history.clear()
            self._last_update_time = time.time()
            self._cached_hash = None
        except Exception as e:
            print(f"[ERROR] Failed to clear history: {str(e)}")
            raise HistoryError(f"Clear history failed: {str(e)}")

    @synchronized("_lock")
    def save_history(self, file_path: Optional[str] = None) -> None:
        """
        Save the confidence history to a file.

        Args:
            file_path: Optional file path. Uses config path if None.

        Raises:
            SerializationError: If saving fails.
        """
        if not self._config.enable_persistence:
            return
        try:
            file_path = file_path or self._config.confidence_history_file
            history_dict = self.to_dict()
            with open(file_path, 'w') as f:
                json.dump(history_dict, f)
        except Exception as e:
            print(f"[ERROR] Failed to save history: {str(e)}")
            raise SerializationError(f"Failed to save history: {str(e)}")

    @synchronized("_lock")
    def _load_history(self, file_path: str) -> None:
        """Load confidence history from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.from_dict(data)
        except Exception as e:
            print(f"[ERROR] Failed to load history: {str(e)}")
            raise SerializationError(f"Failed to load history: {str(e)}")

    @synchronized("_lock")
    def _validate_history(self) -> None:
        """Validate the loaded history."""
        try:
            for confidence in self._confidence_history:
                if not isinstance(confidence, (int, float)):
                    raise ValidationError(f"Invalid confidence value type: {type(confidence)}")
                if not 0.0 <= confidence <= 1.0:
                    raise ValidationError(f"Confidence out of range: {confidence}")
        except Exception as e:
            print(f"[WARNING] Validation failed, continuing with partial history: {str(e)}")
            if self._config.strict_validation:
                raise ValidationError(f"History validation failed: {str(e)}")

    def get_history_hash(self) -> str:
        """Generate a hash of the current history state."""
        try:
            if self._cached_hash is None:
                history_str = json.dumps(list(self._confidence_history))
                self._cached_hash = hashlib.sha256(history_str.encode()).hexdigest()
            return self._cached_hash
        except Exception as e:
            print(f"[ERROR] Failed to generate history hash: {str(e)}")
            return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary for serialization."""
        try:
            return {
                "version": self.VERSION,
                "confidence_history": list(self._confidence_history),
                "last_update": self._last_update_time
            }
        except Exception as e:
            print(f"[ERROR] History serialization failed: {str(e)}")
            raise SerializationError(f"History serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load history from dictionary."""
        try:
            version = data.get("version", "1.0")
            if version != self.VERSION:
                self._migrate_history(data, version)
            else:
                self._confidence_history = deque(data["confidence_history"], maxlen=self._config.max_confidence_history)
                self._last_update_time = data.get("last_update", time.time())
                self._cached_hash = None
        except Exception as e:
            print(f"[ERROR] Failed to load history from dictionary: {str(e)}")
            raise SerializationError(f"Failed to load history from dictionary: {str(e)}")

    def _migrate_history(self, data: Dict[str, Any], version: str) -> None:
        """Migrate history from older version."""
        try:
            if version == "1.0":
                self._confidence_history = deque(data["confidence_history"], maxlen=self._config.max_confidence_history)
                self._last_update_time = data.get("last_update", time.time())
                self._cached_hash = None
            else:
                raise SerializationError(f"Unsupported history version: {version}")
        except Exception as e:
            print(f"[ERROR] History migration failed: {str(e)}")
            raise SerializationError(f"History migration failed: {str(e)}")

    def reset(self) -> None:
        """Reset the history to initial state."""
        try:
            self._confidence_history.clear()
            self._last_update_time = time.time()
            self._cached_hash = None
        except Exception as e:
            print(f"[ERROR] Failed to reset history: {str(e)}")
            raise HistoryError(f"Failed to reset history: {str(e)}")

    def _start_backup_thread(self) -> None:
        """Start the backup thread."""
        if not self._config.enable_persistence:
            return

        def backup_loop():
            while True:
                try:
                    time.sleep(self._config.backup_interval)
                    self.save_history(f"{self._config.confidence_history_file}.backup")
                except Exception as e:
                    print(f"[ERROR] Backup thread error: {str(e)}")
                    time.sleep(60)  # Wait a minute before retrying

        try:
            thread = threading.Thread(target=backup_loop, daemon=True)
            thread.start()
        except Exception as e:
            print(f"[ERROR] Failed to start backup thread: {str(e)}")
            raise HistoryError(f"Failed to start backup thread: {str(e)}")

@dataclass
class ErrorRecord:
    """Data class for error records."""
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    timestamp: float
    additional_info: Dict[str, Any] = field(default_factory=dict)

class IErrorHandler:
    """Interface for error handlers."""
    def handle_error(self, record: ErrorRecord) -> None:
        """Handle an error record."""
        pass

class ErrorRecordBridge:
    """Bridge for error recording to avoid circular dependencies."""
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
            self._handlers: List[IErrorHandler] = []
            self._error_counts = defaultdict(int)
            self._error_history = defaultdict(lambda: deque(maxlen=10))
            self._recent_errors = deque(maxlen=100)

    def register_handler(self, handler: IErrorHandler) -> None:
        """Register an error handler."""
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)

    def unregister_handler(self, handler: IErrorHandler) -> None:
        """Unregister an error handler."""
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def record_error(self, error_type: str, error_message: str, 
                    stack_trace: Optional[str] = None,
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Record an error and notify handlers."""
        with self._lock:
            record = ErrorRecord(
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                timestamp=time.time(),
                additional_info=additional_info or {}
            )
            
            # Update error tracking
            self._error_counts[error_type] += 1
            self._error_history[error_type].append(record)
            self._recent_errors.append(record)
            
            # Notify handlers
            for handler in self._handlers:
                try:
                    handler.handle_error(record)
                except Exception as e:
                    print(f"[ERROR] Error handler failed: {str(e)}")

    def get_error_count(self, error_type: str) -> int:
        """Get the count of errors of a specific type."""
        with self._lock:
            return self._error_counts[error_type]

    def get_recent_errors(self) -> List[ErrorRecord]:
        """Get the most recent errors."""
        with self._lock:
            return list(self._recent_errors)

    def get_error_history(self, error_type: str) -> List[ErrorRecord]:
        """Get the history of errors of a specific type."""
        with self._lock:
            return list(self._error_history[error_type])

# Example usage and testing
if __name__ == "__main__":
    import unittest

    class TestConfidenceHistory(unittest.TestCase):
        def setUp(self):
            self.config_manager = ConfigManager("sovl_config.json")
            self.history = ConfidenceHistory(self.config_manager)

        def test_add_confidence(self):
            self.history.add_confidence(0.85)
            self.assertEqual(len(self.history.get_confidence_history()), 1)
            self.assertEqual(self.history.get_confidence_history()[0], 0.85)

        def test_save_load(self):
            self.history.add_confidence(0.92)
            self.history.save_history("test_history.json")
            new_history = ConfidenceHistory(self.config_manager)
            new_history._load_history("test_history.json")
            self.assertEqual(list(new_history.get_confidence_history()), [0.92])

        def test_reset(self):
            self.history.add_confidence(0.78)
            self.history.reset()
            self.assertEqual(len(self.history.get_confidence_history()), 0)

        def test_invalid_confidence(self):
            with self.assertRaises(ValidationError):
                self.history.add_confidence(1.5)
            with self.assertRaises(ValidationError):
                self.history.add_confidence("invalid")

    if __name__ == "__main__":
        unittest.main()
