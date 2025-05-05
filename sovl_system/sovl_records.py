from typing import Optional, Dict, Any, Deque, Callable, List
from collections import deque, defaultdict
import time
from dataclasses import dataclass, field
from threading import Lock, RLock
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, synchronized

"""
Module for error record management and bridging, decoupled from other modules to avoid circular dependencies.
Provides thread-safe storage, notification, and querying of error records.
"""

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
