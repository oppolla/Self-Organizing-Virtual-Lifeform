from typing import Dict, Any, Optional
import torch
import time
import traceback
from threading import Lock
from dataclasses import dataclass
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_memory_manager import GPUMemoryManager, RAMManager

"""
Facade for hardware access, abstracting GPU and CPU operations to decouple
direct torch calls from other modules. Ensures compatibility in CUDA and non-CUDA
environments.
"""

class HardwareError(Exception):
    """Raised for hardware-related errors."""
    pass

@dataclass
class HardwareConfig:
    """Configuration for hardware manager."""
    enable_cuda: bool = True  # Whether to attempt CUDA operations
    memory_query_interval: float = 0.1  # Seconds between memory queries
    mock_memory_total_mb: float = 8192.0  # Mock total memory for non-CUDA environments

    def validate(self) -> None:
        """Validate configuration parameters."""
        try:
            if not isinstance(self.enable_cuda, bool):
                raise HardwareError("enable_cuda must be boolean")
            if self.memory_query_interval <= 0:
                raise HardwareError("memory_query_interval must be positive")
            if self.mock_memory_total_mb <= 0:
                raise HardwareError("mock_memory_total_mb must be positive")
        except HardwareError as e:
            raise HardwareError(f"Configuration validation failed: {str(e)}")

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'HardwareConfig':
        """Create configuration from ConfigManager."""
        try:
            config = cls(
                enable_cuda=config_manager.get("hardware.enable_cuda", torch.cuda.is_available()),
                memory_query_interval=config_manager.get("hardware.memory_query_interval", 0.1),
                mock_memory_total_mb=config_manager.get("hardware.mock_memory_total_mb", 8192.0)
            )
            config.validate()
            return config
        except Exception as e:
            raise HardwareError(f"Failed to create HardwareConfig: {str(e)}")

class HardwareManager:
    """Manages hardware access for memory and device operations."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize the hardware manager.

        Args:
            config_manager: Configuration manager instance.
            logger: Logger for event recording.
        """
        self.config_manager = config_manager
        self.logger = logger
        self._config = HardwareConfig.from_config_manager(config_manager)
        self._lock = Lock()
        self._last_memory_query: float = 0.0
        self._cached_memory_stats: Optional[Dict[str, float]] = None
        self._cuda_available = self._check_cuda_availability()
        self._log_training_event("hardware_initialized", {
            "cuda_available": self._cuda_available,
            "mock_memory_total_mb": self._config.mock_memory_total_mb
        })

    def _check_cuda_availability(self) -> bool:
        """
        Check if CUDA is available and enabled.

        Returns:
            True if CUDA is available and enabled, False otherwise.
        """
        try:
            if not self._config.enable_cuda:
                self._log_training_event("cuda_disabled", {"message": "CUDA disabled by configuration"})
                return False
            if not torch.cuda.is_available():
                self._log_training_event("cuda_unavailable", {"message": "CUDA not available on system"})
                return False
            return True
        except Exception as e:
            self._log_error("Failed to check CUDA availability", e)
            return False

    def get_memory_stats(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Get memory statistics for specified device."""
        try:
            if device and device.type == 'cuda':
                gpu_manager = GPUMemoryManager(self.config_manager, self.logger)
                gpu_stats = gpu_manager.get_gpu_usage()
                return {
                    'device': str(device),
                    'type': 'cuda',
                    'allocated': gpu_stats.get('gpu_usage', 0.0),
                    'available': gpu_stats.get('gpu_available', 0.0),
                    'usage_percentage': gpu_stats.get('usage_percentage', 0.0)
                }
            else:
                ram_manager = RAMManager(self.config_manager, self.logger)
                ram_stats = ram_manager.check_memory_health()
                return {
                    'device': 'cpu',
                    'type': 'cpu',
                    'total': ram_stats.get('total', 0.0),
                    'available': ram_stats.get('available', 0.0),
                    'usage_percentage': ram_stats.get('usage_percent', 0.0)
                }
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to get memory stats: {str(e)}",
                error_type="memory_stats_error",
                stack_trace=traceback.format_exc()
            )
            return {}

    def get_detailed_memory_stats(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Get detailed memory statistics (e.g., for CUDA memory_stats).

        Args:
            device: Target device. Uses default CUDA device or CPU if None.

        Returns:
            Dictionary with detailed memory stats or empty dict for CPU.
        """
        try:
            with self._lock:
                if self._cuda_available and (device is None or device.type == "cuda"):
                    device = device or torch.device("cuda:0")
                    stats = torch.cuda.memory_stats(device)
                    return {
                        "allocated_bytes_current": stats.get("allocated_bytes.all.current", 0),
                        "reserved_bytes_current": stats.get("reserved_bytes.all.current", 0),
                        "active_bytes_current": stats.get("active_bytes.all.current", 0),
                        "inactive_bytes_current": stats.get("inactive_bytes.all.current", 0)
                    }
                return {}
        except Exception as e:
            self._log_error("Failed to get detailed memory stats", e)
            return {}

    def clear_memory_cache(self, device: Optional[torch.device] = None) -> None:
        """
        Clear memory cache for the specified device or default CUDA.

        Args:
            device: Target device. Uses default CUDA device if None.

        Raises:
            HardwareError: If cache clearing fails.
        """
        try:
            with self._lock:
                if self._cuda_available and (device is None or device.type == "cuda"):
                    torch.cuda.empty_cache()
                    self._log_training_event("memory_cache_cleared", {
                        "device": str(device or "cuda:0")
                    })
        except Exception as e:
            self._log_error("Failed to clear memory cache", e)
            raise HardwareError(f"Clear memory cache failed: {str(e)}")

    def get_device_properties(self, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Get properties of the specified device or default CUDA/CPU.

        Args:
            device: Target device. Uses default CUDA device or CPU if None.

        Returns:
            Dictionary with device properties (e.g., name, total_memory_mb).
        """
        try:
            with self._lock:
                if self._cuda_available and (device is None or device.type == "cuda"):
                    device = device or torch.device("cuda:0")
                    props = torch.cuda.get_device_properties(device)
                    return {
                        "name": props.name,
                        "total_memory_mb": props.total_memory / 1024 / 1024,
                        "major": props.major,
                        "minor": props.minor
                    }
                return {
                    "name": "CPU",
                    "total_memory_mb": self._config.mock_memory_total_mb,
                    "major": 0,
                    "minor": 0
                }
        except Exception as e:
            self._log_error("Failed to get device properties", e)
            raise HardwareError(f"Device properties query failed: {str(e)}")

    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available and enabled.

        Returns:
            True if CUDA is available and enabled, False otherwise.
        """
        return self._cuda_available

    def get_default_device(self) -> torch.device:
        """
        Get the default device (CUDA if available, else CPU).

        Returns:
            Default torch.device.
        """
        try:
            return torch.device("cuda:0" if self._cuda_available else "cpu")
        except Exception as e:
            self._log_error("Failed to get default device", e)
            return torch.device("cpu")

    def _log_training_event(self, event_type: str, additional_info: Dict[str, Any], level: str = "info") -> None:
        """
        Log a training event with standardized metadata.

        Args:
            event_type: Type of the event.
            additional_info: Additional event data.
            level: Log level (debug, info, warning, error).
        """
        try:
            metadata = {
                "timestamp": time.time(),
                **additional_info
            }
            self.logger.log_training_event(
                event_type=f"hardware_{event_type}",
                message=f"Hardware event: {event_type}",
                level=level,
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log training event: {str(e)}")

    def _log_error(self, message: str, error: Exception, level: str = "error") -> None:
        """
        Log an error with standardized metadata.

        Args:
            message: Error message.
            error: Exception instance.
            level: Log level (error or warning).
        """
        try:
            metadata = {
                "error": str(error),
                "stack_trace": traceback.format_exc()
            }
            self.logger.log_error(
                error_msg=message,
                error_type="hardware_error",
                stack_trace=traceback.format_exc(),
                additional_info=metadata,
                level=level
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    from sovl_config import ConfigManager
    from sovl_logger import LoggingManager
    import unittest

    class TestHardwareManager(unittest.TestCase):
        def setUp(self):
            self.logger = LoggingManager("test_logs.jsonl")
            self.config_manager = ConfigManager("sovl_config.json", self.logger)
            self.hardware = HardwareManager(self.config_manager, self.logger)

        def test_memory_stats_cuda(self):
            if self.hardware.is_cuda_available():
                stats = self.hardware.get_memory_stats()
                self.assertIn("allocated_mb", stats)
                self.assertIn("reserved_mb", stats)
                self.assertIn("total_memory_mb", stats)
                self.assertIn("available_mb", stats)
                self.assertGreaterEqual(stats["total_memory_mb"], 0)
                self.assertGreaterEqual(stats["allocated_mb"], 0)

        def test_memory_stats_cpu(self):
            self.config_manager.set("hardware.enable_cuda", False)
            hardware = HardwareManager(self.config_manager, self.logger)
            stats = hardware.get_memory_stats()
            self.assertEqual(stats["total_memory_mb"], hardware._config.mock_memory_total_mb)
            self.assertGreaterEqual(stats["allocated_mb"], 0)

        def test_device_properties(self):
            props = self.hardware.get_device_properties()
            self.assertIn("name", props)
            self.assertIn("total_memory_mb", props)
            self.assertIn("major", props)
            self.assertIn("minor", props)

        def test_default_device(self):
            device = self.hardware.get_default_device()
            self.assertIsInstance(device, torch.device)
            if self.hardware.is_cuda_available():
                self.assertEqual(device.type, "cuda")
            else:
                self.assertEqual(device.type, "cpu")

    if __name__ == "__main__":
        unittest.main()
