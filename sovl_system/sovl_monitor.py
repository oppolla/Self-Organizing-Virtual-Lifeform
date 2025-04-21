from typing import Dict, Any, List
from threading import Thread, Event, Lock
from collections import deque
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_trainer import TrainingCycleManager
from sovl_curiosity import CuriosityManager
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_events import MemoryEventDispatcher, MemoryEventTypes
from sovl_state import SOVLState, StateManager
import time
import traceback

class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize system monitor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        try:
            # Get memory stats from managers
            ram_stats = self.ram_manager.check_memory_health()
            gpu_stats = self.gpu_manager.check_memory_health()
            
            return {
                "ram_stats": ram_stats,
                "gpu_stats": gpu_stats
            }
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to collect metrics: {str(e)}",
                error_type="metrics_collection_error",
                stack_trace=traceback.format_exc()
            )
            return {
                "ram_stats": {"status": "error"},
                "gpu_stats": {"status": "error"}
            }

class MemoryMonitor:
    """Monitors system memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize the memory monitor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
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
