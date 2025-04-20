import torch
import json
import os
from collections import deque, defaultdict
from threading import Lock
import time
import traceback
from typing import Optional, Dict, List, Tuple, Any, Union
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_utils import memory_usage, safe_divide
from sovl_config import ConfigManager
from sovl_hardware import HardwareManager
import gc
import torch.cuda as cuda

class RAMManager:
    """Manages RAM memory for the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize RAMManager with configuration and logger."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self.hardware = HardwareManager(config_manager, logger)
        
        # Initialize RAM
        self._initialize_ram()
        
        # Log initialization
        self._logger.record_event(
            event_type="ram_manager_initialized",
            message="RAM manager initialized",
            level="info"
        )

    def _initialize_ram(self) -> None:
        """Initialize RAM management systems."""
        with self._memory_lock:
            try:
                # Get RAM config section
                ram_config = self._config_manager.get_section("ram_config", {})
                
                # Initialize RAM-specific parameters
                self.memory_threshold = ram_config.get("memory_threshold", 0.85)
                self.memory_decay_rate = ram_config.get("memory_decay_rate", 0.95)
                self.max_batch_size = ram_config.get("max_batch_size", 32)
                self.initial_batch_size = ram_config.get("initial_batch_size", 8)
                self.batch_size = self.initial_batch_size
                
                # Log successful initialization
                self._logger.record_event(
                    event_type="ram_initialized",
                    message="RAM initialized successfully",
                    level="info"
                )
                
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Failed to initialize RAM: {str(e)}",
                    error_type="ram_error",
                    stack_trace=traceback.format_exc()
                )
                raise

    def check_memory_health(self) -> Dict[str, float]:
        """Check RAM health and return metrics."""
        try:
            # Get current memory usage
            memory_stats = self.hardware.get_detailed_memory_stats()
            
            # Calculate memory health metrics
            ram_usage = memory_stats.get("ram_usage", 0.0)
            ram_available = memory_stats.get("ram_available", 0.0)
            ram_total = memory_stats.get("ram_total", 0.0)
            
            # Calculate health score
            health_score = 1.0 - (ram_usage / ram_total) if ram_total > 0 else 0.0
            
            # Log health check
            self._logger.record_event(
                event_type="ram_health_check",
                message="RAM health check completed",
                level="info",
                additional_info={
                    "ram_usage": ram_usage,
                    "ram_available": ram_available,
                    "health_score": health_score
                }
            )
            
            return {
                "ram_usage": ram_usage,
                "ram_available": ram_available,
                "health_score": health_score
            }
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to check RAM health: {str(e)}",
                error_type="health_check_error",
                stack_trace=traceback.format_exc()
            )
            raise

class GPUMemoryManager:
    """Manages GPU memory for the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize GPUMemoryManager with configuration and logger."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self.hardware = HardwareManager(config_manager, logger)
        self._allocated_memory = {}  # Track allocated memory pointers
        
        # Initialize GPU
        self._initialize_gpu()
        
        # Log initialization
        self._logger.record_event(
            event_type="gpu_manager_initialized",
            message="GPU manager initialized",
            level="info"
        )

    def _initialize_gpu(self) -> None:
        """Initialize GPU memory management systems."""
        with self._memory_lock:
            try:
                # Get GPU config section
                gpu_config = self._config_manager.get_section("gpu_config", {})
                
                # Initialize GPU-specific parameters
                self.gpu_threshold = gpu_config.get("gpu_threshold", 0.85)
                self.gpu_decay_rate = gpu_config.get("gpu_decay_rate", 0.95)
                self.max_gpu_memory = gpu_config.get("max_gpu_memory_mb", 1024) * 1024 * 1024  # Convert MB to bytes
                
                # Log successful initialization
                self._logger.record_event(
                    event_type="gpu_initialized",
                    message="GPU initialized successfully",
                    level="info"
                )
                
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Failed to initialize GPU: {str(e)}",
                    error_type="gpu_error",
                    stack_trace=traceback.format_exc()
                )
                raise

    def allocate_gpu_memory(self, size_bytes: int) -> Optional[int]:
        """Allocate GPU memory using CUDA."""
        try:
            if not torch.cuda.is_available():
                self._logger.log_error(
                    error_msg="CUDA is not available",
                    error_type="cuda_error"
                )
                return None
                
            with self._memory_lock:
                # Check if we have enough memory
                current_usage = self.get_detailed_gpu_memory_stats()
                if current_usage['allocated'] + size_bytes > self.max_gpu_memory:
                    self._logger.log_error(
                        error_msg="Not enough GPU memory available",
                        error_type="memory_allocation_error"
                    )
                    return None
                
                # Allocate memory
                ptr = cuda.malloc(size_bytes)
                self._allocated_memory[ptr] = size_bytes
                
                self._logger.record_event(
                    event_type="gpu_memory_allocated",
                    message=f"Allocated {size_bytes} bytes of GPU memory",
                    level="info",
                    additional_info={
                        "size_bytes": size_bytes,
                        "total_allocated": current_usage['allocated'] + size_bytes
                    }
                )
                
                return ptr
                
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to allocate GPU memory: {str(e)}",
                error_type="memory_allocation_error",
                stack_trace=traceback.format_exc()
            )
            return None

    def free_gpu_memory(self, ptr: int) -> bool:
        """Free GPU memory using CUDA."""
        try:
            if not torch.cuda.is_available():
                self._logger.log_error(
                    error_msg="CUDA is not available",
                    error_type="cuda_error"
                )
                return False
                
            with self._memory_lock:
                if ptr not in self._allocated_memory:
                    self._logger.log_error(
                        error_msg="Attempted to free unallocated memory",
                        error_type="memory_free_error"
                    )
                    return False
                
                # Free memory
                cuda.free(ptr)
                size_bytes = self._allocated_memory.pop(ptr)
                
                self._logger.record_event(
                    event_type="gpu_memory_freed",
                    message=f"Freed {size_bytes} bytes of GPU memory",
                    level="info",
                    additional_info={
                        "size_bytes": size_bytes,
                        "total_allocated": self.get_detailed_gpu_memory_stats()['allocated']
                    }
                )
                
                return True
                
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to free GPU memory: {str(e)}",
                error_type="memory_free_error",
                stack_trace=traceback.format_exc()
            )
            return False

    def get_detailed_gpu_memory_stats(self) -> Dict[str, int]:
        """Get detailed GPU memory statistics using CUDA."""
        try:
            if not torch.cuda.is_available():
                return {
                    'allocated': 0,
                    'reserved': 0,
                    'max_allocated': 0,
                    'cached': 0
                }
                
            with self._memory_lock:
                stats = {
                    'allocated': cuda.memory_allocated(),
                    'reserved': cuda.memory_reserved(),
                    'max_allocated': cuda.max_memory_allocated(),
                    'cached': cuda.memory_cached()
                }
                
                self._logger.record_event(
                    event_type="gpu_memory_stats",
                    message="Retrieved detailed GPU memory statistics",
                    level="info",
                    additional_info=stats
                )
                
                return stats
                
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to get GPU memory stats: {str(e)}",
                error_type="memory_stats_error",
                stack_trace=traceback.format_exc()
            )
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'cached': 0
            }

    def get_gpu_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage metrics."""
        try:
            if not torch.cuda.is_available():
                return {
                    "gpu_usage": 0.0,
                    "gpu_available": 0.0,
                    "usage_percentage": 0.0
                }
                
            # Get detailed stats
            stats = self.get_detailed_gpu_memory_stats()
            
            # Calculate metrics
            gpu_usage = stats['allocated']
            gpu_total = self.max_gpu_memory
            gpu_available = gpu_total - gpu_usage
            usage_percentage = (gpu_usage / gpu_total) if gpu_total > 0 else 0.0
            
            # Log GPU usage
            self._logger.record_event(
                event_type="gpu_usage_check",
                message="GPU usage check completed",
                level="info",
                additional_info={
                    "gpu_usage": gpu_usage,
                    "gpu_available": gpu_available,
                    "usage_percentage": usage_percentage
                }
            )
            
            return {
                "gpu_usage": gpu_usage,
                "gpu_available": gpu_available,
                "usage_percentage": usage_percentage
            }
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to get GPU usage: {str(e)}",
                error_type="gpu_usage_error",
                stack_trace=traceback.format_exc()
            )
            raise
