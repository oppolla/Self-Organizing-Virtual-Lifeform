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
from sovl_utils import log_memory_usage, safe_divide
from sovl_config import ConfigManager
from sovl_hardware import HardwareManager
from sovl_error import ErrorManager, ErrorRecord
import gc
import torch.cuda as cuda
import contextlib

class RAMManager:
    """Manages RAM memory for the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize RAMManager with configuration and logger."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self.hardware = HardwareManager(config_manager, logger)
        
        # Initialize error manager
        self._initialize_error_manager()
        
        # Initialize RAM
        self._initialize_ram()
        
        # Log initialization
        self._logger.record_event(
            event_type="ram_manager_initialized",
            message="RAM manager initialized",
            level="info"
        )

    def _initialize_error_manager(self):
        """Initialize error manager with memory-specific configuration."""
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,
            config_manager=self._config_manager,
            error_cooldown=1.0
        )
        
        # Register memory-specific thresholds
        self.error_manager.severity_thresholds.update({
            "ram_allocation": 3,    # 3 allocation failures before critical
            "ram_threshold": 5,     # 5 threshold violations before critical
            "ram_health": 2        # 2 health check failures before critical
        })
        
        # Register recovery strategies
        self.error_manager.recovery_strategies.update({
            "ram_allocation_error": self._recover_ram_allocation,
            "ram_threshold_error": self._recover_ram_threshold,
            "ram_health_error": self._recover_ram_health
        })

    def _recover_ram_allocation(self, record: ErrorRecord) -> None:
        """Recovery strategy for RAM allocation errors."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Reduce batch size if applicable
            current_batch = self._config_manager.get("ram_config.batch_size", self.initial_batch_size)
            new_batch = max(1, current_batch // 2)
            self._config_manager.update("ram_config.batch_size", new_batch)
            
            self._logger.record_event(
                "ram_allocation_recovery",
                "Recovered from RAM allocation error",
                level="info",
                additional_info={"new_batch_size": new_batch}
            )
        except Exception as e:
            self._logger.record_event(
                "ram_recovery_failed",
                f"Failed to recover from RAM allocation error: {str(e)}",
                level="error"
            )

    def _recover_ram_threshold(self, record: ErrorRecord) -> None:
        """Recovery strategy for RAM threshold violations."""
        try:
            # Increase threshold temporarily
            self.memory_threshold = min(0.95, self.memory_threshold + 0.05)
            
            # Force garbage collection
            gc.collect()
            
            self._logger.record_event(
                "ram_threshold_recovery",
                "Adjusted RAM threshold",
                level="info",
                additional_info={"new_threshold": self.memory_threshold}
            )
        except Exception as e:
            self._logger.record_event(
                "ram_recovery_failed",
                f"Failed to recover from RAM threshold error: {str(e)}",
                level="error"
            )

    def _recover_ram_health(self, record: ErrorRecord) -> None:
        """Recovery strategy for RAM health check failures."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Reset memory parameters
            self.batch_size = self.initial_batch_size
            self.memory_threshold = self._config_manager.get("ram_config.memory_threshold", 0.85)
            
            self._logger.record_event(
                "ram_health_recovery",
                "Reset RAM parameters",
                level="info"
            )
        except Exception as e:
            self._logger.record_event(
                "ram_recovery_failed",
                f"Failed to recover from RAM health error: {str(e)}",
                level="error"
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
                self.error_manager.handle_error(
                    error=e,
                    error_type="ram_initialization_error",
                    severity=2,
                    additional_info={"stage": "ram_initialization"}
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
            
            # Check if usage exceeds threshold
            if ram_usage / ram_total > self.memory_threshold:
                self.error_manager.handle_error(
                    error=MemoryError("RAM usage exceeds threshold"),
                    error_type="ram_threshold_error",
                    severity=1,
                    additional_info={
                        "usage": ram_usage,
                        "total": ram_total,
                        "threshold": self.memory_threshold
                    }
                )
            
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
            self.error_manager.handle_error(
                error=e,
                error_type="ram_health_error",
                severity=2,
                additional_info={"stage": "health_check"}
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
        
        # Initialize error manager
        self._initialize_error_manager()
        
        # Initialize GPU
        self._initialize_gpu()
        
        # Log initialization
        self._logger.record_event(
            event_type="gpu_manager_initialized",
            message="GPU manager initialized",
            level="info"
        )

    def _initialize_error_manager(self):
        """Initialize error manager with GPU-specific configuration."""
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,
            config_manager=self._config_manager,
            error_cooldown=1.0
        )
        
        # Register GPU-specific thresholds
        self.error_manager.severity_thresholds.update({
            "gpu_allocation": 3,    # 3 allocation failures before critical
            "gpu_threshold": 5,     # 5 threshold violations before critical
            "cuda_error": 2        # 2 CUDA errors before critical
        })
        
        # Register recovery strategies
        self.error_manager.recovery_strategies.update({
            "gpu_allocation_error": self._recover_gpu_allocation,
            "gpu_threshold_error": self._recover_gpu_threshold,
            "cuda_error": self._recover_cuda_error
        })

    def _recover_gpu_allocation(self, record: ErrorRecord) -> None:
        """Recovery strategy for GPU allocation errors."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Free any unused allocations
            self._cleanup_unused_allocations()
            
            self._logger.record_event(
                "gpu_allocation_recovery",
                "Recovered from GPU allocation error",
                level="info"
            )
        except Exception as e:
            self._logger.record_event(
                "gpu_recovery_failed",
                f"Failed to recover from GPU allocation error: {str(e)}",
                level="error"
            )

    def _recover_gpu_threshold(self, record: ErrorRecord) -> None:
        """Recovery strategy for GPU threshold violations."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Adjust threshold temporarily
            self.gpu_threshold = min(0.95, self.gpu_threshold + 0.05)
            
            self._logger.record_event(
                "gpu_threshold_recovery",
                "Adjusted GPU threshold",
                level="info",
                additional_info={"new_threshold": self.gpu_threshold}
            )
        except Exception as e:
            self._logger.record_event(
                "gpu_recovery_failed",
                f"Failed to recover from GPU threshold error: {str(e)}",
                level="error"
            )

    def _recover_cuda_error(self, record: ErrorRecord) -> None:
        """Recovery strategy for CUDA errors."""
        try:
            # Reset CUDA device
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_device = torch.cuda.current_device()
                torch.cuda.device(current_device)
            
            self._logger.record_event(
                "cuda_error_recovery",
                "Reset CUDA device",
                level="info"
            )
        except Exception as e:
            self._logger.record_event(
                "cuda_recovery_failed",
                f"Failed to recover from CUDA error: {str(e)}",
                level="error"
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
                self.error_manager.handle_error(
                    error=e,
                    error_type="gpu_initialization_error",
                    severity=2,
                    additional_info={"stage": "gpu_initialization"}
                )
                raise

    def allocate_gpu_memory(self, size_bytes: int) -> Optional[int]:
        """Allocate GPU memory using CUDA."""
        try:
            if not torch.cuda.is_available():
                self.error_manager.handle_error(
                    error=RuntimeError("CUDA is not available"),
                    error_type="cuda_error",
                    severity=1,
                    additional_info={"stage": "memory_allocation"}
                )
                return None
                
            with self._memory_lock:
                # Check if we have enough memory
                current_usage = self.get_detailed_gpu_memory_stats()
                if current_usage['allocated'] + size_bytes > self.max_gpu_memory:
                    self.error_manager.handle_error(
                        error=MemoryError("Not enough GPU memory available"),
                        error_type="gpu_allocation_error",
                        severity=1,
                        additional_info={
                            "requested": size_bytes,
                            "available": self.max_gpu_memory - current_usage['allocated']
                        }
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
            self.error_manager.handle_error(
                error=e,
                error_type="gpu_allocation_error",
                severity=2,
                additional_info={
                    "size_bytes": size_bytes,
                    "stage": "memory_allocation"
                }
            )
            return None

    def free_gpu_memory(self, ptr: int) -> bool:
        """Free GPU memory using CUDA."""
        try:
            if not torch.cuda.is_available():
                self.error_manager.handle_error(
                    error=RuntimeError("CUDA is not available"),
                    error_type="cuda_error",
                    severity=1,
                    additional_info={"stage": "memory_free"}
                )
                return False
                
            with self._memory_lock:
                if ptr not in self._allocated_memory:
                    self.error_manager.handle_error(
                        error=ValueError("Attempted to free unallocated memory"),
                        error_type="gpu_allocation_error",
                        severity=1,
                        additional_info={"ptr": ptr}
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
            self.error_manager.handle_error(
                error=e,
                error_type="gpu_allocation_error",
                severity=2,
                additional_info={
                    "ptr": ptr,
                    "stage": "memory_free"
                }
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
                
                # Check if usage exceeds threshold
                if stats['allocated'] / self.max_gpu_memory > self.gpu_threshold:
                    self.error_manager.handle_error(
                        error=MemoryError("GPU memory usage exceeds threshold"),
                        error_type="gpu_threshold_error",
                        severity=1,
                        additional_info={
                            "allocated": stats['allocated'],
                            "max_memory": self.max_gpu_memory,
                            "threshold": self.gpu_threshold
                        }
                    )
                
                self._logger.record_event(
                    event_type="gpu_memory_stats",
                    message="Retrieved detailed GPU memory statistics",
                    level="info",
                    additional_info=stats
                )
                
                return stats
                
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="gpu_stats_error",
                severity=2,
                additional_info={"stage": "memory_stats"}
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

class GenerationMemoryManager:
    """Manages memory for text generation tasks in the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, ram_manager: RAMManager, gpu_manager: GPUMemoryManager):
        """Initialize GenerationMemoryManager with configuration and memory managers."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
        # Initialize memory tracking
        self._tensor_metadata = defaultdict(dict)
        self._embedding_cache = {}
        self._max_cache_size = 1000
        
        # Initialize memory threshold
        self.memory_threshold = min(
            self.ram_manager.memory_threshold,
            self.gpu_manager.gpu_threshold
        )
        
        # Log initialization
        self._logger.record_event(
            event_type="generation_memory_manager_initialized",
            message="Generation memory manager initialized",
            level="info"
        )

    def manage_memory(self) -> None:
        """Manage memory usage with adaptive thresholds."""
        try:
            # Update memory threshold adaptively
            self.memory_threshold = self._calculate_adaptive_threshold()
            
            # Check memory health with adaptive threshold
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            if ram_health["usage_percentage"] > self.memory_threshold:
                self.ram_manager.recover_memory_threshold(self.memory_threshold)
            
            if gpu_health["usage_percentage"] > self.memory_threshold:
                self.gpu_manager.recover_memory_threshold(self.memory_threshold)
                self._offload_old_tensors()
            
            # Log memory health with adaptive threshold
            self._log_memory_health()
            
        except Exception as e:
            self._handle_error("memory_management", e)

    def _offload_old_tensors(self) -> None:
        """Move old tensors to CPU with tracking."""
        try:
            current_time = time.time()
            tensors_to_offload = []
            
            # Identify tensors to offload
            for tensor_id, metadata in self._tensor_metadata.items():
                if current_time - metadata['last_access'] > 300:  # 5 minutes
                    tensors_to_offload.append((tensor_id, metadata))
            
            # Batch process offloading using memory managers
            if tensors_to_offload:
                with self._memory_lock:
                    for tensor_id, metadata in tensors_to_offload:
                        tensor = metadata['tensor']
                        if tensor.device.type == 'cuda':
                            # Use GPU memory manager for proper CUDA memory handling
                            self.gpu_manager.free_gpu_memory(tensor.data_ptr())
                            # Move to RAM using RAM manager
                            cpu_tensor = tensor.cpu()
                            self._tensor_metadata[tensor_id]['tensor'] = cpu_tensor
                            self._tensor_metadata[tensor_id]['device'] = 'cpu'
                            
            # Log offloading results
            self._logger.record_event(
                event_type="tensor_offload",
                message=f"Offloaded {len(tensors_to_offload)} tensors",
                level="info"
            )
        except Exception as e:
            self._handle_error("tensor_offload", e)

    def check_memory_health(self) -> bool:
        """Check memory health with adaptive thresholds."""
        try:
            # Update memory threshold adaptively
            self.memory_threshold = self._calculate_adaptive_threshold()
            
            # Check both RAM and GPU health
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Calculate overall health score
            ram_score = 1.0 - (ram_health["usage_percentage"] / 100.0)
            gpu_score = 1.0 - (gpu_health["usage_percentage"] / 100.0)
            overall_score = (ram_score + gpu_score) / 2.0
            
            # Trigger cleanup if health score is below threshold
            if overall_score < self.memory_threshold:
                self.manage_memory()
                return False
                
            return True
            
        except Exception as e:
            self._handle_error("memory_health_check", e)
            return False

    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive memory threshold based on system load and model size."""
        try:
            base_threshold = self.memory_threshold
            model_size = sum(p.numel() * p.element_size() for p in self._config_manager.get("model_parameters", []))
            total_memory = self.gpu_manager.get_gpu_usage()["total_memory"]
            system_load = self.ram_manager.check_memory_health()["usage_percentage"]
            
            # Calculate adaptive threshold
            adaptive_threshold = base_threshold * (1 - system_load) * (1 - model_size/total_memory)
            return max(0.7, min(0.95, adaptive_threshold))
        except Exception as e:
            self._handle_error("adaptive_threshold_calculation", e)
            return self.memory_threshold

    def optimize_batch_size(self, current_batch_size: int) -> int:
        """Optimize batch size based on memory availability."""
        try:
            memory_usage = self.gpu_manager.get_gpu_usage()["usage_percentage"]
            if memory_usage > 0.9:
                return max(1, current_batch_size // 2)
            elif memory_usage < 0.5:
                return min(self._config_manager.get("max_batch_size", 32), current_batch_size * 2)
            return current_batch_size
        except Exception as e:
            self._handle_error("batch_size_optimization", e)
            return current_batch_size

    def _log_memory_health(self) -> None:
        """Log detailed memory health status with adaptive thresholds."""
        try:
            # Get current memory stats
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Calculate memory pressure
            ram_pressure = ram_health["usage_percentage"] / 100.0
            gpu_pressure = gpu_health["usage_percentage"] / 100.0
            
            # Log memory health details
            self._logger.info(
                f"Memory Health Status:\n"
                f"  RAM Usage: {ram_health['used']:.2f}GB / {ram_health['total']:.2f}GB "
                f"({ram_health['usage_percentage']:.1f}%)\n"
                f"  GPU Usage: {gpu_health['used']:.2f}GB / {gpu_health['total']:.2f}GB "
                f"({gpu_health['usage_percentage']:.1f}%)\n"
                f"  Memory Pressure: RAM={ram_pressure:.2f}, GPU={gpu_pressure:.2f}\n"
                f"  Adaptive Threshold: {self.memory_threshold:.2f}"
            )
            
        except Exception as e:
            self._handle_error("memory_logging", e)

    def _handle_error(self, context: str, error: Exception) -> None:
        """Handle errors using the logger."""
        try:
            self._logger.log_error(
                error_msg=f"Memory management error in {context}: {str(error)}",
                error_type=f"memory_{context}_error",
                stack_trace=traceback.format_exc()
            )
        except Exception as e:
            print(f"Failed to handle error: {str(e)}")  # Fallback error handling

    @contextlib.contextmanager
    def memory_context(self):
        """Context manager for memory-intensive operations."""
        try:
            # Check memory before operation
            self.manage_memory()
            yield
        finally:
            # Clean up after operation
            self.manage_memory()
