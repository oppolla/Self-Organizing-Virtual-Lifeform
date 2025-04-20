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

class MemoriaManager:
    """Manages the core memory system for SOVL."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize MemoriaManager with configuration and logger."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self._state = None
        self._conversation_history = None
        
        # Initialize storage
        self._initialize_storage()
        
        # Log initialization
        self._logger.record_event(
            event_type="memoria_manager_initialized",
            message="Memoria manager initialized",
            level="info"
        )

    def _initialize_storage(self) -> None:
        """Initialize memory storage systems."""
        with self._memory_lock:
            try:
                # Initialize conversation history
                self._conversation_history = ConversationHistory()
                
                # Initialize state
                self._state = SOVLState()
                
                # Log successful initialization
                self._logger.record_event(
                    event_type="memoria_storage_initialized",
                    message="Memoria storage initialized successfully",
                    level="info"
                )
                
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Failed to initialize memoria storage: {str(e)}",
                    error_type="storage_error",
                    stack_trace=traceback.format_exc()
                )
                raise

    def save_state(self, path_prefix: str) -> None:
        """Save current state to disk."""
        try:
            state = {
                "conversation_history": self._conversation_history.get_state(),
                "state": self._state.get_state()
            }
            
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            with open(f"{path_prefix}_memoria.json", 'w') as f:
                json.dump(state, f)
                
            self._logger.record_event(
                event_type="memoria_state_saved",
                message="Memoria state saved successfully",
                level="info"
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to save memoria state: {str(e)}",
                error_type="save_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def load_state(self, path_prefix: str) -> None:
        """Load state from disk."""
        try:
            with open(f"{path_prefix}_memoria.json", 'r') as f:
                state = json.load(f)
                
            self._conversation_history.load_state(state["conversation_history"])
            self._state.load_state(state["state"])
            
            self._logger.record_event(
                event_type="memoria_state_loaded",
                message="Memoria state loaded successfully",
                level="info"
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to load memoria state: {str(e)}",
                error_type="load_error",
                stack_trace=traceback.format_exc()
            )
            raise

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

    def get_gpu_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage metrics."""
        try:
            # Get current GPU usage
            gpu_stats = self.hardware.get_gpu_memory_stats()
            
            # Calculate GPU metrics
            gpu_usage = gpu_stats.get("gpu_usage", 0.0)
            gpu_available = gpu_stats.get("gpu_available", 0.0)
            gpu_total = gpu_stats.get("gpu_total", 0.0)
            
            # Calculate usage percentage
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
