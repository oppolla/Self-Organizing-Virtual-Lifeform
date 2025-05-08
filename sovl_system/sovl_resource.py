import threading
from collections import defaultdict
from typing import Optional, Dict, Any, Union
import torch
import traceback
from transformers import AutoConfig
from sovl_logger import Logger
from sovl_error import ErrorManager

class ResourceManager:
    """Manages system resources with thread-safe operations and validation."""
    
    def __init__(self, logger: Logger, error_manager: Optional[ErrorManager] = None):
        """
        Initialize ResourceManager with logger and optional error manager.
        
        Args:
            logger: Logger instance for event and error logging.
            error_manager: Optional ErrorManager for structured error handling.
        """
        self.resources = defaultdict(int)  # Current resource amounts
        self.max_resources = defaultdict(lambda: None)  # Maximum resource limits
        self._lock = threading.Lock()  # Thread-safe lock
        self.logger = logger
        self.error_manager = error_manager
        self._usage_history = defaultdict(list)  # Track resource usage over time
        
        self.logger.log_event(
            event_type="resource_manager_init",
            message="ResourceManager initialized",
            level="info"
        )

    def acquire(self, resource_type: str, amount: int) -> bool:
        """
        Acquire specified amount of a resource type. Prevents negative resources.
        
        Args:
            resource_type: Type of resource (e.g., 'gpu_memory', 'cpu_memory').
            amount: Amount of resource to acquire (in resource-specific units, e.g., MB).
            
        Returns:
            bool: True if acquisition succeeds, False otherwise.
        """
        try:
            with self._lock:
                available = self.resources[resource_type]
                if available >= amount:
                    self.resources[resource_type] -= amount
                    if self.resources[resource_type] < 0:
                        self.resources[resource_type] = 0
                        self.logger.log_event(
                            event_type="resource_negative_correction",
                            message=f"Resource {resource_type} went negative! Corrected to 0.",
                            level="warning"
                        )
                    self._update_usage_history(resource_type, self.resources[resource_type])
                    self.logger.log_event(
                        event_type="resource_acquire",
                        message=f"Acquired {amount} of {resource_type}. Remaining: {self.resources[resource_type]}",
                        level="info"
                    )
                    return True
                self.logger.log_event(
                    event_type="resource_acquire_failed",
                    message=f"Failed to acquire {amount} of {resource_type}. Available: {available}",
                    level="warning"
                )
                return False
        except Exception as e:
            self._handle_error(
                error=e,
                context="resource_acquire",
                extra_info={"resource_type": resource_type, "amount": amount}
            )
            return False

    def release(self, resource_type: str, amount: int) -> None:
        """
        Release specified amount of a resource type. Prevents over-release above max or negative.
        
        Args:
            resource_type: Type of resource to release.
            amount: Amount of resource to release.
        """
        try:
            with self._lock:
                max_val = self.max_resources[resource_type]
                self.resources[resource_type] += amount
                if max_val is not None and self.resources[resource_type] > max_val:
                    self.resources[resource_type] = max_val
                    self.logger.log_event(
                        event_type="resource_max_correction",
                        message=f"Resource {resource_type} exceeded max! Corrected to {max_val}.",
                        level="warning"
                    )
                if self.resources[resource_type] < 0:
                    self.resources[resource_type] = 0
                    self.logger.log_event(
                        event_type="resource_negative_correction",
                        message=f"Resource {resource_type} went negative on release! Corrected to 0.",
                        level="warning"
                    )
                self._update_usage_history(resource_type, self.resources[resource_type])
                self.logger.log_event(
                    event_type="resource_release",
                    message=f"Released {amount} of {resource_type}. Total: {self.resources[resource_type]}",
                    level="info"
                )
        except Exception as e:
            self._handle_error(
                error=e,
                context="resource_release",
                extra_info={"resource_type": resource_type, "amount": amount}
            )

    def set_resource(self, resource_type: str, amount: int, max_amount: Optional[int] = None) -> None:
        """
        Set the total amount of a resource type. Optionally set a maximum limit.
        
        Args:
            resource_type: Type of resource to set.
            amount: Current amount of resource.
            max_amount: Optional maximum limit for the resource.
        """
        try:
            with self._lock:
                self.resources[resource_type] = max(0, amount)
                if max_amount is not None:
                    self.max_resources[resource_type] = max_amount
                self._update_usage_history(resource_type, self.resources[resource_type])
                self.logger.log_event(
                    event_type="resource_set",
                    message=f"Set {resource_type} to {self.resources[resource_type]} (max: {self.max_resources[resource_type]})",
                    level="info"
                )
        except Exception as e:
            self._handle_error(
                error=e,
                context="resource_set",
                extra_info={"resource_type": resource_type, "amount": amount, "max_amount": max_amount}
            )

    def get_resource_status(self, resource_type: str) -> Dict[str, Any]:
        """
        Get the current status of a resource type.
        
        Args:
            resource_type: Type of resource to query.
            
        Returns:
            Dict containing current amount, max amount, and usage history.
        """
        try:
            with self._lock:
                return {
                    "current": self.resources[resource_type],
                    "max": self.max_resources[resource_type],
                    "usage_history": self._usage_history[resource_type][-100:]  # Limit history to last 100 entries
                }
        except Exception as e:
            self._handle_error(
                error=e,
                context="get_resource_status",
                extra_info={"resource_type": resource_type}
            )
            return {"current": 0, "max": None, "usage_history": []}

    def cleanup(self) -> None:
        """
        Clean up resources, reset state, and clear usage history.
        """
        try:
            with self._lock:
                self.resources.clear()
                self.max_resources.clear()
                self._usage_history.clear()
                self.logger.log_event(
                    event_type="resource_cleanup",
                    message="ResourceManager cleaned up successfully",
                    level="info"
                )
        except Exception as e:
            self._handle_error(
                error=e,
                context="resource_cleanup",
                extra_info={}
            )

    def _update_usage_history(self, resource_type: str, amount: int) -> None:
        """
        Update the usage history for a resource type.
        
        Args:
            resource_type: Type of resource.
            amount: Current amount after operation.
        """
        try:
            self._usage_history[resource_type].append({
                "amount": amount,
                "timestamp": time.time()
            })
            # Trim history to prevent unbounded growth
            if len(self._usage_history[resource_type]) > 1000:
                self._usage_history[resource_type] = self._usage_history[resource_type][-1000:]
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to update usage history: {str(e)}",
                error_type="usage_history_error",
                stack_trace=traceback.format_exc()
            )

    def _handle_error(self, error: Exception, context: str, extra_info: Dict[str, Any]) -> None:
        """
        Handle errors using ErrorManager if available, otherwise log directly.
        
        Args:
            error: Exception to handle.
            context: Context of the error (e.g., 'resource_acquire').
            extra_info: Additional information for logging.
        """
        error_msg = f"Error in {context}: {str(error)}"
        error_context = {
            "context": context,
            "stack_trace": traceback.format_exc(),
            **extra_info
        }
        if self.error_manager:
            self.error_manager.handle_error(
                error=error,
                error_type=type(error).__name__,
                context=error_context
            )
        else:
            self.logger.log_error(
                error_msg=error_msg,
                error_type=type(error).__name__,
                error_context=error_context
            )

def estimate_model_size(model_name: str, quantization: str = "none") -> int:
    """
    Estimate the memory size of a model in MB.
    
    Args:
        model_name: Name or path of the model (e.g., 'bert-base-uncased').
        quantization: Quantization mode ('none', '4bit', '8bit').
        
    Returns:
        int: Estimated size in MB.
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        param_count = sum(p.numel() for p in torch.nn.Module.from_config(config).parameters())
        bytes_per_param = 4  # Default fp32
        if quantization == "4bit":
            bytes_per_param = 0.5
        elif quantization == "8bit":
            bytes_per_param = 1
        elif quantization == "fp16":
            bytes_per_param = 2
        size_bytes = param_count * bytes_per_param
        return size_bytes // (1024 * 1024)  # Convert to MB
    except Exception as e:
        logger = Logger(LoggerConfig())  # Fallback logger
        logger.log_error(
            error_msg=f"Failed to estimate model size for {model_name}: {str(e)}",
            error_type="model_size_estimation_error",
            stack_trace=traceback.format_exc()
        )
        return 2048  # Default to 2GB if estimation fails
