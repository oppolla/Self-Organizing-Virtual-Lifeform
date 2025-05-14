import torch
from typing import Union, Tuple, Optional, List, Dict, Deque, Set, Callable, Any, Type
from collections import deque
import numpy as np
import traceback
from threading import Lock
import traceback
from functools import wraps
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_memory import GPUMemoryManager, RAMManager
from datetime import datetime
import os
import sys

class NumericalGuard:
    """Context manager for numerical stability."""
    def __enter__(self):
        torch.set_grad_enabled(False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(True)

def safe_divide(numerator, denominator, default=0.0):
    """
    Safely divide two numbers, returning a default value if denominator is zero or on error.
    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Value to return if denominator is zero or error occurs.
    Returns:
        The result of numerator / denominator, or default if denominator is zero or error.
    """
    try:
        return numerator / denominator if denominator else default
    except Exception:
        return default

def safe_compare(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Safely compare two floating point numbers."""
    try:
        return abs(a - b) < tolerance
    except Exception:
        return False

def float_gt(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Check if a is greater than b with tolerance."""
    try:
        return a > b + tolerance
    except Exception:
        return False

def validate_quantization_mode(mode: str, config_manager: ConfigManager) -> str:
    """
    Validate and normalize the quantization mode based on configuration.

    Args:
        mode: The quantization mode to validate
        config_manager: ConfigManager instance for fetching valid modes

    Returns:
        The normalized quantization mode

    Raises:
        ValueError: If the mode is invalid or if there are configuration issues
    """
    try:
        # Get valid modes and default mode from config
        valid_modes = config_manager.get(
            "core_config.valid_quantization_modes",
            ["fp16", "int8", "int4"],  # Default list if not in config
            expected_type=list
        )
        default_mode = config_manager.get(
            "core_config.default_quantization_mode",
            "fp16", # Default mode if not specified
            expected_type=str
        )

        # Ensure default_mode is one of the valid_modes
        if default_mode not in valid_modes:
            if not valid_modes:
                raise ValueError("No valid quantization modes defined in configuration.")
            # Fallback to the first valid mode if default is invalid
            default_mode = valid_modes[0]
            print(f"Warning: Default quantization mode '{default_mode}' is not in the list of valid modes: {valid_modes}. "
                  f"Falling back to '{default_mode}'.")

        normalized_mode = mode.lower()

        if normalized_mode not in valid_modes:
            return default_mode

        return normalized_mode

    except Exception as e:
        # Wrap all exceptions in ValueError
        raise ValueError(f"Failed to validate quantization mode: {str(e)}")

def log_memory_usage(label: str = "", device: torch.device = None, logger: Optional[Logger] = None, config_manager: Optional[ConfigManager] = None) -> None:
    """Log memory usage statistics."""
    if logger and config_manager:
        try:
            # Get memory stats from appropriate manager based on device
            if device and device.type == 'cuda':
                gpu_manager = GPUMemoryManager(config_manager, logger)
                gpu_stats = gpu_manager.get_gpu_usage()
                memory_stats = {
                    'gpu_usage': gpu_stats.get('usage_percentage', 0.0),
                    'gpu_allocated': gpu_stats.get('gpu_usage', 0.0),
                    'gpu_available': gpu_stats.get('gpu_available', 0.0)
                }
            else:
                ram_manager = RAMManager(config_manager, logger)
                ram_stats = ram_manager.check_memory_health()
                memory_stats = {
                    'ram_usage': ram_stats.get('usage_percent', 0.0),
                    'ram_available': ram_stats.get('available', 0.0),
                    'ram_total': ram_stats.get('total', 0.0)
                }

            logger.log_memory_usage(
                phase=label,
                device=device,
                additional_info={
                    "memory_stats": memory_stats,
                    "label": label
                }
            )
        except Exception as e:
            logger.log_error(
                error_msg=f"Failed to log memory usage: {str(e)}",
                error_type="memory_logging_error",
                stack_trace=traceback.format_exc()
            )

def dynamic_batch_size(
    base_size: int,
    config_manager: ConfigManager,
    logger: Optional[Logger] = None
) -> int:
    """
    Adjust batch size based on available GPU memory.
    
    Args:
        base_size: Base batch size
        config_manager: Configuration manager instance
        logger: Optional logger for debugging
    
    Returns:
        Adjusted batch size
    """
    if not torch.cuda.is_available():
        return base_size
    
    # Check if base_size is a positive integer
    if base_size <= 0:
        if logger:
            logger.log_error(
                error_msg="Base batch size must be a positive integer.",
                error_type="batch_size_error"
            )
        return 1  # Default to 1 if base_size is invalid
    
    try:
        memory_threshold = config_manager.get("memory_config.memory_threshold", 0.8)
        safety_factor = config_manager.get("memory_config.safety_factor", 0.9)
        
        gpu_manager = GPUMemoryManager(config_manager, logger)
        gpu_stats = gpu_manager.get_gpu_usage()
        
        total_mem = gpu_stats.get('total_memory', 0.0)
        allocated = gpu_stats.get('gpu_usage', 0.0)
        available = (total_mem * memory_threshold * safety_factor) - allocated
        
        if available <= 0:
            adjusted = max(1, base_size // 4)
        else:
            sample_mem = allocated / base_size if base_size > 0 else 1e6
            adjusted = min(base_size, int(available / sample_mem))
            adjusted = max(1, adjusted)
        
        if logger:
            logger.log_training_event(
                event_type="batch_size_adjustment",
                message="Batch size adjusted based on memory",
                level="info",
                additional_info={
                    "base_size": base_size,
                    "adjusted_size": adjusted,
                    "available_memory": available / (1024 ** 3),
                    "memory_threshold": memory_threshold,
                    "safety_factor": safety_factor
                }
            )
        return adjusted
    
    except Exception as e:
        if logger:
            logger.log_error(
                error_msg=f"Dynamic batch size failed: {str(e)}",
                error_type="batch_size_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "base_size": base_size,
                    "error": str(e)
                }
            )
        return max(1, base_size // 4)

def detect_repetitions(
    token_ids: List[int],
    special_ids: Set[int],
    config_manager: ConfigManager,
    logger: Optional[Logger] = None
) -> Optional[Tuple[int, int]]:
    """
    Detect repeating token sequences.
    
    Args:
        token_ids: List of token IDs
        special_ids: Set of special token IDs to ignore
        config_manager: Configuration manager instance
        logger: Optional logger for debugging
    
    Returns:
        (start_idx, end_idx) of first repetition found or None
    """
    try:
        min_rep_length = config_manager.get("processor_config.min_rep_length", 3)
        max_scan = config_manager.get("processor_config.max_rep_scan", 100)
        
        filtered = [i for i in token_ids if i not in special_ids]
        scan_range = min(len(filtered), max_scan)
        
        for i in range(scan_range - 2 * min_rep_length + 1):
            window = filtered[i:i + min_rep_length]
            next_window = filtered[i + min_rep_length:i + 2 * min_rep_length]
            if window == next_window:
                if logger:
                    logger.log_training_event(
                        event_type="repetition_detected",
                        message="Token repetition detected",
                        level="warning",
                        additional_info={
                            "start_idx": i,
                            "end_idx": i + min_rep_length,
                            "length": min_rep_length
                        }
                    )
                return (i, i + min_rep_length)
        return None
    
    except Exception as e:
        if logger:
            logger.log_error(
                error_msg=f"Repetition detection failed: {str(e)}",
                error_type="repetition_detection_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "token_ids_length": len(token_ids),
                    "error": str(e)
                }
            )
        raise

def adjust_temperature(
    base_temp: float,
    temperament_score: float,
    config_manager: ConfigManager,
    logger: Optional[Logger] = None
) -> float:
    """
    Adjust temperature based on temperament and curiosity.
    
    Args:
        base_temp: Base temperature from config
        temperament_score: Current temperament (-1 to 1)
        config_manager: Configuration manager instance
        logger: Optional logger for debugging
    
    Returns:
        Adjusted temperature value
    """
    try:
        with NumericalGuard():
            # Get configuration values
            mood_influence = config_manager.get("controls_config.temp_mood_influence", 0.3)
            min_temp = config_manager.get("controls_config.min_temperature", 0.5)
            max_temp = config_manager.get("controls_config.max_temperature", 1.5)
            curiosity_pressure = config_manager.get("curiosity_config.curiosity_pressure", None)
            
            # Clamp input values
            base_temp = max(min_temp, min(max_temp, base_temp))
            temperament_score = max(-1.0, min(1.0, temperament_score))
            mood_influence = max(0.0, min(1.0, mood_influence))
            
            temp_adjustment = mood_influence * 0.3 * temperament_score
            if curiosity_pressure is not None:
                curiosity_pressure = max(0.0, min(1.0, curiosity_pressure))
                temp_adjustment += curiosity_pressure * 0.1
            
            adjusted_temp = max(min_temp, min(max_temp, base_temp + temp_adjustment))
            
            if logger:
                logger.log_training_event(
                    event_type="temperature_adjustment",
                    message="Temperature adjusted based on temperament and curiosity",
                    level="info",
                    additional_info={
                        "base_temp": base_temp,
                        "temperament_score": temperament_score,
                        "mood_influence": mood_influence,
                        "curiosity_pressure": curiosity_pressure,
                        "adjusted_temp": adjusted_temp
                    }
                )
            return adjusted_temp
    
    except Exception as e:
        if logger:
            logger.log_error(
                error_msg=f"Temperature adjustment failed: {str(e)}",
                error_type="temperature_adjustment_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "base_temp": base_temp,
                    "temperament_score": temperament_score
                }
            )
        return base_temp

def synchronized(lock: Optional[Lock] = None) -> Callable:
    """
    Thread synchronization decorator.
    
    Args:
        lock: Optional Lock instance. If not provided, will use the instance's lock attribute.
        
    Returns:
        Decorated function with thread synchronization.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Use provided lock or instance lock
            lock_to_use = lock if lock is not None else getattr(self, 'lock')
            if not isinstance(lock_to_use, Lock):
                raise AttributeError(f"Lock attribute not found or invalid: {lock_to_use}")
                
            with lock_to_use:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

def validate_components(**components) -> None:
    """
    Validate that all required components are properly initialized.
    
    Args:
        **components: Components to validate with their names as keys
        
    Raises:
        ValueError: If any component is None or invalid
    """
    for name, component in components.items():
        if component is None:
            raise ValueError(f"Required component {name} is None")
        if not hasattr(component, '__class__'):
            raise ValueError(f"Component {name} is not a valid object")

def sync_component_states(state_tracker: Any, components: List[Any]) -> None:
    """
    Synchronize state tracker reference across components.

    Args:
        state_tracker: The main state tracker instance
        components: List of components to sync state with

    Raises:
        ValueError: If state synchronization fails
    """
    try:
        # Ensure state_tracker is not None before proceeding
        if state_tracker is None:
            raise ValueError("Provided state_tracker is None")
            
        for component in components:
            # Check if component is not None and has the attribute
            if component is not None and hasattr(component, 'state_tracker'):
                component.state_tracker = state_tracker
            # Optional: Log or raise if a component doesn't have the attribute?
            # else:
            #     # Handle components without state_tracker attribute if necessary
            #     pass 
                
    except Exception as e:
        # Consider logging the error here using component's logger if available
        # Or re-raise a more specific error
        raise ValueError(f"Failed to sync component states: {str(e)}")

def validate_component_states(state_tracker: Any, components: List[Any]) -> None:
    """
    Validate that all relevant components reference the same state tracker instance.

    Args:
        state_tracker: The main state tracker instance
        components: List of components to validate

    Raises:
        ValueError: If state validation fails (components reference different trackers or None)
    """
    try:
        if state_tracker is None:
            raise ValueError("Cannot validate against a None state_tracker")
            
        mismatched_components = []
        for component in components:
            # Check components that are expected to have a state_tracker
            if component is not None and hasattr(component, 'state_tracker'):
                if getattr(component, 'state_tracker') is not state_tracker:
                    mismatched_components.append(component.__class__.__name__)

        if mismatched_components:
            raise ValueError(f"State tracker mismatch detected in components: {', '.join(mismatched_components)}")

    except Exception as e:
        # Log error if needed
        raise ValueError(f"Failed to validate component states: {str(e)}")

def initialize_component_state(state_tracker: Any, components: List[Any]) -> None:
    """
    Syncs the state tracker reference across components and validates the sync.
    Note: Assumes the core state object (e.g., SOVLState) is initialized elsewhere.

    Args:
        state_tracker: The main state tracker instance
        components: List of components to initialize and validate

    Raises:
        ValueError: If state synchronization or validation fails
    """
    try:
        # Sync the state_tracker reference to all components first
        sync_component_states(state_tracker, components)
        
        # Then validate that the sync was successful
        validate_component_states(state_tracker, components)
        
    except Exception as e:
        # Log error if needed, maybe include component names
        raise ValueError(f"Failed to initialize component state references: {str(e)}")

def validate_layer_indices(
    layers: Union[List[int], int],
    total_layers: int,
    logger: Optional[Logger] = None
) -> bool:
    """
    Validate that layer indices are within valid range for a model.
    
    Args:
        layers: Single layer index or list of layer indices to validate
        total_layers: Total number of layers in the model
        logger: Optional logger for warning messages
        
    Returns:
        bool: True if all indices are valid, False otherwise
    """
    try:
        # Type validation
        if not isinstance(total_layers, int) or total_layers <= 0:
            if logger:
                logger.log_warning(
                    "Invalid total_layers value",
                    additional_info={
                        "total_layers": total_layers,
                        "expected_type": "positive integer"
                    }
                )
            return False

        # Convert single layer index to list and validate type
        if isinstance(layers, int):
            layer_list = [layers]
        elif isinstance(layers, list):
            layer_list = layers
        else:
            if logger:
                logger.log_warning(
                    "Invalid layers type",
                    additional_info={
                        "layers": layers,
                        "expected_type": "int or List[int]"
                    }
                )
            return False

        # Basic validation
        if not layer_list:
            if logger:
                logger.log_warning(
                    "Empty layer list",
                    additional_info={"total_layers": total_layers}
                )
            return False

        # Check for duplicates
        if len(layer_list) != len(set(layer_list)):
            if logger:
                logger.log_warning(
                    "Duplicate layer indices found",
                    additional_info={"layers": layer_list}
                )
            return False

        # Validate all indices in one pass
        invalid_indices = [
            idx for idx in layer_list 
            if not isinstance(idx, int) or idx < 0 or idx >= total_layers
        ]
        
        if invalid_indices:
            if logger:
                logger.log_warning(
                    "Invalid layer indices found",
                    additional_info={
                        "invalid_indices": invalid_indices,
                        "total_layers": total_layers,
                        "valid_range": f"0 to {total_layers - 1}"
                    }
                )
            return False

        return True
        
    except Exception as e:
        if logger:
            logger.log_error(
                error_msg=f"Layer validation failed: {str(e)}",
                error_type="layer_validation_error",
                stack_trace=traceback.format_exc()
            )
        return False

def move_batch_to_device(batch, device):
    """
    Recursively move all tensors in a batch to the specified device.
    Handles nested dictionaries, lists, and tuples.
    Args:
        batch: Input data which may contain tensors (can be dict, list, tuple, tensor, or other)
        device: torch.device to move tensors to
    Returns:
        Same structure as input but with all tensors moved to device
    """
    import torch
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    else:
        return batch

def validate_device_consistency(model: torch.nn.Module, batch: Any, target_device: torch.device) -> Tuple[bool, Optional[str]]:
    """
    Validate that model parameters and batch tensors are on the same device.
    
    Args:
        model: PyTorch model to check
        batch: Batch data containing tensors
        target_device: Expected device for all components
        
    Returns:
        Tuple of (is_consistent, error_message) where error_message is None if consistent
    """
    # Check model parameters
    model_devices = {p.device for p in model.parameters()}
    if len(model_devices) > 1:
        return False, f"Model parameters on multiple devices: {model_devices}"
        
    if len(model_devices) == 0:
        # No parameters found - might be a wrapper or non-standard model
        return True, None
        
    model_device = next(iter(model_devices))
    if model_device != target_device:
        return False, f"Model on {model_device}, expected {target_device}"
    
    # Check batch tensors recursively
    def check_batch_device(data: Any) -> Set[torch.device]:
        if isinstance(data, torch.Tensor):
            return {data.device}
        elif isinstance(data, dict):
            devices = set()
            for v in data.values():
                devices.update(check_batch_device(v))
            return devices
        elif isinstance(data, (list, tuple)):
            devices = set()
            for item in data:
                devices.update(check_batch_device(item))
            return devices
        return set()
    
    batch_devices = check_batch_device(batch)
    if len(batch_devices) > 1:
        return False, f"Batch tensors on multiple devices: {batch_devices}"
    
    if len(batch_devices) == 0:
        # No tensors found in batch
        return True, None
        
    batch_device = next(iter(batch_devices))
    if batch_device != target_device:
        return False, f"Batch on {batch_device}, expected {target_device}"
    
    return True, None

def check_model_health(model, injector, logger=None) -> bool:
    """
    Check the health of the injected model (cross-attention stability).
    Args:
        model: The model to check (should be the base model with injection)
        injector: The CrossAttentionInjector or similar with verify_injection method
        logger: Optional logger for logging results
    Returns:
        bool: True if model is healthy, False if instability is detected
    """
    try:
        config = getattr(model, 'config', None)
        healthy = injector.verify_injection(model, config)
        if not healthy:
            if logger:
                logger.log_error(
                    error_msg="Model health check failed: instability detected after cross-attention injection.",
                    error_type="model_health_check",
                )
        else:
            if logger:
                logger.log_info("Model health check passed: model is stable after injection.")
        return healthy
    except Exception as e:
        if logger:
            logger.log_error(
                error_msg=f"Exception during model health check: {str(e)}",
                error_type="model_health_check",
                stack_trace=traceback.format_exc()
            )
        return False

def calculate_token_map_confidence(logits, *, source="base", allow_scaffold=False, logger=None):
    """
    Calculate confidence score for updating the token map.
    Only use with raw base model logits (not scaffold-augmented).
    Args:
        logits: Output logits from the base model (before any scaffold/cross-attention).
        source: Should be 'base'. If not, logs a warning or raises.
        allow_scaffold: If True, disables the check (for special cases).
        logger: Optional logger for warnings.
    Returns:
        float: Confidence score.
    """
    if source != "base" and not allow_scaffold:
        msg = "calculate_token_map_confidence called with non-base logits!"
        if logger:
            logger.record_event(
                event_type="confidence_circularity_warning",
                message=msg,
                level="warning"
            )
        else:
            print(msg)
        raise ValueError(msg)
    import torch
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs.max(dim=-1).values.mean().item()

def check_adaptation_dependencies(
    logger,
    min_peft_version="0.4.0",
    min_transformers_version=None,
    check_adapters=True,
    check_prefix_tuning=True
):
    """
    Check for PEFT/LoRA/adapter/prefix-tuning dependencies and version compatibility.
    Returns a dict with enabled/disabled flags and reasons for each adaptation method.
    """
    result = {
        "lora_enabled": False,
        "adapters_enabled": False,
        "prefix_tuning_enabled": False,
        "reasons": {}
    }
    # Check PEFT/LoRA
    try:
        import pkg_resources
        import peft
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PrefixTuningConfig
        peft_version = pkg_resources.get_distribution("peft").version
        if pkg_resources.parse_version(peft_version) < pkg_resources.parse_version(min_peft_version):
            msg = f"PEFT version {peft_version} < {min_peft_version}"
            logger.log_error(msg, error_type="dependency_version_error")
            result["reasons"]["lora"] = msg
        else:
            result["lora_enabled"] = True
            result["reasons"]["lora"] = "OK"
    except Exception as e:
        logger.log_error(f"LoRA/PEFT unavailable: {e}", error_type="dependency_error")
        result["reasons"]["lora"] = str(e)
    # Check adapters
    if check_adapters:
        try:
            from transformers.adapters import AdapterConfig
            result["adapters_enabled"] = True
            result["reasons"]["adapters"] = "OK"
        except Exception as e:
            logger.log_error(f"Adapters unavailable: {e}", error_type="dependency_error")
            result["reasons"]["adapters"] = str(e)
    # Check prefix tuning
    if check_prefix_tuning:
        try:
            from peft import PrefixTuningConfig
            result["prefix_tuning_enabled"] = True
            result["reasons"]["prefix_tuning"] = "OK"
        except Exception as e:
            logger.log_error(f"Prefix tuning unavailable: {e}", error_type="dependency_error")
            result["reasons"]["prefix_tuning"] = str(e)
    return result

def ensure_dir_exists(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def backup_file(src, dst):
    """Backup a file from src to dst."""
    if os.path.exists(src):
        with open(src, 'r') as fsrc:
            content = fsrc.read()
        with open(dst, 'w') as fdst:
            fdst.write(content)

def restore_file(src, dst):
    """Restore a file from src to dst."""
    if os.path.exists(src):
        with open(src, 'r') as fsrc:
            content = fsrc.read()
        with open(dst, 'w') as fdst:
            fdst.write(content)

def cleanup_components(component_names, context, logger=None):
    """
    Clean up components in reverse order by calling their cleanup() method if present.
    Args:
        component_names: List of attribute names (strings) to clean up.
        context: The object holding the components as attributes.
        logger: Optional logger for error/info reporting.
    """
    for name in reversed(list(component_names)):
        component = getattr(context, name, None)
        if component and hasattr(component, 'cleanup'):
            try:
                component.cleanup()
                if logger:
                    logger.log_info(f"Cleaned up {name}")
            except Exception as e:
                msg = f"Error cleaning up {name}: {str(e)}"
                if logger:
                    logger.log_error(error_msg=msg, error_type="cleanup_error")
                else:
                    print(msg)

def atomic_file_counter(counter_file, backup_file, lock, logger=None):
    """
    Atomically increment and return a counter stored in a file, with file locking and backup/restore.
    Args:
        counter_file: Path to the file storing the counter.
        backup_file: Path to the backup file.
        lock: A threading lock for process-level atomicity.
        logger: Optional logger for error/info reporting.
    Returns:
        The incremented counter value (int).
    """
    ensure_dir_exists(os.path.dirname(counter_file))
    with lock:
        try:
            with open(counter_file, 'a+') as f:
                # Platform-specific file locking
                try:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX)
                except (ImportError, AttributeError):
                    try:
                        import msvcrt
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    except (ImportError, AttributeError):
                        pass  # Already using a Python lock
                f.seek(0)
                content = f.read().strip()
                last_id = int(content) if content.isdigit() else 0
                last_id += 1
                f.seek(0)
                f.truncate()
                f.write(str(last_id))
            backup_file(counter_file, backup_file)
            return last_id
        except Exception as e:
            restore_file(backup_file, counter_file)
            if logger:
                logger.log_error(error_msg=f"Atomic counter error: {e}", error_type="atomic_counter_error")
            raise

def safe_append_to_file(path, content, encoding="utf-8"):
    """Append content to a file safely, with error handling."""
    try:
        with open(path, "a", encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False

def validate_metadata_fields(example: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate key metadata fields needed for curriculum example selection.
    Args:
        example: Training example with metadata
    Returns:
        Dictionary of validation results for each key field
    """
    metadata = example.get("metadata", {})
    content_metrics = metadata.get("content_metrics", {})
    quality_metrics = metadata.get("quality_metrics", {})
    return {
        "has_metadata": bool(metadata),
        "has_content_metrics": bool(content_metrics),
        "has_quality_metrics": bool(quality_metrics),
        "has_word_count": "word_count" in content_metrics,
        "has_has_code": "has_code" in quality_metrics,
        "has_has_question": "has_question" in quality_metrics
    }

def repair_metadata(example: Dict[str, Any], validation_results: Dict[str, bool]) -> Dict[str, Any]:
    """
    Repair missing metadata fields with appropriate defaults.
    Args:
        example: Training example with metadata
        validation_results: Result of metadata validation
    Returns:
        Example with repaired metadata
    """
    import copy
    fixed_example = copy.deepcopy(example)
    if not validation_results["has_metadata"]:
        fixed_example["metadata"] = {}
    metadata = fixed_example["metadata"]
    if not validation_results["has_content_metrics"]:
        metadata["content_metrics"] = {}
    if not validation_results["has_quality_metrics"]:
        metadata["quality_metrics"] = {}
    content_metrics = metadata["content_metrics"]
    quality_metrics = metadata["quality_metrics"]
    if not validation_results["has_word_count"]:
        if "content" in example and isinstance(example["content"], str):
            content_metrics["word_count"] = len(example["content"].split())
        else:
            content_metrics["word_count"] = 0
    if not validation_results["has_has_code"]:
        if "content" in example and isinstance(example["content"], str):
            quality_metrics["has_code"] = ("```" in example["content"])
        else:
            quality_metrics["has_code"] = False
    if not validation_results["has_has_question"]:
        if "content" in example and isinstance(example["content"], str):
            quality_metrics["has_question"] = ("?" in example["content"])
        else:
            quality_metrics["has_question"] = False
    return fixed_example

def get_metadata_value(metadata: Dict[str, Any], path: str, default_value: Any, expected_type: Optional[Type] = None) -> Any:
    """
    Safely extract a value from nested metadata with type checking.
    Args:
        metadata: The metadata dictionary
        path: Dot-separated path to the value (e.g., "content_metrics.word_count")
        default_value: Default value if path doesn't exist
        expected_type: Expected type of the value (optional)
    Returns:
        The value at the path, or default_value if not found or wrong type
    """
    if not metadata:
        return default_value
    try:
        components = path.split('.')
        current = metadata
        for component in components[:-1]:
            if not isinstance(current, dict) or component not in current:
                return default_value
            current = current[component]
        final_key = components[-1]
        if not isinstance(current, dict) or final_key not in current:
            return default_value
        value = current[final_key]
        if expected_type is not None and not isinstance(value, expected_type):
            return default_value
        return value
    except Exception:
        return default_value

def collate_tensor_batch(batch: list, device: "torch.device") -> dict:
    """
    Collate a list of dicts (with tensor values) into a batch dict of stacked tensors, moved to device.
    Args:
        batch: List[Dict[str, torch.Tensor]]
        device: torch.device
    Returns:
        Dict[str, torch.Tensor] (all tensors stacked and moved to device)
    """
    if not batch:
        return {}
    try:
        collated = {k: torch.stack([item[k] for item in batch]) for k in batch[0] if isinstance(batch[0][k], torch.Tensor)}
        collated = {k: v.to(device) for k, v in collated.items()}
        return collated
    except Exception as e:
        raise RuntimeError(f"Failed to collate tensor batch: {e}")

# --- CLI Feedback and Formatting Utilities ---
def format_file_size(num_bytes: int) -> str:
    """Return human-readable file size (e.g., 1.2 MB)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

def format_timestamp(dt=None) -> str:
    """Return a readable timestamp string (local time)."""
    if dt is None:
        dt = datetime.now()
    elif isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def print_section_header(title: str):
    """Print a bold/underlined section header."""
    print(f"\033[1m{title}\033[0m")

def print_bullet_list(items):
    """Print a list with bullets and indentation."""
    for item in items:
        print(f"  - {item}")

def print_kv_table(d: dict, key_width: int = 16):
    """Print key-value pairs in aligned columns."""
    for k, v in d.items():
        print(f"  {str(k):<{key_width}} {v}")

def progress_bar(percent: float, width: int = 30) -> str:
    """Return a string for a simple progress bar."""
    percent = max(0.0, min(1.0, percent))
    filled = int(width * percent)
    bar = '#' * filled + '-' * (width - filled)
    return f"[{bar}] {int(percent * 100)}%"

def print_success(msg: str):
    """Print a standardized success message (green if supported)."""
    if sys.stdout.isatty():
        print(f"\033[92m✔ {msg}\033[0m")
    else:
        print(f"SUCCESS: {msg}")

def print_error(msg: str):
    """Print a standardized error message (red if supported)."""
    if sys.stderr.isatty():
        print(f"\033[91m✖ {msg}\033[0m", file=sys.stderr)
    else:
        print(f"ERROR: {msg}", file=sys.stderr)

def cosine_similarity(a, b, dim=-1, eps=1e-8):
    """
    Compute cosine similarity between two tensors.
    Args:
        a, b: torch.Tensor
        dim: dimension along which to compute similarity
        eps: small value to avoid division by zero
    Returns:
        torch.Tensor of similarities
    """
    return torch.nn.functional.cosine_similarity(a, b, dim=dim, eps=eps)

def set_nested_dict_value(d: dict, dotted_key: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using a dot-separated key.
    Args:
        d: The dictionary to modify.
        dotted_key: The dot-separated key (e.g., 'a.b.c').
        value: The value to set.
    """
    keys = dotted_key.split('.')
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_nested_dict_value(d: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Get a value from a nested dictionary using a dot-separated key.
    Args:
        d: The dictionary to query.
        dotted_key: The dot-separated key (e.g., 'a.b.c').
        default: The value to return if the key is not found.
    Returns:
        The value at the nested key, or default if not found.
    """
    keys = dotted_key.split('.')
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]