import torch
from typing import Union, Tuple, Optional, List, Dict, Deque, Set, Callable, Any
from collections import deque
import numpy as np
import random
import time
import traceback
from threading import Lock
import traceback
from functools import wraps
from sovl_logger import Logger
from sovl_processor import MetadataProcessor
from sovl_config import ConfigManager
from sovl_memory import GPUMemoryManager, RAMManager
from sovl_state import StateTracker
import logging
import sys
import json
from datetime import datetime, timezone

class NumericalGuard:
    """Context manager for numerical stability."""
    def __enter__(self):
        torch.set_grad_enabled(False)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_grad_enabled(True)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

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