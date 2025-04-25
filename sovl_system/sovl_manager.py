import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from typing import Optional, List, Dict, Any, Tuple
import traceback
import os
from threading import Lock
import time
from functools import wraps  # Import wraps
from sovl_utils import validate_quantization_mode
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager, ErrorRecord
from sovl_state import StateTracker, StateManager

# Decorator function (defined outside the class for clarity)
def _prevent_immediate_retry(recovery_func):
    @wraps(recovery_func)
    def wrapper(self, record: ErrorRecord, *args, **kwargs):
        strategy_name = recovery_func.__name__
        
        # Check for missing hash
        if not hasattr(record, 'hash') or record.hash is None:
            self._log_error(
                "Missing hash in ErrorRecord for retry prevention. Skipping recovery attempt.",
                "decorator_error",
                additional_info={"record_type": type(record).__name__, "record_repr_start": repr(record)[:200]}
            )
            # Skip the recovery attempt entirely if we cannot track it
            return

        recovery_key = (record.hash, strategy_name)

        # Skip if this exact recovery just failed
        if self._last_failed_recovery_key == recovery_key:
            self._log_event(
                f"{strategy_name}_skipped",
                f"Skipping immediate retry of failed {strategy_name} recovery",
                level="warning",
                additional_info={"error_hash": record.hash}
            )
            return  # Skip the recovery attempt

        try:
            # Call the original recovery function
            result = recovery_func(self, record, *args, **kwargs)
            # Reset the flag ONLY if the recovery function completed without raising an exception.
            # NOTE: This signifies the recovery *attempt* finished, not necessarily that the
            # underlying root cause of the error has been resolved. The error might recur.
            self._last_failed_recovery_key = None
            return result
        except Exception as e:
            # Mark this recovery attempt as failed
            self._last_failed_recovery_key = recovery_key
            # Log the failure during the recovery attempt
            self._log_error(
                f"Exception during recovery attempt in {strategy_name}: {str(e)}",
                error_type="recovery_attempt_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_hash": record.hash}
            )
            # Re-raise the exception so the main error handling flow continues
            raise e
    return wrapper

class ModelManager:
    """
    A module for managing model loading, initialization, and switching in the SOVL system.
    Handles base model, scaffold models, tokenizers, and related configurations.
    """
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        """
        Initialize the ModelManager.

        Args:
            config_manager: ConfigManager instance for accessing configuration.
            logger: Logger instance for recording events and errors.
            device: Torch device (cuda/cpu) for model placement.
        """
        self._config_manager = config_manager
        self._logger = logger
        self._device = device
        self._memory_lock = Lock()
        self._last_failed_recovery_key = None  # Track last failed recovery attempt

        # Initialize error manager
        self._initialize_error_manager()

        # Initialize configuration
        self._initialize_config()

        # Model storage
        self.base_model = None
        self.scaffold_models = []  # List to support multiple scaffolds
        self.base_tokenizer = None
        self.scaffold_tokenizers = []  # List to support multiple scaffold tokenizers
        self.scaffold_unk_ids = []  # List to support multiple scaffold UNK IDs
        self.base_config = None
        self.lora_managers = []
        self.active_lora_checkpoint = None

        # Initialize models and tokenizers
        self.load_models()

    def _initialize_error_manager(self):
        """Initialize error manager with model-specific configuration."""
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,  # ModelManager doesn't need state tracking
            config_manager=self._config_manager,
            error_cooldown=1.0
        )
        
        # Register model-specific thresholds
        self.error_manager.severity_thresholds.update({
            "model_loading": 3,  # 3 failures before critical
            "memory": 5,        # 5 memory issues before critical
            "quantization": 2   # 2 quantization failures before critical
        })
        
        # Register recovery strategies
        self._register_recovery_strategies()

    def _register_recovery_strategies(self):
        """Register model-specific error recovery strategies."""
        self.error_manager.recovery_strategies.update({
            "model_loading_error": self._recover_model_loading,
            "memory_allocation_error": self._recover_memory_allocation,
            "quantization_error": self._recover_quantization,
            "tokenizer_error": self._recover_tokenizer
        })

    @_prevent_immediate_retry
    def _recover_model_loading(self, record: ErrorRecord) -> None:
        """Recovery strategy for model loading errors."""
        try:
            original_mode = self.quantization_mode
            new_mode = None
            
            # Determine next quantization mode
            if original_mode == "int4":
                new_mode = "int8"
            elif original_mode == "int8":
                new_mode = "fp16"
            elif original_mode == "fp16":
                self._log_error(
                    "Model loading failed at fp16. Cannot reduce precision further.",
                    error_type="model_loading_recovery_failure",
                    additional_info={"error_hash": record.hash, "quantization": original_mode}
                )
                raise RuntimeError(f"Model loading recovery failed: Already at lowest precision (fp16). Original error: {record.error}") from record.error

            if new_mode:
                self._log_event(
                    "model_loading_recovery_attempt",
                    f"Attempting to switch quantization from {original_mode} to {new_mode}.",
                    level="warning",
                    additional_info={"error_hash": record.hash}
                )
                # Clear existing models before changing mode
                self.cleanup()
                self.quantization_mode = new_mode
                # Attempt reload with new settings
                self.load_models()
                
                self._log_event(
                    "model_loading_recovery_success",
                    f"Successfully recovered from model loading error by switching to {self.quantization_mode}",
                    level="info",
                    additional_info={"error_hash": record.hash, "new_quantization": self.quantization_mode}
                )

        except Exception as e:
            self._log_error(
                f"Exception during model loading recovery attempt (target mode: {self.quantization_mode}): {str(e)}",
                error_type="model_loading_recovery_exception",
                stack_trace=traceback.format_exc(),
                additional_info={"error_hash": record.hash, "recovery_target_quantization": self.quantization_mode}
            )
            raise

    @_prevent_immediate_retry
    def _recover_memory_allocation(self, record: ErrorRecord) -> None:
        """Recovery strategy for memory allocation errors."""
        try:
            mode_changed = False
            original_mode = self.quantization_mode
            new_mode = None

            # Determine next quantization mode
            if original_mode == "fp16":
                new_mode = "int8"
            elif original_mode == "int8":
                new_mode = "int4"
            elif original_mode == "int4":
                self._log_event(
                    "memory_allocation_recovery_info",
                    "Already at int4 quantization. Cannot reduce precision further.",
                    level="warning",
                    additional_info={"error_hash": record.hash}
                )

            # Attempt to change mode if possible
            if new_mode:
                self._log_event(
                    "memory_allocation_recovery_attempt",
                    f"Attempting to switch quantization from {original_mode} to {new_mode} due to memory error.",
                    level="info",
                    additional_info={"error_hash": record.hash}
                )
                try:
                    self.set_quantization_mode(new_mode)
                    mode_changed = True
                    self._log_event(
                        "memory_allocation_recovery_step",
                        f"Successfully changed quantization to {self.quantization_mode}.",
                        level="info",
                        additional_info={"error_hash": record.hash}
                    )
                except Exception as e_quant:
                    self._log_error(
                        f"Failed to change quantization to {new_mode} during memory recovery: {str(e_quant)}",
                        error_type="memory_recovery_quant_fail",
                        stack_trace=traceback.format_exc(),
                        additional_info={"error_hash": record.hash, "target_mode": new_mode}
                    )

            # Clear CUDA cache AFTER attempting quantization change
            cache_cleared = False
            if torch.cuda.is_available():
                self._log_event("memory_allocation_recovery_attempt", "Clearing CUDA cache.", level="info", additional_info={"error_hash": record.hash})
                torch.cuda.empty_cache()
                cache_cleared = True
                self._log_event("memory_allocation_recovery_step", "Cleared CUDA cache.", level="info", additional_info={"error_hash": record.hash})

            # Final log summarizing actions taken
            if mode_changed or cache_cleared:
                self._log_event(
                    "memory_allocation_recovery_completed",
                    f"Memory recovery attempt finished. Mode changed: {mode_changed} (to {self.quantization_mode}). Cache cleared: {cache_cleared}.",
                    level="info",
                    additional_info={"error_hash": record.hash, "final_quantization": self.quantization_mode}
                )
            else:
                self._log_event(
                    "memory_allocation_recovery_noop",
                    "Memory recovery attempt finished. No actions taken (already int4, no CUDA cache cleared).",
                    level="warning",
                    additional_info={"error_hash": record.hash, "final_quantization": self.quantization_mode}
                )

        except Exception as e:
            self._log_error(
                f"Unexpected exception during _recover_memory_allocation: {str(e)}",
                error_type="recovery_internal_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error_hash": record.hash}
            )
            raise

    @_prevent_immediate_retry
    def _recover_quantization(self, record: ErrorRecord) -> None:
        """Recovery strategy for quantization errors."""
        try:
            original_mode = self.quantization_mode
            new_mode = None
            
            # Determine next quantization mode
            if original_mode == "int4":
                new_mode = "int8"
            elif original_mode == "int8":
                new_mode = "fp16"
            elif original_mode == "fp16":
                self._log_error(
                    "Quantization recovery failed: Already at fp16.",
                    error_type="quantization_recovery_failure",
                    additional_info={"error_hash": record.hash, "quantization": original_mode}
                )
                raise RuntimeError(f"Quantization recovery failed: Already at lowest precision (fp16). Original error: {record.error}") from record.error

            if new_mode:
                self._log_event(
                    "quantization_recovery_attempt",
                    f"Attempting to switch quantization from {original_mode} to {new_mode}.",
                    level="warning",
                    additional_info={"error_hash": record.hash}
                )
                self.set_quantization_mode(new_mode)
                self._log_event(
                    "quantization_recovery_success",
                    f"Recovered from quantization error by switching to {self.quantization_mode}",
                    level="info",
                    additional_info={"error_hash": record.hash, "new_quantization": self.quantization_mode}
                )

        except Exception as e:
            self._log_error(
                f"Exception during quantization recovery attempt (target mode: {new_mode}): {str(e)}",
                error_type="quantization_recovery_exception",
                stack_trace=traceback.format_exc(),
                additional_info={"error_hash": record.hash, "recovery_target_quantization": new_mode}
            )
            raise

    @_prevent_immediate_retry
    def _recover_tokenizer(self, record: ErrorRecord) -> None:
        """
        Recovery strategy for tokenizer errors.
        Note: This recovery clears *all* loaded tokenizers (base and scaffold)
        and attempts to reload them from scratch.
        """
        try:
            # Clear existing tokenizers
            self.base_tokenizer = None
            self.scaffold_tokenizers = []
            self.scaffold_unk_ids = []  # Also clear derived info

            self._log_event(
                "tokenizer_recovery_attempt",
                "Cleared all tokenizers. Attempting reload.",
                level="warning",
                additional_info={"error_hash": record.hash}
            )

            # Reload tokenizers
            self._load_tokenizers()

            self._log_event(
                "tokenizer_recovery_success",
                "Successfully reloaded tokenizers after error.",
                level="info",
                additional_info={"error_hash": record.hash}
            )
        except Exception as e:
            self._log_error(
                f"Exception during tokenizer recovery attempt: {str(e)}",
                error_type="tokenizer_recovery_exception",
                stack_trace=traceback.format_exc(),
                additional_info={"error_hash": record.hash}
            )
            raise

    def load_models(self, lora_checkpoint_path: str = None):
        """Load base and scaffold models along with their tokenizers.
        Optionally loads LoRA weights for scaffold models from checkpoint.
        """
        try:
            # Clear existing scaffold resources before loading
            self._clear_scaffold_resources()
            
            # Load tokenizers first
            self._load_tokenizers()
            
            # Load base model
            self._load_base_model()
            
            # Load each scaffold model
            self.scaffold_models = []
            self.lora_managers = []
            for model_name in self.scaffold_model_names:
                scaffold_model, lora_manager = self._load_scaffold_model(model_name, lora_checkpoint_path)
                self.scaffold_models.append(scaffold_model)
                self.lora_managers.append(lora_manager)
            
            self._log_event(
                "model_loading",
                "Successfully loaded all models",
                level="info",
                additional_info={
                    "base_model": self.base_model_name,
                    "scaffold_models": self.scaffold_model_names,
                    "num_scaffolds_loaded": len(self.scaffold_models),
                    "quantization": self.quantization_mode
                }
            )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="model_loading_error",
                severity=2,
                additional_info={
                    "base_model": self.base_model_name,
                    "scaffold_models": self.scaffold_model_names,
                    "quantization": self.quantization_mode,
                    "stage": "model_loading"
                }
            )
            raise

    def _clear_scaffold_resources(self):
        """Clear scaffold models and tokenizers."""
        with self._memory_lock:
            if self.scaffold_models:
                for model in self.scaffold_models:
                    del model
                self.scaffold_models = []
            self.scaffold_tokenizers = []
            self.scaffold_unk_ids = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._log_event("scaffold_cleanup", "Cleared scaffold model/tokenizer resources", level="debug")

    def _load_scaffold_model(self, model_name: str, lora_checkpoint_path: str = None):
        """
        Load a single scaffold model, apply scaffold wrapping, and attach LoRA adapters if enabled.
        Optionally loads LoRA weights from a checkpoint path (for long-term memory).
        """
        try:
            from sovl_scaffold import create_scaffold_with_lora
            scaffold_model, lora_manager = create_scaffold_with_lora(self._config_manager, self._logger, self._error_manager, lora_checkpoint_path)
            return scaffold_model, lora_manager
        except Exception as e:
            self._log_error(
                f"Failed to load scaffold model {model_name}: {str(e)}",
                error_type="scaffold_model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={"model_name": model_name}
            )
            raise

    def _load_base_model(self):
        """
        Load the base model and apply LoRA if enabled.
        """
        try:
            # 1. Load the raw base model (implementation-specific)
            base_model = self._load_raw_base_model()
            
            # 2. Apply LoRA if enabled
            from sovl_engram import LoraAdapterManager
            lora_manager = LoraAdapterManager(self._config_manager, self._logger, self._error_manager)
            if lora_manager.enabled:
                base_model = lora_manager.apply_to(base_model)
                self._logger.record_event(
                    event_type="lora_applied",
                    message=f"LoRA adapters applied to base model",
                    level="info"
                )
            else:
                self._logger.record_event(
                    event_type="lora_skipped",
                    message=f"LoRA not enabled for base model",
                    level="info"
                )
            self.base_model = base_model
            
            self._log_event(
                "base_model_loading",
                "Successfully loaded base model",
                level="info",
                additional_info={
                    "model": self.base_model_name,
                    "quantization": self.quantization_mode
                }
            )
        except Exception as e:
            self._log_error(
                f"Failed to load base model: {str(e)}",
                error_type="base_model_loading_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _load_raw_scaffold_model(self, model_name: str):
        """Load a single scaffold model with appropriate quantization."""
        try:
            with self._memory_lock:
                # Load the scaffold model with appropriate quantization
                if self.quantization_mode == "int4":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_4bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        quantization_config=bnb.nn.QuantizeConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                    )
                elif self.quantization_mode == "int8":
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_8bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                else:  # fp16
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                
                model.eval()
                return model
        except Exception as e:
            self._log_error(
                f"Failed to load raw scaffold model {model_name}: {str(e)}",
                error_type="scaffold_model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={"model_name": model_name}
            )
            raise

    def _load_raw_base_model(self):
        """Load the base model with appropriate quantization."""
        try:
            with self._memory_lock:
                if self.quantization_mode == "int4":
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        load_in_4bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        quantization_config=bnb.nn.QuantizeConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                    )
                elif self.quantization_mode == "int8":
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        load_in_8bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                else:  # fp16
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                
                self.base_model.eval()
                return self.base_model
        except Exception as e:
            self._log_error(
                f"Failed to load raw base model: {str(e)}",
                error_type="base_model_loading_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _load_tokenizers(self):
        """Load tokenizers for both base and scaffold models."""
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                padding_side="left",
                truncation_side="left"
            )
            
            self.scaffold_tokenizers = [AutoTokenizer.from_pretrained(
                name,
                padding_side="left",
                truncation_side="left"
            ) for name in self.scaffold_model_names]
            
            # Update scaffold_unk_ids after loading tokenizers
            self.scaffold_unk_ids = [tokenizer.unk_token_id for tokenizer in self.scaffold_tokenizers]
            
            self._log_event(
                "tokenizer_loading",
                "Successfully loaded tokenizers",
                level="info",
                additional_info={
                    "base_tokenizer": self.base_model_name,
                    "scaffold_tokenizers": self.scaffold_model_names
                }
            )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="tokenizer_error",
                severity=1,
                additional_info={
                    "base_tokenizer": self.base_model_name,
                    "scaffold_tokenizers": self.scaffold_model_names
                }
            )
            raise

    def set_quantization_mode(self, mode: str):
        """Set the quantization mode and reload models if needed."""
        try:
            validate_quantization_mode(mode)
            if mode != self.quantization_mode:
                self.quantization_mode = mode
                self.reload_models()
                
                self._log_event(
                    "quantization_change",
                    f"Changed quantization mode to {mode}",
                    level="info"
                )
        except Exception as e:
            # Log the error before handling it with the manager
            self._log_error(
                f"Failed to complete quantization mode change to {mode}. "
                f"Model state may be invalid (likely empty) due to reload failure. "
                f"Original error during reload: {str(e)}",
                error_type="quantization_change_failed",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "requested_mode": mode,
                    "target_mode": self.quantization_mode
                }
            )
            # Use the error manager
            self.error_manager.handle_error(
                error=e,
                error_type="quantization_error",
                severity=2,
                additional_info={
                    "requested_mode": mode,
                    "target_mode": self.quantization_mode,
                    "stage": "set_quantization_mode"
                }
            )
            raise e

    def reload_models(self):
        """Reload all models with current settings."""
        try:
            self.cleanup()
            self.load_models()
            
            self._log_event(
                "model_reloading",
                "Successfully reloaded all models",
                level="info",
                additional_info={
                    "quantization": self.quantization_mode
                }
            )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="model_loading_error",
                severity=2,
                additional_info={
                    "quantization": self.quantization_mode,
                    "stage": "model_reloading"
                }
            )
            raise

    def cleanup(self):
        """Clean up model resources."""
        try:
            with self._memory_lock:
                if self.base_model is not None:
                    del self.base_model
                    self.base_model = None
                
                self._clear_scaffold_resources()
                
                self._log_event(
                    "cleanup",
                    "Successfully cleaned up model resources",
                    level="info"
                )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="cleanup_error",
                severity=1,
                additional_info={"stage": "model_cleanup"}
            )
            raise

    def _validate_config_value(self, key: str, value: Any, expected_type: type, valid_values: Optional[List[Any]] = None, valid_range: Optional[Tuple[Any, Any]] = None) -> Any:
        """Validate a configuration value against type and constraints."""
        try:
            # Type validation
            if not isinstance(value, expected_type):
                raise ValueError(f"Config {key} must be of type {expected_type.__name__}")
            
            # Value validation
            if valid_values is not None and value not in valid_values:
                raise ValueError(f"Config {key}={value} not in valid values {valid_values}")
            
            # Range validation
            if valid_range is not None:
                min_val, max_val = valid_range
                if not (min_val <= value <= max_val):
                    raise ValueError(f"Config {key}={value} outside valid range [{min_val}, {max_val}]")
            
            return value
        except Exception as e:
            self._log_error(
                f"Config validation failed for {key}: {str(e)}",
                error_type="config_validation_error",
                context="config_validation"
            )
            raise

    def _get_model_memory_usage(self, model: Optional[nn.Module]) -> Optional[Dict[str, Any]]:
        """Get memory usage statistics for a model."""
        if model is None:
            return None
            
        try:
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated()
                }
            return {
                "parameters": sum(p.numel() for p in model.parameters()),
                "buffers": sum(b.numel() for b in model.buffers())
            }
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to get memory usage: {str(e)}",
                error_type="memory_usage_error",
                stack_trace=traceback.format_exc()
            )
            return None

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log an event with standardized format."""
        try:
            self._logger.record_event(
                event_type=event_type,
                message=message,
                level=level,
                additional_info=kwargs.get("additional_info", {})
            )
        except Exception as e:
            print(f"Failed to log event: {str(e)}")

    def _log_error(self, error_msg: str, error_type: str, stack_trace: Optional[str] = None, **kwargs) -> None:
        """Log an error with consistent formatting and context."""
        try:
            self._logger.log_error(
                error_msg=error_msg,
                error_type=error_type,
                stack_trace=stack_trace,
                context=kwargs.get("context", {}),
                additional_info=kwargs.get("additional_info", {})
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")

    def get_base_model(self) -> Optional[nn.Module]:
        """Return the base model. Thread-safe."""
        with self._memory_lock:
            return self.base_model

    def get_scaffold_model(self, index: int = 0) -> Optional[nn.Module]:
        """Return the scaffold model at the specified index. Thread-safe."""
        with self._memory_lock:
            try:
                num_models = len(self.scaffold_models)
                if not 0 <= index < num_models:
                    self._log_event(
                        "scaffold_model_index_error",
                        f"Invalid scaffold model index {index}. Available: {num_models}",
                        level="warning",
                        additional_info={"requested_index": index, "available_models": num_models}
                    )
                    return None
                return self.scaffold_models[index]
            except IndexError:
                self._log_error(
                    f"Internal error: Index {index} invalid for scaffold_models list of size {len(self.scaffold_models)} despite check.",
                    "scaffold_model_access_error"
                )
                return None
            except Exception as e:
                self._log_error(
                    f"Error accessing scaffold model at index {index}: {str(e)}",
                    "scaffold_model_access_error",
                    traceback.format_exc(),
                    additional_info={"index": index}
                )
                return None

    def get_base_tokenizer(self) -> Optional[AutoTokenizer]:
        """Return the base tokenizer. Thread-safe."""
        with self._memory_lock:
            return self.base_tokenizer

    def get_scaffold_tokenizer(self, index: int = 0) -> Optional[AutoTokenizer]:
        """Return the scaffold tokenizer at the specified index. Thread-safe."""
        with self._memory_lock:
            try:
                num_tokenizers = len(self.scaffold_tokenizers)
                if not 0 <= index < num_tokenizers:
                    self._log_event(
                        "scaffold_tokenizer_index_error",
                        f"Invalid scaffold tokenizer index {index}. Available: {num_tokenizers}",
                        level="warning",
                        additional_info={"requested_index": index, "available_tokenizers": num_tokenizers}
                    )
                    return None
                return self.scaffold_tokenizers[index]
            except IndexError:
                self._log_error(
                    f"Internal error: Index {index} invalid for scaffold_tokenizers list of size {len(self.scaffold_tokenizers)} despite check.",
                    "scaffold_tokenizer_access_error"
                )
                return None
            except Exception as e:
                self._log_error(
                    f"Error accessing scaffold tokenizer at index {index}: {str(e)}",
                    "scaffold_tokenizer_access_error",
                    traceback.format_exc(),
                    additional_info={"index": index}
                )
                return None

    def get_scaffold_unk_id(self, index: int = 0) -> Optional[int]:
        """Return the scaffold unknown token ID at the specified index. Thread-safe."""
        with self._memory_lock:
            try:
                num_ids = len(self.scaffold_unk_ids)
                if not 0 <= index < num_ids:
                    self._log_event(
                        "scaffold_unk_id_index_error",
                        f"Invalid scaffold unknown token ID index {index}. Available: {num_ids}",
                        level="warning",
                        additional_info={"requested_index": index, "available_ids": num_ids}
                    )
                    return None
                return self.scaffold_unk_ids[index]
            except IndexError:
                self._log_error(
                    f"Internal error: Index {index} invalid for scaffold_unk_ids list of size {len(self.scaffold_unk_ids)} despite check.",
                    "scaffold_unk_id_access_error"
                )
                return None
            except Exception as e:
                self._log_error(
                    f"Error accessing scaffold unknown token ID at index {index}: {str(e)}",
                    "scaffold_unk_id_access_error",
                    traceback.format_exc(),
                    additional_info={"index": index}
                )
                return None

    def get_num_scaffold_models(self) -> int:
        """Return the number of available scaffold models. Thread-safe."""
        with self._memory_lock:
            return len(self.scaffold_models)

    def _initialize_config(self) -> None:
        """Initialize and validate essential configuration parameters."""
        try:
            # Load base model name
            self.base_model_name = self._validate_config_value(
                "base_model_name",
                self._config_manager.get("model_config.base_model_name"),
                str
            )
            
            # Load scaffold model name(s) - Prefer list `scaffold_model_names`
            scaffold_name_single = self._config_manager.get("model_config.scaffold_model_name")  # Legacy/single name option
            scaffold_names_list = self._config_manager.get("model_config.scaffold_model_names")  # Preferred list option

            if scaffold_names_list is not None:
                # Validate the preferred list format
                if not isinstance(scaffold_names_list, list):
                    raise ValueError("Config 'scaffold_model_names' must be a list.")
                if not all(isinstance(name, str) and name for name in scaffold_names_list):
                    raise ValueError("Config 'scaffold_model_names' must be a list of non-empty strings.")
                self.scaffold_model_names = scaffold_names_list
                if scaffold_name_single is not None:
                    self._logger.record_event(
                        "config_info",
                        "Both 'scaffold_model_names' (list) and 'scaffold_model_name' (string) found. Using the list.",
                        level="info"
                    )
            elif scaffold_name_single is not None:
                # Fallback to the single name if list is not provided
                if not isinstance(scaffold_name_single, str) or not scaffold_name_single:
                    raise ValueError("Config 'scaffold_model_name' must be a non-empty string.")
                self._logger.record_event(
                    "config_fallback_used",
                    "Using legacy 'scaffold_model_name' config. Prefer 'scaffold_model_names' (list) for specifying scaffolds.",
                    level="warning",
                    additional_info={"scaffold_name": scaffold_name_single}
                )
                self.scaffold_model_names = [scaffold_name_single]
            else:
                # Neither specified, initialize as empty list
                self.scaffold_model_names = []
                self._logger.record_event(
                    "config_info",
                    "No scaffold models specified via 'scaffold_model_names' or 'scaffold_model_name'.",
                    level="info"
                )
            
            # Load quantization settings
            self.quantization_mode = self._validate_config_value(
                "quantization_mode",
                self._config_manager.get("model_config.quantization_mode", "fp16"),
                str,
                valid_values=["int4", "int8", "fp16"]
            )
            
            self._log_event(
                "config_initialization",
                "Successfully initialized model configuration",
                level="info",
                additional_info={
                    "base_model": self.base_model_name,
                    "scaffold_models": self.scaffold_model_names,
                    "quantization": self.quantization_mode
                }
            )
            
        except Exception as e:
            self._log_error(
                f"Failed to initialize configuration: {str(e)}",
                error_type="config_initialization_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def set_active_lora_checkpoint(self, lora_checkpoint_path: str):
        """
        Set the active LoRA checkpoint path for long-term memory tracking.
        """
        self.active_lora_checkpoint = lora_checkpoint_path
        self._logger.record_event(
            "lora_checkpoint_update",
            f"Active LoRA checkpoint set to {lora_checkpoint_path}",
            level="info"
        )
