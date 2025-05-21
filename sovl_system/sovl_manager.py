import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from typing import Optional, List, Dict, Any, Tuple
import traceback
import os
from threading import Lock, RLock
import time
from functools import wraps
from sovl_utils import validate_quantization_mode
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager, ErrorRecord
import gc
from sovl_resource import ResourceManager
import curses  # Add curses for native progress bar

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
    Integrates ResourceManager for coordinated GPU memory management.
    """
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device, resource_manager: ResourceManager = None):
        """
        Initialize the ModelManager.
        Args:
            config_manager: ConfigManager instance for accessing configuration.
            logger: Logger instance for recording events and errors.
            device: Torch device (cuda/cpu) for model placement.
            resource_manager: Optional ResourceManager for coordinated resource allocation.
        """
        self._config_manager = config_manager
        self._logger = logger
        self._device = device
        self._memory_lock = RLock()
        self._gpu_lock = Lock()  # Add dedicated GPU lock
        self._last_failed_recovery_key = None  # Track last failed recovery attempt
        self.components = {}  # For compatibility with other modules

        # Initialize error manager FIRST
        self._initialize_error_manager()

        # ResourceManager integration
        if resource_manager is None:
            self.resource_manager = ResourceManager(logger=self._logger, error_manager=self.error_manager)
        else:
            self.resource_manager = resource_manager
        self.components["resource_manager"] = self.resource_manager
        
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
        try:
            self.load_models()
        except Exception:
            self._log_error(
                "Initial model loading failed during ModelManager construction. Attempting cleanup.", 
                "constructor_load_fail",
                stack_trace=traceback.format_exc()
            )
            try:
                self.cleanup() # Attempt cleanup
            except Exception as cleanup_exc:
                self._log_error(
                    f"Cleanup after constructor load failure also failed: {str(cleanup_exc)}", 
                    "constructor_cleanup_fail",
                    stack_trace=traceback.format_exc()
                )
            raise # Re-throw original load_models exception

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
                with self._gpu_lock:
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

    def _estimate_model_size(self, model_name: str) -> int:
        """Estimate model size in MB for resource acquisition."""
        try:
            config = AutoConfig.from_pretrained(model_name)
            dummy_model = AutoModelForCausalLM.from_config(config)
            param_count = sum(p.numel() for p in dummy_model.parameters())
            # Assume fp16 (2 bytes per param) unless quantization is known
            size_bytes = param_count * 2
            return size_bytes // (1024 * 1024)
        except Exception as e:
            self._log_error(
                f"Failed to estimate model size for {model_name}: {str(e)}",
                error_type="model_size_estimation_error",
                stack_trace=traceback.format_exc()
            )
            return 2048  # Default to 2GB if estimation fails

    def load_models(self, lora_checkpoint_path: str = None):
        """Load base and scaffold models along with their tokenizers with progress feedback."""
        try:
            with self._memory_lock:
                print(f"[ModelManager] Starting model loading for {self.base_model_name} and scaffolds {self.scaffold_model_names}")
                self._clear_scaffold_resources()
                self._load_tokenizers()
                # Estimate and acquire GPU memory for base model
                model_size_mb = self._estimate_model_size(self.base_model_name)
                if not self.resource_manager.acquire("gpu_memory", amount=model_size_mb):
                    raise RuntimeError("Insufficient GPU memory for base model")
                try:
                    print(f"[ModelManager] Loading base model: {self.base_model_name}")
                    self._load_base_model()
                    print(f"[ModelManager] Base model loaded successfully")
                except Exception as e:
                    self.resource_manager.release("gpu_memory", amount=model_size_mb)
                    raise
                self.scaffold_models = []
                self.lora_managers = []
                for model_name in self.scaffold_model_names:
                    scaffold_size_mb = self._estimate_model_size(model_name)
                    if not self.resource_manager.acquire("gpu_memory", amount=scaffold_size_mb):
                        raise RuntimeError(f"Insufficient GPU memory for scaffold model {model_name}")
                    try:
                        print(f"[ModelManager] Loading scaffold model: {model_name}")
                        scaffold_model, lora_manager = self._load_scaffold_model(model_name, lora_checkpoint_path)
                        self.scaffold_models.append(scaffold_model)
                        self.lora_managers.append(lora_manager)
                        print(f"[ModelManager] Scaffold model {model_name} loaded successfully")
                    except Exception as e:
                        self.resource_manager.release("gpu_memory", amount=scaffold_size_mb)
                        raise
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
                print(f"[ModelManager] All models loaded successfully")
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="model_loading_error",
                severity=2,
                additional_info={
                    "base_model": getattr(self, 'base_model_name', None),
                    "scaffold_models": getattr(self, 'scaffold_model_names', None),
                    "quantization": getattr(self, 'quantization_mode', None),
                    "stage": "model_loading"
                }
            )
            raise

    def _clear_scaffold_resources(self):
        """Clear scaffold models and tokenizers, and release GPU memory via ResourceManager."""
        with self._memory_lock:
            if self.scaffold_models:
                for model, model_name in zip(self.scaffold_models, self.scaffold_model_names):
                    scaffold_size_mb = self._estimate_model_size(model_name)
                    self.resource_manager.release("gpu_memory", amount=scaffold_size_mb)
                    del model
                self.scaffold_models = []
            self.scaffold_tokenizers = []
            self.scaffold_unk_ids = []
            if torch.cuda.is_available():
                with self._gpu_lock:
                    torch.cuda.empty_cache()
            self._log_event("scaffold_cleanup", "Cleared scaffold model/tokenizer resources", level="debug")

    def _load_scaffold_model(self, model_name: str, lora_checkpoint_path: str = None):
        """
        Load a single scaffold model, apply scaffold wrapping, and attach adaptation (LoRA, Adapters, Prefix Tuning) if enabled.
        Optionally loads LoRA weights from a checkpoint path (for long-term memory).
        """
        try:
            from sovl_scaffold import create_scaffold_with_adaptation
            scaffold_model, lora_manager, method_used = create_scaffold_with_adaptation(self._config_manager, self._logger, self._error_manager, lora_checkpoint_path)
            self._log_event(
                "scaffold_adaptation",
                f"Scaffold model loaded with adaptation method: {method_used}",
                level="info",
                additional_info={"model_name": model_name, "adaptation_method": method_used}
            )
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

    def _load_raw_base_model(self):
        """Load the base model with appropriate quantization and progress feedback."""
        screen = None
        try:
            with self._memory_lock:
                self._log_event(
                    "base_model_download_start",
                    f"Starting download for base model: {self.base_model_name}",
                    level="info",
                    additional_info={"model_name": self.base_model_name}
                )
                screen = self._initialize_curses()
                progress_callback = self._create_progress_callback(self.base_model_name, screen)
                self._last_progress_logged = 0.0

                if self.quantization_mode == "int4":
                    with self._gpu_lock:
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
                            ),
                            progress_callback=progress_callback
                        )
                elif self.quantization_mode == "int8":
                    with self._gpu_lock:
                        self.base_model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_name,
                            load_in_8bit=True,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            progress_callback=progress_callback
                        )
                else:  # fp16
                    with self._gpu_lock:
                        self.base_model = AutoModelForCausalLM.from_pretrained(
                            self.base_model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            progress_callback=progress_callback
                        )

                self.base_model.eval()
                self._log_event(
                    "base_model_download_complete",
                    f"Completed download and loading for base model: {self.base_model_name}",
                    level="info",
                    additional_info={"model_name": self.base_model_name}
                )
                return self.base_model
        except Exception as e:
            if screen:
                self._display_progress_bar(screen, self.base_model_name, 0.0, str(e))
                time.sleep(1)
            self._log_error(
                f"Failed to load raw base model: {str(e)}",
                error_type="base_model_loading_error",
                stack_trace=traceback.format_exc()
            )
            raise
        finally:
            self._cleanup_curses(screen)

    def _load_raw_scaffold_model(self, model_name: str):
        """Load a single scaffold model with appropriate quantization and progress feedback."""
        screen = None
        try:
            with self._memory_lock:
                self._log_event(
                    "scaffold_model_download_start",
                    f"Starting download for scaffold model: {model_name}",
                    level="info",
                    additional_info={"model_name": model_name}
                )
                screen = self._initialize_curses()
                progress_callback = self._create_progress_callback(model_name, screen)
                self._last_progress_logged = 0.0

                if self.quantization_mode == "int4":
                    with self._gpu_lock:
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
                            ),
                            progress_callback=progress_callback
                        )
                elif self.quantization_mode == "int8":
                    with self._gpu_lock:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            load_in_8bit=True,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            progress_callback=progress_callback
                        )
                else:  # fp16
                    with self._gpu_lock:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                            progress_callback=progress_callback
                        )

                model.eval()
                self._log_event(
                    "scaffold_model_download_complete",
                    f"Completed download and loading for scaffold model: {model_name}",
                    level="info",
                    additional_info={"model_name": model_name}
                )
                return model
        except Exception as e:
            if screen:
                self._display_progress_bar(screen, model_name, 0.0, str(e))
                time.sleep(1)
            self._log_error(
                f"Failed to load raw scaffold model {model_name}: {str(e)}",
                error_type="scaffold_model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={"model_name": model_name}
            )
            raise
        finally:
            self._cleanup_curses(screen)

    def _load_tokenizers(self):
        """Load tokenizers for both base and scaffold models with progress feedback."""
        screen = None
        try:
            with self._memory_lock:
                self._log_event(
                    "tokenizer_download_start",
                    f"Starting download for base tokenizer: {self.base_model_name}",
                    level="info",
                    additional_info={"model_name": self.base_model_name}
                )
                screen = self._initialize_curses()
                progress_callback = self._create_progress_callback(f"{self.base_model_name}_tokenizer", screen)
                self._last_progress_logged = 0.0
                self.base_tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    padding_side="left",
                    truncation_side="left",
                    progress_callback=progress_callback
                )
                self._log_event(
                    "tokenizer_download_complete",
                    f"Completed download for base tokenizer: {self.base_model_name}",
                    level="info",
                    additional_info={"model_name": self.base_model_name}
                )

                self.scaffold_tokenizers = []
                self.scaffold_unk_ids = []
                for name in self.scaffold_model_names:
                    self._log_event(
                        "tokenizer_download_start",
                        f"Starting download for scaffold tokenizer: {name}",
                        level="info",
                        additional_info={"model_name": name}
                    )
                    progress_callback = self._create_progress_callback(f"{name}_tokenizer", screen)
                    self._last_progress_logged = 0.0
                    tokenizer = AutoTokenizer.from_pretrained(
                        name,
                        padding_side="left",
                        truncation_side="left",
                        progress_callback=progress_callback
                    )
                    self.scaffold_tokenizers.append(tokenizer)
                    self.scaffold_unk_ids.append(tokenizer.unk_token_id)
                    self._log_event(
                        "tokenizer_download_complete",
                        f"Completed download for scaffold tokenizer: {name}",
                        level="info",
                        additional_info={"model_name": name}
                    )

                self._log_event(
                    "tokenizer_loading",
                    "Successfully loaded all tokenizers",
                    level="info",
                    additional_info={
                        "base_tokenizer": self.base_model_name,
                        "scaffold_tokenizers": self.scaffold_model_names
                    }
                )
        except Exception as e:
            if screen:
                self._display_progress_bar(screen, self.base_model_name, 0.0, str(e))
                time.sleep(1)
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
        finally:
            self._cleanup_curses(screen)

    def set_quantization_mode(self, mode: str):
        """Set the quantization mode and reload models if needed."""
        try:
            validate_quantization_mode(mode)
            with self._memory_lock:
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
        """Reload all models with current settings. Ensures rollback and error safety on failure."""
        try:
            with self._memory_lock:
                self.cleanup()
                try:
                    self.load_models()
                except Exception as load_exc:
                    # Rollback: ensure all model references are set to None and cleanup is called again
                    self._log_event(
                        "reload_rollback",
                        f"load_models failed during reload: {str(load_exc)}. Rolling back to safe state.",
                        level="warning"
                    )
                    with torch.no_grad():
                        self.base_model = None
                        self.scaffold_models = []
                        self.base_tokenizer = None
                        self.scaffold_tokenizers = []
                        self.scaffold_unk_ids = []
                    self.cleanup()
                    raise  # Re-raise the original exception
                self._log_event(
                    "model_reloading",
                    "Successfully reloaded all models",
                    level="info",
                    additional_info={
                        "quantization": self.quantization_mode
                    }
                )
                # Log memory usage after reload
                self.report_gpu_memory_usage()
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
        """Clean up model resources and ensure GPU memory is released."""
        try:
            with self._memory_lock:
                gpu_mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
                self._log_event(
                    "cleanup_start",
                    f"Starting cleanup. GPU memory before: {gpu_mem_before}",
                    level="debug"
                )
                with torch.no_grad():
                    if self.base_model is not None:
                        model_size_mb = self._estimate_model_size(self.base_model_name)
                        self.resource_manager.release("gpu_memory", amount=model_size_mb)
                        del self.base_model
                        self.base_model = None
                    for model, model_name in zip(self.scaffold_models, self.scaffold_model_names):
                        scaffold_size_mb = self._estimate_model_size(model_name)
                        self.resource_manager.release("gpu_memory", amount=scaffold_size_mb)
                    self._clear_scaffold_resources()
                gc.collect()
                if torch.cuda.is_available():
                    with self._gpu_lock:
                        torch.cuda.empty_cache()
                gpu_mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
                self._log_event(
                    "cleanup_end",
                    f"Cleanup complete. GPU memory after: {gpu_mem_after}",
                    level="debug",
                    additional_info={"gpu_mem_before": gpu_mem_before, "gpu_mem_after": gpu_mem_after}
                )
                if gpu_mem_after is not None and gpu_mem_before is not None and gpu_mem_after > 0.5 * gpu_mem_before:
                    self._log_event(
                        "cleanup_warning",
                        f"GPU memory not fully released after cleanup. Before: {gpu_mem_before}, After: {gpu_mem_after}",
                        level="warning"
                    )
                self.report_gpu_memory_usage()
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

    def report_gpu_memory_usage(self):
        """Report current GPU memory usage for monitoring purposes."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            self._log_event(
                "gpu_memory_report",
                f"GPU memory usage - Allocated: {allocated}, Reserved: {reserved}, Max Allocated: {max_allocated}",
                level="info",
                additional_info={
                    "allocated": allocated,
                    "reserved": reserved,
                    "max_allocated": max_allocated
                }
            )
            return {"allocated": allocated, "reserved": reserved, "max_allocated": max_allocated}
        else:
            self._log_event(
                "gpu_memory_report",
                "CUDA not available.",
                level="info"
            )
            return None

    def get_max_context_length(self, model=None) -> int:
        """
        Return the max context length for the given model (or current base model if not specified).
        Checks common config attributes and falls back to config or a safe default.
        """
        if model is None:
            model = self.get_base_model()
        if model is not None and hasattr(model, "config"):
            for attr in ['max_position_embeddings', 'n_ctx', 'seq_length', 'max_seq_len']:
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)
        # Fallback to config or a safe default
        if hasattr(self, "_config_manager"):
            return self._config_manager.get("controls_config.max_seq_length", 2048)
        return 2048

    @property
    def max_context_length(self):
        """
        Property for convenient access to the current base model's max context length.
        """
        return self.get_max_context_length()

    # --- Curses Progress Bar Helpers ---

    def _initialize_curses(self) -> Optional["curses.window"]:
        """Initialize a curses window for progress bar display."""
        try:
            screen = curses.initscr()
            curses.noecho()
            curses.cbreak()
            screen.nodelay(1)
            curses.start_color()
            curses.use_default_colors()
            if curses.has_colors():
                curses.init_pair(1, curses.COLOR_GREEN, -1)  # Green for progress
                curses.init_pair(2, curses.COLOR_RED, -1)   # Red for errors
            return screen
        except Exception as e:
            self._log_error(
                f"Failed to initialize curses for progress bar: {str(e)}",
                error_type="curses_init_error",
                stack_trace=traceback.format_exc()
            )
            return None

    def _cleanup_curses(self, screen: Optional["curses.window"]):
        """Clean up curses to restore terminal state."""
        if screen is not None:
            try:
                screen.nodelay(0)
                curses.nocbreak()
                curses.echo()
                curses.endwin()
            except Exception as e:
                self._log_error(
                    f"Failed to clean up curses: {str(e)}",
                    error_type="curses_cleanup_error",
                    stack_trace=traceback.format_exc()
                )

    def _display_progress_bar(self, screen: Optional["curses.window"], model_name: str, progress: float, error: Optional[str] = None):
        """Display a progress bar in the curses window."""
        if screen is None:
            return
        try:
            with self._memory_lock:
                screen.clear()
                max_y, max_x = screen.getmaxyx()
                bar_width = min(50, max_x - 10)
                filled = int(bar_width * progress)
                bar = '=' * filled + '-' * (bar_width - filled)
                display_name = model_name.split('/')[-1][:max_x-20]
                if error:
                    status = f"Error: {error[:max_x-10]}"
                    color = curses.color_pair(2)
                else:
                    status = f"Downloading {display_name}: [{bar}] {progress*100:.1f}%"
                    color = curses.color_pair(1)
                screen.addstr(0, 0, status[:max_x-1], color)
                screen.refresh()
        except Exception as e:
            self._log_error(
                f"Curses display error for progress bar: {str(e)}",
                error_type="curses_display_error",
                stack_trace=traceback.format_exc()
            )

    def _create_progress_callback(self, model_name: str, screen: Optional["curses.window"]):
        """Create a progress callback for huggingface_hub downloads with curses progress bar."""
        def callback(downloaded: int, total: int):
            if total:
                progress = downloaded / total
                self._display_progress_bar(screen, model_name, progress)
                # Log progress every ~10%
                if not hasattr(self, '_last_progress_logged'):
                    self._last_progress_logged = 0.0
                if progress >= (self._last_progress_logged + 0.1):
                    self._log_event(
                        "model_download_progress",
                        f"Downloading {model_name}: {progress*100:.1f}%",
                        level="info",
                        additional_info={"model_name": model_name, "progress_percent": progress*100}
                    )
                    self._last_progress_logged = progress
        return callback
