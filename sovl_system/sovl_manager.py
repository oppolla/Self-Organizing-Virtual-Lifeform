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
from sovl_utils import validate_quantization_mode
from sovl_config import ConfigManager
from sovl_logger import Logger

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

        # Initialize configuration
        self._initialize_config()

        # Model storage
        self.base_model = None
        self.scaffold_models = []  # List to support multiple scaffolds if needed
        self.base_tokenizer = None
        self.scaffold_tokenizer = None
        self.base_config = None
        self.scaffold_config = None

        # Initialize models and tokenizers
        self.load_models()

    def load_models(self):
        """Load base and scaffold models along with their tokenizers."""
        with self._memory_lock:
            try:
                # Clear existing models and memory first
                self.cleanup()
                
                start_time = time.time()
                self._load_base_model()
                base_load_time = time.time() - start_time
                
                start_time = time.time()
                self._load_scaffold_model()
                scaffold_load_time = time.time() - start_time
                
                start_time = time.time()
                self._load_tokenizers()
                tokenizer_load_time = time.time() - start_time
                
                total_load_time = time.time() - start_time
                
                self._log_event(
                    "models_loaded",
                    "All models and tokenizers loaded successfully",
                    level="info",
                    additional_info={
                        "base_model": self.base_model_name,
                        "scaffold_model": self.scaffold_model_name,
                        "quantization": self.quantization_mode,
                        "load_times": {
                            "base_model": base_load_time,
                            "scaffold_model": scaffold_load_time,
                            "tokenizers": tokenizer_load_time,
                            "total": total_load_time
                        },
                        "memory_usage": {
                            "base_model": self._get_model_memory_usage(self.base_model),
                            "scaffold_model": self._get_model_memory_usage(self.scaffold_models[0]) if self.scaffold_models else None
                        }
                    }
                )
            except Exception as e:
                # Clean up any partially loaded models
                self.cleanup()
                self._log_error(
                    f"Model loading failed: {str(e)}",
                    error_type="model_loading_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "base_model": self.base_model_name,
                        "scaffold_model": self.scaffold_model_name
                    }
                )
                raise

    def _load_scaffold_model(self):
        """Load the scaffold model, optionally with LoRA adapters."""
        try:
            start_time = time.time()
            model_path = self.scaffold_model_path if self.scaffold_model_path else self.scaffold_model_name
            self.scaffold_config = AutoConfig.from_pretrained(model_path)
            config_load_time = time.time() - start_time
            
            start_time = time.time()
            quantization_config = self._get_quantization_config()
            scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=self.scaffold_config,
                **quantization_config
            )
            model_load_time = time.time() - start_time
            
            start_time = time.time()
            if self.enable_lora:
                lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    target_modules=self._config_manager.get("lora_config.lora_target_modules", ["c_attn", "c_proj", "c_fc"]),
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.scaffold_models = [get_peft_model(scaffold_model_raw, lora_config).to(self._device)]
                lora_message = "LoRA adapters applied to scaffold model"
            else:
                self.scaffold_models = [scaffold_model_raw.to(self._device)]
                lora_message = "Scaffold model loaded without LoRA adapters"
            setup_time = time.time() - start_time
            
            self._log_event(
                "scaffold_model_loaded",
                f"{lora_message} from {'local path' if self.scaffold_model_path else 'Hugging Face hub'}: {model_path}",
                level="info",
                additional_info={
                    "model_path": model_path,
                    "lora_enabled": self.enable_lora,
                    "load_times": {
                        "config": config_load_time,
                        "model": model_load_time,
                        "setup": setup_time,
                        "total": config_load_time + model_load_time + setup_time
                    },
                    "memory_usage": self._get_model_memory_usage(self.scaffold_models[0]) if self.scaffold_models else None
                }
            )
        except Exception as e:
            self._log_error(
                f"Failed to load scaffold model: {str(e)}",
                error_type="scaffold_model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": model_path,
                    "lora_enabled": self.enable_lora
                }
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

    def _load_base_model(self):
        """Load the base model with specified quantization."""
        try:
            start_time = time.time()
            model_path = self.base_model_path if self.base_model_path else self.base_model_name
            self.base_config = AutoConfig.from_pretrained(model_path)
            config_load_time = time.time() - start_time
            
            start_time = time.time()
            quantization_config = self._get_quantization_config()
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=self.base_config,
                **quantization_config
            ).to(self._device)
            model_load_time = time.time() - start_time
            
            start_time = time.time()
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
            setup_time = time.time() - start_time
            
            self._log_event(
                "base_model_loaded",
                f"Base model loaded from {'local path' if self.base_model_path else 'Hugging Face hub'}: {model_path}",
                level="info",
                additional_info={
                    "model_path": model_path,
                    "quantization": self.quantization_mode,
                    "load_times": {
                        "config": config_load_time,
                        "model": model_load_time,
                        "setup": setup_time,
                        "total": config_load_time + model_load_time + setup_time
                    },
                    "memory_usage": self._get_model_memory_usage(self.base_model)
                }
            )
        except Exception as e:
            self._log_error(
                f"Failed to load base model: {str(e)}",
                error_type="base_model_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_path": model_path,
                    "quantization": self.quantization_mode
                }
            )
            raise

    def _load_tokenizers(self):
        """Load tokenizers for base and scaffold models."""
        try:
            start_time = time.time()
            base_model_path = self.base_model_path if self.base_model_path else self.base_model_name
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_tokenizer_time = time.time() - start_time
            
            start_time = time.time()
            scaffold_model_path = self.scaffold_model_path if self.scaffold_model_path else self.scaffold_model_name
            self.scaffold_tokenizer = AutoTokenizer.from_pretrained(scaffold_model_path)
            scaffold_tokenizer_time = time.time() - start_time
            
            start_time = time.time()
            # Set padding tokens if not defined
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            if self.scaffold_tokenizer.pad_token is None:
                self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
                
            # Update model configs
            self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
            if self.scaffold_models:
                self.scaffold_models[0].config.pad_token_id = self.scaffold_tokenizer.pad_token_id
                
            # Set scaffold_unk_id if not set
            if self.scaffold_unk_id is None:
                self.scaffold_unk_id = self.scaffold_tokenizer.unk_token_id
                self._config_manager.set("controls_config.scaffold_unk_id", self.scaffold_unk_id)
            setup_time = time.time() - start_time
                
            self._log_event(
                "tokenizers_loaded",
                "Tokenizers loaded and configured",
                level="info",
                additional_info={
                    "base_model": base_model_path,
                    "scaffold_model": scaffold_model_path,
                    "load_times": {
                        "base_tokenizer": base_tokenizer_time,
                        "scaffold_tokenizer": scaffold_tokenizer_time,
                        "setup": setup_time,
                        "total": base_tokenizer_time + scaffold_tokenizer_time + setup_time
                    }
                }
            )
        except Exception as e:
            self._log_error(
                f"Failed to load tokenizers: {str(e)}",
                error_type="tokenizer_loading_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "base_model": base_model_path,
                    "scaffold_model": scaffold_model_path
                }
            )
            raise

    def _get_quantization_config(self) -> Dict[str, Any]:
        """Return quantization configuration based on mode."""
        if self.quantization_mode == "int8":
            return {"load_in_8bit": True}
        elif self.quantization_mode == "int4":
            return {"load_in_4bit": True}
        return {}

    def switch_base_model(self, model_name: str, quantization: Optional[str] = None):
        """
        Switch the base model to a new one.

        Args:
            model_name: Name of the new model (e.g., "gpt2", "bert-base-uncased").
            quantization: Optional quantization mode ("fp16", "int8", "int4").
        """
        with self._memory_lock:
            try:
                # Validate new model configuration before switching
                old_model_name = self.base_model_name
                old_quantization = self.quantization_mode
                
                # Update configuration
                self.base_model_name = self._validate_config_value(
                    "base_model_name",
                    model_name,
                    str
                )
                
                if quantization:
                    self.quantization_mode = self._validate_config_value(
                        "quantization",
                        quantization,
                        str,
                        valid_values=["fp16", "int8", "int4"]
                    )
                
                # Try to load the new model first
                try:
                    self._load_base_model()
                    self._load_tokenizers()  # Reload tokenizers to match new base model
                except Exception as e:
                    # Revert to old configuration if loading fails
                    self.base_model_name = old_model_name
                    self.quantization_mode = old_quantization
                    self._log_error(
                        f"Failed to load new base model: {str(e)}",
                        error_type="base_model_switching_error",
                        stack_trace=traceback.format_exc(),
                        additional_info={
                            "model_name": model_name,
                            "quantization": quantization
                        }
                    )
                    raise
                
                # Update config manager only after successful load
                self._config_manager.set("core_config.base_model_name", self.base_model_name)
                if quantization:
                    self._config_manager.set("core_config.quantization", self.quantization_mode)

                self._log_event(
                    "base_model_switched",
                    f"Switched base model to {model_name} with quantization {self.quantization_mode}",
                    level="info",
                    additional_info={
                        "new_model": model_name,
                        "quantization": self.quantization_mode
                    }
                )
            except Exception as e:
                self._log_error(
                    f"Failed to switch base model: {str(e)}",
                    error_type="base_model_switching_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "model_name": model_name,
                        "quantization": quantization
                    }
                )
                raise

    def switch_scaffold_model(self, model_name: str, quantization: Optional[str] = None, apply_lora: Optional[bool] = None):
        """
        Switch the scaffold model to a new one.

        Args:
            model_name: Name of the new scaffold model.
            quantization: Optional quantization mode ("fp16", "int8", "int4").
            apply_lora: Whether to apply LoRA adapters (defaults to config setting).
        """
        with self._memory_lock:
            try:
                # Validate new model configuration before switching
                old_model_name = self.scaffold_model_name
                old_quantization = self.quantization_mode
                old_lora_enabled = self.enable_lora
                
                # Update configuration
                self.scaffold_model_name = self._validate_config_value(
                    "scaffold_model_name",
                    model_name,
                    str
                )
                
                if quantization:
                    self.quantization_mode = self._validate_config_value(
                        "quantization",
                        quantization,
                        str,
                        valid_values=["fp16", "int8", "int4"]
                    )
                
                if apply_lora is not None:
                    self.enable_lora = self._validate_config_value(
                        "enable_lora_adapters",
                        apply_lora,
                        bool
                    )
                
                # Try to load the new model first
                try:
                    self._load_scaffold_model()
                    self._load_tokenizers()  # Reload tokenizers to match new scaffold model
                except Exception as e:
                    # Revert to old configuration if loading fails
                    self.scaffold_model_name = old_model_name
                    self.quantization_mode = old_quantization
                    self.enable_lora = old_lora_enabled
                    self._log_error(
                        f"Failed to load new scaffold model: {str(e)}",
                        error_type="scaffold_model_switching_error",
                        stack_trace=traceback.format_exc(),
                        additional_info={
                            "model_name": model_name,
                            "quantization": quantization,
                            "lora_enabled": apply_lora
                        }
                    )
                    raise
                
                # Update config manager only after successful load
                self._config_manager.set("core_config.scaffold_model_name", self.scaffold_model_name)
                if quantization:
                    self._config_manager.set("core_config.quantization", self.quantization_mode)
                if apply_lora is not None:
                    self._config_manager.set("lora_config.enable_lora_adapters", self.enable_lora)

                self._log_event(
                    "scaffold_model_switched",
                    f"Switched scaffold model to {model_name} with quantization {self.quantization_mode}",
                    level="info",
                    additional_info={
                        "new_model": model_name,
                        "quantization": self.quantization_mode,
                        "lora_enabled": self.enable_lora
                    }
                )
            except Exception as e:
                self._log_error(
                    f"Failed to switch scaffold model: {str(e)}",
                    error_type="scaffold_model_switching_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "model_name": model_name,
                        "quantization": quantization,
                        "lora_enabled": apply_lora
                    }
                )
                raise

    def set_quantization_mode(self, mode: str):
        """
        Set quantization mode without reloading models.

        Args:
            mode: Quantization mode ("fp16", "int8", "int4").
        """
        try:
            validated_mode = self._validate_config_value(
                "quantization",
                mode,
                str,
                valid_values=["fp16", "int8", "int4"]
            )
            
            if validated_mode != self.quantization_mode:
                self.quantization_mode = validated_mode
                self._config_manager.set("core_config.quantization", validated_mode)
                self._log_event(
                    "quantization_mode_set",
                    f"Quantization mode set to '{validated_mode}'. Restart or reload models to apply.",
                    level="info",
                    additional_info={
                        "mode": validated_mode
                    }
                )
                print(f"Quantization mode set to '{validated_mode}'. Restart or reload models to apply.")
            else:
                print(f"Invalid mode '{mode}' or no change.")
        except Exception as e:
            self._log_error(
                f"Failed to set quantization mode: {str(e)}",
                error_type="quantization_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "mode": mode
                }
            )
            raise

    def reload_models(self):
        """Reload all models with current configuration."""
        with self._memory_lock:
            try:
                # Clean up existing models
                if self.base_model:
                    del self.base_model
                for model in self.scaffold_models:
                    del model
                self.base_model = None
                self.scaffold_models = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reload models
                self.load_models()

                self._log_event(
                    "models_reloaded",
                    "All models reloaded successfully",
                    level="info",
                    additional_info={
                        "base_model": self.base_model_name,
                        "scaffold_model": self.scaffold_model_name,
                        "quantization": self.quantization_mode
                    }
                )
                print("All models reloaded successfully.")
            except Exception as e:
                self._log_error(
                    f"Model reload failed: {str(e)}",
                    error_type="model_reloading_error",
                    stack_trace=traceback.format_exc()
                )
                raise

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
        """Return the base model."""
        return self.base_model

    def get_scaffold_model(self, index: int = 0) -> Optional[nn.Module]:
        """Return the scaffold model at the specified index."""
        return self.scaffold_models[index] if index < len(self.scaffold_models) else None

    def get_base_tokenizer(self) -> Optional[AutoTokenizer]:
        """Return the base tokenizer."""
        return self.base_tokenizer

    def get_scaffold_tokenizer(self) -> Optional[AutoTokenizer]:
        """Return the scaffold tokenizer."""
        return self.scaffold_tokenizer

    def get_scaffold_unk_id(self) -> Optional[int]:
        """Return the scaffold unknown token ID."""
        return self.scaffold_unk_id

    def cleanup(self):
        """Clean up models and free memory."""
        with self._memory_lock:
            try:
                # Clear base model
                if self.base_model is not None:
                    del self.base_model
                    self.base_model = None
                
                # Clear scaffold models
                if self.scaffold_models:
                    for model in self.scaffold_models:
                        del model
                    self.scaffold_models = []
                
                # Clear tokenizers
                self.base_tokenizer = None
                self.scaffold_tokenizer = None
                
                # Clear configurations
                self.base_config = None
                self.scaffold_config = None
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self._log_event(
                    "model_manager_cleanup",
                    "ModelManager cleanup completed",
                    level="info",
                    additional_info={
                        "memory_cleared": True,
                        "gpu_available": torch.cuda.is_available()
                    }
                )
            except Exception as e:
                self._log_error(
                    f"ModelManager cleanup failed: {str(e)}",
                    error_type="model_manager_cleanup_error",
                    stack_trace=traceback.format_exc()
                )
                raise
