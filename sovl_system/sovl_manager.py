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
from sovl_error import ErrorManager, ErrorRecord
from sovl_state import StateTracker

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

        # Initialize error manager
        self._initialize_error_manager()

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

    def _recover_model_loading(self, record: ErrorRecord) -> None:
        """Recovery strategy for model loading errors."""
        try:
            # Clear existing models
            self.cleanup()
            
            # Try reducing quantization level
            if self.quantization_mode == "int4":
                self.quantization_mode = "int8"
            elif self.quantization_mode == "int8":
                self.quantization_mode = "fp16"
                
            # Attempt reload with new settings
            self.load_models()
            
            self._log_event(
                "model_loading_recovery",
                "Recovered from model loading error",
                level="info",
                additional_info={
                    "new_quantization": self.quantization_mode
                }
            )
        except Exception as e:
            self._log_error(
                f"Failed to recover from model loading error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc()
            )

    def _recover_memory_allocation(self, record: ErrorRecord) -> None:
        """Recovery strategy for memory allocation errors."""
        try:
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Reduce model complexity if needed
            if "memory_usage" in record.additional_info:
                if self.quantization_mode == "fp16":
                    self.set_quantization_mode("int8")
                elif self.quantization_mode == "int8":
                    self.set_quantization_mode("int4")
                    
            self._log_event(
                "memory_allocation_recovery",
                "Recovered from memory allocation error",
                level="info",
                additional_info={"new_quantization": self.quantization_mode}
            )
        except Exception as e:
            self._log_error(
                f"Failed to recover from memory allocation: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc()
            )

    def _recover_quantization(self, record: ErrorRecord) -> None:
        """Recovery strategy for quantization errors."""
        try:
            # Try simpler quantization
            if self.quantization_mode == "int4":
                self.set_quantization_mode("int8")
            elif self.quantization_mode == "int8":
                self.set_quantization_mode("fp16")
                
            # Reload models with new quantization
            self.reload_models()
            
            self._log_event(
                "quantization_recovery",
                "Recovered from quantization error",
                level="info",
                additional_info={"new_quantization": self.quantization_mode}
            )
        except Exception as e:
            self._log_error(
                f"Failed to recover from quantization error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc()
            )

    def _recover_tokenizer(self, record: ErrorRecord) -> None:
        """Recovery strategy for tokenizer errors."""
        try:
            # Clear existing tokenizers
            self.base_tokenizer = None
            self.scaffold_tokenizer = None
            
            # Reload tokenizers
            self._load_tokenizers()
            
            self._log_event(
                "tokenizer_recovery",
                "Recovered from tokenizer error",
                level="info"
            )
        except Exception as e:
            self._log_error(
                f"Failed to recover from tokenizer error: {str(e)}",
                error_type="recovery_error",
                stack_trace=traceback.format_exc()
            )

    def load_models(self):
        """Load base and scaffold models along with their tokenizers."""
        try:
            # Load tokenizers first
            self._load_tokenizers()
            
            # Load base model
            self._load_base_model()
            
            # Load scaffold model
            self._load_scaffold_model()
            
            self._log_event(
                "model_loading",
                "Successfully loaded all models",
                level="info",
                additional_info={
                    "base_model": self.base_model_name,
                    "scaffold_model": self.scaffold_model_name,
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
                    "scaffold_model": self.scaffold_model_name,
                    "quantization": self.quantization_mode
                }
            )
            raise

    def _load_scaffold_model(self):
        """Load the scaffold model with appropriate quantization."""
        try:
            with self._memory_lock:
                if self.quantization_mode == "int4":
                    self.scaffold_models = [AutoModelForCausalLM.from_pretrained(
                        self.scaffold_model_name,
                        load_in_4bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        quantization_config=bnb.nn.QuantizeConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                    )]
                elif self.quantization_mode == "int8":
                    self.scaffold_models = [AutoModelForCausalLM.from_pretrained(
                        self.scaffold_model_name,
                        load_in_8bit=True,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )]
                else:  # fp16
                    self.scaffold_models = [AutoModelForCausalLM.from_pretrained(
                        self.scaffold_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )]
                
                for model in self.scaffold_models:
                    model.eval()
                
                self._log_event(
                    "scaffold_model_loading",
                    "Successfully loaded scaffold model",
                    level="info",
                    additional_info={
                        "model": self.scaffold_model_name,
                        "quantization": self.quantization_mode
                    }
                )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="model_loading_error",
                severity=2,
                additional_info={
                    "model": self.scaffold_model_name,
                    "quantization": self.quantization_mode,
                    "stage": "scaffold_model_loading"
                }
            )
            raise

    def _load_base_model(self):
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
            self.error_manager.handle_error(
                error=e,
                error_type="model_loading_error",
                severity=2,
                additional_info={
                    "model": self.base_model_name,
                    "quantization": self.quantization_mode,
                    "stage": "base_model_loading"
                }
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
            
            self.scaffold_tokenizer = AutoTokenizer.from_pretrained(
                self.scaffold_model_name,
                padding_side="left",
                truncation_side="left"
            )
            
            self._log_event(
                "tokenizer_loading",
                "Successfully loaded tokenizers",
                level="info",
                additional_info={
                    "base_tokenizer": self.base_model_name,
                    "scaffold_tokenizer": self.scaffold_model_name
                }
            )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="tokenizer_error",
                severity=1,
                additional_info={
                    "base_tokenizer": self.base_model_name,
                    "scaffold_tokenizer": self.scaffold_model_name
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
            self.error_manager.handle_error(
                error=e,
                error_type="quantization_error",
                severity=2,
                additional_info={
                    "requested_mode": mode,
                    "current_mode": self.quantization_mode
                }
            )
            raise

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
                
                if self.scaffold_models:
                    for model in self.scaffold_models:
                        del model
                    self.scaffold_models = []
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
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
