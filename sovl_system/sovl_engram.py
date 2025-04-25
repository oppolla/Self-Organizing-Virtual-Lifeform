from typing import Any, Dict, Optional
import torch
from sovl_logger import Logger
from sovl_error import ErrorManager, ScaffoldError
from sovl_config import ConfigManager

"""
sovl_engram.py
LoRA Adapter and Adapter Management for SOVL System

This module provides utilities for integrating, configuring, saving, and loading LoRA (Low-Rank Adapter) modules for transformer-based models in the SOVL system.
"""

# Optional: Only import PEFT if available
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None

class LoraAdapterManager:
    """
    Utility for applying, saving, and loading LoRA adapters to transformer models.
    Integrates system-standard logging, error handling, and configuration.
    """
    def __init__(self, config_manager: ConfigManager, logger: Optional[Logger] = None, error_handler: Optional[ErrorManager] = None):
        self.config_manager = config_manager
        self.logger = logger or Logger()
        self.error_handler = error_handler or ErrorManager(self.logger)
        # Fetch LoRA config from system config manager
        self.enabled = config_manager.get("lora.enable", True)
        self.rank = config_manager.get("lora.rank", 8)
        self.alpha = config_manager.get("lora.alpha", 16)
        self.dropout = config_manager.get("lora.dropout", 0.1)
        self.target_modules = config_manager.get("lora.target_modules", ["q_proj", "v_proj"])
        self.task_type = config_manager.get("lora.task_type", "CAUSAL_LM")
        self._validate_config()

    def _validate_config(self):
        """
        Validate LoRA config values. Log warnings/errors for invalid settings.
        """
        if not isinstance(self.rank, int) or self.rank <= 0:
            self.logger.log_warning("LoRA rank should be a positive integer.", event_type="lora_config_warning")
        if not isinstance(self.alpha, (int, float)) or self.alpha <= 0:
            self.logger.log_warning("LoRA alpha should be a positive number.", event_type="lora_config_warning")
        if not isinstance(self.dropout, (int, float)) or not (0.0 <= self.dropout <= 1.0):
            self.logger.log_warning("LoRA dropout should be between 0.0 and 1.0.", event_type="lora_config_warning")
        if not isinstance(self.target_modules, (list, tuple)) or not all(isinstance(m, str) for m in self.target_modules):
            self.logger.log_warning("LoRA target_modules should be a list of strings.", event_type="lora_config_warning")
        valid_task_types = {"CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS", "SEQ_2_SEQ_LM"}
        if self.task_type not in valid_task_types:
            self.logger.log_warning(f"LoRA task_type '{self.task_type}' is not recognized. Using default 'CAUSAL_LM'.", event_type="lora_config_warning")

    def apply_to(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.enabled or LoraConfig is None or get_peft_model is None:
            self.logger.log_debug("LoRA not enabled or PEFT not available; returning original model.", event_type="lora_adapter_skip")
            return model
        try:
            lora_config = LoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                target_modules=self.target_modules,
                lora_dropout=self.dropout,
                bias="none",
                task_type=getattr(TaskType, self.task_type, TaskType.CAUSAL_LM)
            )
            model = get_peft_model(model, lora_config)
            self.logger.log_info(f"LoRA adapters applied: rank={self.rank}, alpha={self.alpha}, modules={self.target_modules}", event_type="lora_adapter_applied")
            return model
        except Exception as e:
            self.error_handler.handle_error(e, operation="apply_lora_adapter", context={"config": self.config_manager.get_section("lora")})
            raise ScaffoldError(f"Failed to apply LoRA adapters: {e}", operation="apply_lora_adapter")

    def save_lora_weights(self, model: torch.nn.Module, path: str):
        try:
            if PeftModel is not None and isinstance(model, PeftModel):
                model.save_pretrained(path)
                self.logger.log_info(f"LoRA weights saved to {path}", event_type="lora_weights_save")
            else:
                raise ScaffoldError("Model is not a PEFT/LoRA model; cannot save LoRA weights.", operation="save_lora_weights")
        except Exception as e:
            self.error_handler.handle_error(e, operation="save_lora_weights", context={"path": path})
            raise

    def load_lora_weights(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        try:
            if PeftModel is not None and hasattr(PeftModel, 'from_pretrained'):
                loaded_model = PeftModel.from_pretrained(model, path)
                self.logger.log_info(f"LoRA weights loaded from {path}", event_type="lora_weights_load")
                return loaded_model
            else:
                raise ScaffoldError("PEFT/LoRA not available or model incompatible.", operation="load_lora_weights")
        except Exception as e:
            self.error_handler.handle_error(e, operation="load_lora_weights", context={"path": path})
            raise

    @staticmethod
    def lora_parameters(model: torch.nn.Module):
        """Yield only LoRA parameters for optimizer."""
        for n, p in model.named_parameters():
            if p.requires_grad and ("lora" in n or "adapter" in n):
                yield p
