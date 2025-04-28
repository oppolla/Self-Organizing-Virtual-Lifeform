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
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PrefixTuningConfig
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None
    PrefixTuningConfig = None

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
        lora_section = config_manager.get_section("lora") if hasattr(config_manager, 'get_section') else None
        self.enabled = config_manager.get("lora.enable", True)
        self.rank = config_manager.get("lora.rank", 8)
        self.alpha = config_manager.get("lora.alpha", 16)
        self.dropout = config_manager.get("lora.dropout", 0.1)
        self.target_modules = config_manager.get("lora.target_modules", ["q_proj", "v_proj"])
        self.task_type = config_manager.get("lora.task_type", "CAUSAL_LM")
        # Fallback config
        self.enable_adapters = config_manager.get("adapters.enable", True)
        self.enable_prefix_tuning = config_manager.get("prefix_tuning.enable", True)
        self._validate_config()

    def _validate_config(self):
        """
        Validate LoRA config values. Log warnings/errors for invalid settings.
        """
        try:
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
                self.task_type = "CAUSAL_LM"
        except Exception as e:
            import traceback
            self.logger.log_error(f"LoRA config validation failed: {e}\n{traceback.format_exc()}", error_type="lora_config_error")

    def apply_with_fallbacks(self, model: torch.nn.Module) -> (torch.nn.Module, str):
        """
        Try to apply LoRA, then Adapters, then Prefix Tuning. Returns (model, method_used).
        """
        # 1. Try LoRA
        if self.enabled and LoraConfig is not None and get_peft_model is not None:
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
                return model, "lora"
            except Exception as e:
                self.logger.log_warning(f"LoRA failed: {e}", event_type="lora_fallback")
        # 2. Try Adapters
        if self.enable_adapters:
            try:
                try:
                    from transformers.adapters import AdapterConfig
                except ImportError:
                    AdapterConfig = None
                if AdapterConfig is not None:
                    adapter_config = AdapterConfig()
                    if hasattr(model, 'add_adapter') and hasattr(model, 'set_active_adapters'):
                        model.add_adapter("default_adapter", config=adapter_config)
                        model.set_active_adapters(["default_adapter"])
                        self.logger.log_info("Adapter applied via adapter-transformers.", event_type="adapter_applied")
                        return model, "adapter"
                    else:
                        self.logger.log_warning("Model does not support adapters API.", event_type="adapter_fallback")
                else:
                    self.logger.log_warning("adapter-transformers not installed.", event_type="adapter_fallback")
            except Exception as e:
                self.logger.log_warning(f"Adapter fallback failed: {e}", event_type="adapter_fallback")
        # 3. Try Prefix Tuning
        if self.enable_prefix_tuning and PrefixTuningConfig is not None and get_peft_model is not None:
            try:
                prefix_config = PrefixTuningConfig(task_type=getattr(TaskType, self.task_type, TaskType.CAUSAL_LM))
                model = get_peft_model(model, prefix_config)
                self.logger.log_info("Prefix Tuning applied via PEFT.", event_type="prefix_tuning_applied")
                return model, "prefix_tuning"
            except Exception as e:
                self.logger.log_warning(f"Prefix Tuning fallback failed: {e}", event_type="prefix_tuning_fallback")
        # 4. Vanilla
        self.logger.log_warning("All adaptation methods failed, using vanilla model.", event_type="adaptation_fallback")
        return model, "vanilla"

    def save_lora_weights(self, model: torch.nn.Module, path: str):
        try:
            if PeftModel is not None and isinstance(model, PeftModel):
                model.save_pretrained(path)
                try:
                    self.logger.log_info(f"LoRA weights saved to {path}", event_type="lora_weights_save")
                except Exception:
                    pass
            else:
                raise ScaffoldError("Model is not a PEFT/LoRA model; cannot save LoRA weights.", operation="save_lora_weights")
        except Exception as e:
            import traceback
            try:
                self.error_handler.handle_error(e, operation="save_lora_weights", context={"path": path})
                self.logger.log_error(f"Failed to save LoRA weights: {e}\n{traceback.format_exc()}", error_type="lora_weights_save_error")
            except Exception:
                pass
            raise

    def load_lora_weights(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        try:
            if PeftModel is not None and hasattr(PeftModel, 'from_pretrained'):
                loaded_model = PeftModel.from_pretrained(model, path)
                try:
                    self.logger.log_info(f"LoRA weights loaded from {path}", event_type="lora_weights_load")
                except Exception:
                    pass
                return loaded_model
            else:
                raise ScaffoldError("PEFT/LoRA not available or model incompatible.", operation="load_lora_weights")
        except Exception as e:
            import traceback
            try:
                self.error_handler.handle_error(e, operation="load_lora_weights", context={"path": path})
                self.logger.log_error(f"Failed to load LoRA weights: {e}\n{traceback.format_exc()}", error_type="lora_weights_load_error")
            except Exception:
                pass
            raise

    @staticmethod
    def lora_parameters(model: torch.nn.Module):
        """Yield only LoRA parameters for optimizer."""
        try:
            for n, p in getattr(model, 'named_parameters', lambda: [])():
                if p.requires_grad and ("lora" in n or "adapter" in n):
                    yield p
        except Exception as e:
            import traceback
            print(f"[LoraAdapterManager] Failed to yield LoRA parameters: {e}\n{traceback.format_exc()}")
