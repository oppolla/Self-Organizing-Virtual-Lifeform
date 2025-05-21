from typing import Any, Dict, Optional, List, Tuple
import torch
from sovl_logger import Logger
from sovl_error import ErrorManager, ScaffoldError
from sovl_config import ConfigManager
from sovl_scaffold import CrossAttentionInjector


"""
LoRA Adapter and Adapter Management for SOVL System
"""

# Optional: Only import PEFT if available
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PrefixTuningConfig
    from transformers.adapters import AdapterConfig
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None
    PrefixTuningConfig = None
    AdapterConfig = None

class LoraAdapterManager:
    """
    Utility for applying, saving, and loading LoRA adapters to transformer models.
    Integrates system-standard logging, error handling, and configuration.
    Implements lazy/on-demand validation and async save/load for low-power systems.
    """
    def __init__(self, config_manager: ConfigManager, logger: Optional[Logger] = None, error_handler: Optional[ErrorManager] = None):
        self.config_manager = config_manager
        self.logger = logger or Logger()
        self.error_handler = error_handler or ErrorManager(self.logger)
        self._validated = False  # For lazy validation
        self._deps_checked = False
        self._dependency_reasons = None
        self._deps = None
        self._adapter_cache = {}  # Cache for adapter application
        self._last_applied_model_id = None
        self._last_applied_config = None
        # LoRA capacity parameters (exposed in config)
        self.rank = config_manager.get("engram_lora.lora_rank", 8)
        self.alpha = config_manager.get("engram_lora.lora_alpha", 16)
        self.dropout = config_manager.get("engram_lora.lora_dropout", 0.1)
        self.target_modules = config_manager.get("lora.target_modules", ["q_proj", "v_proj"])
        self.task_type = config_manager.get("lora.task_type", "CAUSAL_LM")

    def _lazy_validate(self):
        if not self._validated:
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
            self._validated = True

    @staticmethod
    def _check_adaptation_dependencies(logger=None, min_peft_version="0.4.0", check_adapters=True, check_prefix_tuning=True):
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
            if LoraConfig is None or get_peft_model is None or TaskType is None or PeftModel is None:
                raise ImportError("peft or required symbols not available")
            peft_version = pkg_resources.get_distribution("peft").version
            if pkg_resources.parse_version(peft_version) < pkg_resources.parse_version(min_peft_version):
                msg = f"PEFT version {peft_version} < {min_peft_version}"
                if logger: logger.log_error(msg, error_type="dependency_version_error")
                result["reasons"]["lora"] = msg
            else:
                result["lora_enabled"] = True
                result["reasons"]["lora"] = "OK"
        except Exception as e:
            msg = f"LoRA/PEFT unavailable: {e}"
            if logger: logger.log_error(msg, error_type="dependency_error")
            result["reasons"]["lora"] = str(e)
        # Check adapters
        if check_adapters:
            try:
                if AdapterConfig is None:
                    raise ImportError("transformers.adapters.AdapterConfig not available")
                result["adapters_enabled"] = True
                result["reasons"]["adapters"] = "OK"
            except Exception as e:
                msg = f"Adapters unavailable: {e}"
                if logger: logger.log_error(msg, error_type="dependency_error")
                result["reasons"]["adapters"] = str(e)
        # Check prefix tuning
        if check_prefix_tuning:
            try:
                if PrefixTuningConfig is None:
                    raise ImportError("peft.PrefixTuningConfig not available")
                result["prefix_tuning_enabled"] = True
                result["reasons"]["prefix_tuning"] = "OK"
            except Exception as e:
                msg = f"Prefix tuning unavailable: {e}"
                if logger: logger.log_error(msg, error_type="dependency_error")
                result["reasons"]["prefix_tuning"] = str(e)
        return result

    def _lazy_check_dependencies(self):
        if not self._deps_checked:
            deps = self._check_adaptation_dependencies(self.logger)
            self.enabled = deps["lora_enabled"]
            self.enable_adapters = deps["adapters_enabled"]
            self.enable_prefix_tuning = deps["prefix_tuning_enabled"]
            self._dependency_reasons = deps["reasons"]
            self._deps = deps
            self._deps_checked = True
            self.logger.log_info(f"Adaptation dependency check: {self._dependency_reasons}")

    def _get_model_id(self, model):
        # Use id() or a hash of model state_dict keys for cache
        return id(model)

    def apply_with_fallbacks(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
        self._lazy_validate()
        self._lazy_check_dependencies()
        model_id = self._get_model_id(model)
        config_tuple = (self.rank, self.alpha, self.dropout, tuple(self.target_modules), self.task_type)
        # Check cache
        if self._last_applied_model_id == model_id and self._last_applied_config == config_tuple:
            self.logger.log_info("Adapter already applied to this model with the same config. Skipping re-application.")
            return model, "cached"
        # Check compatibility before applying LoRA
        if not self.is_model_compatible(model):
            self.logger.log_warning(
                "Model is not compatible with LoRA. Falling back to vanilla.",
                event_type="lora_compatibility_fallback"
            )
            return model, "vanilla"
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
                self._last_applied_model_id = model_id
                self._last_applied_config = config_tuple
                return model, "lora"
            except Exception as e:
                self.logger.log_warning(f"LoRA failed: {e}", event_type="lora_fallback")
        # 2. Try Adapters
        if self.enable_adapters:
            try:
                if AdapterConfig is not None:
                    adapter_config = AdapterConfig()
                    if hasattr(model, 'add_adapter') and hasattr(model, 'set_active_adapters'):
                        model.add_adapter("default_adapter", config=adapter_config)
                        model.set_active_adapters(["default_adapter"])
                        self.logger.log_info("Adapter applied via adapter-transformers.", event_type="adapter_applied")
                        self._last_applied_model_id = model_id
                        self._last_applied_config = config_tuple
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
                self._last_applied_model_id = model_id
                self._last_applied_config = config_tuple
                return model, "prefix_tuning"
            except Exception as e:
                self.logger.log_warning(f"Prefix Tuning fallback failed: {e}", event_type="prefix_tuning_fallback")
        # 4. Vanilla
        self.logger.log_warning("All adaptation methods failed, using vanilla model.", event_type="adaptation_fallback")
        return model, "vanilla"

    def save_lora_weights(self, model: torch.nn.Module, path: str, async_save: bool = False):
        """Save LoRA weights, optionally in a background thread."""
        def _save():
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
        if async_save:
            import threading
            t = threading.Thread(target=_save, daemon=True)
            t.start()
            return t  # Return thread for optional join
        else:
            _save()

    def load_lora_weights(self, model: torch.nn.Module, path: str, async_load: bool = False, callback: Optional[Any] = None):
        """Load LoRA weights, optionally in a background thread. If async, callback(model) is called on completion."""
        def _load():
            try:
                if PeftModel is not None and hasattr(PeftModel, 'from_pretrained'):
                    loaded_model = PeftModel.from_pretrained(model, path)
                    try:
                        self.logger.log_info(f"LoRA weights loaded from {path}", event_type="lora_weights_load")
                    except Exception:
                        pass
                    if callback:
                        callback(loaded_model)
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
                if callback:
                    callback(None)
                raise
        if async_load:
            import threading
            t = threading.Thread(target=_load, daemon=True)
            t.start()
            return t  # Return thread for optional join
        else:
            return _load()

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

    def is_model_compatible(self, model: torch.nn.Module) -> bool:
        """
        Check if the model is compatible with LoRA (i.e., has the required target modules),
        including wrapped layers from CrossAttentionInjector.
        """
        if not hasattr(model, "named_modules"):
            return False
        for name, module in model.named_modules():
            module_base = name.split(".")[-1]
            if any(target in module_base for target in self.target_modules):
                return True
            # Check for wrapped layers from CrossAttentionInjector
            if hasattr(module, "forward") and "WrappedLayer" in str(type(module)):
                for subname, submodule in module.__dict__.items():
                    if isinstance(submodule, torch.nn.Module) and any(target in subname for target in self.target_modules):
                        return True
        self.logger.log_warning(
            "No compatible LoRA target modules found in model.",
            event_type="lora_compatibility_warning"
        )
        return False

    def is_checkpoint_compatible(self, checkpoint_path: str, model: torch.nn.Module) -> bool:
        """
        Check if a LoRA checkpoint is compatible with the given model.
        
        Args:
            checkpoint_path: Path to the LoRA checkpoint
            model: Model to check compatibility with
            
        Returns:
            bool: True if compatible, False otherwise
        """
        import os
        try:
            # First check if the model is LoRA compatible at all
            if not self.is_model_compatible(model):
                if hasattr(self, "logger") and self.logger:
                    self.logger.log_warning(
                        message=f"Model is not compatible with LoRA configuration",
                        event_type="lora_checkpoint_incompatible"
                    )
                return False
                
            # Check if checkpoint exists
            if not os.path.exists(checkpoint_path):
                if hasattr(self, "logger") and self.logger:
                    self.logger.log_warning(
                        message=f"LoRA checkpoint not found at {checkpoint_path}",
                        event_type="lora_checkpoint_not_found" 
                    )
                return False
                
            # Check if checkpoint has adapter_config.json
            config_path = os.path.join(checkpoint_path, "adapter_config.json")
            if not os.path.exists(config_path):
                if hasattr(self, "logger") and self.logger:
                    self.logger.log_warning(
                        message=f"LoRA adapter_config.json not found in checkpoint",
                        event_type="lora_checkpoint_invalid"
                    )
                return False
                
            # Load and check adapter config
            import json
            with open(config_path, 'r') as f:
                adapter_config = json.load(f)
                
            # Check if target modules match
            checkpoint_modules = set(adapter_config.get("target_modules", []))
            current_modules = set(self.target_modules)
            
            # Check basic compatibility factors
            model_type_matches = adapter_config.get("base_model_name_or_path", "").split("/")[-1] == model.__class__.__name__.lower()
            task_type_matches = adapter_config.get("task_type", "") == self.task_type
            
            # Check if checkpoint modules are a subset of available modules in the model
            available_modules = set()
            for name, _ in model.named_modules():
                module_base = name.split(".")[-1]
                available_modules.add(module_base)
                
            modules_available = all(module in available_modules for module in checkpoint_modules)
                
            if not modules_available:
                if hasattr(self, "logger") and self.logger:
                    self.logger.log_warning(
                        message=f"LoRA checkpoint targets modules not available in model",
                        event_type="lora_checkpoint_incompatible_modules"
                    )
                return False
                
            return True
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                import traceback
                self.logger.log_error(
                    error_msg=f"Failed to check LoRA checkpoint compatibility: {str(e)}",
                    error_type="lora_checkpoint_compatibility_error",
                    stack_trace=traceback.format_exc()
                )
            return False

class SOVLCompositeModel(torch.nn.Module):
    """
    Composite model: frozen base model + one or more scaffold models (with LoRA).
    """
    def __init__(
        self,
        base_model: torch.nn.Module,
        scaffold_models: List[torch.nn.Module],
        config_manager: Any,
        logger: Any,
        error_manager: Any,
        device: Optional[torch.device] = None,
        layers_to_inject: Optional[List[int]] = None,
        injection_strategy: str = 'sequential'
    ):
        super().__init__()
        self.base = base_model
        self.scaffolds = torch.nn.ModuleList(scaffold_models)
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.device = device or next(base_model.parameters()).device

        self._move_to_device(self.device)
        self._freeze_base()
        self._inject_scaffolds(layers_to_inject, injection_strategy)

    def _move_to_device(self, device):
        self.base.to(device)
        for scaffold in self.scaffolds:
            scaffold.to(device)
        self.logger.log_info(f"Moved base and scaffolds to device: {device}", event_type="device_placement")

    def _freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        self.logger.log_info("Base model parameters frozen.", event_type="freeze_base")

    def _inject_scaffolds(self, layers_to_inject, injection_strategy):
        injector = CrossAttentionInjector(self.config_manager, self.logger)
        # TODO: In future, inject all scaffolds (swarm); for now, just the first
        scaffold = self.scaffolds[0]
        if layers_to_inject is None:
            total_layers = injector._get_total_layers(self.base)
            layers_to_inject = [total_layers // 2]
        injector.inject(self.base, scaffold, layers_to_inject, injection_strategy)
        self.logger.log_info(f"Injected scaffold into base model at layers {layers_to_inject}", event_type="scaffold_injection")

    def forward(self, *args, scaffold_index: int = 0, **kwargs):
        """
        Forward pass through the composite model.
        For now, just forwards through the base model (scaffold is injected).
        In future, could select/aggregate scaffolds by index or strategy.
        """
        return self.base(*args, **kwargs)

    def trainable_parameters(self, as_dict: bool = False) -> Any:
        """
        Returns all trainable parameters from all scaffolds.
        If as_dict is True, returns a dict for advanced optimizer configs.
        """
        if as_dict:
            return {f"scaffold_{i}": [p for p in scaffold.parameters() if p.requires_grad]
                    for i, scaffold in enumerate(self.scaffolds)}
        params = []
        for scaffold in self.scaffolds:
            params.extend([p for p in scaffold.parameters() if p.requires_grad])
        return params

# Factory function to build the composite model

def build_sovl_composite_model(
    base_model: torch.nn.Module,
    scaffold_model: torch.nn.Module,
    config_manager: Any,
    logger: Any,
    error_manager: Any,
    device: Optional[torch.device] = None,
    layers_to_inject: Optional[List[int]] = None,
    injection_strategy: str = 'sequential'
) -> SOVLCompositeModel:
    """
    Build a SOVLCompositeModel from a base model and a scaffold model.
    For now, only a single scaffold is supported, but the design is swarm-ready.
    """
    return SOVLCompositeModel(base_model, [scaffold_model], config_manager, logger, error_manager, device, layers_to_inject, injection_strategy)
