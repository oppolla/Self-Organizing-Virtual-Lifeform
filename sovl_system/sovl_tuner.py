from typing import Optional, Dict, Any, List, Protocol
import time
from collections import defaultdict, deque
from dataclasses import dataclass
import traceback
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_utils import NumericalGuard
from sovl_curiosity import CuriosityManager
from sovl_trainer import SOVLTrainer
from sovl_scaffold import CrossAttentionInjector
from sovl_error import ErrorManager, ConfigurationError, ErrorRecord

@dataclass
class ValidationRange:
    """Defines a validation range for a configuration parameter."""
    min_value: Any
    max_value: Any
    description: str = ""

class ICuriosityManager(Protocol):
    """Interface for curiosity management."""
    def get_pressure(self) -> float: ...
    def reduce_pressure(self, amount: float) -> None: ...
    def tune(self, **kwargs) -> None: ...
    def get_pressure_stats(self) -> Dict[str, float]: ...

class ITrainer(Protocol):
    """Interface for model training."""
    def train_step(self, batch: Dict[str, Any]) -> float: ...
    def get_current_parameters(self) -> Dict[str, Any]: ...
    def update_parameters(self, params: Dict[str, Any]) -> None: ...

class ICrossAttentionInjector(Protocol):
    """Interface for cross attention injection."""
    def inject_cross_attention(self, model: Any, scaffold_model: Any, **kwargs) -> None: ...
    def set_influence(self, model: Any, **kwargs) -> None: ...
    def get_cross_attention_layers(self, model: Any, mode: str) -> List[Any]: ...

class SOVLTuner:
    """Centralized module for tuning SOVL system parameters dynamically."""
    
    # Configuration ranges grouped by category
    CONFIG_RANGES: Dict[str, Dict[str, ValidationRange]] = {
    "core_config": {
        "valid_split_ratio": ValidationRange(0.0, 1.0, "Validation split ratio"),
        "random_seed": ValidationRange(0, 2**32, "Random seed"),
        "quantization": ValidationRange("fp16", "int8", "Quantization mode"),
        "hidden_size": ValidationRange(1, None, "Model hidden size"),
        "num_heads": ValidationRange(1, None, "Number of attention heads"),
        "initializer_range": ValidationRange(0.0, None, "Initializer range"),
        "use_dynamic_layers": ValidationRange(True, False, "Dynamic layer usage"),
        "base_model_name": ValidationRange(None, None, "Base model name"),
        "base_model_path": ValidationRange(None, None, "Base model path"),
        "scaffold_model_name": ValidationRange(None, None, "Scaffold model name"),
        "scaffold_model_path": ValidationRange(None, None, "Scaffold model path"),
        "cross_attn_layers": ValidationRange([4, 6], None, "Cross attention layers"),
        "layer_selection_mode": ValidationRange("balanced", ["balanced", "random", "fixed"], "Layer selection mode"),
        "custom_layers": ValidationRange(None, None, "Custom layers"),
        "gradient_checkpointing": ValidationRange(True, False, "Gradient checkpointing"),
        "migration_mode": ValidationRange(True, False, "Migration mode"),
        "device": ValidationRange("cuda", ["cuda", "cpu"], "Device"),
    },
    "model": {
        "quantization_mode": ValidationRange("fp16", "int8", "Quantization mode"),
    },
    "controls_config": {
        "scaffold_weight_cap": ValidationRange(0.0, 1.0, "Scaffold weight cap"),
        "scaffold_unk_id": ValidationRange(0, None, "Scaffold unknown ID"),
        "blend_strength": ValidationRange(0.0, 1.0, "Blend strength"),
        "attention_weight": ValidationRange(0.0, 1.0, "Attention weight"),
        "enable_scaffold": ValidationRange(True, False, "Enable scaffold"),
        "enable_cross_attention": ValidationRange(True, False, "Enable cross attention"),
        "enable_dynamic_cross_attention": ValidationRange(True, False, "Enable dynamic cross attention"),
    },
    "temperament_config": {
        "mood_influence": ValidationRange(0.0, 1.0, "Mood influence"),
        "history_maxlen": ValidationRange(1, None, "History max length"),
        "temp_eager_threshold": ValidationRange(0.0, 1.0, "Eager threshold"),
        "temp_sluggish_threshold": ValidationRange(0.0, 1.0, "Sluggish threshold"),
        "temp_mood_influence": ValidationRange(0.0, 1.0, "Temperament mood influence"),
        "temp_curiosity_boost": ValidationRange(0.0, 1.0, "Curiosity boost"),
        "temp_restless_drop": ValidationRange(0.0, 1.0, "Restless drop"),
        "temp_melancholy_noise": ValidationRange(0.0, 0.1, "Melancholy noise"),
        "conf_feedback_strength": ValidationRange(0.0, 1.0, "Confidence feedback strength"),
        "temp_smoothing_factor": ValidationRange(0.0, 1.0, "Temperament smoothing factor"),
        "temperament_decay_rate": ValidationRange(0.0, 1.0, "Temperament decay rate"),
        "temperament_history_maxlen": ValidationRange(1, None, "Temperament history max length"),
        "confidence_history_maxlen": ValidationRange(1, None, "Confidence history max length"),
    },
    "training_config": {
        "learning_rate": ValidationRange(0.0, None, "Learning rate"),
        "train_epochs": ValidationRange(1, None, "Training epochs"),
        "batch_size": ValidationRange(1, None, "Batch size"),
        "max_seq_length": ValidationRange(1, None, "Maximum sequence length"),
        "sigmoid_scale": ValidationRange(0.0, None, "Sigmoid scale"),
        "sigmoid_shift": ValidationRange(0.0, None, "Sigmoid shift"),
        "lifecycle_capacity_factor": ValidationRange(0.0, None, "Lifecycle capacity factor"),
        "lifecycle_curve": ValidationRange("sigmoid_linear", "exponential", "Lifecycle curve type"),
        "grad_accum_steps": ValidationRange(1, None, "Gradient accumulation steps"),
        "exposure_gain_eager": ValidationRange(1, None, "Eager exposure gain"),
        "exposure_gain_default": ValidationRange(1, None, "Default exposure gain"),
        "max_patience": ValidationRange(0, None, "Maximum patience"),
        "weight_decay": ValidationRange(0.0, None, "Weight decay"),
        "max_grad_norm": ValidationRange(0.0, None, "Maximum gradient norm"),
        "checkpoint_interval": ValidationRange(1, None, "Checkpoint interval"),
        "cosine_min_lr": ValidationRange(0.0, None, "Cosine minimum learning rate"),
        "warmup_ratio": ValidationRange(0.0, 1.0, "Warmup ratio"),
        "warmup_steps": ValidationRange(0, None, "Warmup steps"),
        "total_steps": ValidationRange(1, None, "Total steps"),
        "validate_every_n_steps": ValidationRange(1, None, "Validate every n steps"),
        "dropout_rate": ValidationRange(0.0, 1.0, "Dropout rate"),
        "max_epochs": ValidationRange(1, None, "Maximum epochs"),
        "sleep_conf_threshold": ValidationRange(0.0, 1.0, "Sleep confidence threshold"),
        "sleep_log_min": ValidationRange(1, None, "Minimum sleep log entries"),
        "dream_memory_weight": ValidationRange(0.0, 1.0, "Dream memory weight"),
        "dream_noise_scale": ValidationRange(0.0, None, "Dream noise scale"),
        "dream_prompt_weight": ValidationRange(0.0, 1.0, "Dream prompt weight"),
        "dream_novelty_boost": ValidationRange(0.0, None, "Dream novelty boost"),
        "dream_memory_decay": ValidationRange(0.0, 1.0, "Dream memory decay"),
        "dream_prune_threshold": ValidationRange(0.0, 1.0, "Dream prune threshold"),
        "dream_swing_var": ValidationRange(0.0, None, "Dream swing variance"),
        "dream_lifecycle_delta": ValidationRange(0.0, None, "Dream lifecycle delta"),
        "dream_temperament_on": ValidationRange(True, False, "Dream temperament enabled"),
        "confidence_history_maxlen": ValidationRange(1, None, "Confidence history max length"),
        "temperament_history_maxlen": ValidationRange(1, None, "Temperament history max length"),
        "memory_threshold": ValidationRange(0.0, 1.0, "Memory threshold"),
        "memory_decay_rate": ValidationRange(0.0, 1.0, "Memory decay rate"),
        "scaffold_weight": ValidationRange(0.0, None, "Scaffold weight"),
        "max_dream_memory_mb": ValidationRange(1, None, "Maximum dream memory in MB"),
        "dream_memory_maxlen": ValidationRange(1, None, "Dream memory max length"),
        "repetition_n": ValidationRange(2, None, "Repetition count"),
        "enable_gestation": ValidationRange(True, False, "Enable gestation"),
        "enable_sleep_training": ValidationRange(True, False, "Enable sleep training"),
        "enable_lifecycle_weighting": ValidationRange(True, False, "Enable lifecycle weighting"),
        "enable_dreaming": ValidationRange(True, False, "Enable dreaming"),
        "enable_prompt_driven_dreams": ValidationRange(True, False, "Enable prompt-driven dreams"),
        "use_scaffold_memory": ValidationRange(True, False, "Use scaffold memory"),
        "use_token_map_memory": ValidationRange(True, False, "Use token map memory"),
        "model_name": ValidationRange(None, None, "Model name"),
        "dry_run": ValidationRange(True, False, "Dry run"),
        "dry_run_params": ValidationRange(None, None, "Dry run parameters"),
        "use_amp": ValidationRange(True, False, "Use AMP"),
        "scheduler_type": ValidationRange("linear", ["linear", "cosine", "constant"], "Scheduler type"),
        "checkpoint_path": ValidationRange(None, None, "Checkpoint path"),
        "metrics_to_track": ValidationRange(None, None, "Metrics to track"),
    },
    "scaffold_config": {
        "quantization_mode": ValidationRange("fp16", "int8", "Quantization mode"),
    },
    "dynamic_weighting": {
        "min_weight": ValidationRange(0.0, None, "Minimum weight"),
        "max_weight": ValidationRange(0.0, None, "Maximum weight"),
        "weight_decay": ValidationRange(0.0, None, "Weight decay"),
        "momentum": ValidationRange(0.0, 1.0, "Momentum"),
        "history_size": ValidationRange(1, None, "History size"),
        "enable_dynamic_scaling": ValidationRange(True, False, "Enable dynamic scaling"),
    },
    "preprocessing": {
        "remove_special_chars": ValidationRange(True, False, "Remove special characters"),
        "lowercase": ValidationRange(True, False, "Convert to lowercase"),
        "remove_extra_spaces": ValidationRange(True, False, "Remove extra spaces"),
        "max_length": ValidationRange(1, None, "Maximum length"),
    },
    "augmentation": {
        "synonym_replacement_prob": ValidationRange(0.0, 1.0, "Synonym replacement probability"),
        "word_dropout_prob": ValidationRange(0.0, 1.0, "Word dropout probability"),
        "max_augmentations": ValidationRange(1, None, "Maximum augmentations"),
    },
    "hardware": {
        "enable_cuda": ValidationRange(True, False, "Enable CUDA"),
        "memory_query_interval": ValidationRange(0.0, None, "Memory query interval"),
        "mock_memory_total_mb": ValidationRange(1.0, None, "Mock memory total in MB"),
        "max_scaffold_memory_mb": ValidationRange(1, None, "Maximum scaffold memory in MB"),
    },
    "lora_config": {
        "lora_rank": ValidationRange(1, None, "LoRA rank"),
        "lora_alpha": ValidationRange(1, None, "LoRA alpha"),
        "lora_dropout": ValidationRange(0.0, 1.0, "LoRA dropout"),
    },
    "curiosity_config": {
        "enable_curiosity": ValidationRange(True, False, "Curiosity enabled"),
        "attention_weight": ValidationRange(0.0, 1.0, "Attention weight"),
        "queue_maxlen": ValidationRange(1, None, "Question queue max length"),
        "novelty_history_maxlen": ValidationRange(1, None, "Novelty history max length"),
        "decay_rate": ValidationRange(0.0, 1.0, "Decay rate"),
        "question_timeout": ValidationRange(0.0, None, "Question timeout"),
        "novelty_threshold_spontaneous": ValidationRange(0.0, 1.0, "Spontaneous novelty threshold"),
        "novelty_threshold_response": ValidationRange(0.0, 1.0, "Response novelty threshold"),
        "pressure_threshold": ValidationRange(0.0, 1.0, "Pressure threshold"),
        "pressure_drop": ValidationRange(0.0, 1.0, "Pressure drop"),
        "silence_threshold": ValidationRange(0.0, None, "Silence threshold"),
        "question_cooldown": ValidationRange(0.0, None, "Question cooldown"),
        "weight_ignorance": ValidationRange(0.0, 1.0, "Ignorance weight"),
        "weight_novelty": ValidationRange(0.0, 1.0, "Novelty weight"),
        "max_new_tokens": ValidationRange(1, None, "Maximum new tokens"),
        "base_temperature": ValidationRange(0.0, None, "Base temperature"),
        "temperament_influence": ValidationRange(0.0, 1.0, "Temperament influence"),
        "top_k": ValidationRange(1, None, "Top-k sampling"),
        "max_memory_mb": ValidationRange(1.0, None, "Maximum memory in MB"),
        "pressure_change_cooldown": ValidationRange(0.0, None, "Pressure change cooldown"),
        "min_pressure": ValidationRange(0.0, 1.0, "Minimum pressure"),
        "max_pressure": ValidationRange(0.0, 1.0, "Maximum pressure"),
        "pressure_decay_rate": ValidationRange(0.0, 1.0, "Pressure decay rate"),
        "metrics_maxlen": ValidationRange(1, None, "Metrics max length"),
    },
    "cross_attn_config": {
        "memory_weight": ValidationRange(0.0, 1.0, "Memory weight"),
    },
    "logging_config": {
        "max_log_size_mb": ValidationRange(1, None, "Maximum log size in MB"),
        "backup_count": ValidationRange(0, None, "Backup count"),
        "log_dir": ValidationRange(None, None, "Log directory"),
        "log_file": ValidationRange(None, None, "Log file"),
        "log_level": ValidationRange("INFO", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "Log level"),
    },
    "error_config": {
        "error_cooldown": ValidationRange(0.0, None, "Error cooldown"),
        "warning_threshold": ValidationRange(0.0, None, "Warning threshold"),
        "error_threshold": ValidationRange(0.0, None, "Error threshold"),
        "critical_threshold": ValidationRange(0.0, None, "Critical threshold"),
    },
    "generation_config": {
        "temperature": ValidationRange(0.0, None, "Temperature"),
        "top_p": ValidationRange(0.0, 1.0, "Top-p sampling"),
    },
    "data_config": {
        "batch_size": ValidationRange(1, None, "Batch size"),
        "max_retries": ValidationRange(0, None, "Maximum retries"),
    },
    "memory_config": {
        "memoria": {
            "max_memory_mb": ValidationRange(1, None, "Maximum memory in MB"),
            "garbage_collection_threshold": ValidationRange(0.0, 1.0, "Garbage collection threshold"),
            "memory_decay_rate": ValidationRange(0.0, 1.0, "Memory decay rate"),
            "enable_memory_compression": ValidationRange(True, False, "Enable memory compression"),
            "compression_ratio": ValidationRange(0.0, 1.0, "Compression ratio"),
            "max_compressed_memory_mb": ValidationRange(1, None, "Maximum compressed memory in MB"),
        },
        "ram": {
            "max_ram_mb": ValidationRange(1, None, "Maximum RAM in MB"),
            "ram_threshold": ValidationRange(0.0, 1.0, "RAM threshold"),
            "enable_ram_compression": ValidationRange(True, False, "Enable RAM compression"),
            "ram_compression_ratio": ValidationRange(0.0, 1.0, "RAM compression ratio"),
            "max_compressed_ram_mb": ValidationRange(1, None, "Maximum compressed RAM in MB"),
        },
        "gpu": {
            "max_gpu_memory_mb": ValidationRange(1, None, "Maximum GPU memory in MB"),
            "gpu_memory_threshold": ValidationRange(0.0, 1.0, "GPU memory threshold"),
            "enable_gpu_memory_compression": ValidationRange(True, False, "Enable GPU memory compression"),
            "gpu_compression_ratio": ValidationRange(0.0, 1.0, "GPU compression ratio"),
            "max_compressed_gpu_memory_mb": ValidationRange(1, None, "Maximum compressed GPU memory in MB"),
        },
        "manager": {
            "enable_memoria_manager": ValidationRange(True, False, "Enable memoria manager"),
            "enable_ram_manager": ValidationRange(True, False, "Enable RAM manager"),
            "enable_gpu_memory_manager": ValidationRange(True, False, "Enable GPU memory manager"),
            "memory_sync_interval": ValidationRange(1, None, "Memory sync interval"),
            "enable_memory_monitoring": ValidationRange(True, False, "Enable memory monitoring"),
            "memory_monitoring_interval": ValidationRange(1, None, "Memory monitoring interval"),
        }
    },
    "state_config": {
        "max_history": ValidationRange(1, None, "Maximum history"),
        "state_file": ValidationRange(None, None, "State file"),
    },
    "confidence_config": {
        "history_maxlen": ValidationRange(1, None, "History max length"),
        "weight": ValidationRange(0.0, 1.0, "Confidence weight"),
    },
    "dream_memory_config": {
        "max_memories": ValidationRange(1, None, "Maximum memories"),
        "base_weight": ValidationRange(0.0, 1.0, "Base weight"),
        "max_weight": ValidationRange(0.0, None, "Maximum weight"),
    },
    "lifecycle_config": {
        "fatigue_factor_cap": ValidationRange(0.0, 0.5, "Fatigue factor cap"),
        "fatigue_scaling_factor": ValidationRange(1.0, 5.0, "Fatigue scaling factor"),
    },
    "data_provider": {
        "provider_type": ValidationRange(None, None, "Provider type"),
        "data_path": ValidationRange(None, None, "Data path"),
    },
    "gestation_weighting": {
        "quality_metrics": {
            "code": ValidationRange(0.1, 3.0, "Code weight"),
            "url": ValidationRange(0.1, 3.0, "URL weight"),
            "question": ValidationRange(0.1, 3.0, "Question weight"),
            "exclamation": ValidationRange(0.1, 3.0, "Exclamation weight"),
            "emoji": ValidationRange(0.1, 3.0, "Emoji weight"),
        },
        "content_weights": {
            "word_count_ratio_scale": ValidationRange(0.1, 3.0, "Word count ratio scale"),
            "optimal_word_length_range": {
                "min": ValidationRange(0, None, "Minimum word length"),
                "max": ValidationRange(0, None, "Maximum word length"),
                "weight": ValidationRange(0.1, 3.0, "Word length weight"),
            }
        }
    },
}

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorManager,
        curiosity_manager: Optional[ICuriosityManager] = None,
        trainer: Optional[ITrainer] = None,
        cross_attention_injector: Optional[ICrossAttentionInjector] = None
    ):
        """Initialize the SOVL tuner with configuration and dependencies."""
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.curiosity_manager = curiosity_manager
        self.trainer = trainer
        self.cross_attention_injector = cross_attention_injector
        self._guard = NumericalGuard(logger)

        # Cache configuration sections
        self._config_sections = {
            "core_config": config_manager.get_section("core_config"),
            "controls_config": config_manager.get_section("controls_config"),
            "training_config": config_manager.get_section("training_config"),
            "curiosity_config": config_manager.get_section("curiosity_config"),
            "cross_attn_config": config_manager.get_section("cross_attn_config"),
            "lora_config": config_manager.get_section("lora_config"),
        }

        # Confidence monitoring
        self._confidence_history = deque(maxlen=100)
        self._last_confidence_check = 0.0
        self._confidence_check_interval = 60.0  # seconds

        # Register tuner-specific error handlers
        self._register_error_handlers()

        # Validate configuration on initialization
        self._validate_initial_config()

    def _register_error_handlers(self) -> None:
        """Register tuner-specific error handlers with the error manager."""
        self.error_manager.register_handler("tuner_error", self._handle_tuner_error)
        self.error_manager.register_handler("config_validation_error", self._handle_config_validation_error)
        self.error_manager.register_handler("parameter_validation_error", self._handle_parameter_validation_error)

    def _handle_tuner_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle tuner-specific errors."""
        if "confidence" in context:
            self._monitor_confidence(context["confidence"])

    def _handle_config_validation_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle configuration validation errors."""
        if isinstance(error, ConfigurationError):
            self.logger.log_training_event(
                event_type="config_validation_error",
                message=str(error),
                level="error",
                additional_info=context
            )

    def _handle_parameter_validation_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle parameter validation errors."""
        self.logger.log_training_event(
            event_type="param_validation_error",
            message=str(error),
            level="error",
            additional_info=context
        )

    def _validate_initial_config(self) -> None:
        """Validate all configuration sections during initialization."""
        try:
            required_sections = [
                "core_config", "controls_config", "training_config",
                "curiosity_config", "cross_attn_config", "lora_config"
            ]
            for section in required_sections:
                if not self.config_manager.has_section(section):
                    raise ConfigurationError(f"Missing required configuration section: {section}")

            # Validation rules for each section
            validation_rules = {
                "core_config": {
                    "hidden_size": (lambda x: x > 0, "must be positive"),
                    "quantization": (lambda x: x in ["fp16", "fp32", "int8"], "must be one of: fp16, fp32, int8"),
                    "use_dynamic_layers": (lambda x: isinstance(x, bool), "must be boolean"),
                },
                "controls_config": {
                    "temp_eager_threshold": (lambda x: 0.7 <= x <= 0.9, "must be in [0.7, 0.9]"),
                    "temp_sluggish_threshold": (lambda x: 0.4 <= x <= 0.6, "must be in [0.4, 0.6]"),
                    "temp_mood_influence": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "temp_curiosity_boost": (lambda x: 0.0 <= x <= 0.5, "must be in [0, 0.5]"),
                    "temp_restless_drop": (lambda x: 0.0 <= x <= 0.5, "must be in [0, 0.5]"),
                    "temp_melancholy_noise": (lambda x: 0.0 <= x <= 0.05, "must be in [0, 0.05]"),
                    "conf_feedback_strength": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "temp_smoothing_factor": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "temperament_decay_rate": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "scaffold_weight_cap": (lambda x: 0.5 <= x <= 1.0, "must be in [0.5, 1.0]"),
                    "base_temperature": (lambda x: 0.5 <= x <= 1.5, "must be in [0.5, 1.5]"),
                    "sleep_conf_threshold": (lambda x: 0.5 <= x <= 0.9, "must be in [0.5, 0.9]"),
                    "sleep_time_factor": (lambda x: 0.5 <= x <= 5.0, "must be in [0.5, 5.0]"),
                    "sleep_log_min": (lambda x: 5 <= x <= 20, "must be in [5, 20]"),
                    "dream_swing_var": (lambda x: 0.05 <= x <= 0.2, "must be in [0.05, 0.2]"),
                    "dream_lifecycle_delta": (lambda x: 0.05 <= x <= 0.2, "must be in [0.05, 0.2]"),
                    "dream_temperament_on": (lambda x: isinstance(x, bool), "must be boolean"),
                    "dream_noise_scale": (lambda x: 0.01 <= x <= 0.1, "must be in [0.01, 0.1]"),
                    "dream_memory_weight": (lambda x: 0.0 <= x <= 0.5, "must be in [0, 0.5]"),
                    "dream_memory_maxlen": (lambda x: 5 <= x <= 20, "must be in [5, 20]"),
                    "dream_prompt_weight": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "dream_novelty_boost": (lambda x: 0.0 <= x <= 0.05, "must be in [0, 0.05]"),
                    "dream_memory_decay": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "dream_prune_threshold": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                },
                "curiosity_config": {
                    "enable_curiosity": (lambda x: isinstance(x, bool), "must be boolean"),
                    "novelty_threshold_spontaneous": (lambda x: 0.5 <= x <= 1.0, "must be in [0.5, 1.0]"),
                    "novelty_threshold_response": (lambda x: 0.5 <= x <= 1.0, "must be in [0.5, 1.0]"),
                    "pressure_threshold": (lambda x: 0.5 <= x <= 0.9, "must be in [0.5, 0.9]"),
                    "pressure_drop": (lambda x: 0.1 <= x <= 0.5, "must be in [0.1, 0.5]"),
                    "silence_threshold": (lambda x: 5.0 <= x <= 60.0, "must be in [5.0, 60.0]"),
                    "question_cooldown": (lambda x: 30.0 <= x <= 120.0, "must be in [30.0, 120.0]"),
                    "queue_maxlen": (lambda x: 1 <= x <= 50, "must be in [1, 50]"),
                    "weight_ignorance": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "weight_novelty": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "max_new_tokens": (lambda x: 5 <= x <= 12, "must be in [5, 12]"),
                    "base_temperature": (lambda x: 0.5 <= x <= 1.5, "must be in [0.5, 1.5]"),
                    "temperament_influence": (lambda x: 0.1 <= x <= 0.6, "must be in [0.1, 0.6]"),
                    "top_k": (lambda x: 10 <= x <= 50, "must be in [10, 50]"),
                    "attention_weight": (lambda x: 0.0 <= x <= 1.0, "must be in [0, 1]"),
                    "question_timeout": (lambda x: 60.0 <= x <= 86400.0, "must be in [60.0, 86400.0]"),
                },
                "training_config": {
                    "lifecycle_capacity_factor": (lambda x: 0.001 <= x <= 0.1, "must be in [0.001, 0.1]"),
                    "lifecycle_curve": (lambda x: x in ["sigmoid_linear", "exponential"], "must be one of: sigmoid_linear, exponential"),
                    "lora_capacity": (lambda x: 0 <= x <= 1000, "must be in [0, 1000]"),
                },
            }

            for section, rules in validation_rules.items():
                self.config_manager.validate_section(section, rules)

        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="config_validation_error",
                severity=2,
                additional_info={"operation": "initial_config_validation"}
            )
            raise

    def _monitor_confidence(self, confidence: float) -> None:
        """Monitor confidence scores and adjust parameters if needed."""
        try:
            current_time = time.time()
            if current_time - self._last_confidence_check < self._confidence_check_interval:
                return

            self._last_confidence_check = current_time
            self._confidence_history.append(confidence)

            if len(self._confidence_history) < 10:
                return

            confidences = list(self._confidence_history)
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)

            if variance > 0.1:
                self.logger.log_training_event(
                    event_type="confidence_variance_high",
                    message="High variance detected in confidence scores",
                    level="warning",
                    additional_info={"variance": variance, "mean_confidence": mean_conf, "history_length": len(confidences)}
                )
                current_influence = self._config_sections["curiosity_config"].get("temperament_influence", 0.3)
                new_influence = min(current_influence + 0.1, 0.6)
                if new_influence != current_influence:
                    self.tune_curiosity(temperament_influence=new_influence)
                    self.logger.log_training_event(
                        event_type="temperament_influence_adjusted",
                        message="Adjusted temperament influence to stabilize confidence",
                        level="info",
                        additional_info={"old_influence": current_influence, "new_influence": new_influence, "variance": variance}
                    )
            elif mean_conf < 0.3 and self.curiosity_manager:
                self.logger.log_training_event(
                    event_type="confidence_low",
                    message="Consistently low confidence detected",
                    level="warning",
                    additional_info={"mean_confidence": mean_conf, "history_length": len(confidences)}
                )
                current_pressure = self.curiosity_manager.get_pressure()
                if current_pressure > 0.3:
                    self.curiosity_manager.reduce_pressure(0.1)
                    self.logger.log_training_event(
                        event_type="curiosity_pressure_reduced",
                        message="Reduced curiosity pressure to improve confidence",
                        level="info",
                        additional_info={"old_pressure": current_pressure, "new_pressure": self.curiosity_manager.get_pressure(), "mean_confidence": mean_conf}
                    )
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="confidence_monitoring_error",
                severity=1,
                additional_info={"confidence": confidence}
            )

    def tune_parameters(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Tune system parameters based on performance metrics."""
        try:
            current_params = self.trainer.get_current_parameters() if self.trainer else {}
            if "confidence" in metrics:
                self._monitor_confidence(metrics["confidence"])

            tuned_params = self._adjust_parameters(current_params, metrics)
            if self.trainer:
                for param_name, value in tuned_params.items():
                    if not self.validate_param(param_name, value):
                        raise ConfigurationError(f"Invalid parameter value: {param_name}={value}")
                self.trainer.update_parameters(tuned_params)

            self.logger.log_training_event(
                event_type="parameters_tuned",
                message="System parameters tuned successfully",
                additional_info={"previous_params": current_params, "new_params": tuned_params, "metrics": metrics}
            )
            return tuned_params
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="parameter_tuning_error",
                severity=2,
                additional_info={"metrics": metrics, "batch_size": metrics.get("batch_size", 1)}
            )
            raise

    def _adjust_parameters(self, current_params: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters based on metrics."""
        try:
            tuned_params = current_params.copy()
            if "loss" in metrics:
                loss = metrics["loss"]
                loss_threshold = self.config_manager.get("loss_threshold", 2.0)
                loss_target = self.config_manager.get("loss_target", 0.5)
                if loss > loss_threshold:
                    tuned_params["learning_rate"] *= 0.5
                elif loss < loss_target:
                    tuned_params["learning_rate"] *= 1.1

            if self.curiosity_manager and "performance" in metrics:
                performance = metrics["performance"]
                perf_threshold = self.config_manager.get("performance_threshold", 0.7)
                if performance < perf_threshold:
                    pressure_stats = self.curiosity_manager.get_pressure_stats()
                    performance_gap = perf_threshold - performance
                    reduction_amount = min(0.1, performance_gap * 0.2)
                    if pressure_stats["current_pressure"] > pressure_stats["min_pressure"]:
                        self.curiosity_manager.reduce_pressure(reduction_amount)
                        self.logger.log_training_event(
                            event_type="pressure_adjusted",
                            message="Adjusted curiosity pressure based on performance",
                            level="info",
                            additional_info={
                                "performance": performance,
                                "performance_threshold": perf_threshold,
                                "reduction_amount": reduction_amount,
                                "pressure_stats": pressure_stats
                            }
                        )
            return tuned_params
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="parameter_adjustment_error",
                severity=2,
                additional_info={"current_params": current_params, "metrics": metrics}
            )
            raise

    def validate_param(self, param_name: str, value: Any) -> bool:
        """Validate a parameter against its allowed range or type."""
        try:
            section, key = param_name.split(".", 1)
            validation_rules = self.config_manager.get_validation_rules(section, key)
            if not validation_rules:
                self.logger.log_training_event(
                    event_type="param_validation_warning",
                    message=f"No validation rules found for {param_name}",
                    level="warning",
                    additional_info={"param_name": param_name, "value": value}
                )
                return True

            validator = validation_rules.get("validator")
            if validator and not validator(value):
                self.error_manager.handle_error(
                    error=ConfigurationError(f"Parameter validation failed for {key}"),
                    error_type="parameter_validation_error",
                    severity=1,
                    additional_info={"param_name": param_name, "value": value, "validation_rules": validation_rules}
                )
                return False
            return True
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="parameter_validation_error",
                severity=2,
                additional_info={"param_name": param_name, "value": value}
            )
            return False

    def _update_config_batch(
        self, updates: Dict[str, Any], section: str, notify_component: Optional[callable] = None
    ) -> bool:
        """Helper method to update configuration batch and notify components."""
        if not updates:
            return True

        try:
            success = self.config_manager.update_batch(updates, rollback_on_failure=True)
            if success:
                for key, value in updates.items():
                    param = key.split(".")[-1]
                    self._config_sections[section][param] = value
                self.config_manager.save_config()
                if notify_component:
                    notify_component(updates)
                self.logger.log_training_event(
                    event_type=f"{section}_updated",
                    message=f"{section.capitalize()} parameters updated successfully",
                    additional_info={"updated_params": list(updates.keys())}
                )
            else:
                self.error_manager.handle_error(
                    error=ConfigurationError(f"Failed to update {section} parameters"),
                    error_type="config_update_error",
                    severity=1,
                    additional_info={"section": section, "attempted_params": list(updates.keys())}
                )
            return success
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="config_update_error",
                severity=2,
                additional_info={"section": section, "updates": updates}
            )
            return False

    def tune_curiosity(self, **kwargs: Any) -> bool:
        """Tune curiosity-related parameters with transactional safety."""
        updates = {}
        prefix = "curiosity_config."
        for param, value in kwargs.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if not self.validate_param(full_key, value):
                    self.logger.log_training_event(
                        event_type="curiosity_tuning_validation_failed",
                        message=f"Validation failed for parameter: {param}",
                        level="error",
                        additional_info={"param": param, "value": value}
                    )
                    return False
                updates[full_key] = value

        def notify_curiosity(updates: Dict[str, Any]) -> None:
            if self.curiosity_manager:
                try:
                    manager_params = {k.split(".")[-1]: v for k, v in updates.items()}
                    self.curiosity_manager.tune(**manager_params)
                except Exception as e:
                    self.logger.log_training_event(
                        event_type="curiosity_manager_tune_failed",
                        message=f"Failed to update CuriosityManager: {str(e)}",
                        level="error",
                        additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
                    )

        return self._update_config_batch(updates, "curiosity_config", notify_curiosity)

    def adjust_temperament(self, **kwargs: Any) -> bool:
        """Tune temperament-related parameters with safe ranges and validation."""
        safe_ranges = {
            "temp_smoothing_factor": (0.1, 1.0),
            "temp_eager_threshold": (0.5, 0.9),
            "temp_sluggish_threshold": (0.1, 0.5),
            "temp_mood_influence": (0.1, 0.9),
            "temp_curiosity_boost": (0.1, 0.5),
            "temp_restless_drop": (0.1, 0.5),
            "temp_melancholy_noise": (0.0, 0.2),
            "conf_feedback_strength": (0.1, 0.9),
            "temperament_decay_rate": (0.1, 0.9),
        }
        updates = {}
        prefix = "controls_config."

        for key, value in kwargs.items():
            if value is not None:
                min_val, max_val = safe_ranges[key]
                if not (min_val <= value <= max_val):
                    self.logger.log_training_event(
                        event_type="temperament_parameter_warning",
                        message=f"Parameter {key} out of safe range, clamping to bounds",
                        level="warning",
                        additional_info={"parameter": key, "value": value, "min": min_val, "max": max_val}
                    )
                    value = max(min_val, min(value, max_val))
                updates[f"{prefix}{key}"] = value

        def notify_trainer(updates: Dict[str, Any]) -> None:
            if self.trainer and hasattr(self.trainer, 'state'):
                with self.trainer.state.lock:
                    self.trainer.state.temperament_history.clear()
                    self.trainer.state.temperament_score = 0.5
                    self.logger.log_training_event(
                        event_type="temperament_history_reset",
                        message="Temperament history reset after parameter update",
                        level="info",
                        additional_info={"reason": "temperament parameter update", "updated_params": list(updates.keys())}
                    )

        return self._update_config_batch(updates, "controls_config", notify_trainer)

    def tune_dream(self, **kwargs: Any) -> bool:
        """Tune dreaming-related parameters."""
        updates = {}
        prefix = "controls_config."
        for param, value in kwargs.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if not self.validate_param(full_key, value):
                    return False
                updates[full_key] = value

        def notify_trainer(updates: Dict[str, Any]) -> None:
            if self.trainer and "controls_config.dream_memory_weight" in updates:
                self.trainer.config.dream_memory_weight = updates["controls_config.dream_memory_weight"]

        return self._update_config_batch(updates, "controls_config", notify_trainer)

    def set_sleep_params(self, **kwargs: Any) -> bool:
        """Tune sleep-related parameters."""
        updates = {}
        for param, value in kwargs.items():
            if value is not None:
                if not self.validate_param(param, value):
                    return False
                updates[param] = value

        def notify_trainer(updates: Dict[str, Any]) -> None:
            if self.trainer:
                if "controls_config.sleep_conf_threshold" in updates:
                    self.trainer.config.sleep_conf_threshold = updates["controls_config.sleep_conf_threshold"]
                if "controls_config.sleep_log_min" in updates:
                    self.trainer.config.sleep_log_min = updates["controls_config.sleep_log_min"]
                if "training_config.sleep_max_steps" in updates:
                    self.trainer.config.sleep_max_steps = updates["training_config.sleep_max_steps"]

        return self._update_config_batch(updates, "controls_config", notify_trainer)

    def set_global_blend(self, **kwargs: Any) -> bool:
        """Tune global blend parameters."""
        updates = {}
        prefix = "controls_config."
        for param, value in kwargs.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if not self.validate_param(full_key, value):
                    return False
                updates[full_key] = value

        return self._update_config_batch(updates, "controls_config")

    def tune_lifecycle(self, **kwargs: Any) -> bool:
        """Tune lifecycle-related parameters."""
        updates = {}
        prefix = "training_config."
        for param, value in kwargs.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if not self.validate_param(full_key, value):
                    return False
                updates[full_key] = value

        def notify_trainer(updates: Dict[str, Any]) -> None:
            if self.trainer:
                if "training_config.lifecycle_capacity_factor" in updates:
                    self.trainer.config.lifecycle_capacity_factor = updates["training_config.lifecycle_capacity_factor"]
                if "training_config.lifecycle_curve" in updates:
                    self.trainer.config.lifecycle_curve = updates["training_config.lifecycle_curve"]
                if "training_config.lora_capacity" in updates:
                    self.trainer.lora_capacity = updates["training_config.lora_capacity"]

        return self._update_config_batch(updates, "training_config", notify_trainer)

    def toggle_dynamic_layers(self, enable: bool) -> bool:
        """Toggle dynamic layer usage."""
        full_key = "core_config.use_dynamic_layers"
        if not self.validate_param(full_key, enable):
            return False
        success = self.config_manager.update(full_key, enable)
        if success:
            self._config_sections["core_config"]["use_dynamic_layers"] = enable
            self.config_manager.save_config()
            self.logger.log_training_event(
                event_type="toggle_dynamic_layers",
                message="Dynamic layers toggled",
                additional_info={"enable": enable, "success": success}
            )
        return success

    def set_quantization_mode(self, mode: str) -> bool:
        """Set quantization mode."""
        full_key = "core_config.quantization"
        if not self.validate_param(full_key, mode):
            return False
        success = self.config_manager.update(full_key, mode)
        if success:
            self._config_sections["core_config"]["quantization"] = mode
            self.config_manager.save_config()
            self.logger.log_training_event(
                event_type="set_quantization_mode",
                message="Quantization mode set",
                additional_info={"mode": mode, "success": success}
            )
        return success

    def set_scaffold_influence(
        self,
        weight: Optional[float] = None,
        blend_strength: Optional[float] = None,
        layer_weights: Optional[List[float]] = None,
        base_model: Optional[Any] = None
    ) -> bool:
        """Set scaffold influence for cross-attention layers."""
        if not self.cross_attention_injector or not base_model:
            self.logger.log_training_event(
                event_type="scaffold_influence_error",
                message="CrossAttentionInjector or base_model not provided",
                level="error",
                additional_info={"timestamp": time.time()}
            )
            return False

        if layer_weights is not None:
            try:
                layers = self.cross_attention_injector.get_cross_attention_layers(
                    base_model, mode=self._config_sections["core_config"].get("layer_selection_mode", "balanced")
                )
                if len(layer_weights) != len(layers):
                    self.logger.log_training_event(
                        event_type="scaffold_influence_error",
                        message="Layer weights length mismatch",
                        level="error",
                        additional_info={
                            "expected_layers": len(layers),
                            "provided_weights": len(layer_weights),
                            "layers": layers,
                            "weights": layer_weights
                        }
                    )
                    return False
                for i, weight in enumerate(layer_weights):
                    if not (0.0 <= weight <= 1.0):
                        self.logger.log_training_event(
                            event_type="scaffold_influence_error",
                            message="Invalid layer weight value",
                            level="error",
                            additional_info={"layer_index": i, "weight": weight, "valid_range": (0.0, 1.0)}
                        )
                        return False
                self.logger.log_training_event(
                    event_type="scaffold_influence_validated",
                    message="Layer weights validated successfully",
                    level="info",
                    additional_info={
                        "layer_count": len(layers),
                        "layer_weights": layer_weights,
                        "layer_selection_mode": self._config_sections["core_config"].get("layer_selection_mode", "balanced")
                    }
                )
            except Exception as e:
                self.error_manager.handle_error(
                    error=e,
                    error_type="scaffold_influence_error",
                    severity=2,
                    additional_info={"operation": "layer_weights_validation"}
                )
                return False

        try:
            self.cross_attention_injector.set_influence(
                model=base_model,
                core_config=self._config_sections["core_config"],
                cross_attn_config=self._config_sections["cross_attn_config"],
                training_config=self._config_sections["training_config"],
                controls_config=self._config_sections["controls_config"],
                weight=weight,
                blend_strength=blend_strength,
                layer_weights=layer_weights
            )
            self.logger.log_training_event(
                event_type="scaffold_influence_updated",
                message="Scaffold influence updated successfully",
                level="info",
                additional_info={
                    "weight": weight,
                    "blend_strength": blend_strength,
                    "layer_weights": layer_weights,
                    "layer_count": len(layer_weights) if layer_weights else None
                }
            )
            return True
        except Exception as e:
            self.error_manager.handle_error(
                error=e,
                error_type="scaffold_influence_error",
                severity=2,
                additional_info={"operation": "set_scaffold_influence"}
            )
            return False

    def tune_cross_attention(
        self,
        weight: Optional[float] = None,
        blend_strength: Optional[float] = None,
        layer_weights: Optional[List[float]] = None,
        dynamic_mode: Optional[str] = None,
        base_model: Optional[Any] = None
    ) -> bool:
        """Tune cross-attention settings."""
        success = True
        if any(param is not None for param in [weight, blend_strength, layer_weights]):
            success &= self.set_scaffold_influence(weight, blend_strength, layer_weights, base_model)

        if dynamic_mode is not None:
            full_key = "controls_config.dynamic_cross_attn_mode"
            validated_mode = dynamic_mode if dynamic_mode != "off" else None
            if self.validate_param(full_key, validated_mode):
                success &= self.config_manager.update(full_key, validated_mode)
                if success:
                    self._config_sections["controls_config"]["dynamic_cross_attn_mode"] = validated_mode
                    self.config_manager.save_config()
                    self.logger.log_training_event(
                        event_type="tune_cross_attention",
                        message="Cross-attention settings tuned successfully",
                        additional_info={"dynamic_mode": dynamic_mode, "success": success}
                    )
            else:
                success = False
        return success

    def toggle_memory(
        self,
        mode: str,
        use_scaffold_memory: Optional[bool] = None,
        use_token_map_memory: Optional[bool] = None
    ) -> bool:
        """Toggle memory usage modes."""
        modes = {
            'scaffold_mem': (True, False),
            'token_mem': (False, True),
            'both_mem': (True, True),
            'no_mem': (False, False)
        }
        if mode not in modes and (use_scaffold_memory is None or use_token_map_memory is None):
            self.logger.log_training_event(
                event_type="toggle_memory_error",
                message=f"Invalid memory mode: {mode}. Use: {', '.join(modes.keys())} or specify use_scaffold_memory/use_token_map_memory",
                level="error",
                additional_info={"mode": mode, "timestamp": time.time()}
            )
            return False

        scaffold_mem, token_mem = modes.get(mode, (use_scaffold_memory, use_token_map_memory))
        updates = {
            "controls_config.use_scaffold_memory": scaffold_mem,
            "controls_config.use_token_map_memory": token_mem
        }
        success = self._update_config_batch(updates, "controls_config")
        if success:
            self.logger.log_training_event(
                event_type="toggle_memory",
                message="Memory modes toggled",
                additional_info={
                    "mode": mode if mode in modes else "custom",
                    "scaffold_memory": scaffold_mem,
                    "token_map_memory": token_mem,
                    "success": success
                }
            )
        return success

    def update_component_references(
        self,
        curiosity_manager: Optional[ICuriosityManager] = None,
        trainer: Optional[ITrainer] = None,
        cross_attention_injector: Optional[ICrossAttentionInjector] = None
    ) -> None:
        """Update references to dependent components."""
        if curiosity_manager is not None:
            self.curiosity_manager = curiosity_manager
        if trainer is not None:
            self.trainer = trainer
        if cross_attention_injector is not None:
            self.cross_attention_injector = cross_attention_injector
