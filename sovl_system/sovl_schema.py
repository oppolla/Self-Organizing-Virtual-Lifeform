from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from sovl_config import ConfigSchema

class ValidationSchema:
    """Schema definitions for SOVL configuration validation."""
    
    @staticmethod
    def get_schema() -> Dict[str, Dict[str, ConfigSchema]]:
        """Return the configuration schema."""
        return {
            "core_config": {
                "model_name": ConfigSchema(field="core_config.model_name", type=str, required=True),
                "base_model_path": ConfigSchema(field="core_config.base_model_path", type=str, default=None, nullable=True),
                "scaffold_model_name": ConfigSchema(field="core_config.scaffold_model_name", type=str, default=None, nullable=True),
                "scaffold_model_path": ConfigSchema(field="core_config.scaffold_model_path", type=str, default=None, nullable=True),
                "cross_attn_layers": ConfigSchema(field="core_config.cross_attn_layers", type=list, default=[5, 7]),
                "use_dynamic_layers": ConfigSchema(field="core_config.use_dynamic_layers", type=bool, default=False),
                "layer_selection_mode": ConfigSchema(field="core_config.layer_selection_mode", type=str, default="balanced", validator=lambda x: x in ["balanced", "random", "fixed"]),
                "custom_layers": ConfigSchema(field="core_config.custom_layers", type=list, default=None, nullable=True),
                "valid_split_ratio": ConfigSchema(field="core_config.valid_split_ratio", type=float, default=0.2, range=(0.0, 1.0)),
                "random_seed": ConfigSchema(field="core_config.random_seed", type=int, default=42, range=(0, 2**32)),
                "quantization": ConfigSchema(field="core_config.quantization", type=str, default="fp16", validator=lambda x: x in ["fp16", "int8", "none"]),
                "hidden_size": ConfigSchema(field="core_config.hidden_size", type=int, default=768, range=(1, None)),
                "num_heads": ConfigSchema(field="core_config.num_heads", type=int, default=12, range=(1, None)),
                "gradient_checkpointing": ConfigSchema(field="core_config.gradient_checkpointing", type=bool, default=True),
                "initializer_range": ConfigSchema(field="core_config.initializer_range", type=float, default=0.02, range=(0.0, None)),
                "migration_mode": ConfigSchema(field="core_config.migration_mode", type=bool, default=True),
                "device": ConfigSchema(field="core_config.device", type=str, default="cuda", validator=lambda x: x in ["cuda", "cpu"])
            },
            "controls_config": {
                "enable_scaffold": ConfigSchema(field="controls_config.enable_scaffold", type=bool, default=True),
                "scaffold_weight_cap": ConfigSchema(field="controls_config.scaffold_weight_cap", type=float, default=0.9, range=(0.0, 1.0)),
                "scaffold_unk_id": ConfigSchema(field="controls_config.scaffold_unk_id", type=int, default=0, range=(0, None)),
                "enable_cross_attention": ConfigSchema(field="controls_config.enable_cross_attention", type=bool, default=True),
                "enable_dynamic_cross_attention": ConfigSchema(field="controls_config.enable_dynamic_cross_attention", type=bool, default=False),
                "injection_strategy": ConfigSchema(field="controls_config.injection_strategy", type=str, default="sequential", validator=lambda x: x in ["sequential", "parallel"]),
                "temp_eager_threshold": ConfigSchema(field="controls_config.temp_eager_threshold", type=float, default=0.8, range=(0.7, 0.9)),
                "temp_sluggish_threshold": ConfigSchema(field="controls_config.temp_sluggish_threshold", type=float, default=0.6, range=(0.3, 0.6)),
                "temp_mood_influence": ConfigSchema(field="controls_config.temp_mood_influence", type=float, default=0.0, range=(0.0, 1.0)),
                "temp_curiosity_boost": ConfigSchema(field="controls_config.temp_curiosity_boost", type=float, default=0.5, range=(0.0, 0.5)),
                "temp_restless_drop": ConfigSchema(field="controls_config.temp_restless_drop", type=float, default=0.1, range=(0.0, 0.5)),
                "temp_melancholy_noise": ConfigSchema(field="controls_config.temp_melancholy_noise", type=float, default=0.02, range=(0.0, 0.1)),
                "conf_feedback_strength": ConfigSchema(field="controls_config.conf_feedback_strength", type=float, default=0.5, range=(0.0, 1.0)),
                "temp_smoothing_factor": ConfigSchema(field="controls_config.temp_smoothing_factor", type=float, default=0.0, range=(0.0, 1.0)),
                "temperament_decay_rate": ConfigSchema(field="controls_config.temperament_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
                "temperament_history_maxlen": ConfigSchema(field="controls_config.temperament_history_maxlen", type=int, default=5, range=(3, 10)),
                "confidence_history_maxlen": ConfigSchema(field="controls_config.confidence_history_maxlen", type=int, default=5, range=(3, 10))
            },
            "training_config": {
                "learning_rate": ConfigSchema(field="training_config.learning_rate", type=float, default=2e-5, range=(0.0, None), required=True),
                "train_epochs": ConfigSchema(field="training_config.train_epochs", type=int, default=3, range=(1, None)),
                "batch_size": ConfigSchema(field="training_config.batch_size", type=int, default=2, range=(1, None), required=True),
                "max_seq_length": ConfigSchema(field="training_config.max_seq_length", type=int, default=512, range=(1, None)),
                "sigmoid_scale": ConfigSchema(field="training_config.sigmoid_scale", type=float, default=0.5, range=(0.0, None)),
                "sigmoid_shift": ConfigSchema(field="training_config.sigmoid_shift", type=float, default=5.0, range=(0.0, None)),
                "lifecycle_capacity_factor": ConfigSchema(field="training_config.lifecycle_capacity_factor", type=float, default=0.01, range=(0.0, None)),
                "lifecycle_curve": ConfigSchema(field="training_config.lifecycle_curve", type=str, default="sigmoid_linear", validator=lambda x: x in ["sigmoid_linear", "exponential"]),
                "accumulation_steps": ConfigSchema(field="training_config.accumulation_steps", type=int, default=4, range=(1, None)),
                "exposure_gain_eager": ConfigSchema(field="training_config.exposure_gain_eager", type=int, default=3, range=(1, None)),
                "exposure_gain_default": ConfigSchema(field="training_config.exposure_gain_default", type=int, default=2, range=(1, None)),
                "max_patience": ConfigSchema(field="training_config.max_patience", type=int, default=2, range=(0, None)),
                "dry_run": ConfigSchema(field="training_config.dry_run", type=bool, default=False),
                "dry_run_params": ConfigSchema(field="training_config.dry_run_params", type=dict, default=None, nullable=True),
                "weight_decay": ConfigSchema(field="training_config.weight_decay", type=float, default=0.01, range=(0.0, None)),
                "max_grad_norm": ConfigSchema(field="training_config.max_grad_norm", type=float, default=1.0, range=(0.0, None)),
                "use_amp": ConfigSchema(field="training_config.use_amp", type=bool, default=True),
                "checkpoint_interval": ConfigSchema(field="training_config.checkpoint_interval", type=int, default=1000, range=(1, None)),
                "scheduler_type": ConfigSchema(field="training_config.scheduler_type", type=str, default="linear", validator=lambda x: x in ["linear", "cosine", "constant"]),
                "cosine_min_lr": ConfigSchema(field="training_config.cosine_min_lr", type=float, default=1e-6, range=(0.0, None)),
                "warmup_ratio": ConfigSchema(field="training_config.warmup_ratio", type=float, default=0.1, range=(0.0, 1.0)),
                "warmup_steps": ConfigSchema(field="training_config.warmup_steps", type=int, default=0, range=(0, None)),
                "total_steps": ConfigSchema(field="training_config.total_steps", type=int, default=100000, range=(1, None)),
                "validate_every_n_steps": ConfigSchema(field="training_config.validate_every_n_steps", type=int, default=100, range=(1, None)),
                "checkpoint_path": ConfigSchema(field="training_config.checkpoint_path", type=str, default="checkpoints/sovl_trainer"),
                "dropout_rate": ConfigSchema(field="training_config.dropout_rate", type=float, default=0.1, range=(0.0, 1.0)),
                "metrics_to_track": ConfigSchema(field="training_config.metrics_to_track", type=list, default=["loss", "accuracy", "confidence"]),
                "enable_gestation": ConfigSchema(field="training_config.enable_gestation", type=bool, default=True),
                "enable_sleep_training": ConfigSchema(field="training_config.enable_sleep_training", type=bool, default=True),
                "enable_lifecycle_weighting": ConfigSchema(field="training_config.enable_lifecycle_weighting", type=bool, default=True),
                "sleep_conf_threshold": ConfigSchema(field="training_config.sleep_conf_threshold", type=float, default=0.7, range=(0.0, 1.0)),
                "sleep_log_min": ConfigSchema(field="training_config.sleep_log_min", type=int, default=10, range=(1, None)),
                "dream_memory_weight": ConfigSchema(field="training_config.dream_memory_weight", type=float, default=0.1, range=(0.0, 1.0)),
                "enable_dreaming": ConfigSchema(field="training_config.enable_dreaming", type=bool, default=True),
                "repetition_n": ConfigSchema(field="training_config.repetition_n", type=int, default=3, range=(2, None)),
                "dream_noise_scale": ConfigSchema(field="training_config.dream_noise_scale", type=float, default=0.05, range=(0.0, None)),
                "dream_prompt_weight": ConfigSchema(field="training_config.dream_prompt_weight", type=float, default=0.5, range=(0.0, 1.0)),
                "dream_novelty_boost": ConfigSchema(field="training_config.dream_novelty_boost", type=float, default=0.03, range=(0.0, None)),
                "dream_memory_decay": ConfigSchema(field="training_config.dream_memory_decay", type=float, default=0.95, range=(0.0, 1.0)),
                "dream_prune_threshold": ConfigSchema(field="training_config.dream_prune_threshold", type=float, default=0.1, range=(0.0, 1.0)),
                "temp_melancholy_noise": ConfigSchema(field="training_config.temp_melancholy_noise", type=float, default=0.02, range=(0.0, None)),
                "enable_prompt_driven_dreams": ConfigSchema(field="training_config.enable_prompt_driven_dreams", type=bool, default=True),
                "dream_swing_var": ConfigSchema(field="training_config.dream_swing_var", type=float, default=0.1, range=(0.0, None)),
                "dream_lifecycle_delta": ConfigSchema(field="training_config.dream_lifecycle_delta", type=float, default=0.1, range=(0.0, None)),
                "dream_temperament_on": ConfigSchema(field="training_config.dream_temperament_on", type=bool, default=True),
                "confidence_history_maxlen": ConfigSchema(field="training_config.confidence_history_maxlen", type=int, default=5, range=(1, None)),
                "temperament_history_maxlen": ConfigSchema(field="training_config.temperament_history_maxlen", type=int, default=5, range=(1, None)),
                "memory_threshold": ConfigSchema(field="training_config.memory_threshold", type=float, default=0.85, range=(0.0, 1.0)),
                "memory_decay_rate": ConfigSchema(field="training_config.memory_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
                "use_scaffold_memory": ConfigSchema(field="training_config.use_scaffold_memory", type=bool, default=True),
                "use_token_map_memory": ConfigSchema(field="training_config.use_token_map_memory", type=bool, default=True),
                "scaffold_weight": ConfigSchema(field="training_config.scaffold_weight", type=float, default=1.0, range=(0.0, None)),
                "max_dream_memory_mb": ConfigSchema(field="training_config.max_dream_memory_mb", type=int, default=256, range=(1, None)),
                "dream_memory_maxlen": ConfigSchema(field="training_config.dream_memory_maxlen", type=int, default=10, range=(1, None))
            },
            "dynamic_weighting": {
                "min_weight": ConfigSchema(field="dynamic_weighting.min_weight", type=float, default=0.0, range=(0.0, None)),
                "max_weight": ConfigSchema(field="dynamic_weighting.max_weight", type=float, default=1.0, range=(0.0, None)),
                "weight_decay": ConfigSchema(field="dynamic_weighting.weight_decay", type=float, default=0.01, range=(0.0, None)),
                "momentum": ConfigSchema(field="dynamic_weighting.momentum", type=float, default=0.9, range=(0.0, 1.0)),
                "history_size": ConfigSchema(field="dynamic_weighting.history_size", type=int, default=10, range=(1, None)),
                "enable_dynamic_scaling": ConfigSchema(field="dynamic_weighting.enable_dynamic_scaling", type=bool, default=True),
                "weight_curves": ConfigSchema(field="dynamic_weighting.weight_curves", type=list, default=["linear", "sigmoid_linear"])
            },
            "preprocessing": {
                "remove_special_chars": ConfigSchema(field="preprocessing.remove_special_chars", type=bool, default=True),
                "lowercase": ConfigSchema(field="preprocessing.lowercase", type=bool, default=True),
                "remove_extra_spaces": ConfigSchema(field="preprocessing.remove_extra_spaces", type=bool, default=True),
                "max_length": ConfigSchema(field="preprocessing.max_length", type=int, default=512, range=(1, None))
            },
            "augmentation": {
                "synonym_replacement_prob": ConfigSchema(field="augmentation.synonym_replacement_prob", type=float, default=0.3, range=(0.0, 1.0)),
                "word_dropout_prob": ConfigSchema(field="augmentation.word_dropout_prob", type=float, default=0.1, range=(0.0, 1.0)),
                "max_augmentations": ConfigSchema(field="augmentation.max_augmentations", type=int, default=3, range=(0, None))
            },
            "hardware": {
                "enable_cuda": ConfigSchema(field="hardware.enable_cuda", type=bool, default=True),
                "memory_query_interval": ConfigSchema(field="hardware.memory_query_interval", type=float, default=0.1, range=(0.0, None)),
                "mock_memory_total_mb": ConfigSchema(field="hardware.mock_memory_total_mb", type=float, default=8192.0, range=(0.0, None))
            },
            "lora_config": {
                "lora_rank": ConfigSchema(field="lora_config.lora_rank", type=int, default=8, range=(1, None)),
                "lora_alpha": ConfigSchema(field="lora_config.lora_alpha", type=int, default=16, range=(1, None)),
                "lora_dropout": ConfigSchema(field="lora_config.lora_dropout", type=float, default=0.1, range=(0.0, 1.0)),
                "lora_target_modules": ConfigSchema(field="lora_config.lora_target_modules", type=list, default=["c_attn", "c_proj", "c_fc"])
            },
            "curiosity_config": {
                "enable_curiosity": ConfigSchema(field="curiosity_config.enable_curiosity", type=bool, default=True),
                "attention_weight": ConfigSchema(field="curiosity_config.attention_weight", type=float, default=0.5, range=(0.0, 1.0)),
                "queue_maxlen": ConfigSchema(field="curiosity_config.queue_maxlen", type=int, default=10, range=(1, None)),
                "novelty_history_maxlen": ConfigSchema(field="curiosity_config.novelty_history_maxlen", type=int, default=20, range=(1, None)),
                "decay_rate": ConfigSchema(field="curiosity_config.decay_rate", type=float, default=0.9, range=(0.0, 1.0)),
                "question_timeout": ConfigSchema(field="curiosity_config.question_timeout", type=float, default=3600.0, range=(0.0, None)),
                "novelty_threshold_spontaneous": ConfigSchema(field="curiosity_config.novelty_threshold_spontaneous", type=float, default=0.9, range=(0.0, 1.0)),
                "novelty_threshold_response": ConfigSchema(field="curiosity_config.novelty_threshold_response", type=float, default=0.8, range=(0.0, 1.0)),
                "pressure_threshold": ConfigSchema(field="curiosity_config.pressure_threshold", type=float, default=0.7, range=(0.0, 1.0)),
                "pressure_drop": ConfigSchema(field="curiosity_config.pressure_drop", type=float, default=0.3, range=(0.0, 1.0)),
                "silence_threshold": ConfigSchema(field="curiosity_config.silence_threshold", type=float, default=20.0, range=(0.0, None)),
                "question_cooldown": ConfigSchema(field="curiosity_config.question_cooldown", type=float, default=60.0, range=(0.0, None)),
                "weight_ignorance": ConfigSchema(field="curiosity_config.weight_ignorance", type=float, default=0.7, range=(0.0, 1.0)),
                "weight_novelty": ConfigSchema(field="curiosity_config.weight_novelty", type=float, default=0.3, range=(0.0, 1.0)),
                "max_new_tokens": ConfigSchema(field="curiosity_config.max_new_tokens", type=int, default=8, range=(1, None)),
                "base_temperature": ConfigSchema(field="curiosity_config.base_temperature", type=float, default=1.1, range=(0.0, None)),
                "temperament_influence": ConfigSchema(field="curiosity_config.temperament_influence", type=float, default=0.4, range=(0.0, 1.0)),
                "top_k": ConfigSchema(field="curiosity_config.top_k", type=int, default=30, range=(1, None))
            },
            "cross_attn_config": {
                "memory_weight": ConfigSchema(field="cross_attn_config.memory_weight", type=float, default=0.5, range=(0.0, 1.0))
            },
            "logging_config": {
                "log_dir": ConfigSchema(field="logging_config.log_dir", type=str, default="logs"),
                "log_file": ConfigSchema(field="logging_config.log_file", type=str, default="sovl_logs.jsonl"),
                "log_level": ConfigSchema(field="logging_config.log_level", type=str, default="INFO", validator=lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
                "max_log_size_mb": ConfigSchema(field="logging_config.max_log_size_mb", type=int, default=10, range=(1, None)),
                "backup_count": ConfigSchema(field="logging_config.backup_count", type=int, default=5, range=(0, None))
            },
            "error_config": {
                "error_cooldown": ConfigSchema(field="error_config.error_cooldown", type=float, default=1.0, range=(0.0, None)),
                "warning_threshold": ConfigSchema(field="error_config.warning_threshold", type=float, default=3.0, range=(0.0, None)),
                "error_threshold": ConfigSchema(field="error_config.error_threshold", type=float, default=5.0, range=(0.0, None)),
                "critical_threshold": ConfigSchema(field="error_config.critical_threshold", type=float, default=10.0, range=(0.0, None))
            },
            "generation_config": {
                "temperature": ConfigSchema(field="generation_config.temperature", type=float, default=0.7, range=(0.0, None)),
                "top_p": ConfigSchema(field="generation_config.top_p", type=float, default=0.9, range=(0.0, 1.0))
            },
            "data_config": {
                "batch_size": ConfigSchema(field="data_config.batch_size", type=int, default=1, range=(1, None)),
                "max_retries": ConfigSchema(field="data_config.max_retries", type=int, default=3, range=(0, None))
            },
            "memory_config": {
                "max_memory_mb": ConfigSchema(field="memory_config.max_memory_mb", type=int, default=512, range=(1, None)),
                "garbage_collection_threshold": ConfigSchema(field="memory_config.garbage_collection_threshold", type=float, default=0.7, range=(0.0, 1.0))
            },
            "state_config": {
                "max_history": ConfigSchema(field="state_config.max_history", type=int, default=100, range=(1, None)),
                "state_file": ConfigSchema(field="state_config.state_file", type=str, default="sovl_state.json")
            },
            "temperament_config": {
                "mood_influence": ConfigSchema(field="temperament_config.mood_influence", type=float, default=0.5, range=(0.0, 1.0)),
                "history_maxlen": ConfigSchema(field="temperament_config.history_maxlen", type=int, default=5, range=(1, None)),
                "temp_eager_threshold": ConfigSchema(field="controls_config.temp_eager_threshold", type=float, default=0.8, range=(0.7, 0.9)),
                "temp_sluggish_threshold": ConfigSchema(field="controls_config.temp_sluggish_threshold", type=float, default=0.6, range=(0.3, 0.6)),
                "temp_mood_influence": ConfigSchema(field="controls_config.temp_mood_influence", type=float, default=0.0, range=(0.0, 1.0)),
                "temp_curiosity_boost": ConfigSchema(field="controls_config.temp_curiosity_boost", type=float, default=0.5, range=(0.0, 0.5)),
                "temp_restless_drop": ConfigSchema(field="controls_config.temp_restless_drop", type=float, default=0.1, range=(0.0, 0.5)),
                "temp_melancholy_noise": ConfigSchema(field="controls_config.temp_melancholy_noise", type=float, default=0.02, range=(0.0, 0.1)),
                "conf_feedback_strength": ConfigSchema(field="controls_config.conf_feedback_strength", type=float, default=0.5, range=(0.0, 1.0)),
                "temp_smoothing_factor": ConfigSchema(field="controls_config.temp_smoothing_factor", type=float, default=0.0, range=(0.0, 1.0)),
                "temperament_decay_rate": ConfigSchema(field="controls_config.temperament_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
                "temperament_history_maxlen": ConfigSchema(field="controls_config.temperament_history_maxlen", type=int, default=5, range=(3, 10)),
                "confidence_history_maxlen": ConfigSchema(field="controls_config.confidence_history_maxlen", type=int, default=5, range=(3, 10))
            },
            "confidence_config": {
                "history_maxlen": ConfigSchema(field="confidence_config.history_maxlen", type=int, default=5, range=(1, None)),
                "weight": ConfigSchema(field="confidence_config.weight", type=float, default=0.5, range=(0.0, 1.0))
            }
        }
