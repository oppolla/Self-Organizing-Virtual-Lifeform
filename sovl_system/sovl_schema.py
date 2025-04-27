from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from sovl_config import ConfigSchema

class ValidationSchema:
    """Schema definitions for SOVL configuration validation."""
    
    @staticmethod
    def get_schema() -> Dict[str, Dict[str, ConfigSchema]]:
        """Return the configuration schema."""
        return {
            "core_config": ValidationSchema._get_core_config_schema(),
            "model": ValidationSchema._get_model_schema(),
            "controls_config": ValidationSchema._get_controls_config_schema(),
            "temperament_config": ValidationSchema._get_temperament_config_schema(),
            "training_config": ValidationSchema._get_training_config_schema(),
            "scaffold_config": ValidationSchema._get_scaffold_config_schema(),
            "dynamic_weighting": ValidationSchema._get_dynamic_weighting_schema(),
            "curiosity_config": ValidationSchema._get_curiosity_config_schema(),
            "cross_attn_config": ValidationSchema._get_cross_attn_config_schema(),
            "logging_config": ValidationSchema._get_logging_config_schema(),
            "error_config": ValidationSchema._get_error_config_schema(),
            "generation_config": ValidationSchema._get_generation_config_schema(),
            "data_config": ValidationSchema._get_data_config_schema(),
            "data_provider": ValidationSchema._get_data_provider_schema(),
            "memory_config": ValidationSchema._get_memory_config_schema(),
            "state_config": ValidationSchema._get_state_config_schema(),
            "confidence_config": ValidationSchema._get_confidence_config_schema(),
            "lifecycle_config": ValidationSchema._get_lifecycle_config_schema(),
            "gestation_weighting": ValidationSchema._get_gestation_weighting_schema(),
            "monitoring": ValidationSchema._get_monitoring_schema(),
            "scribed_config": ValidationSchema._get_scribed_config_schema(),
            "io_config": ValidationSchema._get_io_config_schema(),
            "metrics_config": ValidationSchema._get_metrics_config_schema(),
            "metadata_weighting": ValidationSchema._get_metadata_weighting_schema(),
            "dreamer_config": ValidationSchema._get_dreamer_config_schema(),
        }

    @staticmethod
    def _get_core_config_schema() -> Dict[str, ConfigSchema]:
        """Return the core_config schema."""
        return {
            "base_model_name": ConfigSchema(field="core_config.base_model_name", type=str, required=True),
            "base_model_path": ConfigSchema(field="core_config.base_model_path", type=str, default=None, nullable=True),
            "scaffold_model_name": ConfigSchema(
                field="core_config.scaffold_model_name", 
                type=str, 
                required=False, 
                default=None, 
                nullable=True
            ),
            "scaffold_model_names": ConfigSchema(
                field="core_config.scaffold_model_names", 
                type=list, 
                required=False, 
                default=None, 
                nullable=True,
                validator=lambda x: all(isinstance(i, str) for i in x) if x is not None else True
            ),
            "scaffold_model_path": ConfigSchema(field="core_config.scaffold_model_path", type=str, default=None, nullable=True),
            "cross_attn_layers": ConfigSchema(field="core_config.cross_attn_layers", type=list, default=[4, 6]),
            "use_dynamic_layers": ConfigSchema(field="core_config.use_dynamic_layers", type=bool, default=False),
            "layer_selection_mode": ConfigSchema(field="core_config.layer_selection_mode", type=str, default="balanced", validator=lambda x: x in ["balanced", "random", "fixed"]),
            "custom_layers": ConfigSchema(field="core_config.custom_layers", type=list, default=None, nullable=True),
            "valid_split_ratio": ConfigSchema(field="core_config.valid_split_ratio", type=float, default=0.2, range=(0.0, 1.0)),
            "random_seed": ConfigSchema(field="core_config.random_seed", type=int, default=42, range=(0, 2**32)),
            "quantization": ConfigSchema(field="core_config.quantization", type=str, default="fp16", validator=lambda x: x in ["fp16", "int8", "none"]),
            "valid_quantization_modes": ConfigSchema(field="core_config.valid_quantization_modes", type=list, default=["fp16", "int8", "int4"], validator=lambda x: all(isinstance(i, str) and i in ["fp16", "int8", "int4"] for i in x)),
            "default_quantization_mode": ConfigSchema(field="core_config.default_quantization_mode", type=str, default="fp16", validator=lambda x: x in ["fp16", "int8", "int4"]),
            "hidden_size": ConfigSchema(field="core_config.hidden_size", type=int, default=768, range=(1, None)),
            "num_heads": ConfigSchema(field="core_config.num_heads", type=int, default=12, range=(1, None)),
            "gradient_checkpointing": ConfigSchema(field="core_config.gradient_checkpointing", type=bool, default=True),
            "initializer_range": ConfigSchema(field="core_config.initializer_range", type=float, default=0.02, range=(0.0, None)),
            "migration_mode": ConfigSchema(field="core_config.migration_mode", type=bool, default=True),
            "device": ConfigSchema(field="core_config.device", type=str, default="cuda", validator=lambda x: x in ["cuda", "cpu"]),
        }

    @staticmethod
    def _get_model_schema() -> Dict[str, ConfigSchema]:
        """Return the model schema."""
        return {
            "model_path": ConfigSchema(field="model.model_path", type=str, required=True),
            "model_type": ConfigSchema(field="model.model_type", type=str, default="causal_lm", validator=lambda x: x in ["causal_lm", "gpt2"]),
            "quantization_mode": ConfigSchema(field="model.quantization_mode", type=str, default="fp16", validator=lambda x: x in ["fp16", "int8", "none"]),
        }

    @staticmethod
    def _get_generation_hooks_schema() -> Dict[str, ConfigSchema]:
        """Return the generation_hooks schema."""
        return {
            "curiosity":    ConfigSchema(field="generation_hooks.curiosity", type=bool, default=True),
            "temperament":  ConfigSchema(field="generation_hooks.temperament", type=bool, default=True),
            "confidence":   ConfigSchema(field="generation_hooks.confidence", type=bool, default=True),
            "bonding":      ConfigSchema(field="generation_hooks.bonding", type=bool, default=True),
        }

    @staticmethod
    def _get_bonding_config_schema() -> Dict[str, ConfigSchema]:
        """Return the bonding_config schema."""
        return {
            "strong_bond_threshold": ConfigSchema(field="bonding_config.strong_bond_threshold", type=float, default=0.8, range=(0.0, 1.0)),
            "weak_bond_threshold": ConfigSchema(field="bonding_config.weak_bond_threshold", type=float, default=0.3, range=(0.0, 1.0)),
            "default_bond_score": ConfigSchema(field="bonding_config.default_bond_score", type=float, default=0.5, range=(0.0, 1.0)),
            "bond_decay_rate": ConfigSchema(field="bonding_config.bond_decay_rate", type=float, default=0.01, range=(0.0, 1.0)),
            "bond_memory_window": ConfigSchema(field="bonding_config.bond_memory_window", type=int, default=100, range=(1, None)),
            "interaction_weight": ConfigSchema(field="bonding_config.interaction_weight", type=float, default=1.0, range=(0.0, None)),
            "modality_weights": ConfigSchema(field="bonding_config.modality_weights", type=dict, default={"text": 1.0, "face": 0.5, "voice": 0.5}),
            "context_strong": ConfigSchema(field="bonding_config.context_strong", type=str, default="You feel a strong, trusting connection to this user. Be warm, open, and familiar."),
            "context_weak": ConfigSchema(field="bonding_config.context_weak", type=str, default="You feel distant from this user. Be formal and reserved."),
            "context_neutral": ConfigSchema(field="bonding_config.context_neutral", type=str, default="You feel a neutral connection to this user. Be polite and neutral."),
            "bond_sensitivity": ConfigSchema(field="bonding_config.bond_sensitivity", type=float, default=1.0, range=(0.0, None)),
            "enable_bonding": ConfigSchema(field="bonding_config.enable_bonding", type=bool, default=True),
        }    

    @staticmethod
    def _get_controls_config_schema() -> Dict[str, ConfigSchema]:
        """Return the controls_config schema."""
        return {
            "enable_scaffold": ConfigSchema(field="controls_config.enable_scaffold", type=bool, default=True),
            "scaffold_weight_cap": ConfigSchema(field="controls_config.scaffold_weight_cap", type=float, default=0.5, range=(0.0, 1.0)),
            "scaffold_unk_id": ConfigSchema(field="controls_config.scaffold_unk_id", type=int, default=0, range=(0, None)),
            "enable_cross_attention": ConfigSchema(field="controls_config.enable_cross_attention", type=bool, default=True),
            "enable_dynamic_cross_attention": ConfigSchema(field="controls_config.enable_dynamic_cross_attention", type=bool, default=False),
            "injection_strategy": ConfigSchema(field="controls_config.injection_strategy", type=str, default="sequential", validator=lambda x: x in ["sequential", "parallel"]),
            "blend_strength": ConfigSchema(field="controls_config.blend_strength", type=float, default=0.5, range=(0.0, 1.0)),
            "attention_weight": ConfigSchema(field="controls_config.attention_weight", type=float, default=0.5, range=(0.0, 1.0)),
            "max_tokens_per_mapping": ConfigSchema(field="controls_config.max_tokens_per_mapping", type=int, default=3, range=(1, 5)),
            "mapping_similarity_threshold": ConfigSchema(field="controls_config.mapping_similarity_threshold", type=float, default=0.7, range=(0.0, 1.0)),
            "allow_bidirectional_mapping": ConfigSchema(field="controls_config.allow_bidirectional_mapping", type=bool, default=False),
            "fallback_strategy": ConfigSchema(field="controls_config.fallback_strategy", type=str, default="split", validator=lambda x: x in ["split", "merge", "nearest", "unk"]),
            "normalization_level": ConfigSchema(field="controls_config.normalization_level", type=str, default="basic", validator=lambda x: x in ["none", "basic", "aggressive"]),
            "min_semantic_similarity": ConfigSchema(field="controls_config.min_semantic_similarity", type=float, default=0.5, range=(0.0, 1.0)),
            "max_meaning_drift": ConfigSchema(field="controls_config.max_meaning_drift", type=float, default=0.3, range=(0.0, 1.0)),
            "enable_periodic_validation": ConfigSchema(field="controls_config.enable_periodic_validation", type=bool, default=True),
            "conflict_resolution_strategy": ConfigSchema(field="controls_config.conflict_resolution_strategy", type=str, default="keep_highest_conf", validator=lambda x: x in ["keep_first", "keep_last", "keep_highest_conf", "merge"]),
        }

    @staticmethod
    def _get_temperament_config_schema() -> Dict[str, ConfigSchema]:
        """Return the temperament_config schema."""
        return {
            "mood_influence": ConfigSchema(field="temperament_config.mood_influence", type=float, default=0.3, range=(0.0, 1.0)),
            "history_maxlen": ConfigSchema(field="temperament_config.history_maxlen", type=int, default=5, range=(1, None)),
            "temp_eager_threshold": ConfigSchema(field="temperament_config.temp_eager_threshold", type=float, default=0.7, range=(0.0, 1.0)),
            "temp_sluggish_threshold": ConfigSchema(field="temperament_config.temp_sluggish_threshold", type=float, default=0.3, range=(0.0, 1.0)),
            "temp_mood_influence": ConfigSchema(field="temperament_config.temp_mood_influence", type=float, default=0.3, range=(0.0, 1.0)),
            "temp_curiosity_boost": ConfigSchema(field="temperament_config.temp_curiosity_boost", type=float, default=0.2, range=(0.0, 1.0)),
            "temp_restless_drop": ConfigSchema(field="temperament_config.temp_restless_drop", type=float, default=0.2, range=(0.0, 1.0)),
            "temp_melancholy_noise": ConfigSchema(field="temperament_config.temp_melancholy_noise", type=float, default=0.02, range=(0.0, 0.1)),
            "conf_feedback_strength": ConfigSchema(field="temperament_config.conf_feedback_strength", type=float, default=0.5, range=(0.0, 1.0)),
            "temp_smoothing_factor": ConfigSchema(field="temperament_config.temp_smoothing_factor", type=float, default=0.5, range=(0.0, 1.0)),
            "temperament_decay_rate": ConfigSchema(field="temperament_config.temperament_decay_rate", type=float, default=0.9, range=(0.0, 1.0)),
            "temperament_history_maxlen": ConfigSchema(field="temperament_config.temperament_history_maxlen", type=int, default=5, range=(1, None)),
            "confidence_history_maxlen": ConfigSchema(field="temperament_config.confidence_history_maxlen", type=int, default=5, range=(1, None)),
            "temperament_pressure_threshold": ConfigSchema(field="temperament_config.temperament_pressure_threshold", type=float, default=0.5, range=(0.0, 1.0)),
            "temperament_max_pressure": ConfigSchema(field="temperament_config.temperament_max_pressure", type=float, default=1.0, range=(0.0, 1.0)),
            "temperament_min_pressure": ConfigSchema(field="temperament_config.temperament_min_pressure", type=float, default=0.0, range=(0.0, 1.0)),
            "temperament_confidence_adjustment": ConfigSchema(field="temperament_config.temperament_confidence_adjustment", type=float, default=0.5, range=(0.0, 1.0)),
            "temperament_pressure_drop": ConfigSchema(field="temperament_config.temperament_pressure_drop", type=float, default=0.2, range=(0.0, 1.0)),
            "lifecycle_params": ConfigSchema(field="temperament_config.lifecycle_params", type=dict, default={
                "gestation": {"bias": 0.1, "decay": 1.0},
                "active": {"bias": 0.0, "decay": 0.9},
                "sleep": {"bias": -0.1, "decay": 0.8}
            }),
        }

    @staticmethod
    def _get_training_config_schema() -> Dict[str, ConfigSchema]:
        """Return the training_config schema."""
        return {
            "model_name": ConfigSchema(field="training_config.model_name", type=str, default="SmolLM2-360M"),
            "learning_rate": ConfigSchema(field="training_config.learning_rate", type=float, default=1.5e-5, range=(0.0, None), required=True),
            "train_epochs": ConfigSchema(field="training_config.train_epochs", type=int, default=1, range=(1, None)),
            "batch_size": ConfigSchema(field="training_config.batch_size", type=int, default=2, range=(1, None), required=True),
            "max_seq_length": ConfigSchema(field="training_config.max_seq_length", type=int, default=512, range=(1, None)),
            "sigmoid_scale": ConfigSchema(field="training_config.sigmoid_scale", type=float, default=0.5, range=(0.0, None)),
            "sigmoid_shift": ConfigSchema(field="training_config.sigmoid_shift", type=float, default=3.0, range=(0.0, None)),
            "lifecycle_capacity_factor": ConfigSchema(field="training_config.lifecycle_capacity_factor", type=float, default=0.01, range=(0.0, None)),
            "lifecycle_curve": ConfigSchema(field="training_config.lifecycle_curve", type=str, default="sigmoid_linear", validator=lambda x: x in ["sigmoid_linear", "exponential"]),
            "grad_accum_steps": ConfigSchema(field="training_config.grad_accum_steps", type=int, default=4, range=(1, None)),
            "exposure_gain_eager": ConfigSchema(field="training_config.exposure_gain_eager", type=int, default=2, range=(1, None)),
            "exposure_gain_default": ConfigSchema(field="training_config.exposure_gain_default", type=int, default=2, range=(1, None)),
            "max_patience": ConfigSchema(field="training_config.max_patience", type=int, default=2, range=(0, None)),
            "dry_run": ConfigSchema(field="training_config.dry_run", type=bool, default=False),
            "dry_run_params": ConfigSchema(field="training_config.dry_run_params", type=dict, default={
                "max_samples": 4,
                "max_length": 128,
                "validate_architecture": True,
                "skip_training": True
            }),
            "weight_decay": ConfigSchema(field="training_config.weight_decay", type=float, default=0.01, range=(0.0, None)),
            "max_grad_norm": ConfigSchema(field="training_config.max_grad_norm", type=float, default=1.0, range=(0.0, None)),
            "use_amp": ConfigSchema(field="training_config.use_amp", type=bool, default=True),
            "checkpoint_interval": ConfigSchema(field="training_config.checkpoint_interval", type=int, default=1000, range=(1, None)),
            "scheduler_type": ConfigSchema(field="training_config.scheduler_type", type=str, default="linear", validator=lambda x: x in ["linear", "cosine", "constant"]),
            "cosine_min_lr": ConfigSchema(field="training_config.cosine_min_lr", type=float, default=1e-6, range=(0.0, None)),
            "warmup_ratio": ConfigSchema(field="training_config.warmup_ratio", type=float, default=0.1, range=(0.0, 1.0)),
            "warmup_steps": ConfigSchema(field="training_config.warmup_steps", type=int, default=300, range=(0, None)),
            "total_steps": ConfigSchema(field="training_config.total_steps", type=int, default=5000, range=(1, None)),
            "validate_every_n_steps": ConfigSchema(field="training_config.validate_every_n_steps", type=int, default=100, range=(1, None)),
            "checkpoint_path": ConfigSchema(field="training_config.checkpoint_path", type=str, default="checkpoints/sovl_trainer"),
            "dropout_rate": ConfigSchema(field="training_config.dropout_rate", type=float, default=0.1, range=(0.0, 1.0)),
            "max_epochs": ConfigSchema(field="training_config.max_epochs", type=int, default=1, range=(1, None)),
            "metrics_to_track": ConfigSchema(field="training_config.metrics_to_track", type=list, default=["loss", "accuracy", "confidence"]),
            "enable_gestation": ConfigSchema(field="training_config.enable_gestation", type=bool, default=True),
            "enable_sleep_training": ConfigSchema(field="training_config.enable_sleep_training", type=bool, default=True),
            "enable_lifecycle_weighting": ConfigSchema(field="training_config.enable_lifecycle_weighting", type=bool, default=True),
            "sleep_conf_threshold": ConfigSchema(field="training_config.sleep_conf_threshold", type=float, default=0.7, range=(0.0, 1.0)),
            "sleep_log_min": ConfigSchema(field="training_config.sleep_log_min", type=int, default=10, range=(1, None)),
            "dream_memory_weight": ConfigSchema(field="training_config.dream_memory_weight", type=float, default=0.03, range=(0.0, 1.0)),
            "enable_dreaming": ConfigSchema(field="training_config.enable_dreaming", type=bool, default=True),
            "repetition_n": ConfigSchema(field="training_config.repetition_n", type=int, default=3, range=(2, None)),
            "dream_noise_scale": ConfigSchema(field="training_config.dream_noise_scale", type=float, default=0.01, range=(0.0, None)),
            "dream_prompt_weight": ConfigSchema(field="training_config.dream_prompt_weight", type=float, default=0.5, range=(0.0, 1.0)),
            "dream_novelty_boost": ConfigSchema(field="training_config.dream_novelty_boost", type=float, default=0.03, range=(0.0, None)),
            "dream_memory_decay": ConfigSchema(field="training_config.dream_memory_decay", type=float, default=0.95, range=(0.0, 1.0)),
            "dream_prune_threshold": ConfigSchema(field="training_config.dream_prune_threshold", type=float, default=0.1, range=(0.0, 1.0)),
            "enable_prompt_driven_dreams": ConfigSchema(field="training_config.enable_prompt_driven_dreams", type=bool, default=True),
            "dream_swing_var": ConfigSchema(field="training_config.dream_swing_var", type=float, default=0.1, range=(0.0, None)),
            "dream_lifecycle_delta": ConfigSchema(field="training_config.dream_lifecycle_delta", type=float, default=0.1, range=(0.0, None)),
            "dream_temperament_on": ConfigSchema(field="training_config.dream_temperament_on", type=bool, default=False),
            "confidence_history_maxlen": ConfigSchema(field="training_config.confidence_history_maxlen", type=int, default=5, range=(1, None)),
            "temperament_history_maxlen": ConfigSchema(field="training_config.temperament_history_maxlen", type=int, default=5, range=(1, None)),
            "memory_threshold": ConfigSchema(field="training_config.memory_threshold", type=float, default=0.85, range=(0.0, 1.0)),
            "memory_decay_rate": ConfigSchema(field="training_config.memory_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
            "use_scaffold_memory": ConfigSchema(field="training_config.use_scaffold_memory", type=bool, default=True),
            "use_token_map_memory": ConfigSchema(field="training_config.use_token_map_memory", type=bool, default=True),
            "scaffold_weight": ConfigSchema(field="training_config.scaffold_weight", type=float, default=0.3, range=(0.0, None)),
            "max_dream_memory_mb": ConfigSchema(field="training_config.max_dream_memory_mb", type=int, default=128, range=(1, None)),
            "dream_memory_maxlen": ConfigSchema(field="training_config.dream_memory_maxlen", type=int, default=3, range=(1, None)),
        }

    @staticmethod
    def _get_scaffold_config_schema() -> Dict[str, ConfigSchema]:
        """Return the scaffold_config schema."""
        return {
            "model_path": ConfigSchema(field="scaffold_config.model_path", type=str, required=True),
            "model_type": ConfigSchema(field="scaffold_config.model_type", type=str, default="gpt2"),
            "tokenizer_path": ConfigSchema(field="scaffold_config.tokenizer_path", type=str, required=True),
            "quantization_mode": ConfigSchema(field="scaffold_config.quantization_mode", type=str, default="int8", validator=lambda x: x in ["fp16", "int8", "none"]),
        }

    @staticmethod
    def _get_dynamic_weighting_schema() -> Dict[str, ConfigSchema]:
        """Return the dynamic_weighting schema."""
        return {
            "min_weight": ConfigSchema(field="dynamic_weighting.min_weight", type=float, default=0.0, range=(0.0, None)),
            "max_weight": ConfigSchema(field="dynamic_weighting.max_weight", type=float, default=1.0, range=(0.0, None)),
            "weight_decay": ConfigSchema(field="dynamic_weighting.weight_decay", type=float, default=0.01, range=(0.0, None)),
            "momentum": ConfigSchema(field="dynamic_weighting.momentum", type=float, default=0.9, range=(0.0, 1.0)),
            "history_size": ConfigSchema(field="dynamic_weighting.history_size", type=int, default=5, range=(1, None)),
            "enable_dynamic_scaling": ConfigSchema(field="dynamic_weighting.enable_dynamic_scaling", type=bool, default=True),
            "weight_curves": ConfigSchema(field="dynamic_weighting.weight_curves", type=list, default=["linear", "sigmoid_linear"]),
        }

    @staticmethod
    def _get_curiosity_config_schema() -> Dict[str, ConfigSchema]:
        """Return the curiosity_config schema."""
        return {
            "enable_curiosity": ConfigSchema(field="curiosity_config.enable_curiosity", type=bool, default=True),
            "attention_weight": ConfigSchema(field="curiosity_config.attention_weight", type=float, default=0.3, range=(0.0, 1.0)),
            "queue_maxlen": ConfigSchema(field="curiosity_config.queue_maxlen", type=int, default=50, range=(1, None)),
            "novelty_history_maxlen": ConfigSchema(field="curiosity_config.novelty_history_maxlen", type=int, default=20, range=(1, None)),
            "decay_rate": ConfigSchema(field="curiosity_config.decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
            "question_timeout": ConfigSchema(field="curiosity_config.question_timeout", type=float, default=1800.0, range=(0.0, None)),
            "novelty_threshold_spontaneous": ConfigSchema(field="curiosity_config.novelty_threshold_spontaneous", type=float, default=0.8, range=(0.0, 1.0)),
            "novelty_threshold_response": ConfigSchema(field="curiosity_config.novelty_threshold_response", type=float, default=0.8, range=(0.0, 1.0)),
            "pressure_threshold": ConfigSchema(field="curiosity_config.pressure_threshold", type=float, default=0.55, range=(0.0, 1.0)),
            "pressure_drop": ConfigSchema(field="curiosity_config.pressure_drop", type=float, default=0.3, range=(0.0, 1.0)),
            "silence_threshold": ConfigSchema(field="curiosity_config.silence_threshold", type=float, default=20.0, range=(0.0, None)),
            "question_cooldown": ConfigSchema(field="curiosity_config.question_cooldown", type=float, default=60.0, range=(0.0, None)),
            "weight_ignorance": ConfigSchema(field="curiosity_config.weight_ignorance", type=float, default=0.7, range=(0.0, 1.0)),
            "weight_novelty": ConfigSchema(field="curiosity_config.weight_novelty", type=float, default=0.3, range=(0.0, 1.0)),
            "max_new_tokens": ConfigSchema(field="curiosity_config.max_new_tokens", type=int, default=8, range=(1, None)),
            "base_temperature": ConfigSchema(field="curiosity_config.base_temperature", type=float, default=1.1, range=(0.0, None)),
            "temperament_influence": ConfigSchema(field="curiosity_config.temperament_influence", type=float, default=0.4, range=(0.0, 1.0)),
            "top_k": ConfigSchema(field="curiosity_config.top_k", type=int, default=30, range=(1, None)),
            "max_memory_mb": ConfigSchema(field="curiosity_config.max_memory_mb", type=int, default=128, range=(1, None)),
            "pressure_change_cooldown": ConfigSchema(field="curiosity_config.pressure_change_cooldown", type=float, default=60.0, range=(0.0, None)),
            "min_pressure": ConfigSchema(field="curiosity_config.min_pressure", type=float, default=0.1, range=(0.0, 1.0)),
            "max_pressure": ConfigSchema(field="curiosity_config.max_pressure", type=float, default=0.9, range=(0.0, 1.0)),
            "pressure_decay_rate": ConfigSchema(field="curiosity_config.pressure_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
            "metrics_maxlen": ConfigSchema(field="curiosity_config.metrics_maxlen", type=int, default=20, range=(1, None)),
            "min_temperature": ConfigSchema(field="curiosity_config.min_temperature", type=float, default=0.7, range=(0.0, None)),
            "max_temperature": ConfigSchema(field="curiosity_config.max_temperature", type=float, default=1.7, range=(0.0, None)),
            "lifecycle_params": ConfigSchema(field="curiosity_config.lifecycle_params", type=dict, default={
                "gestation": {
                    "pressure_reduction": 0.3,
                    "novelty_boost": 0.2
                },
                "active": {
                    "pressure_reduction": 0.1,
                    "novelty_boost": 0.1
                },
                "sleep": {
                    "pressure_reduction": 0.5,
                    "novelty_boost": 0.3
                }
            }),
        }

    @staticmethod
    def _get_cross_attn_config_schema() -> Dict[str, ConfigSchema]:
        """Return the cross_attn_config schema."""
        return {
            "memory_weight": ConfigSchema(field="cross_attn_config.memory_weight", type=float, default=0.2, range=(0.0, 1.0)),
        }

    @staticmethod
    def _get_logging_config_schema() -> Dict[str, ConfigSchema]:
        """Return the logging_config schema."""
        return {
            "log_dir": ConfigSchema(field="logging_config.log_dir", type=str, default="logs"),
            "log_file": ConfigSchema(field="logging_config.log_file", type=str, default="sovl_logs.jsonl"),
            "log_level": ConfigSchema(field="logging_config.log_level", type=str, default="INFO", validator=lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
            "max_log_size_mb": ConfigSchema(field="logging_config.max_log_size_mb", type=int, default=10, range=(1, None)),
            "backup_count": ConfigSchema(field="logging_config.backup_count", type=int, default=5, range=(0, None)),
        }

    @staticmethod
    def _get_error_config_schema() -> Dict[str, ConfigSchema]:
        """Return the error_config schema."""
        return {
            "error_cooldown": ConfigSchema(field="error_config.error_cooldown", type=float, default=1.0, range=(0.0, None)),
            "warning_threshold": ConfigSchema(field="error_config.warning_threshold", type=float, default=5.0, range=(0.0, None)),
            "error_threshold": ConfigSchema(field="error_config.error_threshold", type=float, default=7.0, range=(0.0, None)),
            "critical_threshold": ConfigSchema(field="error_config.critical_threshold", type=float, default=10.0, range=(0.0, None)),
        }

    @staticmethod
    def _get_generation_config_schema() -> Dict[str, ConfigSchema]:
        """Return the generation_config schema."""
        return {
            "temperature": ConfigSchema(field="generation_config.temperature", type=float, default=0.7, range=(0.0, None)),
            "top_p": ConfigSchema(field="generation_config.top_p", type=float, default=0.9, range=(0.0, 1.0)),
        }

    @staticmethod
    def _get_data_config_schema() -> Dict[str, ConfigSchema]:
        """Return the data_config schema."""
        return {
            "batch_size": ConfigSchema(field="data_config.batch_size", type=int, default=2, range=(1, None)),
            "max_retries": ConfigSchema(field="data_config.max_retries", type=int, default=3, range=(0, None)),
        }

    @staticmethod
    def _get_data_provider_schema() -> Dict[str, ConfigSchema]:
        """Return the data_provider schema."""
        return {
            "provider_type": ConfigSchema(field="data_provider.provider_type", type=str, default="default"),
            "data_path": ConfigSchema(field="data_provider.data_path", type=str, required=True),
        }

    @staticmethod
    def _get_memory_config_schema() -> Dict[str, Dict[str, ConfigSchema]]:
        """Return the memory_config schema."""
        return {
            "memoria": {
                "max_memory_mb": ConfigSchema(field="memory_config.memoria.max_memory_mb", type=int, default=512, range=(1, None)),
                "garbage_collection_threshold": ConfigSchema(field="memory_config.memoria.garbage_collection_threshold", type=float, default=0.7, range=(0.0, 1.0)),
                "memory_decay_rate": ConfigSchema(field="memory_config.memoria.memory_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),
                "enable_memory_compression": ConfigSchema(field="memory_config.memoria.enable_memory_compression", type=bool, default=True),
                "compression_ratio": ConfigSchema(field="memory_config.memoria.compression_ratio", type=float, default=0.5, range=(0.0, 1.0)),
                "max_compressed_memory_mb": ConfigSchema(field="memory_config.memoria.max_compressed_memory_mb", type=int, default=1024, range=(1, None)),
            },
            "ram": {
                "max_ram_mb": ConfigSchema(field="memory_config.ram.max_ram_mb", type=int, default=2048, range=(1, None)),
                "ram_threshold": ConfigSchema(field="memory_config.ram.ram_threshold", type=float, default=0.8, range=(0.0, 1.0)),
                "enable_ram_compression": ConfigSchema(field="memory_config.ram.enable_ram_compression", type=bool, default=True),
                "ram_compression_ratio": ConfigSchema(field="memory_config.ram.ram_compression_ratio", type=float, default=0.6, range=(0.0, 1.0)),
                "max_compressed_ram_mb": ConfigSchema(field="memory_config.ram.max_compressed_ram_mb", type=int, default=4096, range=(1, None)),
            },
            "gpu": {
                "max_gpu_memory_mb": ConfigSchema(field="memory_config.gpu.max_gpu_memory_mb", type=int, default=1024, range=(1, None)),
                "gpu_memory_threshold": ConfigSchema(field="memory_config.gpu.gpu_memory_threshold", type=float, default=0.85, range=(0.0, 1.0)),
                "enable_gpu_memory_compression": ConfigSchema(field="memory_config.gpu.enable_gpu_memory_compression", type=bool, default=True),
                "gpu_compression_ratio": ConfigSchema(field="memory_config.gpu.gpu_compression_ratio", type=float, default=0.7, range=(0.0, 1.0)),
                "max_compressed_gpu_memory_mb": ConfigSchema(field="memory_config.gpu.max_compressed_gpu_memory_mb", type=int, default=2048, range=(1, None)),
            },
            "manager": {
                "enable_memoria_manager": ConfigSchema(field="memory_config.manager.enable_memoria_manager", type=bool, default=True),
                "enable_ram_manager": ConfigSchema(field="memory_config.manager.enable_ram_manager", type=bool, default=True),
                "enable_gpu_memory_manager": ConfigSchema(field="memory_config.manager.enable_gpu_memory_manager", type=bool, default=True),
                "memory_sync_interval": ConfigSchema(field="memory_config.manager.memory_sync_interval", type=int, default=60, range=(1, None)),
                "enable_memory_monitoring": ConfigSchema(field="memory_config.manager.enable_memory_monitoring", type=bool, default=True),
                "memory_monitoring_interval": ConfigSchema(field="memory_config.manager.memory_monitoring_interval", type=int, default=5, range=(1, None)),
            },
        }

    @staticmethod
    def _get_state_config_schema() -> Dict[str, ConfigSchema]:
        """Return the state_config schema."""
        return {
            "max_history": ConfigSchema(field="state_config.max_history", type=int, default=100, range=(1, None)),
            "state_file": ConfigSchema(field="state_config.state_file", type=str, default="sovl_state.json"),
        }

    @staticmethod
    def _get_confidence_config_schema() -> Dict[str, ConfigSchema]:
        """Return the confidence_config schema."""
        return {
            "history_maxlen": ConfigSchema(field="confidence_config.history_maxlen", type=int, default=5, range=(1, None)),
            "weight": ConfigSchema(field="confidence_config.weight", type=float, default=0.5, range=(0.0, 1.0)),
        }

    @staticmethod
    def _get_dream_memory_config_schema() -> Dict[str, ConfigSchema]:
        """Deprecated: Dream memory config is no longer used. Kept for backward compatibility."""
        return {}

    @staticmethod
    def _get_lifecycle_config_schema() -> Dict[str, ConfigSchema]:
        """Return the lifecycle_config schema."""
        return {
            "enable_gestation": ConfigSchema(field="lifecycle_config.enable_gestation", type=bool, default=True),
            "enable_sleep_training": ConfigSchema(field="lifecycle_config.enable_sleep_training", type=bool, default=True),
            "enable_lifecycle_weighting": ConfigSchema(field="lifecycle_config.enable_lifecycle_weighting", type=bool, default=True),
            "lifecycle_capacity_factor": ConfigSchema(field="lifecycle_config.lifecycle_capacity_factor", type=float, default=0.01, range=(0.0, 1.0)),
            "lifecycle_curve": ConfigSchema(field="lifecycle_config.lifecycle_curve", type=str, default="sigmoid_linear", validator=lambda x: x in ["sigmoid_linear", "exponential"]),
            "sleep_conf_threshold": ConfigSchema(field="lifecycle_config.sleep_conf_threshold", type=float, default=0.7, range=(0.0, 1.0)),
            "sleep_log_min": ConfigSchema(field="lifecycle_config.sleep_log_min", type=int, default=10, range=(1, None)),
            "exposure_gain_eager": ConfigSchema(field="lifecycle_config.exposure_gain_eager", type=int, default=3, range=(1, None)),
            "exposure_gain_default": ConfigSchema(field="lifecycle_config.exposure_gain_default", type=int, default=2, range=(1, None)),
            "fatigue_factor_cap": ConfigSchema(field="lifecycle_config.fatigue_factor_cap", type=float, default=0.2, range=(0.0, 0.5)),
            "fatigue_scaling_factor": ConfigSchema(field="lifecycle_config.fatigue_scaling_factor", type=float, default=2.0, range=(1.0, 5.0)),
        }
    
    @staticmethod
    def _get_monitoring_schema() -> Dict[str, ConfigSchema]:
        """Return the monitoring schema."""
        return {
            "ram_critical_threshold_percent": ConfigSchema(field="monitoring.ram_critical_threshold_percent", type=float, default=90.0, range=(0.0, 100.0)),
            "gpu_critical_threshold_percent": ConfigSchema(field="monitoring.gpu_critical_threshold_percent", type=float, default=95.0, range=(0.0, 100.0)),
            "ram_critical_usage_mb": ConfigSchema(field="monitoring.ram_critical_usage_mb", type=int, default=8192, range=(1, None)),
            "gpu_critical_usage_mb": ConfigSchema(field="monitoring.gpu_critical_usage_mb", type=int, default=4096, range=(1, None)),
            "traits_update_interval_seconds": ConfigSchema(field="monitoring.traits_update_interval_seconds", type=float, default=0.5, range=(0.1, 10.0)),
            "trait_history_size": ConfigSchema(field="monitoring.trait_history_size", type=int, default=100, range=(10, 1000)),
            "min_samples_for_variance": ConfigSchema(field="monitoring.min_samples_for_variance", type=int, default=10, range=(2, 100)),
            "trait_variance_thresholds": ConfigSchema(field="monitoring.trait_variance_thresholds", type=dict, default={"default": 0.1}, validator=lambda x: all(0 <= v <= 1 for v in x.values())),
        }

    @staticmethod
    def _get_gestation_weighting_schema() -> Dict:
        """Return the gestation_weighting schema."""
        return {
            "type": "object",
            "required": ["metadata_fields", "content_weights", "confidence_weights", "temperament_weights", "context_weights", "timing_weights", "weight_bounds"],
            "properties": {
                "metadata_fields": {
                    "type": "object",
                    "required": ["quality_metrics", "content_metrics", "confidence_metrics", "context_metrics"],
                    "properties": {
                        "quality_metrics": {
                            "type": "object",
                            "required": ["code", "url", "question", "exclamation", "emoji"],
                            "properties": {
                                "code": {
                                    "type": "object",
                                    "required": ["enabled", "weight"],
                                    "properties": {
                                        "enabled": {"type": "boolean", "default": True},
                                        "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.5}
                                    }
                                },
                                "url": {
                                    "type": "object",
                                    "required": ["enabled", "weight"],
                                    "properties": {
                                        "enabled": {"type": "boolean", "default": True},
                                        "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.3}
                                    }
                                },
                                "question": {
                                    "type": "object",
                                    "required": ["enabled", "weight"],
                                    "properties": {
                                        "enabled": {"type": "boolean", "default": True},
                                        "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.2}
                                    }
                                },
                                "exclamation": {
                                    "type": "object",
                                    "required": ["enabled", "weight"],
                                    "properties": {
                                        "enabled": {"type": "boolean", "default": True},
                                        "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.1}
                                    }
                                },
                                "emoji": {
                                    "type": "object",
                                    "required": ["enabled", "weight"],
                                    "properties": {
                                        "enabled": {"type": "boolean", "default": True},
                                        "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.9}
                                    }
                                }
                            }
                        },
                        "content_metrics": {
                            "type": "object",
                            "required": ["word_count", "sentence_count", "avg_word_length", "avg_sentence_length"],
                            "properties": {
                                "word_count": {"type": "integer", "minimum": 0, "default": 0},
                                "sentence_count": {"type": "integer", "minimum": 0, "default": 0},
                                "avg_word_length": {"type": "number", "minimum": 0, "default": 0.0},
                                "avg_sentence_length": {"type": "number", "minimum": 0, "default": 0.0}
                            }
                        },
                        "confidence_metrics": {
                            "type": "object",
                            "required": ["confidence_score", "temperament_score"],
                            "properties": {
                                "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
                                "temperament_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5}
                            }
                        },
                        "context_metrics": {
                            "type": "object",
                            "required": ["is_first_message", "is_last_message", "context_window_size", "response_time"],
                            "properties": {
                                "is_first_message": {"type": "boolean", "default": False},
                                "is_last_message": {"type": "boolean", "default": False},
                                "context_window_size": {"type": "integer", "minimum": 0, "default": 0},
                                "response_time": {"type": "number", "minimum": 0, "default": 0.0}
                            }
                        }
                    }
                },
                "content_weights": {
                    "type": "object",
                    "required": ["word_count_ratio_scale", "optimal_word_length_range", "suboptimal_word_length_weight", "optimal_sentence_count_range", "excessive_sentence_weight", "optimal_sentence_length_range", "excessive_sentence_length_weight"],
                    "properties": {
                        "word_count_ratio_scale": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.5},
                        "optimal_word_length_range": {
                            "type": "object",
                            "required": ["min", "max", "weight"],
                            "properties": {
                                "min": {"type": "number", "minimum": 0, "default": 4},
                                "max": {"type": "number", "minimum": 0, "default": 8},
                                "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.2}
                            }
                        },
                        "suboptimal_word_length_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.8},
                        "optimal_sentence_count_range": {
                            "type": "object",
                            "required": ["min", "max", "weight"],
                            "properties": {
                                "min": {"type": "integer", "minimum": 0, "default": 2},
                                "max": {"type": "integer", "minimum": 0, "default": 5},
                                "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.2}
                            }
                        },
                        "excessive_sentence_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.9},
                        "optimal_sentence_length_range": {
                            "type": "object",
                            "required": ["min", "max", "weight"],
                            "properties": {
                                "min": {"type": "number", "minimum": 0, "default": 10},
                                "max": {"type": "number", "minimum": 0, "default": 20},
                                "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.2}
                            }
                        },
                        "excessive_sentence_length_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.9}
                    }
                },
                "confidence_weights": {
                    "type": "object",
                    "required": ["high_confidence_threshold", "high_confidence_weight", "moderate_confidence_threshold", "moderate_confidence_weight", "low_confidence_threshold", "low_confidence_weight"],
                    "properties": {
                        "high_confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.8},
                        "high_confidence_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.3},
                        "moderate_confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6},
                        "moderate_confidence_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.1},
                        "low_confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.4},
                        "low_confidence_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.8}
                    }
                },
                "temperament_weights": {
                    "type": "object",
                    "required": ["balanced_range", "extreme_temperament_weight"],
                    "properties": {
                        "balanced_range": {
                            "type": "object",
                            "required": ["min", "max", "weight"],
                            "properties": {
                                "min": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.4},
                                "max": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.6},
                                "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.2}
                            }
                        },
                        "extreme_temperament_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.9}
                    }
                },
                "context_weights": {
                    "type": "object",
                    "required": ["first_last_message_weight", "optimal_context_range", "excessive_context_weight"],
                    "properties": {
                        "first_last_message_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.2},
                        "optimal_context_range": {
                            "type": "object",
                            "required": ["min", "max", "weight"],
                            "properties": {
                                "min": {"type": "integer", "minimum": 0, "default": 2},
                                "max": {"type": "integer", "minimum": 0, "default": 4},
                                "weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.1}
                            }
                        },
                        "excessive_context_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 0.9}
                    }
                },
                "timing_weights": {
                    "type": "object",
                    "required": ["max_response_time", "optimal_timing_weight"],
                    "properties": {
                        "max_response_time": {"type": "number", "minimum": 0, "default": 60},
                        "optimal_timing_weight": {"type": "number", "minimum": 0.1, "maximum": 3.0, "default": 1.0}
                    }
                },
                "weight_bounds": {
                    "type": "object",
                    "required": ["min", "max"],
                    "properties": {
                        "min": {"type": "number", "minimum": 0.1, "default": 0.1},
                        "max": {"type": "number", "minimum": 0.1, "default": 3.0}
                    }
                },

                "experience_memory": {
                    "max_short_term": ConfigSchema(
                        field="experience_memory.max_short_term",
                        type=int,
                        default=50,
                        range=(1, 1000),
                        required=False
                    ),
                    "short_term_expiry_seconds": ConfigSchema(
                        field="experience_memory.short_term_expiry_seconds",
                        type=int,
                        default=3600,
                        range=(1, None),
                        required=False
                    ),
                    "embedding_dim": ConfigSchema(
                        field="experience_memory.embedding_dim",
                        type=int,
                        default=128,
                        range=(1, None),
                        required=False
                    ),
                    "long_term_top_k": ConfigSchema(
                        field="experience_memory.long_term_top_k",
                        type=int,
                        default=5,
                        range=(1, 100),
                        required=False
                    ),
                    "long_term_retention_days": ConfigSchema(
                        field="experience_memory.long_term_retention_days",
                        type=int,
                        default=30,
                        range=(1, 3650),
                        required=False
                    ),
                    "memory_logging_level": ConfigSchema(
                        field="experience_memory.memory_logging_level",
                        type=str,
                        default="info",
                        validator=lambda x: x in ["debug", "info", "warning", "error", "critical"],
                        required=False
                    ),
                },

            }
    
        }

    @staticmethod
    def _get_metadata_weighting_schema() -> Dict[str, ConfigSchema]:
        """Return the metadata_weighting schema (event_type weights)."""
        # This schema allows arbitrary event_type keys, but documents common defaults
        return {
            "internal_error_reflection": ConfigSchema(field="metadata_weighting.internal_error_reflection", type=float, default=0.2, range=(0.0, 3.0)),
            "user_message": ConfigSchema(field="metadata_weighting.user_message", type=float, default=1.0, range=(0.0, 3.0)),
            "system_message": ConfigSchema(field="metadata_weighting.system_message", type=float, default=0.5, range=(0.0, 3.0)),
            "dream_memory": ConfigSchema(field="metadata_weighting.dream_memory", type=float, default=0.3, range=(0.0, 3.0)),
        }

    @staticmethod
    def _get_scribed_config_schema() -> Dict[str, ConfigSchema]:
        """Return the scribed_config schema."""
        return {
            "output_path": ConfigSchema(field="scribed_config.output_path", type=str, default="scribe/sovl_scribe.jsonl"),
            "max_file_size_mb": ConfigSchema(field="scribed_config.max_file_size_mb", type=int, default=50, range=(1, None)),
            "buffer_size": ConfigSchema(field="scribed_config.buffer_size", type=int, default=10, range=(1, None)),
        }

    @staticmethod
    def _get_io_config_schema() -> Dict[str, ConfigSchema]:
        """Return the io_config schema."""
        return {
            "field_mapping": ConfigSchema(
                field="io_config.field_mapping",
                type=dict,
                default={"prompt": "prompt", "completion": "completion"},
                required=False,
                validator=lambda v: isinstance(v, dict) and all(isinstance(k, str) and isinstance(val, str) for k, val in v.items())
            ),
            "required_fields": ConfigSchema(
                field="io_config.required_fields",
                type=list,
                default=["prompt", "completion"],
                required=False,
                validator=lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v)
            ),
            "min_string_length": ConfigSchema(
                field="io_config.min_string_length",
                type=int,
                default=1,
                required=False,
                range=(0, None)
            ),
            "max_string_length": ConfigSchema(
                field="io_config.max_string_length",
                type=int,
                default=10000,
                required=False,
                range=(1, None)
            ),
            "strict_validation": ConfigSchema(
                field="io_config.strict_validation",
                type=bool,
                default=False,
                required=False
            ),
            "seed_file": ConfigSchema(
                field="io_config.seed_file",
                type=str,
                default="sovl_seed.jsonl",
                required=False
            ),
            "min_training_entries": ConfigSchema(
                field="io_config.min_training_entries",
                type=int,
                default=10,
                required=False,
                range=(0, None)
            ),
            "shuffle_data": ConfigSchema(
                field="io_config.shuffle_data",
                type=bool,
                default=True,
                required=False
            )
        }

    @staticmethod
    def _get_metrics_config_schema() -> Dict[str, ConfigSchema]:
        """Return the metrics_config schema."""
        return {
            "token_statistics": {
                "enabled": ConfigSchema(
                    field="metrics_config.token_statistics.enabled",
                    type=bool,
                    default=True
                ),
                "ngram_sizes": ConfigSchema(
                    field="metrics_config.token_statistics.ngram_sizes",
                    type=list,
                    default=[2, 3, 4],
                    validator=lambda x: all(isinstance(n, int) and n > 0 for n in x)
                ),
                "special_token_tracking": ConfigSchema(
                    field="metrics_config.token_statistics.special_token_tracking",
                    type=bool,
                    default=True
                ),
                "min_token_frequency": ConfigSchema(
                    field="metrics_config.token_statistics.min_token_frequency",
                    type=int,
                    default=2,
                    range=(1, None)
                ),
                "max_tokens_to_track": ConfigSchema(
                    field="metrics_config.token_statistics.max_tokens_to_track",
                    type=int,
                    default=1000,
                    range=(1, None)
                ),
                "token_pattern_threshold": ConfigSchema(
                    field="metrics_config.token_statistics.token_pattern_threshold",
                    type=float,
                    default=0.1,
                    range=(0.0, 1.0)
                )
            },
            "performance_metrics": {
                "enabled": ConfigSchema(
                    field="metrics_config.performance_metrics.enabled",
                    type=bool,
                    default=True
                ),
                "track_generation_time": ConfigSchema(
                    field="metrics_config.performance_metrics.track_generation_time",
                    type=bool,
                    default=True
                ),
                "track_memory_usage": ConfigSchema(
                    field="metrics_config.performance_metrics.track_memory_usage",
                    type=bool,
                    default=True
                ),
                "track_efficiency": ConfigSchema(
                    field="metrics_config.performance_metrics.track_efficiency",
                    type=bool,
                    default=True
                ),
                "memory_sample_rate": ConfigSchema(
                    field="metrics_config.performance_metrics.memory_sample_rate",
                    type=int,
                    default=1000,
                    range=(100, None)
                ),
                "time_window_ms": ConfigSchema(
                    field="metrics_config.performance_metrics.time_window_ms",
                    type=int,
                    default=5000,
                    range=(1000, None)
                )
            },
            "structure_analysis": {
                "enabled": ConfigSchema(
                    field="metrics_config.structure_analysis.enabled",
                    type=bool,
                    default=True
                ),
                "track_length_metrics": ConfigSchema(
                    field="metrics_config.structure_analysis.track_length_metrics",
                    type=bool,
                    default=True
                ),
                "track_whitespace": ConfigSchema(
                    field="metrics_config.structure_analysis.track_whitespace",
                    type=bool,
                    default=True
                ),
                "min_section_length": ConfigSchema(
                    field="metrics_config.structure_analysis.min_section_length",
                    type=int,
                    default=10,
                    range=(1, None)
                ),
                "max_section_length": ConfigSchema(
                    field="metrics_config.structure_analysis.max_section_length",
                    type=int,
                    default=1000,
                    range=(1, None)
                ),
                "whitespace_ratio_threshold": ConfigSchema(
                    field="metrics_config.structure_analysis.whitespace_ratio_threshold",
                    type=float,
                    default=0.3,
                    range=(0.0, 1.0)
                )
            },
            "relationship_context": {
                "enabled": ConfigSchema(
                    field="metrics_config.relationship_context.enabled",
                    type=bool,
                    default=True
                ),
                "max_context_history": ConfigSchema(
                    field="metrics_config.relationship_context.max_context_history",
                    type=int,
                    default=100,
                    range=(1, None)
                ),
                "reference_decay_rate": ConfigSchema(
                    field="metrics_config.relationship_context.reference_decay_rate",
                    type=float,
                    default=0.95,
                    range=(0.0, 1.0)
                ),
                "temporal_window_size": ConfigSchema(
                    field="metrics_config.relationship_context.temporal_window_size",
                    type=int,
                    default=10,
                    range=(1, None)
                ),
                "min_relationship_strength": ConfigSchema(
                    field="metrics_config.relationship_context.min_relationship_strength",
                    type=float,
                    default=0.1,
                    range=(0.0, 1.0)
                )
            },
            "helper_methods": {
                "indentation_analysis": {
                    "enabled": ConfigSchema(
                        field="metrics_config.helper_methods.indentation_analysis.enabled",
                        type=bool,
                        default=True
                    ),
                    "max_indent_level": ConfigSchema(
                        field="metrics_config.helper_methods.indentation_analysis.max_indent_level",
                        type=int,
                        default=8,
                        range=(1, None)
                    ),
                    "tab_size": ConfigSchema(
                        field="metrics_config.helper_methods.indentation_analysis.tab_size",
                        type=int,
                        default=4,
                        range=(1, None)
                    ),
                    "mixed_indent_warning": ConfigSchema(
                        field="metrics_config.helper_methods.indentation_analysis.mixed_indent_warning",
                        type=bool,
                        default=True
                    )
                },
                "optimization_analysis": {
                    "enabled": ConfigSchema(
                        field="metrics_config.helper_methods.optimization_analysis.enabled",
                        type=bool,
                        default=True
                    ),
                    "performance_threshold": ConfigSchema(
                        field="metrics_config.helper_methods.optimization_analysis.performance_threshold",
                        type=float,
                        default=0.8,
                        range=(0.0, 1.0)
                    ),
                    "memory_threshold": ConfigSchema(
                        field="metrics_config.helper_methods.optimization_analysis.memory_threshold",
                        type=float,
                        default=0.9,
                        range=(0.0, 1.0)
                    ),
                    "optimization_levels": ConfigSchema(
                        field="metrics_config.helper_methods.optimization_analysis.optimization_levels",
                        type=list,
                        default=["basic", "moderate", "aggressive"],
                        validator=lambda x: all(isinstance(level, str) and level in ["basic", "moderate", "aggressive"] for level in x)
                    )
                }
            }
        }

    @staticmethod
    def _get_dreamer_config_schema() -> Dict[str, ConfigSchema]:
        """Return the dreamer_config schema for the Dreamer class."""
        return {
            "dream_max_events_per_cycle": ConfigSchema(field="dreamer_config.dream_max_events_per_cycle", type=int, default=5, range=(1, 100)),
            "dream_novelty_weight": ConfigSchema(field="dreamer_config.dream_novelty_weight", type=float, default=1.0, range=(0.0, 10.0)),
            "dream_confidence_weight": ConfigSchema(field="dreamer_config.dream_confidence_weight", type=float, default=0.0, range=(0.0, 10.0)),
            "dream_selection_strategy": ConfigSchema(field="dreamer_config.dream_selection_strategy", type=str, default="top", validator=lambda x: x in ["top", "random"]),
            "dream_noise_level": ConfigSchema(field="dreamer_config.dream_noise_level", type=float, default=0.2, range=(0.0, 1.0)),
        }

    @staticmethod
    def _get_motivator_config_schema() -> Dict[str, ConfigSchema]:
        """Return the motivator_config schema for the Motivator class."""
        return {
            "decay_rate": ConfigSchema(
                field="motivator_config.decay_rate",
                type=float,
                default=0.01,
                range=(0.0, 1.0)
            ),
            "min_priority": ConfigSchema(
                field="motivator_config.min_priority",
                type=float,
                default=0.1,
                range=(0.0, 1.0)
            ),
            "completion_threshold": ConfigSchema(
                field="motivator_config.completion_threshold",
                type=float,
                default=0.95,
                range=(0.0, 1.0)
            ),
            "reevaluation_interval": ConfigSchema(
                field="motivator_config.reevaluation_interval",
                type=int,
                default=60,
                range=(1, 3600)
            ),
            "memory_check_enabled": ConfigSchema(
                field="motivator_config.memory_check_enabled",
                type=bool,
                default=True
            ),
            "irrelevance_threshold": ConfigSchema(
                field="motivator_config.irrelevance_threshold",
                type=float,
                default=0.2,
                range=(0.0, 1.0)
            ),
            "completion_drive": ConfigSchema(
                field="motivator_config.completion_drive",
                type=float,
                default=0.7,
                range=(0.0, 1.0)
            ),
            "novelty_drive": ConfigSchema(
                field="motivator_config.novelty_drive",
                type=float,
                default=0.2,
                range=(0.0, 1.0)
            ),
        }
