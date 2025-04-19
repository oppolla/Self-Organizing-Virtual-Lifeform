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
                "model_name": ConfigSchema(required=True, type=str),
                "base_model_path": ConfigSchema(required=False, type=str, default=None),
                "scaffold_model_name": ConfigSchema(required=False, type=str, default=None),
                "scaffold_model_path": ConfigSchema(required=False, type=str, default=None),
                "cross_attn_layers": ConfigSchema(required=False, type=list, default=[5, 7]),
                "use_dynamic_layers": ConfigSchema(required=False, type=bool, default=False),
                "layer_selection_mode": ConfigSchema(required=False, type=str, default="balanced", choices=["balanced", "random", "fixed"]),
                "custom_layers": ConfigSchema(required=False, type=list, default=None),
                "valid_split_ratio": ConfigSchema(required=False, type=float, default=0.2, min=0.0, max=1.0),
                "random_seed": ConfigSchema(required=False, type=int, default=42, min=0),
                "quantization": ConfigSchema(required=False, type=str, default="fp16", choices=["fp16", "int8", "none"]),
                "hidden_size": ConfigSchema(required=False, type=int, default=768, min=1),
                "num_heads": ConfigSchema(required=False, type=int, default=12, min=1),
                "gradient_checkpointing": ConfigSchema(required=False, type=bool, default=True),
                "initializer_range": ConfigSchema(required=False, type=float, default=0.02, min=0.0),
                "migration_mode": ConfigSchema(required=False, type=bool, default=True),
                "device": ConfigSchema(required=False, type=str, default="cuda", choices=["cuda", "cpu"])
            },
            "controls_config": {
                "enable_scaffold": ConfigSchema(required=False, type=bool, default=True),
                "scaffold_weight_cap": ConfigSchema(required=False, type=float, default=0.9, min=0.0, max=1.0),
                "scaffold_unk_id": ConfigSchema(required=False, type=int, default=0, min=0),
                "enable_cross_attention": ConfigSchema(required=False, type=bool, default=True),
                "enable_dynamic_cross_attention": ConfigSchema(required=False, type=bool, default=False),
                "injection_strategy": ConfigSchema(required=False, type=str, default="sequential", choices=["sequential", "parallel"])
            },
            "training_config": {
                "learning_rate": ConfigSchema(required=True, type=float, default=2e-5, min=0.0),
                "train_epochs": ConfigSchema(required=False, type=int, default=3, min=1),
                "batch_size": ConfigSchema(required=True, type=int, default=2, min=1),
                "max_seq_length": ConfigSchema(required=False, type=int, default=512, min=1),
                "sigmoid_scale": ConfigSchema(required=False, type=float, default=0.5, min=0.0),
                "sigmoid_shift": ConfigSchema(required=False, type=float, default=5.0, min=0.0),
                "lifecycle_capacity_factor": ConfigSchema(required=False, type=float, default=0.01, min=0.0),
                "lifecycle_curve": ConfigSchema(required=False, type=str, default="sigmoid_linear", choices=["sigmoid_linear", "exponential"]),
                "accumulation_steps": ConfigSchema(required=False, type=int, default=4, min=1),
                "exposure_gain_eager": ConfigSchema(required=False, type=int, default=3, min=1),
                "exposure_gain_default": ConfigSchema(required=False, type=int, default=2, min=1),
                "max_patience": ConfigSchema(required=False, type=int, default=2, min=0),
                "dry_run": ConfigSchema(required=False, type=bool, default=False),
                "dry_run_params": ConfigSchema(required=False, type=dict, default=None),
                "weight_decay": ConfigSchema(required=False, type=float, default=0.01, min=0.0),
                "max_grad_norm": ConfigSchema(required=False, type=float, default=1.0, min=0.0),
                "use_amp": ConfigSchema(required=False, type=bool, default=True),
                "checkpoint_interval": ConfigSchema(required=False, type=int, default=1000, min=1),
                "scheduler_type": ConfigSchema(required=False, type=str, default="linear", choices=["linear", "cosine", "constant"]),
                "cosine_min_lr": ConfigSchema(required=False, type=float, default=1e-6, min=0.0),
                "warmup_ratio": ConfigSchema(required=False, type=float, default=0.1, min=0.0, max=1.0),
                "warmup_steps": ConfigSchema(required=False, type=int, default=0, min=0),
                "total_steps": ConfigSchema(required=False, type=int, default=100000, min=1),
                "validate_every_n_steps": ConfigSchema(required=False, type=int, default=100, min=1),
                "checkpoint_path": ConfigSchema(required=False, type=str, default="checkpoints/sovl_trainer"),
                "dropout_rate": ConfigSchema(required=False, type=float, default=0.1, min=0.0, max=1.0),
                "metrics_to_track": ConfigSchema(required=False, type=list, default=["loss", "accuracy", "confidence"]),
                "enable_gestation": ConfigSchema(required=False, type=bool, default=True),
                "enable_sleep_training": ConfigSchema(required=False, type=bool, default=True),
                "enable_lifecycle_weighting": ConfigSchema(required=False, type=bool, default=True),
                "sleep_conf_threshold": ConfigSchema(required=False, type=float, default=0.7, min=0.0, max=1.0),
                "sleep_log_min": ConfigSchema(required=False, type=int, default=10, min=1),
                "dream_memory_weight": ConfigSchema(required=False, type=float, default=0.1, min=0.0, max=1.0),
                "enable_dreaming": ConfigSchema(required=False, type=bool, default=True),
                "repetition_n": ConfigSchema(required=False, type=int, default=3, min=2),
                "dream_noise_scale": ConfigSchema(required=False, type=float, default=0.05, min=0.0),
                "dream_prompt_weight": ConfigSchema(required=False, type=float, default=0.5, min=0.0, max=1.0),
                "dream_novelty_boost": ConfigSchema(required=False, type=float, default=0.03, min=0.0),
                "dream_memory_decay": ConfigSchema(required=False, type=float, default=0.95, min=0.0, max=1.0),
                "dream_prune_threshold": ConfigSchema(required=False, type=float, default=0.1, min=0.0, max=1.0),
                "temp_melancholy_noise": ConfigSchema(required=False, type=float, default=0.02, min=0.0),
                "enable_prompt_driven_dreams": ConfigSchema(required=False, type=bool, default=True),
                "dream_swing_var": ConfigSchema(required=False, type=float, default=0.1, min=0.0),
                "dream_lifecycle_delta": ConfigSchema(required=False, type=float, default=0.1, min=0.0),
                "dream_temperament_on": ConfigSchema(required=False, type=bool, default=True),
                "confidence_history_maxlen": ConfigSchema(required=False, type=int, default=5, min=1),
                "temperament_history_maxlen": ConfigSchema(required=False, type=int, default=5, min=1),
                "memory_threshold": ConfigSchema(required=False, type=float, default=0.85, min=0.0, max=1.0),
                "memory_decay_rate": ConfigSchema(required=False, type=float, default=0.95, min=0.0, max=1.0),
                "use_scaffold_memory": ConfigSchema(required=False, type=bool, default=True),
                "use_token_map_memory": ConfigSchema(required=False, type=bool, default=True),
                "scaffold_weight": ConfigSchema(required=False, type=float, default=1.0, min=0.0),
                "max_dream_memory_mb": ConfigSchema(required=False, type=int, default=256, min=1),
                "dream_memory_maxlen": ConfigSchema(required=False, type=int, default=10, min=1)
            },
            "dynamic_weighting": {
                "min_weight": ConfigSchema(required=False, type=float, default=0.0, min=0.0),
                "max_weight": ConfigSchema(required=False, type=float, default=1.0, min=0.0),
                "weight_decay": ConfigSchema(required=False, type=float, default=0.01, min=0.0),
                "momentum": ConfigSchema(required=False, type=float, default=0.9, min=0.0, max=1.0),
                "history_size": ConfigSchema(required=False, type=int, default=10, min=1),
                "enable_dynamic_scaling": ConfigSchema(required=False, type=bool, default=True),
                "weight_curves": ConfigSchema(required=False, type=list, default=["linear", "sigmoid_linear"])
            },
            "preprocessing": {
                "remove_special_chars": ConfigSchema(required=False, type=bool, default=True),
                "lowercase": ConfigSchema(required=False, type=bool, default=True),
                "remove_extra_spaces": ConfigSchema(required=False, type=bool, default=True),
                "max_length": ConfigSchema(required=False, type=int, default=512, min=1)
            },
            "augmentation": {
                "synonym_replacement_prob": ConfigSchema(required=False, type=float, default=0.3, min=0.0, max=1.0),
                "word_dropout_prob": ConfigSchema(required=False, type=float, default=0.1, min=0.0, max=1.0),
                "max_augmentations": ConfigSchema(required=False, type=int, default=3, min=0)
            },
            "hardware": {
                "enable_cuda": ConfigSchema(required=False, type=bool, default=True),
                "memory_query_interval": ConfigSchema(required=False, type=float, default=0.1, min=0.0),
                "mock_memory_total_mb": ConfigSchema(required=False, type=float, default=8192.0, min=0.0)
            },
            "lora_config": {
                "lora_rank": ConfigSchema(required=False, type=int, default=8, min=1),
                "lora_alpha": ConfigSchema(required=False, type=int, default=16, min=1),
                "lora_dropout": ConfigSchema(required=False, type=float, default=0.1, min=0.0, max=1.0),
                "lora_target_modules": ConfigSchema(required=False, type=list, default=["c_attn", "c_proj", "c_fc"])
            },
            "curiosity_config": {
                "enable_curiosity": ConfigSchema(required=False, type=bool, default=True),
                "attention_weight": ConfigSchema(required=False, type=float, default=0.5, min=0.0, max=1.0),
                "queue_maxlen": ConfigSchema(required=False, type=int, default=10, min=1),
                "novelty_history_maxlen": ConfigSchema(required=False, type=int, default=20, min=1),
                "decay_rate": ConfigSchema(required=False, type=float, default=0.9, min=0.0, max=1.0),
                "question_timeout": ConfigSchema(required=False, type=float, default=3600.0, min=0.0),
                "novelty_threshold_spontaneous": ConfigSchema(required=False, type=float, default=0.9, min=0.0, max=1.0),
                "novelty_threshold_response": ConfigSchema(required=False, type=float, default=0.8, min=0.0, max=1.0),
                "pressure_threshold": ConfigSchema(required=False, type=float, default=0.7, min=0.0, max=1.0),
                "pressure_drop": ConfigSchema(required=False, type=float, default=0.3, min=0.0, max=1.0),
                "silence_threshold": ConfigSchema(required=False, type=float, default=20.0, min=0.0),
                "question_cooldown": ConfigSchema(required=False, type=float, default=60.0, min=0.0),
                "weight_ignorance": ConfigSchema(required=False, type=float, default=0.7, min=0.0, max=1.0),
                "weight_novelty": ConfigSchema(required=False, type=float, default=0.3, min=0.0, max=1.0),
                "max_new_tokens": ConfigSchema(required=False, type=int, default=8, min=1),
                "base_temperature": ConfigSchema(required=False, type=float, default=1.1, min=0.0),
                "temperament_influence": ConfigSchema(required=False, type=float, default=0.4, min=0.0, max=1.0),
                "top_k": ConfigSchema(required=False, type=int, default=30, min=1)
            },
            "cross_attn_config": {
                "memory_weight": ConfigSchema(required=False, type=float, default=0.5, min=0.0, max=1.0)
            },
            "logging_config": {
                "log_dir": ConfigSchema(required=False, type=str, default="logs"),
                "log_file": ConfigSchema(required=False, type=str, default="sovl_logs.jsonl"),
                "log_level": ConfigSchema(required=False, type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
                "max_log_size_mb": ConfigSchema(required=False, type=int, default=10, min=1),
                "backup_count": ConfigSchema(required=False, type=int, default=5, min=0)
            },
            "error_config": {
                "error_cooldown": ConfigSchema(required=False, type=float, default=1.0, min=0.0),
                "warning_threshold": ConfigSchema(required=False, type=float, default=3.0, min=0.0),
                "error_threshold": ConfigSchema(required=False, type=float, default=5.0, min=0.0),
                "critical_threshold": ConfigSchema(required=False, type=float, default=10.0, min=0.0)
            },
            "generation_config": {
                "temperature": ConfigSchema(required=False, type=float, default=0.7, min=0.0),
                "top_p": ConfigSchema(required=False, type=float, default=0.9, min=0.0, max=1.0)
            },
            "data_config": {
                "batch_size": ConfigSchema(required=False, type=int, default=1, min=1),
                "max_retries": ConfigSchema(required=False, type=int, default=3, min=0)
            },
            "memory_config": {
                "max_memory_mb": ConfigSchema(required=False, type=int, default=512, min=1),
                "garbage_collection_threshold": ConfigSchema(required=False, type=float, default=0.7, min=0.0, max=1.0)
            },
            "state_config": {
                "max_history": ConfigSchema(required=False, type=int, default=100, min=1),
                "state_file": ConfigSchema(required=False, type=str, default="sovl_state.json")
            },
            "temperament_config": {
                "mood_influence": ConfigSchema(required=False, type=float, default=0.5, min=0.0, max=1.0),
                "history_maxlen": ConfigSchema(required=False, type=int, default=5, min=1)
            },
            "confidence_config": {
                "history_maxlen": ConfigSchema(required=False, type=int, default=5, min=1),
                "weight": ConfigSchema(required=False, type=float, default=0.5, min=0.0, max=1.0)
            }
        }
