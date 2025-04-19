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
                "model_name": ConfigSchema(field="core_config.model_name", type=str, required=True),  # Name of the primary model (e.g., "gpt2", "bert"), required for model initialization
                "base_model_path": ConfigSchema(field="core_config.base_model_path", type=str, default=None, nullable=True),  # Path to the primary model's pretrained weights; None uses default model hub
                "scaffold_model_name": ConfigSchema(field="core_config.scaffold_model_name", type=str, default=None, nullable=True),  # Name of the scaffold model for auxiliary tasks; None disables scaffold
                "scaffold_model_path": ConfigSchema(field="core_config.scaffold_model_path", type=str, default=None, nullable=True),  # Path to scaffold model's pretrained weights; None uses default hub
                "cross_attn_layers": ConfigSchema(field="core_config.cross_attn_layers", type=list, default=[5, 7]),  # List of layer indices for cross-attention integration; affects model architecture
                "use_dynamic_layers": ConfigSchema(field="core_config.use_dynamic_layers", type=bool, default=False),  # If True, dynamically selects cross-attention layers during training
                "layer_selection_mode": ConfigSchema(field="core_config.layer_selection_mode", type=str, default="balanced", validator=lambda x: x in ["balanced", "random", "fixed"]),  # Method for selecting cross-attention layers: balanced (even spread), random, or fixed
                "custom_layers": ConfigSchema(field="core_config.custom_layers", type=list, default=None, nullable=True),  # Custom list of layer indices for cross-attention; used when layer_selection_mode is "fixed"
                "valid_split_ratio": ConfigSchema(field="core_config.valid_split_ratio", type=float, default=0.2, range=(0.0, 1.0)),  # Fraction of data used for validation split; 0.2 means 20% validation
                "random_seed": ConfigSchema(field="core_config.random_seed", type=int, default=42, range=(0, 2**32)),  # Seed for random operations to ensure reproducibility
                "quantization": ConfigSchema(field="core_config.quantization", type=str, default="fp16", validator=lambda x: x in ["fp16", "int8", "none"]),  # Model quantization type: fp16 (half-precision), int8 (8-bit integer), or none
                "hidden_size": ConfigSchema(field="core_config.hidden_size", type=int, default=768, range=(1, None)),  # Size of the model's hidden layers; affects model capacity
                "num_heads": ConfigSchema(field="core_config.num_heads", type=int, default=12, range=(1, None)),  # Number of attention heads in transformer layers; impacts attention mechanism
                "gradient_checkpointing": ConfigSchema(field="core_config.gradient_checkpointing", type=bool, default=True),  # If True, uses gradient checkpointing to reduce memory usage during training
                "initializer_range": ConfigSchema(field="core_config.initializer_range", type=float, default=0.02, range=(0.0, None)),  # Standard deviation for model weight initialization; affects training stability
                "migration_mode": ConfigSchema(field="core_config.migration_mode", type=bool, default=True),  # If True, enables model weight migration for compatibility with pretrained models
                "device": ConfigSchema(field="core_config.device", type=str, default="cuda", validator=lambda x: x in ["cuda", "cpu"]),  # Hardware device for model execution: cuda (GPU) or cpu
            },
            "controls_config": {
                "enable_scaffold": ConfigSchema(field="controls_config.enable_scaffold", type=bool, default=True),  # If True, enables scaffold model for auxiliary guidance during training
                "scaffold_weight_cap": ConfigSchema(field="controls_config.scaffold_weight_cap", type=float, default=0.9, range=(0.0, 1.0)),  # Maximum weight for scaffold model influence; caps its contribution
                "scaffold_unk_id": ConfigSchema(field="controls_config.scaffold_unk_id", type=int, default=0, range=(0, None)),  # Token ID for unknown tokens in scaffold model; 0 is typically the default
                "enable_cross_attention": ConfigSchema(field="controls_config.enable_cross_attention", type=bool, default=True),  # If True, enables cross-attention between primary and scaffold models
                "enable_dynamic_cross_attention": ConfigSchema(field="controls_config.enable_dynamic_cross_attention", type=bool, default=False),  # If True, dynamically adjusts cross-attention weights during training
                "injection_strategy": ConfigSchema(field="controls_config.injection_strategy", type=str, default="sequential", validator=lambda x: x in ["sequential", "parallel"]),  # Method for injecting scaffold outputs: sequential or parallel processing
            },
            "training_config": {
                "learning_rate": ConfigSchema(field="training_config.learning_rate", type=float, default=2e-5, range=(0.0, None), required=True),  # Learning rate for optimizer; controls training step size
                "train_epochs": ConfigSchema(field="training_config.train_epochs", type=int, default=3, range=(1, None)),  # Number of training epochs; higher values increase training time
                "batch_size": ConfigSchema(field="training_config.batch_size", type=int, default=2, range=(1, None), required=True),  # Number of samples per training batch; affects memory and speed
                "max_seq_length": ConfigSchema(field="training_config.max_seq_length", type=int, default=512, range=(1, None)),  # Maximum sequence length for input tokens; longer sequences use more memory
                "sigmoid_scale": ConfigSchema(field="training_config.sigmoid_scale", type=float, default=0.5, range=(0.0, None)),  # Scaling factor for sigmoid lifecycle weighting; adjusts training dynamics
                "sigmoid_shift": ConfigSchema(field="training_config.sigmoid_shift", type=float, default=5.0, range=(0.0, None)),  # Shift parameter for sigmoid lifecycle curve; affects weight progression
                "lifecycle_capacity_factor": ConfigSchema(field="training_config.lifecycle_capacity_factor", type=float, default=0.01, range=(0.0, None)),  # Factor for model capacity adjustment during lifecycle; impacts resource allocation
                "lifecycle_curve": ConfigSchema(field="training_config.lifecycle_curve", type=str, default="sigmoid_linear", validator=lambda x: x in ["sigmoid_linear", "exponential"]),  # Curve type for lifecycle weighting: sigmoid_linear or exponential
                "accumulation_steps": ConfigSchema(field="training_config.accumulation_steps", type=int, default=4, range=(1, None)),  # Number of gradient accumulation steps; simulates larger batch sizes
                "exposure_gain_eager": ConfigSchema(field="training_config.exposure_gain_eager", type=int, default=3, range=(1, None)),  # Gain for eager exposure in training; boosts early learning for certain samples
                "exposure_gain_default": ConfigSchema(field="training_config.exposure_gain_default", type=int, default=2, range=(1, None)),  # Default gain for sample exposure; controls learning emphasis
                "max_patience": ConfigSchema(field="training_config.max_patience", type=int, default=2, range=(0, None)),  # Maximum epochs without improvement before early stopping
                "dry_run": ConfigSchema(field="training_config.dry_run", type=bool, default=False),  # If True, runs training in simulation mode without weight updates
                "dry_run_params": ConfigSchema(field="training_config.dry_run_params", type=dict, default=None, nullable=True),  # Parameters for dry run mode; None uses defaults
                "weight_decay": ConfigSchema(field="training_config.weight_decay", type=float, default=0.01, range=(0.0, None)),  # Weight decay for regularization; prevents overfitting
                "max_grad_norm": ConfigSchema(field="training_config.max_grad_norm", type=float, default=1.0, range=(0.0, None)),  # Maximum gradient norm for clipping; stabilizes training
                "use_amp": ConfigSchema(field="training_config.use_amp", type=bool, default=True),  # If True, uses automatic mixed precision for faster training
                "checkpoint_interval": ConfigSchema(field="training_config.checkpoint_interval", type=int, default=1000, range=(1, None)),  # Steps between model checkpoint saves
                "scheduler_type": ConfigSchema(field="training_config.scheduler_type", type=str, default="linear", validator=lambda x: x in ["linear", "cosine", "constant"]),  # Learning rate scheduler type: linear, cosine, or constant
                "cosine_min_lr": ConfigSchema(field="training_config.cosine_min_lr", type=float, default=1e-6, range=(0.0, None)),  # Minimum learning rate for cosine scheduler
                "warmup_ratio": ConfigSchema(field="training_config.warmup_ratio", type=float, default=0.1, range=(0.0, 1.0)),  # Fraction of training steps for learning rate warmup
                "warmup_steps": ConfigSchema(field="training_config.warmup_steps", type=int, default=0, range=(0, None)),  # Number of warmup steps; overrides warmup_ratio if set
                "total_steps": ConfigSchema(field="training_config.total_steps", type=int, default=100000, range=(1, None)),  # Total training steps; defines training duration
                "validate_every_n_steps": ConfigSchema(field="training_config.validate_every_n_steps", type=int, default=100, range=(1, None)),  # Steps between validation runs
                "checkpoint_path": ConfigSchema(field="training_config.checkpoint_path", type=str, default="checkpoints/sovl_trainer"),  # Directory for saving model checkpoints
                "dropout_rate": ConfigSchema(field="training_config.dropout_rate", type=float, default=0.1, range=(0.0, 1.0)),  # Dropout rate for regularization; prevents overfitting
                "metrics_to_track": ConfigSchema(field="training_config.metrics_to_track", type=list, default=["loss", "accuracy", "confidence"]),  # Metrics to monitor during training
                "enable_gestation": ConfigSchema(field="training_config.enable_gestation", type=bool, default=True),  # If True, enables gestation phase for gradual model warmup (assumed proprietary)
                "enable_sleep_training": ConfigSchema(field="training_config.enable_sleep_training", type=bool, default=True),  # If True, enables sleep training for memory consolidation (assumed proprietary)
                "enable_lifecycle_weighting": ConfigSchema(field="training_config.enable_lifecycle_weighting", type=bool, default=True),  # If True, applies lifecycle-based sample weighting during training
                "sleep_conf_threshold": ConfigSchema(field="training_config.sleep_conf_threshold", type=float, default=0.7, range=(0.0, 1.0)),  # Confidence threshold for sleep training; filters samples
                "sleep_log_min": ConfigSchema(field="training_config.sleep_log_min", type=int, default=10, range=(1, None)),  # Minimum log entries for sleep training; ensures sufficient data
                "dream_memory_weight": ConfigSchema(field="training_config.dream_memory_weight", type=float, default=0.1, range=(0.0, 1.0)),  # Weight for dream memory in training; affects replay influence
                "enable_dreaming": ConfigSchema(field="training_config.enable_dreaming", type=bool, default=True),  # If True, enables dreaming mechanism for synthetic sample generation (assumed proprietary)
                "repetition_n": ConfigSchema(field="training_config.repetition_n", type=int, default=3, range=(2, None)),  # Number of repetitions for dream samples; enhances learning
                "dream_noise_scale": ConfigSchema(field="training_config.dream_noise_scale", type=float, default=0.05, range=(0.0, None)),  # Scale of noise added to dream samples; increases variability
                "dream_prompt_weight": ConfigSchema(field="training_config.dream_prompt_weight", type=float, default=0.5, range=(0.0, 1.0)),  # Weight for prompt-driven dream generation; balances real vs. synthetic data
                "dream_novelty_boost": ConfigSchema(field="training_config.dream_novelty_boost", type=float, default=0.03, range=(0.0, None)),  # Boost for novel dream samples; encourages exploration
                "dream_memory_decay": ConfigSchema(field="training_config.dream_memory_decay", type=float, default=0.95, range=(0.0, 1.0)),  # Decay rate for dream memory; controls retention
                "dream_prune_threshold": ConfigSchema(field="training_config.dream_prune_threshold", type=float, default=0.1, range=(0.0, 1.0)),  # Threshold for pruning low-value dream samples
                "temp_melancholy_noise": ConfigSchema(field="training_config.temp_melancholy_noise", type=float, default=0.02, range=(0.0, None)),  # Noise scale for temperament-driven dreaming; adds emotional variation (assumed proprietary)
                "enable_prompt_driven_dreams": ConfigSchema(field="training_config.enable_prompt_driven_dreams", type=bool, default=True),  # If True, generates dreams based on input prompts
                "dream_swing_var": ConfigSchema(field="training_config.dream_swing_var", type=float, default=0.1, range=(0.0, None)),  # Variance for dream sample swings; controls diversity
                "dream_lifecycle_delta": ConfigSchema(field="training_config.dream_lifecycle_delta", type=float, default=0.1, range=(0.0, None)),  # Delta for lifecycle adjustments in dreaming; affects progression
                "dream_temperament_on": ConfigSchema(field="training_config.dream_temperament_on", type=bool, default=True),  # If True, applies temperament influence to dreaming (assumed proprietary)
                "confidence_history_maxlen": ConfigSchema(field="training_config.confidence_history_maxlen", type=int, default=5, range=(1, None)),  # Maximum length of confidence history for training
                "temperament_history_maxlen": ConfigSchema(field="training_config.temperament_history_maxlen", type=int, default=5, range=(1, None)),  # Maximum length of temperament history for training
                "memory_threshold": ConfigSchema(field="training_config.memory_threshold", type=float, default=0.85, range=(0.0, 1.0)),  # Threshold for memory retention in training; filters low-value memories
                "memory_decay_rate": ConfigSchema(field="training_config.memory_decay_rate", type=float, default=0.95, range=(0.0, 1.0)),  # Decay rate for training memories; controls forgetting
                "use_scaffold_memory": ConfigSchema(field="training_config.use_scaffold_memory", type=bool, default=True),  # If True, uses scaffold model memory for training
                "use_token_map_memory": ConfigSchema(field="training_config.use_token_map_memory", type=bool, default=True),  # If True, uses token map memory for training; enhances context
                "scaffold_weight": ConfigSchema(field="training_config.scaffold_weight", type=float, default=1.0, range=(0.0, None)),  # Weight for scaffold model in training; balances influence
                "max_dream_memory_mb": ConfigSchema(field="training_config.max_dream_memory_mb", type=int, default=256, range=(1, None)),  # Maximum memory (MB) for dream storage
                "dream_memory_maxlen": ConfigSchema(field="training_config.dream_memory_maxlen", type=int, default=10, range=(1, None)),  # Maximum number of dream samples stored
            },
            "dynamic_weighting": {
                "min_weight": ConfigSchema(field="dynamic_weighting.min_weight", type=float, default=0.0, range=(0.0, None)),  # Minimum weight for dynamic weighting; sets lower bound
                "max_weight": ConfigSchema(field="dynamic_weighting.max_weight", type=float, default=1.0, range=(0.0, None)),  # Maximum weight for dynamic weighting; sets upper bound
                "weight_decay": ConfigSchema(field="dynamic_weighting.weight_decay", type=float, default=0.01, range=(0.0, None)),  # Decay rate for dynamic weights; prevents overfitting
                "momentum": ConfigSchema(field="dynamic_weighting.momentum", type=float, default=0.9, range=(0.0, 1.0)),  # Momentum for weight updates; smooths adjustments
                "history_size": ConfigSchema(field="dynamic_weighting.history_size", type=int, default=10, range=(1, None)),  # Size of history for dynamic weighting calculations
                "enable_dynamic_scaling": ConfigSchema(field="dynamic_weighting.enable_dynamic_scaling", type=bool, default=True),  # If True, enables dynamic scaling of weights
                "weight_curves": ConfigSchema(field="dynamic_weighting.weight_curves", type=list, default=["linear", "sigmoid_linear"]),  # List of curves for weight adjustments; defines scaling behavior
            },
            "preprocessing": {
                "remove_special_chars": ConfigSchema(field="preprocessing.remove_special_chars", type=bool, default=True),  # If True, removes special characters from input text
                "lowercase": ConfigSchema(field="preprocessing.lowercase", type=bool, default=True),  # If True, converts input text to lowercase
                "remove_extra_spaces": ConfigSchema(field="preprocessing.remove_extra_spaces", type=bool, default=True),  # If True, removes extra whitespace from input text
                "max_length": ConfigSchema(field="preprocessing.max_length", type=int, default=512, range=(1, None)),  # Maximum length for preprocessed text; truncates longer inputs
            },
            "augmentation": {
                "synonym_replacement_prob": ConfigSchema(field="augmentation.synonym_replacement_prob", type=float, default=0.3, range=(0.0, 1.0)),  # Probability of replacing words with synonyms; enhances data variety
                "word_dropout_prob": ConfigSchema(field="augmentation.word_dropout_prob", type=float, default=0.1, range=(0.0, 1.0)),  # Probability of dropping words; simulates noise
                "max_augmentations": ConfigSchema(field="augmentation.max_augmentations", type=int, default=3, range=(0, None)),  # Maximum number of augmentations per sample
            },
            "hardware": {
                "enable_cuda": ConfigSchema(field="hardware.enable_cuda", type=bool, default=True),  # If True, enables CUDA for GPU acceleration
                "memory_query_interval": ConfigSchema(field="hardware.memory_query_interval", type=float, default=0.1, range=(0.0, None)),  # Interval (seconds) for querying memory usage
                "mock_memory_total_mb": ConfigSchema(field="hardware.mock_memory_total_mb", type=float, default=8192.0, range=(0.0, None)),  # Simulated total memory (MB) for testing
            },
            "lora_config": {
                "lora_rank": ConfigSchema(field="lora_config.lora_rank", type=int, default=8, range=(1, None)),  # Rank for LoRA adaptation; controls parameter efficiency
                "lora_alpha": ConfigSchema(field="lora_config.lora_alpha", type=int, default=16, range=(1, None)),  # Scaling factor for LoRA; adjusts adaptation strength
                "lora_dropout": ConfigSchema(field="lora_config.lora_dropout", type=float, default=0.1, range=(0.0, 1.0)),  # Dropout rate for LoRA layers; prevents overfitting
                "lora_target_modules": ConfigSchema(field="lora_config.lora_target_modules", type=list, default=["c_attn", "c_proj", "c_fc"]),  # Model modules to apply LoRA; e.g., attention and feed-forward
            },
            "curiosity_config": {
                "enable_curiosity": ConfigSchema(field="curiosity_config.enable_curiosity", type=bool, default=True),  # If True, enables curiosity-driven exploration (assumed proprietary)
                "attention_weight": ConfigSchema(field="curiosity_config.attention_weight", type=float, default=0.5, range=(0.0, 1.0)),  # Weight for curiosity-driven attention; balances exploration
                "queue_maxlen": ConfigSchema(field="curiosity_config.queue_maxlen", type=int, default=10, range=(1, None)),  # Maximum length of curiosity question queue
                "novelty_history_maxlen": ConfigSchema(field="curiosity_config.novelty_history_maxlen", type=int, default=20, range=(1, None)),  # Maximum length of novelty history
                "decay_rate": ConfigSchema(field="curiosity_config.decay_rate", type=float, default=0.9, range=(0.0, 1.0)),  # Decay rate for curiosity novelty; controls forgetting
                "question_timeout": ConfigSchema(field="curiosity_config.question_timeout", type=float, default=3600.0, range=(0.0, None)),  # Timeout (seconds) for curiosity questions
                "novelty_threshold_spontaneous": ConfigSchema(field="curiosity_config.novelty_threshold_spontaneous", type=float, default=0.9, range=(0.0, 1.0)),  # Threshold for spontaneous curiosity triggers
                "novelty_threshold_response": ConfigSchema(field="curiosity_config.novelty_threshold_response", type=float, default=0.8, range=(0.0, 1.0)),  # Threshold for response-driven curiosity
                "pressure_threshold": ConfigSchema(field="curiosity_config.pressure_threshold", type=float, default=0.7, range=(0.0, 1.0)),  # Threshold for curiosity pressure; triggers exploration
                "pressure_drop": ConfigSchema(field="curiosity_config.pressure_drop", type=float, default=0.3, range=(0.0, 1.0)),  # Drop in pressure after curiosity action
                "silence_threshold": ConfigSchema(field="curiosity_config.silence_threshold", type=float, default=20.0, range=(0.0, None)),  # Silence duration (seconds) before curiosity triggers
                "question_cooldown": ConfigSchema(field="curiosity_config.question_cooldown", type=float, default=60.0, range=(0.0, None)),  # Cooldown (seconds) between curiosity questions
                "weight_ignorance": ConfigSchema(field="curiosity_config.weight_ignorance", type=float, default=0.7, range=(0.0, 1.0)),  # Weight for ignorance-driven curiosity; prioritizes unknown areas
                "weight_novelty": ConfigSchema(field="curiosity_config.weight_novelty", type=float, default=0.3, range=(0.0, 1.0)),  # Weight for novelty-driven curiosity; encourages new patterns
                "max_new_tokens": ConfigSchema(field="curiosity_config.max_new_tokens", type=int, default=8, range=(1, None)),  # Maximum new tokens for curiosity-generated outputs
                "base_temperature": ConfigSchema(field="curiosity_config.base_temperature", type=float, default=1.1, range=(0.0, None)),  # Base temperature for curiosity sampling; controls randomness
                "temperament_influence": ConfigSchema(field="curiosity_config.temperament_influence", type=float, default=0.4, range=(0.0, 1.0)),  # Influence of temperament on curiosity; adds emotional context
                "top_k": ConfigSchema(field="curiosity_config.top_k", type=int, default=30, range=(1, None)),  # Top-k sampling for curiosity outputs; limits token choices
            },
            "cross_attn_config": {
                "memory_weight": ConfigSchema(field="cross_attn_config.memory_weight", type=float, default=0.5, range=(0.0, 1.0)),  # Weight for memory in cross-attention; balances past and present
            },
            "logging_config": {
                "log_dir": ConfigSchema(field="logging_config.log_dir", type=str, default="logs"),  # Directory for storing log files
                "log_file": ConfigSchema(field="logging_config.log_file", type=str, default="sovl_logs.jsonl"),  # Name of the log file; uses JSONL format
                "log_level": ConfigSchema(field="logging_config.log_level", type=str, default="INFO", validator=lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),  # Logging verbosity level
                "max_log_size_mb": ConfigSchema(field="logging_config.max_log_size_mb", type=int, default=10, range=(1, None)),  # Maximum log file size (MB) before rotation
                "backup_count": ConfigSchema(field="logging_config.backup_count", type=int, default=5, range=(0, None)),  # Number of backup log files to keep
            },
            "error_config": {
                "error_cooldown": ConfigSchema(field="error_config.error_cooldown", type=float, default=1.0, range=(0.0, None)),  # Cooldown (seconds) after an error before retrying
                "warning_threshold": ConfigSchema(field="error_config.warning_threshold", type=float, default=3.0, range=(0.0, None)),  # Threshold for logging warnings; based on error frequency
                "error_threshold": ConfigSchema(field="error_config.error_threshold", type=float, default=5.0, range=(0.0, None)),  # Threshold for logging errors; based on severity
                "critical_threshold": ConfigSchema(field="error_config.critical_threshold", type=float, default=10.0, range=(0.0, None)),  # Threshold for critical errors; may halt execution
            },
            "generation_config": {
                "temperature": ConfigSchema(field="generation_config.temperature", type=float, default=0.7, range=(0.0, None)),  # Temperature for text generation; higher values increase randomness
                "top_p": ConfigSchema(field="generation_config.top_p", type=float, default=0.9, range=(0.0, 1.0)),  # Top-p sampling for generation; filters low-probability tokens
            },
            "data_config": {
                "batch_size": ConfigSchema(field="data_config.batch_size", type=int, default=1, range=(1, None)),  # Batch size for data loading; affects preprocessing speed
                "max_retries": ConfigSchema(field="data_config.max_retries", type=int, default=3, range=(0, None)),  # Maximum retries for failed data operations
            },
            "memory_config": {
                "max_memory_mb": ConfigSchema(field="memory_config.max_memory_mb", type=int, default=512, range=(1, None)),  # Maximum memory (MB) for general storage
                "garbage_collection_threshold": ConfigSchema(field="memory_config.garbage_collection_threshold", type=float, default=0.7, range=(0.0, 1.0)),  # Memory usage threshold for triggering garbage collection
            },
            "state_config": {
                "max_history": ConfigSchema(field="state_config.max_history", type=int, default=100, state_configrange=(1, None)),  # Maximum number of state history entries to store
                "state_file": ConfigSchema(field="state_config.state_file", type=str, default="sovl_state.json"),  # File for saving system state
            },
            "temperament_config": {
                "mood_influence": ConfigSchema(field="temperament_config.mood_influence", type=float, default=0.5, range=(0.0, 1.0)),  # Influence of mood on model behavior; adds emotional context (assumed proprietary)
                "history_maxlen": ConfigSchema(field="temperament_config.history_maxlen", type=int, default=5, range=(1, None)),  # Maximum length of mood history
            },
            "confidence_config": {
                "history_maxlen": ConfigSchema(field="confidence_config.history_maxlen", type=int, default=5, range=(1, None)),  # Maximum length of confidence history
                "weight": ConfigSchema(field="confidence_config.weight", type=float, default=0.5, range=(0.0, 1.0)),  # Weight for confidence scores in decision-making
            }
        }
