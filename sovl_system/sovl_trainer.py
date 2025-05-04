from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Type
import torch
import time
import math
from collections import deque, defaultdict
import traceback
from sovl_scaffold import ScaffoldProvider
from sovl_error import ErrorManager, ConfigurationError
from sovl_config import ConfigManager
from sovl_processor import MetadataProcessor
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_logger import Logger, LoggerConfig
from transformers import get_linear_schedule_with_warmup
from sovl_engram import LoraAdapterManager
from sovl_io import JSONLLoader, StreamingJSONLoader, ScribeJSONLBatchLoader
import threading
import gc
from sovl_utils import validate_metadata_fields, repair_metadata, get_metadata_value, collate_tensor_batch, move_batch_to_device
from sovl_dreamer import Dreamer

# TrainingConfig: holds all training-related configuration groups loaded from ConfigManager.
@dataclass
class TrainingConfig:
    # LoggingConfig: parameters for log file management and training error handling.
    @dataclass
    class LoggingConfig:
        """Logging configuration parameters."""
        log_file: str = "training_logs.jsonl"
        max_size_mb: int = 10
        compress_old: bool = True
        max_in_memory_logs: int = 100
        rotation_count: int = 5
        max_log_age_days: int = 30
        prune_interval_hours: int = 48
        memory_threshold_mb: int = 100
        gpu_memory_threshold: float = 0.85
        error_cooldown: float = 1.0
        max_recent_errors: int = 100
        logging_verbosity: str = "info"
        error_handling_config: Dict[str, Any] = field(default_factory=lambda: {
            "max_history_per_error": 10,
            "critical_threshold": 5,
            "warning_threshold": 10,
            "retry_attempts": 3,
            "retry_delay": 1.0,
            "memory_recovery_attempts": 3,
            "memory_recovery_delay": 1.0
        })
    
    # OptimizerConfig: hyperparameters for the training optimizer.
    @dataclass
    class OptimizerConfig:
        """Optimizer configuration parameters."""
        type: str = "adamw"
        learning_rate: float = 2e-5
        weight_decay: float = 0.01
        grad_accum_steps: int = 4
        max_grad_norm: float = 1.0
        
    # SchedulerConfig: settings for learning rate schedule (warmup, total steps, etc.).
    @dataclass
    class SchedulerConfig:
        """Learning rate scheduler configuration."""
        type: str = "linear"
        warmup_steps: int = 0
        total_steps: int = 100000
        cosine_min_lr: float = 1e-6
        warmup_ratio: float = 0.1
        
    # MemoryConfig: batch size, sequence length, mixed precision, and patience.
    @dataclass
    class MemoryConfig:
        """Memory and batch configuration."""
        batch_size: int = 1
        max_seq_length: int = 512
        use_amp: bool = False
        max_patience: int = 2
        
    # TrainingParams: core loop parameters like epochs, validation frequency, and checkpoints.
    @dataclass
    class TrainingParams:
        """Core training parameters."""
        max_epochs: int = 3
        validate_every_n_steps: int = 500
        checkpoint_interval: int = 5000
        checkpoint_path: str = "checkpoints/sovl_trainer"
        dropout_rate: float = 0.1
        metrics_to_track: List[str] = field(default_factory=lambda: ["loss", "accuracy", "confidence"])
    
    # Initialize TrainingConfig by pulling values from ConfigManager and validating.
    def __init__(self, config_manager: ConfigManager):
        """Initialize training configuration from ConfigManager."""
        self.config_manager = config_manager
        self.optimizer = self.OptimizerConfig()
        self.scheduler = self.SchedulerConfig()
        self.memory = self.MemoryConfig()
        self.params = self.TrainingParams()
        self.logging = self.LoggingConfig()
        self._lock = threading.Lock()  # Initialize thread lock
        self._load_config()
        
    # Load and validate all config sections (optimizer, scheduler, memory, params, logging).
    def _load_config(self) -> None:
        """Load and validate training configuration."""
        try:
            if not self.config_manager.has_section("training"):
                raise ConfigurationError("Missing 'training' section in configuration")
            
            # Load optimizer config
            self.optimizer.type = self.config_manager.get("training.optimizer.type", "adamw")
            self.optimizer.learning_rate = self.config_manager.get("training.learning_rate", 2e-5)
            self.optimizer.weight_decay = self.config_manager.get("training.weight_decay", 0.01)
            self.optimizer.grad_accum_steps = self.config_manager.get("training.grad_accum_steps", 4)
            self.optimizer.max_grad_norm = self.config_manager.get("training.max_grad_norm", 1.0)
            
            # Load scheduler config
            self.scheduler.type = self.config_manager.get("training.scheduler_type", "linear")
            self.scheduler.warmup_steps = self.config_manager.get("training.warmup_steps", 0)
            self.scheduler.total_steps = self.config_manager.get("training.total_steps", 100000)
            self.scheduler.cosine_min_lr = self.config_manager.get("training.cosine_min_lr", 1e-6)
            self.scheduler.warmup_ratio = self.config_manager.get("training.warmup_ratio", 0.1)
            
            # Load memory config
            self.memory.batch_size = self.config_manager.get("training.batch_size", 1)
            self.memory.max_seq_length = self.config_manager.get("training.max_seq_length", 512)
            self.memory.use_amp = self.config_manager.get("training.use_amp", False)
            self.memory.max_patience = self.config_manager.get("training.max_patience", 2)
            
            # Load training params
            self.params.max_epochs = self.config_manager.get("training.max_epochs", 3)
            self.params.validate_every_n_steps = self.config_manager.get("training.validate_every_n_steps", 500)
            self.params.checkpoint_interval = self.config_manager.get("training.checkpoint_interval", 5000)
            self.params.checkpoint_path = self.config_manager.get("training.checkpoint_path", "checkpoints/sovl_trainer")
            self.params.dropout_rate = self.config_manager.get("training.dropout_rate", 0.1)
            self.params.metrics_to_track = self.config_manager.get(
                "training.metrics_to_track",
                ["loss", "accuracy", "confidence"]
            )
            
            # Load logging config
            self.logging.log_file = self.config_manager.get("training.logging.log_file", "training_logs.jsonl")
            self.logging.max_size_mb = self.config_manager.get("training.logging.max_size_mb", 10)
            self.logging.compress_old = self.config_manager.get("training.logging.compress_old", True)
            self.logging.max_in_memory_logs = self.config_manager.get("training.logging.max_in_memory_logs", 100)
            self.logging.rotation_count = self.config_manager.get("training.logging.rotation_count", 5)
            self.logging.max_log_age_days = self.config_manager.get("training.logging.max_log_age_days", 30)
            self.logging.prune_interval_hours = self.config_manager.get("training.logging.prune_interval_hours", 48)
            self.logging.memory_threshold_mb = self.config_manager.get("training.logging.memory_threshold_mb", 100)
            self.logging.gpu_memory_threshold = self.config_manager.get("training.logging.gpu_memory_threshold", 0.85)
            self.logging.error_cooldown = self.config_manager.get("training.logging.error_cooldown", 1.0)
            self.logging.max_recent_errors = self.config_manager.get("training.logging.max_recent_errors", 100)
            self.logging.logging_verbosity = self.config_manager.get("training.logging.logging_verbosity", "info")
            self.logging.error_handling_config = self.config_manager.get(
                "training.logging.error_handling_config",
                {
                    "max_history_per_error": 10,
                    "critical_threshold": 5,
                    "warning_threshold": 10,
                    "retry_attempts": 3,
                    "retry_delay": 1.0,
                    "memory_recovery_attempts": 3,
                    "memory_recovery_delay": 1.0
                }
            )
            
            # Validate configurations
            self._validate()
            
        except KeyError as e:
            raise ConfigurationError(f"Missing configuration key: {str(e)}")
        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration type: {str(e)}")
        except Exception as e:
            raise ConfigurationError(
                f"Unexpected error loading training configuration: {str(e)}",
                traceback.format_exc()
            )
            
    # Assert correctness of loaded configuration (value ranges, types, etc.).
    def _validate(self) -> None:
        """Validate configuration parameters."""
        try:
            # Validate optimizer config
            assert self.optimizer.learning_rate > 0, "Learning rate must be positive"
            assert self.optimizer.weight_decay >= 0, "Weight decay must be non-negative"
            assert self.optimizer.grad_accum_steps >= 1, "Gradient accumulation steps must be at least 1"
            assert self.optimizer.max_grad_norm > 0, "Max gradient norm must be positive"
            
            # Validate scheduler config
            assert self.scheduler.warmup_steps >= 0, "Warmup steps must be non-negative"
            assert self.scheduler.total_steps > 0, "Total steps must be positive"
            assert self.scheduler.cosine_min_lr > 0, "Cosine min learning rate must be positive"
            assert 0 <= self.scheduler.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
            
            # Validate memory config
            assert self.memory.batch_size > 0, "Batch size must be positive"
            assert self.memory.max_seq_length > 0, "Max sequence length must be positive"
            assert isinstance(self.memory.use_amp, bool), "use_amp must be a boolean"
            assert self.memory.max_patience >= 0, "Max patience must be non-negative"
            
            # Validate training params
            assert self.params.max_epochs > 0, "Max epochs must be positive"
            assert self.params.validate_every_n_steps > 0, "Validate every n steps must be positive"
            assert self.params.checkpoint_interval > 0, "Checkpoint interval must be positive"
            assert isinstance(self.params.checkpoint_path, str), "Checkpoint path must be a string"
            assert 0 <= self.params.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
            
            # Validate logging config
            assert isinstance(self.logging.log_file, str) and self.logging.log_file.endswith(".jsonl"), "log_file must be a .jsonl file path"
            assert 0 < self.logging.max_size_mb <= 100, "max_size_mb must be between 0 and 100"
            assert isinstance(self.logging.compress_old, bool), "compress_old must be a boolean"
            assert 100 <= self.logging.max_in_memory_logs <= 10000, "max_in_memory_logs must be between 100 and 10000"
            assert 1 <= self.logging.max_log_age_days <= 365, "max_log_age_days must be between 1 and 365"
            assert 1 <= self.logging.prune_interval_hours <= 168, "prune_interval_hours must be between 1 and 168"
            assert 10 <= self.logging.memory_threshold_mb <= 1000, "memory_threshold_mb must be between 10 and 1000"
            assert 0.1 <= self.logging.gpu_memory_threshold <= 1.0, "gpu_memory_threshold must be between 0.1 and 1.0"
            assert 0.1 <= self.logging.error_cooldown <= 60.0, "error_cooldown must be between 0.1 and 60.0"
            assert 10 <= self.logging.max_recent_errors <= 1000, "max_recent_errors must be between 10 and 1000"
            
            # Validate error handling config
            error_config = self.logging.error_handling_config
            assert isinstance(error_config, dict), "error_handling_config must be a dictionary"
            required_error_keys = {"max_history_per_error", "critical_threshold", "warning_threshold"}
            assert all(key in error_config for key in required_error_keys), \
                f"error_handling_config must contain all required keys: {required_error_keys}"
            assert error_config["max_history_per_error"] > 0, "max_history_per_error must be positive"
            assert error_config["critical_threshold"] >= 0, "critical_threshold must be non-negative"
            assert error_config["warning_threshold"] >= 0, "warning_threshold must be non-negative"
            assert error_config.get("retry_attempts", 0) >= 0, "retry_attempts must be non-negative"
            assert error_config.get("retry_delay", 0) >= 0, "retry_delay must be non-negative"
            assert error_config.get("memory_recovery_attempts", 0) >= 0, "memory_recovery_attempts must be non-negative"
            assert error_config.get("memory_recovery_delay", 0) >= 0, "memory_recovery_delay must be non-negative"
            
        except AssertionError as e:
            raise ConfigurationError(f"Invalid training configuration: {str(e)}")
            
    # Update a training config key, propagate to ConfigManager, and reload settings.
    def update(self, key: str, value: Any) -> bool:
        """Update a configuration parameter."""
        with self._lock:
            try:
                # Update in config manager
                success = self.config_manager.update(f"training.{key}", value)
                # Only reload config if update succeeded, and do it inside the lock
                if success:
                    self._load_config()
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to update training configuration: {str(e)}",
                    traceback.format_exc()
                )
            return success
        
    # Retrieve a training config parameter via ConfigManager.
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration parameter."""
        try:
            return self.config_manager.get(f"training.{key}", default)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to get training configuration: {str(e)}",
                traceback.format_exc()
            )
            
    # Validate that all required keys exist in the 'training' section.
    def validate_section(self) -> bool:
        """Validate the training configuration section."""
        try:
            required_keys = [
                "learning_rate", "grad_accum_steps", "weight_decay",
                "warmup_steps", "total_steps", "max_grad_norm",
                "use_amp", "max_patience", "batch_size", "max_epochs",
                "validate_every_n_steps", "checkpoint_interval",
                "checkpoint_path", "scheduler_type", "cosine_min_lr",
                "warmup_ratio", "dropout_rate", "max_seq_length"
            ]
            
            return self.config_manager.validate_section("training", required_keys)
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to validate training configuration section: {str(e)}",
                traceback.format_exc()
            )

# TrainingWorkflowManager: orchestrates full multi-phase training cycles (train, sleep, gestation, dream).
class TrainingWorkflowManager:
    """Manages training cycles, sleep training, and gestation/dream cycles."""
    
    def __init__(self, trainer: 'SOVLTrainer', event_handler: TrainingEventHandler):
        self.trainer = trainer
        self.event_handler = event_handler
        # Safely get attributes from trainer
        self.logger = getattr(trainer, '_logger', None)
        self.state = getattr(trainer, '_training_state', None)
        self.config = getattr(trainer, 'config', None)
        self.model_manager = getattr(trainer, 'model_manager', None)  # <-- Add reference to model manager
        
        # Initialize locks for thread synchronization
        self._training_lock = threading.RLock()  # Recursive lock for training operations
        self._gestation_lock = threading.RLock()  # Recursive lock for gestation operations
        self._model_lock = threading.RLock()     # Lock for model access
        self._resource_locks = {}                # Dictionary to store locks for specific resources
        
    def _get_resource_lock(self, resource_name):
        """Get or create a lock for a specific resource."""
        if resource_name not in self._resource_locks:
            self._resource_locks[resource_name] = threading.RLock()
        return self._resource_locks[resource_name]
        
    def run_gestation_cycle(self, conversation_history: List[dict]) -> None:
        """Run gestation cycle using pre-enriched, preprocessed batches. Expects conversation_history as a list of ready-to-train batches (dicts or tensors)."""
        with self._gestation_lock:  # Ensure only one gestation cycle runs at a time
            import time
            import os
            import gc
            from datetime import datetime
            import traceback
            import torch

            # Set mode to 'gestating' at the start
            state_manager = getattr(self.trainer, 'state_manager', None)
            if state_manager and hasattr(state_manager, 'set_mode'):
                state_manager.set_mode('gestating')
            if state_manager and hasattr(state_manager, 'set_gestation_progress'):
                state_manager.set_gestation_progress(0.0)

            # Get model/scaffold/LoRA managers
            device = getattr(self.trainer, 'device', None)
            logger = getattr(self, 'logger', None)
            model_manager = getattr(self, 'model_manager', None)
            dreamer = getattr(self.trainer, 'dreamer', None)
            state = getattr(self, 'state', None)

            # Error handling config
            error_handling_cfg = None
            if hasattr(self, 'config') and self.config and hasattr(self.config, 'logging'):
                error_handling_cfg = getattr(self.config.logging, 'error_handling_config', None)
            retry_attempts = (error_handling_cfg or {}).get('retry_attempts', 1)
            retry_delay = (error_handling_cfg or {}).get('retry_delay', 0.0)

            # Validate input
            if not conversation_history or not isinstance(conversation_history, list):
                if logger:
                    logger.log_error("Invalid or missing batch list for gestation cycle.", event_type="gestation_invalid_batches")
                return

            # Validate model_manager and scaffold models with proper synchronization
            scaffold_count = 0
            lora_count = 0
            with self._get_resource_lock('model_manager'):
                if model_manager:
                    scaffold_count = len(getattr(model_manager, 'scaffold_models', []))
                    lora_count = len(getattr(model_manager, 'lora_managers', []))
            if scaffold_count == 0 or lora_count == 0:
                if logger:
                    logger.log_error(f"No scaffold models ({scaffold_count}) or LoRA managers ({lora_count}) available for gestation cycle.", event_type="gestation_scaffold_lora_missing")
                return
            if scaffold_count != lora_count:
                if logger:
                    logger.log_warning(
                        f"Mismatch between scaffold models ({scaffold_count}) and LoRA managers ({lora_count}). Using minimum count ({min(scaffold_count, lora_count)}).",
                        event_type="gestation_scaffold_lora_mismatch",
                        additional_info={"scaffold_count": scaffold_count, "lora_count": lora_count}
                    )
            min_count = min(scaffold_count, lora_count)
            successful_scaffolds = []

            # Run garbage collection before starting the loop
            gc.collect()
            torch.cuda.empty_cache()

            for idx in range(min_count):
                with self._get_resource_lock('model_manager'):
                    scaffold_model = model_manager.scaffold_models[idx]
                    lora_manager = model_manager.lora_managers[idx]
                optimizer = None
                outputs = None
                loss = None
                try:
                    last_exception = None
                    for attempt in range(retry_attempts):
                        try:
                            # Force garbage collection
                            gc.collect()
                            torch.cuda.empty_cache()
                            # Use the preprocessed batch directly
                            batch = conversation_history[idx] if idx < len(conversation_history) else None
                            if batch is None:
                                raise ValueError(f"No batch provided for scaffold {idx}")
                            break
                        except Exception as e:
                            last_exception = e
                            if logger:
                                logger.log_error(
                                    f"Error accessing preprocessed batch (scaffold {idx}, attempt {attempt+1}): {str(e)}",
                                    error_type="gestation_batch_error",
                                    stack_trace=traceback.format_exc(),
                                    additional_info={"scaffold_index": idx, "attempt": attempt+1}
                                )
                            if attempt < retry_attempts - 1 and retry_delay > 0:
                                time.sleep(retry_delay)
                    else:
                        raise RuntimeError(f"Gestation batch access failed for scaffold {idx} after {retry_attempts} attempts: {last_exception}")

                    # Retry logic for training step with synchronized access to the scaffold model
                    last_exception = None
                    for attempt in range(retry_attempts):
                        try:
                            if optimizer is not None:
                                del optimizer
                            if outputs is not None:
                                del outputs
                            if loss is not None:
                                del loss
                            gc.collect()
                            torch.cuda.empty_cache()
                            with self._model_lock:
                                optimizer = torch.optim.AdamW(
                                    lora_manager.lora_parameters(scaffold_model),
                                    lr=self.config.optimizer.learning_rate
                                )
                                scaffold_model.train()
                                optimizer.zero_grad()
                                outputs = scaffold_model(**batch)
                                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                                loss.backward()
                                optimizer.step()
                            loss_value = loss.item()
                            if logger:
                                logger.log_info(
                                    f"Gestation LoRA training step complete (scaffold {idx}). Loss: {loss_value}",
                                    event_type="gestation_lora_train",
                                    additional_info={"scaffold_index": idx}
                                )
                            break
                        except Exception as e:
                            last_exception = e
                            if logger:
                                logger.log_error(
                                    f"Error during gestation LoRA training step (scaffold {idx}, attempt {attempt+1}): {str(e)}",
                                    error_type="gestation_lora_training_error",
                                    stack_trace=traceback.format_exc(),
                                    additional_info={"scaffold_index": idx, "attempt": attempt+1}
                                )
                            if attempt < retry_attempts - 1 and retry_delay > 0:
                                time.sleep(retry_delay)
                    else:
                        raise RuntimeError(f"Gestation LoRA training failed for scaffold {idx} after {retry_attempts} attempts: {last_exception}")

                    # Save LoRA weights as long-term memory, with rollback on failure
                    lora_dir = "lora_checkpoints"
                    os.makedirs(lora_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    lora_path = os.path.join(lora_dir, f"lora_{idx}_{timestamp}.pt")
                    try:
                        if optimizer is not None:
                            del optimizer
                        if outputs is not None:
                            del outputs
                        if loss is not None:
                            del loss
                        gc.collect()
                        torch.cuda.empty_cache()
                        with self._get_resource_lock('lora_manager'):
                            lora_manager.save_lora_weights(scaffold_model, lora_path)
                        with self._get_resource_lock('model_manager'):
                            if model_manager:
                                model_manager.set_active_lora_checkpoint(lora_path)
                        if logger:
                            logger.log_info(
                                f"LoRA weights saved after gestation for scaffold {idx} to {lora_path}",
                                event_type="gestation_lora_save",
                                additional_info={"scaffold_index": idx}
                            )
                    except Exception as e:
                        if os.path.exists(lora_path):
                            os.remove(lora_path)
                        if logger:
                            logger.log_error(
                                f"Error saving LoRA weights after gestation (scaffold {idx}): {str(e)}",
                                error_type="gestation_lora_save_error",
                                stack_trace=traceback.format_exc(),
                                additional_info={"scaffold_index": idx}
                            )
                        raise RuntimeError(f"Failed to save LoRA weights for scaffold {idx}: {e}")
                    successful_scaffolds.append(idx)
                finally:
                    if optimizer is not None:
                        del optimizer
                    if outputs is not None:
                        del outputs
                    if loss is not None:
                        del loss
                    gc.collect()
                    torch.cuda.empty_cache()
            if len(successful_scaffolds) != min_count:
                raise RuntimeError("Not all scaffolds completed gestation training successfully")
            if state is not None and hasattr(state, 'update_after_gestation'):
                try:
                    with self._get_resource_lock('state'):
                        state.update_after_gestation()
                    if logger:
                        logger.log_info("SOVL state updated after gestation.", event_type="gestation_state_update")
                except Exception as e:
                    if logger:
                        logger.log_error(
                            f"Error updating SOVL state after gestation: {str(e)}",
                            error_type="gestation_state_update_error",
                            stack_trace=traceback.format_exc()
                        )
            if dreamer is not None:
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    with self._get_resource_lock('dreamer'):
                        dreamer.run_dream_cycle()
                    if logger:
                        logger.log_info("Dreamer cycle completed after gestation.", event_type="gestation_dreamer")
                except Exception as e:
                    if logger:
                        logger.log_error(
                            f"Error during Dreamer cycle: {str(e)}",
                            error_type="gestation_dreamer_error",
                            stack_trace=traceback.format_exc()
                        )
            gc.collect()
            torch.cuda.empty_cache()
            if state_manager and hasattr(state_manager, 'set_mode'):
                state_manager.set_mode('online')
            if state_manager and hasattr(state_manager, 'set_gestation_progress'):
                state_manager.set_gestation_progress(1.0)

    def run_dream_cycle(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Run a dream cycle with proper synchronization."""
        with self._get_resource_lock('dreamer'):  # Lock access to dreamer
            # Dream cycle implementation
            dreamer = getattr(self.trainer, 'dreamer', None)
            if dreamer is None:
                return
                
            try:
                # Implementation here
                pass
            except Exception as e:
                logger = getattr(self, 'logger', None)
                if logger:
                    logger.log_error(
                        f"Error during manual dream cycle: {str(e)}",
                        error_type="manual_dream_cycle_error",
                        stack_trace=traceback.format_exc()
                    )

# SOVLTrainer: top-level interface tying config, managers, and execution logic for end-to-end training.
class SOVLTrainer:
    """Manages training operations and memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager,
        model: torch.nn.Module,
        device: torch.device,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
            model: The PyTorch model to be trained (MUST be pre-wrapped/adapted by ModelManager)
            device: The torch.device to run training on
            tokenizer: The tokenizer associated with the model
        """
        self._config_manager = config_manager
        self._logger = logger
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.batch_preparer = BatchPreparer(tokenizer, device)
        
        # Initialize training config with logging settings
        self.config = TrainingConfig(config_manager)
        
        # Configure logger with training-specific settings
        logger_config = LoggerConfig(
            log_file=self.config.logging.log_file,
            max_size_mb=self.config.logging.max_size_mb,
            compress_old=self.config.logging.compress_old,
            max_in_memory_logs=self.config.logging.max_in_memory_logs,
            rotation_count=self.config.logging.rotation_count,
            max_log_age_days=self.config.logging.max_log_age_days,
            prune_interval_hours=self.config.logging.prune_interval_hours,
            memory_threshold_mb=self.config.logging.memory_threshold_mb,
            gpu_memory_threshold=self.config.logging.gpu_memory_threshold,
            error_cooldown=self.config.logging.error_cooldown,
            max_recent_errors=self.config.logging.max_recent_errors,
            error_handling_config=self.config.logging.error_handling_config
        )
        self._logger.configure(logger_config)
        
        # Initialize metadata processor for processing training data
        self.metadata_processor = TrainingSampleEnricher(config_manager, logger)
        
        # Initialize ErrorManager for TrainingManager
        from sovl_error import ErrorManager
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,  # Optionally provide a state tracker if needed
            config_manager=config_manager,
            error_cooldown=self.config.logging.error_cooldown
        )
        
        # Log initialization
        self._logger.log_event(
            event_type="trainer_initialization",
            message="SOVLTrainer initialized with TrainingManager and logging configuration",
            level="info",
            additional_info={
                "config": {
                    "optimizer": self.config.optimizer.__dict__,
                    "scheduler": self.config.scheduler.__dict__,
                    "memory": self.config.memory.__dict__,
                    "params": self.config.params.__dict__,
                    "logging": self.config.logging.__dict__
                },
                "device": str(self.device),
                "model_class": self.model.__class__.__name__
            }
        )

    def train_on_scribe_journal(self, scribe_path: str, batch_size: int = 32, default_weight: float = 1.0, epochs: int = 1):
        """
        Train on batches from the scribe journal, tracking all 'memory' fields used for training.
        Returns the set of trained memory strings.
        """
        from sovl_io import ScribeJSONLBatchLoader
        trained_memories = set()
        loader = ScribeJSONLBatchLoader(scribe_path, batch_size, default_weight)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.optimizer.learning_rate)
        for epoch in range(epochs):
            for batch_texts, batch_weights in loader:
                # Track all memory strings in this batch
                for entry in batch_texts:
                    if 'memory' in entry:
                        trained_memories.add(entry['memory'])
                # Prepare batch for model (assume batch_preparer handles 'memory' field)
                batch = self.batch_preparer.prepare(batch_texts)
                batch = move_batch_to_device(batch, self.device)
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                # Apply per-sample weighting if needed (not shown here)
                loss.backward()
                optimizer.step()
        return trained_memories

@dataclass
class InterpretationConfig:
    """Configuration for metadata interpretation rules."""
    # Content weighting
    complexity_weight: float = 0.3
    novelty_weight: float = 0.2
    quality_weight: float = 0.3
    temporal_weight: float = 0.2
    
    # Learning rate adjustments
    base_learning_rate: float = 2e-5
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-4
    complexity_lr_factor: float = 0.8
    
    # Sample importance thresholds
    min_importance: float = 0.1
    max_importance: float = 5.0
    importance_decay: float = 0.95
    
    # Quality thresholds
    min_quality_score: float = 0.3
    confidence_threshold: float = 0.7
    token_diversity_threshold: float = 0.4
    
    # Temporal factors
    recency_half_life: float = 86400  # 24 hours in seconds
    max_age_factor: float = 0.5
    
    # Attention mechanisms
    attention_scale: float = 1.0
    max_attention_boost: float = 2.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0 <= self.complexity_weight <= 1
        assert 0 <= self.novelty_weight <= 1
        assert 0 <= self.quality_weight <= 1
        assert 0 <= self.temporal_weight <= 1
        assert math.isclose(
            sum([self.complexity_weight, self.novelty_weight, 
                 self.quality_weight, self.temporal_weight]), 
            1.0, 
            rel_tol=1e-9
        )
        assert self.min_learning_rate <= self.base_learning_rate <= self.max_learning_rate
        assert 0 < self.importance_decay <= 1
        assert 0 <= self.min_quality_score <= 1
        assert 0 <= self.confidence_threshold <= 1

class ContentType(Enum):
    """Enumeration of content types for specialized handling."""
    NATURAL_LANGUAGE = "natural_language"
    CODE = "code"
    MIXED = "mixed"
    CONVERSATION = "conversation"
    SYSTEM = "system"

class MetadataInterpreter:
    """Interprets metadata to inform training parameters and sample importance."""
    
    def __init__(self, config: Optional[InterpretationConfig] = None, config_manager: Optional[object] = None, logger: Optional[object] = None):
        """Initialize the metadata interpreter.
        Args:
            config: Configuration for interpretation rules
            config_manager: Optional ConfigManager for dynamic config access
            logger: Optional Logger for debug/info logging
        """
        self.config = config or InterpretationConfig()
        self.config.validate()
        self.config_manager = config_manager
        self.logger = logger
        # Track interpretation history
        self._interpretation_history = defaultdict(list)
        self._sample_stats = defaultdict(float)
        # Load event type weights from config_manager if available
        if config_manager is not None:
            self.event_type_weights = config_manager.get_section("metadata_weighting") or {}
            if self.logger:
                self.logger.log_info(
                    f"Loaded event_type_weights: {self.event_type_weights}",
                    event_type="metadata_weighting_init"
                )
        else:
            self.event_type_weights = {}
            
    def interpret_metadata(
        self, 
        metadata: Dict[str, Any],
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Interpret metadata and return training parameters.
        Args:
            metadata: The metadata to interpret
            current_time: Current timestamp (defaults to time.time())
        Returns:
            Dictionary of interpreted parameters
        """
        current_time = current_time or time.time()
        # Extract core metrics
        content_metrics = self._extract_content_metrics(metadata)
        quality_score = self._calculate_quality_score(metadata)
        temporal_factor = self._calculate_temporal_factor(metadata, current_time)
        novelty_score = self._assess_novelty(metadata)
        # Calculate importance weight
        importance_weight = self._calculate_importance_weight(
            content_metrics["complexity"],
            quality_score,
            temporal_factor,
            novelty_score
        )
        # Apply event_type weighting if available
        event_type = metadata.get("event_type")
        event_type_weight = 1.0
        if event_type and self.event_type_weights:
            pre_weight = importance_weight
            event_type_weight = self.event_type_weights.get(event_type, 1.0)
            importance_weight *= event_type_weight
            if self.logger:
                self.logger.log_debug(
                    f"Applied event_type_weight={event_type_weight:.3f} to importance: {pre_weight:.3f} -> {importance_weight:.3f}",
                    event_type="event_type_weight_application"
                )
        # Debug log of interpretation input/output
        if self.logger:
            self.logger.log_debug(
                f"Interpreting metadata for event_type={event_type} | "
                f"complexity={content_metrics['complexity']:.3f}, quality={quality_score:.3f}, "
                f"temporal={temporal_factor:.3f}, novelty={novelty_score:.3f} | "
                f"event_type_weight={event_type_weight:.3f}, importance={importance_weight:.3f}",
                event_type="metadata_interpretation"
            )
        # Determine learning parameter adjustments
        learning_params = self._determine_learning_parameters(
            content_metrics,
            quality_score,
            importance_weight
        )
        # Calculate attention focus
        attention_params = self._calculate_attention_parameters(
            content_metrics,
            metadata
        )
        # Update interpretation history
        self._update_history(metadata, importance_weight, quality_score)
        return {
            "sample_importance": importance_weight,
            "learning_params": learning_params,
            "attention_params": attention_params,
            "quality_metrics": {
                "overall_quality": quality_score,
                "novelty": novelty_score,
                "temporal_relevance": temporal_factor
            },
            "content_metrics": content_metrics,
            "event_type_weight": event_type_weight,
            "event_type": event_type
        }
        
    def _extract_content_metrics(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize content-related metrics."""
        # Get content metrics from metadata
        metrics = metadata.get("content_metrics", {})
        token_stats = metadata.get("token_stats", {})
        
        # Calculate complexity score
        word_count = metrics.get("word_count", 0)
        sentence_count = metrics.get("sentence_count", 0)
        avg_word_length = metrics.get("avg_word_length", 0)
        token_diversity = token_stats.get("token_diversity", 0)
        
        complexity = min(1.0, (
            (math.log2(word_count + 1) / 10) * 0.3 +
            (avg_word_length / 10) * 0.3 +
            token_diversity * 0.4
        ))
        
        # Determine content type
        content_type = self._determine_content_type(metadata)
        
        # Calculate structure score
        structure_metrics = metadata.get("structure_metrics", {})
        structure_score = self._calculate_structure_score(structure_metrics)
        
        return {
            "complexity": complexity,
            "token_diversity": token_diversity,
            "structure_score": structure_score,
            "content_type": content_type,
            "size_factor": min(1.0, math.log2(word_count + 1) / 10)
        }

    def _determine_content_type(self, metadata: Dict[str, Any]) -> ContentType:
        """Determine the type of content from metadata."""
        quality_metrics = metadata.get("quality_metrics", {})
        
        has_code = quality_metrics.get("has_code", False)
        has_conversation = quality_metrics.get("has_conversation", False)
        is_system = metadata.get("is_system_message", False)
        
        if is_system:
            return ContentType.SYSTEM
        elif has_code and not has_conversation:
            return ContentType.CODE
        elif has_conversation and not has_code:
            return ContentType.CONVERSATION
        elif has_code and has_conversation:
            return ContentType.MIXED
        else:
            return ContentType.NATURAL_LANGUAGE

    def _calculate_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate overall quality score from metadata metrics."""
        quality_metrics = metadata.get("quality_metrics", {})
        performance_metrics = metadata.get("performance_metrics", {})
        
        # Get base metrics
        confidence = metadata.get("confidence_score", 0.0)
        token_quality = quality_metrics.get("token_quality", 0.0)
        error_rate = quality_metrics.get("error_rate", 0.0)
        
        # Performance factors
        efficiency = performance_metrics.get("efficiency", {})
        performance_score = float(efficiency.get("optimization_level", "low") == "high")
        
        # Calculate weighted quality score
        quality_score = (
            confidence * 0.4 +
            token_quality * 0.3 +
            (1 - error_rate) * 0.2 +
            performance_score * 0.1
        )
        
        return max(self.config.min_quality_score, min(1.0, quality_score))

    def _calculate_temporal_factor(
        self, 
        metadata: Dict[str, Any],
        current_time: float
    ) -> float:
        """Calculate temporal relevance factor."""
        timestamp = metadata.get("timestamp_unix", current_time)
        age = current_time - timestamp
        
        # Apply exponential decay based on age
        temporal_factor = math.exp(-age / self.config.recency_half_life)
        
        # Ensure minimum temporal factor
        return max(self.config.max_age_factor, temporal_factor)

    def _assess_novelty(self, metadata: Dict[str, Any]) -> float:
        """Assess content novelty based on metadata."""
        content_metrics = metadata.get("content_metrics", {})
        token_stats = metadata.get("token_stats", {})
        
        # Get novelty indicators
        token_diversity = token_stats.get("token_diversity", 0.0)
        pattern_uniqueness = token_stats.get("pattern_stats", {}).get("bigram_diversity", 0.0)
        
        # Calculate novelty score
        novelty_score = (
            token_diversity * 0.6 +
            pattern_uniqueness * 0.4
        )
        
        return min(1.0, novelty_score)

    def _calculate_importance_weight(
        self,
        complexity: float,
        quality: float,
        temporal_factor: float,
        novelty: float
    ) -> float:
        """Calculate overall sample importance weight."""
        importance = (
            complexity * self.config.complexity_weight +
            quality * self.config.quality_weight +
            temporal_factor * self.config.temporal_weight +
            novelty * self.config.novelty_weight
        )
        # Log calculation breakdown
        if self.logger:
            self.logger.log_debug(
                f"Importance calculation: complexity={complexity:.3f}*{self.config.complexity_weight} + "
                f"quality={quality:.3f}*{self.config.quality_weight} + "
                f"temporal={temporal_factor:.3f}*{self.config.temporal_weight} + "
                f"novelty={novelty:.3f}*{self.config.novelty_weight} = {importance:.3f}",
                event_type="importance_weight_calc"
            )
        # Apply bounds
        clipped = False
        if importance < self.config.min_importance or importance > self.config.max_importance:
            clipped = True
        importance = max(
            self.config.min_importance,
            min(self.config.max_importance, importance)
        )
        if clipped and self.logger:
            self.logger.log_warning(
                f"Importance weight clipped to bounds: {importance:.3f}",
                event_type="importance_weight_clipped"
            )
        return importance

    def _determine_learning_parameters(
        self,
        content_metrics: Dict[str, float],
        quality_score: float,
        importance_weight: float
    ) -> Dict[str, float]:
        """Determine learning rate and related parameters."""
        # Base learning rate adjustment
        complexity_factor = math.pow(
            self.config.complexity_lr_factor,
            content_metrics["complexity"]
        )
        
        learning_rate = self.config.base_learning_rate * complexity_factor
        
        # Bound learning rate
        learning_rate = max(
            self.config.min_learning_rate,
            min(self.config.max_learning_rate, learning_rate)
        )
        
        # Additional training parameters
        return {
            "learning_rate": learning_rate,
            "weight_decay": 0.01 * quality_score,  # Adjust regularization based on quality
            "dropout": 0.1 + (1 - quality_score) * 0.2,  # More dropout for lower quality
            "gradient_scale": importance_weight
        }

    def _calculate_attention_parameters(
        self,
        content_metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate attention mechanism parameters."""
        structure_metrics = metadata.get("structure_metrics", {})
        
        # Base attention scale
        attention_scale = self.config.attention_scale
        
        # Boost attention for complex content
        if content_metrics["complexity"] > 0.7:
            attention_scale *= min(
                self.config.max_attention_boost,
                1 + content_metrics["complexity"]
            )
            
        # Adjust for content type
        content_type = content_metrics["content_type"]
        if content_type == ContentType.CODE:
            attention_scale *= 1.2  # Boost attention for code
        elif content_type == ContentType.MIXED:
            attention_scale *= 1.1  # Slight boost for mixed content
            
        return {
            "scale": attention_scale,
            "dropout": 0.1,  # Base attention dropout
            "head_importance": [1.0] * 12  # Per-head importance (if applicable)
        }

    def _calculate_structure_score(
        self,
        structure_metrics: Dict[str, Any]
    ) -> float:
        """Calculate structural quality score."""
        if not structure_metrics:
            return 0.5
            
        # Extract metrics
        length_metrics = structure_metrics.get("length_metrics", {})
        whitespace_metrics = structure_metrics.get("whitespace_metrics", {})
        
        # Calculate structure score components
        length_score = min(1.0, length_metrics.get("avg_line_length", 0) / 100)
        whitespace_ratio = whitespace_metrics.get("whitespace_ratio", 0)
        indentation_score = min(1.0, whitespace_metrics.get("indentation_levels", 0) / 5)
        
        # Combine scores
        return (length_score * 0.4 + 
                whitespace_ratio * 0.3 +
                indentation_score * 0.3)

    def _update_history(
        self,
        metadata: Dict[str, Any],
        importance: float,
        quality: float
    ) -> None:
        """Update interpretation history for tracking."""
        sample_id = metadata.get("sample_id", str(time.time()))
        
        self._interpretation_history[sample_id].append({
            "timestamp": time.time(),
            "importance": importance,
            "quality": quality
        })
        
        # Update running statistics
        self._sample_stats["total_samples"] += 1
        self._sample_stats["avg_importance"] = (
            (self._sample_stats["avg_importance"] * 
             (self._sample_stats["total_samples"] - 1) +
             importance) / self._sample_stats["total_samples"]
        )
        self._sample_stats["avg_quality"] = (
            (self._sample_stats["avg_quality"] * 
             (self._sample_stats["total_samples"] - 1) +
             quality) / self._sample_stats["total_samples"]
        )

    def get_interpretation_stats(self) -> Dict[str, Any]:
        """Get current interpretation statistics."""
        return {
            "total_samples": self._sample_stats["total_samples"],
            "average_importance": self._sample_stats["avg_importance"],
            "average_quality": self._sample_stats["avg_quality"],
            "history_size": sum(len(h) for h in self._interpretation_history.values())
        }

def move_batch_to_device(batch: dict, device: "torch.device") -> dict:
    """
    Move all tensors in a batch dict to the specified device.
    Args:
        batch: Dict[str, torch.Tensor]
        device: torch.device
    Returns:
        Dict[str, torch.Tensor] (all tensors moved to device)
    """
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

class TrainingSampleEnricher:
    """
    Enriches and validates training samples by adding/correcting metadata fields.
    Wraps the enrich_and_validate method from MetadataProcessor.
    """
    def __init__(self, config_manager: 'ConfigManager', logger: 'Logger'):
        from sovl_processor import MetadataProcessor
        self.processor = MetadataProcessor(config_manager, logger)
        self.logger = logger

    def enrich(self, sample: dict, source_metadata: dict = None, session_id: str = None) -> dict:
        # Use enrich_and_validate for metadata enrichment
        try:
            _, enriched = self.processor.enrich_and_validate(
                origin="TrainingPipeline",
                event_type="training_sample",
                event_data=sample,
                source_metadata=source_metadata or {},
                session_id=session_id
            )
            return enriched
        except Exception as e:
            self.logger.log_warning(f"Metadata enrichment failed: {e}", event_type="metadata_enrichment_warning")
            return sample

class BatchPreparer:
    """
    Tokenizes, collates, and moves a batch of samples to the correct device.
    Uses the tokenizer and batch/tensor helpers.
    """
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def prepare(self, samples: list) -> dict:
        # Tokenize and collate samples
        batch = []
        for sample in samples:
            # Assume sample has 'prompt' and 'response' fields
            encoded = self.tokenizer(
                sample["prompt"],
                text_target=sample.get("response"),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=getattr(self.tokenizer, 'model_max_length', 512)
            )
            # Flatten batch dimension
            encoded = {k: v.squeeze(0) if v.dim() == 2 and v.size(0) == 1 else v for k, v in encoded.items()}
            batch.append(encoded)
        return collate_tensor_batch(batch, self.device)
