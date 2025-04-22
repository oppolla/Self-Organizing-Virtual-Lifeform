from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import time
import uuid
import math
import os
import threading
import random
from collections import deque, defaultdict
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
import traceback
from sovl_scaffold import ScaffoldProvider
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_io import ConfigurationError
from sovl_confidence import ConfidenceCalculator
from sovl_temperament import TemperamentSystem
from sovl_experience import MemoriaManager, MetadataProcessor
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_logger import Logger, LoggerConfig

# File-level: Core training module for SOVL.
# - Manages loading of training configuration, setting up optimizers and schedulers,
#   executing training and validation loops, handling checkpoints, and orchestrating
#   complex multi-phase workflows (gestation, dreaming, sleep sessions).

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
        max_in_memory_logs: int = 1000
        rotation_count: int = 5
        max_log_age_days: int = 30
        prune_interval_hours: int = 24
        memory_threshold_mb: int = 100
        gpu_memory_threshold: float = 0.85
        error_cooldown: float = 1.0
        max_recent_errors: int = 100
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
        batch_size: int = 2
        max_seq_length: int = 512
        use_amp: bool = True
        max_patience: int = 2
        
    # TrainingParams: core loop parameters like epochs, validation frequency, and checkpoints.
    @dataclass
    class TrainingParams:
        """Core training parameters."""
        max_epochs: int = 3
        validate_every_n_steps: int = 100
        checkpoint_interval: int = 1000
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
        self._load_config()
        
    # Load and validate all config sections (optimizer, scheduler, memory, params, logging).
    def _load_config(self) -> None:
        """Load and validate training configuration."""
        try:
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
            self.memory.batch_size = self.config_manager.get("training.batch_size", 2)
            self.memory.max_seq_length = self.config_manager.get("training.max_seq_length", 512)
            self.memory.use_amp = self.config_manager.get("training.use_amp", True)
            self.memory.max_patience = self.config_manager.get("training.max_patience", 2)
            
            # Load training params
            self.params.max_epochs = self.config_manager.get("training.max_epochs", 3)
            self.params.validate_every_n_steps = self.config_manager.get("training.validate_every_n_steps", 100)
            self.params.checkpoint_interval = self.config_manager.get("training.checkpoint_interval", 1000)
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
            self.logging.max_in_memory_logs = self.config_manager.get("training.logging.max_in_memory_logs", 1000)
            self.logging.rotation_count = self.config_manager.get("training.logging.rotation_count", 5)
            self.logging.max_log_age_days = self.config_manager.get("training.logging.max_log_age_days", 30)
            self.logging.prune_interval_hours = self.config_manager.get("training.logging.prune_interval_hours", 24)
            self.logging.memory_threshold_mb = self.config_manager.get("training.logging.memory_threshold_mb", 100)
            self.logging.gpu_memory_threshold = self.config_manager.get("training.logging.gpu_memory_threshold", 0.85)
            self.logging.error_cooldown = self.config_manager.get("training.logging.error_cooldown", 1.0)
            self.logging.max_recent_errors = self.config_manager.get("training.logging.max_recent_errors", 100)
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
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load training configuration: {str(e)}",
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
            assert isinstance(self.logging.error_handling_config, dict), "error_handling_config must be a dictionary"
            required_error_keys = {"max_history_per_error", "critical_threshold", "warning_threshold"}
            assert all(key in self.logging.error_handling_config for key in required_error_keys), \
                f"error_handling_config must contain all required keys: {required_error_keys}"
            
        except AssertionError as e:
            raise ConfigurationError(f"Invalid training configuration: {str(e)}")
            
    # Update a training config key, propagate to ConfigManager, and reload settings.
    def update(self, key: str, value: Any) -> bool:
        """Update a configuration parameter."""
        try:
            # Update in config manager
            success = self.config_manager.update(f"training.{key}", value)
            
            if success:
                # Reload configuration to ensure consistency
                self._load_config()
                
            return success
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to update training configuration: {str(e)}",
                traceback.format_exc()
            )
            
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

# TrainingManager: sets up optimizer and scheduler, prepares batches, and runs train/validate steps.
class TrainingManager:
    """Manages core training operations."""
    
    def __init__(self, config: TrainingConfig, model: torch.nn.Module, device: torch.device):
        """Initialize training manager."""
        self.config = config
        self.model = model
        self.device = device
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if self.config.memory.use_amp else None
        self.step_count = 0
        self.epoch_count = 0
        self.best_metrics = {}
        self.metrics_history = defaultdict(list)
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.config.optimizer.type.lower()
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_type = self.config.scheduler.type.lower()
        if scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.config.scheduler.total_steps
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler.total_steps,
                eta_min=self.config.scheduler.cosine_min_lr
            )
        elif scheduler_type == "constant":
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
    @torch.no_grad()
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training with memory optimization."""
        # Move tensors to device and truncate sequences if needed
        prepared_batch = {}
        max_length = self.config.memory.max_seq_length
        
        for key, tensor in batch.items():
            # Move to device
            tensor = tensor.to(self.device)
            
            # Truncate sequences if needed
            if key in ["input_ids", "attention_mask"] and tensor.size(1) > max_length:
                tensor = tensor[:, :max_length]
            
            prepared_batch[key] = tensor
            
        return prepared_batch
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with gradient accumulation and mixed precision."""
        self.model.train()
        metrics = {}
        
        # Prepare batch
        batch = self._prepare_batch(batch)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.config.memory.use_amp):
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.optimizer.grad_accum_steps
            
        # Backward pass with gradient scaling
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Gradient accumulation
        if (self.step_count + 1) % self.config.optimizer.grad_accum_steps == 0:
            # Clip gradients
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optimizer.max_grad_norm
            )
            
            # Update weights
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
        # Update metrics
        metrics["loss"] = loss.item() * self.config.optimizer.grad_accum_steps
        if hasattr(outputs, "logits"):
            metrics["accuracy"] = (outputs.logits.argmax(dim=-1) == batch["labels"]).float().mean().item()
            
        self.step_count += 1
        return metrics
        
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation with memory optimization."""
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = 0
        
        for batch in val_loader:
            # Prepare batch
            batch = self._prepare_batch(batch)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.config.memory.use_amp):
                outputs = self.model(**batch)
                
            # Update metrics
            metrics["loss"] += outputs.loss.item()
            if hasattr(outputs, "logits"):
                metrics["accuracy"] += (outputs.logits.argmax(-1) == batch["labels"]).float().mean().item()
            num_batches += 1
            
        # Average metrics
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        # Update best metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.best_metrics or value < self.best_metrics[metric_name]:
                self.best_metrics[metric_name] = value
                
        # Update metrics history
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
            
        return metrics
        
    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training checkpoint with metadata."""
        if path is None:
            path = self.config.params.checkpoint_path
            
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "best_metrics": self.best_metrics,
            "metrics_history": dict(self.metrics_history),
            "config": self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint with validation."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        # Load scheduler state
        if self.scheduler and checkpoint["scheduler_state"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            
        # Load scaler state
        if self.scaler and checkpoint["scaler_state"]:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
            
        # Load training state
        self.step_count = checkpoint["step_count"]
        self.epoch_count = checkpoint["epoch_count"]
        self.best_metrics = checkpoint["best_metrics"]
        self.metrics_history = defaultdict(list, checkpoint["metrics_history"])

# TrainingEventHandler: processes training lifecycle events for logging and monitoring.
class TrainingEventHandler:
    """Handles training-related events and updates system state."""
    
    def __init__(self, logger: Logger, state: TrainingState):
        self.logger = logger
        self.state = state

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float) -> None:
        """Handle training completion event."""
        self.state.update_data_exposure(data_exposure)
        self.logger.record({
            "event": "training_complete",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

    def handle_gestation_complete(self, batch_size: int, avg_loss: float) -> None:
        """Handle gestation completion event."""
        self.state.update_gestation_metrics(batch_size, avg_loss)
        self.logger.record({
            "event": "gestation_complete",
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Handle dream completion event."""
        self.state.update_dream_metrics(dream_prompt, is_novel, memory_count)
        self.logger.record({
            "event": "dream_complete",
            "dream_prompt": dream_prompt,
            "is_novel": is_novel,
            "memory_count": memory_count,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float) -> None:
        """Handle sleep training completion event."""
        self.state.update_sleep_metrics(batch_size, data_exposure)
        self.logger.record({
            "event": "sleep_train_complete",
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

# TrainingWorkflowManager: orchestrates full multi-phase training cycles (train, sleep, gestation, dream).
class TrainingWorkflowManager:
    """Manages training cycles, sleep training, and gestation/dream cycles."""
    
    def __init__(self, trainer: 'SOVLTrainer', event_handler: TrainingEventHandler):
        self.trainer = trainer
        self.event_handler = event_handler
        self.logger = trainer.logger
        self.state = trainer.state
        self.config = trainer.config

    def run_training_cycle(self, batch: List[Dict[str, Any]], scaffold_provider: Optional[ScaffoldProvider] = None) -> Tuple[float, Dict[str, Any]]:
        """Run a complete training cycle."""
        try:
            # Get batch size from memory manager
            batch_size = self.trainer.memory_manager.get_batch_size()
            
            # Run training step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=scaffold_provider,
                dry_run=False
            )
            
            # Update state and log event
            self.event_handler.handle_training_complete(
                epoch=self.state.epoch,
                avg_loss=loss,
                data_exposure=metrics.get("data_exposure", 0.0)
            )
            
            return loss, metrics
            
        except Exception as e:
            self.logger.error(f"Error in training cycle: {str(e)}")
            raise

    def run_sleep_training(self, batch: List[Dict[str, Any]]) -> None:
        """Run sleep training cycle."""
        try:
            # Get batch size from memory manager
            batch_size = self.trainer.memory_manager.get_batch_size()
            
            # Run sleep training step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=None,
                dry_run=False
            )
            
            # Update state and log event
            self.event_handler.handle_sleep_train_complete(
                batch_size=batch_size,
                data_exposure=metrics.get("data_exposure", 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in sleep training: {str(e)}")
            raise

    def run_gestation_cycle(self, conversation_history: List[Dict[str, str]]) -> None:
        """Run gestation cycle with metadata enrichment."""
        try:
            batch = []
            
            # Process conversation history with metadata if available
            if hasattr(self.trainer, 'metadata_processor'):
                # Prepare training pairs with metadata
                training_pairs = self.trainer.metadata_processor.prepare_training_pairs(conversation_history)
                
                # Convert training pairs to batch format
                for pair in training_pairs:
                    # Convert to model inputs
                    input_text = pair["input"]
                    output_text = pair["output"]
                    
                    # Tokenize inputs/outputs (assuming tokenizer is available)
                    tokenizer = getattr(self.trainer, 'tokenizer', None)
                    if tokenizer:
                        inputs = tokenizer(input_text, return_tensors="pt").to(self.trainer.device)
                        outputs = tokenizer(output_text, return_tensors="pt").to(self.trainer.device)
                        
                        # Create batch item
                        batch_item = {
                            "input_ids": inputs.input_ids[0],
                            "attention_mask": inputs.attention_mask[0],
                            "labels": outputs.input_ids[0],
                            "content": input_text + "\n" + output_text,  # Include raw content for metadata collection
                            "metadata": pair["metadata"]  # Include the collected metadata
                        }
                        batch.append(batch_item)
            else:
                # Fall back to standard processing if metadata processor not available
                # ... existing conversion from conversation history to batch ...
                pass
                
            if not batch:
                self._logger.log_warning("No training pairs generated for gestation cycle")
                return
                
            # Get batch size from memory manager
            batch_size = len(batch)
            if hasattr(self.trainer, 'memory_manager') and hasattr(self.trainer.memory_manager, 'get_batch_size'):
                batch_size = min(batch_size, self.trainer.memory_manager.get_batch_size())
            
            # Run gestation step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch[:batch_size],  # Use only batch_size items
                scaffold_provider=None,
                dry_run=False
            )
            
            # Update state and log event
            self.event_handler.handle_gestation_complete(
                batch_size=batch_size,
                avg_loss=loss
            )
            
        except Exception as e:
            self.logger.error(f"Error in gestation cycle: {str(e)}")
            raise

    def run_dream_cycle(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Run dream cycle."""
        try:
            # Update state and log event
            self.event_handler.handle_dream_complete(
                dream_prompt=dream_prompt,
                is_novel=is_novel,
                memory_count=memory_count
            )
            
        except Exception as e:
            self.logger.error(f"Error in dream cycle: {str(e)}")
            raise

# TrainingCycleManager: manages individual phases like gestation for experiential training.
class TrainingCycleManager:
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self._config_manager = config_manager
        self._logger = logger
        # Memoria manager handles experiential aspects only
        self.memoria_manager = MemoriaManager(config_manager, logger)
        # Memory managers handle RAM and GPU resources
        self.ram_manager = RAMManager(config_manager, logger)
        self.gpu_manager = GPUMemoryManager(config_manager, logger)
        
    def _prepare_gestation_batch(self, batch_size: int) -> int:
        """Prepare batch size based on memory health."""
        try:
            # Check memory health
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Adjust batch size based on memory health
            if ram_health.get("status") == "warning" or gpu_health.get("status") == "warning":
                return max(1, batch_size // 2)
            elif ram_health.get("status") == "critical" or gpu_health.get("status") == "critical":
                return 1
                
            return batch_size
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to prepare gestation batch: {str(e)}",
                error_type="batch_preparation_error",
                stack_trace=traceback.format_exc()
            )
            return 1

# SOVLTrainer: top-level interface tying config, managers, and execution logic for end-to-end training.
class SOVLTrainer:
    """Manages training operations and memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        memoria_manager: MemoriaManager,  # For experiential aspects
        ram_manager: RAMManager,          # For RAM memory management
        gpu_manager: GPUMemoryManager     # For GPU memory management
    ):
        """
        Initialize trainer.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            memoria_manager: MemoriaManager instance for experiential memory management
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
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
        self.metadata_processor = MetadataProcessor(config_manager, logger)
        
        # Initialize training state tracking
        self._training_state = {
            "current_epoch": 0,
            "total_steps": 0,
            "last_checkpoint": None,
            "best_metrics": {},
            "error_history": deque(maxlen=self.config.logging.max_recent_errors)
        }
        
        # Log initialization
        self._logger.log_event(
            event_type="trainer_initialization",
            message="SOVLTrainer initialized with logging configuration",
            level="info",
            additional_info={
                "config": {
                    "optimizer": self.config.optimizer.__dict__,
                    "scheduler": self.config.scheduler.__dict__,
                    "memory": self.config.memory.__dict__,
                    "params": self.config.params.__dict__,
                    "logging": self.config.logging.__dict__
                }
            }
        )

    def _prepare_gestation_batch(self, batch_size: int) -> int:
        """Prepare batch for gestation training with memory awareness."""
        try:
            # Check memory health
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Adjust batch size based on memory health
            if not ram_health['is_healthy'] or not gpu_health['is_healthy']:
                adjusted_size = max(1, batch_size // 2)
                self._logger.record_event(
                    event_type="batch_size_adjusted",
                    message="Batch size adjusted due to memory constraints",
                    level="warning",
                    additional_info={
                        "original_size": batch_size,
                        "adjusted_size": adjusted_size,
                        "ram_health": ram_health,
                        "gpu_health": gpu_health
                    }
                )
                return adjusted_size
                
            return batch_size
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to prepare gestation batch: {str(e)}",
                error_type="batch_preparation_error",
                stack_trace=traceback.format_exc()
            )
            return 1  # Return minimum batch size on error

    def train_with_curriculum(self, training_cycle: int, all_examples: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Train using a curriculum that gradually increases complexity.
        
        Args:
            training_cycle: Current training cycle number
            all_examples: All available training examples
            
        Returns:
            Tuple of (loss, metrics)
        """
        try:
            # Log start of training cycle
            self._logger.log_event(
                event_type="curriculum_training_start",
                message=f"Starting curriculum training cycle {training_cycle}",
                level="info",
                additional_info={
                    "cycle": training_cycle,
                    "total_examples": len(all_examples)
                }
            )
            
            # Create training manager if needed
            if not hasattr(self, 'training_manager'):
                self._logger.log_error(
                    error_msg="Training manager not initialized",
                    error_type="training_manager_error",
                    additional_info={"cycle": training_cycle}
                )
                return 0.0, {}
                
            # Select examples based on current curriculum stage
            selected_examples = self._select_curriculum_examples(training_cycle, all_examples)
            
            # No examples available
            if not selected_examples:
                self._logger.log_error(
                    error_msg="No suitable examples found for curriculum training",
                    error_type="training_data_error",
                    additional_info={"cycle": training_cycle}
                )
                return 0.0, {}
                
            # Get batch size (respecting memory constraints)
            batch_size = self._prepare_gestation_batch(self.config.memory.batch_size)
            
            # Create batch from selected examples
            batch = selected_examples[:batch_size]
            
            # Check memory health before training
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            if not ram_health['is_healthy'] or not gpu_health['is_healthy']:
                self._logger.log_event(
                    event_type="memory_warning",
                    message="Memory health check failed before training",
                    level="warning",
                    additional_info={
                        "ram_health": ram_health,
                        "gpu_health": gpu_health,
                        "batch_size": batch_size
                    }
                )
            
            # Perform training step
            try:
                # Use contextual batching for more advanced stages
                if training_cycle > 300 and hasattr(self.training_manager, 'train_with_contextual_batching'):
                    result = self.training_manager.train_with_contextual_batching(batch)
                    loss = result.get("loss", 0.0)
                    metrics = result.get("metrics", {})
                else:
                    # Standard training for early stages
                    loss, metrics = self.training_manager.train_step_with_scaffold(
                        batch=batch,
                        scaffold_provider=None,
                        dry_run=False
                    )
                    
                # Log curriculum stage
                stage = "beginner" if training_cycle < 100 else "intermediate" if training_cycle < 500 else "advanced"
                self._logger.log_event(
                    event_type="curriculum_training_complete",
                    message=f"Completed curriculum training step (stage: {stage})",
                    level="info",
                    additional_info={
                        "cycle": training_cycle,
                        "stage": stage,
                        "examples_count": len(selected_examples),
                        "batch_size": len(batch),
                        "loss": loss,
                        "metrics": metrics,
                        "ram_health": ram_health,
                        "gpu_health": gpu_health
                    }
                )
                    
                # Update training state
                self._training_state["total_steps"] += 1
                self._training_state["best_metrics"].update(metrics)
                
                return loss, metrics
                
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Failed to execute curriculum training: {str(e)}",
                    error_type="curriculum_training_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={
                        "cycle": training_cycle,
                        "batch_size": batch_size,
                        "ram_health": ram_health,
                        "gpu_health": gpu_health
                    }
                )
                return 0.0, {}
                
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Unexpected error in curriculum training: {str(e)}",
                error_type="curriculum_training_error",
                stack_trace=traceback.format_exc(),
                additional_info={"cycle": training_cycle}
            )
            return 0.0, {}
    
    def _select_curriculum_examples(self, training_cycle: int, all_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select examples with appropriate complexity for current training cycle.
        
        Args:
            training_cycle: Current training cycle
            all_examples: All available examples
            
        Returns:
            List of selected examples
        """
        # No metadata processor or examples
        if not hasattr(self, 'metadata_processor') or not all_examples:
            return all_examples
            
        # Early cycles (0-100): Focus on simpler examples with clear patterns
        if training_cycle < 100:
            # Filter for shorter, clearer examples
            filtered = []
            for ex in all_examples:
                metadata = ex.get("metadata", {})
                content_metrics = metadata.get("content_metrics", {})
                word_count = content_metrics.get("word_count", 100)
                
                # Select shorter examples (< 50 words)
                if word_count < 50:
                    filtered.append(ex)
                    
            return filtered if filtered else all_examples[:min(len(all_examples), 10)]
            
        # Middle cycles (100-500): Focus on medium complexity, diversity
        elif training_cycle < 500:
            # Select diverse examples including some complex ones
            selected = []
            categories_seen = set()
            
            # Try to get diverse examples
            for ex in all_examples:
                metadata = ex.get("metadata", {})
                content_metrics = metadata.get("content_metrics", {})
                quality_metrics = metadata.get("quality_metrics", {})
                
                # Create category signature
                word_count = content_metrics.get("word_count", 0)
                has_code = quality_metrics.get("has_code", False)
                has_question = quality_metrics.get("has_question", False)
                
                # Categorize by length and content type
                length_cat = "short" if word_count < 50 else "medium" if word_count < 150 else "long"
                content_cat = "code" if has_code else "question" if has_question else "plain"
                category = f"{length_cat}_{content_cat}"
                
                # Select if category not seen yet
                if category not in categories_seen:
                    selected.append(ex)
                    categories_seen.add(category)
                    
                # Stop once we have enough diverse examples
                if len(selected) >= 20:
                    break
                    
            return selected if selected else all_examples[:min(len(all_examples), 20)]
                
        # Later cycles (500+): Focus on complex examples
        else:
            # Filter for more complex content
            complex_examples = []
            
            for ex in all_examples:
                metadata = ex.get("metadata", {})
                content_metrics = metadata.get("content_metrics", {})
                quality_metrics = metadata.get("quality_metrics", {})
                
                word_count = content_metrics.get("word_count", 0)
                has_code = quality_metrics.get("has_code", False)
                
                # Select longer examples or those with code
                if word_count > 100 or has_code:
                    complex_examples.append(ex)
                    
            return complex_examples if complex_examples else all_examples[:min(len(all_examples), 20)]
