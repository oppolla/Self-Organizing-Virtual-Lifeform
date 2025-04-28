from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
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
from sovl_io import JSONLLoader
import threading

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
    
    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        device: torch.device,
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[Logger] = None,
        error_manager: Optional[ErrorManager] = None
    ):
        """
        Initialize training manager.
        Args:
            config: TrainingConfig instance
            model: torch.nn.Module (MUST be pre-wrapped/adapted by ModelManager)
            device: torch.device
            config_manager: ConfigManager instance
            logger: Logger instance
            error_manager: ErrorManager instance
        """
        self.config = config
        self.model = model
        self.device = device
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        # LoRA manager is used ONLY for checkpointing and optimizer param selection
        self.lora_manager = None
        if self.config_manager and self.logger and self.error_manager:
            self.lora_manager = LoraAdapterManager(self.config_manager, self.logger, self.error_manager)
        # Use only LoRA parameters if LoRA is enabled, else all parameters
        if self.lora_manager and self.lora_manager.enabled:
            params = self.lora_manager.lora_parameters(self.model)
        else:
            params = self.model.parameters()
        self.optimizer = self._create_optimizer(params)
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if self.config.memory.use_amp and torch.cuda.is_available() else None
        self.step_count = 0
        self.epoch_count = 0
        self.optimizer_step_count = 0
        self.best_metrics = {}
        self.metrics_history = defaultdict(list)
        
    def _create_optimizer(self, params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.config.optimizer.type.lower()
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                params,
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
        """Prepare batch for training with memory optimization and robust device handling."""
        prepared_batch = {}
        max_length = self.config.memory.max_seq_length
        device = self.device
        # Check device availability
        if device.type == "cuda" and not torch.cuda.is_available():
            if hasattr(self, "logger") and self.logger:
                self.logger.log_warning(
                    message=f"CUDA device requested but is not available. Falling back to CPU.",
                    event_type="device_warning"
                )
            device = torch.device("cpu")
        for key, tensor in batch.items():
            if not isinstance(tensor, torch.Tensor):
                if hasattr(self, "logger") and self.logger:
                    self.logger.log_warning(
                        message=f"Non-tensor value for key {key} in batch",
                        event_type="batch_preparation_warning"
                    )
                continue
            # Only move if not already on the correct device
            if tensor.device != device:
                try:
                    tensor = tensor.to(device)
                except RuntimeError as e:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.log_error(
                            error_msg=f"Failed to move tensor to device {device}: {str(e)}",
                            error_type="device_transfer_error"
                        )
                    raise
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
        
        # Determine if we should use AMP
        use_amp = self.config.memory.use_amp and self.scaler is not None
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Ensure labels are present if needed
            if "labels" not in batch and "input_ids" in batch:
                batch["labels"] = batch["input_ids"].clone()
            
            outputs = self.model(**batch)
            
            # Check for loss
            if not hasattr(outputs, 'loss') or outputs.loss is None:
                return {"loss": 0.0}  # Return default metrics if no loss
                
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
            
            # Increment optimizer step count and set data_exposure
            self.optimizer_step_count += 1
            metrics["data_exposure"] = float(self.optimizer_step_count)
            
        # Update metrics
        metrics["loss"] = loss.item() * self.config.optimizer.grad_accum_steps
        
        # Calculate accuracy if possible
        if hasattr(outputs, "logits") and "labels" in batch:
            try:
                logits = outputs.logits.detach()
                labels = batch["labels"].detach()
                
                # Ensure labels are on the same device as logits
                if labels.device != logits.device:
                    labels = labels.to(logits.device)
                    
                # Handle different tensor dimensions
                if logits.dim() == 3 and labels.dim() == 2:  # Sequence model
                    # Ignore padding index (-100)
                    active_loss = labels.view(-1) != -100
                    active_logits = logits.view(-1, logits.size(-1))[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    if active_labels.numel() > 0:
                        metrics["accuracy"] = (active_logits.argmax(dim=-1) == active_labels).float().mean().item()
                    else:
                        metrics["accuracy"] = 0.0
                elif logits.dim() == 2 and labels.dim() == 1:  # Classification
                    metrics["accuracy"] = (logits.argmax(dim=-1) == labels).float().mean().item()
                else:
                    metrics["accuracy"] = 0.0
            except Exception as e:
                metrics["accuracy"] = 0.0
                
        self.step_count += 1
        return metrics
        
    @torch.no_grad()
    def validate(self, val_loader: 'StreamingJSONLoader') -> Dict[str, float]:
        """Run validation with memory optimization."""
        self.model.eval()
        metrics = defaultdict(float)
        num_batches = 0
        
        for batch in val_loader:
            # Prepare batch
            batch = self._prepare_batch(batch)
            
            # Determine if we should use AMP
            use_amp = self.config.memory.use_amp and self.scaler is not None
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Ensure labels are present if needed
                if "labels" not in batch and "input_ids" in batch:
                    batch["labels"] = batch["input_ids"].clone()
                    
                outputs = self.model(**batch)
                
                # Update metrics
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    metrics["loss"] += outputs.loss.item()
                    
                if hasattr(outputs, "logits") and "labels" in batch:
                    try:
                        logits = outputs.logits.detach()
                        labels = batch["labels"].detach()
                        
                        # Ensure labels are on the same device as logits
                        if labels.device != logits.device:
                            labels = labels.to(logits.device)
                            
                        # Handle different tensor dimensions
                        if logits.dim() == 3 and labels.dim() == 2:  # Sequence model
                            active_loss = labels.view(-1) != -100
                            active_logits = logits.view(-1, logits.size(-1))[active_loss]
                            active_labels = labels.view(-1)[active_loss]
                            if active_labels.numel() > 0:
                                metrics["accuracy"] += (active_logits.argmax(dim=-1) == active_labels).float().mean().item()
                        elif logits.dim() == 2 and labels.dim() == 1:  # Classification
                            metrics["accuracy"] += (logits.argmax(dim=-1) == labels).float().mean().item()
                    except Exception as e:
                        pass  # Skip accuracy calculation on error
                        
            num_batches += 1
            
        # Average metrics only if we have batches
        if num_batches > 0:
            metrics = {k: v / num_batches for k, v in metrics.items()}
        else:
            metrics = {k: 0.0 for k in metrics.keys()}
            
        # Update best metrics based on primary metric (loss)
        primary_metric = "loss"
        if primary_metric in metrics:
            current_best = self.best_metrics.get(primary_metric, float('inf'))
            if metrics[primary_metric] < current_best:
                self.best_metrics[primary_metric] = metrics[primary_metric]
                
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
            "optimizer_step_count": self.optimizer_step_count,
            "epoch_count": self.epoch_count,
            "best_metrics": self.best_metrics,
            "metrics_history": dict(self.metrics_history),
            "config": self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint with validation."""
        # Check if checkpoint file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found at {path}")
            
        # Determine map location based on CUDA availability
        map_location = self.device if torch.cuda.is_available() else torch.device('cpu')
        
        try:
            checkpoint = torch.load(path, map_location=map_location)
            
            # Validate checkpoint structure
            required_keys = ["model_state", "optimizer_state", "step_count", "epoch_count"]
            if not all(key in checkpoint for key in required_keys):
                raise ValueError(f"Checkpoint file at {path} is missing required keys.")
                
            # Load model state
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
            
            # Load optimizer state
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as e:
                raise RuntimeError(f"Could not load optimizer state: {e}")
                
            # Load scheduler state if available
            if self.scheduler and checkpoint.get("scheduler_state"):
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                except Exception as e:
                    pass  # Continue without scheduler state
                    
            # Load scaler state if available
            if self.scaler and checkpoint.get("scaler_state"):
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state"])
                except Exception as e:
                    pass  # Continue without scaler state
                    
            # Load training state
            self.step_count = checkpoint["step_count"]
            self.optimizer_step_count = checkpoint.get("optimizer_step_count", 0)
            self.epoch_count = checkpoint["epoch_count"]
            self.best_metrics = checkpoint.get("best_metrics", {})
            self.metrics_history = defaultdict(list, checkpoint.get("metrics_history", {}))
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
            
    def save_lora_checkpoint(self, path: str):
        if self.lora_manager:
            self.lora_manager.save_lora_weights(self.model, path)
            
    def load_lora_checkpoint(self, path: str):
        if self.lora_manager:
            self.model = self.lora_manager.load_lora_weights(self.model, path)

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
        
    def run_training_cycle(self, batch: List[Dict[str, Any]], scaffold_provider: Optional[ScaffoldProvider] = None) -> Tuple[float, Dict[str, Any]]:
        """Run a complete training cycle using the modular pipeline with error recovery and notification."""
        device = getattr(self.trainer, 'device', None)
        batch_preparer = getattr(self.trainer, 'batch_preparer', None)
        logger = getattr(self, 'logger', None)
        training_manager = getattr(self.trainer, 'training_manager', None)
        error_manager = getattr(self.trainer, 'error_manager', None)
        error_handling_cfg = None
        # Get error handling config if available
        if hasattr(self, 'config') and self.config and hasattr(self.config, 'logging'):
            error_handling_cfg = getattr(self.config.logging, 'error_handling_config', None)
        retry_attempts = (error_handling_cfg or {}).get('retry_attempts', 1)
        retry_delay = (error_handling_cfg or {}).get('retry_delay', 0.0)
        last_exception = None
        for attempt in range(retry_attempts):
            # Prepare batch using modular batch preparer
            try:
                collated_batch = batch_preparer.prepare(batch)
            except Exception as e:
                last_exception = e
                if logger:
                    logger.log_error(
                        f"Failed to prepare training batch: {str(e)}",
                        error_type="training_cycle_batch_preparation_error",
                        stack_trace=traceback.format_exc(),
                        additional_info={"attempt": attempt+1}
                    )
                if error_manager:
                    error_manager.notify_error(
                        error_type="training_cycle_batch_preparation_error",
                        error_msg=str(e),
                        context={"attempt": attempt+1, "batch": batch}
                    )
                if attempt < retry_attempts - 1 and retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            # Run training step
            try:
                metrics = training_manager.train_step(batch=collated_batch)
                loss = metrics.get("loss", 0.0)
                return loss, metrics
            except Exception as e:
                last_exception = e
                if logger:
                    logger.log_error(
                        f"Error during training step: {str(e)}",
                        error_type="training_cycle_training_error",
                        stack_trace=traceback.format_exc(),
                        additional_info={"attempt": attempt+1}
                    )
                if error_manager:
                    error_manager.notify_error(
                        error_type="training_cycle_training_error",
                        error_msg=str(e),
                        context={"attempt": attempt+1, "batch": batch}
                    )
                if attempt < retry_attempts - 1 and retry_delay > 0:
                    time.sleep(retry_delay)
                continue
        # If all attempts failed
        if logger:
            logger.log_error(
                f"Training cycle failed after {retry_attempts} attempts: {str(last_exception)}",
                error_type="training_cycle_final_failure",
                stack_trace=traceback.format_exc() if last_exception else None
            )
        if error_manager:
            error_manager.notify_error(
                error_type="training_cycle_final_failure",
                error_msg=str(last_exception),
                context={"batch": batch}
            )
        return 0.0, {"status": "error", "error": str(last_exception) if last_exception else "unknown_error"}
        
    def run_gestation_cycle(self, conversation_history: List[Dict[str, str]]) -> None:
        """Run gestation cycle with metadata enrichment using the modular pipeline. Supports multiple scaffolds and robust validation."""
        # Use modular pipeline for metadata enrichment and batch preparation
        metadata_processor = getattr(self.trainer, 'metadata_processor', None)
        tokenizer = getattr(self.trainer, 'tokenizer', None)
        device = getattr(self.trainer, 'device', None)
        batch_preparer = getattr(self.trainer, 'batch_preparer', None)
        logger = getattr(self, 'logger', None)
        model_manager = getattr(self, 'model_manager', None)
        training_manager = getattr(self.trainer, 'training_manager', None)
        dreamer = getattr(self.trainer, 'dreamer', None)
        state = getattr(self, 'state', None)
    
        # Validate conversation history
        if not conversation_history or not isinstance(conversation_history, list) or not all(isinstance(s, dict) for s in conversation_history):
            if logger:
                logger.log_error("Invalid or missing conversation history for gestation cycle.", event_type="gestation_invalid_conversation")
            return
    
        # Validate model_manager and scaffold models
        scaffold_count = 0
        lora_count = 0
        if model_manager:
            scaffold_count = len(getattr(model_manager, 'scaffold_models', []))
            lora_count = len(getattr(model_manager, 'lora_managers', []))
        if scaffold_count == 0 or lora_count == 0:
            if logger:
                logger.log_error(f"No scaffold models ({scaffold_count}) or LoRA managers ({lora_count}) available for gestation cycle.", event_type="gestation_scaffold_lora_missing")
            return
    
        # Iterate over all scaffold models/LoRA managers (future-proofing)
        for idx in range(min(scaffold_count, lora_count)):
            scaffold_model = model_manager.scaffold_models[idx]
            lora_manager = model_manager.lora_managers[idx]
            try:
                # Prepare training batch from conversation history
                enriched_samples = [metadata_processor.enrich(sample) for sample in conversation_history]
                batch = batch_preparer.prepare(enriched_samples)
            except Exception as e:
                if logger:
                    logger.log_error(f"Error during gestation batch preparation (scaffold {idx}): {str(e)}", error_type="gestation_batch_error", stack_trace=traceback.format_exc(), additional_info={"scaffold_index": idx})
                continue
            
            try:
                optimizer = torch.optim.AdamW(lora_manager.lora_parameters(scaffold_model), lr=2e-5)
                scaffold_model.train()
                optimizer.zero_grad()
                outputs = scaffold_model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
                optimizer.step()
                if logger:
                    logger.log_info(f"Gestation LoRA training step complete (scaffold {idx}). Loss: {loss.item()}", event_type="gestation_lora_train", additional_info={"scaffold_index": idx})
            except Exception as e:
                if logger:
                    logger.log_error(f"Error during gestation LoRA training step (scaffold {idx}): {str(e)}", error_type="gestation_lora_training_error", stack_trace=traceback.format_exc(), additional_info={"scaffold_index": idx})
                continue
            
            # Save LoRA weights as long-term memory
            try:
                import os
                from datetime import datetime
                lora_dir = "lora_checkpoints"
                os.makedirs(lora_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                lora_path = os.path.join(lora_dir, f"lora_{idx}_{timestamp}.pt")
                lora_manager.save_lora_weights(scaffold_model, lora_path)
                if model_manager:
                    model_manager.set_active_lora_checkpoint(lora_path)
                if logger:
                    logger.log_info(f"LoRA weights saved after gestation for scaffold {idx} to {lora_path}", event_type="gestation_lora_save", additional_info={"scaffold_index": idx})
            except Exception as e:
                if logger:
                    logger.log_error(f"Error saving LoRA weights after gestation (scaffold {idx}): {str(e)}", error_type="gestation_lora_save_error", stack_trace=traceback.format_exc(), additional_info={"scaffold_index": idx})
                continue
            
        # Update SOVL state if available
        if state is not None and hasattr(state, 'update_after_gestation'):
            try:
                state.update_after_gestation()
                if logger:
                    logger.log_info("SOVL state updated after gestation.", event_type="gestation_state_update")
            except Exception as e:
                if logger:
                    logger.log_error(f"Error updating SOVL state after gestation: {str(e)}", error_type="gestation_state_update_error", stack_trace=traceback.format_exc())
    
        # DREAMER integration
        if dreamer is not None:
            try:
                dreamer.run_dream_cycle()
                if logger:
                    logger.log_info("Dreamer cycle completed after gestation.", event_type="gestation_dreamer")
            except Exception as e:
                if logger:
                    logger.log_error(f"Error during Dreamer cycle: {str(e)}", error_type="gestation_dreamer_error", stack_trace=traceback.format_exc())

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
        
        # Initialize TrainingManager with ErrorManager
        self.training_manager = TrainingManager(
            self.config, self.model, self.device, config_manager, logger, self.error_manager
        )
        
        # Initialize training state tracking
        self._training_state = {
            "current_epoch": 0,
            "total_steps": self.training_manager.step_count,
            "last_checkpoint": None,
            "best_metrics": self.training_manager.best_metrics.copy(),
            "error_history": deque(maxlen=self.config.logging.max_recent_errors)
        }
        
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

    def train_with_curriculum(self, training_cycle: int, data_loader: 'StreamingJSONLoader') -> Tuple[float, Dict[str, Any]]:
        """
        Train using a curriculum that gradually increases complexity with modular pipeline.
        Args:
            training_cycle: Current training cycle number
            data_loader: StreamingJSONLoader instance for streaming data
        Returns:
            Tuple of (loss, metrics)
        """
        logger = getattr(self, '_logger', None)
        batch_preparer = getattr(self, 'batch_preparer', None)
        training_manager = getattr(self, 'training_manager', None)
        metadata_processor = getattr(self, 'metadata_processor', None)
        device = getattr(self, 'device', None)
        config = getattr(self, 'config', None)

        if not (logger and batch_preparer and training_manager and metadata_processor and device and config):
            if logger:
                logger.log_warning(
                    "Missing modular pipeline component(s) for curriculum training.",
                    event_type="curriculum_modular_pipeline_missing"
                )
            return 0.0, {"status": "missing_pipeline_component"}

        try:
            logger.log_event(
                event_type="curriculum_training_start",
                message=f"Starting curriculum training cycle {training_cycle}",
                level="info",
                additional_info={"cycle": training_cycle}
            )
            # Stream and enrich samples
            enriched_samples = [metadata_processor.enrich(sample) for sample in data_loader]
            # Select examples for this curriculum stage
            selected_examples = self._select_curriculum_examples(training_cycle, enriched_samples)
            if not selected_examples:
                logger.log_warning(
                    "No suitable examples found for curriculum training cycle",
                    event_type="training_data_warning",
                    additional_info={"cycle": training_cycle}
                )
                return 0.0, {"status": "no_examples"}
            batch_size = config.memory.batch_size
            batch_data = selected_examples[:batch_size]
            if not batch_data:
                logger.log_warning(
                    "Batch data is empty after selection/sizing in curriculum training.",
                    event_type="curriculum_empty_batch",
                    additional_info={"cycle": training_cycle, "selected_count": len(selected_examples), "batch_size": batch_size}
                )
                return 0.0, {"status": "empty_batch"}
            # Prepare batch using modular batch preparer
            try:
                collated_batch = batch_preparer.prepare(batch_data)
            except Exception as e:
                logger.log_error(
                    f"Failed to collate curriculum batch: {str(e)}",
                    error_type="curriculum_collation_error",
                    additional_info={"cycle": training_cycle, "batch_keys": list(batch_data[0].keys()) if batch_data else []}
                )
                return 0.0, {"status": "collation_error"}
            # Run training step
            try:
                metrics = training_manager.train_step(batch=collated_batch)
                loss = metrics.get("loss", 0.0)
            except Exception as e:
                logger.log_error(
                    f"Failed during curriculum training step execution: {str(e)}",
                    error_type="curriculum_training_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={"cycle": training_cycle, "batch_size": len(batch_data)}
                )
                return 0.0, {"status": "training_step_error", "error": str(e)}
            # Log curriculum stage
            stage = "beginner" if training_cycle < 100 else "intermediate" if training_cycle < 500 else "advanced"
            logger.log_event(
                event_type="curriculum_training_complete",
                message=f"Completed curriculum training step (stage: {stage})",
                level="info",
                additional_info={"cycle": training_cycle, "stage": stage, "examples_selected": len(selected_examples), "batch_size": len(batch_data), "loss": loss, "metrics": metrics}
            )
            return loss, metrics
        except Exception as e:
            logger.log_error(
                f"Unexpected error in train_with_curriculum: {str(e)}",
                error_type="curriculum_unexpected_error",
                stack_trace=traceback.format_exc(),
                additional_info={"cycle": training_cycle}
            )
            return 0.0, {"status": "unexpected_error", "error": str(e)}
    
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

def collate_tensor_batch(batch: list, device: "torch.device") -> dict:
    """
    Collate a list of dicts (with tensor values) into a batch dict of stacked tensors, moved to device.
    Args:
        batch: List[Dict[str, torch.Tensor]]
        device: torch.device
    Returns:
        Dict[str, torch.Tensor] (all tensors stacked and moved to device)
    """
    if not batch:
        return {}
    try:
        collated = {k: torch.stack([item[k] for item in batch]) for k in batch[0] if isinstance(batch[0][k], torch.Tensor)}
        collated = {k: v.to(device) for k, v in collated.items()}
        return collated
    except Exception as e:
        raise RuntimeError(f"Failed to collate tensor batch: {e}")

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

# ================== END BATCH & TENSOR HELPERS ==================

# ================== MODULAR DATA PIPELINE INTERFACES (Phase 3) ==================

class StreamingJSONLoader:
    """
    Loads training samples from a JSONL file, yielding one sample at a time (streaming).
    Wraps the JSONLLoader from sovl_io for thread safety and validation.
    """
    def __init__(self, file_path: str, config_manager: 'ConfigManager', logger: 'Logger', error_manager: 'ErrorManager'):
        from sovl_io import JSONLLoader
        # Always use config_manager to get the canonical data path
        if file_path is None:
            file_path = config_manager.get("data_provider.data_path")
        self.loader = JSONLLoader(config_manager, logger, error_manager)
        self.file_path = file_path
        self.logger = logger

    def __iter__(self):
        # Stream and yield validated samples using the new streaming method
        try:
            for sample in self.loader.stream_jsonl(self.file_path):
                yield sample
        except Exception as e:
            self.logger.log_error(f"Failed to stream JSONL data: {e}", error_type="jsonl_stream_error")
            return

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

import json
from datetime import datetime
import random

class Dreamer:
    """
    Dream system for SOVL: selects, generates, and logs dream events from the last active period.
    Each dream is a surreal narration, with optional dream noise for creativity.
    """
    def __init__(self, config_manager, scribe_path, logger, metadata_processor, scribe_event_fn, error_manager=None):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = logger
        self.metadata_processor = metadata_processor
        self.scribe_event_fn = scribe_event_fn  # Function to log a scribe event (e.g., capture_scribe_event)
        self.error_manager = error_manager
        # Configurable dream parameters
        self.max_dreams = config_manager.get("dream_max_events_per_cycle", 5)
        self.novelty_weight = config_manager.get("dream_novelty_weight", 1.0)
        self.confidence_weight = config_manager.get("dream_confidence_weight", 0.0)
        self.selection_strategy = config_manager.get("dream_selection_strategy", "top")
        self.dream_noise_level = config_manager.get("dream_noise_level", 0.2)

    def extract_last_active_period(self):
        """
        Extract events from the last active period (since last 'wake' event).
        Returns: List of scribe log event dicts.
        """
        events = []
        try:
            loader = JSONLLoader(self.config_manager, self.logger, self.error_manager) if self.error_manager else JSONLLoader(self.config_manager, self.logger, None)
            
            # First pass: Find the index of the last 'wake' event
            last_wake_idx = None
            current_idx = 0
            for entry in loader.stream_jsonl(self.scribe_path):
                if entry.get("event_type") == "wake":
                    last_wake_idx = current_idx
                current_idx += 1
            
            # Second pass: Collect events from last_wake_idx (or start) using streaming
            current_idx = 0
            for entry in loader.stream_jsonl(self.scribe_path):
                if last_wake_idx is None or current_idx > last_wake_idx:
                    events.append(entry)
                current_idx += 1
                
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    f"Failed to stream scribe log with JSONLLoader: {str(e)}",
                    error_type="scribe_stream_error",
                    stack_trace=traceback.format_exc()
                )
        
        return events

    def score_and_select_dreams(self, events):
        """
        Score and select dream candidates based on novelty/confidence.
        Returns: List of selected dream event dicts.
        """
        scored = []
        for event in events:
            meta = event.get("metadata", {})
            novelty = meta.get("novelty", 0)
            confidence = meta.get("confidence", 1)
            score = self.novelty_weight * novelty - self.confidence_weight * confidence
            scored.append((score, event))
        scored.sort(reverse=True, key=lambda x: x[0])
        if self.selection_strategy == "top":
            selected = [e for (_, e) in scored[:self.max_dreams]]
        elif self.selection_strategy == "random":
            selected = [e for (_, e) in random.sample(scored, min(self.max_dreams, len(scored)))]
        else:
            selected = [e for (_, e) in scored[:self.max_dreams]]
        return selected

    def add_dream_noise(self, dream_event):
        noise_level = self.dream_noise_level
        ed = dream_event["event_data"].copy()
        # Shuffle words or insert random tokens in text fields
        for key, value in ed.items():
            if isinstance(value, str) and random.random() < noise_level:
                words = value.split()
                if words:
                    random.shuffle(words)
                    if random.random() < noise_level:
                        words.insert(
                            random.randint(0, len(words)),
                            random.choice(["???", "dream", "echo", "phantom", "mist", "fragment"])
                        )
                    ed[key] = " ".join(words)
        # Mutate metadata
        meta = dream_event["metadata"].copy()
        if "novelty" in meta:
            meta["novelty"] += random.uniform(-0.1, 0.1) * noise_level
        if "confidence" in meta:
            meta["confidence"] += random.uniform(-0.1, 0.1) * noise_level
        dream_event["event_data"] = ed
        dream_event["metadata"] = meta
        return dream_event

    def narrate_dream(self, dream_event):
        """
        Generate a surreal narrative for the dream event.
        """
        prompt = dream_event["event_data"].get("prompt", "")
        response = dream_event["event_data"].get("response", "")
        # Simple surreal narration: merge, shuffle, and add dream-like phrases
        parts = [prompt, response]
        random.shuffle(parts)
        narration = f"In the midst of swirling thoughts, a dream emerged: {parts[0]} ... Suddenly, {parts[1]} ... The boundaries of meaning blurred."
        if random.random() < self.dream_noise_level:
            narration += f" {random.choice(['A phantom word echoed.', 'Mist enveloped the memory.', 'Fragments danced in the void.'])}"
        return narration.strip()

    def generate_dream_events(self, dream_candidates):
        dreams = []
        now = datetime.now().isoformat()
        for event in dream_candidates:
            dream_event = {
                "timestamp_iso": now,
                "event_type": "dream",
                "event_data": event.get("event_data", {}),
                "metadata": event.get("metadata", {}),
                "dreamed_from": event.get("event_type", "unknown")
            }
            dream_event = self.add_dream_noise(dream_event)
            narration = self.narrate_dream(dream_event)
            # Only the narration is logged as the main dream content
            dream_event["narration"] = narration
            dreams.append(dream_event)
        return dreams

    def log_dreams(self, dreams):
        for dream in dreams:
            try:
                self.scribe_event_fn(
                    event_type="dream",
                    event_data={"narration": dream["narration"]},
                    metadata=dream["metadata"],
                    dreamed_from=dream["dreamed_from"],
                    timestamp_iso=dream["timestamp_iso"]
                )
                self.logger.info(f"Dream event logged: {dream['narration']}")
            except Exception as e:
                self.logger.log_error(f"Failed to log dream event: {e}")

    def run_dream_cycle(self):
        """
        Main entry: extract, select, generate, and log dreams.
        """
        events = self.extract_last_active_period()
        dream_candidates = self.score_and_select_dreams(events)
        dreams = self.generate_dream_events(dream_candidates)
        self.log_dreams(dreams)
