from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import time
import uuid
import math
import os
import threading
import random
from collections import deque
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM
import traceback
from sovl_scaffold import ScaffoldProvider
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_io import ConfigurationError
from sovl_confidence import ConfidenceCalculator
from sovl_temperament import TemperamentSystem
import hashlib
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize training configuration from ConfigManager.
        
        Args:
            config_manager: ConfigManager instance for accessing configuration
        """
        self.config_manager = config_manager
        self._load_config()
        
    def _load_config(self) -> None:
        """Load and validate training configuration."""
        try:
            # Get training section from config
            training_config = self.config_manager.get_section("training")
            if not training_config:
                raise ConfigurationError("Training configuration section is missing or empty.")
            
            # Load required parameters with immediate validation
            self.learning_rate = self.config_manager.get("training.learning_rate", 2e-5)
            assert self.learning_rate > 0, "Learning rate must be positive"
            
            self.grad_accum_steps = self.config_manager.get("training.grad_accum_steps", 4)
            assert self.grad_accum_steps >= 1, "Gradient accumulation steps must be at least 1"
            
            self.weight_decay = self.config_manager.get("training.weight_decay", 0.01)
            assert self.weight_decay >= 0, "Weight decay must be non-negative"
            
            self.warmup_steps = self.config_manager.get("training.warmup_steps", 0)
            assert self.warmup_steps >= 0, "Warmup steps must be non-negative"
            
            self.total_steps = self.config_manager.get("training.total_steps", 100000)
            assert self.total_steps > 0, "Total steps must be positive"
            
            self.max_grad_norm = self.config_manager.get("training.max_grad_norm", 1.0)
            assert self.max_grad_norm > 0, "Max gradient norm must be positive"
            
            self.use_amp = self.config_manager.get("training.use_amp", True)
            assert isinstance(self.use_amp, bool), "use_amp must be a boolean"
            
            self.max_patience = self.config_manager.get("training.max_patience", 2)
            assert self.max_patience >= 0, "Max patience must be non-negative"
            
            self.batch_size = self.config_manager.get("training.batch_size", 2)
            assert self.batch_size > 0, "Batch size must be positive"
            
            self.max_epochs = self.config_manager.get("training.max_epochs", 3)
            assert self.max_epochs > 0, "Max epochs must be positive"
            
            self.validate_every_n_steps = self.config_manager.get("training.validate_every_n_steps", 100)
            assert self.validate_every_n_steps > 0, "Validate every n steps must be positive"
            
            self.checkpoint_interval = self.config_manager.get("training.checkpoint_interval", 1000)
            assert self.checkpoint_interval > 0, "Checkpoint interval must be positive"
            
            self.checkpoint_path = self.config_manager.get("training.checkpoint_path", "checkpoints/sovl_trainer")
            assert isinstance(self.checkpoint_path, str), "Checkpoint path must be a string"
            
            self.scheduler_type = self.config_manager.get("training.scheduler_type", "linear")
            assert self.scheduler_type in ["linear", "cosine", "constant"], "Invalid scheduler type"
            
            self.cosine_min_lr = self.config_manager.get("training.cosine_min_lr", 1e-6)
            assert self.cosine_min_lr > 0, "Cosine min learning rate must be positive"
            
            self.warmup_ratio = self.config_manager.get("training.warmup_ratio", 0.1)
            assert 0 <= self.warmup_ratio <= 1, "Warmup ratio must be between 0 and 1"
            
            self.dropout_rate = self.config_manager.get("training.dropout_rate", 0.1)
            assert 0 <= self.dropout_rate <= 1, "Dropout rate must be between 0 and 1"
            
            self.max_seq_length = self.config_manager.get("training.max_seq_length", 512)
            assert self.max_seq_length > 0, "Max sequence length must be positive"
            
            # Load metrics configuration
            self.metrics_to_track = self.config_manager.get(
                "training.metrics_to_track",
                ["loss", "accuracy", "confidence"]
            )
            assert isinstance(self.metrics_to_track, list), "Metrics to track must be a list"
            
            # Load lifecycle configuration
            self.enable_gestation = self.config_manager.get("training.enable_gestation", True)
            assert isinstance(self.enable_gestation, bool), "enable_gestation must be a boolean"
            
            self.enable_sleep_training = self.config_manager.get("training.enable_sleep_training", True)
            assert isinstance(self.enable_sleep_training, bool), "enable_sleep_training must be a boolean"
            
            self.enable_lifecycle_weighting = self.config_manager.get("training.enable_lifecycle_weighting", True)
            assert isinstance(self.enable_lifecycle_weighting, bool), "enable_lifecycle_weighting must be a boolean"
            
            self.lifecycle_capacity_factor = self.config_manager.get("training.lifecycle_capacity_factor", 0.01)
            assert self.lifecycle_capacity_factor > 0, "Lifecycle capacity factor must be positive"
            
            self.lifecycle_curve = self.config_manager.get("training.lifecycle_curve", "sigmoid_linear")
            assert self.lifecycle_curve in ["sigmoid_linear", "exponential"], "Invalid lifecycle curve"
            
            # Load sleep configuration
            self.sleep_conf_threshold = self.config_manager.get("training.sleep_conf_threshold", 0.7)
            assert 0 <= self.sleep_conf_threshold <= 1, "Sleep confidence threshold must be between 0 and 1"
            
            self.sleep_log_min = self.config_manager.get("training.sleep_log_min", 10)
            assert self.sleep_log_min > 0, "Sleep log minimum must be positive"
            
            self.exposure_gain_eager = self.config_manager.get("training.exposure_gain_eager", 3)
            assert self.exposure_gain_eager > 0, "Exposure gain eager must be positive"
            
            self.exposure_gain_default = self.config_manager.get("training.exposure_gain_default", 2)
            assert self.exposure_gain_default > 0, "Exposure gain default must be positive"
            
            # Load dream configuration
            self.dream_memory_weight = self.config_manager.get("training.dream_memory_weight", 0.1)
            assert 0 <= self.dream_memory_weight <= 1, "Dream memory weight must be between 0 and 1"
            
            self.enable_dreaming = self.config_manager.get("training.enable_dreaming", True)
            assert isinstance(self.enable_dreaming, bool), "enable_dreaming must be a boolean"
            
            self.repetition_n = self.config_manager.get("training.repetition_n", 3)
            assert self.repetition_n >= 2, "Repetition check length must be at least 2"
            
            self.sigmoid_scale = self.config_manager.get("training.sigmoid_scale", 0.5)
            assert self.sigmoid_scale > 0, "Sigmoid scale must be positive"
            
            self.sigmoid_shift = self.config_manager.get("training.sigmoid_shift", 5.0)
            assert self.sigmoid_shift >= 0, "Sigmoid shift must be non-negative"
            
            self.dream_noise_scale = self.config_manager.get("training.dream_noise_scale", 0.05)
            assert self.dream_noise_scale >= 0, "Dream noise scale must be non-negative"
            
            self.dream_prompt_weight = self.config_manager.get("training.dream_prompt_weight", 0.5)
            assert 0 <= self.dream_prompt_weight <= 1, "Dream prompt weight must be between 0 and 1"
            
            self.dream_novelty_boost = self.config_manager.get("training.dream_novelty_boost", 0.03)
            assert self.dream_novelty_boost >= 0, "Dream novelty boost must be non-negative"
            
            self.dream_memory_decay = self.config_manager.get("training.dream_memory_decay", 0.95)
            assert 0 <= self.dream_memory_decay <= 1, "Dream memory decay must be between 0 and 1"
            
            self.dream_prune_threshold = self.config_manager.get("training.dream_prune_threshold", 0.1)
            assert 0 <= self.dream_prune_threshold <= 1, "Dream prune threshold must be between 0 and 1"
            
            self.temp_melancholy_noise = self.config_manager.get("training.temp_melancholy_noise", 0.02)
            assert self.temp_melancholy_noise >= 0, "Temperament melancholy noise must be non-negative"
            
            self.enable_prompt_driven_dreams = self.config_manager.get("training.enable_prompt_driven_dreams", True)
            assert isinstance(self.enable_prompt_driven_dreams, bool), "enable_prompt_driven_dreams must be a boolean"
            
            self.dream_swing_var = self.config_manager.get("training.dream_swing_var", 0.1)
            assert self.dream_swing_var >= 0, "Dream swing variance must be non-negative"
            
            self.dream_lifecycle_delta = self.config_manager.get("training.dream_lifecycle_delta", 0.1)
            assert self.dream_lifecycle_delta >= 0, "Dream lifecycle delta must be non-negative"
            
            self.dream_temperament_on = self.config_manager.get("training.dream_temperament_on", True)
            assert isinstance(self.dream_temperament_on, bool), "dream_temperament_on must be a boolean"
            
            # Load history configuration
            self.confidence_history_maxlen = self.config_manager.get("training.confidence_history_maxlen", 5)
            assert self.confidence_history_maxlen > 0, "Confidence history maxlen must be positive"
            
            self.temperament_history_maxlen = self.config_manager.get("training.temperament_history_maxlen", 5)
            assert self.temperament_history_maxlen > 0, "Temperament history maxlen must be positive"
            
            # Load dry run configuration
            self.dry_run = self.config_manager.get("training.dry_run", False)
            assert isinstance(self.dry_run, bool), "dry_run must be a boolean"
            
            self.dry_run_params = self.config_manager.get("training.dry_run_params", None)
            
            # Load memory configuration
            self.memory_threshold = self.config_manager.get("training.memory_threshold", 0.85)
            assert 0 <= self.memory_threshold <= 1, "Memory threshold must be between 0 and 1"
            
            self.memory_decay_rate = self.config_manager.get("training.memory_decay_rate", 0.95)
            assert 0 <= self.memory_decay_rate <= 1, "Memory decay rate must be between 0 and 1"
            
            self.use_scaffold_memory = self.config_manager.get("training.use_scaffold_memory", True)
            assert isinstance(self.use_scaffold_memory, bool), "use_scaffold_memory must be a boolean"
            
            self.use_token_map_memory = self.config_manager.get("training.use_token_map_memory", True)
            assert isinstance(self.use_token_map_memory, bool), "use_token_map_memory must be a boolean"
            
            self.scaffold_weight = self.config_manager.get("training.scaffold_weight", 1.0)
            assert self.scaffold_weight > 0, "Scaffold weight must be positive"
            
            # Final validation
            self._validate()
            
        except AssertionError as e:
            raise ConfigurationError(
                f"Invalid training configuration: {str(e)}",
                traceback.format_exc()
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load training configuration: {str(e)}",
                traceback.format_exc()
            )
            
    def _validate(self) -> None:
        """Validate configuration parameters."""
        try:
            # Validate relationships between parameters
            assert self.warmup_steps <= self.total_steps, "Warmup steps must be less than or equal to total steps"
            assert self.validate_every_n_steps <= self.total_steps, "Validate every n steps must be less than or equal to total steps"
            assert self.checkpoint_interval <= self.total_steps, "Checkpoint interval must be less than or equal to total steps"
            assert self.batch_size * self.grad_accum_steps > 0, "Effective batch size must be positive"
            
            # Validate memory configuration
            if self.use_scaffold_memory and self.use_token_map_memory:
                assert self.memory_threshold > 0.5, "Memory threshold should be higher when using both memory types"
            
            # Validate dream configuration
            if self.enable_dreaming:
                assert self.dream_memory_weight > 0, "Dream memory weight must be positive when dreaming is enabled"
                assert self.dream_memory_decay > 0, "Dream memory decay must be positive when dreaming is enabled"
            
            # Validate lifecycle configuration
            if self.enable_lifecycle_weighting:
                assert self.lifecycle_capacity_factor > 0, "Lifecycle capacity factor must be positive when lifecycle weighting is enabled"
            
        except AssertionError as e:
            raise ConfigurationError(
                f"Invalid training configuration relationships: {str(e)}",
                traceback.format_exc()
            )
            
    def update(self, key: str, value: Any) -> bool:
        """
        Update a configuration parameter.
        
        Args:
            key: Configuration key to update
            value: New value for the configuration key
            
        Returns:
            bool: True if update was successful, False otherwise
        """
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
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.
        
        Args:
            key: Configuration key to get
            default: Default value if key is not found
            
        Returns:
            Any: Configuration value
        """
        try:
            return self.config_manager.get(f"training.{key}", default)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to get training configuration: {str(e)}",
                traceback.format_exc()
            )
            
    def validate_section(self) -> bool:
        """
        Validate the training configuration section.
        
        Returns:
            bool: True if validation successful, False otherwise
        """
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

class TrainingManager:
    """Manages core training operations."""
    def __init__(
        self, 
        config: TrainingConfig, 
        model: torch.nn.Module, 
        device: torch.device, 
        loss_fn: Callable, 
        tokenizer: Any, 
        curiosity_manager: Optional[CuriosityManager] = None,
        confidence_calculator: Optional[ConfidenceCalculator] = None,
        temperament_system: Optional[TemperamentSystem] = None,
        memoria_manager: Optional[MemoriaManager] = None,
        ram_manager: Optional[RAMManager] = None,
        gpu_manager: Optional[GPUMemoryManager] = None
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.curiosity_manager = curiosity_manager
        self.confidence_calculator = confidence_calculator
        self.temperament_system = temperament_system
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = self._init_scheduler()
        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == "cuda" else None

    def _init_scheduler(self) -> Optional[Any]:
        """Initialize learning rate scheduler with confidence and temperament awareness."""
        warmup_steps = self.config.warmup_steps or int(self.config.warmup_ratio * self.config.total_steps)
        
        # Adjust warmup steps based on confidence and temperament if available
        if self.confidence_calculator and self.temperament_system:
            try:
                confidence = self.confidence_calculator.get_current_confidence()
                mood = self.temperament_system.mood_label
                
                # Adjust warmup based on confidence and mood
                confidence_factor = 1.0 + (1.0 - confidence) * 0.2
                mood_factor = 1.0
                if mood == "Cautious":
                    mood_factor = 1.2  # Longer warmup for cautious mood
                elif mood == "Curious":
                    mood_factor = 0.8  # Shorter warmup for curious mood
                
                warmup_steps = int(warmup_steps * confidence_factor * mood_factor)
            except Exception as e:
                self.logger.warning(f"Failed to adjust warmup steps based on confidence and temperament: {str(e)}")
        
        if self.config.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, self.config.total_steps)
        elif self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps - warmup_steps,
                eta_min=self.config.cosine_min_lr
            )
        return None

    def train_step_with_scaffold(
        self,
        batch: List[Dict[str, Any]],
        scaffold_provider: Optional[ScaffoldProvider] = None,
        dry_run: bool = False,
        dry_run_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute a single training step with scaffold support and memory-aware processing."""
        try:
            # Check memory health before processing
            ram_health = self.ram_manager.check_memory_health() if self.ram_manager else {"is_healthy": True}
            gpu_health = self.gpu_manager.check_memory_health() if self.gpu_manager else {"is_healthy": True}
            
            # Adjust batch size based on memory health
            if not ram_health['is_healthy'] or not gpu_health['is_healthy']:
                adjusted_size = max(1, self.config.batch_size // 2)
                self.logger.record_event(
                    event_type="batch_size_adjusted",
                    message="Batch size adjusted due to memory constraints",
                    level="warning",
                    additional_info={
                        "original_size": self.config.batch_size,
                        "adjusted_size": adjusted_size,
                        "ram_health": ram_health,
                        "gpu_health": gpu_health
                    }
                )
                self.config.batch_size = adjusted_size
            
            # Prepare batch and move to device
            prepared_batch = self._prepare_batch(batch)
            prepared_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in prepared_batch.items()}
            
            # Get scaffold context if available
            scaffold_context = None
            if scaffold_provider:
                try:
                    scaffold_context = scaffold_provider(prepared_batch)
                    if scaffold_context is not None:
                        scaffold_context = scaffold_context.to(self.device)
                        # Save scaffold context to memory if enabled
                        if self.config.use_scaffold_memory and self.memoria_manager:
                            self.memoria_manager.save_state("scaffold", scaffold_context)
                except Exception as e:
                    self.error_manager.handle_scaffold_error(e, {
                        "batch_size": len(batch),
                        "step": self.global_step
                    })
                    raise
            
            # Forward pass with memory monitoring
            outputs = self._forward_pass(prepared_batch, scaffold_context)
            
            # Calculate loss
            loss = self._calculate_loss(outputs, prepared_batch)
            
            # Backward pass
            if not dry_run:
                loss.backward()
                self._optimizer_step()
            
            # Update metrics
            metrics = self._update_metrics(outputs, loss)
            
            # Update curiosity if available
            if self.curiosity_manager:
                self._update_curiosity(metrics)
            
            # Log memory usage after step
            if self.ram_manager and self.gpu_manager:
                ram_usage = self.ram_manager.get_memory_usage()
                gpu_usage = self.gpu_manager.get_memory_usage()
                self.logger.record_event(
                    event_type="memory_usage",
                    message="Memory usage after training step",
                    level="info",
                    additional_info={
                        "ram_usage": ram_usage,
                        "gpu_usage": gpu_usage
                    }
                )
            
            return loss.item(), metrics
            
        except Exception as e:
            return self.error_manager.handle_training_error(e, {
                "step": self.global_step,
                "batch_size": len(batch),
                "dry_run": dry_run
            })

    def _update_curiosity(self, metrics: Dict[str, Any]) -> None:
        """Update curiosity metrics if curiosity manager is available."""
        if not self.curiosity_manager:
            return
            
        try:
            self.curiosity_manager.update_metrics(
                question=None,  # No specific question for training
                score=metrics.get('confidence', 0.5),
                spontaneous=False,
                answered=True,
                conversation_id=self.state.conversation_id,
                state_hash=self.state.get_state_hash()
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "metrics": metrics,
                "conversation_id": self.state.conversation_id
            })
            
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training with memory-aware processing."""
        try:
            # Check memory health before processing
            ram_health = self.ram_manager.check_memory_health() if self.ram_manager else {"is_healthy": True}
            gpu_health = self.gpu_manager.check_memory_health() if self.gpu_manager else {"is_healthy": True}
            
            # Adjust batch size if memory health is poor
            if not ram_health['is_healthy'] or not gpu_health['is_healthy']:
                adjusted_size = max(1, self.config.batch_size // 2)
                self.logger.record_event(
                    event_type="batch_size_adjusted",
                    message="Batch size adjusted during batch preparation",
                    level="warning",
                    additional_info={
                        "original_size": self.config.batch_size,
                        "adjusted_size": adjusted_size,
                        "ram_health": ram_health,
                        "gpu_health": gpu_health
                    }
                )
                self.config.batch_size = adjusted_size
            
            # Prepare batch data
            batch_size = len(batch)
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            
            # Save batch to memory if enabled
            if self.config.use_token_map_memory and self.memoria_manager:
                self.memoria_manager.save_state("batch", {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        except Exception as e:
            self.error_manager.handle_training_error(e, {
                "step": "batch_preparation",
                "batch_size": len(batch)
            })
            raise
            
    def _forward_pass(
        self,
        batch: Dict[str, torch.Tensor],
        scaffold_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Execute forward pass with memory-aware processing."""
        try:
            # Check memory health before processing
            ram_health = self.ram_manager.check_memory_health() if self.ram_manager else {"is_healthy": True}
            gpu_health = self.gpu_manager.check_memory_health() if self.gpu_manager else {"is_healthy": True}
            
            # Adjust batch size if memory health is poor
            if not ram_health['is_healthy'] or not gpu_health['is_healthy']:
                adjusted_size = max(1, self.config.batch_size // 2)
                self.logger.record_event(
                    event_type="batch_size_adjusted",
                    message="Batch size adjusted during forward pass",
                    level="warning",
                    additional_info={
                        "original_size": self.config.batch_size,
                        "adjusted_size": adjusted_size,
                        "ram_health": ram_health,
                        "gpu_health": gpu_health
                    }
                )
                self.config.batch_size = adjusted_size
            
            # Execute forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                scaffold_context=scaffold_context
            )
            
            # Save outputs to memory if enabled
            if self.config.use_token_map_memory and self.memoria_manager:
                self.memoria_manager.save_state("outputs", outputs)
            
            return outputs
            
        except Exception as e:
            self.error_manager.handle_training_error(e, {
                "step": "forward_pass",
                "batch_size": len(batch)
            })
            raise
            
    def _calculate_loss(
        self,
        outputs: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate loss."""
        try:
            # ... existing loss calculation code ...
            pass
        except Exception as e:
            self.error_manager.handle_training_error(e, {
                "step": "loss_calculation",
                "batch_size": len(batch)
            })
            raise
            
    def _optimizer_step(self) -> None:
        """Execute optimizer step."""
        try:
            # ... existing optimizer step code ...
            pass
        except Exception as e:
            self.error_manager.handle_training_error(e, {
                "step": "optimizer_step",
                "global_step": self.global_step
            })
            raise
            
    def _update_metrics(
        self,
        outputs: torch.Tensor,
        loss: torch.Tensor
    ) -> Dict[str, Any]:
        """Update training metrics."""
        try:
            # ... existing metrics update code ...
            pass
        except Exception as e:
            self.error_manager.handle_training_error(e, {
                "step": "metrics_update",
                "global_step": self.global_step
            })
            raise

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

    def run_gestation_cycle(self, batch: List[Dict[str, Any]]) -> None:
        """Run gestation cycle."""
        try:
            # Get batch size from memory manager
            batch_size = self.trainer.memory_manager.get_batch_size()
            
            # Run gestation step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch,
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

class TrainingCycleManager:
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = MemoriaManager(config_manager, logger)
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

class SOVLTrainer:
    """Manages training operations and memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        memoria_manager: MemoriaManager,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize trainer.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            memoria_manager: MemoriaManager instance for core memory management
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
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
