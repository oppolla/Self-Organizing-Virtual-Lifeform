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

@dataclass
class DreamMemoryConfig:
    """Configuration for dream memory behavior."""
    max_memories: int = 100
    novelty_boost: float = 0.03
    base_weight: float = 0.1
    max_weight: float = 1.5
    decay_rate: float = 0.95
    prune_threshold: float = 0.1
    noise_scale: float = 0.05
    melancholy_noise: float = 0.02

class DreamMemory:
    """Thread-safe dream memory management system."""
    def __init__(self, config: DreamMemoryConfig, device: torch.device):
        self.memory = deque(maxlen=config.max_memories)
        self.config = config
        self.device = device
        self.lock = threading.Lock()

    def add_memory(self, prompt: str, hidden_state: torch.Tensor, is_novel: bool, temperament: float = 0.0) -> None:
        """Add a memory with automatic pruning."""
        with self.lock:
            self._maintain_memory()
            weight = self._calculate_memory_weight(is_novel)
            noisy_state = self._apply_noise(hidden_state, temperament)
            self.memory.append({
                "vector": noisy_state,
                "weight": weight,
                "prompt": prompt,
                "timestamp": time.time()
            })

    def _calculate_memory_weight(self, is_novel: bool) -> float:
        """Calculate weight with novelty boost."""
        weight = self.config.base_weight + (self.config.novelty_boost if is_novel else 0.0)
        return min(weight, self.config.max_weight)

    def _apply_noise(self, hidden_state: torch.Tensor, temperament: float) -> torch.Tensor:
        """Apply temperament-adjusted noise to hidden state."""
        noise_level = self.config.noise_scale + (self.config.melancholy_noise if temperament < -0.5 else 0.0)
        noise = torch.randn_like(hidden_state) * noise_level
        return (hidden_state + noise).detach().cpu()

    def _maintain_memory(self) -> None:
        """Apply decay and prune weak memories."""
        updated_memory = deque(maxlen=self.memory.maxlen)
        for m in self.memory:
            new_weight = m["weight"] * self.config.decay_rate
            if new_weight > self.config.prune_threshold:
                m["weight"] = new_weight
                updated_memory.append(m)
        self.memory = updated_memory

    def get_memories(self, n: int = 5) -> List[dict]:
        """Get top-n memories by weight."""
        with self.lock:
            return sorted(self.memory, key=lambda x: -x["weight"])[:n]

    def __len__(self) -> int:
        """Return current number of memories."""
        with self.lock:
            return len(self.memory)

def collate_batch(batch: List[dict], pad_token_id: int, max_seq_length: int, tokenizer: Any) -> Dict[str, torch.Tensor]:
    """Collate batch of prompt-completion pairs into tensors."""
    prompts = [item["prompt"] for item in batch]
    completions = [item["completion"] for item in batch]
    full_texts = [p + c for p, c in zip(prompts, completions)]

    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    prompt_mask = prompt_encodings["attention_mask"]
    labels = torch.where(prompt_mask.bool(), -100, input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt": prompts
    }

class LifecycleManager:
    """Manages model lifecycle and exposure tracking."""
    def __init__(self, config: TrainingConfig, model: torch.nn.Module, state: Optional[Any]):
        self.config = config
        self.state = state
        self.data_exposure = 0
        self.lora_capacity = sum(p.numel() for p in model.parameters() if p.requires_grad) * config.lifecycle_capacity_factor

    def get_life_curve_weight(self) -> float:
        """Calculate lifecycle weight based on data exposure."""
        x = self.data_exposure / self.lora_capacity if self.lora_capacity > 0 else 0
        if self.config.lifecycle_curve == "sigmoid_linear":
            return min(1.0, 1 / (1 + math.exp(-self.config.sigmoid_scale * (x - self.config.sigmoid_shift))))
        return min(1.0, 1 - math.exp(-x))

    def update_exposure(self, prompts: List[str], temperament_score: float) -> None:
        """Update data exposure based on prompts and temperament."""
        exposure_gain = self.config.exposure_gain_eager if temperament_score > 0.5 else self.config.exposure_gain_default
        if self.state:
            for prompt in prompts:
                if prompt not in self.state.seen_prompts:
                    self.state.add_seen_prompt(prompt)
                    self.data_exposure += exposure_gain

class DreamManager:
    """Manages dream generation and memory."""
    
    def __init__(self, state_manager: StateManager, error_manager: ErrorManager):
        self.state_manager = state_manager
        self.error_manager = error_manager
        self.logger = logging.getLogger(__name__)
        
    def _check_memory_usage(self, tensor: torch.Tensor) -> bool:
        """Check if adding a tensor would exceed memory limits."""
        state = self.state_manager.get_current_state()
        memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)  # Convert to MB
        return state.total_dream_memory_mb + memory_size <= state.config.max_dream_memory_mb
        
    def _add_to_memory(self, dream_entry: Dict[str, Any]) -> bool:
        """Add dream to memory if within limits."""
        try:
            state = self.state_manager.get_current_state()
            tensor = dream_entry["tensor"]
            
            if not self._check_memory_usage(tensor):
                self.logger.warning("Memory limit would be exceeded - pruning old memories")
                self._maintain_memory()
                
                # Check again after pruning
                if not self._check_memory_usage(tensor):
                    return False
                    
            memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)
            state.total_dream_memory_mb += memory_size
            state.dream_memory.append(dream_entry)
            return True
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, {"operation": "add_to_memory"})
            return False
            
    def _maintain_memory(self):
        """Maintain dream memory within limits."""
        try:
            state = self.state_manager.get_current_state()
            while state.total_dream_memory_mb > state.config.max_dream_memory_mb and state.dream_memory:
                removed = state.dream_memory.popleft()
                tensor = removed["tensor"]
                memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                state.total_dream_memory_mb -= memory_size
                
        except Exception as e:
            self.error_manager.handle_memory_error(e, {"operation": "maintain_memory"})
            
    def dream(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate a dream from a prompt."""
        try:
            state = self.state_manager.get_current_state()
            
            # Check if prompt has been seen
            if prompt in state.seen_prompts:
                return None
                
            # Maintain memory before generating new dream
            self._maintain_memory()
            
            # Generate dream tensor (placeholder for actual implementation)
            dream_tensor = torch.randn(512)  # Example size
            
            dream_entry = {
                "prompt": prompt,
                "tensor": dream_tensor,
                "timestamp": time.time()
            }
            
            if self._add_to_memory(dream_entry):
                state.seen_prompts.add(prompt)
                return dream_entry
                
            return None
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, {"operation": "dream", "prompt": prompt})
            return None
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            state = self.state_manager.get_current_state()
            return state.get_dream_memory_stats()
            
        except Exception as e:
            self.error_manager.handle_memory_error(e, {"operation": "get_memory_stats"})
            return {}

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
        temperament_system: Optional[TemperamentSystem] = None
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.curiosity_manager = curiosity_manager
        self.confidence_calculator = confidence_calculator
        self.temperament_system = temperament_system
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
        """Execute a single training step with scaffold support and temperament awareness."""
        try:
            start_time = time.time()
            
            # Prepare batch data
            batch_size = len(batch)
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            
            # Get scaffold context if available
            scaffold_context = None
            if scaffold_provider:
                scaffold_start_time = time.time()
                scaffold_context = scaffold_provider(batch)
                scaffold_time = time.time() - scaffold_start_time
            
            # Adjust learning rate based on confidence and temperament if available
            if self.confidence_calculator and self.temperament_system and not dry_run:
                try:
                    confidence = self.confidence_calculator.get_current_confidence()
                    mood = self.temperament_system.mood_label
                    
                    # Adjust learning rate based on confidence and mood
                    confidence_factor = 1.0 + confidence * 0.1
                    mood_factor = 1.0
                    if mood == "Cautious":
                        mood_factor = 0.8  # Reduce learning rate for cautious mood
                    elif mood == "Curious":
                        mood_factor = 1.2  # Increase learning rate for curious mood
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * confidence_factor * mood_factor
                except Exception as e:
                    self.logger.warning(f"Failed to adjust learning rate based on confidence and temperament: {str(e)}")
            
            # Forward pass
            forward_start_time = time.time()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_context=scaffold_context
            )
            forward_time = time.time() - forward_start_time
            
            # Calculate loss
            loss = outputs.loss
            if self.config.grad_accum_steps > 1:
                loss = loss / self.config.grad_accum_steps
            
            # Backward pass
            backward_start_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_start_time
            
            # Optimizer step
            optimizer_start_time = time.time()
            if (self.global_step + 1) % self.config.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            optimizer_time = time.time() - optimizer_start_time
            
            # Calculate metrics
            metrics = {
                "loss": loss.item(),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": batch_size,
                "grad_norm": self._calculate_grad_norm(),
                "timing": {
                    "total": time.time() - start_time,
                    "scaffold": scaffold_time if scaffold_provider else None,
                    "forward": forward_time,
                    "backward": backward_time,
                    "optimizer": optimizer_time
                },
                "memory": {
                    "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                    "reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else None
                }
            }
            
            # Update confidence and temperament metrics if available
            if self.confidence_calculator and self.temperament_system and not dry_run:
                try:
                    # Generate embeddings for confidence calculation
                    with torch.no_grad():
                        embeddings = self.model.get_input_embeddings()(input_ids)
                        mean_embeddings = embeddings.mean(dim=1)
                    
                    # Update confidence with batch information
                    self.confidence_calculator.update_metrics(
                        metric_name="training_batch",
                        value=loss.item(),
                        embeddings=mean_embeddings
                    )
                    
                    # Add confidence and temperament metrics to training metrics
                    metrics["confidence_score"] = self.confidence_calculator.get_current_confidence()
                    metrics["temperament_mood"] = self.temperament_system.mood_label
                    metrics["temperament_score"] = self.temperament_system.current_score
                except Exception as e:
                    self.logger.warning(f"Failed to update confidence and temperament metrics: {str(e)}")
            
            # Log metrics
            self.logger.record({
                "event": "training_step",
                "step": self.global_step,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            self.global_step += 1
            return loss.item(), metrics
            
        except Exception as e:
            self.logger.record({
                "error": f"Training step failed: {str(e)}",
                "step": self.global_step,
                "batch_size": len(batch),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _calculate_grad_norm(self) -> float:
        """Calculate the gradient norm across all parameters."""
        try:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        except Exception as e:
            self.logger.record({
                "warning": f"Failed to calculate gradient norm: {str(e)}",
                "timestamp": time.time()
            })
            return 0.0

    def run_training_cycle(self, batch: List[Dict[str, Any]], scaffold_provider: Optional[ScaffoldProvider] = None) -> Tuple[float, Dict[str, Any]]:
        """Run a complete training cycle."""
        try:
            start_time = time.time()
            loss, metrics = self.train_step_with_scaffold(batch, scaffold_provider)
            
            # Log cycle metrics
            self.logger.record({
                "event": "training_cycle_complete",
                "step": self.global_step,
                "loss": loss,
                "metrics": metrics,
                "cycle_time": time.time() - start_time,
                "timestamp": time.time()
            })
            
            return loss, metrics
            
        except Exception as e:
            self.logger.record({
                "error": f"Training cycle failed: {str(e)}",
                "step": self.global_step,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def validate(self, data: Union[List[dict], Any], scaffold_provider: Optional[Callable] = None) -> Tuple[float, Dict[str, float]]:
        """Validate model performance with curiosity awareness."""
        self.model.eval()
        total_loss, total_correct, total_tokens, batches = 0.0, 0, 0, 0
        metrics = {metric: 0.0 for metric in self.config.metrics_to_track}
        data_iter = data if not isinstance(data, (list, tuple)) else [
            data[i:i + self.config.batch_size] for i in range(0, len(data), self.config.batch_size)
        ]

        with torch.no_grad():
            for batch in data_iter:
                if isinstance(batch, (list, tuple)):
                    batch = collate_batch(batch, self.tokenizer.pad_token_id, self.config.max_seq_length, self.tokenizer)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                scaffold_context = scaffold_provider(batch) if scaffold_provider else None
                if scaffold_context:
                    scaffold_context = scaffold_context.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.use_amp else torch.bfloat16):
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = self.loss_fn(outputs.logits, batch["labels"])

                total_loss += loss.item()
                batches += 1

                if "accuracy" in metrics:
                    preds = outputs.logits.argmax(dim=-1)
                    mask = batch["labels"] != -100
                    total_correct += (preds[mask] == batch["labels"][mask]).sum().item()
                    total_tokens += mask.sum().item()
                    metrics["accuracy"] = total_correct / total_tokens if total_tokens else 0.0
                if "perplexity" in metrics:
                    metrics["perplexity"] = torch.exp(loss).item()
                
                # Update curiosity metrics if available
                if self.curiosity_manager:
                    try:
                        # Generate embeddings for curiosity calculation
                        embeddings = self.model.get_input_embeddings()(batch["input_ids"])
                        mean_embeddings = embeddings.mean(dim=1)
                        
                        # Update curiosity with validation batch information
                        self.curiosity_manager.update_metrics(
                            metric_name="validation_batch",
                            value=loss.item(),
                            embeddings=mean_embeddings
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update curiosity metrics during validation: {str(e)}")

        avg_loss = total_loss / batches if batches else 0.0
        metrics["loss"] = avg_loss
        return avg_loss, metrics

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
    """Manages the orchestration of training cycles, including sleep training."""
    
    def __init__(
        self,
        config: SOVLConfig,
        logger: Logger,
        device: torch.device,
        state_manager: StateManager,
        curiosity_manager: CuriosityManager
    ):
        self.config = config
        self.logger = logger
        self.device = device
        self.state_manager = state_manager
        self.curiosity_manager = curiosity_manager
        self.data_manager = data_manager

        # Training configuration
        training_config = config.get("training_config", {})
        self.learning_rate = training_config.get("learning_rate", 1.5e-5)
        self.batch_size = training_config.get("batch_size", 4)
        self.grad_accum_steps = training_config.get("grad_accum_steps", 4)  # Renamed from accumulation_steps
        self.warmup_steps = training_config.get("warmup_steps", 300)
        self.model_name = training_config.get("model_name", "SmolLM2-360M") 
        
         # Dream memory configuration
        dream_memory_config = config.get("dream_memory_config", {})
        self.max_dream_memories = dream_memory_config.get("max_memories", 100)
        self.dream_base_weight = dream_memory_config.get("base_weight", 0.1)
        self.dream_max_weight = dream_memory_config.get("max_weight", 1.5)
        
        self.logger.record_event(
            event_type="training_cycle_manager_init",
            message="TrainingCycleManager initialized",
            level="info",
            additional_info={
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
                "model_name": self.model_name,
                "max_dream_memories": self.max_dream_memories
            }
        ) # New parameter
             # Initialize trainer with current state
        self.trainer = SOVLTrainer(
            config=config,
            state=state_manager.get_state(),
            curiosity_manager=curiosity_manager,
            error_manager=ErrorManager(),
            device=device,
            logger=logger
        )
        
    def run_training_cycle(
        self,
        train_data: List[Dict[str, Any]],
        valid_data: List[Dict[str, Any]],
        scaffold_provider: Optional[Callable] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a complete training cycle with validation.

        Args:
            train_data: List of training data dictionaries.
            valid_data: List of validation data dictionaries.
            scaffold_provider: Optional provider for scaffold context.
            epochs: Number of training epochs, defaults to config value.
            batch_size: Size of each training batch, defaults to config value.

        Returns:
            Dict containing training results or status if cycle is skipped.

        Raises:
            Exception: If training fails due to unexpected errors.
        """
        try:
            # Get current state
            state = self.state_manager.get_state()
            
            # Update trainer state if needed
            if self.trainer.state != state:
                self.trainer.state = state
                self.trainer.curiosity_manager.set_state(state)
            
            # Get configuration parameters using type-safe keys
            max_epochs = epochs or self.config_manager.get(ConfigKeys.TRAINING_MAX_EPOCHS, 10)
            batch_size = batch_size or self.config_manager.get(ConfigKeys.TRAINING_BATCH_SIZE, 4)
            grad_accum_steps = self.config_manager.get(ConfigKeys.TRAINING_GRAD_ACCUM_STEPS, 4)
            model_name = self.config_manager.get(ConfigKeys.TRAINING_MODEL_NAME, "SmolLM2-360M")
            
            # Dream memory configuration
            max_dream_memories = self.config_manager.get(ConfigKeys.DREAM_MEMORY_MAX_MEMORIES, 100)
            dream_base_weight = self.config_manager.get("dream_memory_config.base_weight", 0.1)
            dream_max_weight = self.config_manager.get("dream_memory_config.max_weight", 1.5)
            
            # Manage dream memory (if applicable)
            if hasattr(self.trainer, 'dream_memory'):
                # Trim dream memory to max_memories
                if len(self.trainer.dream_memory) > max_dream_memories:
                    self.trainer.dream_memory = self.trainer.dream_memory[-max_dream_memories:]
                    self.logger.record_event(
                        event_type="dream_memory_trimmed",
                        message="Dream memory trimmed to max_memories",
                        level="info",
                        additional_info={
                            "max_memories": max_dream_memories,
                            "conversation_id": self.state.history.conversation_id
                        }
                    )
                
                # Apply weights to dream memory entries
                for memory in self.trainer.dream_memory:
                    memory["weight"] = min(
                        dream_max_weight,
                        max(dream_base_weight, memory.get("weight", dream_base_weight))
                    )
            
            # Run training cycle through trainer
            results = self.trainer.run_training_cycle(
                train_data=train_data,
                validation_data=valid_data,
                scaffold_provider=scaffold_provider,
                max_epochs=max_epochs,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,  # Pass updated parameter
                model_name=model_name  # Pass new parameter
            )
            
            # Save updated state
            self.state_manager.save_state()
            
            # Log successful training cycle
            self.logger.log_training_event(
                event_type="training_cycle_complete",
                epoch=max_epochs,
                batch_size=batch_size,
                data_exposure=results.get("data_exposure", 0.0),
                conversation_id=self.state.history.conversation_id,
                state_hash=self.state.state_hash,
                additional_info={
                    **results,
                    "grad_accum_steps": grad_accum_steps,
                    "model_name": model_name,
                    "dream_memory_size": len(self.trainer.dream_memory) if hasattr(self.trainer, 'dream_memory') else 0
                }
            )
            
            return results
            
        except Exception as e:
            self.logger.record({
                "error": f"Training cycle failed: {str(e)}",
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id
            })
            raise
            
    def run_sleep_training(self, log_entries: List[Dict[str, Any]]) -> None:
        """Run sleep training on dream-generated content."""
        try:
            if not self.config.enable_sleep_training:
                self.logger.record_event(
                    event_type="sleep_training_skipped",
                    message="Sleep training disabled",
                    level="info",
                    conversation_id=self.state.history.conversation_id,
                    state_hash=self.state.state_hash
                )
                return
                
            self.logger.record_event(
                event_type="sleep_training_start",
                message="Starting sleep training",
                level="info",
                conversation_id=self.state.history.conversation_id,
                state_hash=self.state.state_hash
            )
            
            # Get current state
            state = self.state_manager.get_state()
            
            # Update trainer state if needed
            if self.trainer.state != state:
                self.trainer.state = state
                self.trainer.curiosity_manager.set_state(state)
            
            # Run sleep training through trainer
            self.trainer.sleep_train(log_entries)
            self.trainer.last_trained = time.time()
            self.trainer.last_weight = self.trainer.get_life_curve_weight()
            
            # Update temperament if enabled
            if self.config.enable_temperament:
                self.trainer._update_temperament()
                self.trainer.last_temperament_score = self.trainer.temperament_system.score
                
            # Save updated state
            self.state_manager.save_state()
                
            self.logger.record_event(
                event_type="sleep_training_complete",
                message="Sleep training completed successfully",
                level="info",
                conversation_id=self.state.history.conversation_id,
                state_hash=self.state.state_hash,
                additional_info={
                    "last_weight": self.trainer.last_weight,
                    "last_temperament_score": self.trainer.last_temperament_score
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Sleep training failed: {str(e)}",
                error_type="sleep_training_error",
                stack_trace=traceback.format_exc(),
                conversation_id=self.state.history.conversation_id,
                state_hash=self.state.state_hash
            )
            raise

class SOVLTrainer:
    """Main trainer class for SOVL system."""
    
    def __init__(
        self,
        config: TrainingConfig,
        state: SOVLState,
        curiosity_manager: CuriosityManager,
        error_manager: ErrorManager,
        device: torch.device,
        logger: Logger
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            state: System state
            curiosity_manager: Required curiosity manager instance
            error_manager: Required error manager instance
            device: Device to use for training
            logger: Logger instance
        """
        # Validate state
        if not isinstance(state, SOVLState):
            raise ValueError("state must be an instance of SOVLState")
            
        # Validate curiosity manager
        if not isinstance(curiosity_manager, CuriosityManager):
            raise ValueError("curiosity_manager must be an instance of CuriosityManager")
            
        # Validate error manager
        if not isinstance(error_manager, ErrorManager):
            raise ValueError("error_manager must be an instance of ErrorManager")
            
        # Validate device
        if not isinstance(device, torch.device):
            raise ValueError("device must be an instance of torch.device")
            
        self.config = config
        self.state = state
        self.logger = logger
        self.curiosity_manager = curiosity_manager
        self.error_manager = error_manager
        self.device = device
        
        # Validate state synchronization
        if not hasattr(self.curiosity_manager, 'state') or self.curiosity_manager.state != state:
            self.curiosity_manager.set_state(state)
            
        # Initialize model and components
        self._initialize_model()
        self._initialize_components()
        
        # Log initialization
        self._log_event("trainer_initialized", {
            "state_hash": self.state.state_hash,
            "conversation_id": self.state.history.conversation_id,
            "device": str(self.device)
        })
        
    def _initialize_model(self) -> None:
        """Initialize model and optimizer."""
        try:
            # Initialize model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Move model to specified device
            self.model = self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error initializing model: {str(e)}",
                error_type="model_initialization_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "model_name": self.config.model_name,
                    "device": str(self.device)
                }
            )
            raise
            
    def _initialize_components(self) -> None:
        """Initialize training components."""
        try:
            # Initialize dream manager
            self.dream_manager = DreamManager(
                state_manager=self.state_manager,
                error_manager=self.error_manager
            )
            
            # Initialize lifecycle manager
            self.lifecycle_manager = LifecycleManager(
                config=self.state.config,
                model=self.model,
                state=self.state
            )
            
            # Initialize training workflow manager
            self.workflow_manager = TrainingWorkflowManager(
                trainer=self,
                event_handler=TrainingEventHandler(
                    logger=self.logger,
                    state=self.state
                )
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error initializing components: {str(e)}",
                error_type="component_initialization_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "state_hash": self.state.state_hash,
                    "conversation_id": self.state.history.conversation_id
                }
            )
            raise

    def train_step_with_scaffold(
        self,
        batch: List[Dict[str, Any]],
        scaffold_provider: Optional[ScaffoldProvider] = None,
        dry_run: bool = False,
        dry_run_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute a single training step with scaffold support."""
        try:
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
                except Exception as e:
                    self.error_manager.handle_scaffold_error(e, {
                        "batch_size": len(batch),
                        "step": self.global_step
                    })
                    raise
            
            # Forward pass
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
        """Prepare batch for training."""
        try:
            # ... existing batch preparation code ...
            pass
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
        """Execute forward pass."""
        try:
            # ... existing forward pass code ...
            pass
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
