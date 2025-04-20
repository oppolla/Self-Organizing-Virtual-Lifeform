from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass, field
import torch
import uuid
from threading import Lock
import time
import re
import traceback
import hashlib
import json
import os
import threading
from datetime import datetime
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, safe_divide, safe_compare, synchronized
from sovl_records import ConfidenceHistory
from sovl_memory import MemoriaManager, RAMManager, GPUMemoryManager

class StateError(Exception):
    """Raised for invalid state operations or data."""
    pass

@dataclass
class CuriosityConfig:
    """Configuration for curiosity-related parameters."""
    max_questions: int
    max_novelty_scores: int
    decay_rate: float
    hidden_size: int
    question_timeout: float

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.max_questions > 0, "max_questions must be positive"
        assert self.max_novelty_scores > 0, "max_novelty_scores must be positive"
        assert 0 <= self.decay_rate <= 1, "decay_rate must be in [0, 1]"
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.question_timeout > 0, "question_timeout must be positive"

@dataclass
class ConversationConfig:
    """Configuration for conversation-related parameters."""
    max_messages: int

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.max_messages > 0, "max_messages must be positive"

@dataclass
class SOVLConfig:
    """Configuration for SOVL state parameters."""
    # Core configuration
    dream_memory_maxlen: int
    temperament_history_maxlen: int
    confidence_history_maxlen: int
    hidden_size: int
    max_seen_prompts: int
    quantization_mode: str
    sleep_max_steps: int
    prompt_timeout: float
    temperament_decay_rate: float
    scaffold_unk_id: int
    lora_capacity: int
    max_dream_memory_mb: float = 512.0
    
    # Temperament configuration
    temperament_melancholy_noise: float = 0.1
    temperament_influence: float = 0.3
    temperament_base_temperature: float = 0.7
    temperament_swing_threshold: float = 0.2
    temperament_stability_threshold: float = 0.1
    
    # Training configuration
    learning_rate: float = 2e-5
    grad_accum_steps: int = 4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    total_steps: int = 100000
    max_grad_norm: float = 1.0
    use_amp: bool = True
    max_patience: int = 2
    batch_size: int = 2
    max_epochs: int = 3
    validate_every_n_steps: int = 100
    checkpoint_interval: int = 1000
    checkpoint_path: str = "checkpoints/sovl_trainer"
    scheduler_type: str = "linear"
    cosine_min_lr: float = 1e-6
    warmup_ratio: float = 0.1
    dropout_rate: float = 0.1
    max_seq_length: int = 512
    metrics_to_track: List[str] = field(default_factory=lambda: ["loss", "accuracy", "confidence"])
    
    # Lifecycle configuration
    enable_gestation: bool = True
    enable_sleep_training: bool = True
    enable_lifecycle_weighting: bool = True
    lifecycle_capacity_factor: float = 0.01
    lifecycle_curve: str = "sigmoid_linear"
    sleep_conf_threshold: float = 0.7
    sleep_log_min: int = 10
    exposure_gain_eager: int = 3
    exposure_gain_default: int = 2
    
    # Dream configuration
    dream_memory_weight: float = 0.1
    enable_dreaming: bool = True
    repetition_n: int = 3
    sigmoid_scale: float = 0.5
    sigmoid_shift: float = 5.0
    dream_noise_scale: float = 0.05
    dream_prompt_weight: float = 0.5
    dream_novelty_boost: float = 0.03
    dream_memory_decay: float = 0.95
    dream_prune_threshold: float = 0.1
    temp_melancholy_noise: float = 0.02
    enable_prompt_driven_dreams: bool = True
    dream_swing_var: float = 0.1
    dream_lifecycle_delta: float = 0.1
    dream_temperament_on: bool = True
    
    # Memory configuration
    memory_threshold: float = 0.85
    memory_decay_rate: float = 0.95
    use_scaffold_memory: bool = True
    use_token_map_memory: bool = True
    scaffold_weight: float = 1.0
    
    # Curiosity configuration
    weight_ignorance: float = 0.3
    weight_novelty: float = 0.7
    metrics_maxlen: int = 100
    novelty_threshold_spontaneous: float = 0.7
    novelty_threshold_curious: float = 0.5
    curiosity_decay_rate: float = 0.95
    curiosity_queue_maxlen: int = 10
    curiosity_question_timeout: float = 3600.0

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'SOVLConfig':
        """Create SOVLConfig instance from ConfigManager."""
        try:
            config = cls(
                **{key: config_manager.get(f"{section}.{key}", default)
                   for section, keys in {
                       "controls_config": [
                           "dream_memory_maxlen", "temperament_history_maxlen", "confidence_history_maxlen",
                           "max_seen_prompts", "prompt_timeout", "temperament_decay_rate", "scaffold_unk_id"
                       ],
                       "core_config": ["hidden_size", "quantization"],
                       "training_config": [
                           "sleep_max_steps", "learning_rate", "grad_accum_steps", "weight_decay",
                           "warmup_steps", "total_steps", "max_grad_norm", "use_amp", "max_patience",
                           "batch_size", "max_epochs", "validate_every_n_steps", "checkpoint_interval",
                           "checkpoint_path", "scheduler_type", "cosine_min_lr", "warmup_ratio",
                           "dropout_rate", "max_seq_length", "metrics_to_track", "lora_capacity"
                       ],
                       "lifecycle_config": [
                           "enable_gestation", "enable_sleep_training", "enable_lifecycle_weighting",
                           "lifecycle_capacity_factor", "lifecycle_curve", "sleep_conf_threshold",
                           "sleep_log_min", "exposure_gain_eager", "exposure_gain_default"
                       ],
                       "dream_config": [
                           "dream_memory_weight", "enable_dreaming", "repetition_n", "sigmoid_scale",
                           "sigmoid_shift", "dream_noise_scale", "dream_prompt_weight", "dream_novelty_boost",
                           "dream_memory_decay", "dream_prune_threshold", "temp_melancholy_noise",
                           "enable_prompt_driven_dreams", "dream_swing_var", "dream_lifecycle_delta",
                           "dream_temperament_on"
                       ],
                       "memory_config": [
                           "memory_threshold", "memory_decay_rate", "use_scaffold_memory",
                           "use_token_map_memory", "scaffold_weight"
                       ],
                       "curiosity_config": [
                           "weight_ignorance", "weight_novelty", "metrics_maxlen",
                           "novelty_threshold_spontaneous", "novelty_threshold_curious",
                           "curiosity_decay_rate", "curiosity_queue_maxlen", "curiosity_question_timeout"
                       ]
                   }.items()
                   for key, default in cls.__dataclass_fields__.items()
                   if key in keys}
            )
            config.validate()
            return config
        except Exception as e:
            raise StateError(f"Failed to create SOVLConfig: {str(e)}")

    def validate(self) -> None:
        """Validate configuration parameters."""
        try:
            # Core configuration
            assert self.dream_memory_maxlen > 0, "dream_memory_maxlen must be positive"
            assert self.temperament_history_maxlen > 0, "temperament_history_maxlen must be positive"
            assert self.confidence_history_maxlen > 0, "confidence_history_maxlen must be positive"
            assert self.hidden_size > 0, "hidden_size must be positive"
            assert self.max_seen_prompts > 0, "max_seen_prompts must be positive"
            assert self.quantization_mode in ["fp16", "fp32", "int8"], "invalid quantization_mode"
            assert self.sleep_max_steps > 0, "sleep_max_steps must be positive"
            assert self.prompt_timeout > 0, "prompt_timeout must be positive"
            assert 0 <= self.temperament_decay_rate <= 1, "temperament_decay_rate must be in [0, 1]"
            assert self.scaffold_unk_id >= 0, "scaffold_unk_id must be non-negative"
            assert self.lora_capacity >= 0, "lora_capacity must be non-negative"
            assert self.max_dream_memory_mb > 0, "max_dream_memory_mb must be positive"
            
            # Temperament configuration
            assert 0 <= self.temperament_melancholy_noise <= 1, "temperament_melancholy_noise must be in [0, 1]"
            assert 0 <= self.temperament_influence <= 1, "temperament_influence must be in [0, 1]"
            assert 0 <= self.temperament_base_temperature <= 2, "temperament_base_temperature must be in [0, 2]"
            assert 0 <= self.temperament_swing_threshold <= 1, "temperament_swing_threshold must be in [0, 1]"
            assert 0 <= self.temperament_stability_threshold <= 1, "temperament_stability_threshold must be in [0, 1]"
            
            # Training configuration
            assert self.learning_rate > 0, "learning_rate must be positive"
            assert self.grad_accum_steps >= 1, "grad_accum_steps must be at least 1"
            assert self.max_grad_norm > 0, "max_grad_norm must be positive"
            assert self.scheduler_type in ["linear", "cosine", "constant"], "invalid scheduler_type"
            assert self.lifecycle_curve in ["sigmoid_linear", "exponential"], "invalid lifecycle_curve"
            
            # Dream configuration
            assert self.repetition_n >= 2, "repetition_n must be at least 2"
            assert self.sigmoid_scale > 0, "sigmoid_scale must be positive"
            assert self.sigmoid_shift >= 0, "sigmoid_shift must be non-negative"
            assert self.dream_noise_scale >= 0, "dream_noise_scale must be non-negative"
            assert 0 <= self.dream_prompt_weight <= 1, "dream_prompt_weight must be in [0, 1]"
            assert self.dream_novelty_boost >= 0, "dream_novelty_boost must be non-negative"
            assert 0 <= self.dream_memory_decay <= 1, "dream_memory_decay must be in [0, 1]"
            assert 0 <= self.dream_prune_threshold <= 1, "dream_prune_threshold must be in [0, 1]"
            assert self.dream_swing_var >= 0, "dream_swing_var must be non-negative"
            assert self.dream_lifecycle_delta >= 0, "dream_lifecycle_delta must be non-negative"
            
            # Memory configuration
            assert 0 <= self.memory_threshold <= 1, "memory_threshold must be in [0, 1]"
            assert 0 <= self.memory_decay_rate <= 1, "memory_decay_rate must be in [0, 1]"
            assert 0 <= self.scaffold_weight <= 1, "scaffold_weight must be in [0, 1]"
            
            # Curiosity configuration
            assert 0 <= self.weight_ignorance <= 1, "weight_ignorance must be in [0, 1]"
            assert 0 <= self.weight_novelty <= 1, "weight_novelty must be in [0, 1]"
            assert self.metrics_maxlen > 0, "metrics_maxlen must be positive"
            assert 0 <= self.novelty_threshold_spontaneous <= 1, "novelty_threshold_spontaneous must be in [0, 1]"
            assert 0 <= self.novelty_threshold_curious <= 1, "novelty_threshold_curious must be in [0, 1]"
            assert 0 <= self.curiosity_decay_rate <= 1, "curiosity_decay_rate must be in [0, 1]"
            assert self.curiosity_queue_maxlen > 0, "curiosity_queue_maxlen must be positive"
            assert self.curiosity_question_timeout > 0, "curiosity_question_timeout must be positive"
        except AssertionError as e:
            raise StateError(f"Configuration validation failed: {str(e)}")

@dataclass
class TrainingState:
    """Manages training-specific state and metrics."""
    last_trained: float = 0.0
    last_weight: float = 0.0
    sleep_confidence_sum: float = 0.0
    sleep_confidence_count: int = 0
    data_exposure: float = 0.0
    lora_capacity: float = 0.0
    gestation_metrics: Dict[str, Any] = field(default_factory=dict)
    dream_metrics: Dict[str, Any] = field(default_factory=dict)
    sleep_metrics: Dict[str, Any] = field(default_factory=dict)
    data_quality_metrics: Dict[str, Any] = field(default_factory=lambda: {
        'avg_input_length': 0.0,
        'avg_output_length': 0.0,
        'pair_completeness': 0.0,
        'last_validation_time': 0.0,
        'validation_errors': []
    })

    def update_gestation_metrics(self, batch_size: int, avg_loss: float) -> None:
        """Update gestation training metrics."""
        self.gestation_metrics.update({
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time()
        })

    def update_dream_metrics(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Update dream cycle metrics."""
        self.dream_metrics.update({
            "dream_prompt": dream_prompt,
            "is_novel": is_novel,
            "memory_count": memory_count,
            "timestamp": time.time()
        })

    def update_sleep_metrics(self, batch_size: int, data_exposure: float) -> None:
        """Update sleep training metrics."""
        self.sleep_metrics.update({
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time()
        })

    def update_data_exposure(self, exposure: float) -> None:
        """Update data exposure."""
        self.data_exposure = exposure

    def update_data_quality(self, formatted_training_data: List[Dict[str, str]]) -> None:
        """Update data quality metrics."""
        if not formatted_training_data:
            return
            
        total_pairs = len(formatted_training_data)
        valid_pairs = 0
        total_input_length = 0
        total_output_length = 0
        validation_errors = []
        
        for pair in formatted_training_data:
            try:
                if not isinstance(pair, dict):
                    validation_errors.append("Invalid pair type")
                    continue
                    
                if 'input' not in pair or 'output' not in pair:
                    validation_errors.append("Missing required fields")
                    continue
                    
                if not isinstance(pair['input'], str) or not isinstance(pair['output'], str):
                    validation_errors.append("Invalid field types")
                    continue
                    
                if not pair['input'].strip() or not pair['output'].strip():
                    validation_errors.append("Empty content")
                    continue
                    
                valid_pairs += 1
                total_input_length += len(pair['input'])
                total_output_length += len(pair['output'])
                
            except Exception as e:
                validation_errors.append(str(e))
                
        self.data_quality_metrics.update({
            'avg_input_length': total_input_length / valid_pairs if valid_pairs > 0 else 0,
            'avg_output_length': total_output_length / valid_pairs if valid_pairs > 0 else 0,
            'pair_completeness': valid_pairs / total_pairs if total_pairs > 0 else 0,
            'last_validation_time': time.time(),
            'validation_errors': validation_errors
        })

    def get_state_hash(self) -> str:
        """Generate a hash of the current training state."""
        state_dict = {
            "last_trained": self.last_trained,
            "last_weight": self.last_weight,
            "sleep_confidence_sum": self.sleep_confidence_sum,
            "sleep_confidence_count": self.sleep_confidence_count,
            "data_exposure": self.data_exposure,
            "lora_capacity": self.lora_capacity,
            "data_quality": self.data_quality_metrics
        }
        return hashlib.md5(json.dumps(state_dict, sort_keys=True).encode()).hexdigest()

class StateBase:
    """Base class for state management with common utilities."""
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()

    def log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(event_type=event_type, message=message, level=level, **kwargs)

    def log_error(self, error_msg: str, error_type: str = "state_error", **kwargs) -> None:
        """Log an error with stack trace."""
        self.logger.log_error(error_msg=error_msg, error_type=error_type, stack_trace=traceback.format_exc(), **kwargs)

    def validate_number(self, value: Any, name: str, min_value: Optional[float] = None) -> float:
        """Validate a numeric value."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}")
        return float(value)

    def validate_tensor(self, tensor: torch.Tensor, expected_dim: int, name: str) -> None:
        """Validate a tensor's shape."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid {name} type: {type(tensor)}")
        if tensor.shape[-1] != expected_dim:
            raise ValueError(f"{name} shape {tensor.shape} mismatches expected dimension {expected_dim}")

class CuriosityState(StateBase):
    """Manages curiosity-related state and question prioritization."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        super().__init__(config_manager, logger)
        self._config = self._load_curiosity_config()
        self.device = device
        self.unanswered_questions: Deque[Tuple[str, float, Optional[torch.Tensor]]] = deque(maxlen=self._config.max_questions)
        self.last_question_time: float = 0.0
        self.pressure: float = 0.0
        self.novelty_scores: Deque[float] = deque(maxlen=self._config.max_novelty_scores)
        self.question_count: int = 0
        self.log_event("curiosity_state_initialized", "Curiosity state initialized", config=str(self._config))

    def _load_curiosity_config(self) -> CuriosityConfig:
        """Load curiosity configuration."""
        config = CuriosityConfig(
            max_questions=self.config_manager.get("controls_config.curiosity_queue_maxlen", 10),
            max_novelty_scores=self.config_manager.get("controls_config.novelty_history_maxlen", 20),
            decay_rate=self.config_manager.get("controls_config.curiosity_decay_rate", 0.9),
            hidden_size=self.config_manager.get("core_config.hidden_size", 768),
            question_timeout=self.config_manager.get("controls_config.curiosity_question_timeout", 3600.0)
        )
        config.validate()
        return config

    @synchronized("lock")
    def update_question_history(self, question: str, timestamp: float) -> None:
        """Update question history and related state."""
        try:
            self.last_question_time = timestamp
            self.question_count += 1
            self.log_event(
                "question_history_updated", "Question history updated",
                question=question, timestamp=timestamp, question_count=self.question_count,
                queue_length=len(self.unanswered_questions)
            )
            self._update_pressure()
        except Exception as e:
            self.log_error(f"Failed to update question history: {str(e)}")
            raise StateError(f"Question history update failed: {str(e)}")

    @synchronized("lock")
    def add_question(self, question: str, score: float, context_vector: Optional[torch.Tensor] = None) -> None:
        """Add a new question with score and optional context vector."""
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string")
            score = self.validate_number(score, "Score", min_value=0.0)
            if context_vector is not None:
                self.validate_tensor(context_vector, self._config.hidden_size, "Context vector")
            with NumericalGuard():
                self.unanswered_questions.append((question, score, context_vector))
                self.question_count += 1
                self.last_question_time = time.time()
                self._update_pressure()
                self.log_event(
                    "question_added", "New question added to curiosity state",
                    question=question, score=score, has_context_vector=context_vector is not None,
                    question_count=self.question_count, queue_length=len(self.unanswered_questions)
                )
        except Exception as e:
            self.log_error(f"Failed to add question: {str(e)}", question=question, score=score)
            raise StateError(f"Add question failed: {str(e)}")

    @synchronized("lock")
    def prioritize_questions(self) -> None:
        """Sort unanswered questions by score."""
        try:
            sorted_questions = sorted(self.unanswered_questions, key=lambda x: x[1], reverse=True)
            self.unanswered_questions = deque(sorted_questions, maxlen=self._config.max_questions)
            self.log_event(
                "questions_prioritized", "Questions prioritized by score",
                question_count=len(self.unanswered_questions)
            )
        except Exception as e:
            self.log_error("Question prioritization failed")
            raise StateError(f"Question prioritization failed: {str(e)}")

    @synchronized("lock")
    def prune_old_questions(self, timeout: float) -> None:
        """Remove questions older than timeout."""
        try:
            current_time = time.time()
            while self.unanswered_questions and current_time - self.last_question_time > timeout:
                question, _, _ = self.unanswered_questions.popleft()
                self.log_event("old_question_pruned", "Old question pruned", question=question)
            self._update_pressure()
        except Exception as e:
            self.log_error("Question pruning failed")
            raise StateError(f"Question pruning failed: {str(e)}")

    @synchronized("lock")
    def _update_pressure(self) -> None:
        """Update curiosity pressure based on questions and novelty."""
        try:
            with NumericalGuard():
                base_pressure = safe_divide(
                    len(self.unanswered_questions), max(1, self._config.max_questions), logger=self.logger
                )
                novelty_avg = safe_divide(
                    sum(self.novelty_scores), max(1, len(self.novelty_scores)), logger=self.logger
                ) if self.novelty_scores else 0.0
                self.pressure = max(0.0, min(1.0, base_pressure * (1.0 + novelty_avg) * self._config.decay_rate))
                self.log_event(
                    "pressure_updated", "Curiosity pressure updated",
                    pressure=self.pressure, unanswered_count=len(self.unanswered_questions), novelty_avg=novelty_avg
                )
        except Exception as e:
            self.log_error("Pressure update failed")
            raise StateError(f"Pressure update failed: {str(e)}")

    @synchronized("lock")
    def add_novelty_score(self, score: float) -> None:
        """Add a novelty score and decay existing scores."""
        try:
            score = self.validate_number(score, "Novelty score", min_value=0.0)
            with NumericalGuard():
                self.novelty_scores.append(score * self._config.decay_rate)
                self._update_pressure()
                self.log_event(
                    "novelty_score_added", "Novelty score added",
                    score=score, novelty_scores_count=len(self.novelty_scores)
                )
        except Exception as e:
            self.log_error(f"Failed to add novelty score: {str(e)}", score=score)
            raise StateError(f"Add novelty score failed: {str(e)}")

    @synchronized("lock")
    def get_context_vector(self) -> Optional[torch.Tensor]:
        """Compute a weighted average of context vectors from questions."""
        try:
            if not self.unanswered_questions:
                return None
            vectors = [v for _, _, v in self.unanswered_questions if v is not None]
            scores = [s for _, s, v in self.unanswered_questions if v is not None]
            if not vectors:
                return None
            with NumericalGuard():
                weights = torch.tensor(scores, dtype=torch.float32)
                weights = weights / (weights.sum() + 1e-8)
                stacked = torch.stack(vectors)
                weighted_avg = (stacked * weights.view(-1, 1)).sum(dim=0)
                self.log_event(
                    "context_vector_computed", "Context vector computed from questions",
                    vector_shape=list(weighted_avg.shape), question_count=len(vectors)
                )
                return weighted_avg
        except Exception as e:
            self.log_error("Context vector computation failed")
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curiosity state to dictionary."""
        try:
            with self.lock:
                return {
                    "unanswered_questions": [
                        (q, s, v.cpu().numpy().tolist() if v is not None else None)
                        for q, s, v in self.unanswered_questions
                    ],
                    "last_question_time": self.last_question_time,
                    "pressure": self.pressure,
                    "novelty_scores": list(self.novelty_scores),
                    "question_count": self.question_count,
                    "version": "1.1"
                }
        except Exception as e:
            self.log_error("Curiosity state serialization failed")
            raise StateError(f"Curiosity state serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load curiosity state from dictionary."""
        try:
            with self.lock:
                version = data.get("version", "1.0")
                if version not in ["1.0", "1.1"]:
                    self.log_event(
                        "unsupported_version", f"Unsupported curiosity state version: {version}",
                        level="warning", version=version
                    )
                self.unanswered_questions = deque(maxlen=self._config.max_questions)
                for q, s, v in data.get("unanswered_questions", []):
                    context_vector = (
                        torch.tensor(v, dtype=torch.float32, device=self.device)
                        if v is not None and len(v) == self._config.hidden_size else None
                    )
                    self.unanswered_questions.append((q, float(s), context_vector))
                self.last_question_time = float(data.get("last_question_time", 0.0))
                self.pressure = float(data.get("pressure", 0.0))
                self.novelty_scores = deque(
                    [float(s) for s in data.get("novelty_scores", [])], maxlen=self._config.max_novelty_scores
                )
                self.question_count = int(data.get("question_count", 0))
                self.log_event(
                    "curiosity_state_loaded", "Curiosity state loaded from dictionary",
                    question_count=self.question_count, pressure=self.pressure, version=version
                )
        except Exception as e:
            self.log_error("Failed to load curiosity state", data_keys=list(data.keys()))
            raise StateError(f"Curiosity state loading failed: {str(e)}")

    def generate_curiosity_question(self, state, tokenizer, model, context, spontaneous: bool = False) -> Optional[str]:
        """Generate a curiosity-driven question based on the current state."""
        try:
            if not self.config_manager.get("curiosity_enabled", True):
                return None
            question = "What is the meaning of life?"  # Placeholder logic
            self.log_event(
                "curiosity_question_generated", "Curiosity-driven question generated",
                question=question, spontaneous=spontaneous
            )
            return question
        except Exception as e:
            self.log_error("Failed to generate curiosity question")
            return None

    def check_silence(self, state, tokenizer, model, context) -> None:
        """Check for prolonged silence and generate a question if needed."""
        try:
            current_time = time.time()
            if current_time - self.last_question_time > self._config.question_timeout:
                question = self.generate_curiosity_question(state, tokenizer, model, context, spontaneous=True)
                if question:
                    print(f"Curiosity Question: {question}")
                    self.last_question_time = current_time
        except Exception as e:
            self.log_error("Failed to check silence")

    def tune_curiosity(self, pressure: Optional[float] = None, decay_rate: Optional[float] = None, question_timeout: Optional[float] = None) -> None:
        """Tune curiosity parameters."""
        try:
            with self.lock:
                if pressure is not None:
                    self.pressure = self.validate_number(pressure, "Pressure", min_value=0.0)
                if decay_rate is not None:
                    self._config.decay_rate = self.validate_number(decay_rate, "Decay rate", min_value=0.0)
                if question_timeout is not None:
                    self._config.question_timeout = self.validate_number(question_timeout, "Question timeout", min_value=0.0)
                self.log_event(
                    "curiosity_tuned", "Curiosity parameters tuned",
                    pressure=self.pressure, decay_rate=self._config.decay_rate, question_timeout=self._config.question_timeout
                )
        except Exception as e:
            self.log_error("Failed to tune curiosity")
            raise StateError(f"Tune curiosity failed: {str(e)}")

    def reset_for_conversation(self, conversation_id: str) -> None:
        """Reset curiosity state for a new conversation."""
        try:
            with self.lock:
                self.unanswered_questions.clear()
                self.last_question_time = time.time()
                self.pressure = 0.0
                self.novelty_scores.clear()
                self.question_count = 0
                self.log_event(
                    "curiosity_reset", "Curiosity state reset for new conversation",
                    conversation_id=conversation_id
                )
        except Exception as e:
            self.log_error("Failed to reset curiosity state")
            raise StateError(f"Reset curiosity failed: {str(e)}")

class ConversationHistory:
    """Manages conversation messages with unique ID."""
    def __init__(self, maxlen: int, conversation_id: Optional[str] = None):
        self._config = ConversationConfig(max_messages=maxlen)
        self._config.validate()
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages: Deque[Dict[str, str]] = deque(maxlen=self._config.max_messages)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        try:
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Role must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError("Content must be a string")
            self.messages.append({"role": role, "content": content})
        except Exception as e:
            raise StateError(f"Failed to add message: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation history to dictionary."""
        return {"conversation_id": self.conversation_id, "messages": list(self.messages)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], maxlen: int) -> 'ConversationHistory':
        """Create conversation history from dictionary."""
        history = cls(maxlen=maxlen, conversation_id=data.get("conversation_id"))
        for msg in data.get("messages", []):
            history.add_message(msg["role"], msg["content"])
        return history

@dataclass
class DataStats:
    """Tracks data loading and quality statistics."""
    total_entries: int = 0
    valid_entries: int = 0
    invalid_entries: int = 0
    last_load_time: float = 0.0
    average_entry_length: float = 0.0
    validation_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    data_quality_score: float = 0.0
    data_diversity_score: float = 0.0
    last_update_time: float = 0.0

    def update(self, total_entries: int, valid_entries: int, invalid_entries: int,
              validation_errors: Dict[str, int], average_entry_length: float) -> None:
        """Update data statistics."""
        self.total_entries = total_entries
        self.valid_entries = valid_entries
        self.invalid_entries = invalid_entries
        self.last_load_time = time.time()
        self.average_entry_length = average_entry_length
        self.validation_errors = validation_errors
        self.last_update_time = time.time()
        self.data_quality_score = valid_entries / total_entries if total_entries > 0 else 0.0
        self.data_diversity_score = min(1.0, average_entry_length / 1000.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_entries": self.total_entries, "valid_entries": self.valid_entries,
            "invalid_entries": self.invalid_entries, "last_load_time": self.last_load_time,
            "average_entry_length": self.average_entry_length, "validation_errors": dict(self.validation_errors),
            "data_quality_score": self.data_quality_score, "data_diversity_score": self.data_diversity_score,
            "last_update_time": self.last_update_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataStats':
        """Create from dictionary."""
        stats = cls()
        stats.total_entries = data.get("total_entries", 0)
        stats.valid_entries = data.get("valid_entries", 0)
        stats.invalid_entries = data.get("invalid_entries", 0)
        stats.last_load_time = data.get("last_load_time", 0.0)
        stats.average_entry_length = data.get("average_entry_length", 0.0)
        stats.validation_errors = defaultdict(int, data.get("validation_errors", {}))
        stats.data_quality_score = data.get("data_quality_score", 0.0)
        stats.data_diversity_score = data.get("data_diversity_score", 0.0)
        stats.last_update_time = data.get("last_update_time", 0.0)
        return stats

class SOVLState(StateBase):
    """Manages the state of the SOVL system."""
    STATE_VERSION = "1.0"

    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        """Initialize SOVL state with configuration and dependencies."""
        super().__init__(config_manager, logger)
        self._device = device
        self.memoria_manager = MemoriaManager(config_manager, logger)
        self.ram_manager = RAMManager(config_manager, logger)
        self.gpu_manager = GPUMemoryManager(config_manager, logger)
        
    def _update_memory_usage(self) -> None:
        """Update memory usage statistics."""
        try:
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.get_gpu_usage()
            
            self._logger.record_event(
                event_type="memory_usage_updated",
                message="Memory usage statistics updated",
                level="info",
                additional_info={
                    "ram_health": ram_health,
                    "gpu_health": gpu_health
                }
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to update memory usage: {str(e)}",
                error_type="memory_update_error",
                stack_trace=traceback.format_exc()
            )

    def _initialize_state(self) -> None:
        """Initialize state components."""
        try:
            self.seen_prompts = set()
            self.temperament_score = 0.0
            self.last_temperament_score = 0.0
            self.temperament_history = deque(maxlen=self.config_manager.get("controls_config.temperament_history_maxlen", 5))
            self.dream_memory = deque(maxlen=self.config_manager.get("controls_config.dream_memory_maxlen", 10))  # Ensure maxlen is set correctly
            self.total_dream_memory_mb = 0.0
            self.history = ConversationHistory(
                maxlen=self.config_manager.get("controls_config.max_messages", 100),
                conversation_id=str(uuid.uuid4())
            )
            self.data_stats = DataStats()
            self.version = self.STATE_VERSION
            self._memory_threshold = self.config_manager.get("memory_config.memory_threshold", 0.85)
            self._memory_decay_rate = self.config_manager.get("memory_config.memory_decay_rate", 0.95)
            self.log_event(
                "state_initialized", "SOVL state components initialized",
                conversation_id=self.history.conversation_id, state_hash=self.state_hash()
            )
        except Exception as e:
            self.log_error(f"Failed to initialize state components: {str(e)}", error_type="state_component_initialization_error")
            raise

    @synchronized("lock")
    def _prune_dream_memory(self) -> None:
        """Prune dream memory if it exceeds memory threshold."""
        try:
            max_memory = self.config_manager.get("memory_config.max_dream_memory_mb", 512.0)
            if self.total_dream_memory_mb <= max_memory:
                return
            sorted_memories = sorted(self.dream_memory, key=lambda x: (x["weight"], x["timestamp"]), reverse=True)
            keep_count = int(len(sorted_memories) * self._memory_decay_rate)
            self.dream_memory = deque(sorted_memories[:keep_count], maxlen=self.dream_memory.maxlen)
            self._update_memory_usage()
            self.log_event(
                "dream_memory_pruned", "Dream memory pruned due to size constraints",
                before_size=self.total_dream_memory_mb, after_size=self._calculate_memory_usage(),
                kept_memories=len(self.dream_memory)
            )
        except Exception as e:
            self.log_error(f"Failed to prune dream memory: {str(e)}", error_type="memory_pruning_error")
            raise

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {"gpu_allocated": None, "gpu_reserved": None, "gpu_memory_percent": None}
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                stats.update({
                    "gpu_allocated": allocated,
                    "gpu_reserved": reserved,
                    "gpu_memory_percent": (allocated / total) * 100 if total > 0 else None
                })
            except Exception as e:
                self.log_event("memory_stats_failed", f"Failed to get GPU memory stats: {str(e)}", level="warning")
        return stats

    def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage percentage."""
        try:
            return self._get_memory_stats().get("gpu_memory_percent", 0.0)
        except Exception as e:
            self.log_event("memory_calculation_failed", f"Failed to calculate memory usage: {str(e)}", level="warning")
            return 0.0

    def _compress_tensor(self, tensor: torch.Tensor, target_device: Optional[str] = None) -> Dict[str, Any]:
        """Compress tensor for storage.
        
        Args:
            tensor: The tensor to compress
            target_device: Optional target device to move tensor to before compression
                          If None, tensor will be moved to CPU
        """
        try:
            with NumericalGuard():
                # Move tensor to target device if specified, otherwise move to CPU
                if target_device is not None:
                    tensor = tensor.to(target_device)
                else:
                    tensor = tensor.cpu()
                np_array = tensor.numpy()
                return {"shape": np_array.shape, "dtype": str(np_array.dtype), "data": np_array.tobytes()}
        except Exception as e:
            self.log_error(f"Failed to compress tensor: {str(e)}", error_type="tensor_compression_error")
            raise

    def _decompress_tensor(self, compressed: Dict[str, Any], target_device: Optional[str] = None) -> torch.Tensor:
        """Decompress tensor from storage.
        
        Args:
            compressed: The compressed tensor data
            target_device: Optional target device to move tensor to after decompression
                          If None, tensor will be moved to the instance's device
        """
        try:
            with NumericalGuard():
                np_array = np.frombuffer(compressed["data"], dtype=np.dtype(compressed["dtype"])).reshape(compressed["shape"])
                tensor = torch.tensor(np_array)
                # Move tensor to target device if specified, otherwise use instance's device
                if target_device is not None:
                    tensor = tensor.to(target_device)
                else:
                    tensor = tensor.to(self._device)
                return tensor
        except Exception as e:
            self.log_error(f"Failed to decompress tensor: {str(e)}", error_type="tensor_decompression_error")
            raise

    def add_confidence(self, confidence: float) -> None:
        """Add a confidence score to the history."""
        self._confidence_history.add_confidence(confidence)

    def get_confidence_history(self) -> Deque[float]:
        """Get the confidence history."""
        return self._confidence_history.get_confidence_history()

    def clear_confidence_history(self) -> None:
        """Clear the confidence history."""
        self._confidence_history.clear_history()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        with self._lock:
            try:
                # Get base state data
                state_data = {
                    "version": self.STATE_VERSION,
                    "confidence_history": list(self._confidence_history.get_history()),
                    "dream_memory": [self._compress_tensor(tensor) for tensor in self._dream_memory],
                    "temperament_history": list(self._temperament_history),
                    "seen_prompts": list(self._seen_prompts),
                    "training_state": self._training_state.__dict__,
                    "conversation_metadata": self._conversation_metadata
                }
                
                # Use MemoriaManager to save state
                self.memoria_manager.save_state("state")
                
                # Use storage classes to save their respective data
                self._dream_storage.load_state([self._compress_tensor(tensor) for tensor in self._dream_memory])
                self._token_map_storage.load_state(self._token_map)
                self._state_storage.load_state(state_data)
                
                return state_data
            except Exception as e:
                self.log_error(f"Failed to convert state to dict: {str(e)}", "state_serialization_error")
                raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_manager: ConfigManager, logger: Logger, device: torch.device) -> 'SOVLState':
        """Create SOVLState instance from dictionary."""
        try:
            # Create new state instance
            state = cls(config_manager, logger, device)
            
            # Use MemoriaManager to load state
            state.memoria_manager.load_state("state")
            
            # Use storage classes to load their respective data
            dream_memory_data = state._dream_storage.get_state()
            token_map_data = state._token_map_storage.get_state()
            state_data = state._state_storage.get_state()
            
            # Update state with loaded data
            with state._lock:
                state._confidence_history = ConfidenceHistory(state._config.confidence_history_maxlen)
                state._confidence_history.add_many(data.get("confidence_history", []))
                
                state._dream_memory = deque(maxlen=state._config.dream_memory_maxlen)
                for compressed in data.get("dream_memory", []):
                    tensor = state._decompress_tensor(compressed)
                    state._dream_memory.append(tensor)
                
                state._temperament_history = deque(maxlen=state._config.temperament_history_maxlen)
                state._temperament_history.extend(data.get("temperament_history", []))
                
                state._seen_prompts = set(data.get("seen_prompts", []))
                state._training_state = TrainingState(**data.get("training_state", {}))
                state._conversation_metadata = data.get("conversation_metadata", {})
            
            # Validate loaded state
            state._validate_state()
            
            return state
        except Exception as e:
            logger.log_error(f"Failed to create state from dict: {str(e)}", "state_deserialization_error")
            raise StateError(f"Failed to create state from dict: {str(e)}")

    def _validate_state(self) -> None:
        """Validate state integrity."""
        try:
            assert isinstance(self._seen_prompts, set), "seen_prompts must be a set"
            assert isinstance(self.temperament_score, float), "temperament_score must be a float"
            assert isinstance(self._confidence_history, deque), "confidence_history must be a deque"
            assert isinstance(self._temperament_history, deque), "temperament_history must be a deque"
            assert isinstance(self._dream_memory, deque), "dream_memory must be a deque"
            assert 0 <= self.temperament_score <= 1, "temperament_score must be in [0, 1]"
            assert all(0 <= score <= 1 for score in self._confidence_history), "confidence scores must be in [0, 1]"
            assert all(0 <= score <= 1 for score in self._temperament_history), "temperament scores must be in [0, 1]"
            for memory in self._dream_memory:
                assert isinstance(memory, torch.Tensor), "dream memory must be a torch.Tensor"
                assert 0 <= memory.item() <= 1, "dream memory must be in [0, 1]"
            assert self.total_dream_memory_mb >= 0, "total_dream_memory_mb must be non-negative"
        except AssertionError as e:
            self.log_error(f"State validation failed: {str(e)}", error_type="state_validation_error")
            raise StateError(f"State validation failed: {str(e)}")

    @synchronized("lock")
    def get_cached(self, key: str, ttl: float = 60.0) -> Any:
        """Get cached value with time-to-live."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp <= ttl:
                return value
        return None

    @synchronized("lock")
    def set_cached(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, time.time())

    @synchronized("lock")
    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def state_hash(self) -> str:
        """Generate a hash of the current state."""
        state_dict = {
            "seen_prompts": tuple(self._seen_prompts),
            "temperament_score": self.temperament_score,
            "last_temperament_score": self.last_temperament_score,
            "confidence_history": tuple(self._confidence_history),
            "temperament_history": tuple(self._temperament_history),
            "dream_memory": tuple(
                (self._compress_tensor(tensor), tensor.item()) for tensor in self._dream_memory
            ),
            "total_dream_memory_mb": self.total_dream_memory_mb,
            "training_state": self._training_state.__dict__,
            "conversation_metadata": self._conversation_metadata
        }
        return hashlib.md5(json.dumps(state_dict, sort_keys=True).encode()).hexdigest()

    @synchronized("lock")
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history as a list of messages.
        
        Returns:
            List[Dict[str, str]]: List of messages, each containing 'role' and 'content'
        """
        return list(self.history.messages)

    @synchronized("lock")
    def get_conversation_metadata(self) -> Dict[str, Any]:
        """Get metadata about the current conversation.
        
        Returns:
            Dict[str, Any]: Dictionary containing conversation_id and message count
        """
        return {
            "conversation_id": self.history.conversation_id,
            "message_count": len(self.history.messages),
            "max_messages": self.history._config.max_messages
        }

    def save_state(self, path_prefix: str) -> None:
        """Save state using memoria manager."""
        self.memoria_manager.save_state(path_prefix)
        
    def load_state(self, path_prefix: str) -> None:
        """Load state using memoria manager."""
        self.memoria_manager.load_state(path_prefix)

class StateManager:
    """Manages system state and memory operations."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        memoria_manager: MemoriaManager,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize state manager.
        
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
        
    def save_state(self, state: Dict[str, Any]) -> None:
        """Save system state with memory awareness."""
        try:
            # Check memory health before saving
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Save state using memoria manager
            self.memoria_manager.save_state("system_state", state)
            
            # Log state save with memory health info
            self._logger.record_event(
                event_type="state_saved",
                message="System state saved",
                level="info",
                additional_info={
                    "state_size": len(str(state)),
                    "ram_health": ram_health,
                    "gpu_health": gpu_health
                }
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to save state: {str(e)}",
                error_type="state_save_error",
                stack_trace=traceback.format_exc()
            )
            
    def load_state(self) -> Dict[str, Any]:
        """Load system state with memory awareness."""
        try:
            # Check memory health before loading
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Load state using memoria manager
            state = self.memoria_manager.load_state("system_state")
            
            # Log state load with memory health info
            self._logger.record_event(
                event_type="state_loaded",
                message="System state loaded",
                level="info",
                additional_info={
                    "state_size": len(str(state)) if state else 0,
                    "ram_health": ram_health,
                    "gpu_health": gpu_health
                }
            )
            
            return state
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to load state: {str(e)}",
                error_type="state_load_error",
                stack_trace=traceback.format_exc()
            )
            return {}

class UserProfileState(StateBase):
    """Manages user profiles for bonding score calculations with simplicity and elegance."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        super().__init__(config_manager, logger)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.max_inputs = self.config_manager.get("bond_config.max_recent_inputs", 10)
        self.max_lexicon = self.config_manager.get("bond_config.max_lexicon_size", 1000)
        self.log_event("user_profile_init", "User profile state initialized", max_inputs=self.max_inputs)

    @synchronized("lock")
    def update(self, conversation_id: str, user_input: str, session_start: float) -> None:
        """Update user profile with new input dynamically."""
        try:
            profile = self.profiles.setdefault(conversation_id, {
                "lexicon": defaultdict(int),
                "interactions": 0,
                "session_time": 0.0,
                "inputs": deque(maxlen=self.max_inputs),
                "last_interaction": time.time()
            })
            for word in re.findall(r'\w+', user_input.lower()):
                profile["lexicon"][word] += 1
            if len(profile["lexicon"]) > self.max_lexicon:
                profile["lexicon"] = dict(sorted(profile["lexicon"].items(), key=lambda x: x[1], reverse=True)[:self.max_lexicon])
            profile["inputs"].append(user_input)
            profile["interactions"] += 1
            profile["session_time"] += time.time() - session_start
            profile["last_interaction"] = time.time()
            self.log_event("profile_updated", "Profile updated", conversation_id=conversation_id)
        except Exception as e:
            self.log_error(error_msg=f"Profile update failed: {str(e)}", conversation_id=conversation_id)
            raise StateError(f"Profile update failed: {str(e)}")

    @synchronized("lock")
    def get(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieve user profile elegantly."""
        profile = self.profiles.get(conversation_id, {
            "lexicon": defaultdict(int),
            "interactions": 0,
            "session_time": 0.0,
            "inputs": deque(maxlen=self.max_inputs),
            "last_interaction": time.time()
        })
        self.log_event("profile_retrieved", "Profile retrieved", conversation_id=conversation_id, level="debug")
        return profile

    @synchronized("lock")
    def reset(self, conversation_id: str) -> None:
        """Reset user profile cleanly."""
        try:
            self.profiles[conversation_id] = {
                "lexicon": defaultdict(int),
                "interactions": 0,
                "session_time": 0.0,
                "inputs": deque(maxlen=self.max_inputs),
                "last_interaction": time.time()
            }
            self.log_event("profile_reset", "Profile reset", conversation_id=conversation_id)
        except Exception as e:
            self.log_error(error_msg=f"Profile reset failed: {str(e)}", conversation_id=conversation_id)
            raise StateError(f"Profile reset failed: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profiles simply for persistence."""
        try:
            with self.lock:
                return {
                    "profiles": {
                        cid: {
                            "lexicon": dict(p["lexicon"]),
                            "interactions": p["interactions"],
                            "session_time": p["session_time"],
                            "inputs": list(p["inputs"]),
                            "last_interaction": p["last_interaction"]
                        } for cid, p in self.profiles.items()
                    },
                    "version": "1.0"
                }
        except Exception as e:
            self.log_error(error_msg=f"Profile serialization failed: {str(e)}")
            raise StateError(f"Profile serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load profiles dynamically from serialized data."""
        try:
            with self.lock:
                self.profiles.clear()
                for cid, p in data.get("profiles", {}).items():
                    self.profiles[cid] = {
                        "lexicon": defaultdict(int, p.get("lexicon", {})),
                        "interactions": int(p.get("interactions", 0)),
                        "session_time": float(p.get("session_time", 0.0)),
                        "inputs": deque(p.get("inputs", []), maxlen=self.max_inputs),
                        "last_interaction": float(p.get("last_interaction", time.time()))
                    }
                self.log_event("profiles_loaded", "Profiles loaded", profile_count=len(self.profiles))
        except Exception as e:
            self.log_error(error_msg=f"Profile loading failed: {str(e)}")
            raise StateError(f"Profile loading failed: {str(e)}")
