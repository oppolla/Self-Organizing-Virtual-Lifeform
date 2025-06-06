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
from sovl_curiosity import CuriosityConfig
from sovl_utils import NumericalGuard, safe_divide, safe_compare, synchronized
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_data import DataStats
from sovl_interfaces import StateAccessor
import sys
from collections import Counter
import numpy as np

class StateError(Exception):
    """Raised for invalid state operations or data."""
    pass

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
        self.identified_users = {}  # key: signature_hash, value: user profile dict
        self.goals = []  # Add goals list for persistence
        self._validate_config()

    def _validate_config(self) -> None:
        """Basic configuration validation with fallbacks."""
        try:
            # Set default values for critical configs
            self._memory_threshold = self.config_manager.get("memory_config.memory_threshold", 0.85)
            self._memory_decay_rate = self.config_manager.get("memory_config.memory_decay_rate", 0.95)
            self._max_history = self.config_manager.get("state.max_history", 100)
            self._state_file = self.config_manager.get("state.state_file", "state.json")
            
            # Validate numeric values
            if not 0 <= self._memory_threshold <= 1:
                self.log_event("config_warning", "Invalid memory threshold, using default", level="warning")
                self._memory_threshold = 0.85
                
            if not 0 <= self._memory_decay_rate <= 1:
                self.log_event("config_warning", "Invalid memory decay rate, using default", level="warning")
                self._memory_decay_rate = 0.95
                
            if self._max_history <= 0:
                self.log_event("config_warning", "Invalid max history, using default", level="warning")
                self._max_history = 100
                
        except Exception as e:
            self.log_error(f"Configuration validation failed: {str(e)}")
            # Use safe defaults
            self._memory_threshold = 0.85
            self._memory_decay_rate = 0.95
            self._max_history = 100
            self._state_file = "state.json"

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

    def add_identified_user(self, signature_hash: str, profile: dict):
        """Add or update an identified user profile in the central registry."""
        with self.lock:
            self.identified_users[signature_hash] = profile

    def get_identified_user(self, signature_hash: str):
        """Retrieve a user profile by signature hash."""
        with self.lock:
            return self.identified_users.get(signature_hash)

    def get_all_identified_users(self):
        """Return all identified user profiles."""
        with self.lock:
            return dict(self.identified_users)

    def get_goals(self):
        with self.lock:
            return [goal.copy() for goal in self.goals]

    def set_goals(self, goals):
        with self.lock:
            self.goals = [goal.copy() for goal in goals]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for persistence."""
        with self.lock:
            try:
                state_data = {
                    "version": "1.0",
                    "identified_users": self.identified_users,
                    "goals": [goal.copy() for goal in self.goals]
                }
                return state_data
            except Exception as e:
                self.log_error(f"Failed to convert state to dict: {str(e)}")
                raise StateError(f"State serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load state from dictionary."""
        try:
            with self.lock:
                version = data.get("version", "1.0")
                if version not in ["1.0"]:
                    self.log_event(
                        "unsupported_version", f"Unsupported state version: {version}",
                        level="warning", version=version
                    )
                self.identified_users = data.get("identified_users", {})
                self.goals = [goal.copy() for goal in data.get("goals", [])]
                self.log_event(
                    "state_loaded", "State loaded from dictionary",
                    version=version
                )
        except Exception as e:
            self.log_error(f"Failed to load state: {str(e)}")
            raise StateError(f"State loading failed: {str(e)}")

class CuriosityState(StateBase, DictSerializable):
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
        """Serialize curiosity state to dictionary using DictSerializable."""
        try:
            with self.lock:
                # Use DictSerializable's logic for the main fields, but handle unanswered_questions specially
                result = super().to_dict()
                result["unanswered_questions"] = [
                    (q, s, v.cpu().numpy().tolist() if v is not None else None)
                    for q, s, v in self.unanswered_questions
                ]
                result["last_question_time"] = self.last_question_time
                result["pressure"] = self.pressure
                result["novelty_scores"] = list(self.novelty_scores)
                result["question_count"] = self.question_count
                result["version"] = "1.1"
                return result
        except Exception as e:
            self.log_error("Curiosity state serialization failed")
            raise StateError(f"Curiosity state serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load curiosity state from dictionary using DictSerializable."""
        try:
            with self.lock:
                # Use DictSerializable's logic for the main fields, but handle unanswered_questions specially
                for k, v in data.items():
                    if k in ["unanswered_questions", "novelty_scores", "last_question_time", "pressure", "question_count", "version"]:
                        continue  # handled below
                    setattr(self, k, v)
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
                # version is informational only
                self.log_event(
                    "curiosity_state_loaded", "Curiosity state loaded from dictionary",
                    question_count=self.question_count, pressure=self.pressure, version=data.get("version", "1.0")
                )
        except Exception as e:
            self.log_error("Failed to load curiosity state", data_keys=list(data.keys()))
            raise StateError(f"Curiosity state loading failed: {str(e)}")

class ConversationHistory:
    """Manages conversation messages with unique ID."""
    def __init__(self, maxlen: int, conversation_id: Optional[str] = None):
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self.max_messages = maxlen
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages: Deque[Dict[str, str]] = deque(maxlen=self.max_messages)

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

class DictSerializable:
    """Mixin for shared to_dict/from_dict logic with hooks for custom field handling."""
    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            result[k] = self._serialize_field(v)
        return result

    @classmethod
    def from_dict(cls, data):
        obj = cls.__new__(cls)
        for k, v in data.items():
            setattr(obj, k, cls._deserialize_field(v))
        return obj

    @staticmethod
    def _serialize_field(value):
        if isinstance(value, deque):
            return list(value)
        if isinstance(value, defaultdict):
            return dict(value)
        # Add more custom type handling as needed
        return value

    @staticmethod
    def _deserialize_field(value):
        # Add custom deserialization logic as needed
        return value

class SOVLState(StateBase, DictSerializable):
    """Manages the state of the SOVL system."""
    STATE_VERSION = "1.0"

    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        """Initialize SOVL state with configuration and dependencies."""
        super().__init__(config_manager, logger)
        self.device = device
        self._initialize_state()
        from collections import deque
        self.confidence_history = deque(maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 100))
        self.data_stats = DataStats()
        self.state_version = 0  # Version for optimistic locking and concurrency control
        # --- Conversation context for per-conversation interaction data ---
        self.conversation_context: Dict[str, Dict[str, Any]] = {}
        self._context_lock = Lock()
        self.active_temporal_prompt: Optional[str] = None

    def _initialize_state(self) -> None:
        """Initialize state components with safe defaults."""
        try:
            # Core state components
            self._training_state = TrainingState()
            self._conversation_metadata = {}
            self._cache = {}
            
            # Memory and tracking
            self.seen_prompts = set()
            self.temperament_score = 0.0
            self.last_temperament_score = 0.0
            self.temperament_history = deque(
                maxlen=self.config_manager.get("controls_config.temperament_history_maxlen", 5)
            )
            
            # Conversation tracking
            self.history = ConversationHistory(
                maxlen=self.config_manager.get("controls_config.max_messages", 100),
                conversation_id=str(uuid.uuid4())
            )
            
            # Curiosity state additions
            from collections import defaultdict, deque
            self.curiosity_metrics = defaultdict(list)
            self.curiosity_exploration_queue = deque(maxlen=self.config_manager.get("curiosity_config.exploration_queue_maxlen", 100))
            
            # short_term_memory: stores recent memory/context items for integration and recall
            self.short_term_memory = deque(
                maxlen=self.config_manager.get("memory_config.short_term_memory_maxlen", 100)
            )
            
            self.version = self.STATE_VERSION
            self.mode: str = "online"  # online, gestating, offline, pending_gestation, dreaming, meditating
            self.gestation_progress: float = 0.0  # 0.0 to 1.0
            self.dreaming_progress: float = 0.0  # 0.0 to 1.0
            self.meditating_progress: float = 0.0  # 0.0 to 1.0
            # --- Initialize conversation_context ---
            self.conversation_context = {}
            self._context_lock = Lock()
            
            self.log_event(
                "state_initialized",
                "SOVL state components initialized",
                conversation_id=self.history.conversation_id
            )
            
            self.state_version = 1  # Initial version after state is initialized
            
        except Exception as e:
            self.log_error(f"Failed to initialize state components: {str(e)}")
            raise StateError(f"State initialization failed: {str(e)}")

    def _validate_state(self) -> None:
        """Validate state integrity."""
        try:
            # Check core components exist and have correct types
            assert hasattr(self, '_confidence_history') and isinstance(self.confidence_history, deque), "Invalid _confidence_history"
            assert hasattr(self, '_training_state') and isinstance(self._training_state, TrainingState), "Invalid _training_state"
            assert hasattr(self, 'history') and isinstance(self.history, ConversationHistory), "Invalid history"
            assert hasattr(self, 'seen_prompts') and isinstance(self.seen_prompts, set), "seen_prompts must be a set"
            assert hasattr(self, 'temperament_score') and isinstance(self.temperament_score, float), "temperament_score must be a float"
            assert hasattr(self, '_cache') and isinstance(self._cache, dict), "Invalid _cache"

            # Check collections
            assert isinstance(self.temperament_history, deque), "temperament_history must be a deque"
            
            # Check value ranges
            assert 0 <= self.temperament_score <= 1, f"temperament_score {self.temperament_score} must be in [0, 1]"
            assert all(0 <= score <= 1 for score in self.temperament_history), "temperament scores must be in [0, 1]"
            
        except AssertionError as e:
            self.log_error(f"State validation failed: {str(e)}")
            raise StateError(f"State validation failed: {str(e)}")

    def _initialize_memory_managers(self) -> None:
        """Initialize and validate memory managers."""
        try:
            self.ram_manager = RAMManager(self.config_manager, self.logger)
            self.gpu_manager = GPUMemoryManager(self.config_manager, self.logger)
            
            # Basic health checks
            if not hasattr(self.ram_manager, 'check_memory_health'):
                self.log_event("ram_manager_warning", "RAMManager missing health check", level="warning")
            if not hasattr(self.gpu_manager, 'check_memory_health'):
                self.log_event("gpu_manager_warning", "GPUManager missing health check", level="warning")
                
        except Exception as e:
            self.log_error(f"Failed to initialize memory managers: {str(e)}")
            raise StateError("Memory manager initialization failed")

    def update_data_stats(self, stats: Dict[str, Any]) -> None:
        """Update data statistics."""
        try:
            self.data_stats.update(
                total_entries=stats.get("total_entries", 0),
                valid_entries=stats.get("valid_entries", 0),
                invalid_entries=stats.get("invalid_entries", 0),
                validation_errors=stats.get("validation_errors", defaultdict(int)),
                average_entry_length=stats.get("avg_entry_length", 0.0)
            )
            self.log_event(
                "data_stats_updated", "Data statistics updated",
                total_entries=self.data_stats.total_entries,
                valid_entries=self.data_stats.valid_entries,
                invalid_entries=self.data_stats.invalid_entries
            )
        except Exception as e:
            self.log_error(f"Failed to update data statistics: {str(e)}", error_type="data_stats_update_error")
            raise

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
                    tensor = tensor.to(self.device)
                return tensor
        except Exception as e:
            self.log_error(f"Failed to decompress tensor: {str(e)}", error_type="tensor_decompression_error")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization using DictSerializable."""
        with self.lock:
            try:
                result = super().to_dict()
                # Custom handling for special fields
                result["version"] = self.STATE_VERSION
                result["confidence_history"] = list(self.confidence_history)
                result["temperament_history"] = list(self.temperament_history)
                result["seen_prompts"] = list(self.seen_prompts)
                result["training_state"] = self._training_state.__dict__
                result["conversation_history"] = self.history.to_dict() if hasattr(self, "history") else None
                result["conversation_metadata"] = self._conversation_metadata
                result["goals"] = [goal.copy() for goal in self.goals]
                result["curiosity_metrics"] = {k: list(v) for k, v in getattr(self, "curiosity_metrics", {}).items()}
                result["curiosity_exploration_queue"] = list(getattr(self, "curiosity_exploration_queue", []))
                result["mode"] = self.mode
                result["gestation_progress"] = self.gestation_progress
                result["dreaming_progress"] = self.dreaming_progress
                result["meditating_progress"] = self.meditating_progress
                if hasattr(self, 'short_term_memory') and hasattr(self.short_term_memory, 'to_dict'):
                    result['short_term_memory'] = self.short_term_memory.to_dict()
                # --- Serialize conversation_context ---
                from collections import deque
                result["conversation_context"] = {}
                for cid, ctx in self.conversation_context.items():
                    result["conversation_context"][cid] = {
                        "last_interaction_time": ctx.get("last_interaction_time", 0.0),
                        "inputs": list(ctx.get("inputs", deque())),
                        "word_set_cache": [list(ws) for ws in ctx.get("word_set_cache", deque())],
                        # Optionally add 'repetition_history' if present
                        **({"repetition_history": ctx["repetition_history"]} if "repetition_history" in ctx else {})
                    }
                result["active_temporal_prompt"] = self.active_temporal_prompt
                return result
            except Exception as e:
                self.log_error(f"Failed to convert state to dict: {str(e)}")
                raise StateError(f"State serialization failed: {str(e)}")

    def _populate_from_dict(self, data: Dict[str, Any]) -> None:
        """Populate state from dictionary data using DictSerializable."""
        try:
            # Use DictSerializable's logic for the main fields, but handle special fields below
            for k, v in data.items():
                if k in [
                    "confidence_history", "temperament_history", "seen_prompts", "training_state",
                    "conversation_history", "conversation_metadata", "goals", "curiosity_metrics",
                    "curiosity_exploration_queue", "short_term_memory", "mode", "gestation_progress",
                    "dreaming_progress", "meditating_progress", "version"
                ]:
                    continue  # handled below
                setattr(self, k, v)
            from collections import deque, defaultdict
            self.confidence_history = deque(data.get("confidence_history", []), maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 100))
            self.temperament_history.clear()
            self.temperament_history.extend(data.get("temperament_history", []))
            self.temperament_score = float(data.get("temperament_score", 0.0))
            self.last_temperament_score = float(data.get("last_temperament_score", 0.0))
            self.seen_prompts = set(data.get("seen_prompts", []))
            training_state_data = data.get("training_state", {})
            if isinstance(training_state_data, dict):
                self._training_state = TrainingState(**training_state_data)
            history_data = data.get("conversation_history", {})
            if isinstance(history_data, dict):
                max_messages = self.config_manager.get("controls_config.max_messages", 100)
                self.history = ConversationHistory.from_dict(history_data, maxlen=max_messages)
            self._conversation_metadata = data.get("conversation_metadata", {})
            self.goals = [goal.copy() for goal in data.get("goals", [])]
            metrics = data.get("curiosity_metrics", {})
            self.curiosity_metrics = defaultdict(list)
            for k, v in metrics.items():
                self.curiosity_metrics[k] = list(v)
            queue = data.get("curiosity_exploration_queue", [])
            self.curiosity_exploration_queue = deque(queue, maxlen=self.config_manager.get("curiosity_config.exploration_queue_maxlen", 100))
            if 'short_term_memory' in data and hasattr(self, 'short_term_memory') and hasattr(self.short_term_memory, 'from_dict'):
                self.short_term_memory.from_dict(data['short_term_memory'])
            self.mode = data.get("mode", "online")
            self.gestation_progress = float(data.get("gestation_progress", 0.0))
            self.dreaming_progress = float(data.get("dreaming_progress", 0.0))
            self.meditating_progress = float(data.get("meditating_progress", 0.0))
            # --- Deserialize conversation_context ---
            self.conversation_context = {}
            cc_data = data.get("conversation_context", {})
            for cid, ctx in cc_data.items():
                self.conversation_context[cid] = {
                    "last_interaction_time": ctx.get("last_interaction_time", 0.0),
                    "inputs": deque(ctx.get("inputs", []), maxlen=10),
                    "word_set_cache": deque([set(ws) for ws in ctx.get("word_set_cache", [])], maxlen=10),
                    # Optionally add 'repetition_history' if present
                    **({"repetition_history": ctx["repetition_history"]} if "repetition_history" in ctx else {})
                }
            self.active_temporal_prompt = data.get("active_temporal_prompt")
            self._validate_state()
        except Exception as e:
            self.log_error(f"Failed to populate state from dict: {str(e)}")
            raise StateError(f"State population failed: {str(e)}")

    def save_state(self, path_prefix: str) -> None:
        """Save state to disk (MemoriaManager deprecated/removed)."""
        try:
            state_dict = self.to_dict()
            # Implement your own save logic here, e.g. using pickle, json, or another persistence method
            with open(f"{path_prefix}_state.json", "w") as f:
                import json
                json.dump(state_dict, f)
            self.log_event(
                "state_saved",
                f"State saved to {path_prefix}_state.json",
                state_size=len(str(state_dict))
            )
        except Exception as e:
            self.log_error(f"Failed to save state: {str(e)}")
            raise StateError(f"State save failed: {str(e)}")

    def load_state(self, path_prefix: str) -> None:
        """Load state from disk (MemoriaManager deprecated/removed)."""
        try:
            import os
            import json
            state_path = f"{path_prefix}_state.json"
            if not os.path.exists(state_path):
                self.log_event(
                    "state_load_empty",
                    f"No state data found at {state_path}",
                    level="warning"
                )
                return
            with open(state_path, "r") as f:
                state_dict = json.load(f)
            if not isinstance(state_dict, dict):
                raise StateError(f"Invalid state data type: {type(state_dict)}")
            self._populate_from_dict(state_dict)
        except Exception as e:
            self.log_error(f"Failed to load state: {str(e)}")
            raise StateError(f"State load failed: {str(e)}")

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
            "seen_prompts": tuple(self.seen_prompts),
            "temperament_score": self.temperament_score,
            "last_temperament_score": self.last_temperament_score,
            "confidence_history": tuple(self.confidence_history),
            "temperament_history": tuple(self.temperament_history),
            "gestation_progress": self.gestation_progress,
            "dreaming_progress": self.dreaming_progress,
            "meditating_progress": self.meditating_progress
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
            "max_messages": self.history.max_messages
        }

    def validate(self):
        """
        Validate state consistency. Raises ValueError if any invariant is violated.
        Checks include: temperament_score and confidence in [0,1], dream memory size, etc.
        """
        errors = []
        if not (0.0 <= getattr(self, 'temperament_score', 0.5) <= 1.0):
            errors.append("temperament_score out of range")
        if hasattr(self, 'confidence') and not (0.0 <= self.confidence <= 1.0):
            errors.append("confidence out of range")
        if errors:
            raise ValueError(f"SOVLState validation failed: {errors}")

    def clone(self):
        """
        Return a deep copy of the state for transactional/atomic updates.
        """
        import copy
        return copy.deepcopy(self)

    # Document the update protocol
    # All updates to SOVLState should go through StateManager, use state_version for concurrency control,
    # and call validate() after every update to ensure consistency.

    @property
    def embeddings(self) -> list:
        """
        Return a list of all memory embeddings from short-term and dream memory.
        Extend this to include other sources as needed.
        """
        result = []
        # Short-term memory (if available and structured as a list of dicts with 'embedding')
        if hasattr(self, "short_term_memory"):
            result.extend([msg["embedding"] for msg in self.short_term_memory if "embedding" in msg])
        return result

    @synchronized("_context_lock")
    def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get or initialize the context for a conversation."""
        from collections import deque
        if conversation_id not in self.conversation_context:
            self.conversation_context[conversation_id] = {
                "last_interaction_time": time.time(),
                "inputs": deque(maxlen=10),
                "word_set_cache": deque(maxlen=10),
                "consecutive_repetitions": 0,
                "extreme_streak": 0
            }
        ctx = self.conversation_context[conversation_id]
        # Ensure all required fields exist (for robustness if context was partially loaded)
        if "inputs" not in ctx or not isinstance(ctx["inputs"], deque):
            ctx["inputs"] = deque(maxlen=10)
        if "word_set_cache" not in ctx or not isinstance(ctx["word_set_cache"], deque):
            ctx["word_set_cache"] = deque(maxlen=10)
        if "last_interaction_time" not in ctx:
            ctx["last_interaction_time"] = time.time()
        if "consecutive_repetitions" not in ctx:
            ctx["consecutive_repetitions"] = 0
        if "extreme_streak" not in ctx:
            ctx["extreme_streak"] = 0
        return ctx

    @synchronized("_context_lock")
    def update_conversation_context(self, conversation_id: str, data: Dict[str, Any]) -> None:
        """Update the context for a conversation with new data (thread-safe)."""
        self.conversation_context[conversation_id] = data

class StateManager(StateAccessor):
    """Manages the global state with thread-safe operations.
    Implements the StateAccessor interface to provide a standard way to access state.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager,
        device: torch.device
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self.device = device
        self._lock = threading.RLock()  # Use recursive lock for nested calls
        self._current_state = None
        self._state_version = 0
        self._initialize_state()
        
    def _initialize_state(self):
        """Initialize the state."""
        try:
            # Only initialize if not already initialized
            if self._current_state is None:
                self._current_state = SOVLState(
                    self.config_manager, 
                    self.logger,
                    self.device
                )
                self._current_state._initialize_memory_managers()
        except Exception as e:
            self.logger.log_error(
                f"Failed to initialize state: {str(e)}",
                error_type="state_initialization_error"
            )
            raise
            
    def set_state(self, state):
        """Set the current state."""
        with self._lock:
            self._current_state = state
            self._state_version += 1
            
    def get_state(self) -> Dict[str, Any]:
        """Get the current state dictionary.
        
        Returns:
            Dict[str, Any]: The current state.
        """
        with self._lock:
            if self._current_state is None:
                return {}
            return self._current_state.to_dict()
    
    def get_state_version(self) -> int:
        """Get the current state version.
        
        Returns:
            int: Current state version number.
        """
        with self._lock:
            return self._state_version
            
    def save_state(self, state: SOVLState, path_prefix: str) -> None:
        """Save state to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            
            # Save state to file
            state_file = f"{path_prefix}_state.json"
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
                
            self.logger.record_event(
                event_type="state_saved",
                message=f"State saved to {state_file}",
                state_hash=state.state_hash()
            )
        except Exception as e:
            self.logger.log_error(
                f"Failed to save state: {str(e)}",
                error_type="state_save_error"
            )
            raise
            
    def load_state(self, path_prefix: str) -> SOVLState:
        """Load state from disk."""
        try:
            state_file = f"{path_prefix}_state.json"
            
            # Check if files exist
            if not os.path.exists(state_file):
                raise FileNotFoundError(f"State file not found: {state_file}")
                
            # Load state from file
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
                
            # Create new state object
            state = SOVLState(
                self.config_manager,
                self.logger,
                self.device
            )
            
            # Populate state from dict
            state._populate_from_dict(state_dict)
            
            # Initialize memory managers
            state._initialize_memory_managers()
            
            self.logger.record_event(
                event_type="state_loaded",
                message=f"State loaded from {state_file}",
                state_hash=state.state_hash()
            )
            
            # Set as current state
            with self._lock:
                self._current_state = state
                self._state_version += 1
                
            return state
        except Exception as e:
            self.logger.log_error(
                f"Failed to load state: {str(e)}",
                error_type="state_load_error"
            )
            raise
            
    def update_state_atomic(self, update_fn, max_retries=3) -> bool:
        """
        Update state atomically using an update function, with optimistic concurrency.
        The update function is called outside the lock. If the state changes during the update,
        the operation is retried up to max_retries times.
        The update function should be pure (no side effects) and fast.
        Returns True if update was successful, False otherwise.
        """
        for attempt in range(max_retries):
            with self._lock:
                if self._current_state is None:
                    self._initialize_state()
                state_clone = self._current_state.clone()
                state_version = self._state_version
            # Call update_fn outside the lock
            updated_state = update_fn(state_clone)
            # Validate and commit under lock, only if state hasn't changed
            with self._lock:
                if self._state_version != state_version:
                    continue  # State changed, retry
                if updated_state is not None and self.validate_state(updated_state.to_dict()):
                    self._current_state = updated_state
                    self._state_version += 1
                    return True
                else:
                    self.logger.log_error(
                        "State update failed: invalid state returned by update function",
                        error_type="state_update_error"
                    )
                    return False
        self.logger.log_error(
            "State update failed: state changed during update_fn, max retries exceeded",
            error_type="state_update_error"
        )
        return False

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate a state dictionary.
        
        Args:
            state: The state dictionary to validate.
            
        Returns:
            bool: True if state is valid, False otherwise.
        """
        try:
            # Basic validation - check for required fields
            required_fields = ["version", "timestamp", "confidence_history"]
            for field in required_fields:
                if field not in state:
                    self.logger.log_error(
                        f"State validation failed: missing required field '{field}'",
                        error_type="state_validation_error"
                    )
                    return False
                    
            # Check version compatibility
            if state["version"] not in ["1.0"]:
                self.logger.log_error(
                    f"State validation failed: unsupported version {state['version']}",
                    error_type="state_validation_error"
                )
                return False
                
            return True
        except Exception as e:
            self.logger.log_error(
                f"State validation failed: {str(e)}",
                error_type="state_validation_error"
            )
            return False

    def read_state_atomic(self, read_fn):
        """Safely read from state under lock using a pure function."""
        with self._lock:
            if self._current_state is None:
                self._initialize_state()
            return read_fn(self._current_state)

    def set_mode(self, new_mode: str):
        with self._lock:
            if self._current_state:
                self._current_state.mode = new_mode
                self._state_version += 1
    def get_mode(self) -> str:
        with self._lock:
            if self._current_state and hasattr(self._current_state, 'mode'):
                return self._current_state.mode
            return "online"

    def set_gestation_progress(self, value: float):
        with self._lock:
            if self._current_state:
                self._current_state.gestation_progress = value
    def get_gestation_progress(self) -> float:
        with self._lock:
            if self._current_state and hasattr(self._current_state, 'gestation_progress'):
                return self._current_state.gestation_progress
            return 0.0

    def set_dreaming_progress(self, value: float):
        with self._lock:
            if self._current_state:
                self._current_state.dreaming_progress = value

    def get_dreaming_progress(self) -> float:
        with self._lock:
            if self._current_state and hasattr(self._current_state, 'dreaming_progress'):
                return self._current_state.dreaming_progress
            return 0.0

    def set_meditating_progress(self, value: float):
        with self._lock:
            if self._current_state:
                self._current_state.meditating_progress = value
    def get_meditating_progress(self) -> float:
        with self._lock:
            if self._current_state and hasattr(self._current_state, 'meditating_progress'):
                return self._current_state.meditating_progress
            return 0.0

    def set_aspiration_doctrine(self, doctrine: str):
        """Set the current aspiration doctrine and persist it."""
        with self._lock:
            if self._current_state is not None:
                self._current_state.aspiration_doctrine = doctrine
                self._state_version += 1

    def get_aspiration_doctrine(self) -> str:
        """Get the current aspiration doctrine, or a default if not set."""
        with self._lock:
            if self._current_state is not None and hasattr(self._current_state, 'aspiration_doctrine'):
                return getattr(self._current_state, 'aspiration_doctrine', "Be open to new experiences.")
            return "Be open to new experiences."

class StateTracker(StateBase):
    """Tracks component states and their changes."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize state tracker with configuration and logger."""
        super().__init__(config_manager, logger)
        self._component_states = {}  # Track component states
        self._state_history = deque(maxlen=100)  # Keep last 100 states
        self._state_changes = deque(maxlen=50)  # Keep last 50 state changes
        self._system_state = None  # Track overall system state
        
    def _validate_component_state(self, component_name: str, state: Dict[str, Any]) -> bool:
        """Basic validation of component state."""
        try:
            if not isinstance(state, dict):
                self.log_error(f"Invalid state type for {component_name}: {type(state)}")
                return False
                
            # Check for required fields if configured
            required_fields = self.config_manager.get(f"state.{component_name}.required_fields", [])
            for field in required_fields:
                if field not in state:
                    self.log_error(f"Missing required field {field} in {component_name} state")
                    return False
                    
            return True
        except Exception as e:
            self.log_error(f"State validation failed: {str(e)}")
            return False
            
    def update_component_state(self, component_name: str, state: Dict[str, Any]) -> None:
        """Update component state and record the change."""
        with self.lock:
            if not self._validate_component_state(component_name, state):
                self.log_error(f"Invalid state for component {component_name}")
                return
                
            old_state = self._component_states.get(component_name, {})
            self._component_states[component_name] = state.copy()
                
            # Record state change
            change = {
                "type": "component_update",
                "component": component_name,
                "old_state": old_state,
                "new_state": self._component_states[component_name].copy(),
                "timestamp": time.time()
            }
            self._state_changes.append(change)
            
            # Add current state to history
            self._state_history.append(self._component_states.copy())
            
            # Log state change
            self.log_event(
                event_type="component_state_change",
                message=f"Component state updated: {component_name}",
                level="debug" if self.logger.is_debug_enabled() else "info",
                additional_info={
                    "component": component_name,
                    "old_state": old_state,
                    "new_state": self._component_states[component_name].copy()
                }
            )
            
    def get_state(self) -> Dict[str, Any]:
        """Get current system state."""
        with self.lock:
            if not self._system_state:
                return {}
            return self._system_state.to_dict() if hasattr(self._system_state, 'to_dict') else self._system_state
            
    def get_component_state(self, component_name: str) -> Dict[str, Any]:
        """Get current state of a specific component."""
        with self.lock:
            return self._component_states.get(component_name, {}).copy()
            
    def get_component_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all components."""
        with self.lock:
            return {k: v.copy() for k, v in self._component_states.items()}
            
    def get_state_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent state history."""
        with self.lock:
            return [state.copy() for state in list(self._state_history)[-limit:]]
            
    def get_state_changes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent state changes."""
        with self.lock:
            return [change.copy() for change in list(self._state_changes)[-limit:]]
            
    def get_state_stats(self) -> Dict[str, Any]:
        """Get state tracking statistics."""
        with self.lock:
            return {
                "total_states": len(self._state_history),
                "total_changes": len(self._state_changes),
                "component_count": len(self._component_states),
                "current_state_age": time.time() - self._system_state.timestamp if self._system_state else None,
                "last_change_time": self._state_changes[-1]["timestamp"] if self._state_changes else None,
                "change_types": {
                    change_type: count
                    for change_type, count in Counter(
                        change["type"] for change in self._state_changes
                    ).items()
                }
            }
            
    def update_state(self, key: str, value: Any) -> None:
        """Update state with new value and record the change."""
        with self.lock:
            if not self._system_state:
                self._system_state = {}
                
            old_value = self._system_state.get(key, None)
            self._system_state[key] = value
                
            # Record state change
            change = {
                "type": "state_update",
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "timestamp": time.time()
            }
            self._state_changes.append(change)
            
            # Add current state to history
            self._state_history.append(self._system_state.copy())
            
            # Log state change
            self.log_event(
                event_type="state_change",
                message=f"State updated: {key}",
                level="debug" if self.logger.is_debug_enabled() else "info",
                additional_info={
                    "key": key,
                    "old_value": old_value,
                    "new_value": value
                }
            )
            
    def clear_history(self) -> None:
        """Clear state history and changes."""
        with self.lock:
            self._state_history.clear()
            self._state_changes.clear()
            
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information about state tracking."""
        with self.lock:
            return {
                "current_state": self.get_state(),
                "component_states": self.get_component_states(),
                "state_stats": self.get_state_stats(),
                "recent_changes": self.get_state_changes(5),
                "recent_history": self.get_state_history(5)
            }
        
class UserProfileState(StateBase, DictSerializable):
    """Manages user profiles for bonding score calculations with simplicity and elegance."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        super().__init__(config_manager, logger)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.max_inputs = self.config_manager.get("bond_config.max_recent_inputs", 10)
        self.max_lexicon = self.config_manager.get("bond_config.max_lexicon_size", 1000)
        self.nickname_buffer_size = self.config_manager.get("bond_config.nickname_buffer_size", 5)
        self.log_event("user_profile_init", "User profile state initialized", max_inputs=self.max_inputs)

    def _default_profile(self) -> Dict[str, Any]:
        return {
            "lexicon": defaultdict(int),
            "interactions": 0,
            "session_time": 0.0,
            "inputs": deque(maxlen=self.max_inputs),
            "last_interaction": time.time(),
            "nickname": "",
            "early_interactions": [],
            "bond_score": 0.5  # Default bond score
        }

    def _get_or_create_profile(self, user_id: str) -> dict:
        profile = self.profiles.get(user_id)
        if not profile:
            profile = self._default_profile()
            self.profiles[user_id] = profile
        return profile

    @synchronized("lock")
    def get_profile_field(self, user_id: str, field: str, default=None):
        profile = self._get_or_create_profile(user_id)
        return profile.get(field, default)

    @synchronized("lock")
    def set_profile_field(self, user_id: str, field: str, value):
        profile = self._get_or_create_profile(user_id)
        profile[field] = value

    @synchronized("lock")
    def update(self, user_id: str, user_input: str, session_start: float) -> None:
        """Update user profile with new input dynamically."""
        try:
            profile = self._get_or_create_profile(user_id)
            for word in re.findall(r'\w+', user_input.lower()):
                profile["lexicon"][word] += 1
            if len(profile["lexicon"]) > self.max_lexicon:
                profile["lexicon"] = dict(sorted(profile["lexicon"].items(), key=lambda x: x[1], reverse=True)[:self.max_lexicon])
            profile["inputs"].append(user_input)
            profile["interactions"] += 1
            profile["session_time"] += time.time() - session_start
            profile["last_interaction"] = time.time()
            if not profile["nickname"]:
                profile["early_interactions"].append(user_input)
            self.log_event("profile_updated", "Profile updated", user_id=user_id)
        except Exception as e:
            self.log_error(error_msg=f"Profile update failed: {str(e)}", user_id=user_id)
            raise StateError(f"Profile update failed: {str(e)}")

    @synchronized("lock")
    def get(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user profile elegantly."""
        profile = self._get_or_create_profile(user_id)
        self.log_event("profile_retrieved", "Profile retrieved", user_id=user_id, level="debug")
        return profile.copy()

    @synchronized("lock")
    def reset(self, user_id: str) -> None:
        """Reset user profile cleanly."""
        try:
            self.profiles[user_id] = self._default_profile()
            self.log_event("profile_reset", "Profile reset", user_id=user_id)
        except Exception as e:
            self.log_error(error_msg=f"Profile reset failed: {str(e)}", user_id=user_id)
            raise StateError(f"Profile reset failed: {str(e)}")

    # Compatibility: keep old methods as wrappers
    @synchronized("lock")
    def get_bond_score(self, user_id: str) -> float:
        return self.get_profile_field(user_id, "bond_score", 0.5)

    @synchronized("lock")
    def set_bond_score(self, user_id: str, value: float) -> None:
        self.set_profile_field(user_id, "bond_score", float(value))

    @synchronized("lock")
    def get_nickname(self, user_id: str) -> str:
        return self.get_profile_field(user_id, "nickname", "")

    @synchronized("lock")
    def set_nickname(self, user_id: str, value: str) -> None:
        self.set_profile_field(user_id, "nickname", value)

    @synchronized("lock")
    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        return {uid: p.copy() for uid, p in self.profiles.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profiles simply for persistence using DictSerializable."""
        try:
            with self.lock:
                # Use DictSerializable's logic for the main fields, but handle profiles specially
                result = super().to_dict()
                # Custom handling for profiles field
                result["profiles"] = {}
                for uid, p in self.profiles.items():
                    try:
                        lexicon = dict(p["lexicon"]) if isinstance(p["lexicon"], (dict, defaultdict)) else {}
                        inputs = list(p["inputs"]) if isinstance(p["inputs"], (deque, list)) else []
                        early_interactions = list(p.get("early_interactions", [])) if isinstance(p.get("early_interactions", []), (list, deque)) else []
                        result["profiles"][uid] = {
                            "lexicon": lexicon,
                            "interactions": int(p.get("interactions", 0)),
                            "session_time": float(p.get("session_time", 0.0)),
                            "inputs": inputs,
                            "last_interaction": float(p.get("last_interaction", time.time())),
                            "nickname": p.get("nickname", ""),
                            "early_interactions": early_interactions,
                            "bond_score": float(p.get("bond_score", 0.5)),
                        }
                    except Exception as profile_exc:
                        self.log_error(error_msg=f"Profile serialization failed for {uid}: {str(profile_exc)}")
                        continue
                result["version"] = "2.0"
                return result
        except Exception as e:
            self.log_error(error_msg=f"Profile serialization failed: {str(e)}")
            raise StateError(f"Profile serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load profiles dynamically from serialized data using DictSerializable."""
        try:
            with self.lock:
                # Use DictSerializable's logic for the main fields, but handle profiles specially
                for k, v in data.items():
                    if k == "profiles":
                        continue  # handled below
                    setattr(self, k, v)
                self.profiles.clear()
                for uid, p in data.get("profiles", {}).items():
                    try:
                        lexicon = defaultdict(int, p.get("lexicon", {})) if isinstance(p.get("lexicon", {}), dict) else defaultdict(int)
                        interactions = int(p.get("interactions", 0))
                        session_time = float(p.get("session_time", 0.0))
                        raw_inputs = p.get("inputs", [])
                        if not isinstance(raw_inputs, list):
                            self.log_error(error_msg=f"Malformed 'inputs' for {uid}")
                            raw_inputs = []
                        inputs = deque(raw_inputs, maxlen=self.max_inputs)
                        last_interaction = float(p.get("last_interaction", time.time()))
                        nickname = p.get("nickname", "")
                        early_interactions = list(p.get("early_interactions", [])) if isinstance(p.get("early_interactions", []), list) else []
                        bond_score = float(p.get("bond_score", 0.5))
                        self.profiles[uid] = {
                            "lexicon": lexicon,
                            "interactions": interactions,
                            "session_time": session_time,
                            "inputs": inputs,
                            "last_interaction": last_interaction,
                            "nickname": nickname,
                            "early_interactions": early_interactions,
                            "bond_score": bond_score,
                        }
                    except Exception as profile_exc:
                        self.log_error(error_msg=f"Profile loading failed for {uid}: {str(profile_exc)}")
                        continue
                self.log_event("profiles_loaded", "Profiles loaded", profile_count=len(self.profiles))
        except Exception as e:
            self.log_error(error_msg=f"Profile serialization failed: {str(e)}")
            raise StateError(f"Profile serialization failed: {str(e)}")
