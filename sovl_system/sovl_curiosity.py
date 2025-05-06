import time
from typing import Any, Dict, List, Optional, Deque, Tuple
from collections import deque, defaultdict, Counter
import traceback
import threading
import torch
from torch import nn
from datetime import datetime
from sovl_error import ErrorManager
from sovl_state import SOVLState, StateManager
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_queue import capture_scribe_event
from sovl_memory import RAMManager, GPUMemoryManager
import json
from dataclasses import dataclass
from sovl_utils import cosine_similarity
from sovl_recaller import DialogueContextManager

# Unified output function for all utterances (user or system)
def output_response(text: str):
    """Outputs a response to the user. Replace with UI logic as needed."""
    print(text)

@dataclass
class CuriosityConfig:
    max_questions: int
    max_novelty_scores: int
    decay_rate: float
    hidden_size: int
    question_timeout: float

    def validate(self):
        # Optional: add validation logic here
        pass

class Curiosity:
    """Computes curiosity scores based on ignorance and novelty."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Optional[Any] = None,
        max_memory_mb: float = 512.0,
        batch_size: int = 32,
        ram_manager: Optional[RAMManager] = None,
        gpu_manager: Optional[GPUMemoryManager] = None
    ):
        # Get configuration values
        self.weight_ignorance = config_manager.get("curiosity_config.weight_ignorance", 0.7)
        self.weight_novelty = config_manager.get("curiosity_config.weight_novelty", 0.3)
        self.metrics_maxlen = config_manager.get("curiosity_config.novelty_history_maxlen", 1000)
        # New: cache and prune batch size config
        self.embedding_cache_maxlen = config_manager.get("curiosity_config.embedding_cache_maxlen", 1000)
        self.embedding_cache_prune_batch = config_manager.get("curiosity_config.embedding_cache_prune_batch", 100)
        self.embedding_cache_backup_enabled = config_manager.get("curiosity_config.embedding_cache_backup_enabled", False)
        self.embedding_cache_backup_path = config_manager.get("curiosity_config.embedding_cache_backup_path", "embedding_cache_backup.jsonl")
        self.background_pruning_enabled = config_manager.get("curiosity_config.background_pruning_enabled", True)
        self.similarity_early_exit_threshold = config_manager.get("curiosity_config.similarity_early_exit_threshold", 0.99)
        self.adaptive_batch_min = config_manager.get("curiosity_config.adaptive_batch_min", 8)
        self.adaptive_batch_max = config_manager.get("curiosity_config.adaptive_batch_max", 128)
        
        self._validate_weights(self.weight_ignorance, self.weight_novelty)
        self.logger = logger
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        
        # Integrate memory managers
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
        # Initialize components
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.metrics = deque(maxlen=self.metrics_maxlen)
        self.embedding_cache = {}
        self.lock = threading.Lock()
        self.curiosity_score = 0.0  # For external nudges
        # For incremental/background pruning
        self._prune_in_progress = False
        self._prune_event = threading.Event()
        self._prune_thread = None
        self._prune_shutdown = False
        if self.background_pruning_enabled:
            self._prune_thread = threading.Thread(target=self._background_prune_loop, daemon=True)
            self._prune_thread.start()
        
    def _validate_weights(self, ignorance: float, novelty: float) -> None:
        """Validate weight parameters."""
        if not (0.0 <= ignorance <= 1.0 and 0.0 <= novelty <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        if abs(ignorance + novelty - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def _update_memory_usage(self) -> None:
        """Update memory usage tracking using RAM and GPU managers if available."""
        if self.ram_manager and self.gpu_manager:
            try:
                with self.lock:
                    ram_stats = self.ram_manager.check_memory_health()
                    gpu_stats = self.gpu_manager.get_gpu_usage()
                    if self.logger:
                        self.logger.record_event(
                            event_type="memory_usage_updated",
                            message="Memory usage updated",
                            level="info",
                            additional_info={
                                "ram_stats": ram_stats,
                                "gpu_stats": gpu_stats
                            }
                        )
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        error_msg=f"Memory usage tracking failed: {str(e)}",
                        error_type="curiosity_memory_error",
                        stack_trace=traceback.format_exc()
                    )
        else:
            # No managers: do nothing or fallback
            pass

    def _prune_cache(self) -> None:
        """Signal background thread to prune cache if needed."""
        usage_high = False
        if self.ram_manager and self.gpu_manager:
            try:
                ram_stats = self.ram_manager.check_memory_health()
                gpu_stats = self.gpu_manager.get_gpu_usage()
                if ram_stats.get("usage_percentage", 0) > 0.8 or gpu_stats.get("usage_percentage", 0) > 0.8:
                    usage_high = True
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        error_msg=f"Cache pruning memory check failed: {str(e)}",
                        error_type="curiosity_memory_error",
                        stack_trace=traceback.format_exc()
                    )
        else:
            if len(self.embedding_cache) > self.embedding_cache_maxlen:
                usage_high = True
        if usage_high and self.background_pruning_enabled:
            self._prune_event.set()
        elif usage_high:
            # Fallback: prune on main thread if background pruning is disabled
            self._prune_cache_main_thread()

    def _prune_cache_main_thread(self):
        """Prune cache on the main thread (used if background pruning is disabled)."""
        with self.lock:
            sorted_cache = sorted(
                self.embedding_cache.items(),
                key=lambda x: x[1].get('last_access', 0)
            )
            initial_cache_size = len(self.embedding_cache)
            prune_batch = self.embedding_cache_prune_batch
            prune_limit = min(prune_batch, initial_cache_size)
            pruned_count = 0
            if self.embedding_cache_backup_enabled and prune_limit > 0:
                try:
                    backup_path = self.embedding_cache_backup_path
                    with open(backup_path, "a", encoding="utf-8") as f:
                        for key, value in sorted_cache[:prune_limit]:
                            json.dump({"key": key, "value": value}, f, default=str)
                            f.write("\n")
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(
                            error_msg=f"Failed to backup pruned embeddings: {str(e)}",
                            error_type="curiosity_prune_backup_error",
                            stack_trace=traceback.format_exc()
                        )
            for i in range(prune_limit):
                key, _ = sorted_cache[i]
                del self.embedding_cache[key]
                pruned_count += 1
            if self.logger:
                self.logger.record_event(
                    event_type="embedding_cache_pruned",
                    message=f"[MainThread] Pruned {pruned_count} embeddings from cache. Remaining: {len(self.embedding_cache)}",
                    level="info",
                    additional_info={
                        "initial_cache_size": initial_cache_size,
                        "pruned_count": pruned_count,
                        "remaining": len(self.embedding_cache)
                    }
                )

    def _background_prune_loop(self):
        """Background thread loop for cache pruning."""
        while not self._prune_shutdown:
            self._prune_event.wait()
            while True:
                with self.lock:
                    if len(self.embedding_cache) <= self.embedding_cache_maxlen or self._prune_shutdown:
                        break
                    sorted_cache = sorted(
                        self.embedding_cache.items(),
                        key=lambda x: x[1].get('last_access', 0)
                    )
                    initial_cache_size = len(self.embedding_cache)
                    prune_batch = self.embedding_cache_prune_batch
                    prune_limit = min(prune_batch, initial_cache_size)
                    pruned_count = 0
                    if self.embedding_cache_backup_enabled and prune_limit > 0:
                        try:
                            backup_path = self.embedding_cache_backup_path
                            with open(backup_path, "a", encoding="utf-8") as f:
                                for key, value in sorted_cache[:prune_limit]:
                                    json.dump({"key": key, "value": value}, f, default=str)
                                    f.write("\n")
                        except Exception as e:
                            if self.logger:
                                self.logger.log_error(
                                    error_msg=f"Failed to backup pruned embeddings: {str(e)}",
                                    error_type="curiosity_prune_backup_error",
                                    stack_trace=traceback.format_exc()
                                )
                    for i in range(prune_limit):
                        key, _ = sorted_cache[i]
                        del self.embedding_cache[key]
                        pruned_count += 1
                    if self.logger:
                        self.logger.record_event(
                            event_type="embedding_cache_pruned",
                            message=f"[Background] Pruned {pruned_count} embeddings from cache. Remaining: {len(self.embedding_cache)}",
                            level="info",
                            additional_info={
                                "initial_cache_size": initial_cache_size,
                                "pruned_count": pruned_count,
                                "remaining": len(self.embedding_cache)
                            }
                        )
                if len(self.embedding_cache) <= self.embedding_cache_maxlen or self._prune_shutdown:
                    break
            self._prune_event.clear()

    def shutdown_prune_thread(self):
        """Cleanly shutdown the background pruning thread."""
        self._prune_shutdown = True
        self._prune_event.set()
        if self._prune_thread is not None:
            self._prune_thread.join()

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress tensor to reduce memory usage, using GPU manager if available."""
        try:
            if self.gpu_manager:
                gpu_stats = self.gpu_manager.get_gpu_usage()
                if gpu_stats.get("usage_percentage", 0) > 0.8:
                    if tensor.dtype == torch.float32:
                        return tensor.half()
            else:
                if tensor.dtype == torch.float32:
                    return tensor.half()
            return tensor
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    error_msg=f"Tensor compression failed: {str(e)}",
                    error_type="curiosity_memory_error",
                    stack_trace=traceback.format_exc()
                )
            return tensor

    def compute_curiosity(
        self,
        state: SOVLState,
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute curiosity score based on novelty only."""
        try:
            memory_embeddings = self._get_valid_memory_embeddings()
            novelty = (
                self._compute_novelty_score(memory_embeddings, query_embedding, device)
                if memory_embeddings and query_embedding is not None
                else 0.0
            )
            final_score = novelty
            self._log_event(
                "curiosity_computed",
                message="Curiosity score computed",
                level="info",
                additional_info={
                    "final_score": final_score,
                    "novelty": novelty,
                    "memory_embeddings_count": len(memory_embeddings)
                }
            )
            return self._clamp_score(final_score)
        except Exception as e:
            self._log_error(f"Curiosity computation failed: {str(e)}")
            return 0.5

    def _get_valid_memory_embeddings(self) -> List[torch.Tensor]:
        """Get valid memory embeddings with memory constraints."""
        try:
            valid_embeddings = []
            batch_size = self.batch_size

            embeddings = state.embeddings

            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                valid_embeddings.extend(batch)

            return valid_embeddings
        except Exception as e:
            self._log_error(f"Failed to get valid memory embeddings: {str(e)}")
            return []

    def _compute_novelty_score(
        self,
        memory_embeddings: List[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute novelty component of curiosity score using batched processing, with early exit and adaptive batch size."""
        try:
            query_embedding = query_embedding.to(device)
            max_similarity = 0.0
            batch_size = self._estimate_adaptive_batch_size()
            for i in range(0, len(memory_embeddings), batch_size):
                batch = memory_embeddings[i:i + batch_size]
                batch_tensors = torch.stack([emb.to(device) for emb in batch])
                similarities = self.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    batch_tensors
                )
                batch_max = similarities.max().item()
                max_similarity = max(max_similarity, batch_max)
                # Early exit if high enough similarity is found
                if max_similarity >= self.similarity_early_exit_threshold:
                    break
            return self._clamp_score(1.0 - max_similarity)
        except Exception as e:
            self._log_error(f"Novelty score computation failed: {str(e)}")
            return 0.0

    def _clamp_score(self, score: float) -> float:
        """Clamp score between 0.0 and 1.0."""
        return max(0.0, min(1.0, score))

    def _log_error(self, message: str, **kwargs) -> None:
        """Log error with standardized format."""
        if self.logger:
            self.logger.log_error(
                error_msg=message,
                error_type="curiosity_error",
                stack_trace=traceback.format_exc(),
                **kwargs
            )

    def nudge_curiosity(self, amount: float):
        """
        Nudge the curiosity score by the given amount (from external modules like SOVLResonator).
        """
        self.curiosity_score = min(max(self.curiosity_score + amount, 0.0), 1.0)
        if self.logger:
            self.logger.record_event(
                event_type="curiosity_nudged",
                message=f"Curiosity nudged by {amount}",
                additional_info={"curiosity_score": self.curiosity_score}
            )

class CuriosityPressure:
    """Manages curiosity pressure accumulation and eruption."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger
    ):
        """
        Initialize curiosity pressure system with configuration-driven parameters.
        
        Args:
            config_manager: Configuration manager instance
            logger: Logger instance for event tracking
        """
        self.config_manager = config_manager
        self.logger = logger
        
        # Fetch and validate configuration parameters
        try:
            config = config_manager.get_section("curiosity_config", {})
            self.base_pressure = self._validate_config_value(
                "base_pressure", config.get("base_pressure", 0.5), (0.0, 1.0)
            )
            self.max_pressure = self._validate_config_value(
                "max_pressure", config.get("max_pressure", 1.0), (0.0, 1.0)
            )
            self.min_pressure = self._validate_config_value(
                "min_pressure", config.get("min_pressure", 0.0), (0.0, 1.0)
            )
            self.decay_rate = self._validate_config_value(
                "decay_rate", config.get("decay_rate", 0.1), (0.0, 1.0)
            )
            self.confidence_adjustment = self._validate_config_value(
                "confidence_adjustment", config.get("confidence_adjustment", 0.5), (0.0, 1.0)
            )
            
            if not (self.min_pressure <= self.base_pressure <= self.max_pressure):
                raise ValueError("Invalid pressure range: min <= base <= max required")
                
        except Exception as e:
            self._log_error(
                f"Failed to initialize curiosity pressure config: {str(e)}",
                error_type="curiosity_pressure_config_error",
                stack_trace=traceback.format_exc()
            )
            raise
            
        self.current_pressure = self.base_pressure
        self.last_update = time.time()
        self._last_eruption_time = 0.0
        self.cooldown = config.get("eruption_cooldown", 30.0)  # seconds, add to config if not present
        
        # Log initialization
        self._log_event(
            "curiosity_pressure_initialized",
            "Curiosity pressure system initialized",
            level="info",
            additional_info={
                "base_pressure": self.base_pressure,
                "max_pressure": self.max_pressure,
                "min_pressure": self.min_pressure,
                "decay_rate": self.decay_rate,
                "confidence_adjustment": self.confidence_adjustment
            }
        )

    def _validate_config_value(self, key: str, value: Any, valid_range: tuple) -> float:
        """
        Validate a configuration value against a range.
        
        Args:
            key: Configuration key
            value: Value to validate
            valid_range: Tuple of (min, max) allowed values
            
        Returns:
            Validated float value
        """
        try:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Config {key} must be a number")
            min_val, max_val = valid_range
            if not (min_val <= value <= max_val):
                raise ValueError(f"Config {key}={value} outside valid range [{min_val}, {max_val}]")
            return float(value)
        except Exception as e:
            self._log_error(
                f"Config validation failed for {key}: {str(e)}",
                error_type="curiosity_config_validation_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def check_eruption(self, threshold: float, drop: float) -> bool:
        """
        Check if pressure exceeds threshold and cooldown has elapsed. If so, drop pressure and return True.
        Returns True if an eruption occurred, else False.
        """
        now = time.time()
        if self.current_pressure >= threshold and (now - self._last_eruption_time > self.cooldown):
            old_pressure = self.current_pressure
            self.current_pressure = max(self.min_pressure, self.current_pressure - drop)
            self._last_eruption_time = now
            self._log_event(
                "curiosity_pressure_erupted",
                "Curiosity pressure eruption occurred",
                level="info",
                additional_info={
                    "old_pressure": old_pressure,
                    "new_pressure": self.current_pressure,
                    "threshold": threshold,
                    "drop": drop
                }
            )
            return True
        return False

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log event with standardized format."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info=kwargs
        )

    def _log_error(self, message: str, error_type: str = "curiosity_pressure_error", **kwargs) -> None:
        """Log error with standardized format."""
        self.logger.log_error(
            error_msg=message,
            error_type=error_type,
            stack_trace=kwargs.get("stack_trace", traceback.format_exc()),
            additional_info=kwargs.get("additional_info", {})
        )

class CuriosityError(Exception):
    """Custom error for curiosity-related failures."""
    pass

class CuriositySystem:
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize curiosity system.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health across all memory managers."""
        try:
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            return {
                "ram_health": ram_health,
                "gpu_health": gpu_health
            }
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to check memory health: {str(e)}",
                error_type="memory_health_error",
                stack_trace=traceback.format_exc()
            )
            return {
                "ram_health": {"status": "error"},
                "gpu_health": {"status": "error"}
            }

class CuriosityManager():
    """
    Handles calculation of curiosity scores, memory embeddings, and exploration decisions.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorManager,
        device: torch.device,
        state_manager=None,
    ):
        """Initialize the curiosity manager with necessary components and configs."""
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.device = device
        self.state_manager = state_manager
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize components
        self._initialize_config()
        
        # Create core components
        self.curiosity = Curiosity(
            config_manager=self.config_manager, 
            logger=self.logger,
            ram_manager=self._ram_manager if hasattr(self, '_ram_manager') else None,
            gpu_manager=self._gpu_manager if hasattr(self, '_gpu_manager') else None
        )
        
        self.curiosity_pressure = CuriosityPressure(
            config_manager=self.config_manager,
            logger=self.logger
        )
        
        # Initialize metrics
        self._curiosity_score = 0.0
        self._last_update = time.time()
        
        # Log initialization
        self._record_event(
            event_type="curiosity_manager_initialized",
            message="CuriosityManager initialized successfully",
            level="info",
            additional_info={
                "device": str(self.device),
                "curiosity_threshold": self.curiosity_threshold
            }
        )

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        try:
            # Get curiosity config section
            curiosity_config = self.config_manager.get_section("curiosity_config", {})
            
            # Initialize enabled state from config
            self.enabled = curiosity_config.get("enable_curiosity", True)
            
            # Validate and set config values
            self.weight_ignorance = self._validate_config_value(
                "weight_ignorance",
                curiosity_config.get("weight_ignorance"),
                (0.0, 1.0)
            )
            
            self.weight_novelty = self._validate_config_value(
                "weight_novelty",
                curiosity_config.get("weight_novelty"),
                (0.0, 1.0)
            )
            
            # Get generation parameters from config
            self.base_temperature = curiosity_config.get("base_temperature")
            self.temperament_influence = curiosity_config.get("temperament_influence")
            self.max_new_tokens = curiosity_config.get("max_new_tokens")
            self.top_k = curiosity_config.get("top_k")
            self.novelty_threshold_response = curiosity_config.get("novelty_threshold_response")
            self.novelty_threshold_spontaneous = curiosity_config.get("novelty_threshold_spontaneous")
            
            # Get temperature bounds
            self.min_temperature = curiosity_config.get("min_temperature", 0.7)
            self.max_temperature = curiosity_config.get("max_temperature", 1.7)
            
            # Add pressure system validation
            self.pressure_threshold = self._validate_config_value(
                "pressure_threshold",
                curiosity_config.get("pressure_threshold"),
                (0.0, 1.0)
            )
            
            self.pressure_drop = self._validate_config_value(
                "pressure_drop",
                curiosity_config.get("pressure_drop"),
                (0.0, 1.0)
            )
            
            self.max_pressure = self._validate_config_value(
                "max_pressure",
                curiosity_config.get("max_pressure"),
                (0.0, 1.0)
            )
            
            self.min_pressure = self._validate_config_value(
                "min_pressure",
                curiosity_config.get("min_pressure"),
                (0.0, 1.0)
            )
            
            self.decay_rate = self._validate_config_value(
                "decay_rate",
                curiosity_config.get("decay_rate"),
                (0.0, 1.0)
            )
            
            self.confidence_adjustment = self._validate_config_value(
                "confidence_adjustment",
                curiosity_config.get("confidence_adjustment", 0.1),
                (0.0, 1.0)
            )
            
            # Initialize pressure system with validated values
            self.pressure = CuriosityPressure(
                config_manager=self.config_manager,
                logger=self.logger
            )
            
            # Log successful initialization
            self._record_event(
                "curiosity_config_initialized",
                "Curiosity configuration initialized successfully",
                level="info",
                additional_info={
                    "enabled": self.enabled,
                    "weight_ignorance": self.weight_ignorance,
                    "weight_novelty": self.weight_novelty,
                    "base_temperature": self.base_temperature,
                    "temperament_influence": self.temperament_influence,
                    "max_new_tokens": self.max_new_tokens,
                    "top_k": self.top_k,
                    "novelty_threshold_response": self.novelty_threshold_response,
                    "novelty_threshold_spontaneous": self.novelty_threshold_spontaneous,
                    "pressure_config": {
                        "base": self.pressure.base_pressure,
                        "max": self.pressure.max_pressure,
                        "min": self.pressure.min_pressure,
                        "decay_rate": self.pressure.decay_rate,
                        "threshold": self.pressure_threshold,
                        "drop": self.pressure_drop,
                        "confidence_adjustment": self.confidence_adjustment
                    }
                }
            )
            
        except Exception as e:
            self._record_error(
                f"Failed to initialize curiosity config: {str(e)}",
                error_type="config_error",
                stack_trace=traceback.format_exc(),
                context="config_initialization"
            )
            raise

    def _validate_config_value(self, key: str, value: Any, valid_range: Tuple[float, float]) -> float:
        """Validate a configuration value against a range."""
        try:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Config {key} must be a number")
            min_val, max_val = valid_range
            if not (min_val <= value <= max_val):
                raise ValueError(f"Config {key}={value} outside valid range [{min_val}, {max_val}]")
            return float(value)
        except Exception as e:
            self._record_error(
                f"Config validation failed for {key}: {str(e)}",
                error_type="config_validation_error",
                context="config_validation"
            )
            raise

    def _get_valid_memory_embeddings(self) -> List[torch.Tensor]:
        """Get valid memory embeddings with memory constraints."""
        try:
            valid_embeddings = []
            batch_size = self.curiosity.batch_size

            embeddings = self.state_manager.get_state().embeddings

            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                valid_embeddings.extend(batch)

            return valid_embeddings
        except Exception as e:
            self._record_error(f"Failed to get valid memory embeddings: {str(e)}")
            return []

    def _record_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Record event with standardized format (logs and sends to scribe)."""
        if self.logger:
            self.logger.log_event(
                event_type=event_type,
                message=message,
                level=level,
                **kwargs
            )
            # Also capture in scribe queue
            capture_scribe_event(
                origin="sovl_curiosity",
                event_type=event_type,
                event_data={
                    "message": message,
                    **kwargs.get("additional_info", {})
                },
                source_metadata={
                    "level": level,
                    "session_id": getattr(self, 'session_id', None)
                },
                session_id=getattr(self, 'session_id', None),
                timestamp=datetime.now()
            )

    def _record_warning(self, event_type: str, message: str, **kwargs) -> None:
        """Log a warning with standardized format."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level="warning",
            additional_info=kwargs
        )

    def _record_error(self, message: str, **kwargs) -> None:
        """Record error with standardized format (logs and sends to scribe)."""
        if self.logger:
            self.logger.log_error(
                error_msg=message,
                error_type="curiosity_error",
                stack_trace=traceback.format_exc(),
                **kwargs
            )
            # Also capture in scribe queue
            capture_scribe_event(
                origin="sovl_curiosity",
                event_type="curiosity_error",
                event_data={
                    "error_message": message,
                    "error_type": "curiosity_error",
                    **kwargs
                },
                source_metadata={
                    "stack_trace": traceback.format_exc(),
                    "session_id": getattr(self, 'session_id', None)
                },
                session_id=getattr(self, 'session_id', None),
                timestamp=datetime.now()
            )

    def update_metrics(self, metric_name: str, value: float) -> bool:
        """Update curiosity metrics atomically in SOVLState."""
        try:
            def update_fn(state):
                if not hasattr(state, "curiosity_metrics"):
                    from collections import defaultdict
                    state.curiosity_metrics = defaultdict(list)
                state.curiosity_metrics[metric_name].append(value)
                maxlen = self.config_manager.get("metrics_maxlen")
                if len(state.curiosity_metrics[metric_name]) > maxlen:
                    state.curiosity_metrics[metric_name] = state.curiosity_metrics[metric_name][-maxlen:]
            self.state_manager.update_state_atomic(update_fn)
            return True
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "metrics_update",
                "metric_name": metric_name,
                "value": value
            })
            return False
            
    def _calculate_novelty(self, prompt: str) -> float:
        """Calculate novelty score for a prompt (1.0 = most novel, 0.0 = not novel)."""
        seen_prompts = self.state_manager.get_seen_prompts()
        if not seen_prompts:
            return 1.0
        similarities = [
            cosine_similarity(
                self.state_manager.get_prompt_embedding(prompt),
                self.state_manager.get_prompt_embedding(seen)
            )
            for seen in seen_prompts
        ]
        return 1.0 - max(similarities) if similarities else 1.0

    def _calculate_ignorance(self, prompt: str) -> float:
        """Calculate ignorance as 1.0 - similarity to best long-term memory match."""
        try:
            # Ensure recaller is available
            if not hasattr(self, 'recaller') or self.recaller is None:
                if self.logger:
                    self.logger.log_error("No recaller (DialogueContextManager) available for ignorance calculation.")
                return 1.0
            # Get embedding for the prompt
            query_embedding = self.recaller.embedding_fn(prompt)
            # Query long-term memory for top match
            results = self.recaller.get_long_term_context(query_embedding=query_embedding, top_k=1)
            if not results or 'embedding' not in results[0]:
                ignorance = 1.0
            else:
                best_embedding = results[0]['embedding']
                # Compute cosine similarity
                similarity = cosine_similarity(query_embedding, best_embedding)
                ignorance = 1.0 - similarity
                ignorance = max(0.0, min(1.0, ignorance))
            self.logger.log_event(
                event_type="ignorance_calculated",
                message="Ignorance calculated for prompt (retrieval-based)",
                additional_info={
                    "prompt": prompt,
                    "ignorance": ignorance,
                    "method": "retrieval_confidence"
                }
            )
            return ignorance
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Ignorance calculation (retrieval) failed: {e}")
            return 1.0

    def calculate_curiosity_score(self, prompt: str) -> float:
        novelty_score = self._calculate_novelty(prompt)
        ignorance_score = self._calculate_ignorance(prompt)
        curiosity_score = 0.5 * novelty_score + 0.5 * ignorance_score
        self.logger.log_event(
            event_type="curiosity_computed",
            message="Curiosity score computed",
            additional_info={
                "prompt": prompt,
                "curiosity_score": curiosity_score,
                "novelty": novelty_score,
                "ignorance": ignorance_score
            }
        )
        return curiosity_score

    def _summarize_knowns(self, prompt: str) -> str:
        """Summarize what the system knows about the prompt."""
        if not isinstance(prompt, str) or not prompt.strip():
            return "Prompt is invalid or empty."
        seen_prompts = self.state_manager.get_seen_prompts()
        if not seen_prompts:
            return "Prompt is new to the system."
        # Use Counter for large lists
        if len(seen_prompts) > 1000:
            prompt_counts = Counter(seen_prompts)
            count = prompt_counts.get(prompt, 0)
        else:
            count = seen_prompts.count(prompt)
        if count > 0:
            return f"Prompt has been seen {count} times."
        return "Prompt is new to the system."

    def _summarize_unknowns(self, prompt: str) -> str:
        """Summarize what the system is ignorant or uncertain about."""
        if not isinstance(prompt, str) or not prompt.strip():
            return "Prompt is invalid or empty."
        try:
            ignorance = self._calculate_ignorance(prompt)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Ignorance calculation failed: {e}")
            return "System could not assess ignorance for this prompt."
        if ignorance > 0.8:
            return "System is highly ignorant about this prompt."
        elif ignorance > 0.5:
            return "System is somewhat ignorant about this prompt."
        else:
            return "System has some knowledge about this prompt."

    def _is_good_question(self, question: str, prompt: str) -> bool:
        """Heuristic to check if a generated question is specific, relevant, and not generic."""
        if not question or len(question) < 5:
            return False
        if question.lower() in ["what is this?", "i don't know.", "unsure"]:
            return False
        if prompt.lower() in question.lower():
            return True
        return True

    # Curiosity question system prompt template (for future development)
    curiosity_prompt_template = (
        "You are an inquisitive digital mind, always seeking to learn and understand more. "
        "Given the following context and what is known and unknown, ask a single, specific question that would help you or others learn something new.\n"
        "Context:\n"
        "{context}\n"
        "Knowns:\n"
        "{knowns}\n"
        "Unknowns:\n"
        "{unknowns}\n"
        "Essential qualities:\n"
        "   - The question must be specific, relevant, and not generic.\n"
        "   - It should be naturally curious, as if you genuinely want to know the answer.\n"
        "   - The question should be open-ended or thought-provoking, not answerable by yes/no.\n"
        "   - Avoid repeating the context verbatim; synthesize and focus on what is truly unknown.\n"
        "Key constraints:\n"
        "   - Do not mention being an AI, computer, or digital entity.\n"
        "   - Do not ask about yourself or your own capabilities.\n"
        "   - Output only the question, with no preamble or explanation.\n"
        "   - If you understand, reply with only the curiosity question."
    )

    def summarize_context(self, context, max_sentences=2):
        """Return the last N sentences from the context string."""
        if not isinstance(context, str) or not context.strip():
            return "No context provided."
        sentences = [s.strip() for s in context.split('.') if s.strip()]
        summary = '. '.join(sentences[-max_sentences:]) + ('.' if sentences else '')
        return summary

    def build_curiosity_prompt(self, context, knowns, unknowns):
        context_summary = self.summarize_context(context)
        knowns_summary = '; '.join(knowns[:3])
        unknowns_summary = '; '.join(unknowns[:3])
        prompt = self.curiosity_prompt_template.format(
            context=context_summary,
            knowns=knowns_summary,
            unknowns=unknowns_summary
        )
        if len(prompt) > 2000:
            context_summary = self.summarize_context(context, max_sentences=1)
            prompt = self.curiosity_prompt_template.format(
                context=context_summary,
                knowns=knowns_summary,
                unknowns=unknowns_summary
            )
        return prompt

    def generate_curiosity_question(self, prompt: str, context: str = None) -> str:
        """Generate a high-quality curiosity question based on the prompt/context."""
        if not isinstance(prompt, str) or not prompt.strip():
            if self.logger:
                self.logger.log_error("Prompt is invalid or empty.")
            return None
        curiosity_score = self.calculate_curiosity_score(prompt)
        if curiosity_score < getattr(self, 'curiosity_threshold', 0.5):
            return None  # Not curious enough to ask
        knowns = [self._summarize_knowns(prompt)]
        unknowns = [self._summarize_unknowns(prompt)]
        meta_prompt = self.build_curiosity_prompt(context or prompt, knowns, unknowns)
        question = None
        try:
            if not hasattr(self, 'generation_manager') or self.generation_manager is None:
                if self.logger:
                    self.logger.log_error("generation_manager is not set.")
                return None
            if not hasattr(self.generation_manager, 'generate_text'):
                if self.logger:
                    self.logger.log_error("generation_manager has no generate_text method.")
                return None
            output = self.generation_manager.generate_text(meta_prompt, num_return_sequences=1)
            if not output or not isinstance(output, list) or not output[0]:
                if self.logger:
                    self.logger.log_error("generate_text returned no output.")
                return None
            question = output[0]
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"generate_text failed: {e}")
            return None
        if self._is_good_question(question, prompt):
            try:
                if self.logger:
                    self.logger.log_event(
                        event_type="curiosity_question_generated",
                        message="Curiosity question generated",
                        additional_info={
                            "prompt": prompt,
                            "context": context,
                            "question": question,
                            "curiosity_score": curiosity_score
                        }
                    )
                try:
                    from sovl_queue import capture_scribe_event
                    capture_scribe_event(
                        origin="sovl_curiosity",
                        event_type="curiosity_question_generated",
                        event_data={
                            "prompt": prompt,
                            "context": context,
                            "question": question,
                            "curiosity_score": curiosity_score
                        },
                        source_metadata={
                            "module": "CuriosityManager",
                            "session_id": getattr(self, 'session_id', None)
                        },
                        session_id=getattr(self, 'session_id', None)
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"capture_scribe_event failed: {e}")
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Logging failed: {e}")
            return question
        else:
            if self.logger:
                self.logger.log_event(
                    event_type="curiosity_question_rejected",
                    message="Generated question did not meet quality standards",
                    additional_info={
                        "prompt": prompt,
                        "context": context,
                        "question": question,
                        "curiosity_score": curiosity_score
                    }
                )
            return None

    def get_curiosity_score(self, prompt: str = None) -> float:
        """
        Get the curiosity score for the given prompt.
        If no prompt is provided, returns the current global curiosity score.
        
        Args:
            prompt: Optional prompt to calculate curiosity for
            
        Returns:
            float: Curiosity score between 0 and 1
        """
        with self._lock:
            if prompt:
                return self.calculate_curiosity_score(prompt)
            return self._curiosity_score
    
    def update_curiosity_score(self, score: float) -> bool:
        """
        Update the global curiosity score.
        
        Args:
            score: New curiosity score between 0 and 1
            
        Returns:
            bool: True if update was successful
        """
        with self._lock:
            try:
                clamped_score = max(0.0, min(1.0, float(score)))
                self._curiosity_score = clamped_score
                self._last_update = time.time()
                
                self._record_event(
                    event_type="curiosity_score_updated",
                    message=f"Global curiosity score updated to {clamped_score:.4f}",
                    level="info"
                )
                
                return True
            except Exception as e:
                self._record_error(
                    message=f"Failed to update curiosity score: {str(e)}",
                    error_type="curiosity_update_error",
                    stack_trace=traceback.format_exc()
                )
                return False
    
    def nudge_curiosity(self, amount: float) -> float:
        """
        Nudge the curiosity score by the given amount.
        
        Args:
            amount: Amount to nudge curiosity (-1.0 to 1.0)
            
        Returns:
            float: Updated curiosity score
        """
        with self._lock:
            try:
                amount = max(-1.0, min(1.0, float(amount)))
                current = self._curiosity_score
                new_score = max(0.0, min(1.0, current + amount))
                self._curiosity_score = new_score
                
                self._record_event(
                    event_type="curiosity_nudged",
                    message=f"Curiosity nudged by {amount:.4f}, from {current:.4f} to {new_score:.4f}",
                    level="info"
                )
                
                return new_score
            except Exception as e:
                self._record_error(
                    message=f"Failed to nudge curiosity: {str(e)}",
                    error_type="curiosity_nudge_error",
                    stack_trace=traceback.format_exc()
                )
                return self._curiosity_score

    def ask_user_curiosity_question(
        self,
        context: str = None,
        spontaneous: bool = False,
        generation_params: Optional[dict] = None,
        num_candidates: int = 3
    ) -> Optional[str]:
        """
        On curiosity eruption, generate a batch of candidate curiosity questions, select the best to ask the user, and log all.
        Output is handled internally and is indistinguishable from a regular response.
        """
        # Check for curiosity eruption
        if not self.curiosity_pressure.check_eruption(self.pressure_threshold, self.pressure_drop):
            return None  # No eruption, do not ask

        # Generate candidate questions
        candidates = self.generate_candidate_curiosity_questions(context, num_candidates)
        if not candidates:
            return None

        # Score and select the best question
        best_question = self.select_best_curiosity_question(candidates, context)
        if not best_question:
            return None

        # Output the selected question
        self.output_curiosity_utterance(best_question)
        # Log the asked question as 'curiosity_question_user'
        capture_scribe_event(
            origin="sovl_curiosity",
            event_type="curiosity_question_user",
            event_data={
                "question": best_question,
                "context": context,
                "spontaneous": spontaneous,
                "timestamp_unix": time.time(),
                "session_id": getattr(self, 'session_id', None)
            },
            source_metadata={
                "module": "CuriosityManager",
                "session_id": getattr(self, 'session_id', None)
            },
            session_id=getattr(self, 'session_id', None)
        )
        # Log each unasked question as 'curiosity_question'
        for q in candidates:
            if q != best_question:
                capture_scribe_event(
                    origin="sovl_curiosity",
                    event_type="curiosity_question",
                    event_data={
                        "question": q,
                        "context": context,
                        "spontaneous": spontaneous,
                        "timestamp_unix": time.time(),
                        "session_id": getattr(self, 'session_id', None)
                    },
                    source_metadata={
                        "module": "CuriosityManager",
                        "session_id": getattr(self, 'session_id', None)
                    },
                    session_id=getattr(self, 'session_id', None)
                )
        return best_question

    def generate_candidate_curiosity_questions(self, context: str, num_candidates: int = 5) -> list:
        """Generate a batch of candidate curiosity questions using the LLM/generation_manager."""
        prompt = context or "What should I be curious about?"
        knowns = [self._summarize_knowns(prompt)]
        unknowns = [self._summarize_unknowns(prompt)]
        meta_prompt = self.build_curiosity_prompt(context or prompt, knowns, unknowns)
        try:
            if not hasattr(self, 'generation_manager') or self.generation_manager is None:
                if self.logger:
                    self.logger.log_error("generation_manager is not set.")
                return []
            if not hasattr(self.generation_manager, 'generate_text'):
                if self.logger:
                    self.logger.log_error("generation_manager has no generate_text method.")
                return []
            outputs = self.generation_manager.generate_text(meta_prompt, num_return_sequences=num_candidates)
            # Filter and clean
            candidates = [q.strip() for q in outputs if q and self._is_good_question(q, prompt)]
            return candidates
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"generate_text failed: {e}")
            return []

    def select_best_curiosity_question(self, candidates: list, context: str = None) -> Optional[str]:
        """Select the best curiosity question from candidates (by curiosity score, novelty, etc.)."""
        if not candidates:
            return None
        # Score by curiosity score (novelty + ignorance)
        scored = [(q, self.calculate_curiosity_score(q)) for q in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def output_curiosity_utterance(self, text: str) -> None:
        """Outputs the curiosity utterance using the unified output function (no labels)."""
        output_response(text)