import time
from typing import Any, Dict, List, Optional, Deque, Tuple
from collections import deque, defaultdict
import traceback
import threading
import math
import torch
from torch import nn
from datetime import datetime
from sovl_error import ErrorManager
from sovl_state import SOVLState
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_queue import capture_scribe_event
from sovl_memory import RAMManager, GPUMemoryManager
import json

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
        """Prune cache if memory usage exceeds threshold, using RAM/GPU managers if available."""
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
            # Fallback: use cache size
            if len(self.embedding_cache) > 10000:
                usage_high = True
        if usage_high:
            sorted_cache = sorted(
                self.embedding_cache.items(),
                key=lambda x: x[1].get('last_access', 0)
            )
            initial_cache_size = len(self.embedding_cache)
            max_prune_fraction = 0.5
            prune_limit = int(initial_cache_size * max_prune_fraction)
            max_iterations = 1000
            pruned_count = 0
            iteration = 0

            # Optional: backup pruned embeddings before deletion
            try:
                backup_path = "embedding_cache_backup.jsonl"
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

            while (
                len(self.embedding_cache) > 0
                and usage_high
                and sorted_cache
                and pruned_count < prune_limit
                and iteration < max_iterations
            ):
                key, _ = sorted_cache.pop(0)
                del self.embedding_cache[key]
                pruned_count += 1
                iteration += 1
                # Optionally re-check memory after each prune
                if self.ram_manager and self.gpu_manager:
                    try:
                        ram_stats = self.ram_manager.check_memory_health()
                        gpu_stats = self.gpu_manager.get_gpu_usage()
                        if ram_stats.get("usage_percentage", 0) < 0.7 and gpu_stats.get("usage_percentage", 0) < 0.7:
                            break
                    except Exception as e:
                        if self.logger:
                            self.logger.log_error(
                                error_msg=f"Memory check failed during pruning: {str(e)}",
                                error_type="curiosity_memory_error",
                                stack_trace=traceback.format_exc()
                            )
                        break
            if iteration >= max_iterations and self.logger:
                self.logger.log_error(
                    error_msg="Pruning stopped due to iteration limit",
                    error_type="curiosity_prune_limit_error"
                )

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
        base_conf: float,
        scaf_conf: float,
        state: SOVLState,
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute curiosity score based on confidence and embeddings with confidence awareness."""
        try:
            # Get memory embeddings with memory limits
            memory_embeddings = self._get_valid_memory_embeddings(state)
            
            # Compute base curiosity score
            ignorance = self._compute_ignorance_score(base_conf, scaf_conf)
            novelty = (
                self._compute_novelty_score(memory_embeddings, query_embedding, device)
                if memory_embeddings and query_embedding is not None
                else 0.0
            )
            
            # Calculate final score
            final_score = self.weight_ignorance * ignorance + self.weight_novelty * novelty
            
            # Log the complete computation
            self._log_event(
                "curiosity_computed",
                message="Curiosity score computed",
                level="info",
                additional_info={
                    "final_score": final_score,
                    "ignorance": ignorance,
                    "novelty": novelty,
                    "memory_embeddings_count": len(memory_embeddings)
                }
            )
            
            return self._clamp_score(final_score)
            
        except Exception as e:
            self._log_error(f"Curiosity computation failed: {str(e)}")
            return 0.5

    def _compute_ignorance_score(self, base_conf: float, scaf_conf: float) -> float:
        """Compute ignorance component of curiosity score."""
        return self._clamp_score(1.0 - (base_conf * 0.5 + scaf_conf * 0.5))

    def _compute_novelty_score(
        self,
        memory_embeddings: List[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute novelty component of curiosity score using batched processing."""
        try:
            query_embedding = query_embedding.to(device)
            
            # Process embeddings in batches
            max_similarity = 0.0
            for i in range(0, len(memory_embeddings), self.batch_size):
                batch = memory_embeddings[i:i + self.batch_size]
                batch_tensors = torch.stack([emb.to(device) for emb in batch])
                
                # Compute similarities in parallel
                similarities = self.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    batch_tensors
                )
                
                max_similarity = max(max_similarity, similarities.max().item())
            
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

    def update(self, confidence: float) -> float:
        """Update pressure based on confidence with time-based decay."""
        try:
            # Validate confidence input
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1.0):
                raise ValueError("Confidence must be a number between 0 and 1")
                
            time_delta = time.time() - self.last_update
            if time_delta < 0:
                raise ValueError("Invalid time delta detected")
                
            self.last_update = time.time()
            
            # Apply time-based decay
            old_pressure = self.current_pressure
            decay = math.exp(-self.decay_rate * time_delta)
            self.current_pressure = (self.current_pressure * decay +
                                   (1 - decay) * (self.base_pressure + (confidence - self.base_pressure) * self.confidence_adjustment))
            
            # Ensure pressure stays within bounds
            self.current_pressure = max(self.min_pressure, min(self.max_pressure, self.current_pressure))
            
            # Log pressure update
            self._log_event(
                "curiosity_pressure_updated",
                "Curiosity pressure updated",
                level="debug",
                additional_info={
                    "old_pressure": old_pressure,
                    "new_pressure": self.current_pressure,
                    "confidence": confidence,
                    "time_delta": time_delta,
                    "decay": decay
                }
            )
            
            return self.current_pressure
            
        except Exception as e:
            self._log_error(
                f"Failed to update pressure: {str(e)}",
                error_type="curiosity_pressure_update_error",
                stack_trace=traceback.format_exc()
            )
            return self.current_pressure

    def should_erupt(self, threshold: float) -> bool:
        """Check if pressure exceeds threshold."""
        try:
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1.0):
                raise ValueError("Threshold must be a number between 0 and 1")
                
            result = self.current_pressure >= threshold
            self._log_event(
                "curiosity_pressure_threshold_check",
                "Checked if pressure exceeds threshold",
                level="debug",
                additional_info={
                    "current_pressure": self.current_pressure,
                    "threshold": threshold,
                    "result": result
                }
            )
            return result
        except Exception as e:
            self._log_error(
                f"Failed to check pressure threshold: {str(e)}",
                error_type="curiosity_pressure_check_error",
                stack_trace=traceback.format_exc()
            )
            return False

    def drop_pressure(self, amount: float) -> None:
        """Reduce pressure by a specified amount."""
        try:
            # Validate amount
            if not isinstance(amount, (int, float)) or amount < 0:
                raise ValueError("Amount must be a non-negative number")
                
            old_pressure = self.current_pressure
            self.current_pressure = max(self.min_pressure, self.current_pressure - amount)
            
            self._log_event(
                "curiosity_pressure_dropped",
                "Reduced curiosity pressure",
                level="debug",
                additional_info={
                    "old_pressure": old_pressure,
                    "new_pressure": self.current_pressure,
                    "amount": amount
                }
            )
        except Exception as e:
            self._log_error(
                f"Failed to drop pressure: {str(e)}",
                error_type="curiosity_pressure_drop_error",
                stack_trace=traceback.format_exc()
            )

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

class CuriosityCallbacks:
    """Handles curiosity-related callbacks."""
    
    def __init__(self, logger: Optional[Any] = None):
        self.callbacks: Dict[str, List[callable]] = {}
        self.logger = logger

    def register_callback(self, event: str, callback: callable) -> None:
        """Register a callback for an event."""
        self.callbacks.setdefault(event, []).append(callback)

    def trigger_callback(self, event: str, **kwargs) -> None:
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                self._log_error(f"Callback {event} failed: {str(e)}")

    def _log_error(self, message: str, **kwargs) -> None:
        """Log error with standardized format."""
        if self.logger:
            self.logger.log_error(
                error_msg=message,
                error_type="curiosity_error",
                stack_trace=traceback.format_exc(),
                **kwargs
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

class CuriosityManager:
    """Manages curiosity-driven exploration and question generation."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorManager,
        device: torch.device,
        state_manager=None,
        lifecycle_manager=None,
        temperament_system=None,
        confidence_calculator=None,
        generation_manager=None
    ):
        """Initialize CuriosityManager with configuration and components."""
        self._config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.device = device
        self.state_manager = state_manager
        self.lifecycle_manager = lifecycle_manager
        self.temperament_system = temperament_system
        self.confidence_calculator = confidence_calculator
        self.generation_manager = generation_manager
        
        # Get global session_id from config
        self.session_id = self._config_manager.get("runtime.session_id")
        if not self.session_id:
            self.logger.log_warning("No global session_id found in config")
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize components
        self.curiosity = Curiosity(
            config_manager=self._config_manager,
            logger=self.logger
        )
        self.pressure = CuriosityPressure(
            config_manager=self._config_manager,
            logger=self.logger
        )
        self.callbacks = CuriosityCallbacks(logger=self.logger)
        self.system = CuriositySystem(
            config_manager=self._config_manager,
            logger=self.logger,
            ram_manager=self.state_manager.ram_manager if self.state_manager else None,
            gpu_manager=self.state_manager.gpu_manager if self.state_manager else None
        )
        
        # Initialize state
        self._state = {
            'metrics': defaultdict(float),
            'exploration_queue': deque(maxlen=self._config_manager.get("curiosity_config.exploration_queue_maxlen", 100)),
            'last_exploration_time': 0.0,
            'exploration_count': 0,
            'curiosity_score': 0.0,
            'pressure': 0.0,
            'session_id': self.session_id
        }
        
        # Log initialization
        self._record_event(
            "curiosity_manager_initialized",
            "CuriosityManager initialized successfully",
            level="info",
            additional_info={
                "config": {
                    "weight_ignorance": self.curiosity.weight_ignorance,
                    "weight_novelty": self.curiosity.weight_novelty,
                    "metrics_maxlen": self.curiosity.metrics_maxlen,
                    "exploration_queue_maxlen": self._config_manager.get("curiosity_config.exploration_queue_maxlen", 100)
                }
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

    def _get_valid_memory_embeddings(self, state: SOVLState) -> List[torch.Tensor]:
        """Get valid memory embeddings with memory constraints."""
        try:
            # Process embeddings in batches to manage memory
            valid_embeddings = []
            batch_size = self.curiosity.batch_size
            
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
                    "session_id": self.session_id
                },
                session_id=self.session_id,
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
                    "session_id": self.session_id
                },
                session_id=self.session_id,
                timestamp=datetime.now()
            )

    def update_metrics(self, metric_name: str, value: float) -> bool:
        """Update curiosity metrics."""
        try:
            maxlen = self.config_manager.get("metrics_maxlen")
            self.metrics[metric_name].append(value)
            if len(self.metrics[metric_name]) > maxlen:
                self.metrics[metric_name].pop(0)
            return True
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "metrics_update",
                "metric_name": metric_name,
                "value": value
            })
            return False
            
    def calculate_curiosity_score(self, prompt: str) -> float:
        """Calculate curiosity score for a prompt."""
        try:
            if not self.state_manager:
                return 0.0
                
            novelty_score = self._calculate_novelty(prompt)
            ignorance_score = self._calculate_ignorance(prompt)
            
            weight_novelty = self.weight_novelty
            weight_ignorance = self.weight_ignorance
            
            return (weight_novelty * novelty_score + 
                   weight_ignorance * ignorance_score)
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "curiosity_score_calculation",
                "prompt": prompt
            })
            return 0.0
            
    def should_explore(self, prompt: str) -> bool:
        """Determine if exploration should be triggered."""
        try:
            curiosity_score = self.calculate_curiosity_score(prompt)
            threshold = self.config_manager.get("novelty_threshold_spontaneous")
            
            return curiosity_score > threshold
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "exploration_check",
                "prompt": prompt
            })
            return False
            
    def queue_exploration(self, prompt: str) -> bool:
        """Queue a prompt for exploration."""
        try:
            self.exploration_queue.append({
                "prompt": prompt,
                "timestamp": time.time(),
                "score": self.calculate_curiosity_score(prompt)
            })
            return True
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "exploration_queue",
                "prompt": prompt
            })
            return False
            
    def get_next_exploration(self) -> Optional[Dict]:
        """Get next prompt for exploration."""
        try:
            if not self.exploration_queue:
                return None
                
            timeout = self.config_manager.get("curiosity_question_timeout")
            current_time = time.time()
            
            while self.exploration_queue:
                item = self.exploration_queue[0]
                if current_time - item["timestamp"] > timeout:
                    self.exploration_queue.popleft()
                else:
                    return item
                    
            return None
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "get_next_exploration"
            })
            return None
            
    def _calculate_novelty(self, prompt: str) -> float:
        """Calculate novelty score for a prompt."""
        try:
            if not self.state_manager:
                return 0.0
                
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
            
            return 1.0 - max(similarities)
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "novelty_calculation",
                "prompt": prompt
            })
            return 0.0
            
    def _calculate_ignorance(self, prompt: str) -> float:
        """Calculate ignorance score for a prompt."""
        try:
            if not self.state_manager:
                return 0.0
                
            confidence = self.state_manager.get_confidence()
            if confidence is None:
                return 1.0
                
            decay_rate = self.config_manager.get("curiosity_decay_rate")
            return math.exp(-decay_rate * confidence)
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "ignorance_calculation",
                "prompt": prompt
            })
            return 0.0

    def generate_curiosity_question(
        self,
        context: str = None,
        spontaneous: bool = False,
        generation_params: Optional[dict] = None
    ) -> Optional[str]:
        """Generate a curiosity-driven question using GenerationManager and capture scribe event."""
        max_retries = 3
        question = None
        last_exception = None

        self._record_event(
            "curiosity_question_generation_started",
            "Starting curiosity question generation",
            level="info",
            additional_info={
                "context": context,
                "spontaneous": spontaneous
            }
        )

        # Use GenerationManager for question generation
        if not hasattr(self, 'generation_manager') or self.generation_manager is None:
            self._record_error("CuriosityManager requires a GenerationManager instance for question generation.")
            return None
        if context is None:
            self._record_error("A context prompt must be provided for curiosity question generation.")
            return None
        if generation_params is None:
            generation_params = {}

        for attempt in range(max_retries):
            try:
                result = self.generation_manager.generate_text(
                    prompt=context,
                    num_return_sequences=1,
                    **generation_params
                )
                question = result[0] if result and isinstance(result, list) else None
                if question:
                    break
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    self.logger.log_warning(
                        f"Curiosity question generation attempt {attempt + 1} failed: {e}"
                    )
                    time.sleep(1)
                    continue
                self.logger.log_error(
                    f"Question generation failed after {max_retries} attempts: {e}"
                )
                question = None

        # Fallback question if all attempts fail
        if not question:
            question = "What is an interesting aspect of this topic?"
            self.logger.log_warning("Using fallback question due to generation failure")

        # Only capture scribe event for successful (non-fallback) generations
        if question and (last_exception is None or question != "What is an interesting aspect of this topic?"):
            from sovl_queue import capture_scribe_event
            capture_scribe_event(
                origin="sovl_curiosity",
                event_type="curiosity_question_generated",
                event_data={
                    "prompt": context,
                    "question": question,
                    "spontaneous": spontaneous,
                    "generation_params": generation_params
                },
                source_metadata={
                    "module": "CuriosityManager",
                    "session_id": getattr(self, 'session_id', None)
                },
                session_id=getattr(self, 'session_id', None)
            )

        if question and (last_exception is None or question != "What is an interesting aspect of this topic?"):
            self._record_event(
                "curiosity_question_generated",
                "Successfully generated curiosity question",
                level="info",
                additional_info={
                    "question": question,
                    "context": context,
                    "spontaneous": spontaneous
                }
            )
        else:
            self._record_event(
                "curiosity_question_generation_failed",
                "Failed to generate curiosity question, using fallback",
                level="warning",
                additional_info={
                    "context": context,
                    "spontaneous": spontaneous
                }
            )
        return question