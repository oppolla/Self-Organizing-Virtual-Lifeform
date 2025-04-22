import time
from typing import Any, Dict, List, Optional, Deque, Tuple
from collections import deque, defaultdict
import traceback
import threading
import math
import torch
from torch import nn
from sovl_error import ErrorHandler
from sovl_state import SOVLState
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_trainer import TrainingCycleManager
from sovl_temperament import TemperamentSystem
from sovl_confidence import ConfidenceCalculator
from sovl_manager import ModelManager
from sovl_schema import ConfigSchema
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager

class Curiosity:
    """Computes curiosity scores based on ignorance and novelty."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Optional[Any] = None,
        max_memory_mb: float = 512.0,
        batch_size: int = 32
    ):
        # Get configuration values
        self.weight_ignorance = config_manager.get("curiosity_config.weight_ignorance", 0.7)
        self.weight_novelty = config_manager.get("curiosity_config.weight_novelty", 0.3)
        self.metrics_maxlen = config_manager.get("curiosity_config.novelty_history_maxlen", 1000)
        
        self._validate_weights(self.weight_ignorance, self.weight_novelty)
        self.logger = logger
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        
        # Initialize memory managers
        self.memoria_manager = MemoriaManager(config_manager, logger)
        self.ram_manager = RAMManager(config_manager, logger)
        self.gpu_manager = GPUMemoryManager(config_manager, logger)
        
        # Initialize components
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.metrics = deque(maxlen=self.metrics_maxlen)
        self.embedding_cache = {}
        self.lock = threading.Lock()
        
        # Initialize memory tracking
        self._update_memory_usage()

    def _validate_weights(self, ignorance: float, novelty: float) -> None:
        """Validate weight parameters."""
        if not (0.0 <= ignorance <= 1.0 and 0.0 <= novelty <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        if abs(ignorance + novelty - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def _update_memory_usage(self) -> None:
        """Update memory usage tracking using the new memory managers."""
        try:
            with self.lock:
                # Get memory usage from RAM manager
                ram_stats = self.ram_manager.check_memory_health()
                gpu_stats = self.gpu_manager.get_gpu_usage()
                
                # Log memory usage
                self._log_event(
                    "memory_usage_updated",
                    "Memory usage updated",
                    level="info",
                    ram_stats=ram_stats,
                    gpu_stats=gpu_stats
                )
        except Exception as e:
            self._log_error(f"Memory usage tracking failed: {str(e)}")

    def _prune_cache(self) -> None:
        """Prune cache if memory usage exceeds threshold using the new memory managers."""
        try:
            with self.lock:
                # Check memory health
                ram_stats = self.ram_manager.check_memory_health()
                gpu_stats = self.gpu_manager.get_gpu_usage()
                
                # If memory usage is high, prune the cache
                if ram_stats.get("usage_percent", 0) > 0.8 or gpu_stats.get("usage_percent", 0) > 0.8:
                    # Sort by last access time and remove oldest entries
                    sorted_cache = sorted(
                        self.embedding_cache.items(),
                        key=lambda x: x[1].get('last_access', 0)
                    )
                    while sorted_cache:
                        key, _ = sorted_cache.pop(0)
                        del self.embedding_cache[key]
                        
                        # Check if memory usage has improved
                        ram_stats = self.ram_manager.check_memory_health()
                        gpu_stats = self.gpu_manager.get_gpu_usage()
                        if ram_stats.get("usage_percent", 0) < 0.7 and gpu_stats.get("usage_percent", 0) < 0.7:
                            break
                            
                    self._log_event(
                        "cache_pruned",
                        "Cache pruned due to high memory usage",
                        level="info",
                        ram_stats=ram_stats,
                        gpu_stats=gpu_stats
                    )
        except Exception as e:
            self._log_error(f"Cache pruning failed: {str(e)}")

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress tensor to reduce memory usage."""
        try:
            # Check GPU memory before compression
            gpu_stats = self.gpu_manager.get_gpu_usage()
            if gpu_stats.get("usage_percent", 0) > 0.8:
                if tensor.dtype == torch.float32:
                    return tensor.half()  # Convert to float16
            return tensor
        except Exception as e:
            self._log_error(f"Tensor compression failed: {str(e)}")
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
            
            # Apply lifecycle stage adjustments if available
            lifecycle_stage = "unknown"
            lifecycle_weight = 1.0
            if self.lifecycle_manager:
                lifecycle_stage = self.lifecycle_manager._lifecycle_stage
                lifecycle_weight = self.lifecycle_manager.get_life_curve_weight()
            
            # Apply confidence-based adjustments if available
            confidence_weight = 1.0
            if self.confidence_calculator:
                # Use the average of base and scaffold confidence as the current confidence
                current_confidence = (base_conf + scaf_conf) / 2.0
                
                # Adjust curiosity based on confidence
                # Higher confidence leads to more exploration
                confidence_weight = 1.0 + (current_confidence - 0.5) * 0.5
                
                # Log confidence-based adjustment
                self._log_event(
                    "confidence_adjustment_applied",
                    message="Applied confidence-based curiosity adjustment",
                    level="info",
                    additional_info={
                        "current_confidence": current_confidence,
                        "confidence_weight": confidence_weight,
                        "base_ignorance": ignorance,
                        "base_novelty": novelty
                    }
                )
            
            # Calculate final score with all adjustments
            base_score = self.weight_ignorance * ignorance + self.weight_novelty * novelty
            final_score = base_score * confidence_weight * lifecycle_weight
            
            # Log the complete computation
            self._log_event(
                "curiosity_computed",
                message="Curiosity score computed with confidence integration",
                level="info",
                additional_info={
                    "base_score": base_score,
                    "final_score": final_score,
                    "lifecycle_stage": lifecycle_stage,
                    "lifecycle_weight": lifecycle_weight,
                    "confidence_weight": confidence_weight,
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

class CuriosityPressure:
    """Manages curiosity pressure accumulation and eruption."""
    
    def __init__(self, base_pressure: float, max_pressure: float, min_pressure: float, decay_rate: float):
        self.base_pressure = base_pressure
        self.max_pressure = max_pressure
        self.min_pressure = min_pressure
        self.decay_rate = decay_rate
        self.current_pressure = base_pressure
        self.last_update = time.time()

    def update(self, confidence: float) -> float:
        """Update pressure based on confidence."""
        time_delta = time.time() - self.last_update
        self.last_update = time.time()

        self.current_pressure = self.base_pressure + (confidence - self.base_pressure) * 0.1
        self.current_pressure = max(self.min_pressure, min(self.max_pressure, self.current_pressure))

        return self.current_pressure

    def should_erupt(self, threshold: float) -> bool:
        """Check if pressure exceeds threshold."""
        return self.current_pressure >= threshold

    def drop_pressure(self, amount: float) -> None:
        """Reduce pressure by a specified amount."""
        self.current_pressure = max(self.min_pressure, self.current_pressure - amount)

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
        memoria_manager: MemoriaManager,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize curiosity system.
        
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
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorHandler,
        device: torch.device,
        state_manager=None,
        lifecycle_manager=None,
        temperament_system=None,
        confidence_calculator=None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.device = device
        self.state_manager = state_manager
        self.lifecycle_manager = lifecycle_manager
        self.temperament_system = temperament_system
        self.confidence_calculator = confidence_calculator
        
        # Initialize memory managers
        self.memoria_manager = MemoriaManager(config_manager, logger)
        self.ram_manager = RAMManager(config_manager, logger)
        self.gpu_manager = GPUMemoryManager(config_manager, logger)
        
        # Initialize components
        self.curiosity = Curiosity(config_manager, logger)
        self.pressure = CuriosityPressure(
            base_pressure=0.5,
            max_pressure=1.0,
            min_pressure=0.0,
            decay_rate=0.95
        )
        self.callbacks = CuriosityCallbacks(logger)
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize state
        self.state = None
        self.exploration_queue = deque()
        self.metrics = defaultdict(list)
        
        # Log initialization
        self._log_event(
            "curiosity_manager_initialized",
            "Curiosity manager initialized",
            level="info"
        )

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        try:
            # Get curiosity config section
            curiosity_config = self.config_manager.get_section("curiosity_config", {})
            
            # Validate and set config values
            self.weight_ignorance = self._validate_config_value(
                "weight_ignorance",
                curiosity_config.get("weight_ignorance", 0.7),
                (0.0, 1.0)
            )
            
            self.weight_novelty = self._validate_config_value(
                "weight_novelty",
                curiosity_config.get("weight_novelty", 0.3),
                (0.0, 1.0)
            )
            
            # Update config with validated values
            curiosity_config.update({
                "weight_ignorance": self.weight_ignorance,
                "weight_novelty": self.weight_novelty
            })
            
            self.config_manager.update_section("curiosity_config", curiosity_config)
            
            # Log successful initialization
            self._log_event(
                "curiosity_config_initialized",
                "Curiosity configuration initialized successfully",
                level="info"
            )
            
        except Exception as e:
            self._log_error(
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
            self._log_error(
                f"Config validation failed for {key}: {str(e)}",
                error_type="config_validation_error",
                context="config_validation"
            )
            raise

    def _get_valid_memory_embeddings(self, state: SOVLState) -> List[torch.Tensor]:
        """Get valid memory embeddings with memory constraints."""
        try:
            # Check memory health before processing
            ram_stats = self.ram_manager.check_memory_health()
            gpu_stats = self.gpu_manager.get_gpu_usage()
            
            if ram_stats.get("usage_percent", 0) > 0.9 or gpu_stats.get("usage_percent", 0) > 0.9:
                self._log_warning(
                    "High memory usage detected during embedding retrieval",
                    ram_stats=ram_stats,
                    gpu_stats=gpu_stats
                )
                return []
            
            # Get embeddings from memoria manager
            embeddings = self.memoria_manager.get_embeddings(state)
            
            # Process embeddings in batches to manage memory
            valid_embeddings = []
            batch_size = self.curiosity.batch_size
            
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                # Check memory health for each batch
                ram_stats = self.ram_manager.check_memory_health()
                gpu_stats = self.gpu_manager.get_gpu_usage()
                
                if ram_stats.get("usage_percent", 0) > 0.9 or gpu_stats.get("usage_percent", 0) > 0.9:
                    self._log_warning(
                        "High memory usage detected during batch processing",
                        ram_stats=ram_stats,
                        gpu_stats=gpu_stats
                    )
                    break
                
                valid_embeddings.extend(batch)
            
            return valid_embeddings
            
        except Exception as e:
            self._log_error(f"Failed to get valid memory embeddings: {str(e)}")
            return []

    def _log_event(self, event_type: str, message: str, level: str = "info", **kwargs) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info=kwargs
        )

    def _log_warning(self, event_type: str, message: str, **kwargs) -> None:
        """Log a warning with standardized format."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level="warning",
            additional_info=kwargs
        )

    def _log_error(self, message: str, **kwargs) -> None:
        """Log an error with standardized format."""
        self.logger.log_error(
            error_msg=message,
            error_type="curiosity_error",
            stack_trace=traceback.format_exc(),
            **kwargs
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

    def generate_question(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        max_length: int = 512
    ) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            if not self.state_manager:
                return None
                
            # Get next exploration item
            item = self.get_next_exploration()
            if not item:
                return None
                
            # Process the prompt through the model
            inputs = tokenizer(
                item["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return question
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "question_generation",
                "max_length": max_length
            })
            return None
            
    def set_state(self, state: Any) -> bool:
        """Set the state for the CuriosityManager."""
        try:
            if not state:
                raise ValueError("State cannot be None")
                
            self.state = state
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "state_set",
                "state_hash": getattr(state, "state_hash", None)
            })
            return False
            
    def reset(self) -> bool:
        """Reset the CuriosityManager state."""
        try:
            self.metrics.clear()
            self.exploration_queue.clear()
            self.state = None
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "manager_reset"
            })
            return False

    def tune(self, **params) -> None:
        """Update curiosity parameters with validation and logging.
        
        Args:
            **params: Key-value pairs of parameters to update
        """
        try:
            for key, value in params.items():
                # Validate parameter exists and is valid
                if not hasattr(self, key):
                    self._log_warning(
                        "invalid_parameter",
                        message=f"Invalid curiosity parameter: {key}",
                        parameter=key,
                        value=value
                    )
                    continue
                    
                # Validate value type and range
                if key in ["pressure", "weight_ignorance", "weight_novelty"]:
                    if not isinstance(value, (int, float)):
                        self._log_warning(
                            "invalid_value_type",
                            message=f"Invalid type for {key}: {type(value)}",
                            parameter=key,
                            value=value
                        )
                        continue
                    if not 0.0 <= value <= 1.0:
                        self._log_warning(
                            "invalid_value_range",
                            message=f"Value out of range for {key}: {value}",
                            parameter=key,
                            value=value
                        )
                        continue
                            
                # Update parameter
                setattr(self, key, value)
                self._log_event(
                    "parameter_updated",
                    message=f"Updated {key} to {value}",
                    parameter=key,
                    value=value
                )
                
        except Exception as e:
            self._log_error(
                "tune_failed",
                message=f"Failed to tune parameters: {str(e)}",
                parameters=params,
                error=str(e)
            )
            raise

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of tracked metrics."""
        try:
            if not self.state_manager:
                return {}
                
            summary = {}
            for metric_name, values in self.metrics.items():
                if values:
                    summary[f"{metric_name}_mean"] = sum(values) / len(values)
                    summary[f"{metric_name}_max"] = max(values)
                    summary[f"{metric_name}_min"] = min(values)
                    
            return summary
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "metrics_summary"
            })
            return {}
            
    def get_exploration_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the exploration queue."""
        try:
            if not self.exploration_queue:
                return {}
                
            current_time = time.time()
            stats = {
                "queue_length": len(self.exploration_queue),
                "avg_score": 0.0,
                "oldest_item_age": 0.0,
                "newest_item_age": 0.0
            }
            
            if self.exploration_queue:
                scores = [item["score"] for item in self.exploration_queue]
                stats["avg_score"] = sum(scores) / len(scores)
                stats["oldest_item_age"] = current_time - self.exploration_queue[0]["timestamp"]
                stats["newest_item_age"] = current_time - self.exploration_queue[-1]["timestamp"]
                
            return stats
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "queue_stats"
            })
            return {}