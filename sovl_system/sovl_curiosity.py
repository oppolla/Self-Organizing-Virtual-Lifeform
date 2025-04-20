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
from sovl_trainer import LifecycleManager
from sovl_temperament import TemperamentSystem
from sovl_confidence import ConfidenceCalculator
from sovl_manager import ModelManager
from sovl_schema import ConfigSchema

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
        
        # Initialize components
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.metrics = deque(maxlen=self.metrics_maxlen)
        self.embedding_cache = {}
        self.lock = threading.Lock()
        self.memory_usage = 0.0
        
        # Initialize memory tracking
        self._update_memory_usage()

    def _validate_weights(self, ignorance: float, novelty: float) -> None:
        """Validate weight parameters."""
        if not (0.0 <= ignorance <= 1.0 and 0.0 <= novelty <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        if abs(ignorance + novelty - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def _update_memory_usage(self) -> None:
        """Update memory usage tracking."""
        try:
            with self.lock:
                self.memory_usage = sum(
                    tensor.element_size() * tensor.nelement() / (1024 * 1024)
                    for tensor in self.embedding_cache.values()
                    if isinstance(tensor, torch.Tensor)  # Ensure tensor is valid
                )
        except Exception as e:
            self._log_error(f"Memory usage tracking failed: {str(e)}")

    def _prune_cache(self) -> None:
        """Prune cache if memory usage exceeds threshold."""
        try:
            with self.lock:
                if self.memory_usage > self.max_memory_mb:
                    # Sort by last access time and remove oldest entries
                    sorted_cache = sorted(
                        self.embedding_cache.items(),
                        key=lambda x: x[1].get('last_access', 0)
                    )
                    while self.memory_usage > self.max_memory_mb * 0.8 and sorted_cache:
                        key, _ = sorted_cache.pop(0)
                        del self.embedding_cache[key]
                        self._update_memory_usage()
        except Exception as e:
            self._log_error(f"Cache pruning failed: {str(e)}")

    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compress tensor to reduce memory usage."""
        try:
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

class CuriosityManager:
    """Manages the curiosity system and its components."""
    
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
        
        # Initialize components with validated configuration
        self.curiosity = Curiosity(
            config_manager=self.config_manager,
            logger=self.logger
        )
        
        # Initialize other components
        self.pressure = CuriosityPressure(
            base_pressure=self.config_manager.get("curiosity_config.pressure_threshold", 0.7),
            max_pressure=1.0,
            min_pressure=0.0,
            decay_rate=self.config_manager.get("curiosity_config.decay_rate", 0.9)
        )
        
        self.callbacks = CuriosityCallbacks(logger=self.logger)
        self.exploration_queue = deque(maxlen=self.config_manager.get("curiosity_config.queue_maxlen", 10))
        
        # Subscribe to config changes
        self.config_manager.subscribe(self._on_config_change)
        
        # Initialize other attributes
        self.metrics = defaultdict(list)
        self._initialized = False
        self.state = None
        self.pressure = 0.0
        self.last_update = time.time()
        self.conversation_id = None
        self.state_hash = None
        self.pressure_mgr = CuriosityPressure(
            base_pressure=0.5,
            max_pressure=0.9,
            min_pressure=0.1,
            decay_rate=0.95
        )
        self.lock = threading.Lock()
        self._pressure_history = deque(maxlen=100)
        self._last_pressure_change = 0.0
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize components
        self._initialize_components()
        
        # Log device initialization
        self._log_event(
            "device_initialized",
            message="Curiosity manager device initialized",
            level="info",
            device=str(self.device),
            device_type=self.device.type
        )
        
    def _initialize_config(self) -> None:
        """Initialize and validate configuration from ConfigManager."""
        try:
            # Load curiosity configuration
            curiosity_config = self.config_manager.get_section("curiosity_config")
            
            # Set configuration parameters with validation
            self._pressure_change_cooldown = float(curiosity_config.get("pressure_change_cooldown", 1.0))
            self._min_pressure = float(curiosity_config.get("min_pressure", 0.1))
            self._max_pressure = float(curiosity_config.get("max_pressure", 0.9))
            self._pressure_decay_rate = float(curiosity_config.get("pressure_decay_rate", 0.95))
            self.curiosity_queue = deque(maxlen=int(curiosity_config.get("curiosity_queue_maxlen", 100)))
            self.max_memory_mb = float(curiosity_config.get("max_dream_memory_mb", 512.0))
            
            # Validate configuration values
            self._validate_config_values()
            
            # Subscribe to configuration changes
            self.config_manager.subscribe(self._on_config_change)
            
            self.logger.record_event(
                event_type="curiosity_config_initialized",
                message="Curiosity configuration initialized successfully",
                level="info",
                additional_info={
                    "pressure_change_cooldown": self._pressure_change_cooldown,
                    "min_pressure": self._min_pressure,
                    "max_pressure": self._max_pressure,
                    "pressure_decay_rate": self._pressure_decay_rate,
                    "queue_maxlen": self.curiosity_queue.maxlen,
                    "max_memory_mb": self.max_memory_mb
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="curiosity_config_initialization_failed",
                message=f"Failed to initialize curiosity configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _validate_config_values(self) -> None:
        """Validate configuration values against defined ranges."""
        try:
            # Validate pressure-related parameters
            if not 0.0 <= self._min_pressure <= 1.0:
                raise ValueError(f"Invalid min_pressure: {self._min_pressure}. Must be between 0.0 and 1.0.")
                
            if not 0.0 <= self._max_pressure <= 1.0:
                raise ValueError(f"Invalid max_pressure: {self._max_pressure}. Must be between 0.0 and 1.0.")
                
            if self._min_pressure >= self._max_pressure:
                raise ValueError(f"min_pressure ({self._min_pressure}) must be less than max_pressure ({self._max_pressure})")
                
            if not 0.0 <= self._pressure_decay_rate <= 1.0:
                raise ValueError(f"Invalid pressure_decay_rate: {self._pressure_decay_rate}. Must be between 0.0 and 1.0.")
                
            if not 0.0 <= self._pressure_change_cooldown <= 10.0:
                raise ValueError(f"Invalid pressure_change_cooldown: {self._pressure_change_cooldown}. Must be between 0.0 and 10.0.")
                
            # Validate memory parameters
            if not 64.0 <= self.max_memory_mb <= 4096.0:
                raise ValueError(f"Invalid max_memory_mb: {self.max_memory_mb}. Must be between 64.0 and 4096.0.")
                
            # Validate queue parameters
            if not 10 <= self.curiosity_queue.maxlen <= 1000:
                raise ValueError(f"Invalid queue_maxlen: {self.curiosity_queue.maxlen}. Must be between 10 and 1000.")
                
        except Exception as e:
            self.logger.record_event(
                event_type="curiosity_config_validation_failed",
                message=f"Configuration validation failed: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        try:
            # Update curiosity weights
            self.curiosity.weight_ignorance = self.config_manager.get("curiosity_config.weight_ignorance", 0.7)
            self.curiosity.weight_novelty = self.config_manager.get("curiosity_config.weight_novelty", 0.3)
            self.curiosity.metrics_maxlen = self.config_manager.get("curiosity_config.novelty_history_maxlen", 1000)
            
            # Update pressure parameters
            self.pressure.base_pressure = self.config_manager.get("curiosity_config.pressure_threshold", 0.7)
            self.pressure.decay_rate = self.config_manager.get("curiosity_config.decay_rate", 0.9)
            
            # Update queue size
            new_queue_size = self.config_manager.get("curiosity_config.queue_maxlen", 10)
            if new_queue_size != self.exploration_queue.maxlen:
                old_queue = list(self.exploration_queue)
                self.exploration_queue = deque(maxlen=new_queue_size)
                self.exploration_queue.extend(old_queue[:new_queue_size])
            
            self._log_event(
                "config_updated",
                message="Curiosity configuration updated",
                level="info",
                additional_info={
                    "weight_ignorance": self.curiosity.weight_ignorance,
                    "weight_novelty": self.curiosity.weight_novelty,
                    "metrics_maxlen": self.curiosity.metrics_maxlen,
                    "pressure_threshold": self.pressure.base_pressure,
                    "decay_rate": self.pressure.decay_rate,
                    "queue_maxlen": new_queue_size
                }
            )
            
        except Exception as e:
            self._log_error(f"Configuration update failed: {str(e)}")
            
    def _initialize_components(self) -> None:
        """Initialize curiosity components with configuration."""
        try:
            # Load curiosity weights from configuration
            curiosity_config = self.config_manager.get_section("curiosity_config")
            weight_ignorance = float(curiosity_config.get("weight_ignorance", 0.7))
            weight_novelty = float(curiosity_config.get("weight_novelty", 0.3))
            metrics_maxlen = int(curiosity_config.get("metrics_maxlen", 1000))
            
            # Initialize components with validated configuration
            self.curiosity = Curiosity(
                config_manager=self.config_manager,
                logger=self.logger
            )
            self.callbacks = CuriosityCallbacks(logger=self.logger)
            self.last_question_time = time.time()
            
            self.logger.record_event(
                event_type="curiosity_components_initialized",
                message="Curiosity components initialized successfully",
                level="info",
                additional_info={
                    "weight_ignorance": weight_ignorance,
                    "weight_novelty": weight_novelty,
                    "metrics_maxlen": metrics_maxlen
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="curiosity_components_initialization_failed",
                message=f"Failed to initialize curiosity components: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            raise
            
    def set_state(self, state: SOVLState) -> None:
        """Set the SOVL state and synchronize with CuriosityState."""
        with self.lock:
            try:
                if not isinstance(state, SOVLState):
                    raise ValueError("State must be an instance of SOVLState")
                    
                # Validate device consistency
                if state.device != self.device:
                    self._log_warning(f"State device ({state.device}) differs from manager device ({self.device})")
                    
                self.state = state
                
                # Validate and synchronize curiosity state
                if not hasattr(state, 'curiosity'):
                    self._log_error("Missing curiosity state in SOVLState")
                    state.curiosity = CuriosityState(self.config_manager, self.logger, self.device)
                    
                # Validate required attributes
                required_attrs = [
                    'pressure', 'novelty_threshold_spontaneous',
                    'novelty_threshold_response', 'pressure_threshold',
                    'pressure_drop', 'silence_threshold',
                    'question_cooldown', 'queue_maxlen'
                ]
                
                missing_attrs = [attr for attr in required_attrs 
                               if not hasattr(state.curiosity, attr)]
                
                if missing_attrs:
                    self._log_error(f"Missing required attributes in CuriosityState: {missing_attrs}")
                    return
                
                # Validate and clamp parameter ranges
                if not (0 <= state.curiosity.pressure <= 1):
                    self._log_warning(f"Invalid pressure value: {state.curiosity.pressure}, clamping to [0,1]")
                    state.curiosity.pressure = max(0, min(1, state.curiosity.pressure))
                
                if not (0 <= state.curiosity.novelty_threshold_spontaneous <= 1):
                    self._log_warning(f"Invalid novelty_threshold_spontaneous: {state.curiosity.novelty_threshold_spontaneous}, clamping to [0,1]")
                    state.curiosity.novelty_threshold_spontaneous = max(0, min(1, state.curiosity.novelty_threshold_spontaneous))
                
                if not (0 <= state.curiosity.novelty_threshold_response <= 1):
                    self._log_warning(f"Invalid novelty_threshold_response: {state.curiosity.novelty_threshold_response}, clamping to [0,1]")
                    state.curiosity.novelty_threshold_response = max(0, min(1, state.curiosity.novelty_threshold_response))
                
                # Synchronize pressure
                self.pressure = state.curiosity.pressure
                
                # Log successful synchronization
                self._log_event("state_sync", {
                    "pressure": self.pressure,
                    "novelty_threshold_spontaneous": state.curiosity.novelty_threshold_spontaneous,
                    "novelty_threshold_response": state.curiosity.novelty_threshold_response,
                    "conversation_id": getattr(state, 'history', {}).get('conversation_id', None),
                    "state_hash": getattr(state, 'state_hash', None),
                    "device": str(self.device)
                })
                
            except Exception as e:
                self._log_error(f"Error synchronizing with SOVLState: {str(e)}")
                raise

    def _generate_query_embedding(self, state: Any, tokenizer: Any, model: Any, query: Optional[str]) -> Optional[torch.Tensor]:
        """Generate query embedding with device validation."""
        try:
            if not query:
                return None
                
            # Ensure model is on correct device
            model = model.to(self.device)
            
            # Tokenize and move to device
            inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                
            return embedding
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "query_embedding",
                "query": query
            })
            return None
            
    def _validate_device(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Validate tensor device and move if necessary."""
        try:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
                
            if tensor.device != self.device:
                self._log_warning(f"Moving {name} from {tensor.device} to {self.device}")
                tensor = tensor.to(self.device)
                
            return tensor
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "device_validation",
                "tensor_name": name,
                "tensor_shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else None
            })
            raise

    def update_pressure(self, confidence: float, timestamp: float) -> None:
        """Update pressure based on confidence and timestamp."""
        with self.lock:
            # Calculate new pressure
            new_pressure = self.pressure_mgr.update(confidence)
            
            # Add to queue
            self._pressure_queue.append((new_pressure, timestamp))
            self._last_pressure_update = timestamp
            
            # Log pressure update
            self.logger.log_event(
                "curiosity_pressure_updated",
                {
                    "pressure": new_pressure,
                    "confidence": confidence,
                    "timestamp": timestamp
                }
            )

    def get_current_pressure(self, timestamp: Optional[float] = None) -> float:
        """Get current pressure from queue."""
        with self.lock:
            if not self._pressure_queue:
                return self.pressure_mgr.current_pressure
                
            if timestamp is None or timestamp - self._last_pressure_update >= self._pressure_update_interval:
                # Get most recent pressure
                pressure, _ = self._pressure_queue[-1]
                return pressure
                
            return self.pressure_mgr.current_pressure

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
            if not self._initialized:
                raise RuntimeError("CuriosityManager not initialized")
                
            novelty_score = self._calculate_novelty(prompt)
            ignorance_score = self._calculate_ignorance(prompt)
            
            weight_novelty = self.config_manager.get("weight_novelty")
            weight_ignorance = self.config_manager.get("weight_ignorance")
            
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
            if not self._initialized:
                return False
                
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
            if not self._initialized:
                return False
                
            self.curiosity_queue.append({
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
            if not self.curiosity_queue:
                return None
                
            timeout = self.config_manager.get("curiosity_question_timeout")
            current_time = time.time()
            
            while self.curiosity_queue:
                item = self.curiosity_queue[0]
                if current_time - item["timestamp"] > timeout:
                    self.curiosity_queue.popleft()
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
            if not self._initialized:
                raise RuntimeError("CuriosityManager not initialized")
                
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
            self._initialized = True
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
            self.curiosity_queue.clear()
            self._initialized = False
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
            with self.lock:
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
                    
                    # Special handling for queue_maxlen
                    if key == "queue_maxlen":
                        self.curiosity_queue = deque(maxlen=value)
                        
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
            if not self._initialized:
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
            if not self._initialized:
                return {}
                
            current_time = time.time()
            stats = {
                "queue_length": len(self.curiosity_queue),
                "avg_score": 0.0,
                "oldest_item_age": 0.0,
                "newest_item_age": 0.0
            }
            
            if self.curiosity_queue:
                scores = [item["score"] for item in self.curiosity_queue]
                stats["avg_score"] = sum(scores) / len(scores)
                stats["oldest_item_age"] = current_time - self.curiosity_queue[0]["timestamp"]
                stats["newest_item_age"] = current_time - self.curiosity_queue[-1]["timestamp"]
                
            return stats
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "queue_stats"
            })
            return {}

    def _get_valid_memory_embeddings(self, state: SOVLState) -> List[torch.Tensor]:
        """
        Get valid memory embeddings while respecting memory limits.
        
        Args:
            state: Current SOVLState instance
            
        Returns:
            List of valid memory embeddings within memory limits
        """
        try:
            memory_embeddings = []
            hidden_size = self.config_manager.get("hidden_size")
            total_size = 0.0
            
            if not hasattr(state, 'dream_memory') or not state.dream_memory:
                return memory_embeddings
                
            for entry in state.dream_memory:
                tensor = entry["tensor"]
                # Calculate memory size in MB
                memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                
                # Check memory limit
                if total_size + memory_size > self.max_memory_mb:
                    self._log_event("memory_limit_reached", {
                        "total_size_mb": total_size,
                        "skipped_size_mb": memory_size,
                        "max_memory_mb": self.max_memory_mb
                    })
                    break
                    
                # Validate tensor shape
                if tensor.shape[-1] == hidden_size:
                    try:
                        # Validate and move tensor to correct device
                        tensor = self._validate_device(tensor, f"dream_memory_tensor_{len(memory_embeddings)}")
                        memory_embeddings.append(tensor)
                        total_size += memory_size
                    except Exception as e:
                        self._log_warning("tensor_validation_failed", {
                            "error": str(e),
                            "tensor_shape": list(tensor.shape)
                        })
                        continue
                else:
                    self._log_warning("invalid_tensor_shape", {
                        "expected_size": hidden_size,
                        "actual_size": tensor.shape[-1]
                    })
                    
            # Update total memory tracking
            self.total_memory_mb = total_size
            
            return memory_embeddings
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "get_valid_memory_embeddings",
                "hidden_size": hidden_size if 'hidden_size' in locals() else None,
                "total_size_mb": total_size if 'total_size' in locals() else None
            })
            return []
            
    def get_pressure(self) -> float:
        """Get current pressure with validation, decay, and confidence awareness."""
        try:
            with self.lock:
                # Get lifecycle stage if available
                lifecycle_stage = "unknown"
                if self.lifecycle_manager:
                    lifecycle_stage = self.lifecycle_manager._lifecycle_stage
                
                # Apply natural decay
                time_delta = time.time() - self.last_update
                if time_delta > 0:
                    decay_rate = self._pressure_decay_rate
                    
                    # Adjust decay rate based on confidence if available
                    if self.confidence_calculator:
                        current_confidence = self.confidence_calculator.calculate_confidence_score(
                            logits=torch.tensor([]),  # Placeholder
                            generated_ids=torch.tensor([]),  # Placeholder
                            state=self.state,
                            error_manager=self.error_manager,
                            context=SystemContext(),  # Placeholder
                            curiosity_manager=self
                        )
                        
                        # Higher confidence leads to faster pressure decay
                        decay_rate *= (1.0 + (current_confidence - 0.5) * 0.5)
                    
                    self.pressure *= (decay_rate ** time_delta)
                    self.last_update = time.time()
                
                # Ensure pressure stays within bounds
                self.pressure = max(self._min_pressure, min(self._max_pressure, self.pressure))
                
                # Record pressure in history with confidence context
                self._pressure_history.append((
                    time.time(),
                    self.pressure,
                    lifecycle_stage,
                    getattr(self.confidence_calculator, "current_confidence", None)
                ))
                
                # Log pressure update with confidence context
                self._log_event(
                    "pressure_updated",
                    message="Pressure updated with confidence context",
                    level="info",
                    additional_info={
                        "current_pressure": self.pressure,
                        "lifecycle_stage": lifecycle_stage,
                        "time_delta": time_delta,
                        "confidence": getattr(self.confidence_calculator, "current_confidence", None)
                    }
                )
                
                return self.pressure
                
        except Exception as e:
            self._log_error(f"Failed to get pressure: {str(e)}")
            return self._min_pressure  # Return minimum pressure on error

    def reduce_pressure(self, amount: float) -> None:
        """Reduce pressure with validation, cooldown, and lifecycle awareness."""
        try:
            with self.lock:
                current_time = time.time()
                
                # Get lifecycle stage if available
                lifecycle_stage = "unknown"
                if self.lifecycle_manager:
                    lifecycle_stage = self.lifecycle_manager._lifecycle_stage
                
                # Check cooldown
                if current_time - self._last_pressure_change < self._pressure_change_cooldown:
                    self._log_warning(
                        "pressure_change_cooldown",
                        message="Pressure change too frequent",
                        last_change=self._last_pressure_change,
                        current_time=current_time,
                        lifecycle_stage=lifecycle_stage
                    )
                    return
                
                # Validate amount
                if not isinstance(amount, (int, float)) or amount <= 0:
                    self._log_warning(
                        "invalid_pressure_reduction",
                        message=f"Invalid pressure reduction amount: {amount}",
                        amount=amount,
                        lifecycle_stage=lifecycle_stage
                    )
                    return
                
                # Adjust reduction amount based on lifecycle stage
                if lifecycle_stage in self.config_manager.get("curiosity_config.lifecycle_params", {}):
                    stage_params = self.config_manager.get(f"curiosity_config.lifecycle_params.{lifecycle_stage}")
                    amount *= stage_params.get("reduction_factor", 1.0)
                
                # Calculate new pressure
                new_pressure = self.pressure - amount
                
                # Ensure pressure stays above minimum
                if new_pressure < self._min_pressure:
                    self._log_warning(
                        "pressure_min_reached",
                        message=f"Pressure reduction would go below minimum: {new_pressure}",
                        current_pressure=self.pressure,
                        reduction=amount,
                        lifecycle_stage=lifecycle_stage
                    )
                    new_pressure = self._min_pressure
                
                # Update pressure
                self.pressure = new_pressure
                self._last_pressure_change = current_time
                self.last_update = current_time
                
                # Log pressure change with lifecycle context
                self._log_event(
                    "pressure_reduced",
                    message=f"Pressure reduced by {amount}",
                    old_pressure=self.pressure + amount,
                    new_pressure=self.pressure,
                    reduction=amount,
                    lifecycle_stage=lifecycle_stage
                )
                
        except Exception as e:
            self._log_error(f"Failed to reduce pressure: {str(e)}")

    def get_pressure_stats(self) -> Dict[str, Any]:
        """Get statistics about pressure changes."""
        try:
            with self.lock:
                if not self._pressure_history:
                    return {
                        "current_pressure": self.pressure,
                        "min_pressure": self._min_pressure,
                        "max_pressure": self._max_pressure,
                        "history_size": 0
                    }
                
                pressures = [p for _, p, _, _ in self._pressure_history]
                return {
                    "current_pressure": self.pressure,
                    "min_pressure": self._min_pressure,
                    "max_pressure": self._max_pressure,
                    "avg_pressure": sum(pressures) / len(pressures),
                    "min_observed": min(pressures),
                    "max_observed": max(pressures),
                    "history_size": len(pressures),
                    "last_change": self._last_pressure_change
                }
                
        except Exception as e:
            self._log_error(f"Failed to get pressure stats: {str(e)}")
            return {}

class CuriosityEngine:
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        config_handler: ConfigHandler,
        model_manager: ModelManager,
        state_tracker: StateTracker,
        error_manager: ErrorManager,
        logger: Logger,
        device: str
    ):
        """
        Initialize the curiosity engine with explicit dependencies.
        
        Args:
            config_handler: Configuration handler
            model_manager: Model manager instance
            state_tracker: State tracker instance
            error_manager: Error manager instance
            logger: Logger instance
            device: Device to use for tensor operations
        """
        self.config_handler = config_handler
        self.model_manager = model_manager
        self.state_tracker = state_tracker
        self.error_manager = error_manager
        self.logger = logger
        self.device = device
        
        # Initialize components
        self.curiosity_manager = self._create_curiosity_manager()
        self.cycle_manager = self._create_training_cycle_manager()
        
        # Log initialization
        self.logger.record_event(
            event_type="curiosity_engine_initialized",
            message="Curiosity engine initialized successfully",
            level="info"
        )
        
    def _create_curiosity_manager(self) -> CuriosityManager:
        """Create and initialize the curiosity manager."""
        try:
            return CuriosityManager(
                config_manager=self.config_handler.config_manager,
                logger=self.logger,
                device=self.device
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            raise
            
    def _create_training_cycle_manager(self) -> TrainingCycleManager:
        """Create and initialize the training cycle manager."""
        try:
            return TrainingCycleManager(
                config=self.config_handler.config_manager.get_section("sovl_config"),
                logger=self.logger,
                device=self.device,
                state_manager=self.state_tracker,
                curiosity_manager=self.curiosity_manager
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0)
            raise
            
    def _validate_configuration(self) -> bool:
        """Validate current configuration state."""
        try:
            if not self.config_handler.validate():
                self.logger.record_event(
                    event_type="config_validation_failed",
                    message="Configuration validation failed, attempting recovery",
                    level="error"
                )
                self.config_handler._refresh_configs()
                if not self.config_handler.validate():
                    self.logger.record_event(
                        event_type="config_recovery_failed",
                        message="Configuration recovery failed",
                        level="error"
                    )
                    return False
            return True
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error during configuration validation: {str(e)}",
                error_type="config_validation_error",
                stack_trace=traceback.format_exc(),
                additional_info={"error": str(e)}
            )
            return False