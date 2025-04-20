import torch
import time
from collections import deque
from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Set
import contextlib
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_utils import calculate_confidence, detect_repetitions, adjust_temperature, synchronized, dynamic_batch_size, memory_usage
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_curiosity import CuriosityManager
from sovl_trainer import LifecycleManager, TrainingConfig
from sovl_temperament import  TemperamentSystem 
from sovl_confidence import  ConfidenceCalculator 
import threading

# Add confidence-related constants at the top of the file
DEFAULT_CONFIDENCE = 0.5
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
MIN_HISTORY_LENGTH = 3
CURIOSITY_PRESSURE_FACTOR = 0.1
DEFAULT_TEMPERAMENT_INFLUENCE = 0.3
DEFAULT_LIFECYCLE_INFLUENCE = 0.2

# Temperament-based confidence adjustments
TEMPERAMENT_MOOD_MULTIPLIERS = {
    "Cautious": 0.8,  # Reduce confidence in cautious mood
    "Balanced": 1.0,  # No adjustment in balanced mood
    "Curious": 1.2    # Increase confidence in curious mood
}

# Lifecycle stage adjustments
LIFECYCLE_STAGE_MULTIPLIERS = {
    "initialization": 0.9,    # More conservative during initialization
    "exploration": 1.1,       # More confident during exploration
    "consolidation": 1.0,     # Normal confidence during consolidation
    "refinement": 0.95        # Slightly more conservative during refinement
}

class GenerationManager:
    """Manages text generation, scaffold integration, and memory handling for the SOVL system."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        base_model: AutoModelForCausalLM,
        scaffolds: List[AutoModelForCausalLM],
        base_tokenizer: AutoTokenizer,
        scaffold_tokenizer: AutoTokenizer,
        state: SOVLState,
        logger: Logger,
        error_manager: ErrorManager,
        cross_attention_injector: Any,
        scaffold_manager: Any,
        device: torch.device,
        curiosity_manager: Any = None,
    ):
        """Initialize GenerationManager with configuration and model components."""
        # Core components
        self._config_manager = config_manager
        self.base_model = base_model.to(device)
        self.scaffolds = [scaffold.to(device) for scaffold in scaffolds]
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer
        self.state = state
        self.logger = logger
        self.error_manager = error_manager
        self.cross_attention_injector = cross_attention_injector
        self.scaffold_manager = scaffold_manager
        self.curiosity_manager = curiosity_manager
        self.device = device
        self.lock = threading.Lock()  # Add lock for synchronization

        # Initialize temperament system
        self._initialize_temperament_system()
        
        # Initialize configuration
        self._initialize_config()
        
        # Initialize lifecycle manager
        self._initialize_lifecycle_manager()
        
        # Log initialization with config values
        self._log_initialization()

        # Memory settings
        self.scaffold_unk_id = self._get_config_value("controls_config.scaffold_unk_id", scaffold_tokenizer.unk_token_id)
        self.use_token_map_memory = self._get_config_value("controls_config.use_token_map_memory", True)
        self.dynamic_cross_attn_mode = self._get_config_value("controls_config.dynamic_cross_attn_mode", None)

        # Generation settings
        self.max_retries = self._get_config_value("controls_config.max_generation_retries", 3)
        self.memory_threshold = self._get_config_value("controls_config.memory_threshold", 0.85)
        self.base_batch_size = self._get_config_value("controls_config.base_batch_size", 1)
        self.generation_callbacks: Dict[str, List[Callable]] = {
            "pre_generate": [],
            "post_generate": []
        }

        # Validate and initialize curiosity state
        self._validate_curiosity_state()

    def _initialize_temperament_system(self) -> None:
        """Initialize the temperament system with validated parameters."""
        try:
            # Get and validate parameters
            params = self._get_validated_temperament_parameters()
            
            # Initialize temperament state
            if not hasattr(self.state, 'temperament_score'):
                self.state.temperament_score = 0.5
            if not hasattr(self.state, 'temperament_history'):
                self.state.temperament_history = deque(maxlen=self._get_config_value("controls_config.temperament_history_maxlen", 10))
            
            # Log initialization
            self.logger.record_event(
                event_type="temperament_system_initialized",
                message="Temperament system initialized with validated parameters",
                level="info",
                additional_info=params
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to initialize temperament system: {str(e)}",
                error_type="temperament_system_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def _get_validated_temperament_parameters(self) -> Dict[str, Any]:
        """Get and validate temperament parameters."""
        # Define safe parameter ranges
        safe_ranges = {
            "temp_smoothing_factor": (0.1, 1.0),
            "temp_eager_threshold": (0.5, 0.9),
            "temp_sluggish_threshold": (0.1, 0.5),
            "temp_mood_influence": (0.1, 0.9),
            "temp_curiosity_boost": (0.1, 0.5),
            "temp_restless_drop": (0.1, 0.5),
            "temp_melancholy_noise": (0.0, 0.2),
            "conf_feedback_strength": (0.1, 0.9),
            "temperament_decay_rate": (0.1, 0.9)
        }
        
        # Get and validate parameters
        params = {}
        for key, (min_val, max_val) in safe_ranges.items():
            value = self._config_manager.get(f"controls_config.{key}", (min_val + max_val) / 2)
            if not (min_val <= value <= max_val):
                self.logger.record_event(
                    event_type="temperament_parameter_warning",
                    message=f"Parameter {key} out of safe range, clamping to bounds",
                    level="warning",
                    additional_info={
                        "parameter": key,
                        "value": value,
                        "min": min_val,
                        "max": max_val
                    }
                )
                value = max(min_val, min(value, max_val))
            params[key] = value
            
        return params

    def update_temperament(self, new_score: float, confidence: float, lifecycle_stage: str) -> None:
        """
        Update the temperament system with new values.
        
        Args:
            new_score: New temperament score (0.0 to 1.0)
            confidence: Confidence level in the update (0.0 to 1.0)
            lifecycle_stage: Current lifecycle stage
        """
        try:
            # Validate inputs
            if not isinstance(new_score, (int, float)) or not 0.0 <= new_score <= 1.0:
                self.logger.record_event(
                    event_type="temperament_update_invalid_score",
                    message=f"Invalid temperament score: {new_score}. Ignoring update.",
                    level="warning",
                    additional_info={
                        "lifecycle_stage": lifecycle_stage,
                        "current_score": self.state.temperament_score
                    }
                )
                return

            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                self.logger.record_event(
                    event_type="temperament_update_invalid_confidence",
                    message=f"Invalid confidence: {confidence}. Ignoring update.",
                    level="warning",
                    additional_info={
                        "lifecycle_stage": lifecycle_stage,
                        "current_score": self.state.temperament_score
                    }
                )
                return
                
            # Get configuration values
            smoothing_factor = self._get_config_value("controls_config.temp_smoothing_factor", 0.5)
            feedback_strength = self._get_config_value("controls_config.conf_feedback_strength", 0.5)
            
            # Update state with new score
            self.state.temperament_score = new_score
            self.state.temperament_history.append(new_score)
            
            # Log the update
            self.logger.record_event(
                event_type="temperament_updated",
                message="Temperament system updated",
                level="info",
                additional_info={
                    "new_score": new_score,
                    "confidence": confidence,
                    "lifecycle_stage": lifecycle_stage,
                    "current_score": self.state.temperament_score,
                    "conversation_id": self.state.history.conversation_id,
                    "state_hash": self.state.state_hash,
                    "smoothing_factor": smoothing_factor,
                    "feedback_strength": feedback_strength
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_update_error",
                message=f"Failed to update temperament: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                    "lifecycle_stage": lifecycle_stage,
                    "current_score": self.state.temperament_score
                }
            )
            raise

    @property
    def current_temperament_score(self) -> float:
        """Get the current temperament score."""
        return self.state.temperament_score
        
    @property
    def mood_label(self) -> str:
        """Get a human-readable mood label based on the current score."""
        score = self.current_temperament_score
        if score < 0.3:
            return "Cautious"
        elif score < 0.7:
            return "Balanced"
        else:
            return "Curious"

    def adjust_parameter(
        self,
        base_value: float,
        parameter_type: str,
        curiosity_pressure: Optional[float] = None
    ) -> float:
        """Adjust a parameter based on current temperament and curiosity pressure."""
        try:
            # Validate inputs
            if not 0.0 <= base_value <= 1.0:
                raise ValueError(f"Base value must be between 0.0 and 1.0, got {base_value}")
            if curiosity_pressure is not None and not 0.0 <= curiosity_pressure <= 1.0:
                raise ValueError(f"Curiosity pressure must be between 0.0 and 1.0, got {curiosity_pressure}")
            
            # Get current temperament score
            current_score = self.current_temperament_score
            
            # Calculate adjustment based on parameter type
            if parameter_type == "temperature":
                # Base adjustment from temperament
                adjustment = (current_score - 0.5) * 0.3  # Scale to ±0.15
                
                # Add curiosity influence if available
                if curiosity_pressure is not None:
                    adjustment += curiosity_pressure * 0.2  # Scale to +0.2
                
                # Apply adjustment with bounds
                adjusted_value = base_value + adjustment
                adjusted_value = max(0.1, min(1.0, adjusted_value))
                
                # Log the adjustment
                self.logger.record_event(
                    event_type="parameter_adjusted",
                    message="Parameter adjusted",
                    level="info",
                    additional_info={
                        "parameter_type": parameter_type,
                        "base_value": base_value,
                        "adjusted_value": adjusted_value,
                        "temperament_score": current_score,
                        "curiosity_pressure": curiosity_pressure,
                        "adjustment": adjustment
                    }
                )
                
                return adjusted_value
                
            else:
                raise ValueError(f"Unsupported parameter type: {parameter_type}")
                
        except Exception as e:
            self.logger.record_event(
                event_type="parameter_adjustment_error",
                message=f"Failed to adjust parameter: {str(e)}",
                level="error",
                additional_info={
                    "parameter_type": parameter_type,
                    "base_value": base_value,
                    "curiosity_pressure": curiosity_pressure
                }
            )
            return base_value  # Return base value on error

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        try:
            # Validate required configuration sections
            required_sections = ["controls_config", "curiosity_config", "training_config"]
            for section in required_sections:
                if not self._config_manager.validate_section(section):
                    raise ValueError(f"Missing required configuration section: {section}")

            # Validate specific configuration values
            self._validate_config_values()

        except Exception as e:
            self._log_error(
                Exception(f"Configuration initialization failed: {str(e)}"),
                "config_initialization",
                traceback.format_exc()
            )
            raise

    def _validate_config_values(self) -> None:
        """Validate specific configuration values."""
        try:
            # Memory threshold validation
            memory_threshold = self._get_config_value("controls_config.memory_threshold", 0.85)
            if not 0.0 <= memory_threshold <= 1.0:
                raise ValueError(f"Invalid memory_threshold: {memory_threshold}")

            # Generation retries validation
            max_retries = self._get_config_value("controls_config.max_generation_retries", 3)
            if not isinstance(max_retries, int) or max_retries < 1:
                raise ValueError(f"Invalid max_generation_retries: {max_retries}")

            # Temperature validation
            base_temperature = self._get_config_value("controls_config.base_temperature", 0.7)
            if not 0.0 <= base_temperature <= 2.0:
                raise ValueError(f"Invalid base_temperature: {base_temperature}")

            # Top-k validation
            top_k = self._get_config_value("curiosity_config.top_k", 30)
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"Invalid top_k: {top_k}")

        except Exception as e:
            self._log_error(
                Exception(f"Configuration validation failed: {str(e)}"),
                "config_validation",
                traceback.format_exc()
            )
            raise

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get a configuration value with validation."""
        try:
            return self._config_manager.get(key, default)
        except Exception as e:
            self._log_error(
                Exception(f"Failed to get config value for {key}: {str(e)}"),
                "config_access",
                traceback.format_exc()
            )
            return default

    def _update_config(self, key: str, value: Any) -> bool:
        """Update a configuration value with validation."""
        try:
            return self._config_manager.update(key, value)
        except Exception as e:
            self._log_error(
                Exception(f"Failed to update config value for {key}: {str(e)}"),
                "config_update",
                traceback.format_exc()
            )
            return False

    def _load_config_sections(self) -> None:
        """Load configuration sections from config manager."""
        self.controls_config = self._config_manager.get_section("controls_config")
        self.curiosity_config = self._config_manager.get_section("curiosity_config")
        self.training_config = self._config_manager.get_section("training_config")
        
    def _log_initialization(self) -> None:
        """Log GenerationManager initialization with config values."""
        self._log_event(
            event_type="generation_manager_init",
            message="GenerationManager initialized successfully",
            level="info",
            additional_info={
                "controls_config": {k: self.controls_config.get(k) for k in [
                    "memory_threshold", "max_generation_retries", "scaffold_unk_id", 
                    "use_token_map_memory", "dynamic_cross_attn_mode", "conversation_history_maxlen",
                    "memory_decay_rate", "enable_repetition_check", "enable_confidence_tracking",
                    "enable_error_listening", "dream_memory_weight", "base_temperature"
                ]},
                "curiosity_config": {k: self.curiosity_config.get(k) for k in [
                    "max_new_tokens", "top_k", "weight_ignorance", "weight_novelty"
                ]}
            }
        )

    def register_callback(self, stage: str, callback: Callable) -> None:
        """Register a callback for generation stages."""
        if stage in self.generation_callbacks:
            self.generation_callbacks[stage].append(callback)
        else:
            self._log_error(
                Exception(f"Invalid callback stage: {stage}"),
                "register_callback"
            )

    def _log_event(self, event_type: str, message: str, level: str = "info", additional_info: Optional[Dict] = None) -> None:
        """Log an event with standardized fields."""
        self.logger.record_event(
            event_type=event_type,
            message=message,
            level=level,
            additional_info={
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "device": str(self.device),
                **(additional_info or {})
            }
        )

    def _log_error(self, error: Exception, context: str, stack_trace: Optional[str] = None) -> None:
        """Log an error with context and stack trace."""
        self.logger.log_error(
            error_msg=str(error),
            error_type=f"generation_{context}_error",
            stack_trace=stack_trace or traceback.format_exc(),
            additional_info={
                "context": context,
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "device": str(self.device),
                "timestamp": time.time()
            }
        )

    def _log_memory_health(self, memory_ratio: float, current_memory: int, total_memory: int) -> None:
        """Log memory health status with standardized metadata."""
        self.logger.log_memory_health(
            model_size=current_memory,
            health_status="healthy" if memory_ratio <= self.memory_threshold else "unhealthy",
            device=self.device,
            additional_info={
                "memory_ratio": memory_ratio,
                "current_memory": current_memory,
                "total_memory": total_memory,
                "threshold": self.memory_threshold,
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "timestamp": time.time()
            }
        )

    def check_memory_health(self) -> bool:
        """Check GPU memory usage and clean up if necessary."""
        if not torch.cuda.is_available():
            return True

        try:
            memory_stats = torch.cuda.memory_stats()
            current_memory = memory_stats["allocated_bytes.all.current"]
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_ratio = current_memory / total_memory

            # Log memory health using standardized method
            self._log_memory_health(memory_ratio, current_memory, total_memory)

            if memory_ratio > self.memory_threshold:
                torch.cuda.empty_cache()
                self._log_event(
                    "memory_cleanup",
                    {
                        "reason": "threshold_exceeded",
                        "memory_ratio": memory_ratio,
                        "threshold": self.memory_threshold
                    }
                )
                return False
            return True
        except Exception as e:
            self._log_error(
                Exception(f"Memory health check failed: {str(e)}"),
                "check_memory_health",
                traceback.format_exc()
            )
            return False

    def _handle_error_prompt(self, error_msg: str) -> str:
        """Generate a response to a system error."""
        try:
            temp_history = self.state.history
            self.state.history = ConversationHistory(
                maxlen=self.controls_config.get("conversation_history_maxlen", 10)
            )
            response = self.generate(
                f"System error detected: {error_msg} What happened?",
                max_new_tokens=self.curiosity_config.get("max_new_tokens", 60),
                temperature=self.controls_config.get("base_temperature", 0.7) + 0.2,
                top_k=self.curiosity_config.get("top_k", 50),
                do_sample=True
            )
            self._log_event(
                "error_prompt_handled",
                {
                    "prompt": f"System error detected: {error_msg} What happened?",
                    "response": response,
                    "is_error_prompt": True,
                    "confidence_score": 0.5
                }
            )
            self.state.history = temp_history
            return response
        except Exception as e:
            self._log_error(
                Exception(f"Error prompt handling failed: {str(e)}"),
                "handle_error_prompt",
                traceback.format_exc()
            )
            return "An error occurred while handling the error prompt"

    def has_repetition(self, output_ids: torch.Tensor, n: int = 3) -> bool:
        """Check for repetition in generated output."""
        try:
            ids = output_ids.tolist()
            special_ids = {
                self.base_tokenizer.pad_token_id,
                self.base_tokenizer.eos_token_id,
                self.base_tokenizer.bos_token_id,
                self.base_tokenizer.unk_token_id
            }
            filtered = [i for i in ids if i not in special_ids]
            for i in range(len(filtered) - 2 * n):
                if filtered[i:i + n] == filtered[i + n:i + 2 * n]:
                    return True
            return False
        except Exception as e:
            self._log_error(
                Exception(f"Repetition check failed: {str(e)}"),
                "has_repetition",
                traceback.format_exc()
            )
            return False

    def tokenize_and_map(
        self,
        prompts: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = 'max_length'
    ) -> Dict[str, torch.Tensor]:
        """Tokenize prompts and map to scaffold tokens."""
        try:
            max_length = max_length or self.training_config.get("max_seq_length", 128)
            prompts = [prompts] if isinstance(prompts, str) else prompts

            batch_size = self.training_config.get("batch_size", 1)
            input_batches = [prompts[i: i + batch_size] for i in range(0, len(prompts), batch_size)]
            all_input_ids = []
            all_attention_masks = []

            for batch in input_batches:
                inputs = self.base_tokenizer(
                    batch,
                    return_tensors='pt',
                    padding=padding,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)
                scaffold_input_ids = self.scaffold_manager.map_sequence(inputs.input_ids)
                scaffold_attention_mask = (
                    scaffold_input_ids != self.scaffold_tokenizer.pad_token_id
                ).int()
                all_input_ids.append(scaffold_input_ids)
                all_attention_masks.append(scaffold_attention_mask)

            return {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
            }
        except Exception as e:
            self._log_error(
                Exception(f"Tokenization failed: {str(e)}"),
                "tokenize_and_map",
                traceback.format_exc()
            )
            raise

    def get_scaffold_hidden_states(self, scaffold_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get hidden states from scaffold model."""
        try:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16
            ):
                scaffold_outputs = self.scaffolds[0](
                    **{k: v for k, v in scaffold_inputs.items() if k in ['input_ids', 'attention_mask']},
                    output_hidden_states=True,
                )
                hidden_states = (
                    scaffold_outputs.hidden_states[-1]
                    if hasattr(scaffold_outputs, 'hidden_states')
                    else scaffold_outputs.base_model_output.hidden_states[-1]
                )
                return hidden_states.detach()
        except Exception as e:
            self._log_error(
                Exception(f"Scaffold hidden states failed: {str(e)}"),
                "get_scaffold_hidden_states",
                traceback.format_exc()
            )
            raise

    @contextlib.contextmanager
    def _scaffold_context(self, scaffold_hidden_states: torch.Tensor):
        """Manage scaffold context with memory monitoring."""
        try:
            with self.state.memory_lock:
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    if mem_before > self.memory_threshold:
                        torch.cuda.empty_cache()
                self._temp_scaffold_context = scaffold_hidden_states.detach()
            yield
        finally:
            self._clear_scaffold_cache()

    def _clear_scaffold_cache(self) -> None:
        """Clear scaffold-related caches with memory optimization."""
        with self.state.memory_lock:
            try:
                if hasattr(self, '_temp_scaffold_context') and self._temp_scaffold_context is not None:
                    if isinstance(self._temp_scaffold_context, torch.Tensor):
                        self._temp_scaffold_context = self._temp_scaffold_context.to('cpu')
                    del self._temp_scaffold_context
                self._temp_scaffold_context = None

                if self.state.last_prompt_embedding is not None:
                    if isinstance(self.state.last_prompt_embedding, torch.Tensor):
                        self.state.last_prompt_embedding = self.state.last_prompt_embedding.to('cpu')
                    del self.state.last_prompt_embedding
                    self.state.last_prompt_embedding = None

                if self.state.dream_memory:
                    new_memory = deque(maxlen=self.state.dream_memory_maxlen)
                    for tensor, weight in self.state.dream_memory:
                        new_memory.append((tensor.to('cpu'), weight))
                    self.state.dream_memory = new_memory

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self._log_event(
                    "scaffold_cache_cleared",
                    {}
                )
            except Exception as e:
                self._log_error(
                    Exception(f"Failed to clear scaffold cache: {str(e)}"),
                    "clear_scaffold_cache",
                    traceback.format_exc()
                )

    def _update_token_map_memory(self, prompt: str, confidence: float) -> None:
        """Update token map weights."""
        if not self.use_token_map_memory:
            return
        try:
            self.scaffold_manager.update_token_map_memory(
                prompt=prompt,
                confidence=confidence,
                tokenizer=self.base_tokenizer,
                memory_decay_rate=self.controls_config.get("memory_decay_rate", 0.95),
            )
        except Exception as e:
            self._log_error(
                Exception(f"Token map update failed: {str(e)}"),
                "update_token_map_memory",
                traceback.format_exc()
            )

    def prepare_for_training(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare data for training."""
        try:
            prompts = [item['prompt'] for item in batch]
            scaffold_inputs = self.tokenize_and_map(prompts)
            scaffold_hidden_states = self.get_scaffold_hidden_states(scaffold_inputs)
            return {
                "scaffold_hidden_states": scaffold_hidden_states,
                "prompts": prompts
            }
        except Exception as e:
            self._log_error(
                Exception(f"Training preparation failed: {str(e)}"),
                "prepare_for_training",
                traceback.format_exc()
            )
            raise

    def _prepare_generation_params(self, max_new_tokens: int, scaffold_weight: Optional[float], **kwargs) -> Dict[str, Any]:
        """Prepare and validate generation parameters."""
        try:
            return {
                "max_new_tokens": max_new_tokens or self._get_config_value("curiosity_config.max_new_tokens", 8),
                "scaffold_weight": scaffold_weight,
                "temperature": kwargs.get("temperature", self._get_config_value("controls_config.base_temperature", 0.7)),
                "top_k": kwargs.get("top_k", self._get_config_value("curiosity_config.top_k", 30)),
                "do_sample": kwargs.get("do_sample", False),
                "prompt_count": 1 if isinstance(kwargs.get("prompts"), str) else len(kwargs.get("prompts", [])),
            }
        except Exception as e:
            self._log_error(
                Exception(f"Failed to prepare generation parameters: {str(e)}"),
                "prepare_generation_params",
                traceback.format_exc()
            )
            raise

    def _compute_dynamic_factor(self) -> Optional[torch.Tensor]:
        """Compute dynamic cross-attention factor based on configuration."""
        if not self.controls_config.get("enable_dynamic_cross_attention", False) or not self.dynamic_cross_attn_mode:
            return None

        try:
            last_conf = self.state.confidence_history[-1] if self.state.confidence_history else 0.5
            if self.dynamic_cross_attn_mode == 'confidence':
                return torch.tensor(last_conf, device=self.device, dtype=torch.float)
            elif self.dynamic_cross_attn_mode == 'temperament':
                return torch.tensor(self.temperament.score, device=self.device, dtype=torch.float)
            return None
        except Exception as e:
            self._log_error(
                Exception(f"Failed to compute dynamic factor: {str(e)}"),
                "compute_dynamic_factor",
                traceback.format_exc()
            )
            return None

    def _prepare_dream_memory(self) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Prepare dream memory tensors if available."""
        dream_memory_info = {"used": False, "tensor_count": 0, "shapes": []}
        memory_tensors = None
        dream_memory_weight = self.controls_config.get("dream_memory_weight", 0.1)

        if self.state.dream_memory and dream_memory_weight > 0:
            try:
                with self.state.memory_lock:
                    dream_tensors, dream_weights = zip(*self.state.dream_memory)
                    dream_memory_info["tensor_count"] = len(dream_tensors)
                    dream_memory_info["shapes"] = [list(t.shape) for t in dream_tensors]
                    for tensor in dream_tensors:
                        if tensor.shape[-1] != self.state.hidden_size:
                            raise ValueError(
                                f"Dream tensor shape {tensor.shape} mismatches hidden_size {self.state.hidden_size}"
                            )
                    dream_tensors = torch.stack([t.detach().to(self.device) for t in dream_tensors])
                    dream_weights = torch.tensor(dream_weights, dtype=torch.float32, device=self.device)
                    memory_tensors = torch.sum(dream_tensors * dream_weights.unsqueeze(-1), dim=0) / dream_weights.sum()
                    dream_memory_info["used"] = True
            except Exception as e:
                dream_memory_info["error"] = str(e)
                self._log_error(
                    Exception(f"Dream memory preparation failed: {str(e)}"),
                    "prepare_dream_memory",
                    traceback.format_exc()
                )

        return memory_tensors, dream_memory_info

    def _handle_repetition(self, seq: torch.Tensor, seq_ids: List[int], outputs: Any) -> List[int]:
        """Handle detected repetition in generated sequence."""
        if self.controls_config.get("enable_repetition_check", True) and self.has_repetition(seq):
            original_text = self.base_tokenizer.decode(seq_ids, skip_special_tokens=True)
            for j in range(len(seq_ids) - 6):
                if all(seq_ids[j + k] == seq_ids[j + k + 3] for k in range(3)):
                    seq_ids = seq_ids[:j + 3]
                    break
            self._log_event(
                "repetition_detected",
                {
                    "original_text": original_text,
                    "truncated_at": j + 3
                }
            )
        return seq_ids

    def _update_curiosity(self, text: str, confidence: float) -> None:
        """Update curiosity state with generated text and confidence."""
        try:
            if not self.curiosity_manager:
                return
                
            # Tokenize and get embeddings for the generated text
            inputs = self.base_tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.base_model(**inputs, output_hidden_states=True)
                text_embedding = outputs.hidden_states[-1].mean(dim=1)
            
            # Update curiosity with the new text and confidence
            self.curiosity_manager.update_curiosity(
                text=text,
                confidence=confidence,
                embedding=text_embedding
            )
            
            # Log curiosity update
            self._log_event(
                "curiosity_update",
                {
                    "text": text[:200],  # Log first 200 chars
                    "confidence": confidence,
                    "embedding_shape": text_embedding.shape,
                    "curiosity_pressure": self.curiosity_manager.get_pressure()
                }
            )
            
        except Exception as e:
            self._log_error(
                Exception(f"curiosity_update_error: {str(e)}"),
                "curiosity_update_error",
                {
                    "text": text[:200],
                    "confidence": confidence
                }
            )

    @synchronized()
    def calculate_confidence_score(self, logits: torch.Tensor, generated_ids: List[int]) -> float:
        """Calculate confidence score for generated output."""
        try:
            if not self.curiosity_manager:
                return 0.5
                
            # Get base and scaffold confidences
            base_conf = self.curiosity_manager.compute_curiosity(
                base_conf=0.5,  # Default base confidence
                scaf_conf=0.5,  # Default scaffold confidence
                state=self.state,
                query_embedding=self.curiosity_manager._generate_query_embedding(
                    self.state,
                    self.base_tokenizer,
                    self.base_model,
                    self.base_tokenizer.decode(generated_ids)
                ),
                device=self.device
            )
            
            return base_conf
            
        except Exception as e:
            self._log_error(
                Exception(f"Confidence score calculation failed: {str(e)}"),
                "calculate_confidence_score",
                traceback.format_exc()
            )
            return 0.5

    def _initialize_lifecycle_manager(self) -> None:
        """Initialize the lifecycle manager with validated parameters."""
        try:
            # Get training config
            training_config = TrainingConfig(self._config_manager)
            
            # Initialize lifecycle manager
            self.lifecycle_manager = LifecycleManager(
                config=training_config,
                model=self.base_model,
                state=self.state
            )
            
            # Log initialization
            self.logger.record_event(
                event_type="lifecycle_manager_initialized",
                message="Lifecycle manager initialized with validated parameters",
                level="info",
                additional_info={
                    "data_exposure": self.lifecycle_manager.data_exposure,
                    "lora_capacity": self.lifecycle_manager.lora_capacity
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to initialize lifecycle manager: {str(e)}",
                error_type="lifecycle_manager_error",
                stack_trace=traceback.format_exc()
            )
            raise

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        lifecycle_stage: str = "active",
        **kwargs
    ) -> List[str]:
        """Generate text using the base model with scaffold integration."""
        try:
            # Check memory health before generation
            self._manage_memory()
            
            # Validate device state
            if not self._validate_device_state():
                raise RuntimeError("Device state validation failed")

            # Log state metadata
            self._log_state_metadata()

            # Update temperament based on curiosity manager if available
            if self.curiosity_manager:
                curiosity_score = self.curiosity_manager.get_curiosity_score()
                confidence = self.curiosity_manager.get_confidence()
                self.update_temperament(curiosity_score, confidence, lifecycle_stage)

            # Calculate curiosity pressure and query embedding
            curiosity_pressure = None
            query_embedding = None
            if self.curiosity_manager:
                curiosity_pressure = self.curiosity_manager.calculate_curiosity_pressure(prompt)
                query_embedding = self.curiosity_manager.get_query_embedding(prompt)

            # Get current temperament state
            current_temperament = self.current_temperament_score
            mood_label = self.mood_label

            # Adjust parameters based on temperament
            adjusted_temperature = self.adjust_parameter(
                base_value=temperature,
                parameter_type="temperature",
                curiosity_pressure=curiosity_pressure
            )
            
            adjusted_max_length = self._adjust_max_length(
                base_length=max_length,
                temperament_score=current_temperament,
                curiosity_pressure=curiosity_pressure
            )

            # Compute base confidence with temperament influence
            base_confidence = self.calculate_confidence_score(
                self._get_last_logits(),
                self.base_tokenizer.encode(prompt)
            ) if hasattr(self, '_get_last_logits') else DEFAULT_CONFIDENCE

            # Apply confidence adjustments with temperament influence
            adjusted_confidence = self._apply_confidence_adjustments(base_confidence)

            # Prepare base generation parameters with temperament-based adjustments
            base_params = {
                "max_length": adjusted_max_length,
                "temperature": adjusted_temperature,
                "top_p": top_p,
                "num_return_sequences": num_return_sequences,
                "do_sample": do_sample,
                **kwargs
            }

            # Create dynamic mood context based on temperament state
            mood_context = self._create_dynamic_mood_context(
                mood_label=mood_label,
                temperament_score=current_temperament,
                confidence=adjusted_confidence
            )
            enhanced_prompt = f"{mood_context}\n{prompt}"

            # Log generation start with temperament-driven parameters
            self.logger.record_event(
                event_type="generation_started",
                message="Starting text generation with temperament-driven parameters",
                level="info",
                additional_info={
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "base_confidence": base_confidence,
                    "adjusted_confidence": adjusted_confidence,
                    "original_params": {
                        "max_length": max_length,
                        "temperature": temperature,
                        "top_p": top_p
                    },
                    "adjusted_params": base_params,
                    "mood_label": mood_label,
                    "temperament_score": current_temperament,
                    "lifecycle_stage": lifecycle_stage
                }
            )

            # Generate text with adjusted parameters
            generated_texts = self._generate_text(
                prompt=enhanced_prompt,
                **base_params
            )

            # Check for repetitions in generated text
            if generated_texts:
                special_ids = {
                    self.base_tokenizer.pad_token_id,
                    self.base_tokenizer.eos_token_id,
                    self.base_tokenizer.bos_token_id,
                    self.base_tokenizer.unk_token_id
                }
                
                for i, text in enumerate(generated_texts):
                    token_ids = self.base_tokenizer.encode(text)
                    repetition = detect_repetitions(
                        token_ids=token_ids,
                        special_ids=special_ids,
                        config_manager=self._config_manager,
                        logger=self.logger
                    )
                    
                    if repetition:
                        self._log_event(
                            "repetition_detected",
                            f"Repetition detected in generated text {i}",
                            level="warning",
                            additional_info={
                                "text": text,
                                "repetition_indices": repetition
                            }
                        )
                        # Truncate text at repetition point
                        generated_texts[i] = self.base_tokenizer.decode(token_ids[:repetition[0]])

            # Update state with generation results
            if generated_texts:
                # Update conversation history
                self.state.history.add_message("assistant", generated_texts[0])
                
                # Update confidence history with temperament context
                self.state.add_confidence(adjusted_confidence)
                
                # Update curiosity state
                if self.curiosity_manager:
                    self.curiosity_manager.update_curiosity_state(
                        prompt=prompt,
                        response=generated_texts[0],
                        query_embedding=query_embedding
                    )

                # Update temperament based on generation results
                self._update_temperament_from_generation(
                    generated_text=generated_texts[0],
                    confidence=adjusted_confidence
                )

            # Log generation completion with temperament updates
            self.logger.record_event(
                event_type="generation_completed",
                message="Text generation completed with temperament updates",
                level="info",
                additional_info={
                    "prompt": prompt,
                    "generated_texts": generated_texts,
                    "base_confidence": base_confidence,
                    "adjusted_confidence": adjusted_confidence,
                    "mood_label": mood_label,
                    "temperament_score": current_temperament,
                    "lifecycle_stage": lifecycle_stage,
                    "conversation_length": len(self.state.history.messages)
                }
            )

            return generated_texts

        except torch.cuda.OutOfMemoryError:
            self._handle_state_driven_error(
                Exception("CUDA out of memory during generation"),
                "generation_oom"
            )
            raise

        except Exception as e:
            self._handle_state_driven_error(e, "generation_error")
            raise

    def _validate_device_state(self) -> bool:
        """Validate that all models are on the correct device."""
        try:
            # Check base model
            if self.base_model.device != self.device:
                self._log_warning(f"Base model device mismatch: {self.base_model.device} != {self.device}")
                self.base_model = self.base_model.to(self.device)
            
            # Check scaffolds
            for i, scaffold in enumerate(self.scaffolds):
                if scaffold.device != self.device:
                    self._log_warning(f"Scaffold {i} device mismatch: {scaffold.device} != {self.device}")
                    self.scaffolds[i] = scaffold.to(self.device)
                    
            return True
        except Exception as e:
            self._log_error(
                Exception(f"device_validation_error: {str(e)}"),
                "device_validation",
                traceback.format_exc()
            )
            return False

    def _log_state_metadata(self) -> None:
        """Log state metadata for debugging."""
        self.logger.record_event(
            event_type="state_metadata",
            message="Logging state metadata",
            level="info",
            additional_info={
                "conversation_id": self.state.history.conversation_id,
                "state_hash": self.state.state_hash,
                "temperament_score": self.current_temperament_score,
                "mood_label": self.mood_label,
                "curiosity_score": self.curiosity_manager.get_curiosity_score() if self.curiosity_manager else None,
                "confidence": self.curiosity_manager.get_confidence() if self.curiosity_manager else None
            }
        )

    def _validate_curiosity_state(self) -> None:
        """Validate and initialize curiosity state if needed."""
        try:
            if not hasattr(self.state, 'curiosity'):
                self.state.curiosity = CuriosityState(
                    pressure=0.0,
                    unanswered_questions=deque(maxlen=self.curiosity_config.get("queue_maxlen", 10)),
                    novelty_scores=deque(maxlen=self.curiosity_config.get("metrics_maxlen", 1000))
                )
                self._log_event(
                    "curiosity_state_init",
                    {
                        "reason": "missing_curiosity_state",
                        "queue_maxlen": self.curiosity_config.get("queue_maxlen", 10),
                        "metrics_maxlen": self.curiosity_config.get("metrics_maxlen", 1000)
                    }
                )
            
            # Validate curiosity manager
            if self.curiosity_manager is None:
                self._log_warning(
                    "curiosity_manager_missing",
                    "Curiosity manager not provided, curiosity features will be disabled"
                )
            else:
                # Ensure curiosity manager is using the same state
                self.curiosity_manager.set_state(self.state)
                self._log_event(
                    "curiosity_manager_sync",
                    {
                        "state_hash": self.state.state_hash,
                        "conversation_id": self.state.history.conversation_id
                    }
                )
                
        except Exception as e:
            self._log_error(
                Exception(f"Failed to validate curiosity state: {str(e)}"),
                "curiosity_validation",
                traceback.format_exc()
            )
            raise

    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        query_embedding: torch.Tensor,
    ) -> float:
        """Compute curiosity score based on confidence and embeddings."""
        try:
            if not self.curiosity_manager:
                return 0.5

            # Get memory embeddings with memory limits
            memory_embeddings = self._get_valid_memory_embeddings(self.state)
            
            # Compute curiosity score
            score = self.curiosity_manager.compute_curiosity(
                base_conf=base_conf,
                scaf_conf=scaf_conf,
                state=self.state,
                query_embedding=query_embedding,
                device=self.device
            )
            
            return score
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to compute curiosity: {str(e)}"),
                "compute_curiosity",
                traceback.format_exc()
            )
            return 0.5

    def _get_valid_memory_embeddings(self, state: SOVLState) -> List[torch.Tensor]:
        """Get valid memory embeddings while respecting memory limits."""
        try:
            memory_embeddings = []
            hidden_size = self._config_manager.get("hidden_size")
            total_size = 0.0
            
            if not hasattr(state, 'dream_memory') or not state.dream_memory:
                return memory_embeddings
                
            for entry in state.dream_memory:
                tensor = entry["tensor"]
                # Calculate memory size in MB
                memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                
                # Check memory limit
                if total_size + memory_size > self._config_manager.get("max_memory_mb", 512.0):
                    self._log_event("memory_limit_reached", {
                        "total_size_mb": total_size,
                        "skipped_size_mb": memory_size,
                        "max_memory_mb": self._config_manager.get("max_memory_mb", 512.0)
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
                    
            return memory_embeddings
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to get valid memory embeddings: {str(e)}"),
                "get_valid_memory_embeddings",
                traceback.format_exc()
            )
            return []

    @synchronized()
    def _manage_memory(self) -> None:
        """Manage memory usage and integrate with state system."""
        try:
            # Get current memory stats using the utility function
            memory_stats = memory_usage(self.device, self._config_manager)
            
            # Log memory usage
            self._log_memory_health(
                memory_ratio=memory_stats.get("allocated", 0.0) / memory_stats.get("reserved", 1.0),
                current_memory=memory_stats.get("allocated", 0),
                total_memory=memory_stats.get("reserved", 1)
            )
            
            # Check if memory usage exceeds threshold
            if memory_stats.get("allocated", 0.0) / memory_stats.get("reserved", 1.0) > self.memory_threshold:
                # Clear scaffold cache
                self._clear_scaffold_cache()
                
                # Prune dream memory if available
                if hasattr(self.state, 'dream_memory'):
                    self.state._prune_dream_memory()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log memory cleanup
                self._log_event(
                    "memory_cleanup",
                    "Memory cleanup performed",
                    level="info",
                    additional_info={
                        "memory_stats": memory_stats,
                        "threshold": self.memory_threshold
                    }
                )
            
            # Update state with current memory stats
            self.state._update_memory_usage()
            
        except Exception as e:
            self._log_error(
                Exception(f"Memory management failed: {str(e)}"),
                "memory_management",
                traceback.format_exc()
            )

    def _get_dynamic_batch_size(self, prompts: List[str]) -> int:
        """Get dynamic batch size based on available memory."""
        try:
            return dynamic_batch_size(
                base_size=self.base_batch_size,
                config_manager=self._config_manager,
                logger=self.logger
            )
        except Exception as e:
            self._log_error(
                Exception(f"Failed to get dynamic batch size: {str(e)}"),
                "dynamic_batch_size",
                traceback.format_exc()
            )
            return self.base_batch_size

    def _adjust_max_length(
        self,
        base_length: int,
        temperament_score: float,
        curiosity_pressure: Optional[float] = None
    ) -> int:
        """Adjust max length based on state parameters."""
        try:
            # Base adjustment from temperament
            adjustment = (temperament_score - 0.5) * 0.2  # Scale to ±0.1
            
            # Add curiosity influence if available
            if curiosity_pressure is not None:
                adjustment += curiosity_pressure * 0.15  # Scale to +0.15
            
            # Apply adjustment with bounds
            adjusted_length = int(base_length * (1 + adjustment))
            adjusted_length = max(50, min(200, adjusted_length))
            
            return adjusted_length
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to adjust max length: {str(e)}"),
                "adjust_max_length",
                traceback.format_exc()
            )
            return base_length

    @synchronized()
    def _handle_state_driven_error(self, error: Exception, context: str) -> None:
        """Handle errors with state-driven recovery strategies."""
        try:
            # Log the error with state context
            self._log_error(error, context)
            
            # Get current state metrics
            memory_stats = memory_usage(self.device, self._config_manager)
            temperament_score = self.current_temperament_score
            confidence_history = self.state.get_confidence_history()
            
            # Determine recovery strategy based on state
            if memory_stats.get("allocated", 0.0) / memory_stats.get("reserved", 1.0) > 0.9:
                # Memory-related error
                self._manage_memory()
                self._log_event(
                    "error_recovery_memory",
                    "Performing memory cleanup as error recovery",
                    level="info",
                    additional_info={
                        "memory_usage": memory_stats,
                        "error_context": context
                    }
                )
                
            elif len(confidence_history) > 0 and confidence_history[-1] < 0.3:
                # Low confidence error
                self._log_event(
                    "error_recovery_confidence",
                    "Adjusting generation parameters due to low confidence",
                    level="info",
                    additional_info={
                        "last_confidence": confidence_history[-1],
                        "error_context": context
                    }
                )
                # Reset confidence history
                self.state.clear_confidence_history()
                
            elif temperament_score < 0.3:
                # Low temperament error
                self._log_event(
                    "error_recovery_temperament",
                    "Adjusting temperament due to error",
                    level="info",
                    additional_info={
                        "temperament_score": temperament_score,
                        "error_context": context
                    }
                )
                # Reset temperament to neutral
                self.update_temperament(0.5, 0.5, "error_recovery")
                
            else:
                # General error recovery
                self._log_event(
                    "error_recovery_general",
                    "Performing general error recovery",
                    level="info",
                    additional_info={
                        "error_context": context,
                        "state_hash": self.state.state_hash,
                        "memory_usage": memory_stats
                    }
                )
                
            # Update state with error information
            self.state.set_cached(
                f"last_error_{context}",
                {
                    "timestamp": time.time(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "memory_stats": memory_stats,
                    "temperament_score": temperament_score,
                    "confidence_history_length": len(confidence_history)
                }
            )
            
        except Exception as recovery_error:
            self._log_error(
                Exception(f"Error recovery failed: {str(recovery_error)}"),
                "error_recovery",
                traceback.format_exc()
            )

    def _create_dynamic_mood_context(
        self,
        mood_label: str,
        temperament_score: float,
        confidence: float
    ) -> str:
        """Create a dynamic mood context based on current temperament state."""
        mood_intensity = abs(temperament_score - 0.5) * 2  # Scale to 0-1
        
        if mood_label == "Cautious":
            return f"[Mood: Cautious (Intensity: {mood_intensity:.2f})] Please provide a careful and well-considered response, focusing on accuracy and reliability."
        elif mood_label == "Curious":
            return f"[Mood: Curious (Intensity: {mood_intensity:.2f})] Please provide an exploratory and creative response, considering novel perspectives."
        else:  # Balanced
            return f"[Mood: Balanced (Intensity: {mood_intensity:.2f})] Please provide a balanced response, considering both reliability and creativity."

    def _update_temperament_from_generation(
        self,
        generated_text: str,
        confidence: float
    ) -> None:
        """Update temperament based on generation results."""
        try:
            # Calculate text characteristics that might influence temperament
            text_length = len(generated_text)
            complexity = len(set(generated_text.split())) / len(generated_text.split())
            
            # Adjust temperament based on generation characteristics
            if confidence > 0.7:
                # Successful generation boosts confidence
                adjustment = 0.05
            elif confidence < 0.3:
                # Low confidence leads to more cautious behavior
                adjustment = -0.05
            else:
                adjustment = 0.0
            
            # Apply adjustment with bounds
            new_score = max(0.0, min(1.0, self.current_temperament_score + adjustment))
            self.update_temperament(new_score, confidence, "generation")
            
        except Exception as e:
            self._log_error(
                Exception(f"Failed to update temperament from generation: {str(e)}"),
                "temperament_update",
                traceback.format_exc()
            )

    def _apply_confidence_adjustments(self, base_confidence: float) -> float:
        """Apply confidence adjustments based on temperament and lifecycle."""
        try:
            # Get current mood and lifecycle stage
            mood_label = self.mood_label
            lifecycle_stage = self.lifecycle_manager.get_lifecycle_stage() if self.lifecycle_manager else "active"
            
            # Apply mood-based multiplier
            mood_multiplier = TEMPERAMENT_MOOD_MULTIPLIERS.get(mood_label, 1.0)
            
            # Apply lifecycle stage multiplier
            lifecycle_multiplier = LIFECYCLE_STAGE_MULTIPLIERS.get(lifecycle_stage, 1.0)
            
            # Calculate adjusted confidence
            adjusted_confidence = base_confidence * mood_multiplier * lifecycle_multiplier
            
            # Log the adjustments
            self.logger.record_event(
                event_type="confidence_adjustment_applied",
                message="Applied confidence adjustments",
                level="info",
                additional_info={
                    "base_confidence": base_confidence,
                    "adjusted_confidence": adjusted_confidence,
                    "mood_label": mood_label,
                    "lifecycle_stage": lifecycle_stage,
                    "mood_multiplier": mood_multiplier,
                    "lifecycle_multiplier": lifecycle_multiplier
                }
            )
            
            return adjusted_confidence
            
        except Exception as e:
            self.logger.record_event(
                event_type="confidence_adjustment_failed",
                message=f"Failed to apply confidence adjustments: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            return base_confidence

    def _adjust_generation_parameters(self, confidence: float, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust generation parameters based on confidence score."""
        try:
            adjusted_params = base_params.copy()
            
            # Adjust temperature based on confidence
            confidence_factor = 1.0 + (confidence - 0.5) * 0.4  # Scale to ±0.2
            adjusted_params["temperature"] = max(0.1, min(2.0, 
                base_params.get("temperature", 0.7) * confidence_factor))
            
            # Adjust max_length based on confidence
            length_factor = 1.0 + (confidence - 0.5) * 0.2  # Scale to ±0.1
            adjusted_params["max_length"] = int(max(50, min(200,
                base_params.get("max_length", 100) * length_factor)))
            
            # Adjust top_p based on confidence
            top_p_factor = 1.0 - (confidence - 0.5) * 0.2  # Scale to ±0.1
            adjusted_params["top_p"] = max(0.1, min(1.0,
                base_params.get("top_p", 0.9) * top_p_factor))
            
            # Log parameter adjustments
            self.logger.record_event(
                event_type="generation_parameters_adjusted",
                message="Adjusted generation parameters based on confidence",
                level="info",
                additional_info={
                    "confidence": confidence,
                    "original_params": base_params,
                    "adjusted_params": adjusted_params
                }
            )
            
            return adjusted_params
            
        except Exception as e:
            self.logger.record_event(
                event_type="parameter_adjustment_failed",
                message=f"Failed to adjust generation parameters: {str(e)}",
                level="error",
                additional_info={"error": str(e), "stack_trace": traceback.format_exc()}
            )
            return base_params

def calculate_confidence(logits: torch.Tensor, generated_ids: torch.Tensor) -> float:
    """Calculate confidence score for generated tokens."""
    try:
        # Get probabilities for generated tokens
        probs = torch.softmax(logits, dim=-1)
        token_probs = torch.gather(probs, -1, generated_ids.unsqueeze(-1)).squeeze(-1)
        
        # Calculate average confidence
        confidence = token_probs.mean().item()
        return max(0.0, min(1.0, confidence))
    except Exception as e:
        logger.record({
            "error": f"Confidence calculation failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })
        return 0.5

def detect_repetitions(token_ids: List[int], special_ids: Set[int], min_rep_length: int = 3) -> Optional[Tuple[int, int]]:
    """Detect repeating token sequences."""
    try:
        filtered = [i for i in token_ids if i not in special_ids]
        for i in range(len(filtered) - 2 * min_rep_length + 1):
            window = filtered[i:i + min_rep_length]
            next_window = filtered[i + min_rep_length:i + 2 * min_rep_length]
            if window == next_window:
                return (i, i + min_rep_length)
        return None
    except Exception as e:
        logger.record({
            "error": f"Repetition detection failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })
        return None

def adjust_temperature(
    base_temp: float,
    temperament_score: float,
    mood_influence: float = 0.3,
    min_temp: float = 0.5,
    max_temp: float = 1.5,
    curiosity_pressure: Optional[float] = None
) -> float:
    """Adjust temperature based on temperament and curiosity."""
    try:
        # Clamp input values
        base_temp = max(min_temp, min(max_temp, base_temp))
        temperament_score = max(-1.0, min(1.0, temperament_score))
        mood_influence = max(0.0, min(1.0, mood_influence))
        
        # Calculate temperature adjustment
        temp_adjustment = mood_influence * 0.3 * temperament_score
        
        # Check if curiosity_pressure is valid before using it
        if curiosity_pressure is not None:
            curiosity_pressure = max(0.0, min(1.0, curiosity_pressure))
            temp_adjustment += curiosity_pressure * 0.1
        
        # Apply adjustment
        adjusted_temp = max(min_temp, min(max_temp, base_temp + temp_adjustment))
        return adjusted_temp
    except Exception as e:
        logger.record({
            "error": f"Temperature adjustment failed: {str(e)}",
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc()
        })
        return base_temp
