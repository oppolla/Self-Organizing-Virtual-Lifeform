import torch
import time
from collections import deque, defaultdict
from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Set
import contextlib
import traceback
from functools import wraps
from datetime import datetime
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_utils import detect_repetitions, adjust_temperature, synchronized, dynamic_batch_size
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_curiosity import CuriosityManager
from sovl_trainer import LifecycleManager, TrainingConfig
from sovl_temperament import TemperamentSystem 
from sovl_confidence import ConfidenceCalculator, calculate_confidence_score
from sovl_queue import capture_scribe_event
from sovl_memory import GenerationMemoryManager
from sovl_scaffold import GenerationScaffoldProvider
from sovl_bonder import BondCalculator
from sovl_primer import GenerationPrimer  # Import GenerationPrimer for trait aggregation and management

class GenerationError(Exception):
    """Raised when text generation fails in a way that should halt upstream processing."""
    pass

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
        generation_hooks: Dict[str, bool] = {},
        dialogue_context_manager: Optional[Any] = None
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
        self.dialogue_context_manager = dialogue_context_manager

        # Generation hooks setup
        self.generation_hooks = generation_hooks or {}
        self.logger.record_event(
            event_type="generation_hooks_initialized",
            message=f"GenerationManager initialized with hooks: {self.generation_hooks}",
            level="info",
            component="GenerationManager"
        )

        self.primer = GenerationPrimer(
            config_manager=self._config_manager,
            logger=self.logger,
            state=self.state,
            error_manager=self.error_manager,
            curiosity_manager=self.curiosity_manager,
            temperament_system=getattr(self, 'temperament_system', None),
            confidence_calculator=getattr(self, 'confidence_calculator', None),
            bond_calculator=getattr(self, 'bond_calculator', None),
            device=self.device,
            lifecycle_manager=getattr(self, 'lifecycle_manager', None),
            scaffold_manager=self.scaffold_manager,
            memory_manager=getattr(self, 'memory_manager', None),
            generation_hooks=self.generation_hooks  # Pass hooks explicitly
        )
        
        # Get global session_id from config
        self.session_id = self._config_manager.get("runtime.session_id")
        if not self.session_id:
            self.logger.log_warning("No global session_id found in config")
        
        # Use state's memory managers
        self.ram_manager = state.ram_manager
        self.gpu_manager = state.gpu_manager
        
        # Initialize memory manager
        self.memory_manager = GenerationMemoryManager(
            config_manager=self._config_manager,
            logger=self.logger,
            ram_manager=self.ram_manager,
            gpu_manager=self.gpu_manager
        )
        
        # Initialize scaffold provider
        self.scaffold_provider = GenerationScaffoldProvider(
            scaffold_model=self.scaffolds[0],  # Use first scaffold model
            scaffold_tokenizer=self.scaffold_tokenizer,
            device=self.device,
            logger=self.logger,
            memory_manager=self.memory_manager
        )
        
        # Enhanced thread safety with multiple locks
        self._locks = {
            'state': Lock(),
            'generation': Lock(),
            'cache': Lock()
        }
        
        # Memory tracking
        self._tensor_metadata = defaultdict(dict)
        self._embedding_cache = {}
        self._max_cache_size = 1000
        
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
        self.base_batch_size = self._get_config_value("controls_config.base_batch_size", 1)
        self.generation_callbacks: Dict[str, List[Callable]] = {
            "pre_generate": [],
            "post_generate": []
        }

        # Validate and initialize curiosity state
        self._validate_curiosity_state()

        # BondCalculator integration
        self.bond_calculator = BondCalculator(config_manager, logger, state)

        self._last_good_memory_context = None  # Cache for fallback memory context

    def _with_lock(self, lock_name: str):
        """Context manager for thread-safe operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self._locks[lock_name]:
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def _handle_error(self, context: str, error: Exception, extra_context: dict = None) -> None:
        """Handle errors using the ErrorManager with state-driven error handling. Optionally accepts extra context for logging."""
        try:
            # Get current state metrics
            state_metrics = {
                'memory_usage': self.memory_manager.get_memory_usage(),
                'confidence': self.state.confidence if hasattr(self.state, 'confidence') else 0.5,
                'temperament_score': self.current_temperament_score,
                'lifecycle_stage': self.state.lifecycle_stage if hasattr(self.state, 'lifecycle_stage') else 'unknown'
            }
            # Merge in any extra context
            if extra_context:
                state_metrics.update(extra_context)
            # Handle error with state context
            self.error_manager.handle_generation_error(
                error=error,
                context=context,
                state=self.state,
                state_metrics=state_metrics
            )
            # Update state based on error
            self._update_state_after_error(error, context)
            # Re-raise critical errors for upstream handling
            if isinstance(error, (ValueError, RuntimeError, GenerationError, IndexError)):
                raise
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to handle error: {str(e)}",
                error_type="error_handling_error",
                stack_trace=traceback.format_exc()
            )

    @synchronized()
    def _handle_state_driven_error(self, error: Exception, context: str) -> None:
        """Handle errors with state-driven recovery strategies."""
        try:
            # Get current state metrics
            state_metrics = {
                'memory_usage': self.memory_manager.get_memory_usage(),
                'confidence': self.state.confidence if hasattr(self.state, 'confidence') else 0.5,
                'temperament_score': self.current_temperament_score,
                'lifecycle_stage': self.state.lifecycle_stage if hasattr(self.state, 'lifecycle_stage') else 'unknown'
            }
            
            # Handle error with state context
            self.error_manager.handle_generation_error(
                error=error,
                context=context,
                state=self.state,
                state_metrics=state_metrics
            )
            
            # Apply state-driven recovery strategies
            self._apply_state_driven_recovery(error, context, state_metrics)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to handle state-driven error: {str(e)}",
                error_type="state_driven_error_handling_error",
                stack_trace=traceback.format_exc()
            )

    def _apply_state_driven_recovery(self, error: Exception, context: str, state_metrics: Dict[str, Any]) -> None:
        """Enhanced: Apply state-driven recovery strategies based on error and current state metrics."""
        try:
            recovery_actions = []
            # Memory management
            if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
                before = self.memory_manager.get_memory_usage() if hasattr(self.memory_manager, 'get_memory_usage') else None
                self.memory_manager.optimize_memory_usage()
                after = self.memory_manager.get_memory_usage() if hasattr(self.memory_manager, 'get_memory_usage') else None
                self._clear_scaffold_cache()
                recovery_actions.append({
                    'action': 'optimize_memory',
                    'before': before,
                    'after': after
                })
            # Batch size adjustment based on confidence
            if state_metrics and state_metrics.get('confidence', 1.0) < 0.3:
                old_batch_size = getattr(self, 'base_batch_size', None)
                self.base_batch_size = max(1, self.base_batch_size // 2)
                recovery_actions.append({
                    'action': 'adjust_batch_size',
                    'old_batch_size': old_batch_size,
                    'new_batch_size': self.base_batch_size
                })
            # Temperament adjustment
            if hasattr(self.state, 'temperament_score'):
                old_temp = self.state.temperament_score
                self.state.temperament_score = max(0.1, self.state.temperament_score - 0.05)
                recovery_actions.append({
                    'action': 'adjust_temperament',
                    'old_temperament': old_temp,
                    'new_temperament': self.state.temperament_score
                })
            # Lifecycle stage update
            if hasattr(self.state, 'lifecycle_stage'):
                old_stage = self.state.lifecycle_stage
                if state_metrics and state_metrics.get('lifecycle_stage') == 'exploration':
                    self.state.lifecycle_stage = 'consolidation'
                    recovery_actions.append({
                        'action': 'update_lifecycle_stage',
                        'old_stage': old_stage,
                        'new_stage': self.state.lifecycle_stage
                    })
            self.logger.record_event(
                event_type="state_driven_recovery",
                message=f"Applied state-driven recovery for {context} error",
                level="info",
                additional_info={
                    'error_type': type(error).__name__,
                    'recovery_actions': recovery_actions,
                    'state_metrics': state_metrics
                }
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to apply state-driven recovery: {str(e)}",
                error_type="state_driven_recovery_error",
                stack_trace=traceback.format_exc()
            )

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

    def _initialize_config(self) -> None:
        """Initialize and validate configuration parameters."""
        try:
            # Validate required configuration sections
            required_sections = ["controls_config", "training_config"]
            for section in required_sections:
                if not self._config_manager.validate_section(section):
                    raise ValueError(f"Missing required configuration section: {section}")

            # Validate specific configuration values
            self._validate_config_values()

        except Exception as e:
            self._handle_error("config_initialization", e)
            raise

    def _validate_config_values(self) -> None:
        """Validate configuration values."""
        try:
            # Memory thresholds
            memory_threshold = self._get_config_value("controls_config.memory_threshold", 0.85)
            if not 0.0 <= memory_threshold <= 1.0:
                raise ValueError(f"Invalid memory_threshold: {memory_threshold}")
                
            # Generation parameters
            max_retries = self._get_config_value("controls_config.max_generation_retries", 3)
            if not isinstance(max_retries, int) or max_retries < 1:
                raise ValueError(f"Invalid max_generation_retries: {max_retries}")
                
            base_temperature = self._get_config_value("controls_config.base_temperature", 0.7)
            if not 0.0 <= base_temperature <= 2.0:
                raise ValueError(f"Invalid base_temperature: {base_temperature}")
                
            # Validate other critical parameters
            self._validate_memory_config()
            self._validate_scaffold_config()
            self._validate_generation_config()
            
        except Exception as e:
            self._handle_error("config_validation", e)
            raise

    def _validate_memory_config(self) -> None:
        """Validate memory-related configuration."""
        batch_size = self._get_config_value("controls_config.base_batch_size", 1)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"Invalid base_batch_size: {batch_size}")
            
        max_cache_size = self._get_config_value("controls_config.max_cache_size", 1000)
        if not isinstance(max_cache_size, int) or max_cache_size < 1:
            raise ValueError(f"Invalid max_cache_size: {max_cache_size}")

    def _validate_scaffold_config(self) -> None:
        """Validate scaffold-related configuration."""
        if not isinstance(self.scaffolds, list) or not self.scaffolds:
            raise ValueError("No scaffold models available")
            
        if not isinstance(self.scaffold_tokenizer, AutoTokenizer):
            raise ValueError("Invalid scaffold tokenizer")

    def _validate_generation_config(self) -> None:
        """Validate generation-related configuration."""
        if not isinstance(self.base_model, AutoModelForCausalLM):
            raise ValueError("Invalid base model")
            
        if not isinstance(self.base_tokenizer, AutoTokenizer):
            raise ValueError("Invalid base tokenizer")

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get a configuration value with validation."""
        try:
            return self._config_manager.get(key, default)
        except Exception as e:
            self._handle_error(f"config_access", e)
            return default

    def _update_config(self, key: str, value: Any) -> bool:
        """Update a configuration value with validation."""
        try:
            return self._config_manager.update(key, value)
        except Exception as e:
            self._handle_error(f"config_update", e)
            return False

    def _load_config_sections(self) -> None:
        """Load configuration sections from config manager."""
        self.controls_config = self._config_manager.get_section("controls_config")
        self.training_config = self._config_manager.get_section("training_config")
        
    def _log_initialization(self) -> None:
        """Log GenerationManager initialization with config values."""
        self.logger.log_event(
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
                "training_config": {k: self.training_config.get(k) for k in [
                    "max_seq_length", "batch_size"
                ]}
            }
        )

    def register_callback(self, stage: str, callback: Callable) -> None:
        """Register a callback for generation stages."""
        if stage in self.generation_callbacks:
            self.generation_callbacks[stage].append(callback)
        else:
            self._handle_error(f"register_callback", Exception(f"Invalid callback stage: {stage}"))

    def _log_memory_health(self) -> None:
        """Log detailed memory health status with adaptive thresholds."""
        try:
            # Get current memory stats
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Calculate memory pressure
            ram_pressure = ram_health["usage_percentage"] / 100.0
            gpu_pressure = gpu_health["usage_percentage"] / 100.0
            
            # Log memory health details
            self.logger.info(
                f"Memory Health Status:\n"
                f"  RAM Usage: {ram_health['used']:.2f}GB / {ram_health['total']:.2f}GB "
                f"({ram_health['usage_percentage']:.1f}%)\n"
                f"  GPU Usage: {gpu_health['used']:.2f}GB / {gpu_health['total']:.2f}GB "
                f"({gpu_health['usage_percentage']:.1f}%)\n"
                f"  Memory Pressure: RAM={ram_pressure:.2f}, GPU={gpu_pressure:.2f}\n"
                f"  Adaptive Threshold: {self.memory_manager.memory_threshold:.2f}"
            )
            
        except Exception as e:
            self._handle_error("memory_logging", e)

    def check_memory_health(self) -> bool:
        """Check memory health with adaptive thresholds and graceful degradation."""
        try:
            # Update memory threshold adaptively
            self.memory_manager.memory_threshold = self._calculate_adaptive_threshold()
            
            # Check both RAM and GPU health
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Calculate overall health score
            ram_score = 1.0 - (ram_health["usage_percentage"] / 100.0)
            gpu_score = 1.0 - (gpu_health["usage_percentage"] / 100.0)
            overall_score = (ram_score + gpu_score) / 2.0
            
            # If memory pressure is high, try graceful degradation first
            if overall_score < self.memory_manager.memory_threshold:
                # Log memory pressure
                self.logger.record_event(
                    event_type="memory_pressure",
                    message="High memory pressure detected, applying graceful degradation",
                    level="warning",
                    additional_info={
                        "ram_score": ram_score,
                        "gpu_score": gpu_score,
                        "overall_score": overall_score,
                        "threshold": self.memory_manager.memory_threshold
                    }
                )
                
                # Try graceful degradation steps
                self._apply_memory_degradation_steps()
                
                # Continue generation with reduced resources
                return True
                
            return True
            
        except Exception as e:
            self._handle_error("memory_health_check", e)
            # Continue with generation even if health check fails
            return True

    def _apply_memory_degradation_steps(self) -> None:
        """Apply graceful degradation steps to reduce memory pressure."""
        try:
            # 1. Clear caches first
            self._clear_scaffold_cache()
            torch.cuda.empty_cache()
            
            # 2. Reduce batch size if possible
            if hasattr(self, 'base_batch_size'):
                self.base_batch_size = max(1, self.base_batch_size // 2)
            
            # 3. Move old tensors to CPU
            if hasattr(self.memory_manager, '_offload_old_tensors'):
                self.memory_manager._offload_old_tensors()
            
            # 4. Trigger memory cleanup in background
            threading.Thread(
                target=self.memory_manager.manage_memory,
                name="memory_cleanup",
                daemon=True
            ).start()
            
        except Exception as e:
            self._handle_error("memory_degradation", e)
            # Continue even if degradation fails

    def _handle_error_prompt(self, error_msg: str) -> str:
        """Generate a response to a system error."""
        request_time = time.time()
        try:
            temp_history = self.state.history
            self.state.history = ConversationHistory(
                maxlen=self.controls_config.get("conversation_history_maxlen", 10)
            )
            response = self.generate(
                f"System error detected: {error_msg} What happened?",
                max_new_tokens=self.controls_config.get("max_new_tokens", 60),
                temperature=self.controls_config.get("base_temperature", 0.7) + 0.2,
                top_k=self.controls_config.get("top_k", 50),
                do_sample=True
            )
            self.state.history = temp_history

            # Log the internal error reflection
            capture_scribe_event(
                origin="sovl_generation",
                event_type="internal_error_reflection",
                event_data={
                    "triggering_error_message": error_msg,
                    "prompt_used": f"System error detected: {error_msg} What happened?",
                    "generated_response": response
                },
                source_metadata={
                    "generation_config": {
                        "max_new_tokens": self.controls_config.get("max_new_tokens", 60),
                        "temperature": self.controls_config.get("base_temperature", 0.7) + 0.2,
                        "top_k": self.controls_config.get("top_k", 50),
                        "do_sample": True
                    },
                    "model_name": getattr(self.base_model.config, "_name_or_path", "unknown"),
                    "device": str(self.device),
                    "internal_call": True
                },
                timestamp=datetime.fromtimestamp(request_time)
            )
            
            return response
        except Exception as e:
            self._handle_error("handle_error_prompt", e)
            # Log the failure to handle the error prompt
            capture_scribe_event(
                origin="sovl_generation",
                event_type="internal_error_reflection_failed",
                event_data={
                    "triggering_error_message": error_msg,
                    "error_message": str(e),
                    "error_type": type(e).__name__
                },
                source_metadata={
                    "internal_call": True
                },
                timestamp=datetime.now()
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
            self._handle_error("has_repetition", e)
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
            self._handle_error("tokenize_and_map", e)
            raise

    def get_scaffold_hidden_states(self, scaffold_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get hidden states from scaffold model."""
        return self.scaffold_provider.get_scaffold_hidden_states(scaffold_inputs)

    @contextlib.contextmanager
    def _scaffold_context(self, scaffold_hidden_states: torch.Tensor):
        """Manage scaffold context with memory monitoring."""
        with self.scaffold_provider.scaffold_context(scaffold_hidden_states) as ctx:
            yield ctx

    def _clear_scaffold_cache(self) -> None:
        """Clear scaffold-related caches with memory optimization."""
        self.scaffold_provider.clear_scaffold_cache()

    def get_num_scaffolds(self) -> int:
        """Get the number of available scaffold models."""
        return len(self.scaffolds)

    def _prepare_generation_params(self, max_new_tokens: int, scaffold_weight: Optional[float], **kwargs) -> Dict[str, Any]:
        """Prepare and validate generation parameters."""
        return self.scaffold_provider.prepare_generation_params(max_new_tokens, scaffold_weight, **kwargs)

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
            self._handle_error("update_token_map_memory", e)

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
            self._handle_error("prepare_for_training", e)
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
            self._handle_error("compute_dynamic_factor", e)
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
                self._handle_error("prepare_dream_memory", e)

        return memory_tensors, dream_memory_info

    def _handle_repetition(self, seq: torch.Tensor, seq_ids: List[int], outputs: Any) -> List[int]:
        """Handle detected repetition in generated sequence."""
        if self.controls_config.get("enable_repetition_check", True) and self.has_repetition(seq):
            original_text = self.base_tokenizer.decode(seq_ids, skip_special_tokens=True)
            for j in range(len(seq_ids) - 6):
                if all(seq_ids[j + k] == seq_ids[j + k + 3] for k in range(3)):
                    seq_ids = seq_ids[:j + 3]
                    break
            self.logger.log_event(
                "repetition_detected",
                {
                    "original_text": original_text,
                    "truncated_at": j + 3
                }
            )
        return seq_ids

    def _update_curiosity(self, text: str, confidence: float) -> None:
        """[DEPRECATED] Use self.primer.update_curiosity_state instead."""
        self.primer.update_curiosity_state(text, confidence)

    @synchronized()
    def calculate_confidence_score(self, logits: torch.Tensor, generated_ids: torch.Tensor, state, error_manager, context, curiosity_manager=None) -> float:
        """Calculate confidence score using the centralized confidence module."""
        return calculate_confidence_score(logits, generated_ids, state, error_manager, context, curiosity_manager)

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

    def state_managed_operation(self, error_context: str):
        """Unified decorator for error handling and state management."""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    # Get state metrics before operation
                    state_metrics = {
                        'memory_usage': self.memory_manager.get_memory_usage(),
                        'confidence': getattr(self.state, 'confidence', 0.5),
                        'temperament_score': self.current_temperament_score,
                        'lifecycle_stage': getattr(self.state, 'lifecycle_stage', 'unknown')
                    }
                    
                    # Execute operation
                    result = func(self, *args, **kwargs)
                    
                    # Update state after successful operation
                    self._update_state_after_operation(error_context, state_metrics, result)
                    
                    return result
                    
                except Exception as e:
                    self.error_manager.handle_generation_error(
                        error=e,
                        context=error_context,
                        state=self.state,
                        state_metrics=state_metrics
                    )
                    # Update state based on error
                    self._update_state_after_error(e, error_context)
                    raise
            return wrapper
        return decorator

    @state_managed_operation("generate_text")
    @GenerationManager._with_lock('generation')
    def generate_text(self, prompt: str, num_return_sequences: int = 1, user_id: str = "default", metadata_entries: list = None, **kwargs) -> List[str]:
        """Generate text with state-driven error handling, recovery, scribe logging, and always-on memory integration."""
        request_time = time.time()
        traits = None
        generated_texts = None
        error = None
        try:
            # --- Always-on Memory: Retrieve context ---
            memory_context = None
            context_retrieved = False
            if self.dialogue_context_manager:
                backoff = 0.1
                for attempt in range(3):
                    try:
                        short_ctx = self.dialogue_context_manager.get_short_term_context()
                        long_ctx = self.dialogue_context_manager.get_long_term_context(user_id=user_id)
                        memory_context = self._compose_memory_context(short_ctx, long_ctx)
                        context_retrieved = True
                        self._last_good_memory_context = memory_context
                        break
                    except Exception as e:
                        self.logger.log_warning(
                            f"Attempt {attempt+1}: Failed to retrieve memory context: {str(e)}",
                            error_type="memory_context_retrieval_error"
                        )
                        if attempt < 2:
                            import time as _time
                            _time.sleep(backoff)
                            backoff *= 2
                if not context_retrieved:
                    if self._last_good_memory_context:
                        memory_context = self._last_good_memory_context
                        self.logger.log_warning(
                            "Using cached memory context due to repeated retrieval failures.",
                            error_type="memory_context_fallback"
                        )
                    else:
                        self.logger.log_warning(
                            "No memory context available after retries and no cache present.",
                            error_type="memory_context_missing"
                        )
            else:
                self.logger.log_warning(
                    "No dialogue context manager available for memory retrieval",
                    error_type="missing_dialogue_context"
                )

            # --- Retrieve all trait values from GenerationPrimer ---
            traits = self.primer.prepare_for_generation(prompt, user_id=user_id, metadata_entries=metadata_entries, **kwargs)

            # --- Validate and sanitize trait values ---
            def validate_trait(name, value, default, valid_range=None):
                if value is None:
                    self.logger.log_warning(f"Trait '{name}' is None, using default value {default}")
                    return default
                if valid_range and not (valid_range[0] <= value <= valid_range[1]):
                    self.logger.log_warning(f"Trait '{name}' value {value} out of range {valid_range}, using default {default}")
                    return default
                return value

            temperament = validate_trait("temperament", traits.get("temperament"), 0.5, (0.0, 1.0))
            curiosity = validate_trait("curiosity", traits.get("curiosity"), 0.0, (0.0, 1.0))
            bond_score = traits.get("bond")
            if bond_score is not None and not isinstance(bond_score, (float, int)):
                self.logger.log_warning(f"Trait 'bond' is not a number: {bond_score}, ignoring bond context")
                bond_score = None
            bond_context = None
            if bond_score is not None:
                bond_context = f"Bond score: {bond_score:.2f}"

            # --- Compose prompt with memory and bond context ---
            composite_prompt = prompt
            if memory_context:
                composite_prompt = f"{memory_context}\n\n{composite_prompt}"
            if bond_context:
                composite_prompt = f"[{bond_context}]\n\n{composite_prompt}"

            # --- Prepare input batch ---
            inputs = self.base_tokenizer(
                composite_prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self._get_config_value("controls_config.max_seq_length", 512)
            )
            model_device = next(self.base_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            # --- Add generation parameters and trait-driven adjustments ---
            gen_kwargs = {"num_return_sequences": num_return_sequences}
            gen_kwargs.update(kwargs)
            # Example: adjust temperature using traits
            base_temp = gen_kwargs.get("temperature", 1.0)
            gen_kwargs["temperature"] = self.primer.adjust_parameter(base_temp, "temperature", temperament=temperament, curiosity=curiosity)

            # --- Generate text ---
            output_sequences = self.base_model.generate(**inputs, **gen_kwargs)
            generated_texts = [self.base_tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]

            # --- Log bond score/context for this generation if present ---
            if bond_score is not None:
                self.logger.record_event(
                    event_type="bond_modulation_applied",
                    message="Bond context applied to generation",
                    level="info",
                    additional_info={
                        "bond_score": bond_score,
                        "bond_context": bond_context,
                        "user_id": user_id,
                        "metadata_entries": metadata_entries
                    }
                )

            # Prepare generation result data for logging
            generation_result = {
                "generated_texts": generated_texts,
                "generation_config_used": self._get_generation_config(),
                "processing_time_ms": (time.time() - request_time) * 1000
            }

            # Assemble capture data
            event_data, source_metadata = ScribeAssembler.assemble_scribe_data(
                manager=self,
                prompt=prompt,
                initial_kwargs=kwargs,
                generation_result=generation_result,
                request_time=request_time,
                session_id=self.session_id,
            )

            # Log the event
            capture_scribe_event(
                origin="sovl_generation",
                event_type="base_generation",
                event_data=event_data,
                source_metadata=source_metadata,
                session_id=self.session_id
            )

            return generated_texts

        except (ValueError, RuntimeError, GenerationError, IndexError) as e:
            error = e
            # Log and propagate critical errors
            self._handle_error("generate_text", e)
            raise
        except Exception as e:
            error = e
            # Log generation error
            capture_scribe_event(
                origin="sovl_generation",
                event_type="generation_error",
                event_data={
                    "prompt": prompt,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "kwargs": kwargs
                },
                source_metadata={
                    "session_id": self.session_id,
                    "request_timestamp_unix": request_time,
                    "model_name": getattr(self.base_model.config, "_name_or_path", "unknown"),
                    "device": str(self.device)
                },
                session_id=self.session_id,
                timestamp=datetime.fromtimestamp(request_time)
            )
            # Handle error through existing mechanism
            self._handle_error("generate_text", e)
            # For broad compatibility, return a fallback error message
            return ["An error occurred during text generation"]
        finally:
            # --- Always sync all traits to state, even on error ---
            try:
                self.primer.update_state_after_operation(
                    context="generate_text",
                    result={
                        "generated_texts": generated_texts,
                        "traits": traits,
                        "error": str(error) if error else None
                    }
                )
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Failed to sync traits to state after generation (finally block): {str(e)}",
                    error_type="trait_state_sync_error",
                    stack_trace=traceback.format_exc()
                )

    def set_system_context(self, system_context):
        """Bind the system context for always-on memory integration."""
        self.system_context = system_context

    def _compose_memory_context(self, short_ctx, long_ctx):
        """Compose a string from short-term and long-term memory context."""
        context_lines = []
        if short_ctx:
            context_lines.append("Short-Term Memory:")
            for msg in short_ctx:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_lines.append(f"{role.capitalize()}: {content}")
        if long_ctx:
            context_lines.append("Long-Term Memory:")
            for msg in long_ctx:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(context_lines) if context_lines else None

    def _prepare_generation_batch(self, prompt: str, num_return_sequences: int = 1) -> Dict[str, Any]:
        """Prepare input batch for text generation."""
        try:
            # Tokenize input
            inputs = self.base_tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self._get_config_value("controls_config.max_seq_length", 512)
            )
            
            # Validate device matches base model
            model_device = next(self.base_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Add generation parameters
            inputs['num_return_sequences'] = num_return_sequences
            
            return inputs
            
        except Exception as e:
            self._handle_error("prepare_generation_batch", e)
            raise ValueError(f"Failed to prepare generation batch: {str(e)}")

    def _get_generation_config(self) -> Dict[str, Any]:
        """Get the configuration for text generation."""
        try:
            config = {
                'max_new_tokens': self._get_config_value("controls_config.max_new_tokens", 100),
                'do_sample': True,
                'temperature': self._get_config_value("controls_config.base_temperature", 0.7),
                'top_k': self._get_config_value("controls_config.top_k", 50),
                'pad_token_id': self.base_tokenizer.pad_token_id,
                'eos_token_id': self.base_tokenizer.eos_token_id
            }
            
            # Apply any state-based adjustments
            if hasattr(self, 'state') and hasattr(self.state, 'temperament_score'):
                config['temperature'] = self.primer.adjust_parameter(
                    config['temperature'],
                    'temperature',
                    self.curiosity_manager.get_pressure() if self.curiosity_manager else None
                )
                
            # Ensure any tensor values are on the correct device
            model_device = next(self.base_model.parameters()).device
            config = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                for k, v in config.items()
            }
                
            return config
            
        except Exception as e:
            self._handle_error("get_generation_config", e)
            # Return safe defaults if error occurs
            return {
                'max_new_tokens': 100,
                'do_sample': True,
                'temperature': 0.7,
                'top_k': 50,
                'pad_token_id': self.base_tokenizer.pad_token_id,
                'eos_token_id': self.base_tokenizer.eos_token_id
            }

    @state_managed_operation("generate_with_state_context")
    def _generate_with_state_context(self, batch: Dict[str, Any]) -> List[str]:
        """Generate text with state context and error handling."""
        # Validate state consistency before generation
        self._validate_state_consistency()
        
        generation_config = self._get_generation_config()
        outputs = self.base_model.generate(**batch, **generation_config)
        generated_texts = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if not generated_texts or not all(isinstance(text, str) for text in generated_texts):
            raise ValueError("Invalid generation output")
        
        return generated_texts

    @state_managed_operation("backchannel_scaffold_prompt")
    @GenerationManager._with_lock('generation')
    def backchannel_scaffold_prompt(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        scaffold_index: int = 0,
        return_logits: bool = False,
        return_hidden_states: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Backchannel communication method to directly prompt the scaffold model."""
        request_time = time.time()

        try:
            # Proactive memory management: check and clear before generation
            if not self.check_memory_health():
                self.memory_manager.manage_memory()
                # After managing, check again and abort if still unhealthy
                if not self.check_memory_health():
                    self.logger.record_event(
                        event_type="memory_exhaustion_abort",
                        message="Aborting scaffold generation due to persistent memory exhaustion.",
                        level="error",
                        additional_info={
                            "scaffold_index": scaffold_index,
                            "prompt_length": len(prompt),
                            "generation_params": kwargs,
                            "memory_usage": self.memory_manager.get_memory_usage()
                        }
                    )
                    raise RuntimeError("Memory exhausted: unable to safely proceed with scaffold generation.")

            # Memory optimization: Check memory health before proceeding
            if not self.check_memory_health():
                self.memory_manager.manage_memory()
                # After managing, check again and abort if still unhealthy
                if not self.check_memory_health():
                    self.logger.record_event(
                        event_type="memory_exhaustion_abort",
                        message="Aborting scaffold generation due to persistent memory exhaustion.",
                        level="error",
                        additional_info={
                            "scaffold_index": scaffold_index,
                            "prompt_length": len(prompt),
                            "generation_params": kwargs,
                            "memory_usage": self.memory_manager.get_memory_usage()
                        }
                    )
                    raise RuntimeError("Memory exhausted: unable to safely proceed with scaffold generation.")
            
            # Validate scaffold index and model early
            if not (0 <= scaffold_index < len(self.scaffolds)):
                raise IndexError(f"Invalid scaffold_index {scaffold_index}. Only {len(self.scaffolds)} scaffolds available")
            
            scaffold_model = self.scaffolds[scaffold_index]
            if not scaffold_model:
                raise RuntimeError("Scaffold model not properly initialized")
            
            # Optimize generation config with smart defaults
            max_length = kwargs.get('max_length', min(512, len(prompt) + max_new_tokens))
            generation_config = {
                'output_scores': return_logits,
                'output_hidden_states': return_hidden_states,
                'return_dict_in_generate': return_logits or return_hidden_states,
                'max_new_tokens': max_new_tokens,
                'pad_token_id': self.scaffold_tokenizer.pad_token_id,
                'early_stopping': kwargs.get('early_stopping', True),
                'do_sample': kwargs.get('do_sample', True),
                'num_beams': kwargs.get('num_beams', 1),
                **kwargs
            }

            # Batch and device optimization
            inputs = self.scaffold_tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device efficiently with validation
            model_device = next(scaffold_model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
            
            # Generation with memory optimization
            with torch.no_grad(), self.memory_manager.track_memory("scaffold_generation"):
                outputs = scaffold_model.generate(
                    **inputs,
                    **generation_config
                )

            # Efficient result handling
            if return_logits or return_hidden_states:
                result = {}
                # Extract text efficiently
                generated_ids = outputs.sequences[0] if hasattr(outputs, 'sequences') else outputs[0]
                result['text'] = self.scaffold_tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                # Only process requested outputs
                if return_logits and hasattr(outputs, 'scores'):
                    with torch.cuda.amp.autocast(enabled=True):
                        result['logits'] = torch.stack(outputs.scores, dim=0)
                
                if return_hidden_states and hasattr(outputs, 'hidden_states'):
                    with torch.cuda.amp.autocast(enabled=True):
                        result['hidden_states'] = outputs.hidden_states
                
                # Add metadata with performance metrics
                result['metadata'] = {
                    'scaffold_index': scaffold_index,
                    'model_name': self.scaffold_tokenizer.name_or_path,
                    'generation_params': generation_config,
                    'input_length': len(inputs['input_ids'][0]),
                    'output_length': len(generated_ids),
                    'memory_usage': self.memory_manager.get_memory_usage(),
                    'generation_time': time.time()  # For tracking generation duration
                }
                
                # Log detailed result
                capture_scribe_event(
                    origin="sovl_generation",
                    event_type="backchannel_scaffold_generated",
                    event_data={
                        "prompt": prompt,
                        "scaffold_index": scaffold_index,
                        "max_new_tokens": max_new_tokens,
                        "generated_text": result['text'],
                        "return_logits": return_logits,
                        "return_hidden_states": return_hidden_states,
                        "generation_params": generation_config
                    },
                    source_metadata={
                        **result['metadata'],
                        "session_id": self.session_id,

                        "request_timestamp_unix": request_time
                    },
                    session_id=self.session_id,
                    timestamp=datetime.fromtimestamp(request_time)
                )
                
                return result
            else:
                # Fast path for text-only return
                generated_text = self.scaffold_tokenizer.decode(
                    outputs[0] if isinstance(outputs, torch.Tensor) else outputs.sequences[0],
                    skip_special_tokens=True
                )
                
                # Log text-only result
                capture_scribe_event(
                    origin="sovl_generation",
                    event_type="backchannel_scaffold_generated",
                    event_data={
                        "prompt": prompt,
                        "scaffold_index": scaffold_index,
                        "max_new_tokens": max_new_tokens,
                        "generated_text": generated_text,
                        "generation_params": generation_config
                    },
                    source_metadata={
                        "scaffold_index": scaffold_index,
                        "model_name": self.scaffold_tokenizer.name_or_path,
                        "input_length": len(inputs['input_ids'][0]),
                        "output_length": len(outputs[0] if isinstance(outputs, torch.Tensor) else outputs.sequences[0]),
                        "memory_usage": self.memory_manager.get_memory_usage(),
                        "generation_time": time.time() - request_time,
                        "session_id": self.session_id,
                        "request_timestamp_unix": request_time
                    },
                    session_id=self.session_id,
                    timestamp=datetime.fromtimestamp(request_time)
                )
                
                return generated_text
            
        except (ValueError, RuntimeError, GenerationError, IndexError) as e:
            # Log and propagate critical errors
            self._handle_error("backchannel_scaffold_prompt", e, {
                'scaffold_index': scaffold_index,
                'prompt_length': len(prompt),
                'generation_params': kwargs,
                'memory_usage': self.memory_manager.get_memory_usage()
            })
            raise
        except Exception as e:
            # Log error
            capture_scribe_event(
                origin="sovl_generation",
                event_type="backchannel_scaffold_error",
                event_data={
                    "prompt": prompt,
                    "scaffold_index": scaffold_index,
                    "max_new_tokens": max_new_tokens,
                    "return_logits": return_logits,
                    "return_hidden_states": return_hidden_states,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "kwargs": kwargs,
                    "traceback": traceback.format_exc()
                },
                source_metadata={
                    "scaffold_index": scaffold_index,
                    "model_name": getattr(self.scaffold_tokenizer, "name_or_path", "unknown"),
                    "session_id": self.session_id,
                    "request_timestamp_unix": request_time,
                    "memory_usage": self.memory_manager.get_memory_usage()
                },
                session_id=self.session_id,
                timestamp=datetime.fromtimestamp(request_time)
            )
            # Handle error through existing mechanism
            self._handle_error("backchannel_scaffold_prompt", e, {
                'scaffold_index': scaffold_index,
                'prompt_length': len(prompt),
                'generation_params': kwargs,
                'memory_usage': self.memory_manager.get_memory_usage()
            })
            # For broad compatibility, raise a GenerationError for upstream detection
            raise GenerationError(f"Failed in backchannel_scaffold_prompt: {e}") from e
        finally:
            # Ensure memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive memory threshold based on system load and model size."""
        try:
            base_threshold = self.memory_manager.memory_threshold
            model_size = sum(p.numel() * p.element_size() for p in self.base_model.parameters())
            total_memory = self.gpu_manager.get_gpu_usage()["total_memory"]
            system_load = self.ram_manager.check_memory_health()["usage_percentage"]
            
            # Calculate adaptive threshold
            adaptive_threshold = base_threshold * (1 - system_load) * (1 - model_size/total_memory)
            return max(0.7, min(0.95, adaptive_threshold))
        except Exception as e:
            self._handle_error("adaptive_threshold_calculation", e)
            return self.memory_manager.memory_threshold

    def _optimize_batch_size(self, current_batch_size: int) -> int:
        """Optimize batch size based on memory availability."""
        try:
            memory_usage = self.gpu_manager.get_gpu_usage()["usage_percentage"]
            if memory_usage > 0.9:
                return max(1, current_batch_size // 2)
            elif memory_usage < 0.5:
                return min(self.max_batch_size, current_batch_size * 2)
            return current_batch_size
        except Exception as e:
            self._handle_error("batch_size_optimization", e)
            return current_batch_size

    def _validate_generation_params(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> None:
        """Validate generation parameters."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
            
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be a positive integer, got {max_new_tokens}")
            
        # Validate temperature if provided
        if "temperature" in kwargs:
            temp = kwargs["temperature"]
            if not isinstance(temp, (int, float)) or temp <= 0:
                raise ValueError(f"temperature must be a positive number, got {temp}")
                
        # Validate other common parameters
        if "top_k" in kwargs:
            top_k = kwargs["top_k"]
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"top_k must be a positive integer, got {top_k}")
                
        if "top_p" in kwargs:
            top_p = kwargs["top_p"]
            if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
                raise ValueError(f"top_p must be between 0 and 1, got {top_p}")

    def _validate_state_consistency(self) -> None:
        """Ensure state consistency by setting defaults if values are missing or invalid."""
        try:
            # Handle temperament_score
            if not hasattr(self.state, 'temperament_score'):
                self.state.temperament_score = 0.5  # Set default
                self.logger.log_event(
                    "state_consistency",
                    {"message": "Set default temperament_score", "value": 0.5}
                )
            elif not 0 <= self.state.temperament_score <= 1:
                self.state.temperament_score = max(0, min(1, self.state.temperament_score))
                self.logger.log_event(
                    "state_consistency",
                    {"message": "Clamped temperament_score", "value": self.state.temperament_score}
                )
            
            # Handle confidence
            if not hasattr(self.state, 'confidence'):
                self.state.confidence = 0.5  # Set default
                self.logger.log_event(
                    "state_consistency",
                    {"message": "Set default confidence", "value": 0.5}
                )
            elif not 0 <= self.state.confidence <= 1:
                self.state.confidence = max(0, min(1, self.state.confidence))
                self.logger.log_event(
                    "state_consistency",
                    {"message": "Clamped confidence", "value": self.state.confidence}
                )
                
        except Exception as e:
            self._handle_error("state_consistency", e)
            # Set safe defaults if error occurs
            self.state.temperament_score = 0.5
            self.state.confidence = 0.5

    def _handle_internal_prompt(self, prompt: str = " ") -> str:
        """Generate a response based on a minimal or provided prompt."""
        request_time = time.time()
        try:
            # Save current history temporarily
            temp_history = self.state.history
            self.state.history = ConversationHistory(
                maxlen=self.controls_config.get("conversation_history_maxlen", 10)
            )
            
            # Generate with higher temperature for more creative/expressive output
            response = self.generate(
                prompt,  # Use the passed or default prompt
                num_return_sequences=1,
                temperature=self.controls_config.get("base_temperature", 0.7) + 0.3,
                top_k=self.controls_config.get("top_k", 50),
                do_sample=True
            )[0]
            
            # Restore history
            self.state.history = temp_history

            # Log the internal thought generation
            capture_scribe_event(
                origin="sovl_generation",
                event_type="internal_thought_generated",
                event_data={
                    "prompt_used": prompt,
                    "generated_response": response
                },
                source_metadata={
                    "generation_config": {
                        "temperature": self.controls_config.get("base_temperature", 0.7) + 0.3,
                        "top_k": self.controls_config.get("top_k", 50),
                        "do_sample": True
                    },
                    "model_name": getattr(self.base_model.config, "_name_or_path", "unknown"),
                    "device": str(self.device),
                    "internal_call": True
                },
                timestamp=datetime.fromtimestamp(request_time)
            )
            
            return response
        except Exception as e:
            self._handle_error("handle_internal_prompt", e)
            # Log the failure to generate internal thought
            capture_scribe_event(
                origin="sovl_generation",
                event_type="internal_thought_generation_failed",
                event_data={
                    "prompt_used": prompt,
                    "error_message": str(e),
                    "error_type": type(e).__name__
                },
                source_metadata={
                    "internal_call": True
                },
                timestamp=datetime.now()
            )
            return "..."

class ScribeAssembler:
    """Assembles the data required for logging generation events."""

    @staticmethod
    def assemble_scribe_data(
        manager: 'GenerationManager', # Pass the manager instance for context
        prompt: str,
        initial_kwargs: Dict[str, Any],
        generation_result: Dict[str, Any],
        request_time: float,
        session_id: Optional[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Assembles the event_data and source_metadata dictionaries for logging.

        Args:
            manager: The GenerationManager instance.
            prompt: The original input prompt.
            initial_kwargs: The original kwargs passed to generate_text.
            generation_result: The dictionary returned by _generate_with_state_context.
            request_time: The timestamp when the generation request was received.
            session_id: External session identifier.

        Returns:
            A tuple containing: (event_data, source_metadata)
        """

        # Retrieve calculated scores from generation_result
        generated_texts = generation_result.get("generated_texts", [])
        calculated_confidence = generation_result.get("calculated_confidence")
        calculated_novelty = generation_result.get("calculated_novelty")

        # Retrieve state-based scores/info from manager
        current_temperament = getattr(manager.state, "temperament_score", None)
        current_lifecycle_stage = manager.lifecycle_manager.get_lifecycle_stage() if hasattr(manager, 'lifecycle_manager') and manager.lifecycle_manager else None
        current_memory_mb = manager.memory_manager.get_memory_usage().get("total_mb") if hasattr(manager, 'memory_manager') and manager.memory_manager else None

        # --- Assemble event_data (Data directly related to the event's core action) ---
        event_data = {
            "prompt": prompt,
            "texts": generated_texts,
            "confidence_score": calculated_confidence,
            "num_return_sequences": initial_kwargs.get("num_return_sequences", 1),
        }

        # --- Assemble source_metadata (Contextual info about the event) ---
        source_metadata = {
            # External Context
            "session_id": session_id,
            # Request Info
            "request_timestamp_unix": request_time,
            "initial_kwargs": initial_kwargs,
            # Config Used
            "generation_config_used": generation_result.get("generation_config_used"),
            # System State Snapshot
            "model_name": getattr(manager.base_model.config, "_name_or_path", "unknown"),
            "device": str(manager.device),
            "temperament_score": current_temperament,
            "lifecycle_stage": current_lifecycle_stage,
            "novelty_score": calculated_novelty,
            "memory_usage_mb": current_memory_mb,
            # Performance
            "processing_time_ms": generation_result.get("processing_time_ms"),
        }

        return event_data, source_metadata
