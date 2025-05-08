import torch
import time
import threading
from collections import deque, defaultdict
from typing import Optional, Dict, Any, List, Union, Callable, Tuple, Set
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
from sovl_trainer import LifecycleManager, TrainingConfig
from sovl_queue import capture_scribe_event
from sovl_memory import GenerationMemoryManager
from sovl_manager import ModelManager
from sovl_primer import GenerationPrimer  
from sovl_resource import ResourceManager

class GenerationError(Exception):
    """Raised when text generation fails in a way that should halt upstream processing."""
    pass

class GenerationManager:
    """
    Manages text generation, scaffold integration, and memory handling for the SOVL system.

    cross_attention_injector: Must provide method 'inject(...)'
    scaffold_manager: Must provide methods 'map_sequence(...)' and 'update_token_map_memory(...)'
    dialogue_context_manager: Must provide 'get_short_term_context(...)' and 'get_long_term_context(...)'
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        base_model: AutoModelForCausalLM,
        base_tokenizer: AutoTokenizer,
        state: SOVLState,
        logger: Logger,
        error_manager: ErrorManager,
        device: torch.device,
        generation_hooks: Dict[str, bool] = {},
        dialogue_context_manager: Optional[Any] = None,
        state_manager: Any = None,
        resource_manager: ResourceManager = None,  # New argument
        model_manager: Any = None  # New argument
    ):
        """Initialize GenerationManager with configuration and model components.
        Args:
            ...
            resource_manager: Optional ResourceManager for coordinated resource allocation.
            model_manager: Optional ModelManager for dynamic context length.
        """
        # Core components
        self._config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.device = device
        self.components = {}  # For compatibility with other modules
        # ResourceManager integration
        if resource_manager is None:
            self.resource_manager = ResourceManager(logger=self.logger)
        else:
            self.resource_manager = resource_manager
        self.components["resource_manager"] = self.resource_manager
        # ModelManager integration
        if model_manager is None:
            self.model_manager = ModelManager(logger=self.logger)
        else:
            self.model_manager = model_manager
        self.components["model_manager"] = self.model_manager
        # Resource acquisition for base_model
        try:
            # Acquire GPU memory for base_model
            model_size_mb = sum(p.numel() * p.element_size() for p in base_model.parameters()) // (1024 * 1024)
            if not self.resource_manager.acquire("gpu_memory", amount=model_size_mb):
                raise RuntimeError(f"Insufficient GPU memory for base model ({model_size_mb} MB)")
            self.base_model = base_model.to(device)
        except Exception as e:
            # Release all acquired resources on failure
            if hasattr(self, 'base_model'):
                self.resource_manager.release("gpu_memory", amount=model_size_mb)
            self.logger.log_error(
                error_msg=f"GenerationManager initialization failed: {str(e)}",
                error_type="generation_manager_init_error",
                stack_trace=traceback.format_exc()
            )
            raise
        self.base_tokenizer = base_tokenizer
        self.state = state
        self.dialogue_context_manager = dialogue_context_manager
        self.state_manager = state_manager
        
        # System context reference (will be set later)
        self._system_context = None

        # Generation hooks setup
        self.generation_hooks = generation_hooks or {}
        self.logger.record_event(
            event_type="generation_hooks_initialized",
            message=f"GenerationManager initialized with hooks: {self.generation_hooks}",
            level="info",
            component="GenerationManager"
        )

        # Lazy initialization flags
        self._initialized_primer = False
        self._initialized_memory_manager = False
        
        # Lazy-loaded components
        self.primer = None
        self.memory_manager = None
        
        # Get global session_id from config
        self.session_id = self._config_manager.get("runtime.session_id")
        if not self.session_id:
            self.logger.log_warning("No global session_id found in config")
        
        # Use memory manager abstraction for RAM/GPU managers
        self._initialize_memory_manager()  # Ensure memory_manager is initialized
        self.ram_manager = self.memory_manager.ram_manager
        self.gpu_manager = self.memory_manager.gpu_manager
        
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
        
        # Initialize configuration
        self._initialize_config()

        # Memory settings
        self.scaffold_unk_id = self._get_config_value("controls_config.scaffold_unk_id", base_tokenizer.unk_token_id)
        self.use_token_map_memory = self._get_config_value("controls_config.use_token_map_memory", True)
        self.dynamic_cross_attn_mode = self._get_config_value("controls_config.dynamic_cross_attn_mode", None)

        # Generation settings
        self.max_retries = self._get_config_value("controls_config.max_generation_retries", 3)
        self.base_batch_size = self._get_config_value("controls_config.base_batch_size", 1)
        self.generation_callbacks: Dict[str, List[Callable]] = {
            "pre_generate": [],
            "post_generate": []
        }

        self._last_good_memory_context = None  # Cache for fallback memory context
        
        # Log successful initialization
        self.logger.log_info("GenerationManager initialized successfully (with lazy component loading)")

        # Interface check for cross_attention_injector
        if self.cross_attention_injector is not None and not callable(getattr(self.cross_attention_injector, 'inject', None)):
            raise TypeError("cross_attention_injector must provide a callable 'inject(...)' method.")

        # Interface check for dialogue_context_manager
        if self.dialogue_context_manager is not None:
            if not callable(getattr(self.dialogue_context_manager, 'get_short_term_context', None)):
                raise TypeError("dialogue_context_manager must provide a callable 'get_short_term_context(...)' method.")
            if not callable(getattr(self.dialogue_context_manager, 'get_long_term_context', None)):
                raise TypeError("dialogue_context_manager must provide a callable 'get_long_term_context(...)' method.")
        
    def _initialize_memory_manager(self) -> None:
        """Lazily initialize the memory manager when needed."""
        if not self._initialized_memory_manager:
            self.logger.log_debug("Initializing memory manager")
            self.memory_manager = GenerationMemoryManager(
                config_manager=self._config_manager,
                logger=self.logger,
                ram_manager=self.ram_manager,
                gpu_manager=self.gpu_manager,
                resource_manager=self.resource_manager  # Pass resource manager if supported
            )
            self._initialized_memory_manager = True
            
    def _initialize_primer(self) -> None:
        """Lazily initialize the primer when needed."""
        if not self._initialized_primer:
            self.logger.log_debug("Initializing generation primer")
            # Ensure memory manager is initialized first
            self._initialize_memory_manager()
            self.primer = GenerationPrimer(
                config_manager=self._config_manager,
                logger=self.logger,
                state=self.state,
                error_manager=self.error_manager,
                device=self.device,
                generation_hooks=self.generation_hooks
            )
            self._initialized_primer = True

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
                'memory_usage': self.memory_manager.get_memory_usage() if hasattr(self, 'memory_manager') and self.memory_manager else {}
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

    def _update_state_after_error(self, error: Exception, context: str):
        """Update state after an error occurs."""
        # Will be implemented in future update
        pass
        
    def set_system_context(self, system_context):
        """Safely set the system context after initialization."""
        self._system_context = system_context
        self.logger.log_info("System context set in GenerationManager")
        
        # Update state_manager if it wasn't provided in the constructor
        if self.state_manager is None and hasattr(system_context, 'state_manager'):
            self.state_manager = system_context.state_manager

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
                    "memory_threshold", "max_generation_retries", "enable_repetition_check",
                    "conversation_history_maxlen", "memory_decay_rate",  
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
        """Lightweight error prompt handler without heavy model calls."""
        request_time = time.time()
        # Log the error
        self.logger.log_error(
            error_msg=f"Handling error prompt: {error_msg}",
            error_type="handle_error_prompt"
        )
        # Emit a minimal scribe event for auditing
        capture_scribe_event(
            origin="sovl_generation",
            event_type="internal_error_reflection",
            event_data={"triggering_error_message": error_msg},
            source_metadata={"internal_call": True},
            timestamp=datetime.fromtimestamp(request_time)
        )
        # Return a static fallback response
        return f"An internal error occurred: {error_msg}"

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

    def state_managed_operation(self, error_context: str):
        """Unified decorator for error handling and state management."""
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    # Get state metrics before operation
                    state_metrics = {
                        'memory_usage': self.memory_manager.get_memory_usage()
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

    def _estimate_optimal_batch_size(self):
        """Estimate the optimal batch size based on available memory and config."""
        min_bs = self._get_config_value("generation_config.min_batch_size", 1)
        max_bs = self._get_config_value("generation_config.max_batch_size", 8)
        default_bs = self._get_config_value("generation_config.default_batch_size", 1)
        mem_per_sample = self._get_config_value("generation_config.mem_per_sample_mb", 100)
        # Try to use GPU memory if available, else RAM
        available_mem = 0
        if hasattr(self, 'gpu_manager') and self.gpu_manager:
            try:
                available_mem = self.gpu_manager.get_available_memory()
            except Exception:
                available_mem = 0
        if not available_mem and hasattr(self, 'ram_manager') and self.ram_manager:
            try:
                available_mem = self.ram_manager.get_available_memory()
            except Exception:
                available_mem = 0
        if not available_mem:
            return default_bs
        for bs in range(max_bs, min_bs - 1, -1):
            if bs * mem_per_sample < available_mem:
                return bs
        return min_bs

    @state_managed_operation("generate_text")
    def generate_text(self, prompt: str, num_return_sequences: int = 1, user_id: str = "default", metadata_entries: list = None, **kwargs) -> List[str]:
        """Generate text with state-driven error handling, recovery, scribe logging, and always-on memory integration.
        Locking is minimized to only the model inference section to prevent deadlocks with StateManager and other modules.
        """
        request_time = time.time()
        # --- Prompt hardening and input sanitization ---
        if not isinstance(prompt, str):
            self.logger.log_error(
                error_msg=f"Prompt is not a string: {type(prompt)}",
                error_type="invalid_prompt_type"
            )
            raise ValueError("Prompt must be a string.")
        import re
        prompt = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', prompt)
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        max_prompt_length = 1024
        if len(prompt) > max_prompt_length:
            self.logger.log_warning(
                f"Prompt length {len(prompt)} exceeds max {max_prompt_length}, truncating.",
                error_type="prompt_truncation"
            )
            prompt = prompt[:max_prompt_length]
        forbidden_phrases = ["<script>", "DROP TABLE"]
        if any(phrase in prompt for phrase in forbidden_phrases):
            self.logger.log_warning(
                "Prompt contains forbidden content.",
                error_type="forbidden_prompt_content"
            )
            raise ValueError("Prompt contains forbidden content.")
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
            traits = self.primer.prepare_for_generation(prompt, user_id=user_id, metadata_entries=metadata_entries, **kwargs)
            # Use the universal prompt assembler from sovl_primer for all context and vibe injection
            composite_prompt = self.primer.assemble_full_prompt(
                user_prompt=prompt,
                memory_context=memory_context or ""
            )
            # --- Prepare input batch ---
            inputs = self.base_tokenizer(
                composite_prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.model_manager.max_context_length
            )
            model_device = next(self.base_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            gen_kwargs = {"num_return_sequences": num_return_sequences}
            gen_kwargs.update(kwargs)
            base_temp = gen_kwargs.get("temperature", 1.0)
            base_top_k = gen_kwargs.get("top_k", self._get_config_value("controls_config.top_k", 50))
            base_top_p = gen_kwargs.get("top_p", self._get_config_value("controls_config.top_p", 0.95))
            gen_kwargs["temperature"] = self.primer.adjust_parameter(base_temp, "temperature")
            gen_kwargs["top_k"] = int(self.primer.adjust_parameter(base_top_k, "top_k"))
            gen_kwargs["top_p"] = self.primer.adjust_parameter(base_top_p, "top_p")
            # --- Model inference under lock ---
            with self._locks['generation']:
                output_sequences = self.base_model.generate(**inputs, **gen_kwargs)
            generated_texts = [self.base_tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
            generation_result = {
                "generated_texts": generated_texts,
                "generation_config_used": self._get_generation_config(),
                "processing_time_ms": (time.time() - request_time) * 1000
            }
            event_data, source_metadata = ScribeAssembler.assemble_scribe_data(
                manager=self,
                prompt=prompt,
                initial_kwargs=kwargs,
                generation_result=generation_result,
                request_time=request_time,
                session_id=self.session_id,
                user_id=user_id,
            )
            capture_scribe_event(
                origin="sovl_generation",
                event_type="user_interaction",
                event_data=event_data,
                source_metadata=source_metadata,
                session_id=self.session_id
            )
            return generated_texts
        except (ValueError, RuntimeError, GenerationError, IndexError) as e:
            error = e
            self._handle_error("generate_text", e)
            raise
        except Exception as e:
            error = e
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
            self._handle_error("generate_text", e)
            return ["An error occurred during text generation"]
        finally:
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
                max_length=self.model_manager.max_context_length
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
                'top_p': self._get_config_value("controls_config.top_p", 0.95),
                'pad_token_id': self.base_tokenizer.pad_token_id,
                'eos_token_id': self.base_tokenizer.eos_token_id
            }
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
                'top_p': 0.95,
                'pad_token_id': self.base_tokenizer.pad_token_id,
                'eos_token_id': self.base_tokenizer.eos_token_id
            }

    @state_managed_operation("backchannel_scaffold_prompt")
    def backchannel_scaffold_prompt(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        scaffold_index: int = 0,
        return_logits: bool = False,
        return_hidden_states: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Backchannel communication method to directly prompt the scaffold model.
        Locking is minimized to only the model inference section to prevent deadlocks with StateManager and other modules.
        """
        request_time = time.time()
        gpu_mem_needed = 512  # Example: 512MB for generation, adjust as needed
        acquired = False
        try:
            if not self.resource_manager.acquire("gpu_memory", amount=gpu_mem_needed):
                raise RuntimeError("Insufficient GPU memory for scaffold generation")
            acquired = True
            # --- Model inference under lock ---
            with self._locks['generation']:
                # Validate scaffold index and model early
                if not (0 <= scaffold_index < len(self.scaffolds)):
                    raise IndexError(f"Invalid scaffold_index {scaffold_index}. Only {len(self.scaffolds)} scaffolds available")
                scaffold_model = self.scaffolds[scaffold_index]
                if not scaffold_model:
                    raise RuntimeError("Scaffold model not properly initialized")
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
                inputs = self.scaffold_tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                model_device = next(scaffold_model.parameters()).device
                inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}
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
                
                # Build event_data dynamically
                if 'text' in result:
                    generated_text = result['text']
                else:
                    generated_text = self.scaffold_tokenizer.decode(
                        outputs[0] if isinstance(outputs, torch.Tensor) else outputs.sequences[0],
                        skip_special_tokens=True
                    )

                event_data = {
                    "user_response": prompt,
                    "generated_text": generated_text,
                    "num_return_sequences": initial_kwargs.get("num_return_sequences", 1),
                }
                # Optionally add detailed fields if present
                if return_logits and 'logits' in result:
                    event_data["logits"] = result["logits"]
                if return_hidden_states and 'hidden_states' in result:
                    event_data["hidden_states"] = result["hidden_states"]
                if 'return_logits' in locals():
                    event_data["return_logits"] = return_logits
                if 'return_hidden_states' in locals():
                    event_data["return_hidden_states"] = return_hidden_states

                # Build source_metadata dynamically
                source_metadata = {
                    "scaffold_index": scaffold_index,
                    "model_name": self.scaffold_tokenizer.name_or_path,
                    "input_length": len(inputs['input_ids'][0]),
                    "output_length": len(outputs[0] if isinstance(outputs, torch.Tensor) else outputs.sequences[0]),
                    "memory_usage": self.memory_manager.get_memory_usage(),
                    "generation_time": time.time() - request_time,
                    "session_id": self.session_id,
                    "request_timestamp_unix": request_time
                }
                if 'metadata' in result:
                    source_metadata.update(result['metadata'])

                # Single scribe event
                capture_scribe_event(
                    origin="sovl_generation",
                    event_type="backchannel_interaction",
                    event_data=event_data,
                    source_metadata=source_metadata,
                    session_id=self.session_id,
                    timestamp=datetime.fromtimestamp(request_time)
                )

                if 'text' in result:
                    return result
                else:
                    return generated_text
            
        except (ValueError, RuntimeError, GenerationError, IndexError) as e:
            if acquired:
                self.resource_manager.release("gpu_memory", amount=gpu_mem_needed)
            self._handle_error("backchannel_scaffold_prompt", e, {
                'scaffold_index': scaffold_index,
                'prompt_length': len(prompt),
                'generation_params': kwargs,
                'memory_usage': self.memory_manager.get_memory_usage()
            })
            raise
        except Exception as e:
            if acquired:
                self.resource_manager.release("gpu_memory", amount=gpu_mem_needed)
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
            if acquired:
                self.resource_manager.release("gpu_memory", amount=gpu_mem_needed)
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
            pass
        except Exception as e:
            self._handle_error("state_consistency", e)

    def _handle_internal_prompt(self, prompt: str = " ") -> str:
        """Generate a response based on a minimal or provided prompt."""
        request_time = time.time()
        try:
            # Save current history temporarily
            if self.state_manager:
                temp_history = self.state_manager.get_state().history
                def update_fn(state):
                    state.history = ConversationHistory(
                        maxlen=self.controls_config.get("conversation_history_maxlen", 10)
                    )
                    return state
                self.state_manager.update_state_atomic(update_fn)
                # Generate with higher temperature for more creative/expressive output
                response = self.generate(
                    prompt,  # Use the passed or default prompt
                    num_return_sequences=1,
                    temperature=self.controls_config.get("base_temperature", 0.7) + 0.3,
                    top_k=self.controls_config.get("top_k", 50),
                    do_sample=True
                )[0]
                def restore_fn(state):
                    state.history = temp_history
                    return state
                self.state_manager.update_state_atomic(restore_fn)
            else:
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
                self.state.history = temp_history

            # Log the internal thought generation
            capture_scribe_event(
                origin="sovl_generation",
                event_type="internal_thought",
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
        user_id: Optional[str] = None,  # <-- add user_id param
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

        # Retrieve state-based scores/info from manager
        current_memory_mb = manager.memory_manager.get_memory_usage().get("total_mb") if hasattr(manager, 'memory_manager') and manager.memory_manager else None

        # --- Assemble event_data (Data directly related to the event's core action) ---
        event_data = {
            "user_response": prompt,
            "generated_text": generated_texts[0] if generated_texts else "",
            "num_return_sequences": initial_kwargs.get("num_return_sequences", 1),
        }

        # --- Assemble source_metadata (Contextual info about the event) ---
        source_metadata = {
            # External Context
            "session_id": session_id,
            "user_id": user_id,  # <-- add user_id to metadata
            # Request Info
            "request_timestamp_unix": request_time,
            "initial_kwargs": initial_kwargs,
            # Config Used
            "generation_config_used": generation_result.get("generation_config_used"),
            # System State Snapshot
            "model_name": getattr(manager.base_model.config, "_name_or_path", "unknown"),
            "device": str(manager.device),
            "memory_usage_mb": current_memory_mb,
            # Performance
            "processing_time_ms": generation_result.get("processing_time_ms"),
        }

        return event_data, source_metadata
