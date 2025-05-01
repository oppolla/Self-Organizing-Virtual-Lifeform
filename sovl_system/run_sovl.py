import argparse
import os
import sys
import torch
import traceback
import json
import signal
import atexit
import time
import functools
import threading
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Core system imports
from sovl_main import SystemContext, SOVLSystem
from sovl_state import StateManager, SOVLState, StateTracker
from sovl_error import ErrorManager, ErrorContext
from sovl_monitor import SystemMonitor, MemoryMonitor, TraitsMonitor

# Other imports
from sovl_curiosity import CuriosityManager
from sovl_io import load_training_data, InsufficientDataError
from sovl_cli import CommandHandler, run_cli
from sovl_utils import (
    safe_compare, memory_usage, log_memory_usage, dynamic_batch_size,
    detect_repetitions, adjust_temperature, synchronized,
    validate_components, sync_component_states, validate_component_states,
    initialize_component_state
)
from sovl_logger import Logger, LoggerConfig
from sovl_config import ConfigManager
from sovl_conductor import SOVLOrchestrator
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_manager import ModelManager
from sovl_scaffold import (
    ScaffoldTokenMapper,
    CrossAttentionLayer,
    CrossAttentionInjector,
    ScaffoldProvider,
    build_scaffold_token_mapping
)
from sovl_resource import ResourceManager

def error_handler(func):
    """Decorator for consistent error handling and logging."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_context = {
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "traceback": traceback.format_exc()
            }
            self.logger.log_error(
                error_msg=str(e),
                error_type=type(e).__name__,
                error_context=error_context
            )
            raise
    return wrapper

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
FORMATTED_TRAINING_DATA = None
VALID_DATA = None
CHECKPOINT_INTERVAL = 1  # Save checkpoint every epoch by default
COMMAND_CATEGORIES = {
    "System": ["quit", "exit", "save", "load", "reset", "status", "help", "monitor"],
    "Training": ["train", "dream"],
    "Generation": ["generate", "echo", "mimic"],
    "Memory": ["memory", "recall", "forget", "recap"],
    "Interaction": ["muse", "flare", "debate", "spark", "reflect"],
    "Debug": ["log", "config", "panic", "glitch"],
    "Advanced": ["tune", "rewind"],
    "History": ["history"]
}

class SOVLRunner:
    """Main class to manage SOVL system execution."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.context = None
        self.model = None
        self.components = None
        self.orchestrator = None
        self.state_manager = None
        self.error_manager = None
        self.model_manager = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None  # Initialize tokenizer
        self.last_checkpoint_time = None
        self.checkpoint_interval = CHECKPOINT_INTERVAL
        self.metrics_history = []
        self.best_validation_loss = float('inf')
        self.patience = 0
        self.max_patience = 3
        self.traits_monitor = None  # Add traits monitor
        
        # Scaffold-related attributes
        self.scaffold_provider = None
        self.scaffold_token_mapper = None
        self.cross_attention_injector = None
        self.scaffold_model = None
        # Add checkpoint lock for concurrency protection
        self._checkpoint_lock = threading.Lock()
        
    @staticmethod
    def _setup_logger() -> Logger:
        """Configure and return logger instance."""
        logger_config = LoggerConfig(
            log_file=f'sovl_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            max_size_mb=10,
            compress_old=True,
            max_in_memory_logs=1000,
            rotation_count=5
        )
        return Logger(logger_config)
    
    @staticmethod
    def _handle_signal(signum: int, frame: Any, logger: Logger, cleanup_fn: callable):
        """Handle system signals for graceful shutdown."""
        logger.log_event(
            event_type="signal_received",
            message=f"Received signal {signum}, initiating graceful shutdown...",
            level="info"
        )
        cleanup_fn()
        sys.exit(0)
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal_handler = lambda signum, frame: self._handle_signal(
            signum, frame, self.logger, self.cleanup
        )
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    @staticmethod
    def _validate_config_file(config_path: str, logger: Logger) -> bool:
        """Validate configuration file format and required fields."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Required sections with their mandatory fields
            required_sections = {
                'core_config': ['base_model_name', 'base_model_path', 'quantization'],
                'training_config': ['learning_rate', 'grad_accum_steps', 'max_grad_norm'],
                'memory_config': ['memory_threshold', 'memory_decay_rate', 'max_memory_mb'],
                'state_config': ['state_save_interval', 'max_backup_files']
            }
            
            # Validate sections and their fields
            for section, fields in required_sections.items():
                if section not in config:
                    logger.log_error(
                        error_msg=f"Missing required configuration section: {section}",
                        error_type="config_validation_error"
                    )
                    return False
                
                for field in fields:
                    if field not in config[section]:
                        logger.log_error(
                            error_msg=f"Missing required field '{field}' in section '{section}'",
                            error_type="config_validation_error"
                        )
                        return False
            
            # Validate specific value ranges
            try:
                # Training config validation
                lr = float(config['training_config']['learning_rate'])
                if lr <= 0:
                    raise ValueError("learning_rate must be positive")
                
                # Memory config validation
                mem_threshold = float(config['memory_config']['memory_threshold'])
                if not 0 <= mem_threshold <= 1:
                    raise ValueError("memory_threshold must be between 0 and 1")
                
                # State config validation
                save_interval = int(config['state_config']['state_save_interval'])
                if not 60 <= save_interval <= 3600:
                    raise ValueError("state_save_interval must be between 60 and 3600 seconds")
                
            except (ValueError, TypeError) as e:
                logger.log_error(
                    error_msg=f"Invalid configuration value: {str(e)}",
                    error_type="config_validation_error"
                )
                return False
                
            return True
            
        except json.JSONDecodeError as e:
            logger.log_error(
                error_msg=f"Invalid JSON format in configuration file: {config_path}",
                error_type="config_validation_error"
            )
            return False
        except Exception as e:
            logger.log_error(
                error_msg=f"Error validating configuration file: {str(e)}",
                error_type="config_validation_error"
            )
            return False
    
    def _on_config_change(self) -> None:
        """Handle configuration changes and update system components."""
        try:
            self.logger.log_event(
                event_type="config_change",
                message="Configuration changed, updating system...",
                level="info"
            )
            
            # Update optimizer settings if changed
            if self.optimizer:
                optimizer_config = self.context.config_manager.get("training.optimizer", {})
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = optimizer_config.get('learning_rate', param_group['lr'])
                    param_group['weight_decay'] = optimizer_config.get('weight_decay', param_group['weight_decay'])
            
            # Update scheduler settings if changed
            if self.scheduler:
                scheduler_config = self.context.config_manager.get("training.scheduler", {})
                if hasattr(self.scheduler, 'warmup_steps'):
                    self.scheduler.warmup_steps = scheduler_config.get('num_warmup_steps', self.scheduler.warmup_steps)
            
            # Update checkpoint interval
            self.checkpoint_interval = self.context.config_manager.get("training.checkpoint_interval", CHECKPOINT_INTERVAL)
            
            # Update max patience for early stopping
            self.max_patience = self.context.config_manager.get("training.max_patience", 3)
            
            self.logger.log_event(
                event_type="config_change",
                message="Configuration update completed",
                level="info"
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error updating configuration: {str(e)}",
                error_type="config_update_error"
            )

    def _initialize_context(self, args: argparse.Namespace) -> SystemContext:
        """Initialize system context with validation and error handling."""
        try:
            # Validate config file exists and is valid
            if not os.path.exists(args.config):
                raise FileNotFoundError(f"Configuration file not found: {args.config}")
            
            if not self._validate_config_file(args.config, self.logger):
                raise ValueError("Configuration validation failed")
            
            # Initialize config manager
            config_manager = ConfigManager(args.config, self.logger)
            if config_manager is None:
                raise ValueError("ConfigManager could not be instantiated")
            config_manager.subscribe(self._on_config_change)
            
            # Validate device
            if args.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. Please use --device cpu")
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                except Exception as e:
                    raise RuntimeError(f"Unable to get CUDA device properties: {str(e)}")
                required_memory = config_manager.get("memory_config.max_memory_mb", 1024) * 1024 * 1024
                if required_memory > total_memory:
                    raise RuntimeError(f"Insufficient GPU memory. Required: {required_memory/1024/1024}MB, Available: {total_memory/1024/1024}MB")
            
            # Create output directory
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise PermissionError("No write permission in output directory")
            
            self.logger.log_event(
                event_type="device_selected",
                message=f"Using {'CUDA device: ' + torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU device'}",
                level="info"
            )
            
            return SystemContext(
                config_path=args.config,
                device=args.device,
                config_manager=config_manager
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=str(e),
                error_type="context_initialization_error"
            )
            raise
    
    def _initialize_scaffold_components(self) -> None:
        """Initialize scaffold-related components, including scaffold_model, with resource management."""
        resource_manager = self.components.get("resource_manager") if hasattr(self, "components") else None
        acquired = False
        try:
            # Example: acquire 1GB GPU memory for scaffold components (adjust as needed)
            if resource_manager:
                if not resource_manager.acquire("gpu_memory", amount=1024):
                    self.logger.log_error(
                        error_msg="Insufficient GPU memory for scaffold components",
                        error_type="resource_error"
                    )
                    raise RuntimeError("Failed to acquire resources for scaffold components")
                acquired = True
            # Initialize scaffold provider
            self.scaffold_provider = ScaffoldProvider()
            self.logger.log_event("Initialized scaffold provider", level="info")
            # Initialize token mapper
            self.scaffold_token_mapper = ScaffoldTokenMapper()
            self.logger.log_event("Initialized scaffold token mapper", level="info")
            # Initialize cross-attention injector
            self.cross_attention_injector = CrossAttentionInjector()
            self.logger.log_event("Initialized cross-attention injector", level="info")
            # Build token mapping
            mapping = build_scaffold_token_mapping()
            self.scaffold_token_mapper.update_mapping(mapping)
            self.logger.log_event("Built scaffold token mapping", level="info")
            # Initialize scaffold model using ModelManager if available
            model_manager = None
            if hasattr(self, "components") and self.components and "model_manager" in self.components:
                model_manager = self.components["model_manager"]
            if model_manager is not None:
                self.scaffold_model = model_manager.get_scaffold_model()
                if self.scaffold_model is None:
                    self.logger.log_error(error_msg="Failed to initialize scaffold model from ModelManager", error_type="scaffold_error")
                    raise RuntimeError("Scaffold model initialization failed")
                else:
                    self.logger.log_event("Initialized scaffold model from ModelManager", level="info")
            else:
                self.logger.log_error(error_msg="ModelManager not available for scaffold model initialization", error_type="scaffold_error")
                raise RuntimeError("ModelManager not available for scaffold model initialization")
            # Optionally, validate compatibility with cross_attention_injector
            if hasattr(self.cross_attention_injector, 'is_compatible'):
                if not self.cross_attention_injector.is_compatible(self.scaffold_model):
                    self.logger.log_error(error_msg="Scaffold model is not compatible with cross-attention injector", error_type="scaffold_error")
                    raise RuntimeError("Incompatible scaffold model")
        except Exception as e:
            if resource_manager and acquired:
                resource_manager.release("gpu_memory", amount=1024)
            self.logger.log_error(f"Failed to initialize scaffold components: {str(e)}")
            raise

    @error_handler
    def _initialize_components(self, context: SystemContext) -> Dict[str, Any]:
        """Initialize core SOVL components with proper dependency handling."""
        # Define component dependencies
        dependency_graph = {
            "resource_manager": set(),
            "model_manager": set(),
            "state_tracker": set(),
            "error_manager": {"state_tracker"},
            "memory_monitor": {"error_manager", "ram_manager", "gpu_manager"},
            "curiosity_manager": {"error_manager", "state_tracker"},
            "memoria_manager": {"error_manager", "state_tracker"}
        }

        components = {}
        initialized = set()

        def initialize_component(name: str):
            """Recursively initialize a component and its dependencies."""
            if name in initialized:
                return

            # Initialize dependencies first
            for dep in dependency_graph.get(name, set()):
                if dep not in initialized:
                    initialize_component(dep)

            # Component-specific initialization
            if name == "resource_manager":
                components[name] = ResourceManager(logger=context.logger, error_manager=components.get("error_manager"))
            elif name == "model_manager":
                components[name] = ModelManager(
                    config_manager=context.config_manager,
                    logger=context.logger,
                    device=context.device
                )
            elif name == "state_tracker":
                components[name] = StateTracker(
                    config_manager=context.config_manager,
                    logger=context.logger
                )
            elif name == "error_manager":
                components[name] = ErrorManager(
                    context=context,
                    state_tracker=components["state_tracker"],
                    config_manager=context.config_manager
                )
            elif name == "memory_monitor":
                components[name] = MemoryMonitor(
                    config_manager=context.config_manager,
                    logger=context.logger,
                    ram_manager=components.get("ram_manager"),
                    gpu_manager=components.get("gpu_manager"),
                    error_manager=components["error_manager"]
                )
            elif name == "curiosity_manager":
                components[name] = CuriosityManager(
                    config_manager=context.config_manager,
                    logger=context.logger,
                    error_manager=components["error_manager"],
                    device=context.device,
                    state_manager=components["state_tracker"]
                )
            elif name == "memoria_manager":
                components[name] = MemoriaManager(
                    config_manager=context.config_manager,
                    logger=context.logger,
                    error_manager=components["error_manager"],
                    state_tracker=components["state_tracker"]
                )

            initialized.add(name)
            self.logger.log_event(
                event_type="component_initialization",
                message=f"Initialized {name}",
                level="info"
            )

        # Initialize all components
        for component_name in dependency_graph.keys():
            initialize_component(component_name)

        return components

    def _initialize_optimizer(self, model: torch.nn.Module) -> None:
        """Initialize optimizer and learning rate scheduler."""
        try:
            # Get optimizer configuration from config manager
            optimizer_config = self.context.config_manager.get("training.optimizer", {})
            optimizer_type = optimizer_config.get("type", "adamw")
            learning_rate = optimizer_config.get("learning_rate", 5e-5)
            weight_decay = optimizer_config.get("weight_decay", 0.01)
            
            # Initialize optimizer
            if optimizer_type.lower() == "adamw":
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
            # Initialize scheduler
            scheduler_config = self.context.config_manager.get("training.scheduler", {})
            scheduler_type = scheduler_config.get("type", "linear")
            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
            num_training_steps = scheduler_config.get("num_training_steps", 1000)
            
            if scheduler_type.lower() == "linear":
                from transformers import get_linear_schedule_with_warmup
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
            self.logger.log_event(
                event_type="optimizer_initialization",
                message=f"Initialized {optimizer_type} optimizer and {scheduler_type} scheduler",
                level="info",
                additional_info={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "num_warmup_steps": num_warmup_steps,
                    "num_training_steps": num_training_steps
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to initialize optimizer: {str(e)}",
                error_type="optimizer_initialization_error"
            )
            raise
    
    def cleanup(self):
        """Release system resources with logging and error handling."""
        try:
            self.logger.log_event(
                event_type="cleanup",
                message="Starting cleanup...",
                level="info"
            )
            
            # Unsubscribe from configuration changes
            if hasattr(self, "context") and self.context and hasattr(self.context, "config_manager") and self.context.config_manager:
                self.context.config_manager.unsubscribe(self._on_config_change)
            
            # Cleanup model manager and model
            if hasattr(self, "model_manager") and self.model_manager:
                self.model_manager.cleanup()
                self.model = None
                if hasattr(self, "tokenizer"):
                    self.tokenizer = None
            
            # Cleanup optimizer and scheduler
            if hasattr(self, "optimizer") and self.optimizer:
                self.optimizer = None
            if hasattr(self, "scheduler") and self.scheduler:
                self.scheduler = None
            
            # Cleanup scaffold components
            for attr in ["scaffold_provider", "scaffold_token_mapper", "cross_attention_injector", "scaffold_model"]:
                if hasattr(self, attr):
                    setattr(self, attr, None)
            
            # Cleanup context and components
            if hasattr(self, "context") and self.context:
                if hasattr(self.context, "cleanup"):
                    self.context.cleanup()
                self.context = None
            if hasattr(self, "components") and self.components:
                self.components = None
            
            # Clear CUDA cache if using GPU
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Add traits monitor cleanup
            if hasattr(self, "traits_monitor") and self.traits_monitor:
                if hasattr(self.traits_monitor, "stop"):
                    self.traits_monitor.stop()
                self.traits_monitor = None
            
            self.logger.log_event(
                event_type="cleanup",
                message="Cleanup completed successfully",
                level="info"
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error during cleanup: {str(e)}",
                error_type="cleanup_error"
            )

    def _run_system(self, args: argparse.Namespace):
        """Run the SOVL system with monitoring and error handling."""
        try:
            self.logger.log_event(
                event_type="system_start",
                message="Initializing SOVL system...",
                level="info"
            )
            
            self.orchestrator = SOVLOrchestrator(
                config_path=args.config,
                log_file=self.logger.config.log_file,
                optimizer=self.optimizer,  # Pass optimizer to orchestrator
                scheduler=self.scheduler   # Pass scheduler to orchestrator
            )
            self.orchestrator.initialize_system()
            self.orchestrator.run()
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error during system execution: {str(e)}",
                error_type="system_execution_error"
            )
            raise
    
    def execute_command(self, sovl_system: SOVLSystem, command: str, args: List[str] = None) -> bool:
        """Execute a command with proper error handling and logging."""
        try:
            args = args or []
            cmd_handler = CommandHandler(sovl_system, self.logger)
            
            # Unified monitor command
            if command == "monitor":
                if not args:
                    # Default behavior: show status of both monitoring systems
                    print("\nMonitoring Status:")
                    print("-----------------")
                    
                    # System monitoring status
                    system_status = cmd_handler.get_monitor_status()
                    print(f"System Monitor: {system_status}")
                    
                    # Traits monitoring status
                    if self.traits_monitor:
                        traits_status = "running" if self.traits_monitor.is_running() else "stopped"
                        print(f"Traits Monitor: {traits_status}")
                    else:
                        print("Traits Monitor: not initialized")
                        
                    print("\nUsage: monitor [system|traits] [start|stop|status]")
                    return True
                    
                monitor_type = args[0]
                action = args[1] if len(args) > 1 else "status"
                
                if monitor_type == "traits":
                    if not self.traits_monitor:
                        print("Traits monitor not initialized")
                        return False
                        
                    if action == "start":
                        self.traits_monitor.start()
                        print("Traits monitoring started. Press 'q' in the monitor window to stop.")
                        return True
                    elif action == "stop":
                        self.traits_monitor.stop()
                        print("Traits monitoring stopped.")
                        return True
                    elif action == "status":
                        status = "running" if self.traits_monitor.is_running() else "stopped"
                        print(f"Traits monitor is {status}")
                        return True
                elif monitor_type == "system":
                    # Handle system monitoring
                    if action == "start":
                        return cmd_handler.start_monitoring()
                    elif action == "stop":
                        return cmd_handler.stop_monitoring()
                    elif action == "status":
                        return cmd_handler.get_monitor_status()
                else:
                    print("Invalid monitor type. Use 'system' or 'traits'")
                    return False
                    
            return cmd_handler.handle_command(command, args)
                
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error executing command {command}: {str(e)}",
                error_type="command_execution_error"
            )
            print(f"Error: {str(e)}")
            return False
    
    @error_handler
    async def _async_save_checkpoint(self, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Asynchronously save system checkpoint with concurrency protection and non-blocking I/O."""
        import uuid
        from concurrent.futures import ThreadPoolExecutor
        current_time = time.time()
        checkpoint_data = {
            "version": "1.0",  # Add versioning for future format changes
            "timestamp": current_time,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "component_states": {
                name: comp.to_dict() for name, comp in self.components.items()
                if self._validate_component_serialization(comp, name)
            }
        }
        temp_path = Path("checkpoints") / f"temp_{current_time}_{uuid.uuid4().hex}.pt"
        final_path = Path("checkpoints") / f"checkpoint_{current_time}.pt"
        loop = asyncio.get_running_loop()
        try:
            with self._checkpoint_lock:
                # Save to temporary file in a thread pool
                await loop.run_in_executor(None, torch.save, checkpoint_data, temp_path)
                # Atomic rename in a thread pool
                await loop.run_in_executor(None, temp_path.rename, final_path)
                self.last_checkpoint_time = current_time
                self.logger.log_event(
                    event_type="checkpoint",
                    message=f"Checkpoint saved to {final_path}",
                    level="info"
                )
            return True
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to save checkpoint: {str(e)}",
                error_type="checkpoint_save_error"
            )
            return False

    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Validate checkpoint data structure and compatibility."""
        try:
            # Check required fields
            required_fields = ["timestamp", "model_state"]
            for field in required_fields:
                if field not in checkpoint_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate model state compatibility (optional deeper check)
            if self.model:
                model_keys = set(self.model.state_dict().keys())
                checkpoint_keys = set(checkpoint_data["model_state"].keys())
                missing_keys = model_keys - checkpoint_keys
                
                if missing_keys:
                    self.logger.log_event(
                        event_type="checkpoint_warning",
                        message=f"Checkpoint missing {len(missing_keys)} model keys",
                        level="warning"
                    )
            
            return True
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Checkpoint validation failed: {str(e)}",
                error_type="checkpoint_validation_error"
            )
            return False

    def _load_partial_state(self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
        """Load partial model state and return success status and missing keys."""
        try:
            model_dict = model.state_dict()
            # Filter out incompatible keys
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            missing_keys = [k for k in model_dict.keys() if k not in filtered_dict]
            
            # Load compatible keys
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            
            return len(filtered_dict) > 0, missing_keys
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Partial state loading failed: {str(e)}",
                error_type="checkpoint_error"
            )
            return False, []

    @error_handler
    def save_checkpoint(self, force: bool = False, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Save system state checkpoint with async support."""
        if not force and self.last_checkpoint_time is not None:
            if time.time() - self.last_checkpoint_time < self.checkpoint_interval:
                return False
            
        try:
            self.logger.log_event(
                event_type="checkpoint",
                message="Saving system checkpoint...",
                level="info"
            )
            
            # Run async save
            return asyncio.run(self._async_save_checkpoint(optimizer))
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to save checkpoint: {str(e)}",
                error_type="checkpoint_error"
            )
            return False

    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Load system state from checkpoint (.pt file). Handles partial restoration and logs missing/incompatible fields."""
        try:
            self.logger.log_event(
                event_type="checkpoint",
                message=f"Loading checkpoint from {checkpoint_path}...",
                level="info"
            )

            # Load checkpoint using torch.load
            checkpoint_data = torch.load(checkpoint_path, map_location=self.context.device)

            # Explicitly check for required fields
            required_fields = ["timestamp", "model_state"]
            missing_fields = [f for f in required_fields if f not in checkpoint_data]
            if missing_fields:
                self.logger.log_error(
                    error_msg=f"Checkpoint missing required fields: {missing_fields}",
                    error_type="checkpoint_schema_error"
                )
                return False

            # Validate checkpoint structure (calls _validate_checkpoint for deeper checks)
            if not self._validate_checkpoint(checkpoint_data):
                return False

            # Load model state (partial if needed)
            if self.model and "model_state" in checkpoint_data:
                success, missing_keys = self._load_partial_state(self.model, checkpoint_data["model_state"])
                if not success:
                    self.logger.log_error(
                        error_msg="Failed to load model state from checkpoint.",
                        error_type="checkpoint_error"
                    )
                    return False
                if missing_keys:
                    self.logger.log_event(
                        event_type="checkpoint_warning",
                        message=f"Model loaded with {len(missing_keys)} missing keys: {missing_keys}",
                        level="warning"
                    )
            else:
                self.logger.log_error(
                    error_msg="No model_state found in checkpoint.",
                    error_type="checkpoint_schema_error"
                )

            # Load optimizer state if present
            if optimizer is not None:
                if "optimizer_state" in checkpoint_data and checkpoint_data["optimizer_state"] is not None:
                    try:
                        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
                        self.logger.log_event(
                            event_type="checkpoint",
                            message="Optimizer state loaded successfully",
                            level="info"
                        )
                    except Exception as e:
                        self.logger.log_error(
                            error_msg=f"Failed to load optimizer state: {str(e)}",
                            error_type="checkpoint_error"
                        )
                        # Continue loading without optimizer state
                else:
                    self.logger.log_event(
                        event_type="checkpoint_warning",
                        message="No optimizer_state found in checkpoint.",
                        level="warning"
                    )

            # Load component states if available
            if "component_states" in checkpoint_data:
                loaded_components = 0
                for name, component_data in checkpoint_data["component_states"].items():
                    if name in self.components:
                        try:
                            self.components[name].from_dict(component_data)
                            loaded_components += 1
                        except Exception as e:
                            self.logger.log_error(
                                error_msg=f"Failed to load state for component {name}: {str(e)}",
                                error_type="checkpoint_error"
                            )
                            # Continue with other components instead of failing completely
                self.logger.log_event(
                    event_type="checkpoint",
                    message=f"Loaded {loaded_components} component states from checkpoint.",
                    level="info"
                )
            else:
                self.logger.log_event(
                    event_type="checkpoint_warning",
                    message="No component_states found in checkpoint.",
                    level="warning"
                )

            self.last_checkpoint_time = checkpoint_data["timestamp"]
            self.logger.log_event(
                event_type="checkpoint",
                message="Checkpoint loaded successfully",
                level="info"
            )
            return True

        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to load checkpoint: {str(e)}",
                error_type="checkpoint_error"
            )
            return False

    def cleanup_old_checkpoints(self, max_checkpoints: int = 5):
        """Remove old checkpoints to manage disk space."""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Update to look for .pt files instead of .json
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"), key=lambda x: x.stat().st_mtime)
            
            # Remove older checkpoints beyond the max limit
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                old_checkpoint.unlink()
                self.logger.log_event(
                    event_type="checkpoint_cleanup",
                    message=f"Removed old checkpoint: {old_checkpoint}",
                    level="info"
                )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to clean up old checkpoints: {str(e)}",
                error_type="checkpoint_cleanup_error"
            )

    @error_handler
    def _run_validation(self, valid_data: List[Dict[str, Any]], batch_size: int) -> Dict[str, float]:
        """Run validation with automatic mixed precision and memory optimization."""
        if not valid_data:
            self.logger.log_event(
                event_type="validation",
                message="No validation data provided",
                level="warning"
            )
            return {}
            
        try:
            self.logger.log_event(
                event_type="validation_start",
                message="Starting validation...",
                level="info"
            )
            
            self.model.eval()
            metrics = defaultdict(float)
            total_batches = 0
            
            # Initialize gradient scaler for mixed precision
            scaler = torch.cuda.amp.GradScaler()
            
            with torch.no_grad():
                for i in range(0, len(valid_data), batch_size):
                    batch = valid_data[i:i + batch_size]
                    if not batch:
                        continue
                        
                    # Use automatic mixed precision
                    with torch.cuda.amp.autocast():
                        inputs = self._prepare_batch(batch)
                        outputs = self.model(**inputs)
                        loss = self._calculate_loss(outputs, inputs)
                        
                    # Update metrics
                    metrics["loss"] += loss.item()
                    total_batches += 1
                    
                    # Clear cache periodically
                    if i % (batch_size * 10) == 0:
                        torch.cuda.empty_cache()
                        
            # Calculate average metrics
            if total_batches > 0:
                return {k: v/total_batches for k, v in metrics.items()}
            return {}
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Validation error: {str(e)}",
                error_type="validation_error"
            )
            return {}

    @error_handler
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for model input with parallel processing. Always includes labels."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        if not batch or "text" not in batch[0]:
            raise ValueError("Each batch item must contain a 'text' field.")
        # Extract texts for parallel processing
        texts = [item["text"] for item in batch]
        # Use thread pool for parallel tokenization
        with ThreadPoolExecutor(max_workers=4) as executor:
            tokenized = list(executor.map(
                lambda text: self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.context.config_manager.get("training.max_seq_length", 512)
                ),
                texts
            ))
        # Find maximum sequence length in batch
        max_length = max(len(t["input_ids"]) for t in tokenized)
        batch_size = len(tokenized)
        # Initialize tensors with padding
        input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id)
        attention_mask = torch.zeros((batch_size, max_length))
        # Fill tensors with tokenized data
        for i, tokens in enumerate(tokenized):
            length = len(tokens["input_ids"])
            input_ids[i, :length] = torch.tensor(tokens["input_ids"])
            attention_mask[i, :length] = 1
        # Prepare labels
        if "labels" in batch[0]:
            label_texts = [item["labels"] for item in batch]
            with ThreadPoolExecutor(max_workers=4) as executor:
                tokenized_labels = list(executor.map(
                    lambda text: self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.context.config_manager.get("training.max_seq_length", 512)
                    ),
                    label_texts
                ))
            labels = torch.full((batch_size, max_length), -100)  # -100 for ignored positions
            for i, tokens in enumerate(tokenized_labels):
                length = min(len(tokens["input_ids"]), max_length)
                labels[i, :length] = torch.tensor(tokens["input_ids"][:length])
        else:
            labels = input_ids.clone()
        # Move tensors to appropriate device
        device = self.context.device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device)
        }

    def _calculate_loss(self, outputs: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate loss for the model outputs."""
        try:
            return torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                inputs["labels"].view(-1),
                ignore_index=-100
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Loss calculation error: {str(e)}",
                error_type="loss_calculation_error"
            )
            raise

    def _update_metrics_history(self, metrics: Dict[str, float], epoch: int) -> None:
        """Update metrics history and handle early stopping."""
        try:
            # Add metrics to history
            self.metrics_history.append({
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            # Check for early stopping
            if metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = metrics["loss"]
                self.patience = 0
            else:
                self.patience += 1
                
            # Log metrics update
            self.logger.log_event(
                event_type="metrics_update",
                message=f"Metrics updated for epoch {epoch}",
                level="info",
                additional_info={
                    "metrics": metrics,
                    "best_validation_loss": self.best_validation_loss,
                    "patience": self.patience
                }
            )
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Metrics update error: {str(e)}",
                error_type="metrics_update_error"
            )
            raise

    def _validate_component_serialization(self, component: Any, name: str) -> bool:
        """Validate that a component has required serialization methods."""
        try:
            if not hasattr(component, 'to_dict') or not hasattr(component, 'from_dict'):
                raise ValueError(f"Component {name} missing required serialization methods")
            
            # Validate scaffold components
            if name in ["scaffold_provider", "scaffold_token_mapper", "cross_attention_injector"]:
                if component is not None and (not hasattr(component, 'to_dict') or not hasattr(component, 'from_dict')):
                    raise ValueError(f"Scaffold component {name} missing required serialization methods")
            
            
            self.logger.log_event(f"Validated component {name} serialization", level="info")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Component {name} serialization validation failed: {str(e)}")
            return False

    def run(self):
        """Main execution method with enhanced argument parsing."""
        parser = argparse.ArgumentParser(
            description="Run the SOVL AI system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument("--config", default="sovl_config.json", help="Path to configuration file")
        parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
        parser.add_argument("--mode", default="train", choices=["train", "generate", "dream"], help="Operation mode")
        parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
        parser.add_argument("--train-data", help="Path to training data file")
        parser.add_argument("--valid-data", help="Path to validation data file")
        parser.add_argument("--test", action="store_true", help="Run system in test mode")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        parser.add_argument("--monitor-interval", type=float, default=1.0, help="Monitoring update interval in seconds")
        parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL, 
                          help="Checkpoint interval in epochs")
        parser.add_argument("--resume-from-checkpoint", help="Path to checkpoint file to resume from")
        parser.add_argument("--validate-every", type=int, default=1, help="Run validation every N epochs")
        parser.add_argument("--max-patience", type=int, default=3, help="Max epochs without validation improvement")
        parser.add_argument("--max-checkpoints", type=int, default=5, help="Maximum number of checkpoints to keep")
        
        args = parser.parse_args()
        
        if args.verbose:
            # Set logger to debug level using SOVL logger config
            self.logger.update_config(log_level="debug")
        
        self.checkpoint_interval = args.checkpoint_interval
        self.max_patience = args.max_patience
        
        self._register_signal_handlers()
        atexit.register(self.cleanup)
        
        self.logger.info("Waking SOVL system...")
        self.logger.info(f"Configuration: {args}")
        
        try:
            self.context = self._initialize_context(args)
            if self.context is None:
                self.logger.log_error(error_msg="Failed to initialize context", error_type="context_error")
                return
            self.components = self._initialize_components(self.context)
            if not self.components:
                self.logger.log_error(error_msg="Component initialization failed or incomplete", error_type="component_error")
                return
                
            # Get model from ModelManager instead of direct tuple indexing
            if "model_manager" in self.components and self.components["model_manager"] is not None:
                model_manager = self.components["model_manager"]
                self.model = model_manager.get_base_model()
                if not isinstance(self.model, torch.nn.Module):
                    self.logger.log_error(error_msg="Invalid model type", error_type="component_error")
                    return
                # Get the tokenizer from the model manager
                self.tokenizer = model_manager.get_base_tokenizer()
                if self.tokenizer is None:
                    # Fallback: try to initialize from transformers
                    try:
                        from transformers import AutoTokenizer
                        model_name_or_path = self.context.config_manager.get("core_config.base_model_name")
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                        self.logger.log_event(
                            event_type="tokenizer_initialization_fallback",
                            message=f"Tokenizer initialized from pretrained: {model_name_or_path}",
                            level="info"
                        )
                    except Exception as e:
                        self.logger.log_error(error_msg=f"Failed to initialize tokenizer from pretrained: {str(e)}", error_type="tokenizer_error")
                        return
                else:
                    self.logger.log_event(
                        event_type="tokenizer_initialization",
                        message="Tokenizer initialized successfully",
                        level="info"
                    )
                # Initialize optimizer after model is ready
                self._initialize_optimizer(self.model)
                if self.optimizer is None:
                    self.logger.log_error(error_msg="Optimizer initialization failed", error_type="optimizer_error")
                    return
            else:
                self.logger.log_error(error_msg="ModelManager not initialized", error_type="component_error")
                return
            
            # Initialize state manager
            self.state_manager = StateManager(
                self.context.config_manager,
                self.logger,
                self.context.device
            )
            self.state_manager.initialize_state()
            
            # Always use self.optimizer for checkpoint loading
            if args.resume_from_checkpoint:
                if not self.load_checkpoint(args.resume_from_checkpoint, optimizer=self.optimizer):
                    self.logger.log_error(
                        error_msg="Failed to load checkpoint, starting fresh",
                        error_type="checkpoint_error"
                    )
            
            # Initialize orchestrator
            try:
                self.orchestrator = SOVLOrchestrator(
                    model=self.model,
                    components=self.components,
                    context=self.context
                )
            except Exception as e:
                self.logger.log_error(error_msg=f"Failed to initialize orchestrator: {str(e)}", error_type="orchestrator_error")
                return
            if not self.orchestrator:
                self.logger.log_error(error_msg="Orchestrator not initialized", error_type="orchestrator_error")
                return
            
            # Run system
            try:
                if args.mode == 'train':
                    self.logger.log_event(
                        event_type="training",
                        message="Starting gestation...",
                        level="info"
                    )
                    
                    # Load training and validation data
                    formatted_training_data = load_training_data(args.train_data) if args.train_data else []
                    valid_data = load_training_data(args.valid_data) if args.valid_data else []
                    
                    if not formatted_training_data:
                        self.logger.warning("No training data available")
                        return
                    
                    # Training loop with validation
                    for epoch in range(args.epochs):
                        self.logger.log_event(
                            event_type="epoch_start",
                            message=f"Starting epoch {epoch + 1}/{args.epochs}",
                            level="info"
                        )
                        # Training phase
                        train_loss = self.orchestrator.train(
                            epochs=1,
                            batch_size=args.batch_size,
                            formatted_training_data=formatted_training_data,
                            valid_data=valid_data,
                            optimizer=self.optimizer,
                            checkpoint_callback=lambda: self.save_checkpoint(optimizer=self.optimizer),
                            validate_every=args.validate_every
                        )
                        
                        # Validation phase
                        if valid_data and (epoch + 1) % args.validate_every == 0:
                            metrics = self.orchestrator.validate(valid_data)
                            valid_loss = metrics.get("loss", float("inf"))
                            self._update_metrics_history(metrics, epoch + 1)
                            self.logger.log_event(
                                event_type="validation",
                                message=f"Epoch {epoch + 1} validation results",
                                level="info",
                                additional_info={
                                    "train_loss": train_loss,
                                    "valid_loss": valid_loss,
                                    "metrics": metrics
                                }
                            )
                            if self.patience >= self.max_patience:
                                self.logger.log_event(
                                    event_type="early_stopping",
                                    message=f"Early stopping triggered after {self.patience} epochs without improvement",
                                    level="info"
                                )
                                break
                        # Save checkpoint and clean up old ones
                        self.save_checkpoint(optimizer=self.optimizer)
                        self.cleanup_old_checkpoints(args.max_checkpoints)
                elif args.mode == 'generate':
                    self.logger.log_event(
                        event_type="generation",
                        message="Starting generation...",
                        level="info"
                    )
                    self.orchestrator.generate()
                elif args.mode == 'dream':
                    self.logger.log_event(
                        event_type="dreaming",
                        message="Starting dreaming...",
                        level="info"
                    )
                    self.orchestrator.dream()
                else:
                    raise ValueError(f"Invalid mode: {args.mode}")
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"System error during operation: {str(e)}",
                    error_type="system_error"
                )
                raise
        except Exception as e:
            self.logger.log_error(
                error_msg=f"System error: {str(e)}",
                error_type="system_error"
            )
            raise
        finally:
            # Save final checkpoint and cleanup, but guard against errors in these methods
            try:
                self.save_checkpoint(force=True, optimizer=self.optimizer)
            except Exception as e:
                self.logger.log_error(error_msg=f"Error saving final checkpoint: {str(e)}", error_type="checkpoint_error")
            try:
                self.cleanup_old_checkpoints(args.max_checkpoints)
            except Exception as e:
                self.logger.log_error(error_msg=f"Error cleaning up old checkpoints: {str(e)}", error_type="checkpoint_error")
            try:
                self.cleanup()
            except Exception as e:
                self.logger.log_error(error_msg=f"Error during cleanup: {str(e)}", error_type="cleanup_error")

def main():
    """Entry point for the SOVL system."""
    runner = SOVLRunner()
    runner.run()  # This does all the setup and returns when ready

    # After setup, launch the CLI if initialization succeeded
    if hasattr(runner, 'context') and runner.context is not None:
        run_cli(runner.context)
    else:
        print("[ERROR] System initialization failed. CLI will not be started.")

if __name__ == "__main__":
    main()
