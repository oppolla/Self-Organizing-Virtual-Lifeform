import argparse
import os
import sys
import torch
import traceback
import json
import signal
import atexit
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sovl_main import SystemContext, SOVLSystem, StateTracker, ErrorManager, MemoryMonitor
from sovl_curiosity import CuriosityEngine
from sovl_io import load_training_data, InsufficientDataError
from sovl_monitor import SystemMonitor
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
from sovl_memory import MemoriaManager, RAMManager, GPUMemoryManager
from sovl_state import StateManager, SOVLState
from sovl_error import ErrorManager, ErrorContext
from sovl_manager import ModelManager
from sovl_scaffold import (
    ScaffoldTokenMapper,
    CrossAttentionLayer,
    CrossAttentionInjector,
    ScaffoldProvider,
    build_scaffold_token_mapping
)

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
        
        # Scaffold-related attributes
        self.scaffold_provider = None
        self.scaffold_token_mapper = None
        self.cross_attention_injector = None
        self.scaffold_model = None
        
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
            config_manager.subscribe(self._on_config_change)
            
            # Validate device
            if args.device == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. Please use --device cpu")
                # Check GPU memory
                total_memory = torch.cuda.get_device_properties(0).total_memory
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
        """Initialize scaffold-related components."""
        try:
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
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize scaffold components: {str(e)}")
            raise

    def _initialize_components(self, context: SystemContext) -> Tuple:
        """Initialize core SOVL components with proper dependency handling."""
        components = []
        component_classes = [
            (ModelLoader, "model loader"),
            (None, "model"),  # Model is loaded separately
            (StateTracker, "state tracker"),
            (ErrorManager, "error manager"),
            (MemoryMonitor, "memory monitor"),
            (CuriosityEngine, "curiosity engine"),
            (MemoriaManager, "memory manager")
        ]
        
        try:
            # Initialize ModelManager first
            self.model_manager = ModelManager(
                config_manager=context.config_manager,
                logger=self.logger,
                device=context.device
            )
            
            # Initialize components in stages
            stage1_components = []  # Basic components without dependencies
            stage2_components = []  # Components with basic dependencies
            stage3_components = []  # Components with complex dependencies
            
            for component_class, name in component_classes:
                try:
                    if component_class is None:  # Handle model loading
                        self.logger.log_event(
                            event_type="component_initialization",
                            message="Loading language model...",
                            level="info"
                        )
                        self.model_manager.load_models()
                        self.model = self.model_manager.get_base_model()
                        self.tokenizer = self.model_manager.get_tokenizer()
                        components.append(self.model)
                        continue
                    
                    # Stage 1: Basic components
                    if name in ["model loader", "state tracker", "memory monitor"]:
                        component = initialize_component(component_class, context)
                        stage1_components.append(component)
                        components.append(component)
                    
                    # Stage 2: Components with basic dependencies
                    elif name == "error manager":
                        if not any(isinstance(c, StateTracker) for c in components):
                            raise RuntimeError("StateTracker must be initialized before ErrorManager")
                        component = ErrorManager(
                            context=context,
                            state_tracker=next(c for c in components if isinstance(c, StateTracker)),
                            config_manager=context.config_manager
                        )
                        self.error_manager = component
                        stage2_components.append(component)
                        components.append(component)
                    
                    # Stage 3: Components with complex dependencies
                    elif name in ["curiosity engine", "memory manager"]:
                        if not self.error_manager:
                            raise RuntimeError("ErrorManager must be initialized before " + name)
                        if not any(isinstance(c, StateTracker) for c in components):
                            raise RuntimeError("StateTracker must be initialized before " + name)
                        
                        if name == "curiosity engine":
                            component = CuriosityEngine(
                                config=context.config_manager,
                                model_manager=self.model_manager,
                                logger=context.logger
                            )
                        else:  # memory manager
                            component = initialize_component(MemoriaManager, context.config_manager, context.device, context.logger)
                        
                        stage3_components.append(component)
                        components.append(component)
                    
                    self.logger.log_event(
                        event_type="component_initialization",
                        message=f"Initialized {name}",
                        level="info"
                    )
                    
                except Exception as e:
                    self.logger.log_error(
                        error_msg=f"Failed to initialize {name}: {str(e)}",
                        error_type="component_initialization_error"
                    )
                    raise
            
            # Validate all components
            validate_components(*components)
            initialize_component_state(components[2], components)
            
            self.logger.log_event(
                event_type="component_initialization",
                message="All components initialized successfully",
                level="info"
            )
            return tuple(components)
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Component initialization failed: {str(e)}",
                error_type="component_initialization_error"
            )
            raise
    
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
            if self.context and self.context.config_manager:
                self.context.config_manager.unsubscribe(self._on_config_change)
            
            # Cleanup model manager and model
            if self.model_manager:
                self.model_manager.cleanup()
                self.model = None
                self.tokenizer = None
            
            # Cleanup optimizer and scheduler
            if self.optimizer:
                self.optimizer = None
            if self.scheduler:
                self.scheduler = None
            
            # Cleanup scaffold components
            if self.scaffold_provider:
                self.scaffold_provider = None
            if self.scaffold_token_mapper:
                self.scaffold_token_mapper = None
            if self.cross_attention_injector:
                self.cross_attention_injector = None
            if self.scaffold_model:
                self.scaffold_model = None
            
            # Cleanup context and components
            if self.context:
                self.context.cleanup()
                self.context = None
            if self.components:
                self.components = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
            return cmd_handler.handle_command(command, args)
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Error executing command {command}: {str(e)}",
                error_type="command_execution_error"
            )
            print(f"Error: {str(e)}")
            return False
    
    def save_checkpoint(self, force: bool = False, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Save system state checkpoint."""
        current_time = time.time()
        if not force and self.last_checkpoint_time is not None:
            if current_time - self.last_checkpoint_time < self.checkpoint_interval:
                return False
                
        try:
            self.logger.log_event(
                event_type="checkpoint",
                message="Saving system checkpoint...",
                level="info"
            )
            
            # Validate component serialization before saving
            for name, component in self.components.items():
                if not self._validate_component_serialization(component, name):
                    self.logger.log_error(
                        error_msg=f"Cannot save checkpoint: Component {name} serialization validation failed",
                        error_type="checkpoint_error"
                    )
                    return False
            
            # Save model state using ModelManager
            model_path = None
            if self.model_manager:
                model_path = self.model_manager.save_model_state(current_time)
                
            # Save system state
            state_path = Path("checkpoints") / f"state_{int(current_time)}.json"
            state_data = {
                "timestamp": current_time,
                "model_path": str(model_path) if model_path else None,
                "components": {
                    name: component.to_dict() 
                    for name, component in self.components.items()
                    if self._validate_component_serialization(component, name)
                }
            }
            
            # Save optimizer state if provided
            if optimizer is not None:
                optimizer_path = Path("checkpoints") / f"optimizer_{int(current_time)}.pt"
                torch.save(optimizer.state_dict(), optimizer_path)
                state_data["optimizer_path"] = str(optimizer_path)
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
                
            self.last_checkpoint_time = current_time
            self.logger.log_event(
                event_type="checkpoint",
                message=f"Checkpoint saved successfully to {state_path}",
                level="info"
            )
            return True
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to save checkpoint: {str(e)}",
                error_type="checkpoint_error"
            )
            return False
            
    def load_checkpoint(self, checkpoint_path: str, optimizer: Optional[torch.optim.Optimizer] = None) -> bool:
        """Load system state from checkpoint."""
        try:
            self.logger.log_event(
                event_type="checkpoint",
                message=f"Loading checkpoint from {checkpoint_path}...",
                level="info"
            )
            
            with open(checkpoint_path, 'r') as f:
                state_data = json.load(f)
                
            # Load model state using ModelManager
            if self.model_manager and state_data["model_path"] is not None:
                self.model_manager.load_model_state(state_data["model_path"])
                self.model = self.model_manager.get_base_model()
                    
            # Load optimizer state if provided
            if optimizer is not None and "optimizer_path" in state_data:
                optimizer_path = Path(state_data["optimizer_path"])
                if optimizer_path.exists():
                    optimizer.load_state_dict(torch.load(optimizer_path))
                    
            # Load component states
            for name, component_data in state_data["components"].items():
                if name in self.components:
                    component = self.components[name]
                    if self._validate_component_serialization(component, name):
                        try:
                            component.from_dict(component_data)
                        except Exception as e:
                            self.logger.log_error(
                                error_msg=f"Failed to load state for component {name}: {str(e)}",
                                error_type="checkpoint_error"
                            )
                            return False
                    else:
                        self.logger.log_error(
                            error_msg=f"Cannot load state for component {name}: Serialization validation failed",
                            error_type="checkpoint_error"
                        )
                        return False
                    
            self.last_checkpoint_time = state_data["timestamp"]
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
            checkpoints = sorted(checkpoint_dir.glob("state_*.json"), key=lambda x: x.stat().st_mtime)
            for old_checkpoint in checkpoints[:-max_checkpoints]:
                old_checkpoint.unlink()
                model_path = checkpoint_dir / f"model_{old_checkpoint.stem.split('_')[1]}.pt"
                if model_path.exists():
                    model_path.unlink()
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

    def _run_validation(self, valid_data: List[Dict[str, Any]], batch_size: int) -> Dict[str, float]:
        """Run validation loop and compute metrics."""
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
            total_loss = 0.0
            total_batches = 0
            metrics = {metric: 0.0 for metric in self.context.config_manager.get("training.metrics_to_track", ["loss", "accuracy", "confidence"])}
            
            with torch.no_grad():
                for i in range(0, len(valid_data), batch_size):
                    try:
                        batch = valid_data[i:i + batch_size]
                        if not batch:
                            continue
                            
                        inputs = self._prepare_batch(batch)
                        outputs = self.model(**inputs)
                        loss = self._calculate_loss(outputs, inputs)
                        
                        total_loss += loss.item()
                        total_batches += 1
                        
                        if "accuracy" in metrics:
                            preds = outputs.logits.argmax(dim=-1)
                            mask = inputs["labels"] != -100
                            correct = (preds[mask] == inputs["labels"][mask]).sum().item()
                            total = mask.sum().item()
                            metrics["accuracy"] += correct / total if total > 0 else 0.0
                            
                        if "perplexity" in metrics:
                            metrics["perplexity"] += torch.exp(loss).item()
                            
                        if "confidence" in metrics:
                            probs = torch.softmax(outputs.logits, dim=-1)
                            max_probs = probs.max(dim=-1)[0]
                            metrics["confidence"] += max_probs[mask].mean().item()
                            
                    except Exception as e:
                        self.logger.log_error(
                            error_msg=f"Error processing batch {i}: {str(e)}",
                            error_type="validation_batch_error"
                        )
                        continue
            
            if total_batches > 0:
                metrics["loss"] = total_loss / total_batches
                for metric in metrics:
                    if metric != "loss":
                        metrics[metric] /= total_batches
            
            return metrics
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Validation error: {str(e)}",
                error_type="validation_error"
            )
            return {}

    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for model input."""
        try:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
                
            # Tokenize and prepare inputs
            texts = [item["text"] for item in batch]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.context.config_manager.get("training.max_seq_length", 512),
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.context.device) for k, v in inputs.items()}
            
            # Prepare labels
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            
            return inputs
            
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Batch preparation error: {str(e)}",
                error_type="batch_preparation_error"
            )
            raise

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
            self.logger.setLevel(logging.DEBUG)
            
        self.checkpoint_interval = args.checkpoint_interval
        self.max_patience = args.max_patience
            
        self._register_signal_handlers()
        atexit.register(self.cleanup)
        
        self.logger.info("Waking SOVL system...")
        self.logger.info(f"Configuration: {args}")
        
        try:
            self.context = self._initialize_context(args)
            self.components = self._initialize_components(self.context)
            self.model = self.components[1]
            
            # Initialize state manager
            self.state_manager = StateManager(
                self.context.config_manager,
                self.logger,
                self.context.device
            )
            self.state_manager.initialize_state()
            
            # Load checkpoint if specified
            optimizer = None  # Note: Optimizer should be initialized in SOVLOrchestrator or passed
            if args.resume_from_checkpoint:
                if not self.load_checkpoint(args.resume_from_checkpoint, optimizer):
                    self.logger.log_error(
                        error_msg="Failed to load checkpoint, starting fresh",
                        error_type="checkpoint_error"
                    )
            
            # Initialize orchestrator
            self.orchestrator = SOVLOrchestrator(
                model=self.model,
                components=self.components,
                context=self.context
            )
            
            # Run system
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
                        checkpoint_callback=lambda: self.save_checkpoint(optimizer=optimizer),
                        validate_every=args.validate_every
                    )
                    
                    # Validation phase
                    if valid_data and (epoch + 1) % args.validate_every == 0:
                        valid_loss, metrics = self.orchestrator.validate(valid_data)
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
                    self.save_checkpoint(optimizer=optimizer)
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
                error_msg=f"System error: {str(e)}",
                error_type="system_error"
            )
            raise
        finally:
            # Save final checkpoint
            self.save_checkpoint(force=True, optimizer=optimizer)
            self.cleanup_old_checkpoints(args.max_checkpoints)
            self.cleanup()

def main():
    """Entry point for the SOVL system."""
    runner = SOVLRunner()
    runner.run()

if __name__ == "__main__":
    main()
