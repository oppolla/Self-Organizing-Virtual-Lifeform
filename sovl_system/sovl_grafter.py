from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import importlib
import os
import sys
import time
import traceback
import hashlib
from dataclasses import dataclass
from threading import Lock
from collections import OrderedDict
import torch
from sovl_logger import Logger, LoggerConfig
from sovl_config import ConfigManager, ConfigSchema
from sovl_state import SOVLState, StateManager
from sovl_error import ErrorManager
from sovl_utils import safe_execute, NumericalGuard

if TYPE_CHECKING:
    from sovl_main import SOVLSystem, SystemContext

""" This is the future plugin manager of the SOVL System"""

class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass

class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""
    pass

class PluginValidationError(PluginError):
    """Raised when a plugin fails validation."""
    pass

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    priority: int = 0  # Lower number = higher priority
    enabled: bool = True
    config_requirements: List[ConfigSchema] = None

@dataclass
class PluginContext:
    """Enhanced context object for plugins with emotional system integration."""
    vibe_profile: Optional[Any] = None  # VibeProfile
    shame_profile: Optional[Any] = None  # ShameProfile
    bond_score: float = 0.5
    thin_ice_level: int = 0
    system_context: Optional['SystemContext'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary format."""
        return {
            "vibe": self.vibe_profile.to_dict() if self.vibe_profile else None,
            "shame": self.shame_profile.to_dict() if self.shame_profile else None,
            "bond_score": self.bond_score,
            "thin_ice_level": self.thin_ice_level
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'PluginContext':
        """Create context from dictionary."""
        return PluginContext(
            vibe_profile=d.get("vibe"),
            shame_profile=d.get("shame"),
            bond_score=d.get("bond_score", 0.5),
            thin_ice_level=d.get("thin_ice_level", 0)
        )

class PluginInterface(ABC):
    """Abstract base class for plugins."""
    
    @abstractmethod
    def initialize(self, system: 'SOVLSystem', context: 'SystemContext') -> None:
        """Initialize plugin with system and context access."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any], *args, **kwargs) -> Any:
        """Execute plugin's main functionality."""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method."""
        pass
    
    def validate(self) -> bool:
        """Validate plugin requirements and compatibility."""
        return True

    def get_state(self) -> Dict[str, Any]:
        """Get plugin state for serialization."""
        try:
            metadata = self.get_metadata()
            return {
                "name": metadata.name,
                "version": metadata.version,
                "enabled": metadata.enabled,
                "state_version": "1.1"  # Updated version
            }
        except Exception as e:
            raise PluginError(f"Plugin state serialization failed: {str(e)}")

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load plugin state."""
        try:
            version = state.get("state_version", "1.0")
            if version not in ["1.0", "1.1"]:
                raise PluginValidationError(f"Unsupported plugin state version: {version}")
        except Exception as e:
            raise PluginError(f"Plugin state loading failed: {str(e)}")

    # New emotional system integration methods
    def on_vibe_change(self, vibe_profile: Any, context: Dict[str, Any]) -> None:
        """Called when system vibe changes."""
        pass
    
    def on_shame_detect(self, shame_profile: Any, context: Dict[str, Any]) -> None:
        """Called when shame is detected."""
        pass
    
    def on_bond_change(self, bond_score: float, context: Dict[str, Any]) -> None:
        """Called when bond score changes."""
        pass
    
    def get_vibe_modifiers(self) -> Dict[str, float]:
        """Return modifiers for vibe calculation."""
        return {}
    
    def get_shame_triggers(self) -> List[str]:
        """Return additional shame triggers."""
        return []
    
    def get_bond_factors(self) -> Dict[str, float]:
        """Return additional bonding factors."""
        return {}

    # New primer-specific methods
    def modify_system_prompt(self, base_prompt: str) -> str:
        """Modify or enhance the system prompt.
        Args:
            base_prompt: The original system prompt
        Returns:
            Modified system prompt
        """
        return base_prompt
    
    def get_generation_parameters(self) -> Dict[str, Any]:
        """Provide custom generation parameters.
        Returns:
            Dictionary of parameter adjustments like temperature, top_p, etc.
        """
        return {}
    
    def pre_prompt_assembly(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before prompt assembly to modify context.
        Args:
            context: The current generation context
        Returns:
            Modified context
        """
        return context
    
    def post_prompt_assembly(self, assembled_prompt: str) -> str:
        """Called after prompt assembly for final modifications.
        Args:
            assembled_prompt: The fully assembled prompt
        Returns:
            Modified prompt
        """
        return assembled_prompt
    
    def get_memory_context(self) -> Optional[str]:
        """Provide additional memory context for generation.
        Returns:
            String of memory context or None
        """
        return None
    
    def get_trait_modifiers(self) -> Dict[str, Dict[str, float]]:
        """Provide modifiers for trait calculations.
        Returns:
            Dictionary of trait modifiers like:
            {
                "curiosity": {"base_boost": 0.1},
                "temperament": {"stability": 0.2}
            }
        """
        return {}

class PluginManager:
    """Manages plugin lifecycle, registration, and execution."""
    
    # Define configuration schema for plugin manager
    SCHEMA = [
        ConfigSchema(
            field="plugin_config.plugin_directory",
            type=str,
            default="plugins",
            validator=lambda x: os.path.isabs(x) or x.strip(),
            required=True
        ),
        ConfigSchema(
            field="plugin_config.enabled_plugins",
            type=list,
            default=[],
            validator=lambda x: all(isinstance(i, str) for i in x),
            nullable=True
        ),
        ConfigSchema(
            field="plugin_config.max_plugins",
            type=int,
            default=10,
            range=(1, 100),
            required=True
        ),
        ConfigSchema(
            field="plugin_config.plugin_timeout",
            type=float,
            default=30.0,
            range=(1.0, 300.0),
            required=True
        ),
        ConfigSchema(
            field="plugin_config.allow_dynamic_loading",
            type=bool,
            default=True,
            required=True
        ),
        ConfigSchema(
            field="plugin_config.log_plugin_errors",
            type=bool,
            default=True,
            required=True
        ),
    ]

    def __init__(
        self,
        context: 'SystemContext',
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorManager,
        state: SOVLState
    ):
        """Initialize with system context and components."""
        self.context = context
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.state = state
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_lock = Lock()
        self.state_version = "1.1"
        self.state_hash = None
        self.system = None

        # Initialize hooks with modern system events
        self.execution_hooks = {
            # Generation hooks
            "pre_generate": [],
            "post_generate": [],
            # Training hooks
            "on_training_step": [],
            "on_training_epoch": [],
            # Lifecycle hooks
            "on_gestation": [],
            "on_dream": [],
            "on_curiosity": [],
            # State hooks
            "on_state_save": [],
            "on_state_load": [],
            # Error hooks
            "on_error": [],
            # Memory hooks
            "on_memory_cleanup": [],
            "on_memory_threshold": [],
            # System hooks
            "on_system_pause": [],
            "on_system_resume": [],
            "on_system_shutdown": [],
            # Vibe hooks
            "pre_vibe_sculpt": [],
            "post_vibe_sculpt": [],
            "on_vibe_change": [],
            # Shame hooks
            "pre_shame_detect": [],
            "post_shame_detect": [],
            "on_thin_ice": [],
            # Bond hooks
            "pre_bond_calculate": [],
            "post_bond_calculate": [],
            "on_bond_change": [],
            # Primer hooks
            "pre_prompt_assembly": [],
            "post_prompt_assembly": [],
            "modify_system_prompt": [],
            "adjust_generation_params": [],
            "provide_memory_context": [],
            "modify_traits": []
        }

        # Register schema with config manager
        self.config_manager.register_schema(self.SCHEMA)
        
        # Load configuration
        self._load_config()
        self._initialize_plugin_directory()
        self._update_state_hash()
        
        self.logger.record({
            "event": "plugin_manager_initialized",
            "plugin_directory": self.plugin_dir,
            "enabled_plugins": self.enabled_plugins,
            "state_hash": self.state_hash,
            "timestamp": time.time()
        })

    def _load_config(self) -> None:
        """Load plugin manager configuration."""
        try:
            self.plugin_dir = self.config_manager.get("plugin_config.plugin_directory", "plugins")
            self.enabled_plugins = self.config_manager.get("plugin_config.enabled_plugins", [])
            self.max_plugins = self.config_manager.get("plugin_config.max_plugins", 10)
            self.plugin_timeout = self.config_manager.get("plugin_config.plugin_timeout", 30.0)
            self.allow_dynamic_loading = self.config_manager.get("plugin_config.allow_dynamic_loading", True)
            self.log_plugin_errors = self.config_manager.get("plugin_config.log_plugin_errors", True)
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_config_load",
                error_message=f"Failed to load plugin configuration: {str(e)}",
                error=e
            )
            raise PluginError(f"Failed to load plugin configuration: {str(e)}")

    def set_system(self, system: 'SOVLSystem') -> None:
        """
        Set the SOVLSystem reference for the plugin manager.
        
        Args:
            system: The SOVLSystem instance
        """
        self.system = system
        self.logger.record({
            "event": "plugin_manager_system_set",
            "timestamp": time.time(),
            "conversation_id": self.state.history.conversation_id
        })

    def _initialize_plugin_directory(self) -> None:
        """Ensure plugin directory exists."""
        try:
            with self.plugin_lock:
                os.makedirs(self.plugin_dir, exist_ok=True)
                self.logger.record({
                    "event": "plugin_directory_initialized",
                    "directory": self.plugin_dir,
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to initialize plugin directory: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id
            })
            raise PluginError(f"Failed to initialize plugin directory: {str(e)}")

    def _update_state_hash(self) -> None:
        """Compute a hash of plugin manager state."""
        try:
            state_str = (
                f"{len(self.plugins)}:{','.join(sorted(self.plugins.keys()))}:"
                f"{sum(len(hooks) for hooks in self.execution_hooks.values())}"
            )
            self.state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.record({
                "error": f"Plugin manager state hash update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.state.history.conversation_id
            })

    def register_plugin(self, plugin: PluginInterface) -> bool:
        """Register a plugin with validation."""
        with self.plugin_lock:
            try:
                if len(self.plugins) >= self.max_plugins:
                    self.logger.record({
                        "error": f"Maximum plugin limit ({self.max_plugins}) reached",
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                    return False

                if not plugin.validate():
                    raise PluginValidationError(f"Plugin {plugin.__class__.__name__} failed validation")
                
                metadata = plugin.get_metadata()
                if metadata.name in self.plugins:
                    self.logger.record({
                        "warning": f"Plugin {metadata.name} already registered",
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                    return False
                
                # Register plugin-specific configuration schema
                if metadata.config_requirements:
                    self.config_manager.register_schema(metadata.config_requirements)
                    for schema in metadata.config_requirements:
                        if not self.config_manager.get(schema.field, None):
                            self.config_manager.update(schema.field, schema.default)

                self.plugins[metadata.name] = plugin
                self._update_state_hash()
                self.logger.record({
                    "event": "plugin_registered",
                    "plugin_name": metadata.name,
                    "version": metadata.version,
                    "priority": metadata.priority,
                    "state_hash": self.state_hash,
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return True
            except Exception as e:
                self.logger.record({
                    "error": f"Plugin registration failed: {str(e)}",
                    "plugin_class": plugin.__class__.__name__,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
                return False

    def load_plugins(self, system: 'SOVLSystem', max_retries: int = 3) -> int:
        """Load plugins with improved error handling."""
        loaded_count = 0
        with self.plugin_lock:
            for plugin_name in self.enabled_plugins:
                for attempt in range(max_retries):
                    try:
                        plugin = self._load_plugin_module(plugin_name)
                        if plugin:
                            with NumericalGuard():
                                # Pass both system and context
                                plugin.initialize(system, self.context)
                            if self.register_plugin(plugin):
                                loaded_count += 1
                                break
                    except Exception as e:
                        self.error_manager.handle_error(
                            error_type="plugin_load_error",
                            error_message=f"Attempt {attempt + 1} failed to load plugin {plugin_name}: {str(e)}",
                            error_context={
                                "plugin_name": plugin_name,
                                "attempt": attempt + 1
                            },
                            error=e
                        )
                        if attempt == max_retries - 1:
                            self.logger.record({
                                "warning": f"Plugin {plugin_name} failed to load after {max_retries} attempts",
                                "timestamp": time.time()
                            })
                        time.sleep(0.1)
            self._update_state_hash()
            self.logger.record({
                "event": "plugin_load_complete",
                "loaded_count": loaded_count,
                "total_attempted": len(self.enabled_plugins),
                "state_hash": self.state_hash,
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id
            })
        return loaded_count

    def _load_plugin_module(self, plugin_name: str) -> Optional[PluginInterface]:
        """Dynamically load a plugin module."""
        if not self.allow_dynamic_loading:
            self.logger.record({
                "warning": f"Dynamic loading disabled for plugin {plugin_name}",
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id
            })
            return None

        try:
            module_path = os.path.join(self.plugin_dir, plugin_name)
            if not os.path.exists(module_path):
                raise PluginLoadError(f"Plugin directory {module_path} does not exist")
            
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}",
                os.path.join(module_path, "__init__.py")
            )
            if spec is None:
                raise PluginLoadError(f"Failed to create spec for plugin {plugin_name}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{plugin_name}"] = module
            spec.loader.exec_module(module)
            
            plugin_class = getattr(module, "Plugin", None)
            if not plugin_class or not issubclass(plugin_class, PluginInterface):
                raise PluginValidationError(f"Plugin {plugin_name} does not implement PluginInterface")
            
            return plugin_class()
        except Exception as e:
            if self.log_plugin_errors:
                self.logger.record({
                    "error": f"Plugin module load failed for {plugin_name}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
            return None

    def register_hook(self, hook_name: str, callback: Callable, plugin_name: str, priority: int = 0) -> bool:
        """Register a callback for a specific hook with improved error handling."""
        if hook_name not in self.execution_hooks:
            self.error_manager.handle_error(
                error_type="hook_registration",
                error_message=f"Invalid hook name: {hook_name}",
                error_context={
                    "hook_name": hook_name,
                    "plugin_name": plugin_name
                }
            )
            return False
        
        with self.plugin_lock:
            try:
                self.execution_hooks[hook_name].append({
                    "callback": callback,
                    "plugin_name": plugin_name,
                    "priority": priority
                })
                self.execution_hooks[hook_name].sort(key=lambda x: x["priority"])
                self._update_state_hash()
                self.logger.record({
                    "event": "hook_registered",
                    "hook_name": hook_name,
                    "plugin_name": plugin_name,
                    "priority": priority,
                    "state_hash": self.state_hash,
                    "timestamp": time.time()
                })
                return True
            except Exception as e:
                self.error_manager.handle_error(
                    error_type="hook_registration",
                    error_message=f"Hook registration failed: {str(e)}",
                    error_context={
                        "hook_name": hook_name,
                        "plugin_name": plugin_name,
                        "priority": priority
                    },
                    error=e
                )
                return False

    def execute_hook(self, hook_name: str, context: Dict[str, Any], *args, **kwargs) -> List[Any]:
        """Execute all callbacks registered for a hook with improved error handling."""
        results = []
        if hook_name not in self.execution_hooks:
            return results
        
        with self.plugin_lock:
            for hook in self.execution_hooks[hook_name]:
                try:
                    start_time = time.time()
                    with NumericalGuard():
                        result = safe_execute(
                            hook["callback"],
                            args=(context,) + args,
                            kwargs=kwargs,
                            logger=self.logger,
                            timeout=self.plugin_timeout
                        )
                    elapsed = time.time() - start_time
                    
                    if elapsed > self.plugin_timeout:
                        self.error_manager.handle_error(
                            error_type="hook_timeout",
                            error_message=f"Hook execution exceeded timeout",
                            error_context={
                                "hook_name": hook_name,
                                "plugin_name": hook["plugin_name"],
                                "elapsed": elapsed,
                                "timeout": self.plugin_timeout
                            }
                        )
                    
                    results.append(result)
                    self.logger.record({
                        "event": "hook_executed",
                        "hook_name": hook_name,
                        "plugin_name": hook["plugin_name"],
                        "result": str(result)[:200],  # Truncate for logging
                        "elapsed": elapsed,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="hook_execution",
                        error_message=f"Hook execution failed: {str(e)}",
                        error_context={
                            "hook_name": hook_name,
                            "plugin_name": hook["plugin_name"]
                        },
                        error=e
                    )
        return results

    def execute_plugin(self, plugin_name: str, context: Dict[str, Any], *args, **kwargs) -> Any:
        """Execute a specific plugin."""
        with self.plugin_lock:
            plugin = self.plugins.get(plugin_name)
            if not plugin:
                self.logger.record({
                    "warning": f"Plugin {plugin_name} not found",
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return None
            
            try:
                start_time = time.time()
                with NumericalGuard():
                    result = safe_execute(
                        plugin.execute,
                        args=(context,) + args,
                        kwargs=kwargs,
                        logger=self.logger,
                        timeout=self.plugin_timeout
                    )
                elapsed = time.time() - start_time
                if elapsed > self.plugin_timeout:
                    self.logger.record({
                        "warning": f"Plugin execution for {plugin_name} exceeded timeout ({self.plugin_timeout}s)",
                        "elapsed": elapsed,
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                self.logger.record({
                    "event": "plugin_executed",
                    "plugin_name": plugin_name,
                    "result": str(result)[:200],
                    "elapsed": elapsed,
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return result
            except Exception as e:
                if self.log_plugin_errors:
                    self.logger.record({
                        "error": f"Plugin execution failed for {plugin_name}: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id
                    })
                return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin."""
        with self.plugin_lock:
            plugin = self.plugins.get(plugin_name)
            if not plugin:
                self.logger.record({
                    "warning": f"Plugin {plugin_name} not found",
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return False
            
            try:
                with NumericalGuard():
                    plugin.cleanup()
                del self.plugins[plugin_name]
                
                # Remove plugin's hooks
                for hook_name in self.execution_hooks:
                    self.execution_hooks[hook_name] = [
                        hook for hook in self.execution_hooks[hook_name]
                        if hook["plugin_name"] != plugin_name
                    ]
                
                self._update_state_hash()
                self.logger.record({
                    "event": "plugin_unloaded",
                    "plugin_name": plugin_name,
                    "state_hash": self.state_hash,
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return True
            except Exception as e:
                if self.log_plugin_errors:
                    self.logger.record({
                        "error": f"Plugin unload failed for {plugin_name}: {str(e)}",
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc(),
                        "conversation_id": self.state.history.conversation_id
                    })
                return False

    def cleanup(self) -> None:
        """Cleanup all plugins."""
        with self.plugin_lock:
            try:
                for plugin_name in list(self.plugins.keys()):
                    self.unload_plugin(plugin_name)
                self.plugins.clear()
                self.execution_hooks = {k: [] for k in self.execution_hooks}
                self._update_state_hash()
                self.logger.record({
                    "event": "plugin_manager_cleanup",
                    "state_hash": self.state_hash,
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
            except Exception as e:
                self.logger.record({
                    "error": f"Plugin manager cleanup failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })

    def get_plugin_metadata(self) -> Dict[str, PluginMetadata]:
        """Return metadata for all loaded plugins."""
        with self.plugin_lock:
            return {name: plugin.get_metadata() for name, plugin in self.plugins.items()}

    def validate_plugin_config(self, plugin_name: str) -> bool:
        """Validate plugin configuration requirements."""
        with self.plugin_lock:
            plugin = self.plugins.get(plugin_name)
            if not plugin:
                self.logger.record({
                    "warning": f"Plugin {plugin_name} not found for config validation",
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return False
            
            metadata = plugin.get_metadata()
            if not metadata.config_requirements:
                return True
            
            try:
                for schema in metadata.config_requirements:
                    value = self.config_manager.get(schema.field, None)
                    if value is None and schema.required:
                        self.logger.record({
                            "warning": f"Missing required config for plugin {plugin_name}: {schema.field}",
                            "suggested": f"Default: {schema.default}",
                            "timestamp": time.time(),
                            "conversation_id": self.state.history.conversation_id
                        })
                        return False
                    if value is not None:
                        if not isinstance(value, schema.type):
                            self.logger.record({
                                "warning": f"Invalid type for {schema.field}: expected {schema.type.__name__}, got {type(value).__name__}",
                                "timestamp": time.time(),
                                "conversation_id": self.state.history.conversation_id
                            })
                            return False
                        if schema.validator and not schema.validator(value):
                            self.logger.record({
                                "warning": f"Invalid value for {schema.field}: {value}",
                                "timestamp": time.time(),
                                "conversation_id": self.state.history.conversation_id
                            })
                            return False
                        if schema.range and not (schema.range[0] <= value <= schema.range[1]):
                            self.logger.record({
                                "warning": f"Value for {schema.field} out of range {schema.range}: {value}",
                                "timestamp": time.time(),
                                "conversation_id": self.state.history.conversation_id
                            })
                            return False
                return True
            except Exception as e:
                self.logger.record({
                    "error": f"Plugin config validation failed for {plugin_name}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
                return False

    def update_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Update plugin-specific configuration."""
        with self.plugin_lock:
            try:
                updates = {f"plugin_config.{plugin_name}.{key}": value for key, value in config.items()}
                success = self.config_manager.update_batch(updates, rollback_on_failure=True)
                if success:
                    self._update_state_hash()
                    self.logger.record({
                        "event": "plugin_config_updated",
                        "plugin_name": plugin_name,
                        "config_keys": list(config.keys()),
                        "state_hash": self.state_hash,
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                return success
            except Exception as e:
                self.logger.record({
                    "error": f"Plugin config update failed for {plugin_name}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
                return False

    def to_dict(self, max_retries: int = 3) -> Dict[str, Any]:
        """Serialize plugin manager state to dictionary."""
        for attempt in range(max_retries):
            try:
                with self.plugin_lock:
                    state_dict = {
                        "version": self.state_version,
                        "state_hash": self.state_hash,
                        "plugins": {
                            name: plugin.to_dict()
                            for name, plugin in self.plugins.items()
                        },
                        "enabled_plugins": self.enabled_plugins,
                        "execution_hooks": {
                            name: [
                                {
                                    "plugin_name": hook["plugin_name"],
                                    "priority": hook["priority"]
                                }
                                for hook in hooks
                            ]
                            for name, hooks in self.execution_hooks.items()
                        }
                    }
                    self.logger.record({
                        "event": "plugin_manager_state_serialized",
                        "plugin_count": len(self.plugins),
                        "state_hash": self.state_hash,
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                    return state_dict
            except Exception as e:
                self.logger.record({
                    "error": f"Plugin manager state serialization failed on attempt {attempt + 1}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
                if attempt == max_retries - 1:
                    raise PluginError(f"Plugin manager state serialization failed: {str(e)}")
                time.sleep(0.1)

    def from_dict(self, data: Dict[str, Any], system: 'SOVLSystem', max_retries: int = 3) -> None:
        """Load plugin manager state from dictionary."""
        for attempt in range(max_retries):
            try:
                with self.plugin_lock:
                    version = data.get("version", "1.0")
                    if version != self.state_version:
                        self.logger.record({
                            "warning": f"Plugin manager state version mismatch: expected {self.state_version}, got {version}",
                            "timestamp": time.time(),
                            "conversation_id": self.state.history.conversation_id
                        })

                    # Clear existing plugins
                    self.cleanup()

                    # Load enabled plugins
                    self.enabled_plugins = data.get("enabled_plugins", self.enabled_plugins)
                    self.config_manager.update(
                        "plugin_config.enabled_plugins",
                        self.enabled_plugins
                    )

                    # Load plugins
                    for plugin_data in data.get("plugins", {}).values():
                        plugin_name = plugin_data.get("name")
                        if plugin_name:
                            plugin = self._load_plugin_module(plugin_name)
                            if plugin:
                                plugin.initialize(system, self.context)
                                if self.register_plugin(plugin):
                                    plugin.load_state(plugin_data)

                    # Restore hooks (callbacks are re-registered during initialize)
                    self.execution_hooks = {
                        name: [
                            {
                                "callback": None,  # Callbacks restored by plugins
                                "plugin_name": hook["plugin_name"],
                                "priority": hook["priority"]
                            }
                            for hook in hooks
                        ]
                        for name, hooks in data.get("execution_hooks", {}).items()
                    }

                    self._update_state_hash()
                    self.logger.record({
                        "event": "plugin_manager_state_loaded",
                        "plugin_count": len(self.plugins),
                        "state_hash": self.state_hash,
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                    break
            except Exception as e:
                self.logger.record({
                    "error": f"Plugin manager state loading failed on attempt {attempt + 1}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
                if attempt == max_retries - 1:
                    raise PluginError(f"Plugin manager state loading failed: {str(e)}")
                time.sleep(0.1)

# Example plugin implementation
class ExamplePlugin(PluginInterface):
    def __init__(self):
        self.system = None
        self.context = None
        self.state_version = "1.1"
        self.logger = None
        self.error_manager = None
    
    def initialize(self, system: 'SOVLSystem', context: 'SystemContext') -> None:
        """Initialize plugin with system and context access."""
        self.system = system
        self.context = context
        self.logger = context.logger
        self.error_manager = context.error_handler
        
        try:
            # Register for core system hooks
            self.system.plugin_manager.register_hook(
                "pre_generate",
                self.pre_generate_hook,
                plugin_name="example_plugin",
                priority=10
            )
            
            # Register for emotional system hooks
            hook_registrations = [
                ("pre_vibe_sculpt", self.pre_vibe_hook),
                ("on_shame_detect", self.shame_hook),
                ("on_bond_change", self.bond_hook),
                ("on_thin_ice", self.thin_ice_hook)
            ]
            
            # Register for primer hooks
            primer_hooks = [
                ("pre_prompt_assembly", self.enhance_context),
                ("modify_system_prompt", self.enhance_system_prompt),
                ("provide_memory_context", self.provide_custom_memory),
                ("adjust_generation_params", self.adjust_generation_params),
                ("modify_traits", self.modify_traits)
            ]
            hook_registrations.extend(primer_hooks)
            
            for hook_name, callback in hook_registrations:
                self.system.plugin_manager.register_hook(
                    hook_name,
                    callback,
                    plugin_name="example_plugin",
                    priority=10
                )
            
            self.logger.record({
                "event": "example_plugin_initialized",
                "timestamp": time.time()
            })
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_initialization",
                error_message=f"Failed to initialize example plugin: {str(e)}",
                error=e
            )
            raise PluginError(f"Plugin initialization failed: {str(e)}")
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example_plugin",
            version="1.1.0",
            description="Example plugin for SOVLSystem with emotional system and primer integration",
            author="xAI",
            dependencies=[],
            priority=10,
            config_requirements=[
                ConfigSchema(
                    field="plugin_config.example_plugin.enabled",
                    type=bool,
                    default=True,
                    required=True
                ),
                ConfigSchema(
                    field="plugin_config.example_plugin.mode",
                    type=str,
                    default="default",
                    validator=lambda x: x in ["default", "advanced"],
                    required=True
                )
            ]
        )
    
    def execute(self, context: Dict[str, Any], *args, **kwargs) -> Any:
        """Execute plugin's main functionality."""
        try:
            with NumericalGuard():
                # Example: Access system components through context
                ram_usage = self.context.ram_manager.get_usage()
                gpu_usage = self.context.gpu_manager.get_usage()
                
                result = {
                    "status": "executed",
                    "context": context,
                    "system_stats": {
                        "ram_usage": ram_usage,
                        "gpu_usage": gpu_usage
                    }
                }
                
                self.logger.record({
                    "event": "example_plugin_executed",
                    "result": str(result)[:200],  # Truncate for logging
                    "timestamp": time.time()
                })
                
                return result
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_execution",
                error_message=f"Failed to execute example plugin: {str(e)}",
                error=e
            )
            return None

    # Primer-specific hook implementations
    def enhance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-prompt assembly hook to enhance context."""
        try:
            # Add custom context enhancements
            context["enhanced"] = True
            context["plugin_context"] = {
                "last_execution": time.time(),
                "system_state": "stable"
            }
            return context
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_context_enhancement",
                error_message=f"Failed to enhance context: {str(e)}",
                error=e
            )
            return context

    def enhance_system_prompt(self, base_prompt: str) -> str:
        """Modify system prompt hook."""
        try:
            # Add custom instructions based on system state
            state = self.system.state_manager.get_state()
            if hasattr(state, 'confidence') and state.confidence > 0.8:
                return f"{base_prompt}\nEnhanced with high confidence context awareness."
            return base_prompt
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_prompt_enhancement",
                error_message=f"Failed to enhance system prompt: {str(e)}",
                error=e
            )
            return base_prompt

    def provide_custom_memory(self) -> Optional[str]:
        """Memory context hook."""
        try:
            if self.context.dialogue_context_manager:
                recent_context = self.context.dialogue_context_manager.get_recent_context(n=1)
                if recent_context:
                    return f"Previous interaction theme: {recent_context[0].get('theme', 'general')}"
            return None
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_memory_context",
                error_message=f"Failed to provide memory context: {str(e)}",
                error=e
            )
            return None

    def adjust_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generation parameters hook."""
        try:
            # Adjust parameters based on system state
            state = self.system.state_manager.get_state()
            if hasattr(state, 'confidence'):
                if state.confidence > 0.8:
                    params["temperature"] = max(0.1, params.get("temperature", 0.7) - 0.1)
                elif state.confidence < 0.3:
                    params["temperature"] = min(1.0, params.get("temperature", 0.7) + 0.1)
            return params
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_param_adjustment",
                error_message=f"Failed to adjust generation parameters: {str(e)}",
                error=e
            )
            return params

    def modify_traits(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Trait modification hook."""
        try:
            # Modify traits based on plugin logic
            if "curiosity" in traits:
                traits["curiosity"] = min(1.0, traits["curiosity"] * 1.1)
            if "confidence" in traits:
                traits["confidence"] = max(0.1, min(1.0, traits["confidence"] + 0.05))
            return traits
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_trait_modification",
                error_message=f"Failed to modify traits: {str(e)}",
                error=e
            )
            return traits

    # Existing emotional system hooks...
    def pre_generate_hook(self, context: Dict[str, Any]) -> None:
        """Example hook implementation."""
        try:
            self.logger.record({
                "event": "example_plugin_pre_generate",
                "context": str(context)[:200],
                "timestamp": time.time()
            })
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_hook_execution",
                error_message=f"Failed to execute pre_generate hook: {str(e)}",
                error=e
            )
    
    def pre_vibe_hook(self, context: Dict[str, Any]) -> None:
        """Modify vibe calculation."""
        try:
            vibe_profile = context.get('vibe_profile')
            if vibe_profile and hasattr(vibe_profile, 'dimensions'):
                # Example: Boost energy if bond is strong
                bond_score = context.get('bond_score', 0.5)
                if bond_score > 0.7:
                    vibe_profile.dimensions['energy_base_energy'] = min(
                        1.0,
                        vibe_profile.dimensions.get('energy_base_energy', 0.5) * 1.1
                    )
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_vibe_hook",
                error_message=f"Failed to execute vibe hook: {str(e)}",
                error=e
            )
    
    def shame_hook(self, context: Dict[str, Any]) -> None:
        """React to shame detection."""
        try:
            shame_profile = context.get('shame_profile')
            if shame_profile and hasattr(shame_profile, 'frustration_score'):
                if shame_profile.frustration_score > 0.8:
                    self.logger.record({
                        "event": "high_frustration_detected",
                        "message": "Critical frustration level detected",
                        "level": "warning",
                        "timestamp": time.time()
                    })
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_shame_hook",
                error_message=f"Failed to execute shame hook: {str(e)}",
                error=e
            )
    
    def bond_hook(self, context: Dict[str, Any]) -> None:
        """React to bond changes."""
        try:
            bond_score = context.get('bond_score', 0.5)
            if bond_score > 0.9:
                self.logger.record({
                    "event": "strong_bond_achieved",
                    "message": "Strong bond established",
                    "level": "info",
                    "timestamp": time.time()
                })
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_bond_hook",
                error_message=f"Failed to execute bond hook: {str(e)}",
                error=e
            )
    
    def thin_ice_hook(self, context: Dict[str, Any]) -> None:
        """React to thin ice state changes."""
        try:
            thin_ice_level = context.get('thin_ice_level', 0)
            if thin_ice_level > 2:
                self.logger.record({
                    "event": "critical_thin_ice",
                    "message": "System in critical thin ice state",
                    "level": "warning",
                    "timestamp": time.time(),
                    "thin_ice_level": thin_ice_level
                })
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_thin_ice_hook",
                error_message=f"Failed to execute thin ice hook: {str(e)}",
                error=e
            )
    
    def get_vibe_modifiers(self) -> Dict[str, float]:
        """Provide custom vibe modifiers."""
        return {
            "energy_boost": 0.1,
            "flow_dampen": -0.05
        }
    
    def get_shame_triggers(self) -> List[str]:
        """Provide additional shame triggers."""
        return [
            "completely wrong",
            "not helping at all",
            "waste of time"
        ]
    
    def get_bond_factors(self) -> Dict[str, float]:
        """Provide custom bonding factors."""
        return {
            "empathy_weight": 0.3,
            "consistency_weight": 0.2
        }
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        try:
            if self.logger:
                self.logger.record({
                    "event": "example_plugin_cleanup",
                    "timestamp": time.time()
                })
            self.system = None
            self.context = None
            self.logger = None
            self.error_manager = None
        except Exception as e:
            if self.error_manager:
                self.error_manager.handle_error(
                    error_type="plugin_cleanup",
                    error_message=f"Failed to cleanup example plugin: {str(e)}",
                    error=e
                )

def initialize_plugin_manager(system: 'SOVLSystem') -> PluginManager:
    """Initialize and return a plugin manager instance."""
    plugin_manager = PluginManager(
        context=system.context,
        config_manager=system.config_manager,
        logger=system.logger,
        error_manager=system.error_manager,
        state=system.state
    )
    plugin_manager.load_plugins(system)
    return plugin_manager
