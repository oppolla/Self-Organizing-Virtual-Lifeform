from typing import Dict, List, Optional, Callable, Any
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
from sovl_state import SOVLState
from sovl_utils import safe_execute, NumericalGuard


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

class PluginInterface(ABC):
    """Abstract base class for plugins."""
    
    @abstractmethod
    def initialize(self, system: 'SOVLSystem') -> None:
        """Initialize the plugin with access to the SOVLSystem."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any], *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method."""
        pass
    
    def validate(self) -> bool:
        """Validate plugin requirements and compatibility."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize plugin state to dictionary."""
        try:
            metadata = self.get_metadata()
            return {
                "name": metadata.name,
                "version": metadata.version,
                "enabled": metadata.enabled,
                "state_version": "1.0"
            }
        except Exception as e:
            raise PluginError(f"Plugin serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load plugin state from dictionary."""
        try:
            version = data.get("state_version", "1.0")
            if version != "1.0":
                raise PluginValidationError(f"Unsupported plugin state version: {version}")
            # Plugins can override to restore custom state
        except Exception as e:
            raise PluginError(f"Plugin state loading failed: {str(e)}")

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

    def __init__(self, config_manager: ConfigManager, logger: Logger, state: SOVLState):
        """
        Initialize PluginManager with configuration, logging, and state.

        Args:
            config_manager: ConfigManager instance for parameters
            logger: Logger instance for recording events
            state: SOVLState instance for system state
        """
        self.config_manager = config_manager
        self.logger = logger
        self.state = state
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_lock = Lock()
        self.state_version = "1.1"
        self.state_hash = None
        self.system = None  # Will be set when system is initialized

        # Register schema
        self.config_manager.register_schema(self.SCHEMA)
        
        # Load configuration with validation
        self.plugin_dir = self.config_manager.get("plugin_config.plugin_directory", "plugins")
        self.enabled_plugins = self.config_manager.get("plugin_config.enabled_plugins", [])
        self.max_plugins = self.config_manager.get("plugin_config.max_plugins", 10)
        self.plugin_timeout = self.config_manager.get("plugin_config.plugin_timeout", 30.0)
        self.allow_dynamic_loading = self.config_manager.get("plugin_config.allow_dynamic_loading", True)
        self.log_plugin_errors = self.config_manager.get("plugin_config.log_plugin_errors", True)

        # Initialize hooks
        self.execution_hooks = {
            "pre_generate": [],
            "post_generate": [],
            "on_training_step": [],
            "on_gestation": [],
            "on_dream": [],
            "on_curiosity": [],
            "on_error": [],
            "on_state_save": [],
            "on_state_load": []
        }

        self._initialize_plugin_directory()
        self._update_state_hash()
        self.logger.record({
            "event": "plugin_manager_initialized",
            "plugin_directory": self.plugin_dir,
            "enabled_plugins": self.enabled_plugins,
            "state_hash": self.state_hash,
            "timestamp": time.time(),
            "conversation_id": self.state.history.conversation_id
        })

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
        """Load plugins from the plugin directory with retry logic."""
        loaded_count = 0
        with self.plugin_lock:
            for plugin_name in self.enabled_plugins:
                for attempt in range(max_retries):
                    try:
                        plugin = self._load_plugin_module(plugin_name)
                        if plugin:
                            with NumericalGuard():
                                plugin.initialize(system)
                            if self.register_plugin(plugin):
                                loaded_count += 1
                                break
                    except Exception as e:
                        self.logger.record({
                            "error": f"Attempt {attempt + 1} failed to load plugin {plugin_name}: {str(e)}",
                            "timestamp": time.time(),
                            "stack_trace": traceback.format_exc(),
                            "conversation_id": self.state.history.conversation_id
                        })
                        if attempt == max_retries - 1:
                            self.logger.record({
                                "warning": f"Plugin {plugin_name} failed to load after {max_retries} attempts",
                                "timestamp": time.time(),
                                "conversation_id": self.state.history.conversation_id
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
        """Register a callback for a specific hook."""
        if hook_name not in self.execution_hooks:
            self.logger.record({
                "warning": f"Invalid hook name: {hook_name}",
                "timestamp": time.time(),
                "conversation_id": self.state.history.conversation_id
            })
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
                    "timestamp": time.time(),
                    "conversation_id": self.state.history.conversation_id
                })
                return True
            except Exception as e:
                self.logger.record({
                    "error": f"Hook registration failed: {str(e)}",
                    "hook_name": hook_name,
                    "plugin_name": plugin_name,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.state.history.conversation_id
                })
                return False

    def execute_hook(self, hook_name: str, context: Dict[str, Any], *args, **kwargs) -> List[Any]:
        """Execute all callbacks registered for a hook."""
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
                        self.logger.record({
                            "warning": f"Hook execution for {hook['plugin_name']} exceeded timeout ({self.plugin_timeout}s)",
                            "hook_name": hook_name,
                            "elapsed": elapsed,
                            "timestamp": time.time(),
                            "conversation_id": self.state.history.conversation_id
                        })
                    results.append(result)
                    self.logger.record({
                        "event": "hook_executed",
                        "hook_name": hook_name,
                        "plugin_name": hook["plugin_name"],
                        "result": str(result)[:200],  # Truncate for logging
                        "elapsed": elapsed,
                        "timestamp": time.time(),
                        "conversation_id": self.state.history.conversation_id
                    })
                except Exception as e:
                    if self.log_plugin_errors:
                        self.logger.record({
                            "error": f"Hook execution failed for {hook['plugin_name']}: {str(e)}",
                            "hook_name": hook_name,
                            "timestamp": time.time(),
                            "stack_trace": traceback.format_exc(),
                            "conversation_id": self.state.history.conversation_id
                        })
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
                                plugin.initialize(system)
                                if self.register_plugin(plugin):
                                    plugin.from_dict(plugin_data)

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
        self.state_version = "1.0"
    
    def initialize(self, system: 'SOVLSystem') -> None:
        self.system = system
        self.system.plugin_manager.register_hook(
            "pre_generate",
            self.pre_generate_hook,
            plugin_name="example_plugin",
            priority=10
        )
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for SOVLSystem",
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
        with NumericalGuard():
            return {"status": "executed", "context": context}
    
    def pre_generate_hook(self, context: Dict[str, Any]) -> None:
        self.system.logger.record({
            "event": "example_plugin_pre_generate",
            "context": str(context)[:200],
            "timestamp": time.time(),
            "conversation_id": self.system.state.history.conversation_id
        })
    
    def cleanup(self) -> None:
        self.system = None

def initialize_plugin_manager(system: 'SOVLSystem') -> PluginManager:
    """Initialize and return a plugin manager instance."""
    plugin_manager = PluginManager(system.config_manager, system.logger, system.state)
    plugin_manager.load_plugins(system)
    return plugin_manager
