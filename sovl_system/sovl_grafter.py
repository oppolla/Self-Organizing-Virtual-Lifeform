from typing import Dict, List, Optional, Any, TYPE_CHECKING
from abc import ABC, abstractmethod
import importlib
import os
import time
from dataclasses import dataclass
from threading import Lock
from sovl_logger import Logger
from sovl_config import ConfigManager, ConfigSchema
from sovl_state import SOVLState
from sovl_error import ErrorManager
from sovl_utils import safe_execute

if TYPE_CHECKING:
    from sovl_main import SOVLSystem, SystemContext

class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    enabled: bool = True

class PluginInterface(ABC):
    """Abstract base class for primer-focused plugins."""
    
    @abstractmethod
    def initialize(self, system: 'SOVLSystem', context: 'SystemContext') -> None:
        """Initialize plugin with system and context access."""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def cleanup(self) -> None:
        """Optional cleanup method."""
        pass

    # Primer-specific methods
    def modify_system_prompt(self, base_prompt: str) -> str:
        """Modify or enhance the system prompt."""
        return base_prompt
    
    def get_generation_parameters(self) -> Dict[str, Any]:
        """Provide custom generation parameters."""
        return {}
    
    def pre_prompt_assembly(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before prompt assembly to modify context."""
        return context
    
    def post_prompt_assembly(self, assembled_prompt: str) -> str:
        """Called after prompt assembly for final modifications."""
        return assembled_prompt
    
    def get_memory_context(self) -> Optional[str]:
        """Provide additional memory context for generation."""
        return None
    
    def get_trait_modifiers(self) -> Dict[str, Dict[str, float]]:
        """Provide modifiers for trait calculations."""
        return {}

class PluginManager:
    """Manages primer-focused plugins."""
    
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
        )
    ]

    def __init__(
        self,
        context: 'SystemContext',
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorManager,
        state: SOVLState
    ):
        self.context = context
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.state = state
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_lock = Lock()
        
        # Initialize primer-specific hooks
        self.execution_hooks = {
            "pre_prompt_assembly": [],
            "post_prompt_assembly": [],
            "modify_system_prompt": [],
            "adjust_generation_params": [],
            "provide_memory_context": [],
            "modify_traits": []
        }

        # Register schema and load config
        self.config_manager.register_schema(self.SCHEMA)
        self._load_config()
        self._initialize_plugin_directory()

    def _load_config(self) -> None:
        """Load plugin manager configuration."""
        try:
            self.plugin_dir = self.config_manager.get("plugin_config.plugin_directory", "plugins")
            self.enabled_plugins = self.config_manager.get("plugin_config.enabled_plugins", [])
        except Exception as e:
            raise PluginError(f"Failed to load plugin configuration: {str(e)}")

    def _initialize_plugin_directory(self) -> None:
        """Ensure plugin directory exists."""
        try:
            os.makedirs(self.plugin_dir, exist_ok=True)
        except Exception as e:
            raise PluginError(f"Failed to initialize plugin directory: {str(e)}")

    def register_plugin(self, plugin: PluginInterface) -> bool:
        """Register a plugin."""
        with self.plugin_lock:
            try:
                metadata = plugin.get_metadata()
                if metadata.name in self.plugins:
                    return False
                
                self.plugins[metadata.name] = plugin
                return True
            except Exception as e:
                self.error_manager.handle_error(
                    error_type="plugin_registration",
                    error_message=f"Plugin registration failed: {str(e)}",
                    error=e
                )
                return False

    def load_plugins(self, system: 'SOVLSystem') -> int:
        """Load enabled plugins."""
        loaded_count = 0
        with self.plugin_lock:
            for plugin_name in self.enabled_plugins:
                try:
                    plugin = self._load_plugin_module(plugin_name)
                    if plugin:
                        plugin.initialize(system, self.context)
                        if self.register_plugin(plugin):
                            loaded_count += 1
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="plugin_load_error",
                        error_message=f"Failed to load plugin {plugin_name}: {str(e)}",
                        error=e
                    )
        return loaded_count

    def _load_plugin_module(self, plugin_name: str) -> Optional[PluginInterface]:
        """Load a plugin module."""
        try:
            module_path = os.path.join(self.plugin_dir, plugin_name)
            if not os.path.exists(module_path):
                raise PluginError(f"Plugin directory {module_path} does not exist")
            
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}",
                os.path.join(module_path, "__init__.py")
            )
            if not spec or not spec.loader:
                raise PluginError(f"Failed to create spec for plugin {plugin_name}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            plugin_class = getattr(module, "Plugin", None)
            if not plugin_class or not issubclass(plugin_class, PluginInterface):
                raise PluginError(f"Plugin {plugin_name} does not implement PluginInterface")
            
            return plugin_class()
        except Exception as e:
            self.error_manager.handle_error(
                error_type="plugin_load",
                error_message=f"Failed to load plugin module {plugin_name}: {str(e)}",
                error=e
            )
            return None

    def modify_system_prompt(self, base_prompt: str) -> str:
        """Apply all plugin modifications to system prompt."""
        with self.plugin_lock:
            modified_prompt = base_prompt
            for plugin in self.plugins.values():
                try:
                    modified_prompt = safe_execute(
                        plugin.modify_system_prompt,
                        args=(modified_prompt,),
                        logger=self.logger
                    ) or modified_prompt
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="prompt_modification",
                        error_message=f"Failed to modify prompt: {str(e)}",
                        error=e
                    )
            return modified_prompt

    def get_combined_generation_parameters(self) -> Dict[str, Any]:
        """Combine generation parameters from all plugins."""
        with self.plugin_lock:
            combined_params = {}
            for plugin in self.plugins.values():
                try:
                    params = safe_execute(
                        plugin.get_generation_parameters,
                        logger=self.logger
                    ) or {}
                    combined_params.update(params)
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="generation_params",
                        error_message=f"Failed to get generation parameters: {str(e)}",
                        error=e
                    )
            return combined_params

    def process_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process context through all plugins."""
        with self.plugin_lock:
            processed_context = context.copy()
            for plugin in self.plugins.values():
                try:
                    processed_context = safe_execute(
                        plugin.pre_prompt_assembly,
                        args=(processed_context,),
                        logger=self.logger
                    ) or processed_context
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="context_processing",
                        error_message=f"Failed to process context: {str(e)}",
                        error=e
                    )
            return processed_context

    def get_combined_memory_context(self) -> str:
        """Combine memory context from all plugins."""
        with self.plugin_lock:
            contexts = []
            for plugin in self.plugins.values():
                try:
                    context = safe_execute(
                        plugin.get_memory_context,
                        logger=self.logger
                    )
                    if context:
                        contexts.append(context)
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="memory_context",
                        error_message=f"Failed to get memory context: {str(e)}",
                        error=e
                    )
            return "\n".join(contexts)

    def get_combined_trait_modifiers(self) -> Dict[str, Dict[str, float]]:
        """Combine trait modifiers from all plugins."""
        with self.plugin_lock:
            combined_modifiers = {}
            for plugin in self.plugins.values():
                try:
                    modifiers = safe_execute(
                        plugin.get_trait_modifiers,
                        logger=self.logger
                    ) or {}
                    for trait, values in modifiers.items():
                        if trait not in combined_modifiers:
                            combined_modifiers[trait] = {}
                        combined_modifiers[trait].update(values)
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="trait_modifiers",
                        error_message=f"Failed to get trait modifiers: {str(e)}",
                        error=e
                    )
            return combined_modifiers

    def cleanup(self) -> None:
        """Cleanup all plugins."""
        with self.plugin_lock:
            for plugin in self.plugins.values():
                try:
                    plugin.cleanup()
                except Exception as e:
                    self.error_manager.handle_error(
                        error_type="plugin_cleanup",
                        error_message=f"Failed to cleanup plugin: {str(e)}",
                        error=e
                    )
            self.plugins.clear()

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
