from typing import Dict, Any, Optional, List
from sovl_grafter import PluginInterface, PluginMetadata, ConfigSchema
import time

class Plugin(PluginInterface):
    """Example plugin demonstrating plugin system capabilities."""
    
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
        self.error_manager = context.error_manager
        
        try:
            self.system.plugin_manager.register_hook(
                "pre_generate",
                self.pre_generate_hook,
                plugin_name="example_plugin",
                priority=10
            )
            
            self.logger.record({
                "event": "example_plugin_initialized",
                "timestamp": time.time()
            })
        except Exception as e:
            if self.error_manager:
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
            description="Example plugin for SOVLSystem",
            author="xAI",
            dependencies=[],
            priority=10,
            enabled=True,
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
                ram_usage = self.context.ram_manager.get_usage() if hasattr(self.context, 'ram_manager') else None
                gpu_usage = self.context.gpu_manager.get_usage() if hasattr(self.context, 'gpu_manager') else None
                
                result = {
                    "status": "executed",
                    "context": context,
                    "system_stats": {
                        "ram_usage": ram_usage,
                        "gpu_usage": gpu_usage
                    }
                }
                
                if self.logger:
                    self.logger.record({
                        "event": "example_plugin_executed",
                        "result": str(result)[:200],
                        "timestamp": time.time()
                    })
                
                return result
        except Exception as e:
            if self.error_manager:
                self.error_manager.handle_error(
                    error_type="plugin_execution",
                    error_message=f"Failed to execute example plugin: {str(e)}",
                    error=e
                )
            return None
    
    def pre_generate_hook(self, context: Dict[str, Any]) -> None:
        """Hook executed before generating a response."""
        try:
            if self.logger:
                self.logger.record({
                    "event": "example_plugin_pre_generate",
                    "context": str(context)[:200],
                    "timestamp": time.time(),
                    "conversation_id": self.system.state.history.conversation_id if hasattr(self.system, 'state') else None
                })
        except Exception as e:
            if self.error_manager:
                self.error_manager.handle_error(
                    error_type="plugin_hook_execution",
                    error_message=f"Failed to execute pre_generate hook: {str(e)}",
                    error=e
                )
    
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