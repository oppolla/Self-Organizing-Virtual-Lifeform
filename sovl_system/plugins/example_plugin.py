from typing import Dict, Any, Optional
from sovl_grafter import PluginInterface, PluginMetadata
from sovl_main import SOVLSystem, SystemContext

class Plugin(PluginInterface):
    """Example plugin demonstrating primer integration capabilities.
    
    The SOVL plugin system allows you to modify and enhance the behavior of the AI system
    through the primer integration. Each plugin can:
    1. Modify the system prompt
    2. Adjust generation parameters
    3. Add context before and after prompt assembly
    4. Provide memory context
    5. Influence personality traits
    
    To create your own plugin:
    1. Create a Python file in the plugins directory
    2. Add your plugin name to enabled_plugins in sovl_config.json
    
    Required methods:
    - initialize(system, context)
    - get_metadata()
    
    All other methods are optional - implement only what you need.
    """
    
    def __init__(self):
        """
        Initialize plugin state.
        """
        self.system = None
        self.context = None
    
    def initialize(self, system: SOVLSystem, context: SystemContext) -> None:
        """Set up the plugin with system access.
        
        This method is called when the plugin is loaded. Use it to:
        - Store references to system and context
        - Set up any resources needed by the plugin
        - Initialize any plugin state
        
        Args:
            system: Main SOVL system instance
            context: Shared system context
        """
        self.system = system
        self.context = context
    
    def get_metadata(self) -> PluginMetadata:
        """
        Define plugin information.
        """
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin demonstrating primer integration",
            author="xAI",
            enabled=True
        )
    
    def modify_system_prompt(self, base_prompt: str) -> str:
        """
        Optional: Add custom instructions to the system prompt.
        """
        return f"{base_prompt}\nAdditional instructions:\n" \
               f"1. Always provide code examples when explaining technical concepts\n" \
               f"2. Use clear section headings for complex explanations\n" \
               f"3. Include error handling considerations in code suggestions"
    
    def get_generation_parameters(self) -> Dict[str, Any]:
        """
        Optional: Configure AI generation parameters (temperature, top_p, etc).
        """
        return {
            "temperature": 0.7,  # Balanced between creativity and consistency
            "top_p": 0.9,       # Allow some diversity in responses
            "max_tokens": 2000,  # Support longer responses
            "presence_penalty": 0.1,  # Slight encouragement for diverse topics
            "frequency_penalty": 0.1   # Slight discouragement of repetition
        }
    
    def pre_prompt_assembly(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional: Modify context before prompt assembly.
        """
        context.update({
            "code_style": {
                "prefer_type_hints": True,
                "max_line_length": 88,
                "docstring_style": "google"
            },
            "technical_depth": "intermediate",
            "explanation_style": "detailed"
        })
        return context
    
    def post_prompt_assembly(self, assembled_prompt: str) -> str:
        """
        Optional: Modify the final prompt before it's sent to the AI.
        """
        return f"{assembled_prompt}\n\nRemember to:\n" \
               f"- Consider edge cases\n" \
               f"- Explain trade-offs\n" \
               f"- Provide practical examples"
    
    def get_memory_context(self) -> Optional[str]:
        """
        Optional: Add historical or contextual information.
        """
        return "\nRecent topics discussed:\n" \
               "- Code organization best practices\n" \
               "- Error handling patterns\n" \
               "- Performance optimization techniques"
    
    def get_trait_modifiers(self) -> Dict[str, Dict[str, float]]:
        """
        Optional: Modify AI personality traits.
        """
        return {
            "technical_depth": {
                "base_value": 0.8,     # Prefer detailed technical explanations
                "context_weight": 0.3   # Moderately influenced by context
            },
            "explanation_detail": {
                "base_value": 0.7,     # Detailed but not overwhelming
                "context_weight": 0.4   # More responsive to context
            },
            "code_focus": {
                "base_value": 0.9,     # Strong emphasis on code examples
                "context_weight": 0.5   # Highly context-dependent
            }
        }
    
    def cleanup(self) -> None:
        """
        Optional: Clean up plugin resources.
        """
        self.system = None
        self.context = None 