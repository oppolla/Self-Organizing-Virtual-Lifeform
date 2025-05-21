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
    1. Create a new directory in the plugins folder
    2. Create an __init__.py file with a Plugin class that inherits from PluginInterface
    3. Implement the required methods (initialize, get_metadata)
    4. Implement any optional methods you want to use to modify the system's behavior
    5. Add your plugin name to the enabled_plugins list in the configuration
    """
    
    def __init__(self):
        """Initialize plugin state.
        
        The plugin maintains references to:
        - system: The main SOVLSystem instance for accessing system components
        - context: The SystemContext for accessing shared resources
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
        """Define plugin information.
        
        This metadata is used by the plugin manager to:
        - Identify the plugin
        - Track versions
        - Provide descriptions
        - Control plugin state (enabled/disabled)
        
        Returns:
            PluginMetadata object with plugin information
        """
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin demonstrating primer integration",
            author="xAI",
            enabled=True
        )
    
    def modify_system_prompt(self, base_prompt: str) -> str:
        """Enhance the system prompt that defines AI behavior.
        
        Use this method to:
        - Add specific instructions
        - Define constraints
        - Set behavioral guidelines
        - Include custom context
        
        The modifications here directly influence how the AI understands
        its role and how it should respond.
        
        Args:
            base_prompt: The original system prompt
        
        Returns:
            Modified system prompt with additional instructions
        """
        return f"{base_prompt}\nAdditional instructions:\n" \
               f"1. Always provide code examples when explaining technical concepts\n" \
               f"2. Use clear section headings for complex explanations\n" \
               f"3. Include error handling considerations in code suggestions"
    
    def get_generation_parameters(self) -> Dict[str, Any]:
        """Configure AI response generation parameters.
        
        Use this method to adjust:
        - temperature: Controls randomness (0.0-1.0)
        - top_p: Controls diversity of token selection
        - max_tokens: Limits response length
        - presence_penalty: Encourages topic diversity
        - frequency_penalty: Discourages repetition
        
        These parameters influence how the AI generates its responses,
        affecting creativity, consistency, and style.
        
        Returns:
            Dictionary of generation parameters
        """
        return {
            "temperature": 0.7,  # Balanced between creativity and consistency
            "top_p": 0.9,       # Allow some diversity in responses
            "max_tokens": 2000,  # Support longer responses
            "presence_penalty": 0.1,  # Slight encouragement for diverse topics
            "frequency_penalty": 0.1   # Slight discouragement of repetition
        }
    
    def pre_prompt_assembly(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify context before the prompt is assembled.
        
        This is called before the final prompt is created, allowing you to:
        - Add custom context
        - Set preferences
        - Include relevant information
        - Modify existing context
        
        The context provided here influences how the prompt is assembled
        and how the AI understands the current situation.
        
        Args:
            context: Current context dictionary
        
        Returns:
            Modified context dictionary
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
        """Modify the final prompt before it's sent to the AI.
        
        This is your last chance to modify the prompt, useful for:
        - Adding final reminders
        - Including global context
        - Enforcing constraints
        - Adding consistent suffixes
        
        Args:
            assembled_prompt: The fully assembled prompt
        
        Returns:
            Final modified prompt
        """
        return f"{assembled_prompt}\n\nRemember to:\n" \
               f"- Consider edge cases\n" \
               f"- Explain trade-offs\n" \
               f"- Provide practical examples"
    
    def get_memory_context(self) -> Optional[str]:
        """Provide additional memory context for the AI.
        
        Use this to add relevant historical or contextual information like:
        - Previous interactions
        - Important topics
        - Ongoing themes
        - User preferences
        
        This context helps maintain consistency across interactions
        and provides important background information.
        
        Returns:
            String of memory context or None
        """
        return "\nRecent topics discussed:\n" \
               "- Code organization best practices\n" \
               "- Error handling patterns\n" \
               "- Performance optimization techniques"
    
    def get_trait_modifiers(self) -> Dict[str, Dict[str, float]]:
        """Define or modify AI personality traits.
        
        Use this to influence the AI's personality by adjusting:
        - base_value: The default strength of the trait (0.0-1.0)
        - context_weight: How much context influences the trait (0.0-1.0)
        
        These traits influence how the AI expresses itself and
        handles different situations.
        
        Returns:
            Dictionary of trait definitions and their parameters
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
        """Clean up plugin resources.
        
        Called when the plugin is being unloaded. Use this to:
        - Close connections
        - Free resources
        - Reset state
        - Clean up any temporary data
        """
        self.system = None
        self.context = None 