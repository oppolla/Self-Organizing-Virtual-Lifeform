# SOVL Plugin Development Guide

This guide outlines how to create and integrate new plugins into the SOVL system, specifically for the `sovl_grafter` component, using the `PluginManager`.

## Directory Structure

Each plugin should reside in its own subdirectory within the main `plugins/` directory. The subdirectory should be named after the plugin itself. Inside this subdirectory, you must have at least one Python file named `<plugin_name>.py` (where `<plugin_name>` matches the directory name). This file will contain your plugin's logic.

**Example Structure:**

```
plugins/
└── example_plugin/
    └── example_plugin.py
└── another_plugin/
    ├── another_plugin.py
    └── helper_module.py  # Optional additional files
```

Here, `example_plugin` and `another_plugin` are the names of the plugins.

## Plugin Python Module (`<plugin_name>/<plugin_name>.py`)

The main Python file for your plugin (e.g., `example_plugin/example_plugin.py`) contains the core logic and must define a class named `Plugin` that inherits from `PluginInterface`.

**Core Requirements:**

1.  **`Plugin` Class:** You must define a Python class named `Plugin` that inherits from `sovl_grafter.PluginInterface`.
2.  **`get_metadata()` Method:** This method must be implemented and return a `PluginMetadata` object.
    *   `PluginMetadata` fields:
        *   `name` (str): The unique name of the plugin. This **must** match the subdirectory name and the base name of the main `.py` file.
        *   `version` (str): The version of your plugin (e.g., "1.0.0").
        *   `description` (str): A clear, concise description of what the plugin does.
        *   `author` (str): The author of the plugin.
        *   `enabled` (bool, optional): Defaults to `True`. If set to `False` in the metadata, the plugin will not be activated even if listed in `enabled_plugins` in the configuration. (Note: Primary enablement is controlled via `sovl_config.json`).
3.  **`initialize()` Method:** This method must be implemented.
    *   `initialize(self, system: 'SOVLSystem', context: 'SystemContext') -> None:`
    *   This method is called when the plugin is loaded. Use it to store references to the main `SOVLSystem` instance and the shared `SystemContext`, perform any setup, or initialize plugin-specific state.

**Optional Hook Methods:**

Your `Plugin` class can optionally implement any ofthe following methods from `PluginInterface` to hook into different parts of the SOVL system's primer and generation lifecycle:

*   `modify_system_prompt(self, base_prompt: str) -> str:`
    *   Modify or enhance the base system prompt.
*   `get_generation_parameters(self) -> Dict[str, Any]:`
    *   Provide custom generation parameters (e.g., temperature, top_p) that will be merged with other parameters.
*   `pre_prompt_assembly(self, context: Dict[str, Any]) -> Dict[str, Any]:`
    *   Called before the main prompt components are assembled. Allows modification of the context dictionary that will be used for prompt templating.
*   `post_prompt_assembly(self, assembled_prompt: str) -> str:`
    *   Called after the prompt has been assembled from its various components. Allows final modifications to the complete prompt string.
*   `get_memory_context(self) -> Optional[str]:`
    *   Provide additional string-based context that can be incorporated into the system's memory or prompt, such as recent interactions or relevant information.
*   `cleanup(self) -> None:`
    *   Called when the system is shutting down or the plugin is being unloaded. Use this for any cleanup tasks (e.g., releasing resources).

### Python Example (`plugins/example_plugin/example_plugin.py`):

Refer to the `plugins/example_plugin/example_plugin.py` file in the codebase for a comprehensive example. A snippet is provided below:

```python
from typing import Dict, Any, Optional
from sovl_grafter import PluginInterface, PluginMetadata # Assuming sovl_grafter is accessible
# If SOVLSystem and SystemContext are in sovl_main, you might import them like this:
# from sovl_main import SOVLSystem, SystemContext 
# Adjust imports based on your actual project structure.

# Placeholder for actual SOVLSystem and SystemContext if not directly importable
class SOVLSystem: pass
class SystemContext: pass


class Plugin(PluginInterface):
    """
    Example plugin demonstrating primer integration capabilities.
    """
    
    def __init__(self):
        self.system: Optional[SOVLSystem] = None
        self.context: Optional[SystemContext] = None
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_plugin", # Must match directory and file name
            version="1.0.0",
            description="An example plugin to demonstrate functionality.",
            author="Your Name",
            enabled=True 
        )

    def initialize(self, system: SOVLSystem, context: SystemContext) -> None:
        self.system = system
        self.context = context
        # Example: self.system.logger.log(f"Plugin {self.get_metadata().name} initialized.")
        print(f"Plugin {self.get_metadata().name} initialized.")


    def modify_system_prompt(self, base_prompt: str) -> str:
        return f"{base_prompt}\nThis is an addition from the example_plugin."

    # Implement other optional methods as needed...

    def cleanup(self) -> None:
        # Example: self.system.logger.log(f"Plugin {self.get_metadata().name} cleaning up.")
        print(f"Plugin {self.get_metadata().name} cleaning up.")
        self.system = None
        self.context = None
```

## Plugin Loading

The `PluginManager` automatically discovers and loads enabled plugins by:

1.  Reading the `enabled_plugins` list from the `sovl_config.json` file. Each entry in this list should correspond to a plugin's directory name.
2.  For each enabled plugin name:
    *   It looks for a subdirectory with that name inside the `plugin_directory` (e.g., `plugins/`).
    *   It attempts to load `<plugin_name>/<plugin_name>.py` (e.g., `plugins/example_plugin/example_plugin.py`).
    *   It instantiates the `Plugin` class found in that file.
    *   It calls the `initialize()` method on the plugin instance.

Ensure your plugin directory and main Python file are named identically, and that this name is correctly listed in `enabled_plugins` in your `sovl_config.json`.

## Configuration

Plugins are primarily enabled or disabled by listing their directory names in the `"plugin_config.enabled_plugins"` array within your `sovl_config.json` file.

Example `sovl_config.json` snippet:
```json
{
  "plugin_config": {
    "plugin_directory": "plugins", // Default, can be changed
    "enabled_plugins": [
      "example_plugin",
      "another_plugin"
      // Add other enabled plugin names here
    ]
  }
  // ... other configurations
}
```

The `plugin_directory` field specifies where the `PluginManager` looks for plugin subdirectories. 