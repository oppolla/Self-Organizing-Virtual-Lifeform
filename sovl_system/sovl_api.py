from sovl_conductor import SOVLOrchestrator
from sovl_main import SOVLSystem, SystemContext
from typing import Any, Callable, Optional, List, Dict
import os
import json
import shutil
import datetime

"""
sovl_api.py

This module provides a high-level, user-friendly API for initializing, controlling, and interacting with the SOVL (Self-Organizing Virtual Lifeform) system.

-------------------------------------------------------------------------------
OVERVIEW
-------------------------------------------------------------------------------
- The SOVLAPI class wraps the internal SOVL system and orchestrator, exposing a minimal, stable, and extensible interface for external users and applications.
- Designed for developers who want to embed, automate, or control SOVL from their own Python code, scripts, or services.
- The API is intentionally bare bones: it raises exceptions for errors and does not force any logging or error handling on the user. This allows maximum flexibility and integration into a wide variety of environments.

-------------------------------------------------------------------------------
KEY FEATURES
-------------------------------------------------------------------------------
- Lifecycle control: Start, stop, pause, resume, and reload the SOVL system.
- Input/Output: Send input (text, commands, or structured data) and retrieve output.
- Command execution: Execute system commands programmatically (mirroring CLI functionality).
- Plugin/Hook registration: Register plugins or hooks for extensibility and custom integration.
- System state and metrics: Access system state, metrics, and recent events for monitoring or advanced control.

-------------------------------------------------------------------------------
USAGE EXAMPLE
-------------------------------------------------------------------------------
    from sovl_api import SOVLAPI

    api = SOVLAPI(config_path="my_sovl_config.json")
    api.start()
    api.send_input("Hello, SOVL!")
    print(api.get_output())
    api.stop()

-------------------------------------------------------------------------------
IMPORTANT NOTES
-------------------------------------------------------------------------------
- Exceptions: Most methods may raise NotImplementedError if the underlying system does not support the requested operation, or other exceptions if errors occur. Users are responsible for handling exceptions as appropriate for their application.
- Extensibility: Plugins and hooks can be registered for custom behaviors, but their invocation depends on the system's support for such extensions.
- Thread Safety: This API does not guarantee thread safety. If you use it from multiple threads, ensure proper synchronization.
- Logging: No logging is performed by default. Users may add their own logging as needed.

-------------------------------------------------------------------------------
SEE ALSO
-------------------------------------------------------------------------------
- SOVL documentation and README for more details on system capabilities and configuration.
- The CLI (sovl_cli.py) for interactive command-line usage.
- The orchestrator (sovl_conductor.py) for advanced orchestration and resource management.

-------------------------------------------------------------------------------
"""

class SOVLAPI:
    """
    High-level API for initializing, controlling, and interacting with the SOVL system.
    Wraps the orchestrator and system, providing a stable interface for external integration.

    Note:
        Methods may raise NotImplementedError or other exceptions if the underlying system does not support the requested operation.
        Users are responsible for handling exceptions as appropriate for their application.
    """
    def __init__(self, config_path: str = "sovl_config.json"):
        """
        Initialize the SOVL API and underlying system.
        Args:
            config_path: Path to the SOVL configuration file.
        """
        self.orchestrator = SOVLOrchestrator(config_path=config_path)
        self.orchestrator.initialize_system()
        self.system = self.orchestrator.system  # SOVLSystem instance
        self.context = self.orchestrator.components.get("system_context", None)
        self.plugins: List[Any] = []
        self.hooks: Dict[str, List[Callable]] = {}

    def start(self):
        """
        Start the SOVL system's main loop or processing.
        Raises:
            Exception: If the system fails to start.
        """
        self.orchestrator.run()

    def stop(self):
        """
        Shutdown the SOVL system and release resources.
        Raises:
            Exception: If the system fails to shut down.
        """
        self.orchestrator.shutdown()

    def send_input(self, input_data: Any) -> Any:
        """
        Send input (text, command, or structured data) to the system.
        Args:
            input_data: Input for the system (e.g., text, dict)
        Returns:
            System response or output
        Raises:
            NotImplementedError: If the system does not support input.
            Exception: For any system-level error.
        """
        if hasattr(self.system, "process_input"):
            return self.system.process_input(input_data)
        raise NotImplementedError("System does not support process_input.")

    def get_output(self) -> Any:
        """
        Retrieve the most recent output from the system.
        Returns:
            Last system output
        Raises:
            NotImplementedError: If the system does not support output retrieval.
            Exception: For any system-level error.
        """
        if hasattr(self.system, "get_last_output"):
            return self.system.get_last_output()
        raise NotImplementedError("System does not support get_last_output.")

    def execute_command(self, command: str, args: Optional[List[Any]] = None) -> Any:
        """
        Execute a command in the SOVL system (mirrors CLI functionality).
        Args:
            command: Command string
            args: Optional list of arguments
        Returns:
            Command result or output
        Raises:
            NotImplementedError: If the system does not support command execution.
            Exception: For any system-level error.
        """
        if hasattr(self.system, "execute_command"):
            return self.system.execute_command(command, args or [])
        raise NotImplementedError("System does not support command execution.")

    def register_plugin(self, plugin: Any):
        """
        Register a plugin or adapter with the system.
        Args:
            plugin: Plugin object (should have a register(system) method, if needed)
        """
        self.plugins.append(plugin)
        if hasattr(plugin, "register"):
            plugin.register(self.system)

    def register_hook(self, hook_type: str, hook_fn: Callable):
        """
        Register a hook function for a specific event or type.
        Args:
            hook_type: Event or hook type (str)
            hook_fn: Callable to register
        """
        self.hooks.setdefault(hook_type, []).append(hook_fn)

    def get_system_state(self) -> dict:
        """
        Get the current system state as a dictionary.
        Returns:
            System state dict
        Raises:
            NotImplementedError: If the system does not support state retrieval.
            Exception: For any system-level error.
        """
        if hasattr(self.system, "get_system_state"):
            return self.system.get_system_state()
        raise NotImplementedError("System does not support get_system_state.")

    def get_metrics(self) -> dict:
        """
        Get current system metrics (if available).
        Returns:
            Metrics dict (may be empty if not supported)
        """
        if hasattr(self.system, "get_metrics"):
            return self.system.get_metrics()
        return {}

    def get_recent_events(self, n: int = 10) -> list:
        """
        Get a list of recent system events (if available).
        Args:
            n: Number of events to retrieve
        Returns:
            List of event dicts (may be empty if not supported)
        """
        if hasattr(self.system, "get_recent_events"):
            return self.system.get_recent_events(n)
        return []

    def pause(self) -> bool:
        """
        Pause the system (if supported).
        Returns:
            True if paused, False otherwise
        """
        if hasattr(self.system, "pause"):
            return self.system.pause()
        return False

    def resume(self) -> bool:
        """
        Resume the system (if supported).
        Returns:
            True if resumed, False otherwise
        """
        if hasattr(self.system, "resume"):
            return self.system.resume()
        return False

    def reload_config(self) -> bool:
        """
        Reload the system configuration (if supported).
        Returns:
            True if reloaded, False otherwise
        """
        if hasattr(self.orchestrator, "reload_config"):
            return self.orchestrator.reload_config()
        return False

    CONFIG_FILE = "sovl_config.json"
    DEFAULTS_FILE = "sovl_config.defaults.json"

    def get_config(self, section: str = None, key: str = None):
        """
        Get the full config, a section, or a specific key (dot notation supported).
        """
        with open(self.CONFIG_FILE, "r") as f:
            config = json.load(f)
        if section is None and key is None:
            return config
        node = config
        if section:
            for p in section.split('.'):
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    raise KeyError(f"Section '{section}' not found in config.")
        if key:
            for p in key.split('.'):
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    raise KeyError(f"Key '{key}' not found in config.")
        return node

    def set_config(self, key: str, value: any):
        """
        Set a config value using dot notation, with type validation.
        """
        with open(self.CONFIG_FILE, "r") as f:
            config = json.load(f)
        parts = key.split('.')
        node = config
        for p in parts[:-1]:
            if p in node and isinstance(node[p], dict):
                node = node[p]
            else:
                raise KeyError(f"Config key '{key}' not found.")
        last_key = parts[-1]
        if last_key not in node:
            raise KeyError(f"Config key '{key}' not found.")
        current_value = node[last_key]
        expected_type = type(current_value)
        try:
            # Try to cast value to the expected type
            if expected_type is bool:
                # Special handling for bool
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)
            else:
                value = expected_type(value)
        except Exception:
            raise ValueError(f"Type mismatch: expected {expected_type.__name__} for key '{key}'")
        node[last_key] = value
        with open(self.CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        self.reload_config()

    def reset_config(self, section: str = None):
        """
        Reset the whole config or a section to defaults, with backup.
        """
        # Backup current config
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        backup_path = f"{self.CONFIG_FILE}.bak-{ts}"
        shutil.copy2(self.CONFIG_FILE, backup_path)
        with open(self.DEFAULTS_FILE, "r") as f:
            defaults = json.load(f)
        if section:
            with open(self.CONFIG_FILE, "r") as f:
                config = json.load(f)
            if section not in defaults:
                raise KeyError(f"Section '{section}' not found in defaults.")
            config[section] = defaults[section]
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        else:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(defaults, f, indent=4)
        self.reload_config()

    def search_config(self, term: str):
        """
        Search for config keys containing the term (dot notation).
        """
        with open(self.CONFIG_FILE, "r") as f:
            config = json.load(f)
        results = []
        def search(node, path=[]):
            if isinstance(node, dict):
                for k, v in node.items():
                    search(v, path + [k])
            else:
                key_path = '.'.join(path)
                if term.lower() in key_path.lower():
                    results.append(key_path)
        search(config)
        return results

    def help_config(self, key: str):
        """
        Return value, type, and a placeholder for description for a config key.
        """
        value = self.get_config(key=key)
        return {
            "key": key,
            "value": value,
            "type": type(value).__name__,
            "description": "(No inline description. See docs or schema for details.)"
        }
