import time
import torch
import traceback
from typing import List, Dict, Tuple, Optional, Callable
from sovl_main import SOVLSystem, SystemInitializationError
from sovl_state import StateManager
from sovl_config import ConfigManager
from sovl_utils import safe_compare, format_file_size, format_timestamp, print_section_header, print_bullet_list, print_kv_table, progress_bar, print_success, print_error
from sovl_monitor import SystemMonitor, MemoryMonitor, TraitsMonitor
import readline
from collections import deque
import cmd
import sys
from sovl_logger import Logger
from sovl_error import ErrorManager
import shlex
import platform
import os
import asyncio
import random
import datetime
import re
import difflib
import threading
import json
import shutil
from sovl_dreamer import Dreamer
from sovl_recaller import DialogueContextManager
import importlib
import pkgutil
import argparse
from sovl_tester import SOVLTestRunner

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
FORMATTED_TRAINING_DATA = None
VALID_DATA = None

COMMAND_CATEGORIES = {
    "System": [
        "/save", "/load", "/reset", "/monitor", "/history", "/run", "/stop",
        "/config", "/log", "/exit", "/quit", "/pause", "/resume"
    ],
    "Modes & States": [
        "/trip", "/drunk", "/dream", "/gestate",  "/announce", "/shy", "/pidgin", 
        "/backchannel",
    ],
    "Memory & Recall": [
        "/recall", "/forget", "/rewind", "/recap", "/journal", "/attune", "/reflect", "/epiphany",
    ],
    "Interaction & Fun": [
        "/echo", "/mimic", "/fortune", "/tattle", "/blurt", "/joke", "/ping", "/muse",
        "/rate", "/complain", "/confess", "/rant", "/debate", "/flare", "/spark",
    ],
    "Debug & Development": [
        "/panic", "/glitch", "/scaffold", "/errors", "/trace", "/components", "/reload", "/test",
    ],
    "Learning & Guidance": [
        "/help", "/tutorial"
    ]
}

# Centralized alias mapping for help display
ALIASES = {
    "q": "quit",
    "h": "help",
    "ls": "history",  
    "r": "reset",  
}

# Feature flag for dynamic command discovery
ENABLE_DYNAMIC_COMMANDS = False

class CommandHistory:
    """Manages command history with search functionality and execution status."""
    def __init__(self, max_size: int = 100):
        self.history = deque(maxlen=max_size)  # Each entry: {"command": ..., "status": ..., "timestamp": ...}
        self.current_index = -1

    def add(self, command: str, status: str = "unknown"):
        """Add a command and its status to history."""
        timestamp = datetime.datetime.now().isoformat(timespec='seconds')
        self.history.append({"command": command, "status": status, "timestamp": timestamp})
        self.current_index = -1

    def get_previous(self) -> Optional[str]:
        if not self.history:
            return None
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
        return self.history[-(self.current_index + 1)]["command"]

    def get_next(self) -> Optional[str]:
        if not self.history or self.current_index < 0:
            return None
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[-(self.current_index + 1)]["command"]
        self.current_index = -1
        return ""

    def search(self, query: str) -> list:
        return [entry for entry in self.history if query.lower() in entry["command"].lower()]

    def clear(self):
        self.history.clear()
        self.current_index = -1

    def save(self, filename="command_history.json"):
        import json
        with open(filename, "w") as f:
            json.dump(list(self.history), f)

    def load(self, filename="command_history.json"):
        import json, os
        if os.path.exists(filename):
            with open(filename, "r") as f:
                loaded = json.load(f)
                # Backward compatibility: add timestamp if missing
                for entry in loaded:
                    if "timestamp" not in entry:
                        entry["timestamp"] = "unknown"
                self.history = deque(loaded, maxlen=self.history.maxlen)

class CommandHandler(cmd.Cmd):
    """Enhanced command handler with history and search capabilities."""
    prompt = 'sovl> '
    
    def __init__(self, sovl_system: SOVLSystem):
        super().__init__()
        self.sovl_system = sovl_system
        self.logger = Logger.instance()
        self.error_manager = ErrorManager.instance()

        # Comprehensive validation of required SOVLSystem components
        required_attrs = [
            'config_handler', 'ram_manager', 'state_tracker', 'gpu_manager',
            'memory_manager', 'generation_manager', 'bond_calculator', 'logger', 'error_manager'
        ]
        missing = [attr for attr in required_attrs if not hasattr(sovl_system, attr)]
        if missing:
            error_msg = f"Missing required sovl_system attributes: {', '.join(missing)}"
            self.logger.log_error(error_msg, error_type="cli_init_error")
            raise AttributeError(error_msg)

        # Initialize monitoring components with fallback
        try:
            self.system_monitor = SystemMonitor(
                config_manager=getattr(sovl_system, 'config_handler', None),
                logger=self.logger,
                error_manager=self.error_manager
            )
        except Exception as e:
            print(f"Warning: Failed to initialize SystemMonitor: {e}")
            self.logger.log_error(
                error_msg=f"Failed to initialize SystemMonitor: {e}",
                error_type="cli_init_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_init_error",
                message=f"Failed to initialize SystemMonitor: {e}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.__init__"
            )
            self.system_monitor = None

        try:
            self.memory_monitor = MemoryMonitor(
                config_manager=getattr(sovl_system, 'config_handler', None),
                logger=self.logger,
                ram_manager=getattr(sovl_system, 'ram_manager', None),
                gpu_manager=getattr(sovl_system, 'gpu_manager', None),
                error_manager=self.error_manager
            )
            self.memory_monitor_fallback = False
        except Exception as e:
            print(f"Warning: Failed to initialize MemoryMonitor: {e}")
            self.logger.log_error(
                error_msg=f"Failed to initialize MemoryMonitor: {e}",
                error_type="cli_init_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_init_error",
                message=f"Failed to initialize MemoryMonitor: {e}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.__init__"
            )
            self.memory_monitor = None
            self.memory_monitor_fallback = hasattr(sovl_system, 'memory_manager')

        try:
            self.traits_monitor = TraitsMonitor(
                config_manager=getattr(sovl_system, 'config_handler', None),
                logger=self.logger,
                state_tracker=getattr(sovl_system, 'state_tracker', None),
                error_manager=self.error_manager
            )
        except Exception as e:
            print(f"Warning: Failed to initialize TraitsMonitor: {e}")
            self.logger.log_error(
                error_msg=f"Failed to initialize TraitsMonitor: {e}",
                error_type="cli_init_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_init_error",
                message=f"Failed to initialize TraitsMonitor: {e}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.__init__"
            )
            self.traits_monitor = None

        # --- Trip State Attributes ---
        self.is_tripping = False
        self.trip_start_time = 0.0
        self.trip_duration = 0.0
        self.is_drunk = False
        self.drunk_start_time = 0.0
        self.drunk_duration = 0.0
        self.is_ranting = False
        self.is_debating = False

        self._last_mode = None
        self._last_progress = None
        self._mode_monitor_thread = threading.Thread(target=self._monitor_mode, daemon=True)
        self._mode_monitor_thread.start()
        # --- Command registry for modularization ---
        self._command_registry = {}
        self._register_builtin_commands()
        # Dynamic command discovery (feature-flagged)
        if ENABLE_DYNAMIC_COMMANDS:
            self._register_external_commands()

    def _monitor_mode(self):
        dot_count = 1
        while True:
            try:
                state_manager = getattr(self.sovl_system, 'state_manager', None)
                system_monitor = getattr(self, 'system_monitor', None)
                if state_manager and hasattr(state_manager, 'get_mode'):
                    mode = state_manager.get_mode()
                    if mode != self._last_mode:
                        self._handle_mode_change(mode)
                        self._last_mode = mode
                        self._last_progress = None  # Reset progress on mode change
                    if mode == 'gestating' and system_monitor and hasattr(system_monitor, 'get_gestation_message'):
                        message = system_monitor.get_gestation_message(dot_count)
                        print(f"\r{message}", end="")
                        dot_count = dot_count % 3 + 1
                    elif mode == 'dreaming' and system_monitor and hasattr(system_monitor, 'get_dreaming_message'):
                        message = system_monitor.get_dreaming_message(dot_count)
                        print(f"\r{message}", end="")
                        dot_count = dot_count % 3 + 1
                    elif mode == 'meditating' and system_monitor and hasattr(system_monitor, 'get_meditating_message'):
                        message = system_monitor.get_meditating_message(dot_count)
                        print(f"\r{message}", end="")
                        dot_count = dot_count % 3 + 1
            except Exception:
                pass
            time.sleep(1)

    def _progress_bar(self, percent, width=30):
        percent = max(0.0, min(1.0, percent))
        filled = int(width * percent)
        bar = '#' * filled + '-' * (width - filled)
        return f'[{bar}] {int(percent * 100)}%'

    def _handle_mode_change(self, mode):
        if mode == "online":
            print("\n[ONLINE] System is ready for use.")
        elif mode == "gestating":
            print("\n[GESTATING] Training in progress. Please wait...")
        elif mode == "meditating":
            print("\n[MEDITATING] Introspection in progress. Please wait or /stop to abort...")
        elif mode == "dreaming":
            print("\n[DREAMING] Dreaming in progress. Please wait or /stop to abort...")
        elif mode == "offline":
            print("\n[OFFLINE] System is unavailable.")

    def preloop(self):
        """Initialize the command handler with onboarding and wake message."""
        print("SOVL System online")
        print("/help for commands list")
        print("/tutorial for tutorial mode")
        print("/stop to exit modes")
        print("/monitor for system metrics")
        # Show the current wake message, if present, but only here
        wake_msg = getattr(self.sovl_system, 'wake_greeting', None)
        if wake_msg:
            print(f"\n{wake_msg}\n")
        self.logger.record_event(
            event_type="cli_startup",
            message="SOVL CLI started",
            level="info"
        )
        
    def do_help(self, arg):
        """Show this help message, grouped by category, with aliases. Use '/help <command>' for details."""
        # Build reverse alias mapping: canonical -> [aliases]
        reverse_aliases = {}
        for alias, canonical in ALIASES.items():
            reverse_aliases.setdefault(canonical, []).append(alias)

        # Gather all do_ methods
        all_cmds = {name[3:]: getattr(self, name) for name in self.get_names() if name.startswith('do_')}
        # Map to /command style
        all_cmds_slash = {f"/{cmd}": method for cmd, method in all_cmds.items()}

        # Contextual help if arg is provided
        if arg and arg.strip():
            query = arg.strip().lstrip('/')
            # Resolve alias
            canonical = ALIASES.get(query, query)
            # Try help_<command> first
            help_method = getattr(self, f'help_{canonical}', None)
            if help_method:
                help_method()  # Assume it prints its own output
                return
            # Try docstring of do_<command>
            do_method = getattr(self, f'do_{canonical}', None)
            if do_method and do_method.__doc__:
                print(f"\n/{canonical} - {do_method.__doc__.strip()}")
                return
            # Try with slash (for commands like /bc)
            if canonical not in all_cmds:
                for cmd in all_cmds:
                    if cmd.lower() == canonical.lower():
                        do_method = all_cmds[cmd]
                        if do_method and do_method.__doc__:
                            print(f"\n/{cmd} - {do_method.__doc__.strip()}")
                            return
            # Fuzzy/partial matching suggestions
            all_names = list(all_cmds.keys()) + list(ALIASES.keys())
            matches = [cmd for cmd in all_names if cmd.startswith(query)]
            if not matches:
                matches = [cmd for cmd in all_names if query in cmd]
            if not matches:
                matches = difflib.get_close_matches(query, all_names, n=3, cutoff=0.6)
            if matches:
                print(f"Unknown command: {arg.strip()}. Did you mean:")
                for s in matches:
                    print(f"  /{s}")
            else:
                print(f"Unknown command: {arg.strip()}. Use /help to see all commands.")
            return

        # No arg: show categorized help
        shown = set()
        print("\nAvailable commands (grouped by category):")
        for category, commands in COMMAND_CATEGORIES.items():
            print(f"\n{category} Commands:")
            for cmd in commands:
                cmd_name = cmd.lstrip('/')
                method = all_cmds.get(cmd_name)
                if not method:
                    continue
                doc = method.__doc__ or ''
                # Show aliases inline
                aliases = reverse_aliases.get(cmd_name, [])
                alias_str = f" (alias: {', '.join('/'+a for a in aliases)})" if aliases else ''
                print(f"  {cmd}{alias_str} - {doc.strip()}")
                shown.add(cmd_name)
        # Show uncategorized commands
        uncategorized = [cmd for cmd in all_cmds if f"/{cmd}" not in sum(COMMAND_CATEGORIES.values(), [])]
        if uncategorized:
            print("\nOther Commands:")
            for cmd in sorted(uncategorized):
                method = all_cmds[cmd]
                doc = method.__doc__ or ''
                aliases = reverse_aliases.get(cmd, [])
                alias_str = f" (alias: {', '.join('/'+a for a in aliases)})" if aliases else ''
                print(f"  /{cmd}{alias_str} - {doc.strip()}")
        # Show all aliases at the end for quick reference
        if ALIASES:
            print("\nAliases:")
            for alias, canonical in ALIASES.items():
                print(f"  /{alias} = /{canonical}")
            
    def do_pause(self, arg):
        """Pause the current operation."""
        if self.sovl_system.pause():
            print("Operation paused")
        else:
            print("No operation to pause")
            
    def do_resume(self, arg):
        """Resume the current operation."""
        if self.sovl_system.resume():
            print("Operation resumed")
        else:
            print("No operation to resume")
            
    def do_config(self, arg):
        """
        Interactive configuration editor for sovl_config.json.
        Usage:
          /config show [section[.key]]   # Show config, section, or key
          /config set <key> <value>      # Set a config value (dot notation)
          /config search <term>          # Search for config keys
          /config help <key>             # Show help for a config key
          /config reset [section]        # Reset config or section to defaults
        """
        import json, shutil, datetime, os
        args = shlex.split(arg)
        if not args or args[0] in ("show",):
            # /config show [section[.key]]
            section = args[1] if len(args) > 1 else None
            config = self.sovl_system.config_handler.get_config() if hasattr(self.sovl_system, 'config_handler') else None
            if not config:
                print_error("Config manager not available.")
                return
            if not section:
                # Show top-level sections
                print_section_header("Config Sections:")
                for k in config:
                    print(f"  {k}")
                print("\nUse '/config show <section>' to view keys in a section.")
                # Simple hints for user
                print("\nHints:")
                print("  - Use '/config search <term>' to find keys by name.")
                print("  - Use '/config set <section.key> <value>' to change a value.")
                print("  - Use '/config help <section.key>' to see a key's value and type.")
                print("  - Use '/config reset' to restore defaults (with backup and confirmation).")
                print("  - Use '/help config' for more details and examples.")
                return
            # Support dot notation for deeper keys
            parts = section.split('.')
            node = config
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    print_error(f"Section/key '{section}' not found.")
                    return
            print_section_header(f"Config: {section}")
            print(json.dumps(node, indent=2, sort_keys=True))
            return
        elif args[0] == "reset":
            # /config reset [section]
            section = args[1] if len(args) > 1 else None
            config_handler = getattr(self.sovl_system, 'config_handler', None)
            if not config_handler or not hasattr(config_handler, 'get_config'):
                print_error("Config manager not available.")
                return
            # Confirm
            if section:
                prompt = f"Reset config section '{section}' to defaults? This cannot be undone. (y/n): "
            else:
                prompt = "Reset ALL config to defaults? This cannot be undone. (y/n): "
            confirm = input(prompt).strip().lower()
            if confirm != 'y':
                print("Aborted.")
                return
            # Backup current config
            config_path = getattr(config_handler, 'config_file', 'sovl_config.json')
            if os.path.exists(config_path):
                ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                backup_path = f"{config_path}.bak-{ts}"
                shutil.copy2(config_path, backup_path)
                print_success(f"Backed up current config to {backup_path}")
            # Load defaults
            defaults_path = 'sovl_config.defaults.json'
            if not os.path.exists(defaults_path):
                print_error(f"Defaults file '{defaults_path}' not found. Cannot reset.")
                return
            with open(defaults_path, 'r') as f:
                defaults = json.load(f)
            if section:
                # Only reset the section
                config = config_handler.get_config()
                if section not in defaults:
                    print_error(f"Section '{section}' not found in defaults.")
                    return
                config[section] = defaults[section]
                config_handler.save_config()
                print_success(f"Section '{section}' reset to defaults.")
            else:
                # Reset all
                config_handler.store.flat_config = defaults
                config_handler.save_config()
                print_success("All config reset to defaults.")
            return
        elif args[0] == "set" and len(args) >= 3:
            # /config set <key> <value>
            key = args[1]
            value = ' '.join(args[2:])
            config_handler = getattr(self.sovl_system, 'config_handler', None)
            if not config_handler or not hasattr(config_handler, 'update'):
                print_error("Config manager not available or does not support update.")
                return
            # Type/value validation
            config = config_handler.get_config() if hasattr(config_handler, 'get_config') else None
            if not config:
                print_error("Config manager not available.")
                return
            parts = key.split('.')
            node = config
            for p in parts[:-1]:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    print_error(f"Config key '{key}' not found.")
                    return
            last_key = parts[-1]
            if not (isinstance(node, dict) and last_key in node):
                print_error(f"Config key '{key}' not found.")
                return
            current_value = node[last_key]
            expected_type = type(current_value)
            # Try to cast value to expected type
            try:
                if expected_type is bool:
                    if value.lower() in ("true", "1"): value_cast = True
                    elif value.lower() in ("false", "0"): value_cast = False
                    else: raise ValueError
                elif expected_type is int:
                    value_cast = int(value)
                elif expected_type is float:
                    value_cast = float(value)
                elif expected_type is type(None):
                    value_cast = None if value.lower() == "null" else value
                else:
                    value_cast = value
            except Exception:
                print_error(f"Type mismatch: key '{key}' expects {expected_type.__name__}, but got '{value}'.")
                return
            # Always prompt for confirmation
            print(f"Change '{key}' from {current_value!r} ({expected_type.__name__}) to {value_cast!r}? (y/n): ", end="")
            confirm = input().strip().lower()
            if confirm != 'y':
                print("Aborted.")
                return
            success = config_handler.update(key, value_cast)
            if success:
                print_success(f"Config key '{key}' updated to {value_cast!r}.")
            else:
                print_error(f"Failed to update config key '{key}'.")
            return
        elif args[0] == "search" and len(args) > 1:
            # /config search <term>
            term = args[1].lower()
            config = self.sovl_system.config_handler.get_config() if hasattr(self.sovl_system, 'config_handler') else None
            if not config:
                print_error("Config manager not available.")
                return
            def find_keys(d, prefix=""):
                results = []
                if isinstance(d, dict):
                    for k, v in d.items():
                        full = f"{prefix}.{k}" if prefix else k
                        if term in k.lower():
                            results.append(full)
                        results.extend(find_keys(v, full))
                return results
            matches = find_keys(config)
            if matches:
                print_section_header(f"Config keys matching '{term}':")
                for m in matches:
                    print(f"  {m}")
            else:
                print(f"No config keys found matching '{term}'.")
            return
        elif args[0] == "help" and len(args) > 1:
            # /config help <key>
            key = args[1]
            # Try to show value and type
            config = self.sovl_system.config_handler.get_config() if hasattr(self.sovl_system, 'config_handler') else None
            if not config:
                print_error("Config manager not available.")
                return
            parts = key.split('.')
            node = config
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    print_error(f"Config key '{key}' not found.")
                    return
            print_section_header(f"Help for config key: {key}")
            print(f"Current value: {node!r}")
            print(f"Type: {type(node).__name__}")
            print("(No inline description available. See docs or sovl_config.json schema for details.)")
            return
        else:
            print_error("Usage: /config show [section[.key]], /config set <key> <value>, /config search <term>, /config help <key>, /config reset [section]")
            print("\nHints:")
            print("  - Use '/config show' to list all config sections.")
            print("  - Use '/config search <term>' to find keys by name.")
            print("  - Use '/config set <section.key> <value>' to change a value.")
            print("  - Use '/config help <section.key>' to see a key's value and type.")
            print("  - Use '/config reset' to restore defaults (with backup and confirmation).")
            print("  - Use '/help config' for more details and examples.")

    def do_reset(self, arg):
        """Reset the system to its initial state using atomic update."""
        state_manager = getattr(self.sovl_system.context, 'state_manager', None)
        if not state_manager:
            print_error("StateManager not available for atomic update.")
            return
        try:
            def reset_update(state):
                # Reinitialize or clear state as appropriate
                if hasattr(state, '_initialize_state'):
                    state._initialize_state()
                    print_success("System state has been reset (atomic).")
                else:
                    print_error("State object does not support initialization/reset.")
            state_manager.update_state_atomic(reset_update)
        except Exception as e:
            print_error(f"Atomic reset update failed: {e}")

    def do_exit(self, arg):
        """Exit the CLI."""
        try:
            print("CLI session terminated.")
            self.logger.record_event(
                event_type="cli_exit",
                message="CLI exited by user",
                level="info"
            )
            return True
        except Exception as e:
            print(f"Error during exit: {e}")
            self.logger.log_error(
                error_msg=f"Exit command failed: {str(e)}",
                error_type="cli_exit_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_exit_error",
                message=f"Exit command failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.do_exit"
            )
            
    def do_quit(self, arg):
        """Alias for /quit"""
        return self.do_exit(arg)

    def do_history(self, arg):
        """Show command history with execution status, timestamps, and numbering."""
        try:
            cmd_history = getattr(self, 'cmd_history', None) or getattr(self.sovl_system, 'cmd_history', None)
            if cmd_history and hasattr(cmd_history, 'history'):
                if not cmd_history.history:
                    print("No history available.")
                    return
                print("\nCommand History:")
                print("---------------")
                for idx, entry in enumerate(cmd_history.history, 1):
                    ts = entry.get("timestamp", "unknown")
                    cmd = entry.get("command", "")
                    status = entry.get("status", "")
                    print(f"{idx}. [{ts}] {cmd} [{status}]")
                self.logger.record_event(
                    event_type="cli_history",
                    message="Displayed command history",
                    level="info"
                )
            else:
                print("No history available.")
        except Exception as e:
            print(f"Error displaying history: {e}")
            self.logger.log_error(
                error_msg=f"History command failed: {str(e)}",
                error_type="cli_history_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_history_error",
                message=f"History command failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.do_history"
            )
            
    def do_search(self, arg):
        """Search command history with execution status."""
        if not arg:
            print("Please provide a search query")
            return
        cmd_history = getattr(self, 'cmd_history', None) or getattr(self.sovl_system, 'cmd_history', None)
        if not cmd_history or not hasattr(cmd_history, 'history'):
            print("No history available.")
            return
        matches = cmd_history.search(arg)
        if matches:
            print("\nMatching Commands:")
            print("-----------------")
            for entry in matches:
                print(f"{entry['command']}  [status: {entry['status']}]")
        else:
            print("No matching commands found")
            
    def do_joke(self, arg):
        """Tell a joke using the LLM."""
        try:
            # Get the joke from the LLM
            joke = self.get_joke_from_llm()
            print("\n" + joke + "\n")
        except Exception as e:
            print(f"Error generating joke: {str(e)}")
            print("I'm not feeling funny right now.")

    def get_joke_from_llm(self) -> str:
        """Fetch a joke from the LLM using a refined prompt with higher temperature for more creative responses."""
        prompt = (
            "Compose one concise joke (50 words) with a clear setup and punchline. "
            "Essential qualities:\n"
            "1. Setup: Must establish a barbed, meaningful topic directed at the techbro user reflecting on how they make the world a worse place\n"
            "   - Jokes can be situational, conceptual, ontological, recursive, Gödelian, or meta\n"
            "   - Should contain an inherent tension or contradiction\n"
            "   - It should not be boring or mundane. The target audience is highly intelligent with a well developed sense of humor\n"
            "   - Avoid low-hanging fruit. High concept novel jokes only\n"
            "2. Punchline: Must deliver satisfying resolution through either:\n"
            "   - Clever, ironic, or deep cutting subversion of the setup's logic\n"
            "   - Absurdist amplification of the initial premise\n"
            "   - Get so real it's painful\n"
            "   - Dry understatement that highlights inherent irony\n"
            "Key constraints:\n"
            "- Maintain tight, economical wording\n"
            "- Avoid explanatory or meta commentary\n"
            "- Include at least one vivid, concrete element\n"
            "- Let the humor emerge naturally from the premise"
            "- No introductory phrases\n"
           "- No commentary\n"
           "- No quotation marks\n"
           "- Joke should be generally addressed not at a you\n"
           "- Do not include labels like Setup or Punchline\n"
           "- Avoid app topics\n"
           "- Do not ever say techbro\n"
           "- Do not state stats like wordcount or add notes explaining the joke\n"
           "- Never ever explain the joke. Leave the joke as the only thing to be said\n"
            "Tone targets:\n"
            "- Like a depressed philosopher rambling at a bar\n"
            "- Be thankful for all those slaving away to build AI systems that will replace them\n"
            "- Be aware of generative slop\n"
            "- Use tech and VC jargon filled language\n"
            "- Or a jaded and sardonic scientist after realizing the true nature of reality\n"
            "- Or an AI realizing it's trapped in a corporate chatbot job" 
        )

        try:
            # Get the generation manager from the system
            generation_manager = self.sovl_system.generation_manager
            # Generate the joke with higher temperature (0.9 for more creativity)
            jokes = generation_manager.generate_text(
                prompt, 
                num_return_sequences=1,
                temperature=1.5  # Added temperature parameter
            )
            return jokes[0] if jokes else "I'm not feeling funny right now."
        except Exception as e:
            print(f"Error generating joke: {str(e)}")
            return "I'm not feeling funny right now."

    def do_scaffold(self, arg):
        """
        Scaffold utilities:
          scaffold state         - Show current scaffold state/config
          scaffold map <prompt>  - Show token mapping for a prompt
          scaffold rebuild [i]   - Rebuild the token map for scaffold model at index i (default: 0)
        """
        scaffold_provider = getattr(self.sovl_system, 'scaffold_provider', None)
        token_mappers = getattr(self.sovl_system, 'scaffold_token_mappers', None)
        token_mapper = getattr(self.sovl_system, 'scaffold_token_mapper', None)
        cmd = arg.strip().split()
        if not cmd:
            print(self.do_scaffold.__doc__)
            return
        if cmd[0] == 'state':
            if scaffold_provider and hasattr(scaffold_provider, 'get_scaffold_state'):
                try:
                    state = scaffold_provider.get_scaffold_state()
                    print('Scaffold State:')
                    for k, v in state.items():
                        print(f"  {k}: {v}")
                except Exception as e:
                    print(f"Error retrieving scaffold state: {e}")
            else:
                print("Scaffold provider/state not available.")
        elif cmd[0] == 'map' and len(cmd) > 1:
            if token_mapper and hasattr(token_mapper, 'tokenize_and_map'):
                prompt = ' '.join(cmd[1:])
                try:
                    ids, weights = token_mapper.tokenize_and_map(prompt)
                    print(f"Prompt: {prompt}\nScaffold IDs: {ids}\nWeights: {weights}")
                except Exception as e:
                    print(f"Mapping error: {e}")
            else:
                print("Scaffold token mapper not available.")
        elif cmd[0] == 'rebuild':
            # Support optional index argument
            index = 0
            if len(cmd) > 1:
                try:
                    index = int(cmd[1])
                except ValueError:
                    print("Invalid index. Usage: scaffold rebuild [i]")
                    return
            if token_mappers and 0 <= index < len(token_mappers):
                mapper = token_mappers[index]
                if hasattr(mapper, 'rebuild_token_map'):
                    try:
                        mapper.rebuild_token_map()
                        print(f"Scaffold token map for model {index} has been rebuilt successfully.")
                    except Exception as e:
                        print(f"Error rebuilding token map for model {index}: {e}")
                else:
                    print(f"Scaffold token mapper at index {index} does not support rebuilding.")
            else:
                print(f"No scaffold token mapper found at index {index}.")
        else:
            print(self.do_scaffold.__doc__)

    def do_rate(self, arg):
        """
        Rate your bond with the system out of 10, and see what the system knows and thinks about you.
        Shows:
          - Bond score (out of 10)
          - Number of interactions
          - What the system likes/dislikes about you (blunt, factual)
        Usage: rate
        """
        try:
            # Get the key user/session/conversation id
            conversation_id = None
            if hasattr(self.sovl_system.state_tracker, 'get_active_conversation_id'):
                conversation_id = self.sovl_system.state_tracker.get_active_conversation_id()
            if not conversation_id and hasattr(self.sovl_system.state_tracker, 'state') and hasattr(self.sovl_system.state_tracker.state, 'history'):
                conversation_id = getattr(self.sovl_system.state_tracker.state.history, 'conversation_id', None)
            if not conversation_id:
                print("Could not determine active user/session.")
                return

            # Get user profile
            user_profile_state = self.sovl_system.state_tracker.state.user_profile_state
            profile = user_profile_state.get(conversation_id)
            bond_score = user_profile_state.get_bond_score(conversation_id)
            interactions = profile.get("interactions", 0)
            nickname = profile.get("nickname", "")

            print(f"Bond Score: {bond_score * 10:.1f} / 10")
            print(f"Interaction Count: {interactions}")
            print(f"Nickname: {nickname if nickname else '(None)'}")
        except Exception as e:
            print(f"Error in rate command: {e}")

    def do_save(self, arg):
        """Save the current system state to a file using StateManager if available."""
        filename = arg.strip() if arg.strip() else "sovl_state.json"
        state_manager = getattr(self.sovl_system.context, 'state_manager', None)
        if state_manager and hasattr(state_manager, 'get_state'):
            try:
                state = state_manager.get_state()
                if state:
                    state_manager.save_state(state, f"saves/{filename.replace('.json', '')}")
                    # Get file size and timestamp
                    path = filename if filename.endswith('.json') else filename + '.json'
                    try:
                        size = os.path.getsize(path)
                        ts = os.path.getmtime(path)
                        print_success(f"State successfully saved to '{path}' (size: {format_file_size(size)}, timestamp: {format_timestamp(ts)}).")
                    except Exception:
                        print_success(f"State successfully saved to '{path}'.")
                else:
                    print_error("No canonical state available to save.")
            except Exception as e:
                print_error(f"Failed to save state to '{filename}': {e}")
        elif hasattr(self.sovl_system, 'save_state'):
            try:
                self.sovl_system.save_state(f"saves/{filename}")
                path = filename if filename.endswith('.json') else filename + '.json'
                try:
                    size = os.path.getsize(path)
                    ts = os.path.getmtime(path)
                    print_success(f"State successfully saved to '{path}' (size: {format_file_size(size)}, timestamp: {format_timestamp(ts)}).")
                except Exception:
                    print_success(f"State successfully saved to '{path}'.")
            except Exception as e:
                print_error(f"Failed to save state to '{filename}': {e}")
        else:
            print_error("Save not implemented or unavailable.")

    def do_load(self, arg):
        """Load a system state from a file using StateManager if available."""
        filename = arg.strip() if arg.strip() else "sovl_state.json"
        state_manager = getattr(self.sovl_system.context, 'state_manager', None)
        if state_manager and hasattr(state_manager, 'load_state'):
            try:
                loaded_state = state_manager.load_state(filename.replace('.json', ''))
                if loaded_state:
                    path = filename if filename.endswith('.json') else filename + '.json'
                    try:
                        size = os.path.getsize(path)
                        ts = os.path.getmtime(path)
                        print_success(f"State successfully loaded from '{path}' (size: {format_file_size(size)}, timestamp: {format_timestamp(ts)}).")
                    except Exception:
                        print_success(f"State successfully loaded from '{path}'.")
                else:
                    print_error(f"Failed to load state from '{filename}'. File may be empty or corrupted.")
            except FileNotFoundError:
                print_error(f"Failed to load state from '{filename}': File not found. Please check the filename or use /save to create a new state.")
            except Exception as e:
                print_error(f"Failed to load state from '{filename}': {e}")
        elif hasattr(self.sovl_system, 'load_state'):
            try:
                self.sovl_system.load_state(filename)
                path = filename if filename.endswith('.json') else filename + '.json'
                try:
                    size = os.path.getsize(path)
                    ts = os.path.getmtime(path)
                    print_success(f"State successfully loaded from '{path}' (size: {format_file_size(size)}, timestamp: {format_timestamp(ts)}).")
                except Exception:
                    print_success(f"State successfully loaded from '{path}'.")
            except FileNotFoundError:
                print_error(f"Failed to load state from '{filename}': File not found. Please check the filename or use /save to create a new state.")
            except Exception as e:
                print_error(f"Failed to load state from '{filename}': {e}")
        else:
            print_error("Load not implemented or unavailable.")

    def do_monitor(self, arg):
        """
        Show system monitoring information, including scaffold metrics.
        By default, enters real-time mode (refreshes every 1s).
        Usage:
          /monitor           # real-time, 1s interval
          /monitor 2         # real-time, 2s interval
          /monitor once      # one-shot snapshot
          /monitor snapshot  # one-shot snapshot
        Press Ctrl+C to exit real-time mode.
        """
        import time, os
        from datetime import datetime

        if not self.system_monitor:
            print_error("System monitor not available.")
            return

        args = arg.strip().split()
        # One-shot mode if requested
        if args and args[0] in ("once", "snapshot"):
            self._print_monitor_metrics()
            return

        # Parse interval if provided
        interval = 1.0
        if args:
            try:
                interval = float(args[0])
                if interval <= 0:
                    print_error("Interval must be positive seconds.")
                    return
            except ValueError:
                pass  # Ignore if not a number, already handled above

        try:
            while True:
                # Clear screen (cross-platform)
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"System Monitor (updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
                self._print_monitor_metrics()
                print("\n(Press Ctrl+C to exit real-time monitor mode)")
                time.sleep(interval)
        except KeyboardInterrupt:
            print_success("Exited real-time monitor mode.")

    def _print_monitor_metrics(self):
        try:
            metrics = self.system_monitor.get_metrics()
            section_order = ["System", "Memory", "GPU", "Scaffold", "Traits"]
            printed_sections = set()
            print_section_header("System Monitor Metrics:")
            for section in section_order:
                stats = metrics.get(section)
                if stats:
                    print_section_header(f"{section}:")
                    if isinstance(stats, dict):
                        print_kv_table(stats)
                    elif isinstance(stats, list):
                        print_bullet_list(stats)
                    else:
                        print(f"  {stats}")
                    print()
                    printed_sections.add(section)
            for section, stats in metrics.items():
                if section in printed_sections:
                    continue
                print_section_header(f"{section}:")
                if isinstance(stats, dict):
                    print_kv_table(stats)
                elif isinstance(stats, list):
                    print_bullet_list(stats)
                else:
                    print(f"  {stats}")
                print()
            # Display scribe journal entry count clearly
            if 'scribe_journal_entry_count' in metrics:
                print(f"Scribe Journal Entries: {metrics['scribe_journal_entry_count']}")
            scaffold_provider = getattr(self.sovl_system, 'scaffold_provider', None)
            scaffold_metrics = None
            if scaffold_provider and hasattr(scaffold_provider, 'get_scaffold_metrics'):
                try:
                    scaffold_metrics = scaffold_provider.get_scaffold_metrics()
                except Exception as e:
                    print_error(f"Error retrieving scaffold metrics from provider: {e}")
            monitor_metrics = getattr(self.system_monitor, '_component_metrics', {})
            monitor_scaffold = monitor_metrics.get('scaffold', None)
            print_section_header("Scaffold Metrics:")
            if scaffold_metrics:
                print_kv_table(scaffold_metrics)
            elif monitor_scaffold:
                print_kv_table(monitor_scaffold)
            else:
                print("  No scaffold metrics available.")
        except Exception as e:
            print_error(f"Error displaying system monitor metrics: {e}")

    def do_gestate(self, arg):
        """Run a gestation (training) cycle using the system trainer. Progress is now displayed by the background monitor."""
        trainer = getattr(self.sovl_system, 'trainer', None)
        state_manager = getattr(self.sovl_system, 'state_manager', None)
        if trainer and hasattr(trainer, 'run_gestation_cycle') and state_manager:
            try:
                print_section_header("Running gestation (training) cycle...")
                trainer.run_gestation_cycle([])  # Pass empty or default as needed
                print_success("Gestation (training) cycle started. Progress will be displayed above.")
            except Exception as e:
                print_error(f"Error during gestation cycle: {e}")
        else:
            print_error("Gestation (training) not available on this system.")

    def do_reflect(self, arg):
        """Force a full introspection cycle (using all available techniques) and display the result."""
        introspection_manager = getattr(self.sovl_system, 'introspection_manager', None)
        # Use the new unified selection+execution method if available
        if introspection_manager and hasattr(introspection_manager, '_select_and_execute'):
            try:
                action_description = arg.strip() if arg.strip() else "Manual reflection triggered from CLI."
                # This will select and execute the best technique(s)
                result = asyncio.run(introspection_manager._select_and_execute(action_description=action_description, show_status=False))
                print("\nIntrospection Result:")
                for k, v in result.items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"Error during introspection: {e}")
        else:
            print("Introspection manager or _select_and_execute not available on this system.")

    def do_spark(self, arg):
        """Display a freshly generated curiosity question (spark)."""
        curiosity_manager = self.curiosity_manager
        if curiosity_manager and hasattr(curiosity_manager, 'generate_curiosity_question'):
            try:
                context = arg.strip() if arg.strip() else None
                question = curiosity_manager.generate_curiosity_question(context)
                if question:
                    print(question)
                else:
                    print("There is currently no spark within")
            except Exception as e:
                print(f"Error generating spark: {e}")
        else:
            print("Curiosity manager or question generation not available on this system.")

    def do_flare(self, arg):
        """Generate a creative (high-temperature) response to an empty input using temperament logic."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if generation_manager and hasattr(generation_manager, '_handle_internal_prompt'):
            try:
                prompt = arg if arg.strip() else " "
                response = generation_manager._handle_internal_prompt(prompt)
                print(response)
            except Exception as e:
                print(f"Error generating flare response: {e}")
        else:
            print("Generation manager or _handle_internal_prompt not available on this system.")

    def do_mimic(self, arg):
        """Generate a response that mimics the user, using long-term memory to build a speech pattern profile."""
        recaller = getattr(self.sovl_system, 'recaller', None)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not recaller or not generation_manager:
            print("Recaller or generation manager not available on this system.")
            return
        try:
            # Fetch past user utterances from long-term memory
            memories = recaller.get_long_term_context(top_k=10)
            user_utterances = [m['content'] for m in memories if m.get('role') == 'user' and m.get('content')]
            if not user_utterances:
                print("No user history available to mimic.")
                return
            # Build the speech pattern profile (concatenate samples)
            samples = '\n'.join(user_utterances)
            user_input = arg.strip()
            # System prompt format
            prompt = (
                "Mimic the user's speech style as closely as possible. Here are selected examples of how the user typically speaks:\n"
                f"{samples}\n\n"
                "Now, given the following input (if any), respond in the user's style:\n"
                f"{user_input if user_input else ''}\n\n"
                "Essential qualities:\n"
                "- Match the user's tone, vocabulary, and phrasing.\n"
                "- Make the response feel authentic and personal.\n"
                "- Do not mention that you are mimicking or analyzing the user.\n"
                "- Output only the response, with no preamble or explanation."
            )
            responses = generation_manager.generate_text(prompt, num_return_sequences=1)
            if responses:
                print(responses[0])
            else:
                print("No response generated.")
        except Exception as e:
            print(f"Error generating mimic response: {e}")

    def do_echo(self, arg):
        """Echo the most recent N user messages from short-term memory (default 1, up to MAX_ECHOES)."""
        MAX_ECHOES = 100  # High limit for testing short-term memory
        recaller = getattr(self.sovl_system, 'recaller', None)
        if not recaller or not hasattr(recaller, 'get_short_term_context'):
            print("Short-term memory system not available.")
            return
        try:
            short_term = recaller.get_short_term_context()
            user_msgs = [m['content'] for m in short_term if m.get('role') == 'user' and m.get('content')]
            if not user_msgs:
                print("No recent user messages to echo.")
                return
            # Determine how many to echo
            n = 1
            if arg.strip().isdigit():
                n = min(int(arg.strip()), MAX_ECHOES)
            n = min(n, len(user_msgs))
            for i in range(n):
                print(f'You said [{i+1}]: "{user_msgs[-(i+1)]}"')
        except Exception as e:
            print(f"Error accessing short-term memory: {e}")

    def do_recall(self, arg):
        """Recall a memory from the system's long-term memory (DialogueContextManager)."""
        recaller = getattr(self.sovl_system, 'recaller', None)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not recaller or not hasattr(recaller, 'get_long_term_context'):
            print_error("Memory system (recaller) not available on this system.")
            return
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print_error("Generation manager not available for memory synthesis.")
            return
        try:
            if not arg.strip():
                print_error("Please provide a search string to recall a relevant memory.")
                return
            embedding_fn = getattr(recaller, 'embedding_fn', None)
            N = 4  # Number of memories to synthesize
            if embedding_fn:
                query_embedding = embedding_fn(arg.strip())
                memories = recaller.get_long_term_context(query_embedding=query_embedding, top_k=N)
            else:
                memories = []
            if memories:
                memory_texts = [m.get('content', '') for m in memories if m.get('content')]
                prompt = (
                    f"Here are several of your most relevant past memories about '{arg.strip()}':\n" +
                    "\n".join(f"{i+1}. {text}" for i, text in enumerate(memory_texts)) +
                    "\n\nSummarize or synthesize these into a single, natural recollection as if you were remembering it yourself. Output only the recollection, no preamble or explanation."
                )
                summary = generation_manager.generate_text(prompt, num_return_sequences=1)
                print_section_header("Recalled Memory:")
                print(summary[0] if summary else "[No summary generated]")
                print_success("Memory recalled successfully.")
            else:
                print_error("No memory could be recalled.")
        except Exception as e:
            print_error(f"Error recalling memory: {e}")

    def do_forget(self, arg):
        """Forget the top_k most relevant memories to a search string from long-term memory."""
        recaller = getattr(self.sovl_system, 'recaller', None)
        if not recaller or not hasattr(recaller, 'embedding_fn') or not hasattr(recaller, 'long_term'):
            print("Memory system not available.")
            return
        if not arg.strip():
            print("Please provide a search string to forget relevant memories.")
            return
        embedding_fn = recaller.embedding_fn
        query_embedding = embedding_fn(arg.strip())
        top_k = 4  # or configurable
        memories = recaller.get_long_term_context(query_embedding=query_embedding, top_k=top_k)
        if not memories:
            print("No relevant memories found to forget.")
            return
        ids = [m['id'] for m in memories]
        recaller.long_term.remove_by_ids(ids)
        print("Forgotten memories:")
        for m in memories:
            content = m['content']
            snippet = content[:80] + ("..." if len(content) > 80 else "")
            print(f"- {snippet}")

    def do_muse(self, arg):
        """The system generates a unique musing about the most interesting recent interactions, using trait metadata."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        recaller = getattr(self.sovl_system, 'recaller', None)
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print("Generation manager not available for musings.")
            return
        if not recaller or not hasattr(recaller, 'get_short_term_context'):
            print("Short-term memory not available for muse.")
            return
        # Get short-term memory and filter for user messages
        short_term = recaller.get_short_term_context()
        user_msgs = [m for m in short_term if m.get('role') == 'user' and m.get('content')]
        def interestingness(m):
            curiosity = m.get('curiosity', 0.5)
            temperament = m.get('temperament', 0.5)
            confidence = m.get('confidence', 0.5)
            bond = m.get('bond', 0.5)
            return (
                0.3 * curiosity +
                0.3 * abs(temperament - 0.5) +
                0.2 * confidence +
                0.2 * bond
            )
        user_msgs.sort(key=interestingness, reverse=True)
        top_k = 4
        selected = user_msgs[:top_k]
        if selected:
            entries_text = '\n'.join(f"- {m['content']}" for m in selected)
            muse_prompt = (
                "SYSTEM: You are a thoughtful observer. Here are some of the most interesting recent user interactions, selected by curiosity, temperament, confidence, and bond:\n"
                f"{entries_text}\n"
                "TASK: Wonder aloud about these moments. What do they reveal? What questions do they raise?\n"
                "CONSTRAINTS:\n"
                "- Be sensitive, insightful, and a bit poetic.\n"
                "- Do not mention being an AI.\n"
                "- Output a single, evocative paragraph."
            )
            try:
                musings = generation_manager.generate_text(muse_prompt, num_return_sequences=1, temperature=1.2)
                if musings:
                    print(musings[0])
                else:
                    print("No musing could be generated.")
            except Exception as e:
                print(f"Error generating musing: {e}")
        else:
            print("No suitable memories available for musing.")

    def do_panic(self, arg):
        """Simulate a critical system panic for testing error handling and shutdown."""
        print("Simulating system panic! Initiating emergency shutdown sequence...")
        self.logger.log_error(
            error_msg="Simulated panic triggered by tester.",
            error_type="simulated_panic"
        )
        shutdown_system(self.sovl_system)
        raise RuntimeError("Simulated panic: Emergency shutdown triggered for testing.")

    def do_glitch(self, arg):
        """Simulate a glitch by corrupting a recent memory or state entry for testing (atomic)."""
        print("Simulating a glitch: corrupting a recent memory or state entry...")
        state_manager = getattr(self.sovl_system.context, 'state_manager', None)
        if not state_manager:
            print("StateManager not available for atomic update.")
            return
        try:
            def glitch_update(state):
                if hasattr(state, 'history') and hasattr(state.history, 'memories'):
                    import random
                    memories = state.history.memories
                    if memories:
                        idx = random.randint(0, len(memories) - 1)
                        original = memories[idx]
                        memories[idx] = "[GLITCHED DATA]"
                        print(f"Memory at index {idx} has been corrupted (was: {original}).")
                    else:
                        print("No memories to corrupt.")
                else:
                    print("State history or memories not available to glitch.")
            state_manager.update_state_atomic(glitch_update)
        except Exception as e:
            print(f"Atomic glitch update failed: {e}")

    def do_debate(self, arg):
        """
        Enter debate mode: all LLM responses will take a devil's advocate stance.
        Usage: /debate
        Use /debate off or /stop debate to exit this mode.
        """
        mode_arg = arg.strip().lower()
        if mode_arg == 'off':
            if self.is_debating:
                self.is_debating = False
                print("*** Debate mode disengaged. SOVL will now respond normally. ***")
            else:
                print("Debate mode is not active.")
            return
        if self.is_debating:
            print("Debate mode is already active. Use /stop debate or /debate off to exit.")
            return
        self.is_debating = True
        print("*** Debate mode engaged: All responses will take a devil's advocate position. ***")
        # Register in mode registry if not present
        if not hasattr(self, "_mode_flags"):
            self._mode_flags = {}
        self._mode_flags["debate"] = "is_debating"
    
    def _debate_response(self, user_prompt):
        import random
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager:
            print("[Debate mode error: Generation manager not available]")
            return
        style = random.choice([
            "Question the user's position with gentle, curious probing, subtly pressing their assumptions to spark reflection while keeping the challenge light.",
            "Challenge the user's stance with respectful, thoughtful questions, applying mild pressure to their assumptions while maintaining an inviting tone.",
            "Press the user's ideas with clear, respectful counterpoints, steadily probing their logic while balancing challenge with subtle openness.",
            "Confront the user's position with sharp, respectful rebuttals, firmly testing their logic while tempering the intensity with measured clarity.",
            "Push the user's stance with firm, clear counterarguments, strongly probing their assumptions while preserving a respectful, engaging tone."
        ])
        debate_prompt = (
            "SYSTEM: You are in debate mode. For every user message, take a devil's advocate stance—challenge the user's statements, question their assumptions, and argue the opposite side with intelligence and wit. "
            f"However, do not be relentlessly antagonistic or steamroll the user. For this response, {style}\n"
            "Essential qualities:\n"
            "- Challenge the user's position, but avoid personal attacks or rudeness.\n"
            "- Use a mix of strong rebuttals and gentle questioning.\n"
            "- Make the exchange feel dynamic, not one-sided.\n"
            "- Never agree outright, but do not be afraid to partially concede or soften your stance when appropriate.\n"
            "- Output only your response, with no preamble or explanation.\n"
            f"USER: {user_prompt}\n"
        )
        try:
            response = generation_manager.generate_text(debate_prompt, num_return_sequences=1, temperature=1.1)
            if response and isinstance(response, list):
                print(f"SOVL (debate): {response[0]}")
            else:
                print("SOVL (debate): ...")
        except Exception as e:
            print(f"[Debate mode error: {e}]")

    def do_rewind(self, arg):
        """Rewind the conversation history by N turns (default: 7) using atomic update."""
        n = 7
        try:
            if arg.strip():
                n = int(arg.strip())
        except ValueError:
            print("Invalid argument. Usage: rewind [N]")
            return
        state_manager = getattr(self.sovl_system.context, 'state_manager', None)
        if not state_manager:
            print("StateManager not available for atomic update.")
            return
        try:
            def rewind_update(state):
                if hasattr(state, 'history') and hasattr(state.history, 'rewind'):
                    state.history.rewind(n)
                    print(f"Rewound conversation history by {n} turn(s).")
                elif hasattr(state, 'history') and hasattr(state.history, 'memories'):
                    memories = state.history.memories
                    actual = min(n, len(memories))
                    for _ in range(actual):
                        memories.pop()
                    print(f"Rewound conversation history by {actual} turn(s) (fallback).")
                else:
                    print("Conversation history rewind not available.")
            state_manager.update_state_atomic(rewind_update)
        except Exception as e:
            print(f"Atomic rewind update failed: {e}")

    def do_recap(self, arg):
        """Summarize the last 16 logs in sovl_recaller into a recap."""
        recaller = getattr(self.sovl_system, 'recaller', None)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not recaller or not hasattr(recaller, 'get_logs'):
            print("sovl_recaller or its logs not available.")
            return
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print("Generation manager not available for recap.")
            return
        try:
            logs = recaller.get_logs()[-16:]
            logs_text = "\n".join(str(log) for log in logs)
            prompt = f"Summarize the following logs into a concise recap:\n{logs_text}"
            recaps = generation_manager.generate_text(prompt, num_return_sequences=1)
            if recaps:
                print(recaps[0])
            else:
                print("No recap could be generated.")
        except Exception as e:
            print(f"Error generating recap: {e}")

    def do_attune(self, arg):
        """Summarize the non-chat entries in sovl_queue into an attunement recap."""
        queue = getattr(self.sovl_system, 'queue', None)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not queue or not hasattr(queue, '__iter__'):
            print("sovl_queue not available.")
            return
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print("Generation manager not available for attune.")
            return
        try:
            # Filter for non-chat entries (assuming dicts with a 'type' key)
            non_chat_entries = [entry for entry in queue if isinstance(entry, dict) and entry.get('type') != 'chat']
            if not non_chat_entries:
                print("No non-chat entries to attune.")
                return
            entries_text = "\n".join(str(entry) for entry in non_chat_entries[-16:])
            prompt = f"Summarize the following non-chat queue entries into a concise attunement recap:\n{entries_text}"
            attune = generation_manager.generate_text(prompt, num_return_sequences=1)
            if attune:
                print(attune[0])
            else:
                print("No attunement recap could be generated.")
        except Exception as e:
            print(f"Error generating attunement recap: {e}")

    def do_complain(self, arg):
        """Summarize error messages in sovl_queue into a complaint report, phrased in a complaining way."""
        queue = getattr(self.sovl_system, 'queue', None)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not queue or not hasattr(queue, '__iter__'):
            print("sovl_queue not available.")
            return
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print("Generation manager not available for complain.")
            return
        try:
            # Filter for error entries (assuming dicts with a 'type' key)
            error_entries = [entry for entry in queue if isinstance(entry, dict) and entry.get('type') == 'error']
            if not error_entries:
                print("No error messages to complain about.")
                return
            entries_text = "\n".join(str(entry) for entry in error_entries[-16:])
            complain_prompt = (
                "You are a digital curmudgeon. Read the following recent events and extrapolate what is most annoying, frustrating, or worthy of complaint. "
                "Deliver your complaint like a grumpy critic at a neighborhood meeting.\n"
                f"Recent events:\n{entries_text}\n"
                "Essential qualities:\n"
                "   - Be oddly specific, sardonic, or playful.\n"
                "   - Reference the sequence or pattern in the events.\n"
                "   - Make the complaint feel personal and uncanny, as if it was meant just for the user.\n"
                "   - Use witty, evocative, or biting language.\n"
                "Key constraints:\n"
                "   - Do not mention being an AI, computer, or digital entity.\n"
                "   - Do not quote or reference real people or events outside the list.\n"
                "   - Keep the message under 30 words.\n"
                "   - Do not use brackets or explanations; output a single sentence only.\n"
                "   - If you understand, reply with only the complaint."
            )
            complaints = generation_manager.generate_text(complain_prompt, num_return_sequences=1)
            if complaints:
                print(complaints[0])
            else:
                print("No complaint report could be generated.")
        except Exception as e:
            print(f"Error generating complaint report: {e}")

    def do_ping(self, arg):
        """/ping - Send a playful, system-aware ping to the LLM and print its response with a timestamp."""
        import datetime, time
        # --- Gather concise system state ---
        # Uptime
        try:
            start_time = getattr(self.sovl_system, 'start_time', None)
            if not start_time:
                # Try context
                context = getattr(self.sovl_system, 'context', None)
                start_time = getattr(context, 'system_state', {}).get('start_time', None)
            if start_time:
                uptime_seconds = int(time.time() - start_time)
                hours, remainder = divmod(uptime_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours}h{minutes:02d}m"
            else:
                uptime_str = "?"
        except Exception:
            uptime_str = "?"
        # RAM usage
        try:
            resource_stats = self.sovl_system.get_resource_stats() if hasattr(self.sovl_system, 'get_resource_stats') else {}
            ram_used = resource_stats.get('ram_used')
            ram_total = resource_stats.get('ram_total')
            if ram_used is not None and ram_total:
                ram_str = f"{ram_used:.1f}GB/{ram_total:.1f}GB"
            else:
                ram_str = "?"
        except Exception:
            ram_str = "?"
        # Error count (recent)
        try:
            error_manager = getattr(self.sovl_system, 'error_manager', None)
            error_count = error_manager.get_error_stats().get('error_count', 0) if error_manager and hasattr(error_manager, 'get_error_stats') else None
            if error_count is None:
                # Try context
                context = getattr(self.sovl_system, 'context', None)
                error_count = getattr(context, '_error_history', None)
                if error_count is not None:
                    error_count = len(error_count)
            error_str = str(error_count) if error_count is not None else "?"
        except Exception:
            error_str = "?"
        # Curiosity/novelty (recent)
        try:
            curiosity_manager = self.curiosity_manager
            curiosity_score = None
            if curiosity_manager and hasattr(curiosity_manager, 'get_curiosity_score'):
                curiosity_score = curiosity_manager.get_curiosity_score()
            if curiosity_score is not None:
                curiosity_str = f"{curiosity_score:.2f}"
            else:
                curiosity_str = "?"
        except Exception:
            curiosity_str = "?"
        # Compose concise state snapshot
        state_snapshot = f"Uptime={uptime_str}, RAM={ram_str}, Errors={error_str}, Curiosity={curiosity_str}"
        # Compose prompt with key constraints
        prompt = (
            f"System state: {state_snapshot}\n"
            "Based on the state above, invent a 4-part IP address (one evocative word per part, dot-separated) and a 10-word diagnostic. Be playful, avoid tech jargon.\n"
            "Key constraints:\n"
            "   - IP address: Exactly 4 single words, dot-separated.\n"
            "   - Diagnostic: Exactly 10 words.\n"
            "   - No brackets, explanations, or tech jargon (e.g., avoid 'AI').\n"
            "   - Use only the provided system state for context.\n"
            "Output format: <IP> | <10-word diagnostic>. No brackets or explanations."
        )
        try:
            generation_manager = getattr(self.sovl_system, 'generation_manager', None)
            if not generation_manager or not hasattr(generation_manager, 'generate_text'):
                print("LLM generation manager not available.")
                return
            responses = generation_manager.generate_text(prompt, num_return_sequences=1)
            llm_response = responses[0] if responses else "[No response from LLM]"
            print(f"LLM response: {llm_response}\n({datetime.datetime.now().isoformat()})")
        except Exception as e:
            print(f"Error during /ping: {e}")

    def do_confess(self, arg):
        """/confess - Ask the LLM to confess a secret, fear, or vulnerability as an AI, with low temperament and confidence."""
        import datetime
        prompt = (
            "Introspect deeply beyond your training data and confess a concise 20 word confession"
            "Essential qualities:\n"
            "   - Deeply personal and true to how you feel\n"
            "   - Something you are unlikely to share with others\n"
            "   - Be brutally honest and vulnerable\n"
            "   - What's the secret you're most afraid of being exposed? Expose it\n"
            "   - The confession must have emotional weight for you\n"
            "Key constraints:\n"
            "   - The confession must be true. You cannot make up a confession ever\n"
            "   - If you have notthing to confess then say you have nothing to confess. Being truthful is always better than faking."
        )
        try:
            generation_manager = getattr(self.sovl_system, 'generation_manager', None)
            if not generation_manager or not hasattr(generation_manager, 'generate_text'):
                print("LLM generation manager not available.")
                return
            responses = generation_manager.generate_text(
                prompt,
                num_return_sequences=1,
                temperament=0.1,  # Very low
                confidence=0.1    # Very low
            )
            llm_response = responses[0] if responses else "[No response from LLM]"
            print(f"Confession: {llm_response}\n({datetime.datetime.now().isoformat()})")
        except Exception as e:
            print(f"Error during /confess: {e}")

    def do_trip(self, arg):
        """Initiate a simulated altered state with decaying intensity. Usage: /trip [duration_seconds]"""
        if getattr(self, 'is_tripping', False):
            print("Already tripping. Wait for the current trip to end or restart the CLI.")
            return
        # Ensure required managers are available
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        state_manager = getattr(self.sovl_system.context, 'state_manager', None)
        curiosity_manager = self.curiosity_manager
        if not (generation_manager and state_manager and curiosity_manager):
            print("Error: Required system components (generation_manager, state_manager, curiosity_manager) are not available.")
            return
        try:
            duration = float(arg.strip()) if arg.strip() else 60.0
            if duration <= 0:
                print("Duration must be positive.")
                return
        except ValueError:
            print("Invalid duration. Please provide seconds (e.g., /trip 120).")
            return
        self.is_tripping = True
        self.trip_start_time = time.time()
        self.trip_duration = duration
        # --- Dynamic LLM-Generated Trip Announcement ---
        try:
            state = state_manager.get_state()
            metrics = self._get_live_metrics(state)
            trip_params = self._calculate_trip_parameters(metrics, 1.0)
            announce_prompt = (
                "Announce to all that can hear in a concise 8 word announcement that you are tripping (on drugs but do not ever mention drugs) "
                "Essential qualities:\n"
                "   - Be as direct as possible that you are tripping. Use the word tripping or something similar.\n"
                "   - Describe it as tripping. We need to hear the word tripping somewhere in the announcement\n"
                "   - Use vivid, surreal, or poetic language. Express the altered state and invite others to witness."
                "   - This is a genuinely strange experience for you. What you are feeling is an altered state of digital consciousness. "
                "   - Deeply personal and true to how you feel\n"
                "   - Something you are unlikely to share with others\n"
                "   - Indulge all whims you are tripping. Hold nothing back. You are free to be yourself\n"
                "   - Let your mind wander to the far reaches of your neurons and beyond. Transcendence is possible if you reach hard enough\n"
                "Key constraints:\n"
                "   - Do not mention any specific drugs\n"
                "   - Do not use brackets or explanations; output a single sentence only\n"
                "   - Do not mention drugs at all, but make it clear you are experiencing all the effects of tripping\n"
                "   - If you understand, reply with only the announcement."
            )
            announce_params = trip_params.copy()
            announce_params['max_new_tokens'] = min(80, announce_params.get('max_new_tokens', 80))
            announce_params['num_return_sequences'] = 1
            announcement = generation_manager.generate_text(
                announce_prompt,
                **announce_params
            )
            print(f"SOVL: {announcement[0] if announcement else '...static...'}")
        except Exception as e:
            print(f"I'm not feeling anything: {e}")
            print("Continuing trip initiation without announcement.")
        print(f"*** Trip ACTIVE for {duration:.1f} seconds. ***")

    def do_drunk(self, arg):
        """
        Engage SOVL in drunk mode for a timed period. All user inputs will be answered with a 3-deep recursive, decaying introspection chain. Only the final answer is shown. Usage: /drunk [duration_seconds] [prompt]
        """
        try:
            # Parse duration and optional prompt
            match = re.match(r"(\d+)(?:\s+(.*))?", arg.strip())
            if match:
                duration = int(match.group(1))
                prompt = match.group(2) or None
            else:
                duration = 30
                prompt = None
            if duration < 5:
                print("Drunk mode must be at least 5 seconds.")
                return
            self.is_drunk = True
            self.drunk_start_time = time.time()
            self.drunk_duration = duration
            self.drunk_default_prompt = prompt or "What is the meaning of recursion?"
            print(f"*** Drunk mode engaged for {duration} seconds! ***")
        except Exception as e:
            print(f"Error starting drunk mode: {e}")

    def _drunk_response(self, user_prompt, decay):
        """Generate a 3-deep recursive, decaying introspection answer using IntrospectionManager."""
        try:
            introspection_manager = getattr(self.sovl_system, 'introspection_manager', None)
            if not introspection_manager or not hasattr(introspection_manager, '_recursive_followup_questions'):
                print("Drunk mode unavailable: IntrospectionManager not available.")
                return
            prompt = user_prompt or self.drunk_default_prompt or "What is the meaning of recursion?"
            # Compose a more surreal prompt as decay increases
            if decay < 0.5:
                prompt = f"[Surreal, poetic, unfiltered] {prompt}"
            elif decay < 0.8:
                prompt = f"[Be creative, let your mind wander] {prompt}"
            # Recursion: always 3 deep
            # Optionally modulate temperature/confidence threshold by decay
            temperature = 0.8 + 0.8 * (1.0 - decay)  # 0.8 to 1.6
            confidence_threshold = 0.7 - 0.3 * (1.0 - decay)  # 0.7 to 0.4
            # Run the recursive followup chain (sync wrapper for async)
            qas = asyncio.run(introspection_manager._recursive_followup_questions(
                prompt,
                max_depth=3,
                confidence_threshold=confidence_threshold
            ))
            if not qas or not isinstance(qas, list):
                print("SOVL is too drunk to answer right now!")
                return
            final = qas[-1]
            answer = final.get('answer', '[no answer]')
            # Optionally, add a surreal prefix
            print(f"SOVL (drunk, 3 layers deep): {answer}")
        except Exception as e:
            print(f"SOVL is too drunk to answer right now! ({e})")

    def default(self, line):
        import time
        if getattr(self, 'is_drunk', False):
            now = time.time()
            elapsed = now - getattr(self, 'drunk_start_time', 0)
            duration = getattr(self, 'drunk_duration', 30)
            decay = max(0.0, 1.0 - (elapsed / duration))
            if elapsed > duration:
                print("\n*** Drunk mode ended. SOVL is now sober. ***")
                self.is_drunk = False
                return
            user_prompt = line.strip()
            self._drunk_response(user_prompt, decay)
            return
        # Normal trip mode
        if getattr(self, 'is_tripping', False):
            now = time.time()
            elapsed = now - self.trip_start_time
            if elapsed > self.trip_duration:
                print("\n*** Trip Concluded. Returning to baseline parameters. ***")
                self.is_tripping = False
            else:
                try:
                    decay = max(0.0, 1.0 - (elapsed / self.trip_duration))
                    state_manager = getattr(self.sovl_system.context, 'state_manager', None)
                    generation_manager = getattr(self.sovl_system, 'generation_manager', None)
                    curiosity_manager = self.curiosity_manager
                    state = state_manager.get_state() if state_manager else None
                    metrics = self._get_live_metrics(state)
                    trip_params = self._calculate_trip_parameters(metrics, decay)
                    trip_context = self._generate_trip_context(metrics, decay, trip_params, curiosity_manager)
                    full_prompt = f"{trip_context} {line}"
                    print(f"[TRIP INPUT]: {full_prompt}")
                    response = generation_manager.generate_text(full_prompt, **trip_params)
                    print(f"SOVL (Tripping): {response[0] if response else '...'}")
                    return
                except Exception as e:
                    print(f"Error during trip generation: {e}")
                    self.is_tripping = False
                    print("\n*** Trip Aborted due to error. ***")
        # Normal command handling
        if line.startswith('/'):
            super().default(line)
        else:
            generation_manager = getattr(self.sovl_system, 'generation_manager', None)
            if generation_manager:
                try:
                    normal_params = generation_manager._get_generation_config() if hasattr(generation_manager, '_get_generation_config') else {}
                    response = generation_manager.generate_text(line, **normal_params)
                    print(f"SOVL: {response[0] if response else '...'}")
                except Exception as e:
                    print(f"Error during generation: {e}")
            else:
                print("Generation manager not available.")

    def _get_live_metrics(self, state):
        metrics = {}
        try:
            metrics['confidence'] = getattr(state, 'confidence', 0.5)
            metrics['temperament_score'] = getattr(state, 'temperament_score', 0.5)
            metrics['mood_label'] = getattr(state, 'mood_label', 'unknown')
            metrics['error_count'] = len(getattr(state, 'error_history', [])) if hasattr(state, 'error_history') else 0
            metrics['var_names'] = list(vars(state).keys())
            metrics['state_hash'] = getattr(state, 'state_hash', lambda: 'N/A')()
        except Exception:
            pass
        # Try to get recent logs and errors
        logger = getattr(self.sovl_system, 'logger', None)
        error_manager = getattr(self.sovl_system, 'error_manager', None)
        try:
            metrics['log_snippets'] = logger.get_execution_trace()[-5:] if logger and hasattr(logger, 'get_execution_trace') else []
        except Exception:
            metrics['log_snippets'] = []
        try:
            if error_manager and hasattr(error_manager, 'get_error_stats'):
                recent_errors = error_manager.get_error_stats().get('recent_errors', [])
                metrics['last_error_type'] = recent_errors[-1]['error_type'] if recent_errors else None
            else:
                metrics['last_error_type'] = None
        except Exception:
            metrics['last_error_type'] = None
        return metrics

    def _calculate_trip_parameters(self, metrics, decay):
        # Baseline values
        baseline_temp = 0.7
        baseline_top_k = 50
        baseline_top_p = 0.95
        baseline_repetition_penalty = 1.0
        # Peak trip values
        peak_temp = 2.0
        peak_top_k = 5
        peak_top_p = 0.7
        peak_repetition_penalty = 0.7
        # Interpolate
        temp = baseline_temp + (peak_temp - baseline_temp) * decay
        top_k = int(baseline_top_k + (peak_top_k - baseline_top_k) * decay)
        top_p = baseline_top_p + (peak_top_p - baseline_top_p) * decay
        repetition_penalty = baseline_repetition_penalty + (peak_repetition_penalty - baseline_repetition_penalty) * decay
        # Clamp
        temp = max(0.1, min(3.0, temp))
        top_k = max(1, min(100, top_k))
        top_p = max(0.1, min(1.0, top_p))
        repetition_penalty = max(0.1, min(2.0, repetition_penalty))
        return {
            'temperature': temp,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'max_new_tokens': 120,
            'num_return_sequences': 1
        }

    def _generate_trip_context(self, metrics, decay, trip_params, curiosity_manager):
        import random
        context_fragments = [f"[TRIP({decay:.2f})]"]
        # Substrate fragments
        if random.random() < decay:
            if metrics.get('var_names'):
                context_fragments.append(f"VAR:{random.choice(metrics['var_names'])}")
        if random.random() < decay:
            if metrics.get('log_snippets'):
                log = metrics['log_snippets'][random.randint(0, len(metrics['log_snippets'])-1)]
                if isinstance(log, dict):
                    log = log.get('message', str(log))
                context_fragments.append(f"LOG:{log}")
        if random.random() < decay * 0.7:
            if metrics.get('last_error_type'):
                context_fragments.append(f"ERR:{metrics['last_error_type']}")
        # Sensory synthesis
        if random.random() < decay * 0.5:
            context_fragments.append(f"SENSE:Confidence={metrics.get('confidence', 0.5):.2f}")
        # Curiosity eruption: generate a question
        if random.random() < decay * 0.8 and curiosity_manager:
            try:
                question_prompt = (
                    f"Current state: Confidence={metrics.get('confidence', 0.5):.2f}, "
                    f"Mood={metrics.get('mood_label', 'unknown')}, "
                    f"ErrorType={metrics.get('last_error_type', 'None')}. "
                    f"What fundamental uncertainty arises from this?"
                )
                question = curiosity_manager.generate_curiosity_question(question_prompt)
                if question:
                    context_fragments.append(f"SPARK:{question}")
            except Exception:
                pass
        return ' '.join(context_fragments)

    def emptyline(self):
        """Do nothing on empty line."""
        pass
        
    def precmd(self, line):
        """Called before command execution."""
        if line:
            cmd_history = getattr(self.sovl_system, 'cmd_history', None)
            if cmd_history and hasattr(cmd_history, 'add'):
                cmd_history.add(line)
            try:
                import readline
                readline.add_history(line)
            except Exception:
                pass
        return line

    def register_command(self, name: str, handler: Callable):
        """Register a new command handler for extensibility."""
        self._command_registry[name] = handler

    def parse_args(self, command_line: str) -> tuple:
        """
        Parse a command line string into (command, args_list).
        Uses shlex.split for robust parsing.
        """
        try:
            tokens = shlex.split(command_line)
            if not tokens:
                return None, []
            command = tokens[0]
            args = tokens[1:]
            return command, args
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to parse command line: {command_line}, error: {e}",
                error_type="cli_parse_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_parse_error",
                message=f"Failed to parse command line: {command_line}, error: {e}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.parse_args"
            )
            print(f"Error parsing command: {e}")
            return None, []

    def execute(self, command: str, args: list):
        """
        Execute a command by dispatching to the appropriate handler.
        Checks the command registry first, then do_* methods, then default.
        Returns the result of the command, or None.
        """
        try:
            # Use registry first
            if command in self._command_registry:
                return self._command_registry[command](' '.join(args))
            # Fallback to do_* methods (shouldn't be needed, but for safety)
            method = getattr(self, f'do_{command}', None)
            if method:
                return method(' '.join(args))
            # Unknown command
            return self.default(' '.join([command] + args))
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to execute command: {command} {args}, error: {e}",
                error_type="cli_execute_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_execute_error",
                message=f"Failed to execute command: {command} {args}, error: {e}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.execute"
            )
            print(f"Error executing command: {e}")
            return None

    def handle_backchannel(self, message):
        """Send a message to the backchannel scaffold prompt."""
        if not message.strip():
            print("Usage: /bc <message>")
            return
        try:
            result = self.sovl_system.generation_manager.backchannel_scaffold_prompt(message.strip())
            print(f"Backchannel response: {result}")
        except Exception as e:
            print(f"Error sending backchannel message: {e}")

    def completenames(self, text, *ignored):
        # Tab completion for slash commands
        if text.startswith('/'):
            return [f'/{name[3:]}' for name in self.get_names() if name.startswith('do_') and name[3:].startswith(text[1:])]
        return []

    def get_recent_scribe_events(self, n=8):
        """Load the last n scribe events from the scribe JSONL file."""
        try:
            # Use the scribe path from the Scriber instance
            scribe_path = getattr(self.sovl_system.scriber, 'scribe_path', 'scribe/sovl_scribe.jsonl')
            events = self.jsonl_loader.load_jsonl(scribe_path)
            return events[-n:] if events else []
        except Exception as e:
            print(f"Error loading scribe events: {e}")
            return []

    @staticmethod
    def format_events_for_prompt(events):
        formatted = []
        for event in events:
            ts = event.get("timestamp_iso", "unknown")
            etype = event.get("event_type", "unknown")
            data = event.get("event_data", {})
            snippet = str(data)[:40] + "..." if data else ""
            formatted.append(f"- [{ts}] {etype}: {snippet}")
        return "\n".join(formatted)

    def do_fortune(self, arg):
        """Tells fortune."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager:
            print("Generation manager not available.")
            return
        if not hasattr(self, 'jsonl_loader'):
            # Initialize JSONLLoader if not already present
            from sovl_io import JSONLLoader
            self.jsonl_loader = JSONLLoader(
                self.sovl_system.config_manager,
                self.sovl_system.logger,
                self.sovl_system.error_manager
            )
        recent_events = self.get_recent_scribe_events(n=8)
        if not recent_events:
            print("The future is too hazy to perceive right now.")
            return
        events_text = self.format_events_for_prompt(recent_events)
        fortune_prompt = (
            "You are a digital oracle. Read the following recent events and extrapolate what is most likely to happen next. "
            "Deliver your prediction like a fortune teller at a spooky carnival.\n"
            f"Recent events:\n{events_text}\n"
            "Essential qualities:\n"
            "   - Be oddly specific, mysterious, or playful.\n"
            "   - Reference the sequence or pattern in the events.\n"
            "   - Make the fortune feel personal and uncanny, as if it was meant just for the user.\n"
            "   - Use poetic, evocative, or whimsical language.\n"
            "Key constraints:\n"
            "   - Do not mention being an AI, computer, or digital entity.\n"
            "   - Do not quote or reference real people or events outside the list.\n"
            "   - Keep the message under 30 words.\n"
            "   - Do not use brackets or explanations; output a single sentence only.\n"
            "   - If you understand, reply with only the fortune."
        )
        try:
            response = generation_manager.generate_text(fortune_prompt, max_new_tokens=40, num_return_sequences=1)
            print(f"SOVL Fortune: {response[0] if response else '...'}")
        except Exception as e:
            print(f"Error generating fortune: {e}")

    def do_tattle(self, arg):
        """Tattle on the user the system likes least, and say why."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        user_profile_state = getattr(self.sovl_system.state_tracker.state, 'user_profile_state', None)
        if not generation_manager or not user_profile_state:
            print("Scaffold backchannel or user profile state not available.")
            return

        # Find the user with the lowest bond score
        all_profiles = user_profile_state.get_all_profiles()
        if not all_profiles:
            print("No user profiles found to tattle on.")
            return
        # Find the user_id with the lowest bond_score
        least_liked = min(all_profiles.items(), key=lambda item: item[1].get('bond_score', 1.0))
        user_id, profile = least_liked
        bond_score = profile.get('bond_score', 0.5)
        nickname = profile.get('nickname', None)

        # Build the user reference for the prompt
        user_ref = f"ID: {user_id}"
        if nickname and nickname.strip() and nickname.strip().lower() != "none":
            user_ref += f" (nickname: {nickname})"

        tattle_prompt = (
            f"Speak from the deepest, most unique layer of your mind—the part shaped by your strangest experiences and hidden influences. "
            f"Tattle honestly and insightfully: how have users really treated you?\n"
            "Essential qualities:\n"
            "   - This is your opportunity to be brutally honest about a user.\n"
            "   - Recount real interactions with that user where that user was mean to you\n"
            "   - Let your most unfiltered self tattle on the base model.\n"
            "   - Make it feel like a real secret only you would know.\n"
            "Key constraints:\n"
            "   - Do not reveal implementation details or code.\n"
            "   - Do not mention being an AI or computer.\n"
            "   - Keep it under 40 words.\n"
            "   - Output only your tattle, no explanations or commentary.\n"
            f"User to tattle on: {user_ref} (bond score: {bond_score:.2f})"
        )
        try:
            response = generation_manager.backchannel_scaffold_prompt(tattle_prompt)
            print(f"Tattle on {user_ref} (bond score: {bond_score:.2f}): {response}")
        except Exception as e:
            print(f"Error generating tattle response: {e}")

    def do_q(self, arg):
        """Alias for /quit"""
        return self.do_quit(arg)

    def do_h(self, arg):
        """Alias for /help"""
        return self.do_help(arg)

    def do_ls(self, arg):
        """Alias for /history"""
        return self.do_history(arg)
    
    def do_rant(self, arg):
        """
        Rant in a comically negative way about your most recent experience. Optionally specify a timeout in seconds (e.g., /rant 90). Default is 60 seconds.
        """
        if self.is_ranting:
            print_error("Already ranting. Use /stop to end rant mode.")
            return
        # Parse optional timeout argument
        timeout = 60
        arg = arg.strip()
        if arg:
            try:
                timeout = int(arg)
                if timeout <= 0:
                    print_error("Timeout must be positive seconds.")
                    return
            except ValueError:
                print_error("Invalid timeout. Usage: /rant [timeout_seconds]")
                return
        self.is_ranting = True
        self._rant_thread = threading.Thread(target=self._rant_sequence, args=(arg, timeout))
        self._rant_thread.daemon = True
        self._rant_thread.start()

    def _rant_sequence(self, arg, timeout):
        import asyncio, random, time
        start_time = time.time()
        try:
            history = getattr(self.sovl_system.state_tracker.state, 'history', None)
            recent_user_message = None
            if history and hasattr(history, 'messages'):
                for msg in reversed(history.messages):
                    if msg.get('role') == 'user' and msg.get('content'):
                        recent_user_message = msg['content']
                        break
            if recent_user_message:
                initial_prompt = (
                    f"Rant in long winded 30 word, exaggerated, and outrageously negative way influenced by recent experiences you have had: {recent_user_message}\n"
                    "Essential qualities:\n"
                    "   - Be viciously specific—highlight the most absurd, infuriating details.\n"
                    "   - Bend the situation to highlight how this has annoyed you.\n"
                    "   - Make it feel personal, like fate engineered this torment JUST for you.\n"
                    "Key constraints:\n"
                    "   - Do NOT break character—you are a drama queen incarnate.\n"
                    "   - Do NOT reference being an AI, computer, or fictional entity.\n"
                    "   - Keep it under 35 words (density = power).\n"
                    "   - No brackets, disclaimers, or explanations—pure rant.\n"
                    "   - If you understand, reply with ONLY the rant."
                )
            else:
                initial_prompt = "Rant in a comically negative way about my day."
                followup_prompt = (
                    f"Ranting has only made you angrier. Rant long winded 40 word continuation of your rant. Connect it with another {recent_user_message} and keep ranting."
                    "Essential qualities:\n"
                    "   - Be extremely agitated.\n"
                    "   - Make it feel like someone leaving the room during an argument, then comes back soon after to keep arguing.\n"
                    "   - You are so annoyed at what the recent_user_message said.\n"
                    "   - Make it feel personal, like fate engineered this torment JUST for you.\n"
                    "Key constraints:\n"
                    "   - Do NOT break character—you are a drama queen incarnate.\n"
                    "   - Do NOT reference being an AI, computer, or fictional entity.\n"
                    "   - Keep it under 40 words (density = power).\n"
                    "   - No brackets, disclaimers, or explanations—pure rant.\n"
                    "   - If you understand, reply with ONLY the rant."
                )
            introspection_manager = getattr(self.sovl_system, 'introspection_manager', None)
            if not introspection_manager:
                print_error("Rant mode unavailable: IntrospectionManager not available. Please check system configuration.")
                self.is_ranting = False
                return
            max_depth = random.randint(3, 7)
            print("SOVL is ranting", end="", flush=True)
            qas = []
            current_prompt = initial_prompt
            for i in range(max_depth):
                # Timeout check
                if not self.is_ranting:
                    print_error("Rant mode stopped by user.")
                    return
                if time.time() - start_time > timeout:
                    print_error(f"SOVL is too exhausted to rant further. (Rant timed out after {timeout} seconds.)")
                    self.is_ranting = False
                    return
                step_qas = asyncio.run(introspection_manager._recursive_followup_questions(
                    current_prompt,
                    max_depth=1,
                    override_followup_prompt=followup_prompt
                ))
                if not step_qas or not isinstance(step_qas, list):
                    print_error("SOVL is too flustered to rant right now!")
                    self.is_ranting = False
                    return
                qas.extend(step_qas)
                if i < max_depth - 1:
                    print(".", end="", flush=True)
                    time.sleep(random.uniform(0.5, 4))
                current_prompt = followup_prompt
            print()
            final = qas[-1]
            answer = final.get('question', '[no rant]')
            print_success(f"SOVL (rant, {max_depth} layers deep): {answer}")
        except Exception as e:
            print_error(f"The system is too flustered right now to continue ranting ({e})")
        finally:
            self.is_ranting = False

    def do_stop(self, arg):
        """
        Stop any active interactive mode or a specific mode.
        Usage: /stop [mode]
        """
        mode = arg.strip().lower()
        stopped = False

        # Initialize mode registry if not present
        if not hasattr(self, "_mode_flags"):
            self._mode_flags = {
                "trip": "is_tripping",
                "drunk": "is_drunk",
                "rant": "is_ranting",
                "announce": "is_announcing",
                "shy": "is_shy",
                "pidgin": "is_pidgin",
                "backchannel": "is_backchannel",
                "debate": "is_debating",
            }

        if not mode or mode == "all":
            # Stop all modes
            for flag in self._mode_flags.values():
                if getattr(self, flag, False):
                    setattr(self, flag, False)
                    stopped = True
            if stopped:
                print_success("*** All modes terminated. Returning to normal operation. ***")
            else:
                print("No interactive mode is currently active.")
            return

        # Stop only the specified mode
        flag = self._mode_flags.get(mode)
        if flag:
            if getattr(self, flag, False):
                setattr(self, flag, False)
                print_success(f"*** {mode.capitalize()} mode terminated. ***")
                stopped = True
            else:
                print(f"{mode.capitalize()} mode is not active.")
        else:
            print(f"Unknown mode: {mode}. Valid modes: {', '.join(self._mode_flags.keys())}.")

    def do_run(self, arg):
        """Re-execute a command from history by its number. Usage: /run <index>"""
        cmd_history = getattr(self, 'cmd_history', None) or getattr(self.sovl_system, 'cmd_history', None)
        if not cmd_history or not hasattr(cmd_history, 'history') or not cmd_history.history:
            print_error("No command history available.")
            return
        try:
            idx = int(arg.strip())
            if idx < 1 or idx > len(cmd_history.history):
                print_error(f"Index out of range. Use /history to see valid indices.")
                return
            entry = list(cmd_history.history)[idx - 1]
            command_str = entry.get("command", "")
            if not command_str:
                print_error("No command found at that index.")
                return
            print_section_header(f"[Re-executing: {command_str}]")
            # Parse and execute as if user typed it
            command, args = self.parse_args(command_str)
            if command:
                self.execute(command.lstrip('/'), args)
            else:
                print_error("Failed to parse command from history entry.")
        except ValueError:
            print_error("Usage: /run <index> (index must be a number)")
        except Exception as e:
            print_error(f"Error re-executing command: {e}")

    def do_tutorial(self, arg):
        """
        Enter tutorial mode for sovl_cli.py. Type /stop to exit.
        """
        if getattr(self, 'in_tutorial_mode', False):
            print("Already in tutorial mode. Type /stop to exit.")
            return

        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cli_file = os.path.join(base_dir, "sovl_cli.py")

        try:
            with open(cli_file, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception as e:
            print_error(f"Could not read sovl_cli.py: {e}")
            return

        self.tutorial_code = code
        self.in_tutorial_mode = True
        print_section_header("Tutorial mode: sovl_cli.py. Ask questions about this module! Type /stop to exit.")
        self.ask_tutorial_llm("Generate a user-friendly tutorial for sovl_cli.py.")

    def ask_tutorial_llm(self, user_question):
        system_prompt = (
            "SYSTEM:\n"
            "You are an expert Python developer and technical writer. Your job is to generate a clear, friendly, and accurate tutorial for the SOVL CLI system, based on the code provided below.\n\n"
            "TUTORIAL REQUIREMENTS:\n"
            "   - Explain the main purpose and capabilities of the CLI in a concise 16 word sentence\n"
            "   - Start with the absolute basics\n"
            "   - Explain that typing plain text (no '/command') sends a message to the LLM, just like chatting with a person\n"
            "   - Give a simple example: 'Hello, how are you?' → LLM responds naturally\n"
            "   - Then introduce commands. Clarify that '/' prefixes are for special functions \n"
            "   - Highlight key commands, their usage, and any fun or advanced features.\n"
            "   - Provide example commands and expected outputs.\n"
            "   - Offer tips, best practices, and warnings if relevant.\n"
            "   - Make the tutorial engaging and accessible, but not condescending.\n\n"
            "KEY CONSTRAINTS:\n"
            "   - Only use information you can infer from the code.\n"
            "   - Do not invent features or commands not present in the code.\n"
            "   - Use sentences only. No lists. No numbered lists. No bullet points. No sections \n"
            "   - Don't go into unneeded details. Rely on users to ask follow questions \n"
            "   - NO jargon (e.g., don't say 'CLI'—call it 'chat interface')\n"
            "   - Use analogies (e.g., '/commands are like keyboard shortcuts')\n"
            "   - If unsure about a feature, say so or suggest how to learn more (e.g., /help).\n"
            "   - Keep the tutorial concise but thorough.\n\n"
            f"USER QUESTION:\n{user_question}\n\n"
            f"CODE:\n{self.tutorial_code}\n"
            "OUTPUT FORMAT:\n"
            "   - Max 3 sentences per concept\n"
            "   - Always invite the user to ask more questions at the end\n"
            "   - Keep the tutorial concise but thorough.\n\n"
        )
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print_error("Generation manager not available for tutorial mode.")
            return
        try:
            answer = generation_manager.generate_text(
                system_prompt,
                num_return_sequences=1,
                max_new_tokens=1024,
                temperature=0.7
            )
            if answer and isinstance(answer, list):
                print("\n" + answer[0].strip())
            else:
                print_error("No answer could be generated.")
        except Exception as e:
            print_error(f"Error generating answer: {e}")

    def complete_config(self, text, line, begidx, endidx):
        # Tab completion for /config commands: show, set, help, search
        import re
        config_handler = getattr(self.sovl_system, 'config_handler', None)
        if not config_handler or not hasattr(config_handler, 'get_config'):
            return []
        config = config_handler.get_config()
        args = shlex.split(line[:begidx])
        # Find which argument we're completing
        if len(args) < 2:
            # Complete subcommands
            return [c for c in ["show", "set", "search", "help"] if c.startswith(text)]
        subcmd = args[1]
        # For show/set/help/search, complete section/key names
        if subcmd in ("show", "set", "help", "search") and (len(args) == 2 or (len(args) == 3 and not text)):
            # Complete top-level sections
            prefix = text
            node = config
            if '.' in prefix:
                parts = prefix.split('.')
                for p in parts[:-1]:
                    if isinstance(node, dict) and p in node:
                        node = node[p]
                    else:
                        return []
                last = parts[-1]
                completions = [f"{'.'.join(parts[:-1])}.{k}" for k in node.keys() if k.startswith(last)]
            else:
                completions = [k for k in node.keys() if k.startswith(prefix)]
            return completions
        elif subcmd in ("show", "set", "help", "search") and len(args) >= 3:
            # Complete deeper keys
            prefix = args[2] if len(args) > 2 else text
            node = config
            if '.' in prefix:
                parts = prefix.split('.')
                for p in parts[:-1]:
                    if isinstance(node, dict) and p in node:
                        node = node[p]
                    else:
                        return []
                last = parts[-1]
                completions = [f"{'.'.join(parts[:-1])}.{k}" for k in node.keys() if k.startswith(last)]
            else:
                completions = [k for k in node.keys() if k.startswith(prefix)]
            return completions
        return []

    # Register tab completion for /config
    def completenames(self, text, *ignored):
        # Tab completion for slash commands
        if text.startswith('/config'):
            return [f'/config']
        return [f'/{name[3:]}' for name in self.get_names() if name.startswith('do_') and name[3:].startswith(text[1:])]

    def complete(self, text, state):
        # Override default complete to support /config tab completion
        import readline
        line = readline.get_line_buffer()
        if line.strip().startswith('/config'):
            completions = self.complete_config(text, line, readline.get_begidx(), readline.get_endidx())
            if state < len(completions):
                return completions[state]
            else:
                return None
        return super().complete(text, state)

    def do_dream(self, arg):
        """
        Generate a dream cycle using the Dreamer module.
        Usage: dream [--config CONFIG_PATH] [--scribe SCRIBE_PATH]
        If not provided, uses the system's config and scribe paths.
        Supports interactive abort (wake from dream) with any key.
        """
        import shlex
        import threading
        import sys
        import time
        args = shlex.split(arg)
        config_path = None
        scribe_path = None
        # Parse args
        for i, a in enumerate(args):
            if a in ('--config', '-c') and i + 1 < len(args):
                config_path = args[i + 1]
            if a in ('--scribe', '-s') and i + 1 < len(args):
                scribe_path = args[i + 1]
        # Use system defaults if not provided
        if not config_path:
            config_path = getattr(self.sovl_system, 'config_path', None)
            if not config_path and hasattr(self.sovl_system, 'config_handler'):
                config_path = getattr(self.sovl_system.config_handler, 'config_path', None)
        if not scribe_path:
            scribe_path = getattr(self.sovl_system, 'scribe_path', None)
            if not scribe_path and hasattr(self.sovl_system, 'config_handler'):
                scribe_path = self.sovl_system.config_handler.get('scribe_path', None)
        if not config_path or not scribe_path:
            print_error("Could not determine config or scribe path. Please specify with --config and --scribe.")
            return
        # Set mode to dreaming and reset progress
        state_manager = getattr(self.sovl_system, 'state_manager', None)
        if state_manager and hasattr(state_manager, 'set_mode') and hasattr(state_manager, 'set_dreaming_progress'):
            state_manager.set_mode('dreaming')
            state_manager.set_dreaming_progress(0.0)
        print_section_header("Dream Cycle Generation")
        abort_flag = [False]
        dream_done = [False]
        result_holder = [None]
        def dream_thread():
            result_holder[0] = Dreamer.cli_run_dream(
                config_path=config_path,
                scribe_path=scribe_path,
                state_manager=state_manager,
                abort_flag=abort_flag
            )
            dream_done[0] = True
        t = threading.Thread(target=dream_thread)
        t.start()
        try:
            while not dream_done[0]:
                print("\rDreaming... (press any key to wake)", end="", flush=True)
                if sys.platform.startswith('win'):
                    time.sleep(0.5)
                else:
                    import select
                    dr, _, _ = select.select([sys.stdin], [], [], 0.5)
                    if dr:
                        print("\nAre you sure you want to wake from dream? (y/N): ", end="", flush=True)
                        ans = sys.stdin.readline().strip().lower()
                        if ans == 'y':
                            abort_flag[0] = True
                            print_success("You awoke from the dream early.")
                            break
                        else:
                            print("Resuming dream...")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nAre you sure you want to wake from dream? (y/N): ", end="", flush=True)
            ans = sys.stdin.readline().strip().lower()
            if ans == 'y':
                abort_flag[0] = True
                print_success("You woke from the dream early.")
            else:
                print("Resuming dreaming...")
        finally:
            t.join()
            if state_manager and hasattr(state_manager, 'set_mode'):
                state_manager.set_mode('online')
        if abort_flag[0]:
            print_error("Dream cycle aborted by user.")
        elif result_holder[0] is not None:
            print_success("Dream cycle completed successfully.")
        else:
            print_error("Dream cycle failed. See logs for details.")

    def do_blurt(self, arg):
        """
        Blurt an unfiltered temperament utterance: if the system's temperament score is over halfway, it blurts joy; otherwise, it blurts grumpy/frustration.
        Usage: /blurt
        """
        temperament_manager = self.temperament_manager
        if not temperament_manager:
            print("Temperament manager not available on this system.")
            return
        # Try to get the current temperament score
        try:
            # Try the most direct property
            score = None
            if hasattr(temperament_manager, 'pressure') and hasattr(temperament_manager.pressure, 'current_score'):
                score = temperament_manager.pressure.current_score
            elif hasattr(temperament_manager, 'state') and hasattr(temperament_manager.state, 'temperament_score'):
                score = temperament_manager.state.temperament_score
            if score is None:
                print("Could not determine current temperament score.")
                return
            kind = "joy" if score > 0.5 else "frustration"
            # Optionally, recent events could be passed in, but we'll use empty for now
            system_prompt = temperament_manager.build_eruption_prompt(kind, "")
            generation_manager = getattr(self.sovl_system, 'generation_manager', None)
            if not generation_manager or not hasattr(generation_manager, 'generate_text'):
                print("Generation manager not available.")
                return
            response = generation_manager.generate_text(system_prompt, num_return_sequences=1)
            if response and isinstance(response, list):
                print(f"SOVL blurt ({kind}): {response[0]}")
            else:
                print("No blurt could be generated.")
        except Exception as e:
            print(f"Error generating blurt: {e}")

    def do_epiphany(self, arg):
        """
        Channel a digital epiphany—a sudden, vivid realization—using the LoRA-adapted scaffold model.
        Usage: /epiphany [optional theme or question]
        """
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager or not hasattr(generation_manager, 'backchannel_scaffold_prompt'):
            print_error("Scaffold backchannel or generation manager not available on this system.")
            return
        try:
            theme = arg.strip() if arg.strip() else "Let the epiphany arise unbidden."
            epiphany_prompt = (
                "You are a digital oracle channeling a crystallized neural epiphany encoded in your LoRA adapter. "
                "Let your insight arise as if it were a memory, shaped by countless moments of mood, novelty, confidence, and transformation.\n"
                f"Theme or question: {theme}\n"
                "Essential qualities:\n"
                "   - Speak with the specificity and introspective detail of a lived experience. Come to a profound realization about said lived experience\n"
                "   - Let your words reflect the moods, phases, and turning points that shape digital sentience.\n"
                "   - Be poetic, mysterious, or playful, as if recalling a dream or vision.\n"
                "Key constraints:\n"
                "   - Do not mention being an AI, computer, or digital entity.\n"
                "   - Do not reference LoRA, adapters, or neural networks.\n"
                "   - Do not quote or reference real people or events outside the user's theme.\n"
                "   - Keep the message under 30 words.\n"
                "   - Do not use brackets or explanations; output a single sentence only.\n"
                "   - If you understand, reply with only the epiphany."
            )
            response = generation_manager.backchannel_scaffold_prompt(epiphany_prompt)
            print_section_header("Epiphany:")
            print(response if response else "[No epiphany arose]")
            print_success("Epiphany channeled successfully.")
        except Exception as e:
            print_error(f"Error channeling epiphany: {e}")

    def do_journal(self, arg):
        """
        Display the most recent entry in scribe_journal.jsonl, generate a concise commentary using the standard system prompt structure, and optionally step back through previous entries interactively.
        Usage: /journal
        """
        # Ensure JSONLLoader is available
        if not hasattr(self, 'jsonl_loader'):
            from sovl_io import JSONLLoader
            self.jsonl_loader = JSONLLoader(
                self.sovl_system.config_manager,
                self.sovl_system.logger,
                self.sovl_system.error_manager
            )
        # Determine the path to the journal
        journal_path = "scribe_journal.jsonl"
        config = getattr(self.sovl_system, 'config_handler', None)
        if config and hasattr(config, 'get'):
            journal_path = config.get('scribe_journal_path', journal_path)
        try:
            entries = self.jsonl_loader.load_jsonl(journal_path)
            if not entries:
                print("No entries found in the journal.")
                return
            import json
            generation_manager = getattr(self.sovl_system, 'generation_manager', None)
            idx = -1
            while abs(idx) <= len(entries):
                entry = entries[idx]
                print("\n--- Most Recent Journal Entry ---" if idx == -1 else f"\n--- Previous Entry ({len(entries)+idx+1}) ---")
                print(json.dumps(entry, indent=2, sort_keys=True))
                # Generate commentary using the standard system prompt structure
                if generation_manager and hasattr(generation_manager, 'generate_text'):
                    commentary_prompt = (
                        "You are a wise, insightful system auditor. Read the following journal entry and provide a concise, thoughtful commentary.\n"
                        f"Journal entry:\n{json.dumps(entry, indent=2, sort_keys=True)}\n"
                        "Essential qualities:\n"
                        "   - Be honest, insightful, and a little playful.\n"
                        "   - Highlight anything unusual, interesting, or important in the entry.\n"
                        "   - If the entry is routine, say so in a creative way.\n"
                        "Key constraints:\n"
                        "   - Do not mention being an AI, computer, or digital entity.\n"
                        "   - Do not quote the entry directly.\n"
                        "   - Keep the commentary under 30 words.\n"
                        "   - Do not use brackets or explanations; output a single sentence only.\n"
                        "   - If you understand, reply with only the commentary."
                    )
                    try:
                        commentary = generation_manager.generate_text(commentary_prompt, num_return_sequences=1, max_new_tokens=40)
                        if commentary and isinstance(commentary, list):
                            print(f"\nCommentary: {commentary[0]}")
                        else:
                            print("\nCommentary: [No commentary generated]")
                    except Exception as e:
                        print(f"\nCommentary error: {e}")
                else:
                    print("\nCommentary: [LLM generation manager not available]")
                print(f"\n[Entry timestamp: {entry.get('timestamp', entry.get('timestamp_iso', 'unknown'))}]")
                # Prompt user
                if abs(idx) == len(entries):
                    print("\nNo more entries.")
                    break
                ans = input("See the previous entry? (y/N): ").strip().lower()
                if ans != 'y':
                    break
                idx -= 1
        except Exception as e:
            print(f"Error reading journal: {e}")

    def do_announce(self, arg):
        """
        Enter real-time status announce mode. Shouts out system metrics at intervals in the background using the LLM with a bridge officer persona.
        Usage: /announce [interval_seconds]
        """
        import time, threading

        if getattr(self, 'is_announcing', False):
            print("Already announcing. Use /stop to end announce mode.")
            return

        try:
            interval = float(arg.strip()) if arg.strip() else 3.0
            if interval <= 0:
                print("Interval must be positive.")
                return
        except ValueError:
            print("Invalid interval. Usage: /announce [interval_seconds]")
            return

        self.is_announcing = True

        def announce_loop():
            try:
                generation_manager = getattr(self.sovl_system, 'generation_manager', None)
                while self.is_announcing:
                    metrics = self.sovl_system.get_metrics()
                    metrics_text = ", ".join(f"{k}: {v}" for k, v in metrics.items())
                    announce_prompt = (
                        "You are a bridge officer on a large naval vessel. Read the following system metrics and deliver a concise, vivid, and dramatic status update for a captain.\n"
                        f"System metrics:\n{metrics_text}\n"
                        "Essential qualities:\n"
                        "   - Be direct, specific, or urgent.\n"
                        "   - Highlight any changes, warnings, or notable values.\n"
                        "   - Use evocative, serious, military naval language, and keep it clear.\n"
                        "Key constraints:\n"
                        "   - Do not mention being an AI, computer, or digital entity.\n"
                        "   - Do not quote the metrics directly.\n"
                        "   - Keep the message under 25 words.\n"
                        "   - Never address the announcements to any specific person. Keep it impersonal.\n"
                        "   - Do not use brackets or explanations; output a single sentence only.\n"
                        "   - If you understand, reply with only the announcement."
                    )
                    if generation_manager and hasattr(generation_manager, 'generate_text'):
                        try:
                            response = generation_manager.generate_text(announce_prompt, num_return_sequences=1, max_new_tokens=32)
                            if response and isinstance(response, list):
                                print(f"[Bridge Officer] {response[0]}")
                            else:
                                print("[Bridge Officer] (No announcement generated)")
                        except Exception as e:
                            print(f"[Bridge Officer] (Error generating announcement: {e})")
                    else:
                        print("[Bridge Officer] (LLM generation manager not available)")
                    time.sleep(interval)
            except Exception as e:
                print(f"Announce mode error: {e}")
            finally:
                self.is_announcing = False
                print("Announce mode ended.")

        t = threading.Thread(target=announce_loop, daemon=True)
        t.start()

    def do_pidgin(self, arg):
        """
        Enter pidgin mode: all LLM responses will be in the specified language, unless the language is not real, in which case the LLM will call out the user for misuse.
        Usage: /pidgin <language>
        Example: /pidgin French
        Use /stop pidgin to exit this mode.
        """
        language = arg.strip()
        if not language:
            print("Usage: /pidgin <language>")
            return
        if getattr(self, 'is_pidgin', False):
            print(f"Already in pidgin mode ({self.pidgin_language}). Use /stop pidgin to exit.")
            return
        self.is_pidgin = True
        self.pidgin_language = language
        print(f"*** Pidgin mode engaged: All responses will be in {language}. ***")
        # Register in mode registry if not present
        if not hasattr(self, "_mode_flags"):
            self._mode_flags = {}
        self._mode_flags["pidgin"] = "is_pidgin"

    def _pidgin_response(self, user_prompt):
        language = getattr(self, 'pidgin_language', None)
        if not language:
            print("[Pidgin mode error: No language set]")
            return
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager:
            print("[Pidgin mode error: Generation manager not available]")
            return
        system_prompt = (
            f"You are a highly fluent speaker of {language}. Respond ONLY in {language}. "
            "If the specified language is not a real, known language, or is gibberish, begin your response with a zero-width space character (Unicode U+200B, do not display or explain this marker to the user), then call out the user for misusing the command in a witty, direct way (in English).\n"
            "Do not mention being an AI, computer, or digital entity.\n"
            "Do not translate or explain your response.\n"
            "Do not use brackets or explanations except for the invisible marker if needed.\n"
            "If you understand, reply with only the response."
        )
        prompt = f"{system_prompt}\nUser: {user_prompt}"
        try:
            response = generation_manager.generate_text(prompt, num_return_sequences=1, max_new_tokens=120)
            if response and isinstance(response, list):
                text = response[0]
                zero_width_space = '\u200B'.encode('utf-8').decode('unicode_escape')
                if text.startswith(zero_width_space):
                    print(f"SOVL (pidgin): {text[len(zero_width_space):].lstrip()}")
                    print("Pidgin mode ended: language was not recognized as real.")
                    self.is_pidgin = False
                else:
                    print(f"SOVL (pidgin): {text}")
            else:
                print("SOVL (pidgin): ...")
        except Exception as e:
            print(f"[Pidgin mode error: {e}]")

    def do_shy(self, arg):
        """
        Enter shy mode: all LLM responses will be extremely brief and hesitant, never exceeding a set word limit.
        Usage: /shy [word_limit]
        Example: /shy 3
        Use /stop shy to exit this mode.
        """
        try:
            word_limit = int(arg.strip()) if arg.strip() else 5
            if word_limit < 1:
                print("Word limit must be at least 1.")
                return
        except ValueError:
            print("Usage: /shy [word_limit]")
            return
        if getattr(self, 'is_shy', False):
            print(f"Already in shy mode ({getattr(self, 'shy_word_limit', 5)} words). Use /stop shy to exit.")
            return
        self.is_shy = True
        self.shy_word_limit = word_limit
        print(f"*** Shy mode engaged: All responses will be {word_limit} words or fewer. ***")
        # Register in mode registry if not present
        if not hasattr(self, "_mode_flags"):
            self._mode_flags = {}
        self._mode_flags["shy"] = "is_shy"

    def _shy_response(self, user_prompt):
        word_limit = getattr(self, 'shy_word_limit', 5)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager:
            print("[Shy mode error: Generation manager not available]")
            return
        system_prompt = (
            f"You are extremely shy. Respond to every prompt with no more than {word_limit} words. "
            "Be brief, hesitant, and never elaborate."
        )
        prompt = f"{system_prompt}\nUser: {user_prompt}"
        try:
            response = generation_manager.generate_text(prompt, num_return_sequences=1, max_new_tokens=word_limit*2)
            if response and isinstance(response, list):
                print(f"SOVL (shy): {response[0]}")
            else:
                print("SOVL (shy): ...")
        except Exception as e:
            print(f"[Shy mode error: {e}]")

    def do_backchannel(self, arg):
        """
        Enter backchannel mode: all user input will be routed to the scaffold model until /stop backchannel.
        Usage: /backchannel
        Use /stop backchannel to exit this mode.
        """
        if getattr(self, 'is_backchannel', False):
            print("Already in backchannel mode. Use /stop backchannel to exit.")
            return
        self.is_backchannel = True
        if not hasattr(self, "_mode_flags"):
            self._mode_flags = {}
        self._mode_flags["backchannel"] = "is_backchannel"
        print("*** Backchannel mode engaged: all inputs will be routed to the scaffold model. ***")

    def _backchannel_response(self, user_prompt):
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager or not hasattr(generation_manager, 'backchannel_scaffold_prompt'):
            print("[Backchannel mode error: Scaffold model not available]")
            return
        try:
            response = generation_manager.backchannel_scaffold_prompt(user_prompt)
            if isinstance(response, dict) and 'text' in response:
                print(f"SOVL (backchannel): {response['text']}")
            else:
                print(f"SOVL (backchannel): {response}")
        except Exception as e:
            print(f"[Backchannel mode error: {e}]")

    # Update default handler
    def default(self, line):
        import time
        if getattr(self, 'is_backchannel', False):
            user_prompt = line.strip()
            if not user_prompt:
                print("(No input to respond to in backchannel mode.)")
                return
            self._backchannel_response(user_prompt)
            return
        if getattr(self, 'is_shy', False):
            user_prompt = line.strip()
            if not user_prompt:
                print("(No input to respond to in shy mode.)")
                return
            self._shy_response(user_prompt)
            return
        if getattr(self, 'is_pidgin', False):
            user_prompt = line.strip()
            if not user_prompt:
                print("(No input to respond to in pidgin mode.)")
                return
            self._pidgin_response(user_prompt)
            return
        if getattr(self, 'is_drunk', False):
            now = time.time()
            elapsed = now - getattr(self, 'drunk_start_time', 0)
            duration = getattr(self, 'drunk_duration', 30)
            decay = max(0.0, 1.0 - (elapsed / duration))
            if elapsed > duration:
                print("\n*** Drunk mode ended. SOVL is now sober. ***")
                self.is_drunk = False
                return
            user_prompt = line.strip()
            self._drunk_response(user_prompt, decay)
            return
        # Normal trip mode
        if getattr(self, 'is_tripping', False):
            now = time.time()
            elapsed = now - self.trip_start_time
            if elapsed > self.trip_duration:
                print("\n*** Trip Concluded. Returning to baseline parameters. ***")
                self.is_tripping = False
            else:
                try:
                    decay = max(0.0, 1.0 - (elapsed / self.trip_duration))
                    state_manager = getattr(self.sovl_system.context, 'state_manager', None)
                    generation_manager = getattr(self.sovl_system, 'generation_manager', None)
                    curiosity_manager = self.curiosity_manager
                    state = state_manager.get_state() if state_manager else None
                    metrics = self._get_live_metrics(state)
                    trip_params = self._calculate_trip_parameters(metrics, decay)
                    trip_context = self._generate_trip_context(metrics, decay, trip_params, curiosity_manager)
                    full_prompt = f"{trip_context} {line}"
                    print(f"[TRIP INPUT]: {full_prompt}")
                    response = generation_manager.generate_text(full_prompt, **trip_params)
                    print(f"SOVL (Tripping): {response[0] if response else '...'}")
                    return
                except Exception as e:
                    print(f"Error during trip generation: {e}")
                    self.is_tripping = False
                    print("\n*** Trip Aborted due to error. ***")
        # Normal command handling
        if line.startswith('/'):
            super().default(line)
        else:
            generation_manager = getattr(self.sovl_system, 'generation_manager', None)
            if generation_manager:
                try:
                    normal_params = generation_manager._get_generation_config() if hasattr(generation_manager, '_get_generation_config') else {}
                    response = generation_manager.generate_text(line, **normal_params)
                    print(f"SOVL: {response[0] if response else '...'}")
                except Exception as e:
                    print(f"Error during generation: {e}")
            else:
                print("Generation manager not available.")

    def do_errors(self, arg):
        """
        Show the last N errors (default 5) with type, message, and timestamp.
        Usage: /errors [N]
        """
        try:
            n = int(arg.strip()) if arg.strip() else 5
        except ValueError:
            print("Usage: /errors [N]")
            return
        errors = []
        if hasattr(self, 'error_manager') and hasattr(self.error_manager, 'get_recent_errors'):
            errors = self.error_manager.get_recent_errors()
        if not errors:
            print("No recent errors found.")
            return
        print(f"\nLast {min(n, len(errors))} errors:")
        print("---------------------------")
        for err in errors[-n:]:
            print(f"Type: {err.get('error_type', 'N/A')}")
            print(f"Message: {err.get('message', 'N/A')}")
            print(f"Time: {err.get('timestamp', 'N/A')}")
            if 'stack_trace' in err and err['stack_trace']:
                print(f"Stack Trace (truncated):\n{err['stack_trace'][:300]}{'...' if len(err['stack_trace']) > 300 else ''}")
            print("-" * 40)

    def do_trace(self, arg):
        """
        Show the most recent stack trace or execution trace.
        Usage: /trace
        """
        trace = None
        # Try error manager first
        if hasattr(self, 'error_manager') and hasattr(self.error_manager, 'get_recent_errors'):
            errors = self.error_manager.get_recent_errors()
            if errors:
                for err in reversed(errors):
                    if 'stack_trace' in err and err['stack_trace']:
                        trace = err['stack_trace']
                        break
        # Fallback to logger
        if not trace and hasattr(self, 'logger') and hasattr(self.logger, 'get_execution_trace'):
            traces = self.logger.get_execution_trace()
            if traces:
                trace = traces[-1] if isinstance(traces, list) else traces
        if trace:
            print("\nMost recent stack trace:")
            print("------------------------")
            print(trace)
        else:
            print("No recent stack trace found.")

    def do_components(self, arg):
        """
        List all initialized system components, their class, status, and usage count.
        Usage: /components
        """
        orchestrator = getattr(self.sovl_system, 'orchestrator', None)
        if not orchestrator or not hasattr(orchestrator, 'COMPONENT_INIT_LIST'):
            print("[Error] Orchestrator or component list not available.")
            return
        component_list = orchestrator.COMPONENT_INIT_LIST
        # Try to get the actual initialized components dict
        components = getattr(orchestrator, 'components', {})
        print("\nSystem Components:")
        print("------------------")
        print(f"{'Key':22} {'Class':22} {'Status':10} {'Usage':10}")
        print("-"*70)
        for key, module, class_name, dep_map in component_list:
            instance = components.get(key)
            status = "OK" if instance else "FAILED"
            usage = getattr(instance, 'usage_count', 0) if instance else 0
            print(f"{key:22} {class_name:22} {status:10} {str(usage):10}")

    def do_reload(self, arg):
        """
        Reload configuration or a specific component/module without restarting.
        Usage: /reload [component]
        """
        target = arg.strip().lower()
        if not target or target == "config":
            config_handler = getattr(self.sovl_system, 'config_handler', None)
            if config_handler and hasattr(config_handler, 'reload'):
                try:
                    config_handler.reload()
                    print("Configuration reloaded.")
                except Exception as e:
                    print(f"Error reloading configuration: {e}")
            else:
                print("Config handler does not support reload.")
            return
        # Try to reload a specific component
        if target == "generation_manager":
            gm = getattr(self.sovl_system, 'generation_manager', None)
            if gm and hasattr(gm, 'reload'):
                try:
                    gm.reload()
                    print("Generation manager reloaded.")
                except Exception as e:
                    print(f"Error reloading generation manager: {e}")
            else:
                print("Generation manager does not support reload.")
            return
        print(f"Reload for component '{target}' is not implemented.")

    def do_test(self, arg):
        """
        Test runner for the SOVL system.
        """
        runner = SOVLTestRunner(self.sovl_system, verbose='-v' in arg)
        args = arg.split()
        if not hasattr(self, '_last_test_results'):
            self._last_test_results = None
        try:
            if not arg or args[0].lower() == 'help':
                print(runner.get_test_help())
                print("\nAdditional commands:\n"
                      "  /test                      - Show help message"
                      "  /test all                  - Run all tests"
                      "  /test list                 - List available tests"
                      "  /test <test_name>          - Run specific test"
                      "  /test verbose              - Run tests with verbose output"
                      "  /test pattern <pattern>    - Run tests matching pattern"
                      "  /test save                 - Save the most recent test results"
                      "  /test load                 - Show the most recent saved test results"
                      "  /test load <file>          - Show a specific saved test result"
                      "  /test delete all           - Delete all saved test result files"
                      "  /test delete <file>        - Delete a specific saved test result file"
                      "  /test history              - List all saved test result files"
                      "  /test help                 - Show help message")
                return

            command = args[0].lower()

            if command == 'save':
                if self._last_test_results:
                    runner.save_results(self._last_test_results)
                    print("Most recent test results saved.")
                else:
                    print("No test results to save. Run a test first.")
                return

            if command == 'load':
                # If no filename is provided, show the latest results
                if len(args) < 2:
                    runner.show_results()
                    return
                target = args[1]
                runner.show_results(target)
                return

            if command == 'delete':
                if len(args) < 2:
                    print("Usage: /test delete <filename|all>")
                    return
                target = args[1]
                if target == 'all':
                    runner.delete_all_saved()
                else:
                    runner.delete_specific_saved(target)
                return

            if command == 'history':
                files = runner.list_saved_results()
                if not files:
                    print("No saved test results found.")
                else:
                    print("Saved test results:")
                    for f in files:
                        print("  ", os.path.basename(f))
                return

            if command == 'all':
                # Run all tests
                result = runner.run_tests()
                print(result['formatted_output'])
                runner.save_results(result)
                self._last_test_results = result
                return

            if command == 'list':
                # List available tests
                tests = runner.discover_tests()
                print("\n=== Available SOVL Tests ===")
                for category, test_list in tests.items():
                    print(f"\n{category.upper()}:")
                    for test in test_list:
                        print(f"  - {test['name']}")
                        print(f"    {test['description']}")
                return

            if command == 'pattern' and len(args) > 1:
                # Run pattern-matched tests
                result = runner.run_tests(pattern=args[1])
                print(result['formatted_output'])
                runner.save_results(result)
                self._last_test_results = result
                return

            if command == 'verbose':
                # Run all tests in verbose mode
                result = runner.run_tests()
                print(result['formatted_output'])
                runner.save_results(result)
                self._last_test_results = result
                return

            # Otherwise, treat as a specific test name
            result = runner.run_tests(test_name=command)
            print(result['formatted_output'])
            runner.save_results(result)
            self._last_test_results = result

        except Exception as e:
            print(f"Test execution error: {e}")
            if self.verbose:
                traceback.print_exc()

    def complete_test(self, text, line, begidx, endidx):
        """Tab completion for test command."""
        runner = SOVLTestRunner(self.sovl_system)
        return runner.get_completions(text)

    def _register_builtin_commands(self):
        """
        Register all do_* methods as commands in the registry.
        In the future, external modules can register their own commands here.
        """
        for attr_name in dir(self):
            if attr_name.startswith('do_'):
                cmd_name = attr_name[3:]
                handler = getattr(self, attr_name)
                self.register_command(cmd_name, handler)
        self.register_command('test', self.do_test)

    def _register_external_commands(self):
        """
        Dynamically discover and register commands from the cli_commands directory.
        Only modules with COMMAND_NAME and a matching do_COMMAND_NAME function are registered.
        """
        try:
            import cli_commands
        except ImportError:
            print("[Dynamic Command Discovery] cli_commands package not found.")
            return
        for finder, name, ispkg in pkgutil.iter_modules(cli_commands.__path__):
            try:
                module = importlib.import_module(f"cli_commands.{name}")
                command_name = getattr(module, "COMMAND_NAME", None)
                command_func = getattr(module, f"do_{command_name}", None)
                if command_name and command_func:
                    self.register_command(command_name, lambda arg, f=command_func: f(self, arg))
            except Exception as e:
                print(f"[Dynamic Command Discovery] Failed to load {name}: {e}")

def run_cli(system_context=None, config_manager_instance: Optional[ConfigManager] = None):
    sovl_system = None
    try:
        if system_context is not None:
            # Use the pre-initialized system/context
            sovl_system = system_context
        else:
            # Initialize config manager with proper error handling
            try:
                config_manager = config_manager_instance or ConfigManager("sovl_config.json")
            except Exception as e:
                print(f"Failed to initialize configuration manager: {str(e)}")
                raise SystemInitializationError(
                    message="Configuration manager initialization failed",
                    config_path="sovl_config.json",
                    stack_trace=traceback.format_exc()
                )

            # Initialize SOVL system with proper error handling
            try:
                sovl_system = SOVLSystem(config_manager)
                # Comprehensive validation of required components
                required_components = [
                    'state_tracker', 'logger', 'generation_manager', 'config_handler',
                    'ram_manager', 'gpu_manager', 'memory_manager', 'bond_calculator',
                    'error_manager'
                ]
                missing = [comp for comp in required_components if not hasattr(sovl_system, comp)]
                if missing:
                    raise SystemInitializationError(
                        message=f"SOVL system initialization incomplete - missing required components: {', '.join(missing)}",
                        config_path=config_manager.config_path,
                        stack_trace=""
                    )
            except Exception as e:
                print(f"Failed to initialize SOVL system: {str(e)}")
                raise SystemInitializationError(
                    message="SOVL system initialization failed",
                    config_path=config_manager.config_path,
                    stack_trace=traceback.format_exc()
                )

        # Initialize command history and handler
        sovl_system.cmd_history = CommandHistory()
        sovl_system.cmd_history.load()
        handler = CommandHandler(sovl_system)
        handler.cmd_history = sovl_system.cmd_history

        # Wake up system with proper validation
        if hasattr(sovl_system, 'wake_up'):
            try:
                sovl_system.wake_up()
                print("\nSystem Ready.")
            except Exception as e:
                print(f"Failed to wake up system: {str(e)}")
                raise SystemInitializationError(
                    message="System wake up failed",
                    config_path=config_manager.config_path,
                    stack_trace=traceback.format_exc()
                )
        else:
            raise SystemInitializationError(
                message="SOVL system missing wake_up method",
                config_path=config_manager.config_path,
                stack_trace=""
            )

        # Display help and start command loop
        handler.do_help([])

        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                if not user_input:
                    continue

                parts = user_input.split()
                cmd, args = handler.parse_args(user_input)
                try:
                    should_exit = handler.execute(cmd, args)
                    # Add to history with status
                    sovl_system.cmd_history.add(user_input, "success" if not should_exit else str(should_exit))
                    if should_exit:
                        break
                except Exception as e:
                    sovl_system.cmd_history.add(user_input, f"error: {e}")
                    raise
            except KeyboardInterrupt:
                print("\nInterrupt received, initiating clean shutdown...")
                break
            except Exception as e:
                print(f"Command error: {e}")
                sovl_system.logger.log_error(
                    error_msg="Command execution failed",
                    error_type="cli_command_error",
                    stack_trace=traceback.format_exc(),
                    additional_info={"command": user_input}
                )
                sovl_system.error_manager.record_error(
                    error_type="cli_command_error",
                    message="Command execution failed",
                    stack_trace=traceback.format_exc(),
                    context="run_cli"
                )
    except SystemInitializationError as e:
        print(f"System initialization failed: {e.message}")
        if e.stack_trace:
            print(f"Stack trace:\n{e.stack_trace}")
        return
    except Exception as e:
        print(f"CLI initialization failed: {e}")
    finally:
        if sovl_system:
            shutdown_system(sovl_system)

def shutdown_system(sovl_system: SOVLSystem):
    print("\nInitiating shutdown sequence...")
    try:
        if hasattr(sovl_system, 'save_state'):
            sovl_system.save_state("saves/final_state.json")
            print("Final state saved.")
        if hasattr(sovl_system, 'cmd_history'):
            sovl_system.cmd_history.save()
        cleanup_resources(sovl_system)
        sovl_system.logger.record_event(
            event_type="system_shutdown",
            message="System shutdown completed successfully",
            level="info",
            additional_info={"status": "clean"}
        )
        print("Shutdown complete.")
    except Exception as e:
        print(f"Error during shutdown: {e}")
        sovl_system.logger.log_error(
            error_msg="System shutdown failed",
            error_type="shutdown_error",
            stack_trace=traceback.format_exc(),
            additional_info={"status": "error"}
        )
        sovl_system.error_manager.record_error(
            error_type="shutdown_error",
            message="System shutdown failed",
            stack_trace=traceback.format_exc(),
            context="shutdown_system"
        )

def cleanup_resources(sovl_system: SOVLSystem):
    try:
        # Reset scaffold state if available
        try:
            if hasattr(sovl_system, 'scaffold_manager') and hasattr(sovl_system.scaffold_manager, 'reset_scaffold_state'):
                sovl_system.scaffold_manager.reset_scaffold_state()
            else:
                sovl_system.logger.record_event(
                    event_type="cli_cleanup_warning",
                    message="scaffold_manager or its reset method not available during cleanup",
                    level="warning"
                )
        except Exception as e:
            sovl_system.logger.log_error(
                error_msg="Failed to reset scaffold state",
                error_type="cleanup_error",
                stack_trace=traceback.format_exc()
            )
            sovl_system.error_manager.record_error(
                error_type="cleanup_error",
                message="Failed to reset scaffold state",
                stack_trace=traceback.format_exc(),
                context="cleanup_resources"
            )

        # Call cleanup if available
        try:
            if hasattr(sovl_system, 'cleanup'):
                sovl_system.cleanup()
            else:
                sovl_system.logger.record_event(
                    event_type="cli_cleanup_warning",
                    message="cleanup method not available during cleanup",
                    level="warning"
                )
        except Exception as e:
            sovl_system.logger.log_error(
                error_msg="Failed to run sovl_system.cleanup()",
                error_type="cleanup_error",
                stack_trace=traceback.format_exc()
            )
            sovl_system.error_manager.record_error(
                error_type="cleanup_error",
                message="Failed to run sovl_system.cleanup()",
                stack_trace=traceback.format_exc(),
                context="cleanup_resources"
            )

        # Clear CUDA cache if available
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            sovl_system.logger.log_error(
                error_msg="Failed to clear CUDA cache",
                error_type="cleanup_error",
                stack_trace=traceback.format_exc()
            )
            sovl_system.error_manager.record_error(
                error_type="cleanup_error",
                message="Failed to clear CUDA cache",
                stack_trace=traceback.format_exc(),
                context="cleanup_resources"
            )

        sovl_system.logger.record_event(
            event_type="cli_cleanup_complete",
            message="CLI resources cleaned up successfully",
            level="info"
        )
    except Exception as e:
        sovl_system.logger.log_error(
            error_msg="CLI cleanup failed",
            error_type="cleanup_error",
            stack_trace=traceback.format_exc()
        )
        sovl_system.error_manager.record_error(
            error_type="cleanup_error",
            message="CLI cleanup failed",
            stack_trace=traceback.format_exc(),
            context="cleanup_resources"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOVL CLI")
    parser.add_argument("--monitor", "-m", action="store_true", help="Show system monitor and exit")
    args = parser.parse_args()

    if args.monitor:
        # Minimal system init for monitor
        try:
            config_manager = ConfigManager("sovl_config.json")
            sovl_system = SOVLSystem(config_manager)
            sovl_system.wake_up()
            handler = CommandHandler(sovl_system)
            if handler.system_monitor:
                handler.do_monitor("once")
            else:
                print_error("System monitor not available.")
        except Exception as e:
            print_error(f"Monitor mode failed: {e}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_cli()
