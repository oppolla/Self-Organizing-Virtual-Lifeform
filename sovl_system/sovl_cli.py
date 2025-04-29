import time
import torch
import traceback
from typing import List, Dict, Tuple, Optional, Callable
from sovl_main import SOVLSystem
from sovl_config import ConfigManager
from sovl_utils import safe_compare
from sovl_monitor import SystemMonitor, MemoryMonitor, TraitsMonitor
import readline
import rlcompleter
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

# Constants
TRAIN_EPOCHS = 10
BATCH_SIZE = 32
FORMATTED_TRAINING_DATA = None
VALID_DATA = None

COMMAND_CATEGORIES = {
    "System": ["quit", "exit", "save", "load", "reset", "status", "help", "monitor"],
    "Training": ["train", "dream"],
    "Generation": ["echo", "mimic"],
    "Memory": ["recall", "forget", "recap"],
    "Interaction": ["muse", "flare", "debate", "spark", "reflect"],
    "Debug": ["log", "config", "panic", "glitch", "scaffold"],
    "Advanced": ["attune", "rewind"],
    "History": ["history"],
    "Fun": ["joke"],
    "Rating": ["rate"],
    "Complaint": ["complain"]
}

class CommandHistory:
    """Manages command history with search functionality and execution status."""
    def __init__(self, max_size: int = 100):
        self.history = deque(maxlen=max_size)  # Each entry: {"command": ..., "status": ...}
        self.current_index = -1

    def add(self, command: str, status: str = "unknown"):
        """Add a command and its status to history."""
        self.history.append({"command": command, "status": status})
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
                self.history = deque(json.load(f), maxlen=self.history.maxlen)

class CommandHandler(cmd.Cmd):
    """Enhanced command handler with history and search capabilities."""
    prompt = 'sovl> '
    
    def __init__(self, sovl_system: SOVLSystem):
        super().__init__()
        self.sovl_system = sovl_system
        self.debug_mode = False
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

    def preloop(self):
        """Initialize the command handler."""
        print("SOVL Interactive CLI")
        print("Type 'help' for available commands")
        self.logger.record_event(
            event_type="cli_startup",
            message="SOVL CLI started",
            level="info"
        )
        
    def do_help(self, arg):
        """Show help for commands and required SOVLSystem components."""
        try:
            if arg:
                # Show help for specific command
                try:
                    func = getattr(self, 'help_' + arg)
                except AttributeError:
                    try:
                        doc = getattr(self, 'do_' + arg).__doc__
                        if doc:
                            print(doc)
                            return
                    except AttributeError:
                        pass
                    print(f"No help available for {arg}")
                else:
                    func()
            else:
                # Show all commands
                print("\nAvailable commands:")
                print("------------------")
                for category, commands in COMMAND_CATEGORIES.items():
                    print(f"{category}: {', '.join(commands)}")
                print("Type 'help [command]' for more information on a specific command.")
                print("\nRequired SOVLSystem components for CLI:")
                print("--------------------------------------")
                print(
                    "config_handler, ram_manager, state_tracker, gpu_manager, memory_manager, "
                    "generation_manager, bond_calculator, logger, error_manager"
                )
                print("If any of these are missing, some commands may not function.")
            self.logger.record_event(
                event_type="cli_help",
                message=f"Help command invoked for: {arg if arg else 'all'}",
                level="info"
            )
        except Exception as e:
            print(f"Error in help command: {e}")
            self.logger.log_error(
                error_msg=f"Help command failed: {str(e)}",
                error_type="cli_help_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_help_error",
                message=f"Help command failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.do_help"
            )
            
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
            
    def do_metrics(self, arg):
        """Show current metrics."""
        metrics = self.sovl_system.get_metrics()
        print("\nCurrent Metrics:")
        print("---------------")
        for key, value in metrics.items():
            print(f"{key}: {value}")
            
    def do_config(self, arg):
        """Show current configuration."""
        config = self.sovl_system.get_config()
        print("\nCurrent Configuration:")
        print("---------------------")
        for key, value in config.items():
            print(f"{key}: {value}")
            
    def do_exit(self, arg):
        """Exit the CLI."""
        try:
            print("Exiting CLI...")
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
            
    def do_history(self, arg):
        """Show command history with execution status."""
        try:
            cmd_history = getattr(self, 'cmd_history', None) or getattr(self.sovl_system, 'cmd_history', None)
            if cmd_history and hasattr(cmd_history, 'history'):
                for entry in cmd_history.history:
                    print(f"{entry['command']}  [status: {entry['status']}]")
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
            
    def do_debug(self, arg):
        """
        Debug command with various subcommands:
        debug on/off - Enable/disable debug mode
        debug state - Show internal state
        debug components - List active components
        debug errors - Show recent errors
        debug memory - Show memory usage
        debug trace - Show execution trace
        """
        try:
            cmd = arg.split()
            if not cmd:
                print("Debug commands: on/off, state, components, errors, memory, trace")
                return
            elif cmd[0] == "on":
                self.debug_mode = True
                print("Debug mode enabled")
            elif cmd[0] == "off":
                self.debug_mode = False
                print("Debug mode disabled")
                if hasattr(self.logger, 'update_config'):
                    self.logger.update_config(log_level="info")
            elif cmd[0] == "state":
                if hasattr(self, '_show_debug_state'):
                    self._show_debug_state()
                else:
                    print("Debug state function not available.")
            elif cmd[0] == "components":
                if hasattr(self, '_show_debug_components'):
                    self._show_debug_components()
                else:
                    print("Debug components function not available.")
            elif cmd[0] == "errors":
                # Show recent errors from ErrorManager
                errors = self.error_manager.get_recent_errors() if hasattr(self.error_manager, 'get_recent_errors') else []
                if errors:
                    print("\nRecent CLI Errors:")
                    print("-----------------")
                    for error in errors:
                        print(f"Type: {error.get('error_type','N/A')}")
                        print(f"Message: {error.get('message','N/A')}")
                        print(f"Time: {error.get('timestamp','N/A')}")
                        if 'stack_trace' in error:
                            print(f"Stack Trace:\n{error['stack_trace']}")
                        if 'context' in error:
                            print(f"Context: {error['context']}")
                        print("-" * 50)
                else:
                    print("No recent CLI errors found.")
            elif cmd[0] == "memory":
                if hasattr(self, '_show_debug_memory'):
                    self._show_debug_memory()
                else:
                    print("Debug memory function not available.")
            elif cmd[0] == "trace":
                if hasattr(self, '_show_debug_trace'):
                    self._show_debug_trace()
                else:
                    print("Debug trace function not available.")
            else:
                print(f"Unknown debug command: {cmd[0]}")
            print("Type 'debug' for available debug commands")
        except Exception as e:
            print(f"Error in debug command: {e}")
            self.logger.log_error(
                error_msg=f"Debug command failed: {str(e)}",
                error_type="cli_debug_error",
                stack_trace=traceback.format_exc()
            )
            self.error_manager.record_error(
                error_type="cli_debug_error",
                message=f"Debug command failed: {str(e)}",
                stack_trace=traceback.format_exc(),
                context="CommandHandler.do_debug"
            )
            
    def help_debug(self):
        """Show help for debug command."""
        print("\nDebug Commands:")
        print("-------------")
        print("debug on/off        - Enable/disable debug mode")
        print("debug state         - Show internal state")
        print("debug components    - List active components")
        print("debug errors        - Show recent errors")
        print("debug memory        - Show memory usage")
        print("debug trace         - Show execution trace")
        print("\nUse these commands to inspect and troubleshoot the system.")

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
        """
        scaffold_provider = getattr(self.sovl_system, 'scaffold_provider', None)
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
        else:
            print(self.do_scaffold.__doc__)

    def help_scaffold(self):
        """Detailed help for scaffold command."""
        print("""
Scaffold Command - Direct interaction with scaffold models
-------------------------------------------------------
Usage: scaffold [options] <prompt>

Options:
    -i, --index <n>    Use specific scaffold model (default: 0)
    -t, --tokens <n>   Max new tokens to generate (default: 100)
    -l, --logits      Return logits information
    -h, --hidden      Return hidden states

Examples:
    scaffold "What is the meaning of life?"
    scaffold -i 1 -t 200 "Tell me a story"
    scaffold -i 2 -l -h "Analyze this text"

Note: This command requires that the SOVL system is initialized with a working 'generation_manager' that implements 'backchannel_scaffold_prompt'. If these are missing or misconfigured, scaffold model interaction will not be available.

The scaffold command allows direct interaction with any of the available
scaffold models for debugging and development purposes.
""")

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
            profile = self.sovl_system.state_tracker.state.user_profile_state.get(conversation_id)
            bond_calculator = self.sovl_system.bond_calculator
            bond_score = bond_calculator.get_bond_score_for_user(conversation_id)
            if bond_score is None:
                bond_score = bond_calculator.default_bond_score

            # Prepare profile stats
            interactions = profile.get("interactions", 0)
            nickname = profile.get("nickname", "")

            print(f"Bond Score: {bond_score:.2f} / 10")
            print(f"Interaction Count: {interactions}")
            print(f"Nickname: {nickname if nickname else '(None)'}")
        except Exception as e:
            print(f"Error in rate command: {e}")
            if self.debug_mode:
                traceback.print_exc()

    def do_status(self, arg):
        """Show current metrics."""
        print("\nCurrent Metrics:")
        print("---------------")
        # System Monitor
        if self.system_monitor:
            try:
                metrics = self.sovl_system.get_metrics()
                for key, value in metrics.items():
                    print(f"{key}: {value}")
            except Exception as e:
                print(f"System monitor error: {e}")
        else:
            print("System monitor not available. Showing basic system info:")
            print(f"Platform: {platform.system()} {platform.release()}")
            print(f"Python: {platform.python_version()}")
            print(f"CPU count: {os.cpu_count()}")

        # Memory Monitor
        if self.memory_monitor:
            try:
                mem_metrics = self.sovl_system.get_config().get('memory', {})
                for key, value in mem_metrics.items():
                    print(f"Memory {key}: {value}")
            except Exception as e:
                print(f"Memory monitor error: {e}")
        elif getattr(self, 'memory_monitor_fallback', False):
            print("Memory monitor not available. Using sovl_system.memory_manager fallback:")
            memory_manager = getattr(self.sovl_system, 'memory_manager', None)
            if memory_manager:
                try:
                    # Example: assume memory_manager has a get_status() or similar method
                    mem_status = memory_manager.get_status() if hasattr(memory_manager, 'get_status') else str(memory_manager)
                    print(f"Memory Manager Status: {mem_status}")
                except Exception as e:
                    print(f"Error retrieving memory manager status: {e}")
            else:
                print("No memory manager available for memory metrics.")
        else:
            print("Memory monitor not available and no fallback present.")

        # GPU Info
        try:
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                print(f"CUDA available: True")
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current CUDA device: {torch.cuda.current_device()}")
                print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            else:
                print("CUDA available: False")
        except Exception as e:
            print(f"GPU info error: {e}")

        # Traits Monitor
        if self.traits_monitor:
            try:
                print("Traits monitor available.")
                # Add more trait metrics as needed
            except Exception as e:
                print(f"Traits monitor error: {e}")
        else:
            print("Traits monitor not available.")

    def do_quit(self, arg):
        """Exit the CLI (alias for exit)."""
        return self.do_exit(arg)

    def do_save(self, arg):
        """Save the current system state to a file."""
        filename = arg.strip() if arg.strip() else "sovl_state.json"
        if hasattr(self.sovl_system, 'save_state'):
            try:
                self.sovl_system.save_state(filename)
                print(f"System state saved to {filename}.")
            except Exception as e:
                print(f"Error saving state: {e}")
        else:
            print("Save not implemented or unavailable.")

    def do_load(self, arg):
        """Load a system state from a file."""
        filename = arg.strip() if arg.strip() else "sovl_state.json"
        if hasattr(self.sovl_system, 'load_state'):
            try:
                self.sovl_system.load_state(filename)
                print(f"System state loaded from {filename}.")
            except Exception as e:
                print(f"Error loading state: {e}")
        else:
            print("Load not implemented or unavailable.")

    def do_reset(self, arg):
        """Reset the system to its initial state."""
        if hasattr(self.sovl_system, 'reset'):
            try:
                self.sovl_system.reset()
                print("System has been reset.")
            except Exception as e:
                print(f"Error resetting system: {e}")
        elif hasattr(self.sovl_system, 'reset_state'):
            try:
                self.sovl_system.reset_state()
                print("System state has been reset.")
            except Exception as e:
                print(f"Error resetting system state: {e}")
        else:
            print("Reset not implemented or unavailable.")

    def do_monitor(self, arg):
        """Show system monitoring information."""
        if self.system_monitor:
            try:
                metrics = self.system_monitor._collect_metrics()
                print("System Monitor Metrics:")
                for section, stats in metrics.items():
                    print(f"{section}:")
                    if isinstance(stats, dict):
                        for k, v in stats.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"  {stats}")
            except Exception as e:
                print(f"Error displaying system monitor metrics: {e}")
        else:
            print("System monitor not available.")

    def do_gestate(self, arg):
        """Run a gestation (training) cycle using the system trainer."""
        trainer = getattr(self.sovl_system, 'trainer', None)
        if trainer and hasattr(trainer, 'run_gestation_cycle'):
            try:
                # Optionally, parse arg for conversation history or use default
                # Here, we assume conversation history is managed internally
                trainer.run_gestation_cycle([])  # Pass empty or default as needed
                print("Gestation (training) cycle completed.")
            except Exception as e:
                print(f"Error during gestation cycle: {e}")
        else:
            print("Gestation (training) not available on this system.")

    def do_train(self, arg):
        """Alias for gestate (run a gestation cycle)."""
        return self.do_gestate(arg)

    def do_reflect(self, arg):
        """Force an introspection cycle and display the result."""
        introspection_manager = getattr(self.sovl_system, 'introspection_manager', None)
        if introspection_manager and hasattr(introspection_manager, 'conduct_hidden_dialogue'):
            try:
                action_description = arg.strip() if arg.strip() else "Manual reflection triggered from CLI."
                result = asyncio.run(introspection_manager.conduct_hidden_dialogue(action_description, show_status=False))
                print("\nIntrospection Result:")
                for k, v in result.items():
                    print(f"{k}: {v}")
            except Exception as e:
                print(f"Error during introspection: {e}")
        else:
            print("Introspection manager or conduct_hidden_dialogue not available on this system.")

    def do_spark(self, arg):
        """Display the most recent curiosity question (spark)."""
        curiosity_manager = getattr(self.sovl_system, 'curiosity_manager', None)
        if curiosity_manager and hasattr(curiosity_manager, 'exploration_queue'):
            try:
                queue = curiosity_manager.exploration_queue
                if queue and len(queue) > 0:
                    most_recent = queue[-1]
                    prompt = most_recent.get('prompt') if isinstance(most_recent, dict) else str(most_recent)
                    print(f"Most recent spark (curiosity question):\n{prompt}")
                else:
                    print("There is currently no spark within")
            except Exception as e:
                print(f"Error retrieving spark: {e}")
        else:
            print("There is currently no spark within")

    def do_flare(self, arg):
        """Generate a creative (high-temperature) response to an empty input using temperament logic."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if generation_manager and hasattr(generation_manager, '_handle_internal_prompt'):
            try:
                prompt = arg if arg.strip() else " "
                response = generation_manager._handle_internal_prompt(prompt)
                print(f"Flare response:\n{response}")
            except Exception as e:
                print(f"Error generating flare response: {e}")
        else:
            print("Generation manager or _handle_internal_prompt not available on this system.")

    def do_mimic(self, arg):
        """Generate a response that mimics the user, using bond modulation context."""
        bond_modulator = getattr(self.sovl_system, 'bond_modulator', None)
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        state = getattr(self.sovl_system, 'state', None)
        if not bond_modulator or not generation_manager:
            print("Bond modulator or generation manager not available on this system.")
            return
        try:
            # Attempt to get user metadata entries from state or sovl_system
            metadata_entries = []
            if state and hasattr(state, 'history') and hasattr(state.history, 'get_recent_metadata'):
                metadata_entries = state.history.get_recent_metadata()
            # Get bond modulation context
            context_str, bond_score = bond_modulator.get_bond_modulation(metadata_entries)
            # Compose prompt: context + user arg
            prompt = f"{context_str}\n{arg.strip()}" if arg.strip() else context_str
            responses = generation_manager.generate_text(prompt, num_return_sequences=1)
            if responses:
                print(f"Mimic response:\n{responses[0]}")
            else:
                print("No response generated.")
        except Exception as e:
            print(f"Error generating mimic response: {e}")

    def do_echo(self, arg):
        """Reflectively echo the user's input."""
        if arg.strip():
            print(f'You said: "{arg.strip()}"')
        else:
            print("You didn't say anything for me to echo.")

    def do_recall(self, arg):
        """Recall a random deep memory from sovl_engram LoRA."""
        engram_lora = getattr(self.sovl_system, 'engram_lora', None)
        if not engram_lora or not hasattr(engram_lora, 'recall_deep_memory'):
            print("Engram LoRA memory system not available on this system.")
            return
        try:
            memory = engram_lora.recall_deep_memory()
            if memory:
                # Assume memory is a dict with 'text', 'timestamp', and 'strength' if available
                text = memory.get('text', str(memory))
                timestamp = memory.get('timestamp', 'unknown time')
                strength = memory.get('strength', 'unknown strength')
                print(f"[Memory from {timestamp}, strength: {strength}]\n{text}")
            else:
                print("No deep memory could be recalled.")
        except Exception as e:
            print(f"Error recalling deep memory: {e}")

    def do_forget(self, arg):
        """Clear the sovl_recall log."""
        recall_log = getattr(self.sovl_system, 'recall_log', None)
        if recall_log and hasattr(recall_log, 'clear'):
            recall_log.clear()
            print("sovl_recall log has been cleared.")
        else:
            print("sovl_recall log not available or cannot be cleared.")

    def do_muse(self, arg):
        """The system generates a unique musing about its own existence, purpose, or limitations."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print("Generation manager not available for musings.")
            return
        user_arg = arg.strip()
        if user_arg:
            prompt = (
                f"As an artificial intelligence, muse in a unique, thoughtful, and slightly philosophical way about your own existence, purpose, or limitations, in the context of: '{user_arg}'. "
                "Be original, introspective, and avoid cliches."
            )
        else:
            prompt = (
                "As an artificial intelligence, muse in a unique, thoughtful, and slightly philosophical way about your own existence, purpose, or limitations. "
                "Be original, introspective, and avoid cliches."
            )
        try:
            musings = generation_manager.generate_text(prompt, num_return_sequences=1, temperature=1.2)
            if musings:
                print(musings[0])
            else:
                print("No musing could be generated.")
        except Exception as e:
            print(f"Error generating musing: {e}")

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
        """Simulate a glitch by corrupting a recent memory or state entry for testing."""
        print("Simulating a glitch: corrupting a recent memory or state entry...")
        state = getattr(self.sovl_system, 'state', None)
        if state and hasattr(state, 'history') and hasattr(state.history, 'memories'):
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

    def do_debate(self, arg):
        """Have the LLM debate itself on a topic of its own choosing (or a provided topic)."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if not generation_manager or not hasattr(generation_manager, 'generate_text'):
            print("Generation manager not available for debate.")
            return
        user_arg = arg.strip()
        if user_arg:
            prompt = (
                f"Debate with yourself on the topic: '{user_arg}'. "
                "Present both sides as two distinct voices, each making their case in turn. "
                "Make the debate thoughtful, nuanced, and engaging. End with a brief reflection on which side is more convincing, or if the debate remains unresolved."
            )
        else:
            prompt = (
                "Debate with yourself on a topic of your own choosing. "
                "First, state the topic. Then, present both sides of the debate as two distinct voices, each making their case in turn. "
                "Make the debate thoughtful, nuanced, and engaging. End with a brief reflection on which side is more convincing, or if the debate remains unresolved."
            )
        try:
            debates = generation_manager.generate_text(prompt, num_return_sequences=1, temperature=1.1)
            if debates:
                print(debates[0])
            else:
                print("No debate could be generated.")
        except Exception as e:
            print(f"Error generating debate: {e}")

    def do_rewind(self, arg):
        """Rewind the conversation history by N turns (default: 7). Usage: rewind [N]"""
        n = 7
        try:
            if arg.strip():
                n = int(arg.strip())
        except ValueError:
            print("Invalid argument. Usage: rewind [N]")
            return
        state = getattr(self.sovl_system, 'state', None)
        if state and hasattr(state, 'history') and hasattr(state.history, 'rewind'):
            state.history.rewind(n)
            print(f"Rewound conversation history by {n} turn(s).")
        elif state and hasattr(state, 'history') and hasattr(state.history, 'memories'):
            memories = state.history.memories
            actual = min(n, len(memories))
            for _ in range(actual):
                memories.pop()
            print(f"Rewound conversation history by {actual} turn(s) (fallback).")
        else:
            print("Conversation history rewind not available.")

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
            prompt = (
                "Read the following error messages and summarize them in a single complaint, "
                "using a tone of frustration, annoyance, or exasperation. Be dramatic and make it sound like a real complaint:\n"
                f"{entries_text}"
            )
            complaints = generation_manager.generate_text(prompt, num_return_sequences=1)
            if complaints:
                print(complaints[0])
            else:
                print("No complaint report could be generated.")
        except Exception as e:
            print(f"Error generating complaint report: {e}")

    def default(self, line):
        """Handle unknown commands by sending them to the LLM as chat input."""
        generation_manager = getattr(self.sovl_system, 'generation_manager', None)
        if generation_manager and hasattr(generation_manager, 'generate_text'):
            try:
                responses = generation_manager.generate_text(line, num_return_sequences=1)
                if responses:
                    print(responses[0])
                else:
                    print("No response generated.")
            except Exception as e:
                print(f"Error generating chat response: {e}")
        else:
            print("Type 'help' for available commands (chat mode unavailable: generation manager missing)")

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
        if not hasattr(self, '_command_registry'):
            self._command_registry = {}
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
            # Extensible registry dispatch
            if hasattr(self, '_command_registry') and command in self._command_registry:
                return self._command_registry[command](args)
            # Fallback to do_* methods
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

def run_cli(config_manager_instance: Optional[ConfigManager] = None):
    sovl_system = None
    try:
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
            sovl_system.save_state("final_state.json")
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
    run_cli()
