import torch
from typing import Dict, List, Optional, Any, Callable
import time
from threading import Lock
from collections import deque
import json
from sovl_logger import Logger
from sovl_memory import GPUMemoryManager, RAMManager
from sovl_tuner import SOVLTuner
from sovl_resonator_unattached import SOVLResonator
from sovl_curiosity import Curiosity
import traceback
from abc import ABC, abstractmethod
import sovl_state
import random

class SensationNode(ABC):
    """
    Abstract base class for SOVL sensation nodes (e.g., vision, audio, etc.).
    All sensation nodes must implement this interface to integrate with volition.
    """
    @abstractmethod
    def get_status(self) -> dict:
        """
        Return a summary of the node's current status (health, readiness, etc.).
        Returns:
            A dictionary with status information.
        """
        pass

    @abstractmethod
    def get_latest_observation(self) -> dict:
        """
        Return the most recent observation/data from the sensation node.
        Returns:
            A dictionary containing the latest observation.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the node's state if applicable.
        """
        pass

    def close(self) -> None:
        """
        Optional: Clean up resources (e.g., camera, microphone) when shutting down.
        """
        pass

class AutonomyManager:
    """
    A lightweight decision-making framework for autonomous system optimization in the SOVL System.
    Processes system metrics and uses LLM-based reasoning to make decisions, initially for memory health.
    Now also integrates with Motivator for dynamic, emergent goal-driven behavior.
    """
    def __init__(self, config_manager, logger: Logger, device: torch.device, system_ref, tuner: Optional[SOVLTuner] = None, motivator=None):
        """
        Initialize the AutonomyManager.

        Args:
            config_manager: ConfigManager instance for accessing configuration.
            logger: Logger instance for recording events and errors.
            device: Torch device (cuda/cpu) for tensor operations.
            system_ref: Reference to SOVLSystem instance for triggering actions.
            tuner: SOVLTuner instance for dynamic parameter tuning (optional).
            motivator: Motivator instance for goal-driven behavior (optional).
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.system_ref = system_ref
        self.tuner = tuner  # Link to SOVLTuner for dynamic parameter tuning
        self.memory_lock = Lock
        self.motivator = motivator  # <-- Integration point
        # Cache configuration
        self.controls_config = config_manager.get_section("controls_config")
        self.autonomy_config = config_manager.get_section("autonomy_config", {
            "enable_autonomy": True,
            "memory_threshold": 0.85,
            "error_rate_threshold": 0.1,
            "decision_cooldown": 60.0,
            "hysteresis_window": 5,
            "max_history_len": 10,
            "prompt_template": (
                "Given the following system metrics:\n"
                "Memory Usage: {memory_usage:.2%}\n"
                "Error Rate: {error_rate:.2%}\n"
                "Stability Score: {stability_score:.2f}\n"
                "Decide if adjustments are needed. Respond only with 'true' or 'false'."
            ),
            "diagnostic_interval": 300.0,
            "context_window": 3,
            "max_prompt_len": 500,  # New: Limit prompt/response length
            "action_timeout": 10.0,  # New: Timeout for actions
            "fallback_decision_limit": 3,  # New: Fallback after repeated LLM failures
            "strict_mode": False,  # Added: strict vs best-effort mode
            "enable_autosys_control": False  # New: Automated system control (LLM-driven throttle/pause/shutdown)
        })
        
        # State tracking
        self.decision_history = deque(maxlen=self.autonomy_config["max_history_len"])
        self.error_counts = deque(maxlen=self.autonomy_config["hysteresis_window"])
        self.last_decision_time = 0.0
        self.diagnostic_last_run = 0.0
        self.context_memory = deque(maxlen=self.autonomy_config["context_window"])
        self.consecutive_llm_failures = 0
        self.start_time = time.time()  # New: Track system uptime
        self.last_metrics = None
        self.last_decision = None
        self.last_diagnostics = None
        self.last_action_result = None
        self.action_registry = {}
        self.pending_shutdown = False
        
        # Sense nodes registry: name -> SensationNode
        self.sense_nodes: dict = {}

        # RAM and GPU managers
        self.ram_manager = RAMManager()  # For advanced RAM stats
        self.gpu_manager = GPUMemoryManager()  # Already present

        # --- LLM Decision Integration: Attach GenerationManager ---
        # Attempt to get a generation_manager from system_ref, else raise error on first use
        self.generation_manager = getattr(system_ref, 'generation_manager', None)
        if self.generation_manager is None:
            self.logger.record_event(
                event_type="generation_manager_missing",
                message="No generation_manager found on system_ref. LLM decisions will fail until attached.",
                level="error",
                additional_info={"timestamp": time.time()}
            )

        self.logger.record_event(
            event_type="autonomy_manager_initialized",
            message="AutonomyManager initialized",
            level="info",
            additional_info={
                "config": {k: v for k, v in self.autonomy_config.items() if k != "prompt_template"},
                "timestamp": time.time()
            }
        )

    def collect_metrics(self) -> Dict[str, float]:
        """
        Collect system metrics for decision-making (memory usage, error rate, stability score).

        Returns:
            Dict of metric names to values.
        """
        try:
            metrics = {
                "memory_usage": 0.0,
                "error_rate": 0.0,
                "stability_score": 1.0
            }
            # Memory usage with fallback
            try:
                # Use RAMManager for RAM usage
                ram_manager = RAMManager(self.config_manager, self.logger)
                ram_stats = ram_manager.check_memory_health()
                current_mem = ram_stats.get('used', 0.0)
                total_mem = ram_stats.get('total', 0.0)
                metrics["memory_usage"] = current_mem / total_mem if total_mem > 0 else 0.0
            except Exception as e:
                self.logger.record_event(
                    event_type="memory_stats_unavailable",
                    message=f"Failed to get RAM memory stats: {str(e)}",
                    level="warning"
                )
            # Error rate with smoothing
            try:
                recent_logs = self.logger.read(limit=max(1, self.autonomy_config["hysteresis_window"]))
                error_count = sum(1 for log in recent_logs if "error" in log.get("event", "").lower())
                metrics["error_rate"] = error_count / max(1, len(recent_logs))
            except Exception as e:
                self.logger.record_event(
                    event_type="error_rate_calc_error",
                    message=f"Failed to calculate error rate: {str(e)}",
                    level="error",
                    stack_trace=traceback.format_exc(),
                    additional_info={"timestamp": time.time()}
                )
                metrics["error_rate"] = 0.0
            # Stability score
            try:
                metrics["stability_score"] = max(0.0, min(1.0, 1.0 - (0.7 * metrics["memory_usage"] + 0.3 * metrics["error_rate"])))
            except Exception as e:
                self.logger.record_event(
                    event_type="stability_score_error",
                    message=f"Failed to calculate stability score: {str(e)}",
                    level="error",
                    stack_trace=traceback.format_exc(),
                    additional_info={"timestamp": time.time()}
                )
                metrics["stability_score"] = 1.0
            self.last_metrics = metrics
            self.logger.record_event(
                event_type="metrics_collected",
                message="System metrics collected",
                level="info",
                additional_info={"metrics": {k: round(v, 4) for k, v in metrics.items()}, "timestamp": time.time()}
            )
            return metrics
        except Exception as e:
            self.logger.record_event(
                event_type="metrics_collection_failed",
                message=f"Failed to collect metrics: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"timestamp": time.time()}
            )
            return {
                "memory_usage": 0.0,
                "error_rate": 0.0,
                "stability_score": 1.0
            }

    def validate_metrics(self, metrics: Dict[str, float]) -> bool:
        """
        Validate that metrics are within acceptable ranges.

        Args:
            metrics: Dictionary of metric names to values.
        Returns:
            True if valid, raises ValueError if invalid.
        """
        try:
            if not (0.0 <= metrics["memory_usage"] <= 1.0):
                raise ValueError("Memory usage out of range")
            if not (0.0 <= metrics["error_rate"] <= 1.0):
                raise ValueError("Error rate out of range")
            if not (0.0 <= metrics["stability_score"] <= 1.0):
                raise ValueError("Stability score out of range")
            return True
        except Exception as e:
            self.logger.record_event(
                event_type="metrics_validation_failed",
                message=f"Metrics validation failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"metrics": metrics, "timestamp": time.time()}
            )
            if self.autonomy_config.get("strict_mode", False):
                raise
            return False

    def build_prompt(self, metrics: Dict[str, float]) -> str:
        """
        Build a prompt for the LLM based on metrics.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            Formatted prompt string.
        """
        try:
            prompt = self.autonomy_config["prompt_template"].format(**metrics)
            if len(prompt) > self.autonomy_config["max_prompt_len"]:
                prompt = prompt[:self.autonomy_config["max_prompt_len"]]
            self.logger.record_event(
                event_type="prompt_built",
                message="Prompt built for LLM",
                level="info",
                additional_info={"prompt": prompt, "timestamp": time.time()}
            )
            return prompt
        except Exception as e:
            self.logger.record_event(
                event_type="prompt_build_failed",
                message=f"Failed to build prompt: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"metrics": metrics, "timestamp": time.time()}
            )
            if self.autonomy_config.get("strict_mode", False):
                raise
            return ""

    def llm_decide(self, prompt: str, allowed_responses: list) -> str:
        """
        Query the LLM with the prompt and parse the response into an allowed action string.
        Args:
            prompt: The structured prompt string.
            allowed_responses: List of valid responses (e.g., ["continue", "pause", ...] or ["yes", "no"])
        Returns:
            Action string chosen by the LLM.
        """
        self.logger.record_event(
            event_type="llm_prompt_sent",
            message="Prompt sent to LLM",
            level="debug",
            additional_info={"prompt": prompt, "timestamp": time.time(), "allowed_responses": allowed_responses}
        )
        llm_output = None
        try:
            if self.generation_manager is None:
                raise RuntimeError("No generation_manager attached to AutonomyManager.")
            result = self.generation_manager.generate_text(prompt, num_return_sequences=1, user_id="autonomy_manager")
            if isinstance(result, list) and result:
                llm_output = result[0].strip().lower()
            else:
                llm_output = None
        except Exception as e:
            self.logger.record_event(
                event_type="llm_decision_error",
                message=f"LLM decision error: {str(e)}",
                level="error",
                additional_info={"prompt": prompt, "timestamp": time.time()}
            )
            llm_output = None
        self.logger.record_event(
            event_type="llm_decision_output",
            message="LLM decision output",
            level="debug",
            additional_info={"llm_output": llm_output, "timestamp": time.time()}
        )
        # Validate output
        allowed_set = set(map(str.lower, allowed_responses))
        if llm_output not in allowed_set:
            self.logger.record_event(
                event_type="llm_decision_invalid",
                message=f"Invalid LLM action '{llm_output}', falling back to '{allowed_responses[0]}'.",
                level="warning",
                additional_info={"timestamp": time.time()}
            )
            return allowed_responses[0]  # Safe fallback
        return llm_output

    def execute_action(self, action: str) -> None:
        """
        Execute system actions based on the string action name, with robust curiosity/explore fallback.
        Args:
            action: Action string (e.g., 'continue', 'throttle', 'pause', 'explore', etc.)
        """
        if action == "continue":
            return  # No-op for normal operation
        elif action == "throttle":
            if self.tuner:
                self.tuner.update_parameters({
                    "data_config.batch_size": 1,
                    "generation_config.temperature": 0.6
                })
                self.logger.record_event(
                    event_type="autotune_applied",
                    message="Batch size and temperature reduced due to throttle action",
                    additional_info={"timestamp": time.time()}
                )
        elif action == "pause":
            if hasattr(self.system_ref, "pause"):
                self.system_ref.pause()
                self.logger.record_event(event_type="system_paused", message="System paused by LLM decision.")
        elif action == "soft_shutdown":
            if hasattr(self.system_ref, "soft_shutdown"):
                self.system_ref.soft_shutdown()
                self.logger.record_event(event_type="soft_shutdown", message="System entered soft shutdown mode.")
        elif action == "shutdown":
            if hasattr(self.system_ref, "shutdown"):
                self.system_ref.shutdown()
                self.logger.record_event(event_type="shutdown", message="System shutdown initiated by LLM.")
        elif action == "explore":
            # Robust curiosity-driven question generation
            if hasattr(self, "curiosity") and self.curiosity:
                try:
                    question = None
                    if hasattr(self.curiosity, "generate_curiosity_question"):
                        question = self.curiosity.generate_curiosity_question(context="explore", spontaneous=True)
                    self.logger.record_event(
                        event_type="curiosity_question_generated",
                        message=f"Curiosity-driven question: {question}",
                        additional_info={"timestamp": time.time()}
                    )
                    if question and hasattr(self, "motivator") and self.motivator:
                        self.motivator.consider_goal(description=question, source="curiosity")
                except Exception as e:
                    self.logger.record_event(
                        event_type="curiosity_question_failed",
                        message=f"Failed to generate curiosity question: {str(e)}",
                        level="error",
                        additional_info={"timestamp": time.time()}
                    )
            else:
                self.logger.record_event(
                    event_type="explore_action_skipped",
                    message="Curiosity module not available; skipping explore action.",
                    level="warning",
                    additional_info={"timestamp": time.time()}
                )
        else:
            # Try action registry if present
            if hasattr(self, "action_registry") and action in self.action_registry:
                try:
                    self.action_registry[action]()
                except Exception as e:
                    self.logger.record_event(
                        event_type="action_failed",
                        message=f"Failed to execute action '{action}': {str(e)}",
                        level="error",
                        additional_info={"timestamp": time.time()}
                    )
            else:
                self.logger.record_event(
                    event_type="unknown_action",
                    message=f"Unknown action '{action}' received.",
                    level="warning",
                    additional_info={"timestamp": time.time()}
                )

    def aggregate_sensory_context(self) -> dict:
        """
        Collect the latest observation from each registered sense node and system/system resource state.
        Returns:
            Dictionary summarizing all sensory, core system, and resource state.
        """
        context = {}
        for name, node in self.sense_nodes.items():
            try:
                context[name] = node.get_latest_observation()
            except Exception as e:
                self.logger.record_event(
                    event_type="sense_observation_failed",
                    message=f"Failed to get observation from '{name}': {str(e)}",
                    level="warning",
                    additional_info={"timestamp": time.time()}
                )
        # Add system state (battery, error, etc.)
        if hasattr(self.system_ref, 'get_battery_level'):
            try:
                context["battery"] = self.system_ref.get_battery_level()
            except Exception:
                context["battery"] = None
        error_state = None
        if hasattr(self.system_ref, 'get_last_error'):
            try:
                error_state = self.system_ref.get_last_error()
            except Exception:
                error_state = None
        context["error"] = error_state
        if hasattr(self, 'error_counts') and self.error_counts:
            context["error_rate"] = sum(self.error_counts) / len(self.error_counts)
            context["error_count"] = len(self.error_counts)
        else:
            context["error_rate"] = 0.0
            context["error_count"] = 0
        context["throttle_level"] = getattr(self, "throttle_level", 0)
        # Add system resource metrics (RAM, GPU)
        try:
            # Use RAMManager for RAM usage
            ram_manager = RAMManager(self.config_manager, self.logger)
            ram_stats = ram_manager.check_memory_health()
            ram_used = ram_stats.get('used', 0.0)
            ram_total = ram_stats.get('total', None)
            context["ram_usage"] = ram_used
            context["ram_total"] = ram_total
            context["ram_usage_pct"] = ram_used / ram_total if ram_total else None
            # Add advanced RAM stats from RAMManager if available
            if hasattr(self, "ram_manager") and self.ram_manager is not None:
                ram_stats = self.ram_manager.get_stats() if hasattr(self.ram_manager, "get_stats") else {}
                for k, v in ram_stats.items():
                    context[f"rammgr_{k}"] = v
        except Exception:
            context["ram_usage"] = None
            context["ram_total"] = None
            context["ram_usage_pct"] = None
        # GPU memory (use GPUMemoryManager for system standard)
        try:
            if hasattr(self, "gpu_manager") and self.gpu_manager is not None:
                gpu_stats = self.gpu_manager.get_stats() if hasattr(self.gpu_manager, "get_stats") else {}
                for k, v in gpu_stats.items():
                    context[f"gpumgr_{k}"] = v
                # For backward compatibility, also set top-level keys if present
                if "used" in gpu_stats:
                    context["gpu_mem_used"] = gpu_stats["used"]
                if "total" in gpu_stats:
                    context["gpu_mem_total"] = gpu_stats["total"]
                if "used" in gpu_stats and "total" in gpu_stats and gpu_stats["total"]:
                    context["gpu_mem_used_pct"] = gpu_stats["used"] / gpu_stats["total"]
                else:
                    context["gpu_mem_used_pct"] = None
            else:
                # Fallback to previous torch-based logic if no manager
                torch_available = False
                try:
                    import torch
                    torch_available = True
                except ImportError:
                    torch_available = False
                device = getattr(self, "device", None)
                if torch_available and isinstance(device, type(getattr(torch, "device", None) and torch.device("cpu"))):
                    if device.type == "cuda" and torch.cuda.is_available():
                        gpu_used = torch.cuda.memory_allocated(device)
                        gpu_total = torch.cuda.get_device_properties(device).total_memory
                        context["gpu_mem_used"] = gpu_used
                        context["gpu_mem_total"] = gpu_total
                        context["gpu_mem_used_pct"] = gpu_used / gpu_total if gpu_total else None
                    else:
                        context["gpu_mem_used"] = None
                        context["gpu_mem_total"] = None
                        context["gpu_mem_used_pct"] = None
                else:
                    context["gpu_mem_used"] = None
                    context["gpu_mem_total"] = None
                    context["gpu_mem_used_pct"] = None
        except Exception:
            context["gpu_mem_used"] = None
            context["gpu_mem_total"] = None
            context["gpu_mem_used_pct"] = None
        # Centralize curiosity score/label logic here
        if hasattr(self, "curiosity") and self.curiosity:
            context["curiosity_level"] = getattr(self.curiosity, "curiosity_score", 0.0)
            score = context["curiosity_level"]
            if score > 0.7:
                context["curiosity_label"] = "high"
            elif score > 0.4:
                context["curiosity_label"] = "medium"
            else:
                context["curiosity_label"] = "low"
        return context

    def build_structured_prompt(self, context: dict) -> str:
        """
        Convert the aggregated context into a structured prompt for the LLM.
        Args:
            context: Aggregated sensory/system context dict.
        Returns:
            Formatted prompt string.
        """
        prompt_lines = ["System Context:"]
        for key, value in context.items():
            if key.endswith("_pct") and value is not None:
                prompt_lines.append(f"  - {key.replace('_', ' ').capitalize()}: {value:.2%}")
            else:
                prompt_lines.append(f"  - {key.replace('_', ' ').capitalize()}: {value}")
        # Expanded action space for throttling, pausing, shutdown, with explanations
        action_space = [
            "continue: normal operation",
            "throttle: slow down processing to reduce load/errors",
            "pause: temporarily halt all actions to recover",
            "soft_shutdown: enter safe idle mode, await confirmation",
            "shutdown: fully power down (requires confirmation)"
        ]
        if "curiosity_label" in context and context["curiosity_label"] == "high":
            action_space = ["explore: seek new information or try something novel"] + action_space
        prompt_lines.append("Choose action (see explanations):")
        for action in action_space:
            prompt_lines.append(f"  - {action}")
        prompt_lines.append(
            "If errors or resource usage are high, choose throttle, pause, or soft_shutdown. Only choose shutdown if instructed or if the system is unsafe."
        )
        return "\n".join(prompt_lines)

    def check_and_act(self) -> None:
        """
        Main loop to aggregate context, prompt the LLM, and execute the chosen action.
        """
        if not self.autonomy_config["enable_autonomy"]:
            return
        if not getattr(self, "enable_autosys_control", False):
            return
        try:
            with self.memory_lock:
                context = self.aggregate_sensory_context()
                if self.tuner:
                    context["tunable_parameters"] = self.tuner.get_current_parameters()
                prompt = self.build_structured_prompt(context)
                allowed_actions = ["continue", "throttle", "pause", "soft_shutdown", "shutdown"]
                if context.get("curiosity_label") == "high":
                    allowed_actions = ["explore"] + allowed_actions
                    self.logger.record_event(
                        event_type="curiosity_action_space_expanded",
                        message="Curiosity is high, adding 'explore' to allowed actions.",
                        additional_info={"curiosity_level": context["curiosity_level"], "timestamp": time.time()}
                    )
                action = self.llm_decide(prompt, allowed_actions)
                self.execute_action(action)
        except Exception as e:
            self.logger.record_event(
                event_type="autonomy_check_failed",
                message=f"Autonomy check failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"timestamp": time.time()}
            )

    def register_action(self, name: str, func: Callable[[], Any]) -> None:
        """
        Register a new autonomous action.

        Args:
            name: Name of the action.
            func: Callable to execute for this action.
        """
        self.action_registry[name] = func
        self.logger.record_event(
            event_type="action_registered",
            message=f"Action '{name}' registered",
            level="info",
            additional_info={"timestamp": time.time()}
        )

    def get_last_state(self) -> Dict[str, Any]:
        """
        Get the latest decision, metrics, diagnostics, and action result.

        Returns:
            Dictionary of last state information for monitoring or UI.
        """
        return {
            "last_metrics": self.last_metrics,
            "last_decision": self.last_decision,
            "last_diagnostics": self.last_diagnostics,
            "last_action_result": self.last_action_result,
            "context_memory": list(self.context_memory)
        }

    def extend_volition(self, *args, **kwargs) -> None:
        """
        Hook for future volition/cognitive modules (vision, etc.).
        """
        self.logger.record_event(
            event_type="volition_hook_called",
            message="Volition extension hook called",
            level="info",
            additional_info={"args": args, "kwargs": kwargs, "timestamp": time.time()}
        )

    def register_sense(self, name: str, node: SensationNode) -> None:
        """
        Register a new sense node (e.g., vision, audio) with the autonomy manager.
        Args:
            name: Unique name for the node.
            node: Instance of a class implementing SensationNode.
        """
        if name in self.sense_nodes:
            self.logger.record_event(
                event_type="sense_registration_failed",
                message=f"Sense node '{name}' already registered.",
                level="warning",
                additional_info={"timestamp": time.time()}
            )
            return
        self.sense_nodes[name] = node
        self.logger.record_event(
            event_type="sense_registered",
            message=f"Sense node '{name}' registered.",
            level="info",
            additional_info={"timestamp": time.time()}
        )

    def unregister_sense(self, name: str) -> None:
        """
        Unregister a sense node by name.
        Args:
            name: Name of the node to remove.
        """
        if name not in self.sense_nodes:
            self.logger.record_event(
                event_type="sense_unregistration_failed",
                message=f"Sense node '{name}' not found.",
                level="warning",
                additional_info={"timestamp": time.time()}
            )
            return
        node = self.sense_nodes.pop(name)
        try:
            node.close()
        except Exception:
            pass
        self.logger.record_event(
            event_type="sense_unregistered",
            message=f"Sense node '{name}' unregistered.",
            level="info",
            additional_info={"timestamp": time.time()}
        )

    def list_senses(self) -> dict:
        """
        List all registered sense nodes.
        Returns:
            Dict of sense node names and their class names.
        """
        return {name: type(node).__name__ for name, node in self.sense_nodes.items()}

    def act_on_goal(self, goal):
        """
        Example stub for acting on a goal.
        This could be LLM prompting, triggering an action, etc.
        Returns 'completed', 'dropped', or None.
        """
        # For now, just log and simulate completion
        self.logger.record_event(event_type="goal_action", message=f"Acting on goal: {goal.description}")
        # Simulate completion for demonstration
        return 'completed'

class Goal:
    def __init__(self, description, source, context, priority=1.0, status='active', metadata=None, completion=0.0, ephemeral=False, soft_completed=False, last_checked=None, relevance=1.0):
        self.description = description          # Natural language or structured
        self.source = source                   # 'user', 'curiosity', 'pattern', etc.
        self.context = context                 # Dict: user_id, session, etc.
        self.priority = priority               # Fluid, modulated by bond, context, etc.
        self.status = status                   # 'active', 'pending', 'stale', 'completed', etc.
        self.metadata = metadata or {}         # Arbitrary info
        self.created_at = time.time()
        self.last_updated = self.created_at
        self.completion = completion           # 0.0 (not started) to 1.0 (done)
        self.ephemeral = ephemeral             # If True, goal can be dropped easily
        self.soft_completed = soft_completed   # If True, goal was softly completed (not strictly finished)
        self.last_checked = last_checked or time.time()
        self.relevance = relevance             # 0.0 (irrelevant) to 1.0 (highly relevant)

    def decay_priority(self, decay_rate=0.01):
        # Decay priority and relevance over time
        time_passed = time.time() - self.last_updated
        self.priority *= (1 - decay_rate) ** time_passed
        self.relevance *= (1 - decay_rate/2) ** time_passed
        self.last_updated = time.time()

    def to_dict(self):
        return {
            "description": self.description,
            "source": self.source,
            "context": self.context,
            "priority": self.priority,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "completion": self.completion,
            "ephemeral": self.ephemeral,
            "soft_completed": self.soft_completed,
            "last_checked": self.last_checked,
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, data):
        goal = cls(
            description=data["description"],
            source=data["source"],
            context=data["context"],
            priority=data.get("priority", 1.0),
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
            completion=data.get("completion", 0.0),
            ephemeral=data.get("ephemeral", False),
            soft_completed=data.get("soft_completed", False),
            last_checked=data.get("last_checked", time.time()),
            relevance=data.get("relevance", 1.0),
        )
        goal.created_at = data.get("created_at", time.time())
        goal.last_updated = data.get("last_updated", goal.created_at)
        return goal

    def decay_priority(self, decay_rate=0.01):
        # Decay priority over time
        time_passed = time.time() - self.last_updated
        self.priority *= (1 - decay_rate) ** time_passed
        self.last_updated = time.time()

    def to_dict(self):
        return {
            "description": self.description,
            "source": self.source,
            "context": self.context,
            "priority": self.priority,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data):
        goal = cls(
            description=data["description"],
            source=data["source"],
            context=data["context"],
            priority=data.get("priority", 1.0),
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
        )
        goal.created_at = data.get("created_at", time.time())
        goal.last_updated = data.get("last_updated", goal.created_at)
        return goal

class Motivator:
    """
    Central driver of autonomous, dynamic, goal-directed behavior.
    Handles goal emergence, prioritization, decay, completion, and organic evolution.
    """
    def __init__(
        self,
        curiosity,
        memory,
        logger,
        config_manager,
        bonder,
        decay_rate=0.01,
        min_priority=0.1,
        completion_threshold=0.95,
        reevaluation_interval=60,
        memory_check_enabled=True,
        irrelevance_threshold=0.2,
        completion_drive=0.7,
        novelty_drive=0.2,
    ):
        self.curiosity = curiosity
        self.memory = memory
        self.logger = logger
        self.config_manager = config_manager
        self.bonder = bonder
        self.decay_rate = decay_rate
        self.min_priority = min_priority
        self.completion_threshold = completion_threshold
        self.reevaluation_interval = reevaluation_interval
        self.memory_check_enabled = memory_check_enabled
        self.irrelevance_threshold = irrelevance_threshold
        self.completion_drive = completion_drive
        self.novelty_drive = novelty_drive
        self.goals = []  # List[Goal]
        self.load_goals_from_memory()

    # --- Goal Emergence ---
    def consider_goal(self, description, source, context):
        """
        Called when a potential goal arises (user request, curiosity spike, pattern, etc.)
        """
        bond = self.bonder.get_bond(context.get('user_id')) if source == 'user' and context.get('user_id') else 1.0
        priority = self.estimate_priority(description, source, context, bond)
        if self.should_create_goal(priority):
            goal = Goal(description, source, context, priority)
            self.goals.append(goal)
            self.logger.record_event(event_type="goal_created", message=description, additional_info={"priority": priority, "source": source, "context": context})
            self.persist_goals()

    def estimate_priority(self, description, source, context, bond):
        # Use heuristics, bond, curiosity, context, etc. to estimate priority (future: temperament, novelty, etc.)
        base_priority = 1.0
        if source == 'user':
            base_priority *= bond
        # Add more dynamic influences here (curiosity, novelty, etc.)
        return base_priority

    def should_create_goal(self, priority):
        # Probabilistic or threshold-based decision, not hardcoded
        threshold = self.config_manager.get("goal_priority_threshold", 0.5)
        return priority > threshold or random.random() < priority

    # --- Goal Lifecycle Management ---
    def reevaluate_goals(self):
        """
        Periodically update priorities, decay old goals, drop or revive as needed.
        """
        for goal in self.goals:
            # Simulate memory/context checks (stub for now)
            goal.last_checked = time.time()
            # Example: If completion is high, soft-complete
            if goal.completion >= 0.95 and goal.status == 'active':
                goal.status = 'completed'
                goal.soft_completed = True
                self.logger.record_event("goal_soft_completed", message=goal.description)
            # Example: If relevance is low or goal is ephemeral and stale, drop it
            elif goal.relevance < 0.2 or (goal.ephemeral and goal.status == 'stale'):
                goal.status = 'dropped'
                self.logger.record_event("goal_dropped_irrelevant", message=goal.description)
            else:
                goal.decay_priority()
        # Remove dropped/completed goals from active list if you want ephemeral behavior
        self.goals = [g for g in self.goals if g.status not in ('dropped', 'completed') or g.soft_completed]
        self.persist_goals()

    def update_goal_progress(self, goal, progress_delta):
        goal.completion = min(max(goal.completion + progress_delta, 0.0), 1.0)
        if goal.completion >= 1.0:
            self.complete_goal(goal, soft=True)
        else:
            self.persist_goals()

    def complete_goal(self, goal, soft=False):
        goal.status = 'completed'
        goal.priority = 0
        goal.completion = 1.0
        goal.soft_completed = soft
        goal.metadata['completed_at'] = time.time()
        self.logger.record_event(event_type="goal_completed", message=goal.description)
        self.persist_goals()

    def drop_goal(self, goal, reason="decayed"):
        goal.status = 'dropped'
        goal.priority = 0
        goal.metadata['dropped_at'] = time.time()
        self.logger.record_event(event_type="goal_dropped", message=goal.description, additional_info={"reason": reason})
        self.persist_goals()

    # --- Persistence ---
    def persist_goals(self):
        # Save goals to sovl_state for cross-session continuity
        sovl_state_instance = getattr(sovl_state, 'state', sovl_state)
        sovl_state_instance.set_goals([g.to_dict() for g in self.goals])

    def load_goals_from_memory(self):
        sovl_state_instance = getattr(sovl_state, 'state', sovl_state)
        goal_dicts = sovl_state_instance.get_goals() if hasattr(sovl_state_instance, 'get_goals') else []
        self.goals = [Goal.from_dict(g) for g in goal_dicts]

    # --- Reflection/Introspection (Optional) ---
    def reflect_on_goals(self):
        """
        Agent can introspect on its own motivations and progress, for self-dialogue or explanation.
        """
        completed = [g for g in self.goals if g.status == 'completed']
        active = [g for g in self.goals if g.status == 'active']
        stale = [g for g in self.goals if g.status == 'stale']
        summary = f"Completed: {len(completed)}, Active: {len(active)}, Stale: {len(stale)}."
        return summary

    def get_params(self):
        return {
            "decay_rate": self.decay_rate,
            "min_priority": self.min_priority,
            "completion_threshold": self.completion_threshold,
            "reevaluation_interval": self.reevaluation_interval,
            "memory_check_enabled": self.memory_check_enabled,
            "irrelevance_threshold": self.irrelevance_threshold,
            "completion_drive": self.completion_drive,
            "novelty_drive": self.novelty_drive,
        }
