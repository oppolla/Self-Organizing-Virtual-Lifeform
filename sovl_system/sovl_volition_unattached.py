import torch
from typing import Dict, List, Optional, Any
import time
from threading import Lock
from collections import deque
import json
from sovl_utils import memory_usage, log_memory_usage, total_memory
from sovl_logger import Logger
from sovl_memory import GPUMemoryManager, RAMManager
from sovl_tuner import SOVLTuner
import traceback
from typing import Callable
from abc import ABC, abstractmethod

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
    """
    def __init__(self, config_manager, logger: Logger, device: torch.device, system_ref, tuner: Optional[SOVLTuner] = None):
        """
        Initialize the AutonomyManager.

        Args:
            config_manager: ConfigManager instance for accessing configuration.
            logger: Logger instance for recording events and errors.
            device: Torch device (cuda/cpu) for tensor operations.
            system_ref: Reference to SOVLSystem instance for triggering actions.
            tuner: SOVLTuner instance for dynamic parameter tuning (optional).
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.system_ref = system_ref
        self.tuner = tuner  # Link to SOVLTuner for dynamic parameter tuning
        self.memory_lock = Lock
        
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
                if torch.cuda.is_available():
                    gpu_manager = GPUMemoryManager(self.config_manager, self.logger)
                    gpu_stats = gpu_manager.get_gpu_usage()
                    if gpu_stats:
                        current_mem = gpu_stats.get('gpu_usage', 0.0)
                        total_mem = gpu_stats.get('total_memory', 0.0)
                        metrics["memory_usage"] = current_mem / total_mem if total_mem > 0 else 0.0
                    else:
                        self.logger.record_event(
                            event_type="memory_stats_unavailable",
                            message="Memory stats unavailable, using fallback value",
                            level="warning",
                            additional_info={"timestamp": time.time()}
                        )
            except Exception as e:
                self.logger.record_event(
                    event_type="gpu_stats_error",
                    message=f"Failed to get GPU stats: {str(e)}",
                    level="error",
                    stack_trace=traceback.format_exc(),
                    additional_info={"timestamp": time.time()}
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

    def make_decision(self, prompt: str) -> Optional[bool]:
        """
        Use the LLM to make a decision based on the prompt, with fallback for repeated failures.

        Args:
            prompt: Formatted prompt with metrics.

        Returns:
            True if adjustments are needed, False if not, None if decision fails.
        """
        try:
            # Placeholder: Replace with actual LLM call
            # For now, random decision for demonstration
            import random
            result = random.choice([True, False])
            self.last_decision = result
            self.logger.record_event(
                event_type="decision_made",
                message="Decision made by LLM or agent",
                level="info",
                additional_info={"decision": result, "prompt": prompt, "timestamp": time.time()}
            )
            self.consecutive_llm_failures = 0
            return result
        except Exception as e:
            self.consecutive_llm_failures += 1
            self.logger.record_event(
                event_type="decision_failed",
                message=f"Decision-making failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"prompt": prompt, "failures": self.consecutive_llm_failures, "timestamp": time.time()}
            )
            if self.consecutive_llm_failures >= self.autonomy_config.get("fallback_decision_limit", 3):
                self.logger.record_event(
                    event_type="decision_fallback_triggered",
                    message="Fallback triggered after repeated LLM failures",
                    level="warning",
                    additional_info={"timestamp": time.time()}
                )
                self.consecutive_llm_failures = 0
                return False  # Fallback: take safe action
            if self.autonomy_config.get("strict_mode", False):
                raise
            return None

    def execute_action(self, decision: bool) -> bool:
        """
        Execute system actions based on the decision with rollback on failure.

        Args:
            decision: True if adjustments are needed, False otherwise.

        Returns:
            True if actions were executed successfully, False otherwise.
        """
        try:
            action_result = False
            if decision:
                # Use action registry if available
                if self.action_registry:
                    for name, func in self.action_registry.items():
                        try:
                            func_result = func()
                            self.logger.record_event(
                                event_type="action_executed",
                                message=f"Action '{name}' executed",
                                level="info",
                                additional_info={"result": func_result, "timestamp": time.time()}
                            )
                        except Exception as e:
                            self.logger.record_event(
                                event_type="action_failed",
                                message=f"Action '{name}' failed: {str(e)}",
                                level="error",
                                stack_trace=traceback.format_exc(),
                                additional_info={"timestamp": time.time()}
                            )
                # Default system action (if no registry or as fallback)
                if hasattr(self.system_ref, "reset_state"):
                    self.system_ref.reset_state()
                    action_result = True
                    self.logger.record_event(
                        event_type="system_reset",
                        message="System state reset as autonomous action",
                        level="info",
                        additional_info={"timestamp": time.time()}
                    )
                else:
                    self.logger.record_event(
                        event_type="no_reset_method",
                        message="No reset_state method on system_ref",
                        level="warning",
                        additional_info={"timestamp": time.time()}
                    )
            else:
                action_result = True  # No action needed is still success
            self.last_action_result = action_result
            return action_result
        except Exception as e:
            self.logger.record_event(
                event_type="action_execution_failed",
                message=f"Action execution failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"timestamp": time.time()}
            )
            if self.autonomy_config.get("strict_mode", False):
                raise
            self.last_action_result = False
            return False

    def run_self_diagnostic(self) -> Dict[str, Any]:
        """
        Perform periodic self-diagnostic checks on system health.

        Returns:
            Dictionary of diagnostic results.
        """
        try:
            current_time = time.time()
            if current_time - self.diagnostic_last_run < self.autonomy_config["diagnostic_interval"]:
                return self.last_diagnostics or {"status": "skipped"}
            # Example diagnostics: uptime, context memory, last metrics
            diagnostics = {
                "status": "ok",
                "uptime": current_time - self.start_time,
                "context_memory_len": len(self.context_memory),
                "last_metrics": self.last_metrics,
                "timestamp": current_time
            }
            self.last_diagnostics = diagnostics
            self.diagnostic_last_run = current_time
            self.logger.record_event(
                event_type="self_diagnostic_ran",
                message="Self-diagnostic completed",
                level="info",
                additional_info=diagnostics
            )
            return diagnostics
        except Exception as e:
            self.logger.record_event(
                event_type="self_diagnostic_failed",
                message=f"Self-diagnostic failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"timestamp": time.time()}
            )
            return {"status": "failed", "error": str(e)}

    def update_context(self, prompt: str, response: str) -> None:
        """
        Update contextual memory with recent interactions, with truncation.

        Args:
            prompt: Input prompt.
            response: System response.
        """
        try:
            max_len = self.autonomy_config["max_prompt_len"]
            self.context_memory.append({
                "prompt": prompt[:max_len],
                "response": response[:max_len]
            })
            self.logger.record_event(
                event_type="context_updated",
                message="Context memory updated",
                level="info",
                additional_info={"context_length": len(self.context_memory), "timestamp": time.time()}
            )
        except Exception as e:
            self.logger.record_event(
                event_type="context_update_failed",
                message=f"Context update failed: {str(e)}",
                level="error",
                stack_trace=traceback.format_exc(),
                additional_info={"timestamp": time.time()}
            )
            self.reset_context()

    def reset_context(self) -> None:
        """
        Reset contextual memory if corrupted or on error.
        """
        self.context_memory = deque(maxlen=self.autonomy_config["context_window"])
        self.logger.record_event(
            event_type="context_reset",
            message="Context memory reset",
            level="info",
            additional_info={"timestamp": time.time()}
        )

    def aggregate_sensory_context(self) -> dict:
        """
        Collect the latest observation from each registered sense node and system/system resource state.
        Returns:
            Dictionary summarizing all sensory, core system, and resource state.
        """
        context = {}
        # Aggregate sensory nodes
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
            from sovl_utils import memory_usage, total_memory
            ram_used = memory_usage()
            ram_total = total_memory() if callable(total_memory) else None
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
        # GPU memory (if available)
        try:
            if hasattr(self, "device") and self.device and "cuda" in str(self.device):
                import torch
                if torch.cuda.is_available():
                    gpu_used = torch.cuda.memory_allocated(self.device)
                    gpu_total = torch.cuda.get_device_properties(self.device).total_memory
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
        prompt_lines.append("Choose action (see explanations):")
        for action in action_space:
            prompt_lines.append(f"  - {action}")
        prompt_lines.append(
            "If errors or resource usage are high, choose throttle, pause, or soft_shutdown. Only choose shutdown if instructed or if the system is unsafe."
        )
        return "\n".join(prompt_lines)

    def llm_decide(self, prompt: str) -> str:
        """
        Query the LLM with the prompt and parse the response into an action string.
        Args:
            prompt: The structured prompt string.
        Returns:
            Action string chosen by the LLM.
        """
        # Log the prompt and LLM output for audit trail
        self.logger.record_event(
            event_type="llm_prompt_sent",
            message="Prompt sent to LLM",
            level="debug",
            additional_info={"prompt": prompt, "timestamp": time.time()}
        )
        # TODO: Integrate with local LLM inference engine
        # Example stub:
        llm_output = "continue"  # Replace with actual LLM call and output
        self.logger.record_event(
            event_type="llm_decision_output",
            message="LLM decision output",
            level="debug",
            additional_info={"llm_output": llm_output, "timestamp": time.time()}
        )
        # Fallback: if output is invalid, default to continue
        valid_actions = {"continue", "throttle", "pause", "soft_shutdown", "shutdown"}
        if llm_output not in valid_actions:
            self.logger.record_event(
                event_type="llm_decision_invalid",
                message=f"Invalid LLM action '{llm_output}', falling back to 'continue'",
                level="warning",
                additional_info={"timestamp": time.time()}
            )
            return "continue"
        return llm_output

    def execute_action(self, action: str) -> None:
        """
        Map the chosen action string to actuator commands or system behaviors.
        Args:
            action: Action string from LLM.
        """
        # Soft shutdown confirmation logic
        if action == "soft_shutdown":
            self.pending_shutdown = True
            self.logger.record_event(
                event_type="soft_shutdown_initiated",
                message="System entering safe idle mode. Awaiting confirmation for full shutdown.",
                level="critical",
                additional_info={"timestamp": time.time()}
            )
            # Enter idle loop or minimal operation mode here
            return
        if action == "shutdown":
            if getattr(self, "pending_shutdown", False):
                self.logger.record_event(
                    event_type="system_shutdown_confirmed",
                    message="System shutdown confirmed. Powering down.",
                    level="critical",
                    additional_info={"timestamp": time.time()}
                )
                raise SystemExit("Shutdown triggered by LLM decision.")
            else:
                self.logger.record_event(
                    event_type="shutdown_without_confirmation",
                    message="Shutdown requested without prior soft shutdown. Ignoring for safety.",
                    level="warning",
                    additional_info={"timestamp": time.time()}
                )
                return
        if action == "throttle":
            prev = getattr(self, "throttle_level", 0)
            self.throttle_level = min(prev + 1, 5)
            self.logger.record_event(
                event_type="throttle_engaged",
                message=f"Throttling system. New throttle level: {self.throttle_level}",
                level="warning",
                additional_info={"timestamp": time.time()}
            )
            time.sleep(self.throttle_level)
            return
        if action == "pause":
            self.logger.record_event(
                event_type="system_paused",
                message="System paused due to error state.",
                level="warning",
                additional_info={"timestamp": time.time()}
            )
            time.sleep(10)
            return
        if action == "continue":
            self.throttle_level = 0
            self.logger.record_event(
                event_type="system_continue",
                message="System continuing normal operation.",
                level="info",
                additional_info={"timestamp": time.time()}
            )
            return
        # Default: try action registry as before
        if hasattr(self, "action_registry") and action in self.action_registry:
            try:
                self.action_registry[action]()
                self.logger.record_event(
                    event_type="action_executed",
                    message=f"Executed action '{action}'",
                    level="info",
                    additional_info={"timestamp": time.time()}
                )
            except Exception as e:
                self.logger.record_event(
                    event_type="action_execution_failed",
                    message=f"Failed to execute action '{action}': {str(e)}",
                    level="error",
                    additional_info={"timestamp": time.time()}
                )
        else:
            self.logger.record_event(
                event_type="action_unknown",
                message=f"Unknown or unregistered action '{action}'",
                level="warning",
                additional_info={"timestamp": time.time()}
            )

    def check_and_act(self) -> None:
        """
        Main loop to aggregate context, prompt the LLM, and execute the chosen action.
        """
        if not self.autonomy_config["enable_autonomy"]:
            return
        if not getattr(self, "enable_autosys_control", False):
            # Autosys control is off, skip LLM-based system self-regulation
            return
        try:
            with self.memory_lock:  # Ensure thread safety
                # 1. Aggregate sensory input and system state
                context = self.aggregate_sensory_context()
                # 2. Build a structured prompt for the LLM
                prompt = self.build_structured_prompt(context)
                # 3. Query the LLM for the next action
                action = self.llm_decide(prompt)
                # 4. Execute the chosen action
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
