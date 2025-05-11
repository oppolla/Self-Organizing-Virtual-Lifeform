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
import concurrent.futures
import re
import threading
import numpy as np

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

class NoOpTuner:
    def update_parameters(self, params):
        pass
    def get_current_parameters(self):
        return {}

class NoOpMotivator:
    def consider_goal(self, description, source, context):
        print(f"[NoOpMotivator] Goal not created: {description} (no motivator present)")

class AutonomyManager:
    """
    A lightweight decision-making framework for autonomous system optimization in the SOVL System.
    All dependencies are injected explicitly to avoid legacy system_ref patterns.
    """
    def __init__(
        self,
        config_manager,
        logger: Logger,
        device: torch.device,
        generation_manager=None,
        state_manager=None,
        ram_manager=None,
        gpu_manager=None,
        tuner: Optional[SOVLTuner] = None,
        motivator=None,
        get_resource_stats=None,
        pause=None,
        shutdown=None,
        soft_shutdown=None,
        get_battery_level=None,
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.generation_manager = generation_manager
        self.state_manager = state_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self.tuner = tuner if tuner is not None else NoOpTuner()
        self.get_resource_stats = get_resource_stats
        self.pause = pause
        self.shutdown = shutdown
        self.soft_shutdown = soft_shutdown
        self.get_battery_level = get_battery_level
        self.motivator = motivator if motivator is not None else Motivator(
            curiosity=None,
            memory=None,
            logger=self.logger,
            config_manager=self.config_manager,
            bonder=None,
            state_manager=self.state_manager
        )
        self.context_lock = Lock()
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
            "enable_autosys_control": False,  # New: Automated system control (LLM-driven throttle/pause/shutdown)
            "sense_node_cache_ttl": 1.0,  # seconds
            "sense_node_priority": {}  # name -> priority (lower is higher priority)
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
        self.resource_actions = {}  # Modular resource action registry
        self.pending_shutdown = False
        
        # Sense nodes registry: name -> SensationNode
        self.sense_nodes: dict = {}

        # --- LLM Decision Integration: Attach GenerationManager ---
        # Attempt to get a generation_manager from system_ref, else raise error on first use
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

        self._sense_cache = {}  # name -> (observation, timestamp)
        self._sense_cache_ttl = self.config_manager.get("sense_node_cache_ttl", 1.0)  # seconds
        self._sense_node_priority = self.config_manager.get("sense_node_priority", {})  # name -> priority (lower is higher priority)

    def register_resource_action(self, name, func):
        self.resource_actions[name] = func

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
                # Use centralized resource stats
                resource_stats = self.get_resource_stats() if self.get_resource_stats else {}
                current_mem = resource_stats.get('ram_used', 0.0)
                total_mem = resource_stats.get('ram_total', 0.0)
                metrics["memory_usage"] = current_mem / total_mem if total_mem and total_mem > 0 else 0.0
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
        Enhanced: robust output validation, failure tracking, and safer fallbacks.
        """
        self.logger.record_event(
            event_type="llm_prompt_sent",
            message="Prompt sent to LLM",
            level="debug",
            additional_info={"prompt": prompt, "timestamp": time.time(), "allowed_responses": allowed_responses}
        )
        llm_output = None
        action = None
        allowed_set = {resp.lower().strip(): resp for resp in allowed_responses}
        failure_threshold = 3

        try:
            if self.generation_manager is None:
                self.logger.record_event(
                    event_type="generation_manager_missing_critical",
                    message="No generation_manager attached to AutonomyManager. Fallback to safe action.",
                    level="critical",
                    additional_info={"prompt": prompt, "timestamp": time.time()}
                )
                self.consecutive_llm_failures += 1
                return self._safe_fallback_action(allowed_responses)
            result = self.generation_manager.generate_text(prompt, num_return_sequences=1, user_id="autonomy_manager")
            if isinstance(result, list) and result:
                llm_output = result[0]
            else:
                llm_output = ""
        except Exception as e:
            self.logger.record_event(
                event_type="llm_decision_error",
                message=f"LLM decision error: {str(e)}",
                level="error",
                additional_info={"prompt": prompt, "timestamp": time.time()}
            )
            self.consecutive_llm_failures += 1
            return self._safe_fallback_action(allowed_responses)

        # Robust extraction: try to find a valid action in the output
        extracted = None
        if isinstance(llm_output, str):
            # Try to extract a valid action (case-insensitive, word-boundary)
            for resp in allowed_responses:
                pattern = r'\\b' + re.escape(resp) + r'\\b'
                if re.search(pattern, llm_output, re.IGNORECASE):
                    extracted = resp
                    break
            # Try to extract from JSON or quoted string
            if not extracted:
                m = re.search(r'"([a-zA-Z_]+)"', llm_output)
                if m and m.group(1).lower() in allowed_set:
                    extracted = allowed_set[m.group(1).lower()]
            # Fallback: try to match any allowed response as substring
            if not extracted:
                for resp in allowed_responses:
                    if resp.lower() in llm_output.lower():
                        extracted = resp
                        break
        else:
            llm_output = str(llm_output)

        if extracted and extracted.lower() in allowed_set:
            action = allowed_set[extracted.lower()]
            self.consecutive_llm_failures = 0
        else:
            self.consecutive_llm_failures += 1
            self.logger.record_event(
                event_type="llm_decision_invalid",
                message=f"Invalid or unrecognized LLM action '{llm_output}'. Fallback triggered.",
                level="warning",
                additional_info={"timestamp": time.time(), "failures": self.consecutive_llm_failures}
            )
            if self.consecutive_llm_failures >= failure_threshold:
                self.logger.record_event(
                    event_type="llm_decision_fallback_mode",
                    message=f"Consecutive LLM failures ({self.consecutive_llm_failures}) exceeded threshold. Entering fallback mode.",
                    level="error",
                    additional_info={"timestamp": time.time()}
                )
                return self._safe_fallback_action(allowed_responses)
            else:
                return self._safe_fallback_action(allowed_responses)
        self.logger.record_event(
            event_type="llm_decision_output",
            message="LLM decision output",
            level="debug",
            additional_info={"llm_output": llm_output, "action": action, "timestamp": time.time()}
        )
        return action

    def _safe_fallback_action(self, allowed_responses):
        # Prefer throttle or pause if available, else first allowed
        for safer in ["throttle", "pause"]:
            for resp in allowed_responses:
                if resp.lower() == safer:
                    return resp
        return allowed_responses[0]

    def execute_action(self, action: str) -> None:
        """
        Execute system actions based on the string action name, with robust curiosity/explore fallback.
        Args:
            action: Action string (e.g., 'continue', 'throttle', 'pause', 'explore', etc.)
        """
        if action == "continue":
            return  # No-op for normal operation
        elif action == "throttle":
            if isinstance(self.tuner, NoOpTuner):
                self.logger.record_event(
                    event_type="tuner_missing",
                    message="Tuner is required for 'throttle' action but is missing. No parameters updated.",
                    level="warning"
                )
            else:
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
            if self.pause:
                self.pause()
                self.logger.record_event(event_type="system_paused", message="System paused by LLM decision.")
        elif action == "soft_shutdown":
            if self.soft_shutdown:
                self.soft_shutdown()
                self.logger.record_event(event_type="soft_shutdown", message="System entered soft shutdown mode.")
        elif action == "shutdown":
            if self.shutdown:
                self.shutdown()
                self.logger.record_event(event_type="shutdown", message="System shutdown initiated by LLM.")
        elif action == "explore":
            if isinstance(self.motivator, NoOpMotivator):
                self.logger.record_event(
                    event_type="motivator_missing",
                    message="Motivator is required for 'explore' action but is missing. No goal created.",
                    level="warning"
                )
            else:
                if hasattr(self, "curiosity") and self.curiosity:
                    try:
                        question = None
                        if hasattr(self.curiosity, "generate_curiosity_question"):
                            question = self.curiosity.generate_curiosity_question(context="explore", spontaneous=True)
                        self.logger.record_event(
                            event_type="curiosity_question",
                            message=f"Curiosity-driven question: {question}",
                            additional_info={"timestamp": time.time()}
                        )
                        if question:
                            self.motivator.consider_goal(description=question, source="curiosity", context={})
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
        Uses timeout, error isolation, caching, and prioritization.
        """
        context = {}
        now = time.time()
        # Prioritize nodes
        nodes = list(self.sense_nodes.items())
        nodes.sort(key=lambda item: self._sense_node_priority.get(item[0], 100))  # Default low priority

        def get_obs(name, node):
            # Caching
            cached = self._sense_cache.get(name)
            if cached and now - cached[1] < self._sense_cache_ttl:
                return cached[0]
            # Timeout and error isolation
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(node.get_latest_observation)
                try:
                    obs = future.result(timeout=1.0)
                    self._sense_cache[name] = (obs, now)
                    return obs
                except Exception as e:
                    self.logger.record_event(
                        event_type="sense_observation_failed",
                        message=f"Failed or timed out getting observation from '{name}': {str(e)}",
                        level="warning",
                        additional_info={"timestamp": now}
                    )
                    self._sense_cache[name] = (None, now)
                    return None

        for name, node in nodes:
            context[name] = get_obs(name, node)

        # Add system state (battery, error, etc.)
        if self.get_battery_level:
            try:
                context["battery"] = self.get_battery_level()
            except Exception:
                context["battery"] = None
        error_state = None
        if hasattr(self, 'error_counts') and self.error_counts:
            context["error_rate"] = sum(self.error_counts) / len(self.error_counts)
            context["error_count"] = len(self.error_counts)
        else:
            context["error_rate"] = 0.0
            context["error_count"] = 0
        context["throttle_level"] = getattr(self, "throttle_level", 0)
        # Add system resource metrics (RAM, GPU)
        try:
            resource_stats = self.get_resource_stats() if self.get_resource_stats else {}
            context["ram_usage"] = resource_stats.get('ram_used', None)
            context["ram_total"] = resource_stats.get('ram_total', None)
            context["ram_usage_pct"] = resource_stats['ram_used'] / resource_stats['ram_total'] if resource_stats.get('ram_used') is not None and resource_stats.get('ram_total') else None
            context["gpu_mem_used"] = resource_stats.get('gpu_used', None)
            context["gpu_mem_total"] = resource_stats.get('gpu_total', None)
            context["gpu_mem_used_pct"] = resource_stats['gpu_used'] / resource_stats['gpu_total'] if resource_stats.get('gpu_used') is not None and resource_stats.get('gpu_total') else None
        except Exception:
            context["ram_usage"] = None
            context["ram_total"] = None
            context["ram_usage_pct"] = None
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
        If generation_manager is missing, log a critical error and skip LLM-based actions.
        """
        if not self.autonomy_config["enable_autonomy"]:
            return
        if not getattr(self, "enable_autosys_control", False):
            return
        if self.generation_manager is None:
            self.logger.record_event(
                event_type="generation_manager_missing_critical",
                message="No generation_manager available in check_and_act. Skipping LLM-based decision.",
                level="critical",
                additional_info={"timestamp": time.time()}
            )
            return
        try:
            with self.context_lock:
                context = self.aggregate_sensory_context()
                if isinstance(self.tuner, NoOpTuner):
                    self.logger.record_event(
                        event_type="tuner_fallback",
                        message="Tuner is not configured. Using no-op fallback.",
                        level="warning"
                    )
                if isinstance(self.motivator, NoOpMotivator):
                    self.logger.record_event(
                        event_type="motivator_fallback",
                        message="Motivator is not configured. Using no-op fallback.",
                        level="warning"
                    )
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
    GOAL_SCHEMA_VERSION = 2

    def __init__(self, description, source, context, priority=1.0, status='active', metadata=None, completion=0.0, ephemeral=False, ttl=None, impact=1.0, soft_completed=False, last_checked=None, relevance=1.0, created_at=None, last_updated=None):
        self.description = description
        self.source = source
        self.context = context
        self.priority = priority
        self.status = status
        self.metadata = metadata or {}
        self.created_at = created_at if created_at is not None else time.time()
        self.last_updated = last_updated if last_updated is not None else self.created_at
        self.completion = completion
        self.ephemeral = ephemeral
        self.ttl = ttl  # seconds, or None
        self.impact = impact  # 0.0 (trivial) to 1.0 (critical)
        self.soft_completed = soft_completed
        self.last_checked = last_checked or time.time()
        self.relevance = relevance

    @property
    def expires_at(self):
        return self.created_at + self.ttl if self.ttl else None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.last_updated = time.time()

    def decay_priority(self, decay_rate, decay_multiplier, min_priority):
        decay = decay_rate * (decay_multiplier if self.ephemeral else 1.0)
        self.priority = max(self.priority - decay, min_priority)

    def to_dict(self):
        return {
            "version": self.GOAL_SCHEMA_VERSION,
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
            "ttl": self.ttl,
            "impact": self.impact,
            "soft_completed": self.soft_completed,
            "last_checked": self.last_checked,
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, data):
        try:
            version = data.get("version", 2)
            return cls(
                description=data.get("description", "unknown"),
                source=data.get("source", "unknown"),
                context=data.get("context", {}),
                priority=data.get("priority", 1.0),
                status=data.get("status", "active"),
                metadata=data.get("metadata", {}),
                completion=data.get("completion", 0.0),
                ephemeral=data.get("ephemeral", False),
                ttl=data.get("ttl", None),
                impact=data.get("impact", 1.0),
                soft_completed=data.get("soft_completed", False),
                last_checked=data.get("last_checked", time.time()),
                relevance=data.get("relevance", 1.0),
                created_at=data.get("created_at", time.time()),
                last_updated=data.get("last_updated", data.get("created_at", time.time())),
            )
        except Exception as e:
            import traceback
            print(f"Failed to load goal from dict: {e}\n{traceback.format_exc()}")
            return None

# --- Goal Validation Helper ---
def validate_goal(goal) -> bool:
    """Ensure goal meets system invariants"""
    checks = [
        bool(getattr(goal, 'description', None)),
        0 <= getattr(goal, 'priority', 0) <= 1,
        getattr(goal, 'ttl', None) is None or getattr(goal, 'ttl', 1) > 0,
        getattr(goal, 'impact', 0) >= 0
    ]
    return all(checks)

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
        state_manager,
        decay_rate=0.01,
        min_priority=0.1,
        completion_threshold=0.95,
        reevaluation_interval=60,
        memory_check_enabled=True,
        irrelevance_threshold=0.2,
        completion_drive=0.7,
        novelty_drive=0.2,
        max_goals=30,
    ):
        self.curiosity = curiosity
        self.memory = memory
        self.logger = logger
        self.config_manager = config_manager
        self.bonder = bonder
        self.state_manager = state_manager
        # Ephemeral config
        ephemeral_cfg = config_manager.get_section("motivator.ephemeral", {
            "novelty_threshold": 0.3,
            "impact_threshold": 0.4,
            "decay_rate": decay_rate,
            "decay_multiplier": 2.0,
            "stale_threshold": 300,
            "min_impact_to_keep": 0.2,
            "min_priority": min_priority,
        })
        self.ephemeral_cfg = ephemeral_cfg
        self.decay_rate = ephemeral_cfg["decay_rate"]
        self.min_priority = ephemeral_cfg["min_priority"]
        self.completion_threshold = completion_threshold
        self.reevaluation_interval = reevaluation_interval
        self.memory_check_enabled = memory_check_enabled
        self.irrelevance_threshold = irrelevance_threshold
        self.completion_drive = completion_drive
        self.novelty_drive = novelty_drive
        self.max_goals = max_goals
        self.goals = []  # List[Goal]
        self._goal_lock = threading.Lock()
        self.ttl_warning_threshold = self.ephemeral_cfg.get("ttl_warning_threshold", 60)  # seconds
        self.load_goals_from_memory()

    def _is_ephemeral(self, source: str, context: dict) -> bool:
        novelty = context.get("novelty", 0)
        impact = context.get("impact", 1.0)
        expires_at = context.get("expires_at")
        if source == "curiosity" and novelty < self.ephemeral_cfg["novelty_threshold"]:
            return True
        if impact < self.ephemeral_cfg["impact_threshold"]:
            return True
        if expires_at is not None:
            return True
        return False

    def load_goals_from_memory(self):
        """Load goals from state_manager atomically and safely."""
        with self._goal_lock:
            try:
                def get_goals_fn(state):
                    return state.get_goals() if hasattr(state, 'get_goals') else []
                goal_dicts = self.state_manager.read_state_atomic(get_goals_fn)
                loaded_goals = []
                for g in goal_dicts:
                    goal = Goal.from_dict(g)
                    if goal is not None and validate_goal(goal):
                        loaded_goals.append(goal)
                    else:
                        self.logger.record_event(
                            event_type="goal_load_skipped",
                            message="Skipped invalid goal during load.",
                            level="warning"
                        )
                if loaded_goals:
                    self.goals = loaded_goals
                self.logger.record_event(
                    event_type="goals_loaded",
                    message=f"Loaded {len(self.goals)} goals from state.",
                    level="info"
                )
            except Exception as e:
                self.logger.record_event(
                    event_type="goal_load_failed",
                    message=f"Failed to load goals from state_manager: {str(e)}",
                    level="error",
                    stack_trace=traceback.format_exc()
                )
                # Do not overwrite self.goals with []; preserve last known good state

    def persist_goals(self):
        """Persist goals to state_manager atomically with validation."""
        with self._goal_lock:
            def update_fn(state):
                try:
                    goals_to_save = [g.to_dict() for g in self.goals]
                    # Validate goals before saving
                    for g in goals_to_save:
                        if not isinstance(g, dict) or "description" not in g:
                            raise ValueError(f"Invalid goal format: {g}")
                    state.set_goals(goals_to_save)
                    return state
                except Exception as e:
                    self.logger.record_event(
                        event_type="goal_persist_failed",
                        message=f"Failed to persist goals: {str(e)}",
                        level="error",
                        stack_trace=traceback.format_exc()
                    )
                    raise
            try:
                self.state_manager.update_state_atomic(update_fn)
                self.logger.record_event(
                    event_type="goals_persisted",
                    message=f"Persisted {len(self.goals)} goals to state.",
                    level="info"
                )
            except Exception as e:
                self.logger.record_event(
                    event_type="goal_persist_atomic_failed",
                    message=f"Atomic persist failed: {str(e)}",
                    level="error",
                    stack_trace=traceback.format_exc()
                )
                # Optionally: Retry or mark for recovery

    def consider_goal(self, description, source, context, force_persistent=False):
        with self._goal_lock:
            # Enforce max_goals limit
            active_goals = [g for g in self.goals if g.status == 'active']
            if len(active_goals) >= self.max_goals:
                self.logger.record_event(
                    event_type="goal_rejected",
                    message=f"Goal limit reached ({self.max_goals}). Goal not added: {description}",
                    additional_info={"source": source, "context": context}
                )
                return  # Do not add new goal

            # --- Conflict Detection ---
            conflict = self._detect_goal_conflict(description, source, context, active_goals)
            if conflict:
                self.logger.record_event(
                    event_type="goal_conflict",
                    message=f"Goal '{description}' conflicts with existing goal '{conflict.description}'. Rejected.",
                    additional_info={"new_goal": description, "conflict_with": conflict.description, "context": context}
                )
                return  # Reject conflicting goal

            bond = self.bonder.get_bond(context.get('user_id')) if source == 'user' and context.get('user_id') else 1.0
            priority = self.estimate_priority(description, source, context, bond)
            ephemeral = self._is_ephemeral(source, context) and not force_persistent
            ttl = context.get("ttl")
            impact = context.get("impact", 1.0)
            goal = Goal(description, source, context, priority, ephemeral=ephemeral, ttl=ttl, impact=impact)
            if not validate_goal(goal):
                self.logger.record_event(
                    event_type="goal_validation_failed",
                    message=f"Goal failed validation and was not added: {description}",
                    additional_info={"source": source, "context": context, "priority": priority, "ephemeral": ephemeral, "ttl": ttl, "impact": impact}
                )
                return
            self.goals.append(goal)
            self._prioritize_goals()
            self.logger.record_event(event_type="goal_created", message=description, additional_info={"priority": priority, "source": source, "context": context, "ephemeral": ephemeral, "ttl": ttl, "impact": impact})
            self.persist_goals()

    def _detect_goal_conflict(self, description, source, context, active_goals):
        """
        Returns the first conflicting goal if found, else None.
        Conflict logic: same context and mutually exclusive actions (customize as needed).
        """
        new_action = context.get('action')
        new_ctx = context.get('location') or context.get('target')
        for goal in active_goals:
            existing_action = goal.context.get('action')
            existing_ctx = goal.context.get('location') or goal.context.get('target')
            # Example: mutually exclusive if same context/location and different actions
            if new_ctx and existing_ctx and new_ctx == existing_ctx and new_action and existing_action and new_action != existing_action:
                return goal
        return None

    def _prioritize_goals(self):
        """
        Sorts self.goals by priority, relevance, and compatibility.
        Keeps only the top N active goals (configurable).
        """
        n = self.config_manager.get('goal_queue_size', self.max_goals)
        # Only keep top N active goals, drop or deprioritize the rest
        active_goals = [g for g in self.goals if g.status == 'active']
        sorted_goals = sorted(active_goals, key=lambda g: (g.priority, g.relevance), reverse=True)
        for i, goal in enumerate(sorted_goals):
            if i >= n:
                goal.status = 'dropped'
                self.logger.record_event(
                    event_type="goal_dropped_low_priority",
                    message=f"Goal '{goal.description}' dropped due to low priority in queue.",
                    additional_info={"priority": goal.priority, "relevance": goal.relevance}
                )
        # Rebuild self.goals with updated statuses
        self.goals = sorted(self.goals, key=lambda g: (g.status != 'active', -g.priority, -g.relevance))

    def estimate_priority(self, description, source, context, bond):
        # Use heuristics, bond, curiosity, context, etc. to estimate priority (future: temperament, novelty, etc.)
        base_priority = 1.0
        if source == 'user':
            base_priority *= bond
        # Add more dynamic influences here (curiosity, novelty, etc.)
        return base_priority

    def reevaluate_goals(self):
        with self._goal_lock:
            self.check_ttl_warnings()
            now = time.time()
            to_remove = []
            for goal in list(self.goals):
                goal.last_checked = now
                # TTL enforcement
                if goal.ephemeral and goal.ttl and (now > goal.created_at + goal.ttl):
                    to_remove.append((goal, "ttl_expired"))
                # Staleness
                elif goal.ephemeral and (now - goal.last_updated) > self.ephemeral_cfg["stale_threshold"]:
                    to_remove.append((goal, "stale"))
                # Impact-based cleanup
                elif goal.ephemeral and goal.impact < self.ephemeral_cfg["min_impact_to_keep"]:
                    to_remove.append((goal, "low_impact"))
                # Priority-based cleanup
                elif goal.ephemeral and goal.priority <= self.ephemeral_cfg["min_priority"]:
                    to_remove.append((goal, "priority_depleted"))
                # Completion
                elif goal.completion >= self.completion_threshold and goal.status == 'active':
                    goal.status = 'completed'
                    goal.soft_completed = True
                    self.logger.record_event("goal_soft_completed", message=goal.description)
                # Irrelevance
                elif goal.relevance < self.irrelevance_threshold or (goal.ephemeral and goal.status == 'stale'):
                    goal.status = 'dropped'
                    self.logger.record_event("goal_dropped_irrelevant", message=goal.description)
                else:
                    goal.decay_priority(self.decay_rate, self.ephemeral_cfg["decay_multiplier"], self.min_priority)
            for goal, reason in to_remove:
                self.drop_goal(goal, reason)
            self.goals = [g for g in self.goals if g.status not in ('dropped', 'completed') or g.soft_completed]
            self.persist_goals()

    def update_goal_progress(self, goal, progress_delta):
        with self._goal_lock:
            goal.completion = min(max(goal.completion + progress_delta, 0.0), 1.0)
            if goal.completion >= 1.0:
                self.complete_goal(goal, soft=True)
            else:
                self.persist_goals()

    def complete_goal(self, goal, soft=False):
        with self._goal_lock:
            goal.status = 'completed'
            goal.priority = 0
            goal.completion = 1.0
            goal.soft_completed = soft
            goal.metadata['completed_at'] = time.time()
            self.logger.record_event(event_type="goal_completed", message=goal.description)
            self.persist_goals()

    def drop_goal(self, goal, reason="decayed"):
        with self._goal_lock:
            goal.status = 'dropped'
            goal.priority = 0
            goal.metadata['dropped_at'] = time.time()
            self.logger.record_event(event_type="goal_dropped", message=goal.description, additional_info={"reason": reason})
            self.persist_goals()

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

    def get_ephemeral_stats(self):
        """Return stats on ephemeral goals (count, avg_lifespan, completed, dropped)."""
        ephemerals = [g for g in self.goals if getattr(g, 'ephemeral', False)]
        now = time.time()
        lifespans = [now - g.created_at for g in ephemerals]
        return {
            'count': len(ephemerals),
            'avg_lifespan': float(np.mean(lifespans)) if lifespans else 0.0,
            'completed': sum(1 for g in ephemerals if getattr(g, 'status', None) == 'completed'),
            'dropped': sum(1 for g in ephemerals if getattr(g, 'status', None) == 'dropped')
        }

    def check_ttl_warnings(self):
        """Log warnings for soon-to-expire ephemeral goals."""
        now = time.time()
        for goal in self.goals:
            if getattr(goal, 'ephemeral', False) and getattr(goal, 'ttl', None):
                expires_at = getattr(goal, 'created_at', 0) + getattr(goal, 'ttl', 0)
                remaining = expires_at - now
                if 0 < remaining < self.ttl_warning_threshold:
                    self.logger.record_event(
                        event_type="ephemeral_goal_expiring_soon",
                        message=f"Ephemeral goal expiring soon: {getattr(goal, 'description', '')}",
                        additional_info={"remaining_seconds": remaining}
                    )
