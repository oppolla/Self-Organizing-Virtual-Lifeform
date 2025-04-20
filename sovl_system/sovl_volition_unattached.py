import torch
from typing import Dict, List, Optional, Any
import time
from threading import Lock
from collections import deque
import json
from sovl_utils import memory_usage, log_memory_usage
from sovl_logger import Logger

class AutonomyManager:
    """
    A lightweight decision-making framework for autonomous system optimization in the SOVL System.
    Processes system metrics and uses LLM-based reasoning to make decisions, initially for memory health.
    """
    def __init__(self, config_manager, logger: Logger, device: torch.device, system_ref):
        """
        Initialize the AutonomyManager.

        Args:
            config_manager: ConfigManager instance for accessing configuration.
            logger: Logger instance for recording events and errors.
            device: Torch device (cuda/cpu) for tensor operations.
            system_ref: Reference to SOVLSystem instance for triggering actions.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.system_ref = system_ref
        self.memory_lock = Lock()
        
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
            "fallback_decision_limit": 3  # New: Fallback after repeated LLM failures
        })
        
        # State tracking
        self.decision_history = deque(maxlen=self.autonomy_config["max_history_len"])
        self.error_counts = deque(maxlen=self.autonomy_config["hysteresis_window"])
        self.last_decision_time = 0.0
        self.diagnostic_last_run = 0.0
        self.context_memory = deque(maxlen=self.autonomy_config["context_window"])
        self.consecutive_llm_failures = 0
        self.start_time = time.time()  # New: Track system uptime
        
        self.logger.record({
            "event": "autonomy_manager_initialized",
            "config": {k: v for k, v in self.autonomy_config.items() if k != "prompt_template"},  # Compact logging
            "timestamp": time.time()
        })

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
            if torch.cuda.is_available():
                mem_stats = memory_usage(self.device)
                if mem_stats and mem_stats.get('allocated') is not None:
                    current_mem = mem_stats['allocated'] * (1024 ** 3)
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    metrics["memory_usage"] = current_mem / total_mem if total_mem > 0 else 0.0
                else:
                    self.logger.record({
                        "warning": "Memory stats unavailable, using fallback value",
                        "timestamp": time.time()
                    })
            
            # Error rate with smoothing
            recent_logs = self.logger.read(limit=max(1, self.autonomy_config["hysteresis_window"]))
            error_count = sum(1 for log in recent_logs if "error" in log.get("event", "").lower())
            metrics["error_rate"] = error_count / max(1, len(recent_logs))
            
            # Stability score
            metrics["stability_score"] = max(0.0, min(1.0, 1.0 - (0.7 * metrics["memory_usage"] + 0.3 * metrics["error_rate"])))
            
            self.validate_metrics(metrics)
            self.logger.record({
                "event": "metrics_collected",
                "metrics": {k: round(v, 4) for k, v in metrics.items()},  # Compact logging
                "timestamp": time.time()
            })
            return metrics
        except Exception as e:
            self.logger.record({
                "error": f"Failed to collect metrics: {str(e)}",
                "timestamp": time.time()
            })
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
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric '{name}' has invalid type: {type(value)}")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Metric '{name}' out of range: {value}")
        return True

    def build_prompt(self, metrics: Dict[str, float]) -> str:
        """
        Build a prompt for the LLM based on metrics.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            Formatted prompt string.
        """
        prompt = self.autonomy_config["prompt_template"].format(
            memory_usage=metrics["memory_usage"],
            error_rate=metrics["error_rate"],
            stability_score=metrics["stability_score"]
        )
        return prompt[:self.autonomy_config["max_prompt_len"]]  # Truncate if too long

    def make_decision(self, prompt: str) -> Optional[bool]:
        """
        Use the LLM to make a decision based on the prompt, with fallback for repeated failures.

        Args:
            prompt: Formatted prompt with metrics.

        Returns:
            True if adjustments are needed, False if not, None if decision fails.
        """
        try:
            if not self.autonomy_config["enable_autonomy"]:
                return False
                
            current_time = time.time()
            if current_time - self.last_decision_time < self.autonomy_config["decision_cooldown"]:
                self.logger.record({
                    "event": "decision_skipped",
                    "reason": "cooldown",
                    "remaining": round(self.autonomy_config["decision_cooldown"] - (current_time - self.last_decision_time), 2),
                    "timestamp": current_time
                })
                return None
                
            # Fallback if LLM fails repeatedly
            if self.consecutive_llm_failures >= self.autonomy_config["fallback_decision_limit"]:
                metrics = self.collect_metrics()
                decision = metrics["memory_usage"] > self.autonomy_config["memory_threshold"] or \
                          metrics["error_rate"] > self.autonomy_config["error_rate_threshold"]
                self.logger.record({
                    "event": "decision_fallback",
                    "decision": decision,
                    "reason": "llm_failure_limit_reached",
                    "metrics": metrics,
                    "timestamp": current_time
                })
                self.consecutive_llm_failures = 0
                self.last_decision_time = current_time
                return decision
                
            # Generate decision
            response = self.system_ref.generate(
                prompt,
                max_new_tokens=10,
                temperature=self.controls_config.get("base_temperature", 0.7),
                top_k=self.controls_config.get("top_k", 30),
                do_sample=False
            ).strip().lower()
            
            if response not in ["true", "false"]:
                self.consecutive_llm_failures += 1
                self.logger.record({
                    "warning": f"Invalid LLM response: '{response}'",
                    "prompt": prompt[:100],  # Truncate for logging
                    "timestamp": current_time
                })
                return None
                
            self.consecutive_llm_failures = 0
            decision = response == "true"
            self.last_decision_time = current_time
            self.decision_history.append({
                "decision": decision,
                "metrics": {k: round(v, 4) for k in self.collect_metrics()},  # Compact storage
                "timestamp": current_time
            })
            
            self.logger.record({
                "event": "decision_made",
                "decision": decision,
                "response": response,
                "timestamp": current_time
            })
            return decision
            
        except Exception as e:
            self.consecutive_llm_failures += 1
            self.logger.record({
                "error": f"Decision-making failed: {str(e)}",
                "prompt": prompt[:100],
                "timestamp": time.time()
            })
            return None

    def execute_action(self, decision: bool) -> bool:
        """
        Execute system actions based on the decision with rollback on failure.

        Args:
            decision: True if adjustments are needed, False otherwise.

        Returns:
            True if actions were executed successfully, False otherwise.
        """
        if not decision:
            self.logger.record({
                "event": "action_skipped",
                "reason": "no_adjustments_needed",
                "timestamp": time.time()
            })
            return False
            
        start_time = time.time()
        try:
            with self.memory_lock:
                success = False
                rollback_state = {}  # Store state for rollback
                
                # Quantization action
                if self.system_ref.quantization_mode != "int8":
                    rollback_state["quantization_mode"] = self.system_ref.quantization_mode
                    self.system_ref.set_quantization_mode("int8")
                    success = True
                    self.logger.record({
                        "event": "action_executed",
                        "action": "set_quantization_int8",
                        "timestamp": time.time()
                    })
                
                # Batch size action
                current_batch_size = self.system_ref.training_config.get("batch_size", 1)
                if current_batch_size > 1:
                    rollback_state["batch_size"] = current_batch_size
                    new_batch_size = max(1, current_batch_size // 2)
                    self.system_ref.config_manager.update("training_config.batch_size", new_batch_size)
                    self.system_ref.training_config["batch_size"] = new_batch_size
                    self.system_ref.trainer.config.batch_size = new_batch_size
                    success = True
                    self.logger.record({
                        "event": "action_executed",
                        "action": "reduce_batch_size",
                        "new_batch_size": new_batch_size,
                        "timestamp": time.time()
                    })
                
                # CUDA cache action
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    success = True
                    self.logger.record({
                        "event": "action_executed",
                        "action": "clear_cuda_cache",
                        "timestamp": time.time()
                    })
                
                # Dream memory pruning
                if len(self.system_ref.state.dream_memory) > 0:
                    with self.system_ref.state.memory_lock:
                        original_len = len(self.system_ref.state.dream_memory)
                        sorted_mem = sorted(self.system_ref.state.dream_memory, key=lambda x: x[1], reverse=True)
                        keep_len = max(1, original_len // 2)
                        rollback_state["dream_memory"] = list(self.system_ref.state.dream_memory)
                        self.system_ref.state.dream_memory = deque(maxlen=self.system_ref.state.dream_memory_maxlen)
                        for tensor, weight in sorted_mem[:keep_len]:
                            if weight > 0.5:
                                self.system_ref.state.append_dream_memory(tensor.detach().cpu(), weight)
                        if len(self.system_ref.state.dream_memory) < original_len:
                            success = True
                            self.logger.record({
                                "event": "action_executed",
                                "action": "prune_dream_memory",
                                "new_length": len(self.system_ref.state.dream_memory),
                                "timestamp": time.time()
                            })
                
                # Timeout check
                if time.time() - start_time > self.autonomy_config["action_timeout"]:
                    raise TimeoutError("Action execution exceeded timeout")
                
                return success
                
        except Exception as e:
            self.logger.record({
                "error": f"Action execution failed: {str(e)}",
                "timestamp": time.time()
            })
            # Attempt rollback
            try:
                if "quantization_mode" in rollback_state:
                    self.system_ref.set_quantization_mode(rollback_state["quantization_mode"])
                if "batch_size" in rollback_state:
                    self.system_ref.config_manager.update("training_config.batch_size", rollback_state["batch_size"])
                    self.system_ref.training_config["batch_size"] = rollback_state["batch_size"]
                    self.system_ref.trainer.config.batch_size = rollback_state["batch_size"]
                if "dream_memory" in rollback_state:
                    with self.system_ref.state.memory_lock:
                        self.system_ref.state.dream_memory = deque(rollback_state["dream_memory"], maxlen=self.system_ref.state.dream_memory_maxlen)
                self.logger.record({
                    "event": "rollback_executed",
                    "rollback_state": rollback_state,
                    "timestamp": time.time()
                })
            except Exception as re:
                self.logger.record({
                    "error": f"Rollback failed: {str(re)}",
                    "timestamp": time.time()
                })
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
                return {"status": "skipped", "reason": "interval_not_reached"}
                
            diagnostics = {
                "status": "success",
                "timestamp": current_time,
                "memory_usage": 0.0,
                "token_consumption": 0.0,
                "response_latency": 0.0,
                "uptime": current_time - self.start_time,  # New: System uptime
                "anomalies": []
            }
            
            # Memory usage
            metrics = self.collect_metrics()
            diagnostics["memory_usage"] = metrics["memory_usage"]
            if metrics["memory_usage"] > self.autonomy_config["memory_threshold"]:
                diagnostics["anomalies"].append("high_memory_usage")
            
            # Token consumption with cap
            recent_logs = self.logger.read(limit=self.autonomy_config["context_window"])
            token_counts = [
                min(1000, len(self.system_ref.base_tokenizer.encode(log.get("response", ""))))  # Cap at 1000 tokens
                for log in recent_logs if "response" in log
            ]
            diagnostics["token_consumption"] = sum(token_counts) / max(1, len(token_counts)) if token_counts else 0.0
            
            # Response latency
            latencies = [
                log.get("generation_params", {}).get("timestamp", 0.0) - log.get("timestamp", 0.0)
                for log in recent_logs if "generation_params" in log
            ]
            diagnostics["response_latency"] = sum(latencies) / max(1, len(latencies)) if latencies else 0.0
            
            # Anomalies
            if diagnostics["response_latency"] > 5.0:
                diagnostics["anomalies"].append("high_response_latency")
            if diagnostics["token_consumption"] > 500:
                diagnostics["anomalies"].append("high_token_consumption")
                
            self.diagnostic_last_run = current_time
            self.logger.record({
                "event": "self_diagnostic",
                "diagnostics": {k: round(v, 4) if isinstance(v, float) else v for k, v in diagnostics.items()},
                "timestamp": current_time
            })
            
            return diagnostics
            
        except Exception as e:
            self.logger.record({
                "error": f"Self-diagnostic failed: {str(e)}",
                "timestamp": time.time()
            })
            return {"status": "failed", "error": str(e)}

    def update_context(self, prompt: str, response: str):
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
            self.logger.record({
                "event": "context_updated",
                "context_length": len(self.context_memory),
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Context update failed: {str(e)}",
                "timestamp": time.time()
            })
            self.reset_context()

    def reset_context(self):
        """
        Reset contextual memory if corrupted or on error.
        """
        self.context_memory = deque(maxlen=self.autonomy_config["context_window"])
        self.logger.record({
            "event": "context_reset",
            "timestamp": time.time()
        })

    def check_and_act(self):
        """
        Main loop to check metrics and make autonomous decisions.
        """
        if not self.autonomy_config["enable_autonomy"]:
            return
            
        try:
            with self.memory_lock:  # Ensure thread safety
                diagnostics = self.run_self_diagnostic()
                if diagnostics["status"] == "failed":
                    self.logger.record({
                        "warning": "Skipping autonomy check due to diagnostic failure",
                        "timestamp": time.time()
                    })
                    return
                    
                metrics = self.collect_metrics()
                
                self.error_counts.append(metrics["error_rate"])
                avg_error_rate = sum(self.error_counts) / len(self.error_counts) if self.error_counts else 0.0
                if metrics["memory_usage"] < self.autonomy_config["memory_threshold"] and \
                   avg_error_rate < self.autonomy_config["error_rate_threshold"]:
                    self.logger.record({
                        "event": "autonomy_check",
                        "status": "stable",
                        "metrics": {k: round(v, 4) for k in metrics},
                        "timestamp": time.time()
                    })
                    return
                    
                prompt = self.build_prompt(metrics)
                decision = self.make_decision(prompt)
                
                if decision is None:
                    self.logger.record({
                        "warning": "Decision-making returned None, skipping actions",
                        "timestamp": time.time()
                    })
                    return
                    
                success = self.execute_action(decision)
                
                self.logger.record({
                    "event": "autonomy_cycle_complete",
                    "decision": decision,
                    "actions_executed": success,
                    "metrics": {k: round(v, 4) for k in metrics},
                    "diagnostics": {k: round(v, 4) if isinstance(v, float) else v for k, v in diagnostics.items()},
                    "timestamp": time.time()
                })
                
        except Exception as e:
            self.logger.record({
                "error": f"Autonomy check failed: {str(e)}",
                "timestamp": time.time()
            })
