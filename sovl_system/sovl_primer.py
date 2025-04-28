from typing import Optional, Any, Dict
from sovl_curiosity import Curiosity, CuriosityPressure, CuriosityCallbacks
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_confidence import ConfidenceCalculator, calculate_confidence_score
from sovl_bonder import BondCalculator, BondModulator
from sovl_logger import Logger 
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_state import SOVLState  # Import for type clarity and direct integration
import traceback
import threading

class GenerationPrimer:
    """
    GenerationPrimer is the unified integration point for all trait modules, providing trait aggregation,
    parameter adjustment, dynamic trait toggling for generation.

    """
    def __init__(
        self,
        config_manager: Any,
        logger: Logger,
        state: SOVLState,
        error_manager: ErrorManager,
        curiosity_manager: Optional[Any] = None,
        temperament_system: Optional[Any] = None,
        confidence_calculator: Optional[Any] = None,
        bond_calculator: Optional[Any] = None,
        device: Optional[Any] = None,
        lifecycle_manager: Optional[Any] = None,
        scaffold_manager: Optional[Any] = None,
        generation_hooks: Optional[Dict[str, bool]] = None,
        memory_manager: Optional[Any] = None,
        enable_curiosity: bool = True,
        enable_temperament: bool = True,
        enable_confidence: bool = True,
        enable_bond: bool = True,
      
    ):
        self.config_manager = config_manager
        self.logger = logger if logger else Logger()
        # State is now explicitly typed and required
        self.state: SOVLState = state
        self.error_manager = error_manager if error_manager else ErrorManager()
        self.curiosity_manager = curiosity_manager
        self.temperament_system = temperament_system
        self.confidence_calculator = confidence_calculator
        self.bond_calculator = bond_calculator
        self.device = device
        self.lifecycle_manager = lifecycle_manager
        self.scaffold_manager = scaffold_manager
        self.memory_manager = memory_manager
      
        default_hooks = {
            "curiosity": enable_curiosity,
            "temperament": enable_temperament,
            "confidence": enable_confidence,
            "bond": enable_bond
        }
        hooks_from_config = {}
        if isinstance(config_manager, ConfigManager):
            hooks_from_config = config_manager.get("generation_hooks", {})
        self.generation_hooks = {**default_hooks, **(generation_hooks or {}), **hooks_from_config}
        self.logger.record_event(
            event_type="primer_initialized",
            message="GenerationPrimer initialized with config-driven hooks and parameters.",
            level="info",
            component="GenerationPrimer"
        )
        self.logger.record_event(
            event_type="primer_generation_hooks_final",
            message=f"Final merged generation_hooks: {self.generation_hooks}",
            level="info",
            component="GenerationPrimer"
        )
      
        self.curiosity_weight = getattr(self.curiosity_manager, "weight_ignorance", None)
        if isinstance(config_manager, ConfigManager):
            self.curiosity_weight = config_manager.get("curiosity_config.weight_ignorance", self.curiosity_weight)
        self.temperament_decay = None
        if isinstance(config_manager, ConfigManager):
            self.temperament_decay = config_manager.get("temperament_config.decay", None)
       
        self.logger.record_event(
            event_type="primer_initialized",
            message="GenerationPrimer initialized with config-driven hooks and parameters.",
            level="info",
            component="GenerationPrimer"
        )
        self._state_lock = threading.Lock()

    def set_generation_hook(self, trait: str, enabled: bool):
        """
        Enable or disable a specific trait's influence at runtime.
        Only known traits can be toggled. Logs and raises ValueError on invalid trait.
        """
        if trait not in self.get_all_trait_names():
            self.logger.log_error(
                error_msg=f"Attempted to toggle unknown trait '{trait}'",
                error_type="primer_trait_toggle_error",
                component="GenerationPrimer"
            )
            raise ValueError(f"Unknown trait: {trait}")
        self.generation_hooks[trait] = enabled
        self.logger.record_event(
            event_type="generation_hook_update",
            message=f"Trait '{trait}' set to {enabled}",
            level="info",
            component="GenerationPrimer"
        )

    def get_all_trait_names(self) -> list:
        """
        Returns a list of all trait names supported by this primer instance (for validation, UI, analytics).
        """
        # This should match keys in default_hooks and compute_traits
        return ["curiosity", "temperament", "confidence", "bond"]

    def get_enabled_traits(self) -> dict:
        """
        Returns a dictionary of currently enabled traits and their status.
        Useful for debugging, UI, and analytics.
        """
        return {trait: self.generation_hooks.get(trait, True) for trait in self.get_all_trait_names()}

    def handle_error(self, context: str, error: Exception, extra: Optional[dict] = None):
        """Aggregate and report errors with context."""
        error_info = {
            "context": context,
            "state": str(self.state),
            "traits": list(self.generation_hooks.keys()),
            "extra": extra or {}
        }
        self.logger.log_error(
            error_msg=f"[GenerationPrimer] Error in {context}: {error}",
            error_type="primer_critical_error",
            component="GenerationPrimer"
        )
        if self.error_manager:
            self.error_manager.handle_data_error(error, error_info, component="GenerationPrimer")

    def compute_traits(self, **kwargs) -> Dict[str, Any]:
        """
        Aggregates trait values from all connected modules, respecting generation hooks.
        Returns a dictionary with all trait outputs.
        Validates trait managers and state compatibility.
        """
        traits = {}
        state_arg = kwargs.get("state", self.state)
        # Validate state_arg type
        if not isinstance(state_arg, SOVLState):
            self.logger.log_error("Invalid state type for trait computation", error_type="primer_state_type_error")
            return {}

        # Curiosity
        if self.generation_hooks.get("curiosity", True):
            if not self.curiosity_manager or not hasattr(self.curiosity_manager, 'compute_curiosity'):
                self.logger.log_error("Curiosity manager missing or invalid", error_type="primer_curiosity_manager_error")
                traits["curiosity"] = None
            else:
                try:
                    traits["curiosity"] = self.curiosity_manager.compute_curiosity(state=state_arg, **kwargs)
                except Exception as e:
                    traits["curiosity"] = None
                    self.handle_error("curiosity_computation", e, {"kwargs": kwargs})

        # Temperament
        if self.generation_hooks.get("temperament", True):
            if not self.temperament_system or not hasattr(self.temperament_system, 'current_score'):
                self.logger.log_error("Temperament system missing or invalid", error_type="primer_temperament_manager_error")
                traits["temperament"] = None
            else:
                try:
                    traits["temperament"] = self.temperament_system.current_score(state=state_arg)
                except Exception as e:
                    traits["temperament"] = None
                    self.handle_error("temperament_computation", e, {"kwargs": kwargs})

        # Confidence
        if self.generation_hooks.get("confidence", True):
            if not self.confidence_calculator or not hasattr(self.confidence_calculator, 'calculate_confidence_score'):
                self.logger.log_error("Confidence calculator missing or invalid", error_type="primer_confidence_manager_error")
                traits["confidence"] = None
            else:
                try:
                    traits["confidence"] = self.confidence_calculator.calculate_confidence_score(state=state_arg, **kwargs)
                except Exception as e:
                    traits["confidence"] = None
                    self.handle_error("confidence_computation", e, {"kwargs": kwargs})

        # Bond
        if self.generation_hooks.get("bond", True):
            if not self.bond_calculator or not hasattr(self.bond_calculator, 'calculate_bond'):
                self.logger.log_error("Bond calculator missing or invalid", error_type="primer_bond_manager_error")
                traits["bond"] = None
            else:
                try:
                    traits["bond"] = self.bond_calculator.calculate_bond(state=state_arg, **kwargs)
                except Exception as e:
                    traits["bond"] = None
                    self.handle_error("bond_computation", e, {"kwargs": kwargs})

        # Fail fast if required traits are missing
        required_traits = ["curiosity", "temperament"]
        missing = [t for t in required_traits if traits.get(t) is None]
        if missing:
            raise RuntimeError(f"Failed to compute required traits: {missing}")

        return traits

    def get_traits_for_generation(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Prepares all trait influences and returns them, ready to be used by sovl_generation's generate_text.
        This is a placeholder for the actual call to sovl_generation.generate_text.
        """
        traits = self.compute_traits(prompt=prompt, **kwargs)
        return traits

    def adjust_parameter(self, base_value: float, parameter_type: str, **traits) -> float:
        """
        Adjusts a parameter (e.g., temperature) based on trait influences.
        Uses an additive approach for predictability and consistency with GenerationManager.
        """
        try:
            if parameter_type == "temperature":
                temperament = traits.get("temperament", 0.5)
                curiosity = traits.get("curiosity", 0.0)
                adjustment = (temperament - 0.5) * 0.3  # Scale to ±0.15
                if curiosity:
                    adjustment += curiosity * 0.2  # Scale to +0.2
                adjusted_value = base_value + adjustment
                adjusted_value = max(0.1, min(1.0, adjusted_value))
                self.logger.record_event(
                    event_type="parameter_adjusted",
                    message="Parameter adjusted (additive approach)",
                    level="info",
                    additional_info={
                        "parameter_type": parameter_type,
                        "base_value": base_value,
                        "adjusted_value": adjusted_value,
                        "temperament": temperament,
                        "curiosity": curiosity,
                        "adjustment": adjustment
                    }
                )
                return adjusted_value
            else:
                raise ValueError(f"Unsupported parameter type: {parameter_type}")
        except Exception as e:
            self.logger.record_event(
                event_type="parameter_adjustment_error",
                message=f"Failed to adjust parameter: {str(e)}",
                level="error",
                additional_info={
                    "parameter_type": parameter_type,
                    "base_value": base_value,
                    "traits": traits
                }
            )
            return base_value  # Return base value on error

    def assemble_metadata(self, prompt: str, traits: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Assemble metadata for logging/analytics, similar to ScribeAssembler in sovl_generation.
        """
        metadata = {
            "prompt": prompt,
            "traits": traits,
            "state": str(self.state),
            "device": str(self.device) if self.device else None,
            "timestamp": kwargs.get("timestamp"),
            "generation_hooks": self.generation_hooks.copy(),
            "additional": kwargs
        }
        self.logger.record_event(
            event_type="metadata_assembled",
            message=f"Metadata assembled for prompt.",
            level="debug",
            component="GenerationPrimer"
        )
        return metadata

    def compute_curiosity(self, **kwargs):
        if self.curiosity_manager:
            return self.curiosity_manager.compute_curiosity(**kwargs)
        return None

    def get_temperament(self):
        if self.temperament_system:
            return self.temperament_system.current_score() if hasattr(self.temperament_system, "current_score") else None
        return None

    def compute_confidence(self, **kwargs):
        if self.confidence_calculator:
            return self.confidence_calculator.calculate_confidence_score(**kwargs)
        return None

    def compute_bond(self, **kwargs):
        if self.bond_calculator:
            return self.bond_calculator.calculate_bond(**kwargs)
        return None

    def get_trait(self, trait: str, **kwargs):
        """
        Unified trait getter for seamless integration.
        Usage: primer.get_trait('curiosity', ...)
        Returns the computed value for the requested trait, or None if not available/enabled.
        Only known traits are allowed.
        """
        trait_map = {
            "curiosity": self.compute_curiosity,
            "temperament": self.get_temperament,
            "confidence": self.compute_confidence,
            "bond": self.compute_bond
        }
        if trait not in trait_map:
            self.logger.log_error(
                error_msg=f"Attempted to access unknown trait '{trait}'",
                error_type="primer_trait_access_error",
                component="GenerationPrimer"
            )
            raise ValueError(f"Unknown trait: {trait}")
        if self.generation_hooks.get(trait, True):
            try:
                return trait_map[trait](**kwargs)
            except Exception as e:
                self.handle_error(f"get_trait:{trait}", e, {"kwargs": kwargs})
        return None

    def get_all_traits(self, **kwargs) -> Dict[str, Any]:
        """
        Returns a dictionary of all trait values currently enabled, for use by generation.
        This is a synonym for compute_traits, but signals intent for integration.
        """
        return self.compute_traits(**kwargs)

    def prepare_for_generation(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Prepares all trait influences and returns them, ready to be used by sovl_generation's generate_text.
        This is the canonical call for integration.
        """
        traits = self.get_all_traits(prompt=prompt, **kwargs)
       
        return traits

    def update_state(self, new_state: SOVLState):
        """
        Update the state object used by the primer and all trait computations.
        Useful for hot-swapping user/system state at runtime.
        """
        self.state = new_state
        self.logger.record_event(
            event_type="state_updated",
            message="GenerationPrimer state object updated.",
            level="info",
            component="GenerationPrimer"
        )

    def update_curiosity_state(self, *args, **kwargs):
        """
        Update curiosity state post-generation. Delegates to curiosity_manager if available.
        """
        if self.curiosity_manager and hasattr(self.curiosity_manager, "update_metrics"):
            try:
                return self.curiosity_manager.update_metrics(*args, **kwargs)
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Failed to update curiosity state: {str(e)}",
                    error_type="curiosity_update_error",
                    stack_trace=traceback.format_exc()
                )
        else:
            self.logger.log_warning("Curiosity manager not available for curiosity state update.")
        return None

    def update_state_after_error(self, error: Exception, context: str) -> None:
        """
        Update system state after an error occurs. Adjusts confidence and temperament as appropriate.
        Thread-safe, validates attributes, and rolls back on failure.
        """
        with self._state_lock:
            original_confidence = getattr(self.state, 'confidence', None)
            original_temperament = getattr(self.state, 'temperament_score', None)
            try:
                # Validate confidence
                if not hasattr(self.state, 'confidence') or not isinstance(self.state.confidence, (int, float)):
                    self.logger.log_error("Invalid or missing confidence attribute in SOVLState")
                    return
                # Adjust confidence based on error type
                if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
                    self.state.confidence = max(0.1, self.state.confidence - 0.1)
                elif isinstance(error, (ValueError, RuntimeError)):
                    self.state.confidence = max(0.2, self.state.confidence - 0.05)
                # Validate and update temperament_score if present
                if hasattr(self.state, 'temperament_score'):
                    if not isinstance(self.state.temperament_score, (int, float)):
                        self.logger.log_error("Invalid temperament_score type in SOVLState")
                        return
                    self.state.temperament_score = max(0.0, self.state.temperament_score - 0.05)
                self.logger.record_event(
                    event_type="state_updated_after_error",
                    message=f"State updated after {context} error",
                    level="info",
                    additional_info={
                        'error_type': type(error).__name__,
                        'new_confidence': self.state.confidence,
                        'new_temperament': getattr(self.state, 'temperament_score', None)
                    }
                )
            except Exception as e:
                # Rollback on failure
                if original_confidence is not None:
                    self.state.confidence = original_confidence
                if original_temperament is not None and hasattr(self.state, 'temperament_score'):
                    self.state.temperament_score = original_temperament
                self.logger.log_error(
                    error_msg=f"Failed to update state after error: {str(e)}",
                    error_type="state_update_error",
                    stack_trace=traceback.format_exc()
                )

    def handle_state_driven_error(self, error: Exception, context: str, state_metrics: dict = None) -> None:
        """
        Enhanced state-driven error handling. Logs state metrics and invokes explicit recovery.
        """
        if self.error_manager and hasattr(self.error_manager, "handle_generation_error"):
            try:
                self.error_manager.handle_generation_error(error=error, context=context, state=self.state, state_metrics=state_metrics)
                self.logger.record_event(
                    event_type="state_driven_error_handled",
                    message=f"State-driven error handled for context: {context}",
                    level="info",
                    additional_info={
                        "error_type": type(error).__name__,
                        "context": context,
                        "state_metrics": state_metrics
                    }
                )
                # Explicitly invoke recovery after error handling
                self.apply_state_driven_recovery(error, context, state_metrics=state_metrics)
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Failed during state-driven error handling: {str(e)}",
                    error_type="state_driven_error_handling_error",
                    stack_trace=traceback.format_exc()
                )
        else:
            self.logger.log_warning("Error manager not available for state-driven error handling.")
            # Still attempt explicit recovery
            self.apply_state_driven_recovery(error, context, state_metrics=state_metrics)

    def apply_state_driven_recovery(self, error: Exception, context: str, state_metrics: dict = None) -> None:
        """
        Enhanced recovery strategies for errors. Includes explicit actions and logs state metrics.
        Brings logic to parity with sovl_generation: includes memory optimization, batch size, temperament,
        and lifecycle adjustments, all conditional on state_metrics, and logs before/after states.
        """
        recovery_actions = []
        try:
            # Memory optimization (if available)
            if hasattr(self.state, 'ram_manager') and self.state.ram_manager:
                before = self.state.ram_manager.get_usage()
                self.state.ram_manager.optimize_memory()
                after = self.state.ram_manager.get_usage()
                recovery_actions.append({
                    "action": "optimize_memory",
                    "before": before,
                    "after": after
                })
            # Batch size adjustment based on confidence
            if state_metrics and state_metrics.get('confidence', 1.0) < 0.3 and hasattr(self.state, 'batch_size'):
                old_batch_size = self.state.batch_size
                self.state.batch_size = max(1, self.state.batch_size // 2)
                recovery_actions.append({
                    "action": "adjust_batch_size",
                    "old_batch_size": old_batch_size,
                    "new_batch_size": self.state.batch_size
                })
            # Temperament adjustment (if present)
            if hasattr(self.state, 'temperament_score'):
                old_temp = self.state.temperament_score
                self.state.temperament_score = max(0.1, self.state.temperament_score - 0.05)
                recovery_actions.append({
                    'action': 'adjust_temperament',
                    'old_temperament': old_temp,
                    'new_temperament': self.state.temperament_score
                })
            # Lifecycle stage update based on metrics
            if hasattr(self.state, 'lifecycle_stage'):
                old_stage = self.state.lifecycle_stage
                if state_metrics and state_metrics.get('lifecycle_stage') == 'exploration':
                    self.state.lifecycle_stage = 'consolidation'
                    recovery_actions.append({
                        'action': 'update_lifecycle_stage',
                        'old_stage': old_stage,
                        'new_stage': self.state.lifecycle_stage
                    })
            # Call error_manager's recovery if available
            if self.error_manager and hasattr(self.error_manager, "apply_recovery_strategy"):
                self.error_manager.apply_recovery_strategy(error=error, context=context, state=self.state, state_metrics=state_metrics)
            self.logger.record_event(
                event_type="state_driven_recovery_applied",
                message=f"State-driven recovery applied for context: {context}",
                level="info",
                additional_info={
                    "error_type": type(error).__name__,
                    "context": context,
                    "state_metrics": state_metrics,
                    "recovery_actions": recovery_actions
                }
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed during state-driven recovery: {str(e)}",
                error_type="state_driven_recovery_error",
                stack_trace=traceback.format_exc()
            )

    def sync_traits_to_state(self, traits: dict) -> None:
        """
        Automatically syncs all trait values to the corresponding attributes in SOVLState.
        Only updates traits that are present as attributes on the state object.
        Logs a warning if a trait is not found on the state.
        """
        if not traits:
            return
        for trait, value in traits.items():
            if hasattr(self.state, trait):
                try:
                    setattr(self.state, trait, value)
                except Exception as e:
                    self.logger.log_error(
                        error_msg=f"Failed to set state.{trait} to {value}: {str(e)}",
                        error_type="trait_state_sync_error",
                        stack_trace=traceback.format_exc()
                    )
            else:
                self.logger.log_warning(
                    f"Trait '{trait}' not found in SOVLState; skipping state update.",
                    event_type="trait_state_sync_warning"
                )

    def update_state_after_operation(self, context: str = None, result: dict = None) -> None:
        """
        Enhanced state updates post-operation. Handles operation result and logs adjustments.
        Adds fallback logic if SOVLState.update_after_operation is not available.
        Now also syncs all traits in result['traits'] to state.
        """
        try:
            # Sync all traits to state if present in result
            traits = result.get("traits") if result else None
            if traits:
                self.sync_traits_to_state(traits)
            # Fallback: Apply default adjustment if no result and confidence exists
            used_fallback = False
            if not result and hasattr(self.state, 'confidence'):
                self.state.confidence = min(1.0, self.state.confidence + 0.05)  # Default success boost
                used_fallback = True
            # Apply result-based adjustments (legacy deltas)
            if result:
                if "confidence_delta" in result and hasattr(self.state, "confidence"):
                    self.state.confidence = max(0.0, min(1.0, self.state.confidence + result["confidence_delta"]))
                if "temperament_delta" in result and hasattr(self.state, "temperament_score"):
                    self.state.temperament_score = max(0.0, min(1.0, self.state.temperament_score + result["temperament_delta"]))
            # Delegate to state if method exists
            if hasattr(self.state, "update_after_operation"):
                self.state.update_after_operation(context=context)
            else:
                if not used_fallback:
                    self.logger.log_warning("SOVLState.update_after_operation not available, using default adjustments.")
            self.logger.record_event(
                event_type="state_updated_after_operation",
                message=f"State updated after operation: {context}",
                level="info",
                additional_info={
                    'context': context,
                    'confidence': getattr(self.state, 'confidence', None),
                    'temperament': getattr(self.state, 'temperament_score', None),
                    'operation_result': result
                }
            )
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to update state after operation: {str(e)}",
                error_type="state_update_operation_error",
                stack_trace=traceback.format_exc()
            )