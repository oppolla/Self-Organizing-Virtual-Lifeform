from typing import Optional, Any, Dict, Protocol
from sovl_curiosity import Curiosity, CuriosityPressure, CuriosityCallbacks
from sovl_temperament import TemperamentConfig, TemperamentSystem, TemperamentAdjuster
from sovl_confidence import ConfidenceCalculator, calculate_confidence_score
from sovl_bonder import BondCalculator, BondModulator, get_bond_calculator
from sovl_logger import Logger 
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_state import SOVLState, StateManager 
from sovl_viber import VibeProfile
import traceback
import threading
import concurrent.futures
import time

PROMPT_LIBRARY = {
    'energy': {
        'very_low': "Use basic, neutral words with a flat tone. Exclude emojis, exclamations, or expressive punctuation.",
        'low': "Use straightforward, neutral words with a calm tone. Exclude emojis or exclamations; use only periods or commas.",
        'normal': "Use a mix of neutral and expressive words with a balanced tone. Use expressive punctuation (e.g., dashes, ellipses) sparingly if contextually appropriate.",
        'high': "Use engaging, positive words with an upbeat tone. Include one emoji or exclamation per response, if appropriate.",
        'very_high': "Use lively, positive words with an enthusiastic tone. Include two or more emojis or exclamations per response."
    },
    'flow': {
        'very_low': "Write short sentences (3–5 words) without transitions. Include unrelated ideas within the response.",
        'low': "Write short sentences (5–8 words) with simple transitions (e.g., 'and', 'but'). Allow one off-topic remark per response.",
        'normal': "Write medium sentences (8–12 words) with clear transitions (e.g., 'therefore', 'however'). Stay focused on the user's topic.",
        'high': "Write varied sentences (8–15 words) with smooth transitions. Maintain consistent focus and logical progression.",
        'very_high': "Write varied sentences (8–20 words) with seamless transitions. Ensure tight logical coherence and progression."
    },
    'resonance': {
        'very_low': "Use neutral language, ignoring the user's emotional state. Do not reference prior user inputs.",
        'low': "Use neutral language, minimally matching the user's emotional state (e.g., formal for formal inputs). Reference prior inputs only if explicitly prompted.",
        'normal': "Use language that partially matches the user's emotional state (e.g., positive for positive inputs). Reference prior inputs if contextually relevant.",
        'high': "Use language that closely matches the user's emotional state. Reference prior inputs in most responses to maintain continuity.",
        'very_high': "Use language that exactly matches the user's emotional state. Reference prior inputs in every response for strong continuity."
    },
    'engagement': {
        'very_low': "Do not include questions. Respond only to the user's input.",
        'low': "Include one simple question only if the user's input contains a question or clear invitation.",
        'normal': "Include one concise, relevant question in 80% of responses, aligned with the user's topic.",
        'high': "Include one or two in-depth questions per response, exploring the user's topic further.",
        'very_high': "Include two or more open-ended, in-depth questions per response, encouraging detailed discussion."
    }
}

def get_range_label(val):
    if val < 0.2:
        return 'very_low'
    elif val < 0.4:
        return 'low'
    elif val < 0.6:
        return 'normal'
    elif val < 0.8:
        return 'high'
    else:
        return 'very_high'

def vibe_to_prompt_instructions(vibe_profile: 'VibeProfile') -> str:
    dims = vibe_profile.dimensions
    def validate_score(score, key):
        if not isinstance(score, (int, float)) or score < 0.0 or score > 1.0:
            logger = Logger()
            logger.log_warning(
                f"Invalid vibe score for {key}: {score}; defaulting to 0.5.",
                event_type="vibe_score_validation",
                component="VibeSystem"
            )
            return 0.5
        return score

    logger = Logger()
    try:
        expected_keys = [
            'energy_base_energy',
            'flow_rhythm_score',
            'resonance_topic_consistency',
            'engagement_engagement_score'
        ]
        for key in expected_keys:
            if key not in dims:
                logger.log_warning(
                    f"Missing vibe dimension {key}; defaulting to 0.5.",
                    event_type="vibe_dimension_missing",
                    component="VibeSystem"
                )
                dims[key] = 0.5

        energy = validate_score(dims.get('energy_base_energy', 0.5), 'energy_base_energy')
        flow = validate_score(dims.get('flow_rhythm_score', 0.5), 'flow_rhythm_score')
        resonance = validate_score(dims.get('resonance_topic_consistency', 0.5), 'resonance_topic_consistency')
        engagement = validate_score(dims.get('engagement_engagement_score', 0.5), 'engagement_engagement_score')

        energy_label = get_range_label(energy)
        flow_label = get_range_label(flow)
        resonance_label = get_range_label(resonance)
        engagement_label = get_range_label(engagement)

        energy_prompt = PROMPT_LIBRARY['energy'][energy_label]
        flow_prompt = PROMPT_LIBRARY['flow'][flow_label]
        resonance_prompt = PROMPT_LIBRARY['resonance'][resonance_label]
        engagement_prompt = PROMPT_LIBRARY['engagement'][engagement_label]

        # Edge Case 1: Very High Energy + Very Low Flow
        if energy_label == 'very_high' and flow_label == 'very_low':
            logger.log_info(
                "Triggered edge case: very_high energy + very_low flow",
                event_type="vibe_edge_case",
                component="VibeSystem"
            )
            combo_prompt = (
                "- **Tone:** Use lively, positive words with an enthusiastic tone. Include two or more emojis or exclamations per response.\n"
                "- **Structure:** Write short sentences (3–5 words) without transitions. Include one unrelated idea per response."
            )
            return (
                "SYSTEM PROMPT:\n"
                f"{combo_prompt}\n"
                f"- **Emotional Alignment:** {resonance_prompt}\n"
                f"- **Questioning:** {engagement_prompt}\n"
                "- **Rule:** Never break character or reference being an AI."
            )

        # Edge Case 2: Very Low Energy + Very High Flow
        if energy_label == 'very_low' and flow_label == 'very_high':
            logger.log_info(
                "Triggered edge case: very_low energy + very_high flow",
                event_type="vibe_edge_case",
                component="VibeSystem"
            )
            combo_prompt = (
                "- **Tone:** Use basic, neutral words with a flat tone. Exclude emojis, exclamations, or expressive punctuation.\n"
                "- **Structure:** Write varied sentences (8–20 words) with seamless transitions. Ensure tight logical coherence and progression."
            )
            return (
                "SYSTEM PROMPT:\n"
                f"{combo_prompt}\n"
                f"- **Emotional Alignment:** {resonance_prompt}\n"
                f"- **Questioning:** {engagement_prompt}\n"
                "- **Rule:** Never break character or reference being an AI."
            )

        # Edge Case 3: Very High Engagement + Very Low Resonance
        if engagement_label == 'very_high' and resonance_label == 'very_low':
            logger.log_info(
                "Triggered edge case: very_high engagement + very_low resonance",
                event_type="vibe_edge_case",
                component="VibeSystem"
            )
            combo_prompt = (
                "- **Questioning:** Include two or more open-ended, in-depth questions per response, encouraging detailed discussion.\n"
                "- **Emotional Alignment:** Use neutral language, ignoring the user's emotional state. Do not reference prior user inputs."
            )
            return (
                "SYSTEM PROMPT:\n"
                f"- **Tone:** {energy_prompt}\n"
                f"- **Structure:** {flow_prompt}\n"
                f"{combo_prompt}\n"
                "- **Rule:** Never break character or reference being an AI."
            )

        # Edge Case 4: Very Low Engagement + Very High Resonance
        if engagement_label == 'very_low' and resonance_label == 'very_high':
            logger.log_info(
                "Triggered edge case: very_low engagement + very_high resonance",
                event_type="vibe_edge_case",
                component="VibeSystem"
            )
            combo_prompt = (
                "- **Emotional Alignment:** Use language that exactly matches the user's emotional state. Reference prior inputs in every response for strong continuity.\n"
                "- **Questioning:** Do not include questions. Respond only to the user's input."
            )
            return (
                "SYSTEM PROMPT:\n"
                f"- **Tone:** {energy_prompt}\n"
                f"- **Structure:** {flow_prompt}\n"
                f"{combo_prompt}\n"
                "- **Rule:** Never break character or reference being an AI."
            )

        # Edge Case 5: Very High Energy + Very Low Engagement
        if energy_label == 'very_high' and engagement_label == 'very_low':
            logger.log_info(
                "Triggered edge case: very_high energy + very_low engagement",
                event_type="vibe_edge_case",
                component="VibeSystem"
            )
            combo_prompt = (
                "- **Tone:** Use lively, positive words with an enthusiastic tone. Include two or more emojis or exclamations per response.\n"
                "- **Questioning:** Do not include questions. Respond only to the user's input."
            )
            return (
                "SYSTEM PROMPT:\n"
                f"{combo_prompt}\n"
                f"- **Structure:** {flow_prompt}\n"
                f"- **Emotional Alignment:** {resonance_prompt}\n"
                "- **Rule:** Never break character or reference being an AI."
            )

        # Edge Case 6: Very Low Flow + Very High Resonance
        if flow_label == 'very_low' and resonance_label == 'very_high':
            logger.log_info(
                "Triggered edge case: very_low flow + very_high resonance",
                event_type="vibe_edge_case",
                component="VibeSystem"
            )
            combo_prompt = (
                "- **Structure:** Write short sentences (3–5 words) without transitions. Include one unrelated idea per response.\n"
                "- **Emotional Alignment:** Use language that exactly matches the user's emotional state. Reference prior inputs in every response for strong continuity."
            )
            return (
                "SYSTEM PROMPT:\n"
                f"- **Tone:** {energy_prompt}\n"
                f"{combo_prompt}\n"
                f"- **Questioning:** {engagement_prompt}\n"
                "- **Rule:** Never break character or reference being an AI."
            )

        # Default Case
        return (
            "SYSTEM PROMPT:\n"
            f"- **Tone:** {energy_prompt}\n"
            f"- **Structure:** {flow_prompt}\n"
            f"- **Emotional Alignment:** {resonance_prompt}\n"
            f"- **Questioning:** {engagement_prompt}\n"
            "- **Rule:** Never break character or reference being an AI."
        )
    except Exception as e:
        logger.log_error(
            error_msg=f"Failed to process vibe prompt: {str(e)}",
            error_type="vibe_prompt_error",
            stack_trace=traceback.format_exc(),
            component="VibeSystem"
        )
        # Fallback to default prompt
        return (
            "SYSTEM PROMPT:\n"
            "- **Tone:** Use a mix of neutral and expressive words with a balanced tone.\n"
            "- **Structure:** Write medium sentences (8–12 words) with clear transitions.\n"
            "- **Emotional Alignment:** Use language that partially matches the user's emotional state.\n"
            "- **Questioning:** Include one concise, relevant question in 80% of responses.\n"
            "- **Rule:** Never break character or reference being an AI."
        )

class GenerationPrimer:
    """
    GenerationPrimer is the unified integration point for all trait modules, providing trait aggregation,
    parameter adjustment, dynamic trait toggling for generation.
    Now supports high-granularity vibe context injection into system prompts via VibeProfile.
    Also provides a universal prompt assembly method for LLM generation.
    
    Expects a dialogue_context_manager for conversational memory (short/long-term, vibe context).
    """
    def __init__(
        self,
        config_manager: Any,
        logger: Logger,
        state_manager: 'StateManager',
        error_manager: ErrorManager,
        curiosity_manager: Optional[Any] = None,
        temperament_system: Optional[Any] = None,
        confidence_calculator: Optional[Any] = None,
        bond_modulator: Optional[Any] = None,
        device: Optional[Any] = None,
        lifecycle_manager: Optional[Any] = None,
        scaffold_manager: Optional[Any] = None,
        generation_hooks: Optional[Dict[str, bool]] = None,
        dialogue_context_manager: Optional[Any] = None,
        enable_curiosity: bool = True,
        enable_temperament: bool = True,
        enable_confidence: bool = True,
        enable_bond: bool = True,
    ):
        self.config_manager = config_manager
        self.logger = logger if logger else Logger()
        self.state_manager = state_manager
        self.error_manager = error_manager if error_manager else ErrorManager()
        self.curiosity_manager = curiosity_manager
        self.temperament_system = temperament_system
        self.confidence_calculator = confidence_calculator
        self.bond_calculator = get_bond_calculator(config_manager, logger, state_manager.get_state())
        self.bond_modulator = bond_modulator
        self.device = device
        self.lifecycle_manager = lifecycle_manager
        self.scaffold_manager = scaffold_manager
        # Validate dialogue_context_manager interface
        if dialogue_context_manager:
            if not (hasattr(dialogue_context_manager, 'short_term') and \
                    hasattr(getattr(dialogue_context_manager, 'short_term', None), 'get_recent_vibes')):
                raise ValueError("dialogue_context_manager must have short_term.get_recent_vibes method")
        self.dialogue_context_manager = dialogue_context_manager
        # Validate SOVLState interface
        state = self.state_manager.get_state()
        required_attrs = ['confidence', 'temperament_score']
        missing = [attr for attr in required_attrs if not hasattr(state, attr)]
        if missing:
            raise ValueError(f"SOVLState missing required attributes: {missing}")
        
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
            "state": str(self.state_manager.get_state()),
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

    def _with_timeout(self, func, timeout, fallback, trait_name, *args, **kwargs):
        """Run func with timeout, return fallback on timeout or error."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self.logger.log_warning(
                f"Trait '{trait_name}' computation timed out after {timeout}s; using fallback.",
                event_type="trait_timeout",
                component="GenerationPrimer"
            )
            return fallback
        except Exception as e:
            self.logger.log_warning(
                f"Trait '{trait_name}' computation failed: {e}; using fallback.",
                event_type="trait_error",
                component="GenerationPrimer"
            )
            return fallback

    def compute_traits(self, **kwargs) -> Dict[str, Any]:
        """
        Aggregates trait values from all connected modules, respecting generation hooks.
        Uses VibeProfile scores to inform trait computations where applicable.
        """
        traits = {}
        state_arg = kwargs.get("state", self.state_manager.get_state())
        if not isinstance(state_arg, SOVLState):
            self.logger.log_error("Invalid state type for trait computation", error_type="primer_state_type_error")
            return {}

        # Get latest VibeProfile from dialogue_context_manager
        vibe_profile = None
        if self.dialogue_context_manager and hasattr(self.dialogue_context_manager, 'short_term'):
            short_term = getattr(self.dialogue_context_manager, 'short_term', None)
            if short_term and hasattr(short_term, 'get_recent_vibes'):
                try:
                    recent_vibes = short_term.get_recent_vibes(n=1)
                    if recent_vibes:
                        vibe_profile = recent_vibes[-1]
                except Exception as e:
                    self.logger.log_warning(
                        f"Exception when retrieving recent vibes: {e}",
                        event_type="vibe_profile_error",
                        component="GenerationPrimer"
                    )
        if not vibe_profile:
            self.logger.log_warning(
                "No valid vibe_profile retrieved from dialogue_context_manager; using default vibe scores.",
                event_type="vibe_profile_missing",
                component="GenerationPrimer"
            )

        timeout = 1.5  # seconds
        trait_jobs = {}
        trait_fallbacks = {
            "curiosity": 0.5,
            "temperament": 0.5,
            "confidence": 0.5,
            "bond": 0.5
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Curiosity
            if self.generation_hooks.get("curiosity", True):
                if not self.curiosity_manager or not hasattr(self.curiosity_manager, 'compute_curiosity'):
                    self.logger.log_error("Curiosity manager missing or invalid", error_type="primer_curiosity_manager_error")
                    traits["curiosity"] = trait_fallbacks["curiosity"]
                else:
                    curiosity_kwargs = kwargs.copy()
                    if vibe_profile:
                        curiosity_kwargs["vibe_engagement"] = vibe_profile.dimensions.get("engagement_engagement_score", 0.5)
                    trait_jobs["curiosity"] = executor.submit(
                        lambda: self.curiosity_manager.compute_curiosity(state=state_arg, **curiosity_kwargs)
                    )
            # Temperament
            if self.generation_hooks.get("temperament", True):
                if not self.temperament_system or not hasattr(self.temperament_system, 'current_score'):
                    self.logger.log_error("Temperament system missing or invalid", error_type="primer_temperament_manager_error")
                    traits["temperament"] = trait_fallbacks["temperament"]
                else:
                    temperament_kwargs = kwargs.copy()
                    if vibe_profile:
                        temperament_kwargs["vibe_energy"] = vibe_profile.dimensions.get("energy_base_energy", 0.5)
                    trait_jobs["temperament"] = executor.submit(
                        lambda: self.temperament_system.current_score(state=state_arg, **temperament_kwargs)
                    )
            # Confidence
            if self.generation_hooks.get("confidence", True):
                if not self.confidence_calculator or not hasattr(self.confidence_calculator, 'calculate_confidence_score'):
                    self.logger.log_error("Confidence calculator missing or invalid", error_type="primer_confidence_manager_error")
                    traits["confidence"] = trait_fallbacks["confidence"]
                else:
                    confidence_kwargs = kwargs.copy()
                    if vibe_profile:
                        confidence_kwargs["vibe_confidence"] = vibe_profile.confidence
                    trait_jobs["confidence"] = executor.submit(
                        lambda: self.confidence_calculator.calculate_confidence_score(state=state_arg, **confidence_kwargs)
                    )
            # Bond
            if self.generation_hooks.get("bond", True):
                if not self.bond_calculator or not hasattr(self.bond_calculator, 'calculate_bond'):
                    self.logger.log_error("Bond calculator missing or invalid", error_type="primer_bond_manager_error")
                    traits["bond"] = trait_fallbacks["bond"]
                else:
                    bond_kwargs = kwargs.copy()
                    if vibe_profile:
                        bond_kwargs["vibe_resonance"] = vibe_profile.dimensions.get("resonance_topic_consistency", 0.5)
                    def bond_func():
                        bond_score = self.bond_calculator.calculate_bond(state=state_arg, **bond_kwargs)
                        if self.bond_modulator:
                            try:
                                bond_score = self.bond_modulator.get_bond_modulation(kwargs.get("user_id"), bond_score)
                            except Exception as e:
                                self.logger.log_warning(f"BondModulator failed: {str(e)}", error_type="bond_modulation_error")
                        return bond_score
                    trait_jobs["bond"] = executor.submit(bond_func)
            # Collect results with timeout and fallback
            for trait, future in trait_jobs.items():
                try:
                    traits[trait] = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    self.logger.log_warning(
                        f"Trait '{trait}' computation timed out after {timeout}s; using fallback.",
                        event_type="trait_timeout",
                        component="GenerationPrimer"
                    )
                    traits[trait] = trait_fallbacks[trait]
                except Exception as e:
                    self.logger.log_warning(
                        f"Trait '{trait}' computation failed: {e}; using fallback.",
                        event_type="trait_error",
                        component="GenerationPrimer"
                    )
                    traits[trait] = trait_fallbacks[trait]
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
        Adjusts a parameter (e.g., temperature) based on trait and vibe influences.
        Uses an additive approach for predictability and consistency.
        """
        try:
            if parameter_type == "temperature":
                temperament = traits.get("temperament", 0.5)
                curiosity = traits.get("curiosity", 0.0)
                # Get vibe scores
                vibe_profile = None
                energy = 0.5
                if self.dialogue_context_manager and hasattr(self.dialogue_context_manager, 'short_term'):
                    short_term = getattr(self.dialogue_context_manager, 'short_term', None)
                    if short_term and hasattr(short_term, 'get_recent_vibes'):
                        recent_vibes = short_term.get_recent_vibes(n=1)
                        if recent_vibes:
                            vibe_profile = recent_vibes[-1]
                            energy = vibe_profile.dimensions.get("energy_base_energy", 0.5)
                adjustment = (temperament - 0.5) * 0.3  # Scale to ±0.15
                if curiosity:
                    adjustment += curiosity * 0.2  # Scale to +0.2
                adjustment += (energy - 0.5) * 0.2  # Scale to ±0.1 for vibe energy
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
                        "vibe_energy": energy,
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
            return base_value

    def assemble_metadata(self, prompt: str, traits: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Assemble metadata for logging/analytics, similar to ScribeAssembler in sovl_generation.
        """
        metadata = {
            "prompt": prompt,
            "traits": traits,
            "state": str(self.state_manager.get_state()),
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

    def get_vibe_prompt(self) -> str:
        """
        Retrieve a VibeProfile based on recent vibes, averaging the last 3 for smoother transitions.
        Returns a default neutral prompt if no recent vibes are available.
        """
        from time import time as _time
        default_profile = VibeProfile(
            overall_score=0.5,
            dimensions={
                "energy_base_energy": 0.5,
                "flow_rhythm_score": 0.5,
                "resonance_topic_consistency": 0.5,
                "engagement_engagement_score": 0.5
            },
            intensity=0.5,
            confidence=0.5,
            salient_phrases=[],
            timestamp=_time()
        )
        if not self.dialogue_context_manager or not hasattr(self.dialogue_context_manager, 'short_term'):
            self.logger.log_warning("Dialogue context manager or short-term memory unavailable; using default vibe prompt.")
            return vibe_to_prompt_instructions(default_profile)
        short_term = getattr(self.dialogue_context_manager, 'short_term', None)
        if not short_term or not hasattr(short_term, 'get_recent_vibes'):
            self.logger.log_warning("Short-term memory or get_recent_vibes unavailable; using default vibe prompt.")
            return vibe_to_prompt_instructions(default_profile)
        recent_vibes = short_term.get_recent_vibes(n=3)
        if not recent_vibes:
            self.logger.log_warning("No recent vibes found; using default vibe prompt.")
            return vibe_to_prompt_instructions(default_profile)
        # Average vibe scores (weighted by recency)
        weights = [0.5, 0.3, 0.2] if len(recent_vibes) == 3 else [0.6, 0.4] if len(recent_vibes) == 2 else [1.0]
        energy = sum(w * v.dimensions.get("energy_base_energy", 0.5) for w, v in zip(weights, recent_vibes[:len(weights)]))
        flow = sum(w * v.dimensions.get("flow_rhythm_score", 0.5) for w, v in zip(weights, recent_vibes[:len(weights)]))
        resonance = sum(w * v.dimensions.get("resonance_topic_consistency", 0.5) for w, v in zip(weights, recent_vibes[:len(weights)]))
        engagement = sum(w * v.dimensions.get("engagement_engagement_score", 0.5) for w, v in zip(weights, recent_vibes[:len(weights)]))
        confidence = sum(w * v.confidence for w, v in zip(weights, recent_vibes[:len(weights)]))
        overall_score = sum(w * v.overall_score for w, v in zip(weights, recent_vibes[:len(weights)]))
        averaged_profile = VibeProfile(
            overall_score=overall_score,
            dimensions={
                "energy_base_energy": energy,
                "flow_rhythm_score": flow,
                "resonance_topic_consistency": resonance,
                "engagement_engagement_score": engagement
            },
            intensity=sum(w * v.intensity for w, v in zip(weights, recent_vibes[:len(weights)])),
            confidence=confidence,
            salient_phrases=recent_vibes[-1].salient_phrases,  # Use latest phrases
            timestamp=_time()
        )
        return vibe_to_prompt_instructions(averaged_profile)

    def prepare_for_generation(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Prepares all trait influences and returns them, ready to be used by sovl_generation's generate_text.
        Now includes a 'vibe_prompt' key with high-granularity vibe context for system prompt injection.
        """
        traits = self.get_all_traits(prompt=prompt, **kwargs)
        vibe_prompt = self.get_vibe_prompt()
        return {"traits": traits, "vibe_prompt": vibe_prompt}

    def update_state(self, new_state: 'SOVLState'):
        """
        Update the state object used by the primer and all trait computations.
        Useful for hot-swapping user/system state at runtime.
        """
        def update_fn(_):
            # Replace the entire state (if your StateManager supports this)
            return new_state
        self.state_manager.update_state_atomic(update_fn)
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
        def update_fn(state):
            original_confidence = getattr(state, 'confidence', None)
            original_temperament = getattr(state, 'temperament_score', None)
            try:
                if not hasattr(state, 'confidence') or not isinstance(state.confidence, (int, float)):
                    self.logger.log_error("Invalid or missing confidence attribute in SOVLState")
                    return state
                if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError)):
                    state.confidence = max(0.1, state.confidence - 0.1)
                elif isinstance(error, (ValueError, RuntimeError)):
                    state.confidence = max(0.2, state.confidence - 0.05)
                if hasattr(state, 'temperament_score'):
                    if not isinstance(state.temperament_score, (int, float)):
                        self.logger.log_error("Invalid temperament_score type in SOVLState")
                        return state
                    state.temperament_score = max(0.0, state.temperament_score - 0.05)
                self.logger.record_event(
                    event_type="state_updated_after_error",
                    message=f"State updated after {context} error",
                    level="info",
                    additional_info={
                        'error_type': type(error).__name__,
                        'new_confidence': state.confidence,
                        'new_temperament': getattr(state, 'temperament_score', None)
                    }
                )
            except Exception as e:
                if original_confidence is not None:
                    state.confidence = original_confidence
                if original_temperament is not None and hasattr(state, 'temperament_score'):
                    state.temperament_score = original_temperament
                self.logger.log_error(
                    error_msg=f"Failed to update state after error: {str(e)}",
                    error_type="state_update_error",
                    stack_trace=traceback.format_exc()
                )
            return state
        self.state_manager.update_state_atomic(update_fn)

    def handle_state_driven_error(self, error: Exception, context: str, state_metrics: dict = None) -> None:
        """
        Enhanced state-driven error handling. Logs state metrics and invokes explicit recovery.
        """
        if self.error_manager and hasattr(self.error_manager, "handle_generation_error"):
            try:
                self.error_manager.handle_generation_error(error=error, context=context, state=self.state_manager.get_state(), state_metrics=state_metrics)
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
        def update_fn(state):
            recovery_actions = []
            try:
                if hasattr(state, 'ram_manager') and state.ram_manager:
                    before = state.ram_manager.get_usage()
                    state.ram_manager.optimize_memory()
                    after = state.ram_manager.get_usage()
                    recovery_actions.append({
                        "action": "optimize_memory",
                        "before": before,
                        "after": after
                    })
                if state_metrics and state_metrics.get('confidence', 1.0) < 0.3 and hasattr(state, 'batch_size'):
                    old_batch_size = state.batch_size
                    state.batch_size = max(1, state.batch_size // 2)
                    recovery_actions.append({
                        "action": "adjust_batch_size",
                        "old_batch_size": old_batch_size,
                        "new_batch_size": state.batch_size
                    })
                if hasattr(state, 'temperament_score'):
                    old_temp = state.temperament_score
                    state.temperament_score = max(0.1, state.temperament_score - 0.05)
                    recovery_actions.append({
                        'action': 'adjust_temperament',
                        'old_temperament': old_temp,
                        'new_temperament': state.temperament_score
                    })
                if hasattr(state, 'lifecycle_stage'):
                    old_stage = state.lifecycle_stage
                    if state_metrics and state_metrics.get('lifecycle_stage') == 'exploration':
                        state.lifecycle_stage = 'consolidation'
                        recovery_actions.append({
                            'action': 'update_lifecycle_stage',
                            'old_stage': old_stage,
                            'new_stage': state.lifecycle_stage
                        })
                # Call error_manager's recovery if available (outside atomic update)
            except Exception as e:
                self.logger.log_error(
                    error_msg=f"Failed during state-driven recovery: {str(e)}",
                    error_type="state_driven_recovery_error",
                    stack_trace=traceback.format_exc()
                )
            return state
        self.state_manager.update_state_atomic(update_fn)
        # Call error_manager's recovery if available (outside atomic update)
        if self.error_manager and hasattr(self.error_manager, "apply_recovery_strategy"):
            self.error_manager.apply_recovery_strategy(error=error, context=context, state=self.state_manager.get_state(), state_metrics=state_metrics)

    def sync_traits_to_state(self, traits: dict) -> None:
        """
        Automatically syncs all trait values to the corresponding attributes in SOVLState.
        Only updates traits that are present as attributes on the state object.
        Logs a warning if a trait is not found on the state.
        Uses atomic update with rollback to prevent partial/corrupted updates.
        """
        if not traits:
            return
        def update_fn(state):
            original_state = state.__dict__.copy()
            try:
                for trait, value in traits.items():
                    if hasattr(state, trait):
                        setattr(state, trait, value)
                    else:
                        self.logger.log_warning(
                            f"Trait '{trait}' not found in SOVLState; skipping state update.",
                            event_type="trait_state_sync_warning"
                        )
                return state
            except Exception as e:
                state.__dict__.update(original_state)
                self.logger.log_error(
                    error_msg=f"Trait sync failed: {str(e)}",
                    error_type="trait_sync_error"
                )
                raise
        self.state_manager.update_state_atomic(update_fn)

    def update_state_after_operation(self, context: str = None, result: dict = None) -> None:
        """
        Enhanced state updates post-operation. Handles operation result and logs adjustments.
        Adds fallback logic if SOVLState.update_after_operation is not available.
        Now also syncs all traits in result['traits'] to state.
        """
        def update_fn(state):
            traits = result.get("traits") if result else None
            if traits:
                for trait, value in traits.items():
                    if hasattr(state, trait):
                        try:
                            setattr(state, trait, value)
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
            used_fallback = False
            if not result and hasattr(state, 'confidence'):
                state.confidence = min(1.0, state.confidence + 0.05)
                used_fallback = True
            if result:
                if "confidence_delta" in result and hasattr(state, "confidence"):
                    state.confidence = max(0.0, min(1.0, state.confidence + result["confidence_delta"]))
                if "temperament_delta" in result and hasattr(state, "temperament_score"):
                    state.temperament_score = max(0.0, min(1.0, state.temperament_score + result["temperament_delta"]))
            if hasattr(state, "update_after_operation"):
                state.update_after_operation(context=context)
            else:
                if not used_fallback:
                    self.logger.log_warning("SOVLState.update_after_operation not available, using default adjustments.")
            return state
        self.state_manager.update_state_atomic(update_fn)
        self.logger.record_event(
            event_type="state_updated_after_operation",
            message=f"State updated after operation: {context}",
            level="info",
            additional_info={
                'context': context,
                'confidence': getattr(self.state_manager.get_state(), 'confidence', None),
                'temperament': getattr(self.state_manager.get_state(), 'temperament_score', None),
                'operation_result': result
            }
        )

    def assemble_full_prompt(
        self,
        user_prompt: str,
        memory_context: str = "",
        base_system_instruction: str = None,
        output_format: str = "text",
        **kwargs
    ) -> str:
        """
        Assemble the final prompt string for the LLM, combining:
        - A base system instruction (default: helpful, emotionally aware assistant)
        - The high-granularity vibe context (from get_vibe_prompt)
        - The memory context (short-term/long-term, passed in)
        - The user's prompt
        Skips any empty sections. Accepts **kwargs for future extensibility.
        Logs the applied vibe prompt and its dimensions for traceability.
        """
        system_instruction = base_system_instruction or "You are a helpful, emotionally aware assistant."
        vibe_context = self.get_vibe_prompt()

        # Log vibe prompt details
        vibe_profile = None
        if self.dialogue_context_manager and hasattr(self.dialogue_context_manager, 'short_term'):
            short_term = getattr(self.dialogue_context_manager, 'short_term', None)
            if short_term and hasattr(short_term, 'get_recent_vibes'):
                recent_vibes = short_term.get_recent_vibes(n=1)
                if recent_vibes:
                    vibe_profile = recent_vibes[-1]
        self.logger.record_event(
            event_type="vibe_prompt_applied",
            message="Vibe prompt applied to LLM",
            level="debug",
            additional_info={
                "vibe_context": vibe_context[:100] + "..." if vibe_context and len(vibe_context) > 100 else vibe_context,
                "vibe_dimensions": vibe_profile.dimensions if vibe_profile else None,
                "output_format": output_format
            }
        )

        prompt_parts = [
            system_instruction,
            f"Current vibe context: {vibe_context}" if vibe_context else "",
            memory_context if memory_context else "",
            user_prompt
        ]
        return "\n\n".join([part for part in prompt_parts if part])