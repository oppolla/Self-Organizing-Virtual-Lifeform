import time
import re
from collections import deque
from typing import Optional, Dict, Any, TYPE_CHECKING, List, Tuple
from dataclasses import dataclass
from sovl_processor import MetadataProcessor
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_state import SOVLState, StateManager
from sovl_error import ErrorManager, ConfigurationError
# TODO: Resolve StateError. For example:
# If StateError is in sovl_error: from sovl_error import StateError (assuming it's a ConfigurationError or similar)
# Or define it: class StateError(Exception): pass (using a generic base for now)
class StateError(ConfigurationError):
    pass

from sovl_confidence import CuriosityManager, SystemContext
from sovl_utils import synchronized

# For full static type checking of TemperamentSystem and LifecycleManager,
# you would import them, typically like this:
# if TYPE_CHECKING:
#     from sovl_temperament import TemperamentSystem  # Adjust path if necessary
#     from sovl_lifecycle import LifecycleManager    # Adjust path if necessary

@dataclass
class VibeProfile:
    """Represents a detailed snapshot of the conversational vibe."""
    overall_score: float
    dimensions: Dict[str, float]
    intensity: float
    confidence: float
    salient_phrases: List[Tuple[str, Dict[str, float]]] # e.g., [("really great", {"joy": 0.9})]
    timestamp: float

class VibeSculptor:
    """Sculpts conversational vibes as dynamic, empathetic fingerprints."""

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        temperament_system: Optional['TemperamentSystem'] = None,
        lifecycle_manager: Optional['LifecycleManager'] = None
    ):
        """Initialize with config, logger, and optional system components."""
        if not config_manager or not logger:
            raise ValueError("config_manager and logger cannot be None")
        self.config_manager = config_manager
        self.logger = logger
        self.temperament_system = temperament_system
        self.lifecycle_manager = lifecycle_manager
        # self.vibes stores VibeProfile objects now
        self.vibes: deque[VibeProfile] = deque(maxlen=self._get_config("history_maxlen", 20))
        self._load_config()

    def _load_config(self) -> None:
        """Load vibe configuration elegantly."""
        try:
            vibe_config = self.config_manager.get_section("vibe_config", {})
            self.default_vibe_score = vibe_config.get("default_vibe_score", 0.5) # Renamed for clarity
            self.min_vibe = vibe_config.get("min_vibe_score", 0.0)
            self.max_vibe = vibe_config.get("max_vibe_score", 1.0)
            self.switch_threshold = vibe_config.get("switch_threshold", 0.3)
            self.decay_factor = vibe_config.get("decay_factor", 0.9)
            self.weights = {
                "energy": vibe_config.get("energy_weight", 0.25),
                "flow": vibe_config.get("flow_weight", 0.25),
                "resonance": vibe_config.get("resonance_weight", 0.25),
                "curiosity": vibe_config.get("curiosity_weight", 0.25)
            }
            if abs(sum(self.weights.values()) - 1.0) > 1e-6:
                raise ValueError("Vibe weights must sum to 1.0")
            self.logger.record_event(
                event_type="vibe_config_loaded",
                message="Vibe sculptor configured",
                level="info",
                additional_info={"weights": self.weights}
            )
        except Exception as e:
            self.logger.record_event(
                event_type="vibe_config_failed",
                message=f"Vibe config failed: {str(e)}",
                level="error"
            )
            raise StateError(f"Vibe config failed: {str(e)}")

    def _get_config(self, key: str, default: Any) -> Any:
        """Helper to get vibe config values."""
        return self.config_manager.get(f"vibe_config.{key}", default)

    def _get_default_vibe_profile(self) -> VibeProfile:
        """Returns a default VibeProfile."""
        return VibeProfile(
            overall_score=self.default_vibe_score,
            dimensions={"default_energy": 0.5, "default_flow": 0.5, "default_resonance": 0.5, "default_curiosity": 0.5},
            intensity=0.5,
            confidence=0.1, # Low confidence for default
            salient_phrases=[],
            timestamp=time.time()
        )

    def _compute_energy(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure lexical diversity and sentiment as conversational energy, enhanced with metadata."""
        words = re.findall(r'\w+', text.lower())
        base_energy_score = 0.5
        expressiveness_adj = 0.0

        if words:
            ttr = len(set(words)) / len(words)
            pos_words = {'good', 'great', 'happy', 'awesome', 'love'}
            neg_words = {'bad', 'sad', 'hate', 'terrible', 'awful'}
            pos_count = len(set(words) & pos_words)
            neg_count = len(set(words) & neg_words)
            sentiment = pos_count / (pos_count + neg_count) if pos_count + neg_count else 0.5
            base_energy_score = 0.6 * ttr + 0.4 * sentiment
        
        if metadata:
            quality_metrics = metadata.get("quality_metrics", {})
            if quality_metrics.get("has_exclamation"):
                expressiveness_adj += 0.1
            if quality_metrics.get("has_emoji"):
                expressiveness_adj += 0.15
            # New: Adjust sentiment based on historical positivity/negativity
            historical_sentiment = metadata.get("historical_sentiment", 0.5)
            base_energy_score = 0.7 * base_energy_score + 0.3 * historical_sentiment
        
        # Ensure scores are within a reasonable range, e.g., 0-1
        base_energy_score = max(0.0, min(1.0, base_energy_score))
        expressiveness_adj = max(0.0, min(1.0, expressiveness_adj))

        return {"base_energy": base_energy_score, "expressiveness_adj": expressiveness_adj}

    def _compute_flow(self, text: str, profile: Dict, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure syntactic complexity and interaction rhythm as conversational flow."""
        # Syntactic complexity (sentence length)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        avg_length = sum(len(re.findall(r'\w+', s)) for s in sentences) / len(sentences) if sentences else 0.0
        syntax = min(avg_length / 20.0, 1.0)
        
        # Rhythm (input frequency and length)
        inputs = profile.get("inputs", deque(maxlen=10)) # Assuming profile.inputs is a deque
        rhythm_base = min(len(inputs) / 10.0, 1.0) * 0.5 + min(sum(len(i) for i in inputs) / (200.0 * len(inputs) or 1), 1.0) * 0.5
        
        rhythm_adj = 0.0
        if metadata:
            relationship_context = metadata.get("relationship_context", {})
            temporal_tracking = relationship_context.get("temporal_tracking", {})
            elapsed_time_ms = temporal_tracking.get("elapsed_time_ms", 0) # Assuming this key
            if elapsed_time_ms > 5000: # e.g., more than 5 seconds is slow
                rhythm_adj -= 0.1
            elif elapsed_time_ms < 1000 and elapsed_time_ms > 0: # e.g., less than 1 second is fast
                rhythm_adj += 0.1
        
        rhythm_final = max(0.0, min(1.0, rhythm_base + rhythm_adj))
        return {"syntactic_complexity": syntax, "rhythm_score": rhythm_final}

    def _compute_resonance(self, text: str, state: SOVLState, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure topic consistency and temperament alignment as vibe resonance."""
        profile = state.user_profile_state.get(state.history.conversation_id)
        inputs = profile.get("inputs", deque(maxlen=10))
        text_words = set(re.findall(r'\w+', text.lower()))
        
        topic_consistency_base = 0.5
        if inputs and text_words:
            topic_consistency_base = sum(
                len(text_words & set(re.findall(r'\w+', h.lower()))) /
                len(text_words | set(re.findall(r'\w+', h.lower()))) if h else 0.0 # Ensure h is not empty
                for h in inputs
            ) / len(inputs)
        
        topic_consistency_adj = 0.0
        if metadata:
            relationship_context = metadata.get("relationship_context", {})
            reference_tracking = relationship_context.get("reference_tracking", {})
            if reference_tracking.get("references") or reference_tracking.get("parent_message_id"):
                topic_consistency_adj += 0.15
        
        final_topic_consistency = max(0.0, min(1.0, topic_consistency_base + topic_consistency_adj))

        # Temperament alignment (proxy for user mood with _compute_energy - simplified for Phase 1)
        # For Phase 1, _compute_energy doesn't take metadata to avoid circular dependency if called here
        # This will be refined in Phase 2
        user_mood_proxy = self._compute_energy(text).get("base_energy", 0.5) 
        temperament_score = self.temperament_system.get_temperament_score() if self.temperament_system else 0.5
        alignment = 1.0 - abs(temperament_score - user_mood_proxy)
        
        return {"topic_consistency": final_topic_consistency, "temperament_alignment": alignment}

    def _compute_curiosity(self, text: str, curiosity_manager: Optional[CuriosityManager], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure engagement with curiosity-driven questions."""
        base_curiosity_score = 0.5
        if curiosity_manager:
            novelty = curiosity_manager.get_novelty_score(text)
            base_curiosity_score = min(novelty / 0.7, 1.0)

        question_detected_boost = 0.0
        if metadata:
            quality_metrics = metadata.get("quality_metrics", {})
            if quality_metrics.get("has_question"):
                question_detected_boost += 0.2
        
        final_curiosity_score = max(0.0, min(1.0, base_curiosity_score + question_detected_boost))
        return {"curiosity_score": final_curiosity_score}

    @synchronized()
    def sculpt_vibe(
        self,
        user_input: str,
        state: SOVLState,
        error_manager: ErrorManager, # ErrorManager not explicitly used in this version of sculpt_vibe
        context: SystemContext, # SystemContext not explicitly used in this version of sculpt_vibe
        curiosity_manager: Optional[CuriosityManager] = None,
        turn_metadata: Optional[Dict[str, Any]] = None # New parameter
    ) -> VibeProfile: # Return VibeProfile
        """Sculpt a vibe score that resonates with user and system energy."""
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")
            
            conversation_id = getattr(getattr(state, 'history', {}), 'conversation_id', "unknown_conv_id")
            profile_data = getattr(state, 'user_profile_state', {}).get(conversation_id, {})
            # User profile update logic might be handled by SOVLProcessor before calling this
            # state.user_profile_state.update(conversation_id, user_input, state.session_start)

            now = time.time()
            last_interaction_time = profile_data.get("last_interaction", now)
            # Ensure last_interaction_time is not in the future, can happen if profile is not updated yet
            if last_interaction_time > now : last_interaction_time = now 
            decay = self.decay_factor ** ((now - last_interaction_time) / 86400.0) # Daily decay

            energy_components = self._compute_energy(user_input, turn_metadata)
            flow_components = self._compute_flow(user_input, profile_data, turn_metadata)
            # Pass turn_metadata to _compute_resonance, _compute_curiosity if they use it
            resonance_components = self._compute_resonance(user_input, state, turn_metadata)
            curiosity_components = self._compute_curiosity(user_input, curiosity_manager, turn_metadata)

            # Apply decay to the primary scores of each component
            decayed_energy = energy_components.get("base_energy", 0.5) * decay
            decayed_flow = flow_components.get("rhythm_score", 0.5) * decay # Using rhythm_score as primary flow for weighting
            decayed_resonance = resonance_components.get("topic_consistency", 0.5) * decay # Using topic_consistency as primary
            decayed_curiosity = curiosity_components.get("curiosity_score", 0.5) * decay
            
            lifecycle_factor = self.lifecycle_manager.get_lifecycle_factor() if self.lifecycle_manager else 1.0
            
            overall_score = (
                self.weights["energy"] * decayed_energy +
                self.weights["flow"] * decayed_flow +
                self.weights["resonance"] * decayed_resonance +
                self.weights["curiosity"] * decayed_curiosity
            ) * lifecycle_factor
            overall_score = max(self.min_vibe, min(self.max_vibe, overall_score))

            # Combine all dimensional outputs
            all_dimensions = {
                **{f"energy_{k}": v for k,v in energy_components.items()},
                **{f"flow_{k}": v for k,v in flow_components.items()},
                **{f"resonance_{k}": v for k,v in resonance_components.items()},
                **{f"curiosity_{k}": v for k,v in curiosity_components.items()}
            }
            
            # Phase 1: Simple intensity and confidence
            # Intensity could be average of primary decayed scores, or max
            intensity = (decayed_energy + decayed_flow + decayed_resonance + decayed_curiosity) / 4.0 
            intensity = max(0.0, min(1.0, intensity))

            confidence = 0.6 # Base confidence for Phase 1
            if turn_metadata:
                confidence += 0.1
                content_metrics = turn_metadata.get("content_metrics", {})
                word_count = content_metrics.get("word_count", 0)
                if word_count < 3: # Very short input
                    confidence -= 0.2
                elif word_count > 50: # Very long input, potentially more complex
                     confidence += 0.1
            confidence = max(0.1, min(1.0, confidence)) # Ensure confidence is within [0.1, 1.0]


            current_profile = VibeProfile(
                overall_score=overall_score,
                dimensions=all_dimensions,
                intensity=intensity,
                confidence=confidence,
                salient_phrases=[], # Empty for Phase 1
                timestamp=now
            )
            self.vibes.append(current_profile)

            self.logger.record_event(
                event_type="vibe_sculpted",
                message="Vibe profile sculpted",
                level="info",
                additional_info={
                    "profile": current_profile, # Log the full profile
                    "conversation_id": conversation_id
                }
            )
            return current_profile

        except Exception as e:
            self.logger.record_event(
                event_type="vibe_sculpt_failed",
                message=f"Vibe sculpting failed: {str(e)}",
                level="error",
                # Add more context if possible, e.g., user_input snippet safely
            )
            # Optionally use error_manager if passed and relevant here
            # error_manager.handle_data_error(e, {"user_input": user_input[:50]}, conversation_id)
            return self._get_default_vibe_profile()

    def predict_vibe_shift(self, current_vibe_profile: VibeProfile) -> bool: # Takes VibeProfile
        """Predict if a vibe shift is occurring based on recent trends."""
        if len(self.vibes) < 3: # Need at least 2 previous + current to compare
            return False
        # Compare overall_score for simplicity in Phase 1
        recent_overall_scores = [vp.overall_score for vp in list(self.vibes)[-3:-1]] # Last two before current
        if not recent_overall_scores: return False

        avg_recent_score = sum(recent_overall_scores) / len(recent_overall_scores)
        deviation = abs(current_vibe_profile.overall_score - avg_recent_score)
        
        if deviation > self.switch_threshold:
            self.logger.record_event(
                event_type="vibe_shift_detected",
                message="Potential vibe shift detected (overall_score)",
                level="warning",
                additional_info={"current_score": current_vibe_profile.overall_score, "avg_recent_score": avg_recent_score, "deviation": deviation}
            )
            return True
        return False

    def get_vibe_aura(self, state: SOVLState) -> Dict[str, Any]: # Return type updated
        """Generate a vibe 'aura' for visualization, using VibeProfile."""
        profile_data = state.user_profile_state.get(state.history.conversation_id, {})
        
        # For aura, we might want to represent the latest full vibe profile or an aggregation
        # This is a simplified version for Phase 1
        if self.vibes:
            latest_profile = self.vibes[-1]
            aura = {
                "latest_overall_score": latest_profile.overall_score,
                "latest_intensity": latest_profile.intensity,
                "latest_confidence": latest_profile.confidence,
                "latest_dimensions": latest_profile.dimensions,
                "history_length": len(self.vibes)
            }
            if len(self.vibes) > 1:
                trend = (self.vibes[-1].overall_score - self.vibes[0].overall_score) / (len(self.vibes) -1) if len(self.vibes) > 1 else 0.0
                aura["trend_overall_score"] = trend
            return aura
        else:
            # Fallback if no vibes have been sculpted yet
            default_aura_profile = self._get_default_vibe_profile()
            return {
                "latest_overall_score": default_aura_profile.overall_score,
                "latest_intensity": default_aura_profile.intensity,
                "latest_confidence": default_aura_profile.confidence,
                "latest_dimensions": default_aura_profile.dimensions,
                "history_length": 0,
                "trend_overall_score": 0.0
            }
