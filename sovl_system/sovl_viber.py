import time
import re
from collections import deque
from typing import Optional, Dict, Any, TYPE_CHECKING, List, Tuple
from dataclasses import dataclass
from sovl_processor import MetadataProcessor
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_state import SOVLState, StateManager
from sovl_error import ErrorManager, ConfigurationError, ConfigurationError
from sovl_main import SystemContext
from sovl_utils import synchronized
from sovl_curiosity import CuriosityManager
from sovl_temperament import TemperamentSystem


@dataclass
class VibeProfile:
    """Represents a detailed snapshot of the conversational vibe."""
    overall_score: float
    dimensions: Dict[str, float]
    intensity: float
    confidence: float
    salient_phrases: List[Tuple[str, Dict[str, float]]]  # e.g., [("really great", {"joy": 0.9})]
    timestamp_unix: float

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "dimensions": self.dimensions,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "salient_phrases": self.salient_phrases,
            "timestamp_unix": self.timestamp_unix,
        }

    @staticmethod
    def from_dict(d: dict) -> 'VibeProfile':
        return VibeProfile(
            overall_score=d["overall_score"],
            dimensions=d["dimensions"],
            intensity=d["intensity"],
            confidence=d["confidence"],
            salient_phrases=d["salient_phrases"],
            timestamp_unix=d["timestamp_unix"],
        )

class VibeSculptor:
    """Sculpts conversational vibes as dynamic, empathetic fingerprints."""

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        temperament_system: Optional['TemperamentSystem'] = None,
    ):
        """Initialize with config, logger, and optional system components."""
        if not config_manager or not logger:
            raise ValueError("config_manager and logger cannot be None")
        self.config_manager = config_manager
        self.logger = logger
        self.temperament_system = temperament_system
        self.vibes: deque[VibeProfile] = deque(maxlen=self._get_config("history_maxlen", 20))
        self._load_config()

    def _load_config(self) -> None:
        """Load vibe configuration elegantly."""
        try:
            vibe_config = self.config_manager.get_section("vibe_config", {})
            self.default_vibe_score = vibe_config.get("default_vibe_score", 0.5)
            self.min_vibe = vibe_config.get("min_vibe_score", 0.0)
            self.max_vibe = vibe_config.get("max_vibe_score", 1.0)
            self.switch_threshold = vibe_config.get("switch_threshold", 0.3)
            self.decay_factor = vibe_config.get("decay_factor", 0.9)
            self.weights = {
                "energy": vibe_config.get("energy_weight", 0.25),
                "flow": vibe_config.get("flow_weight", 0.25),
                "resonance": vibe_config.get("resonance_weight", 0.25),
                "engagement": vibe_config.get("engagement_weight", 0.25)
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
            raise ConfigurationError(f"Vibe config failed: {str(e)}")

    def _get_config(self, key: str, default: Any) -> Any:
        """Helper to get vibe config values."""
        return self.config_manager.get(f"vibe_config.{key}", default)

    def _get_default_vibe_profile(self) -> VibeProfile:
        """Returns a default VibeProfile."""
        return VibeProfile(
            overall_score=self.default_vibe_score,
            dimensions={"default_energy": 0.5, "default_flow": 0.5, "default_resonance": 0.5, "default_engagement": 0.5},
            intensity=0.5,
            confidence=0.1,
            salient_phrases=[],
            timestamp_unix=time.time()
        )

    def _compute_energy(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure lexical diversity, sentiment, and expressiveness as conversational energy."""
        words = re.findall(r'\w+', text.lower())
        base_energy = 0.5
        expressiveness = 0.0
        diversity = 0.5
        confidence_adj = 0.0

        # Sentiment (simplified, replace with model if available)
        pos_words = {'good', 'great', 'happy', 'awesome', 'love'}
        neg_words = {'bad', 'sad', 'hate', 'terrible', 'awful'}
        pos_count = len(set(words) & pos_words)
        neg_count = len(set(words) & neg_words)
        sentiment = pos_count / (pos_count + neg_count) if pos_count + neg_count else 0.5

        if metadata:
            quality_metrics = metadata.get("prompt_metrics", {}).get("quality_metrics", {})
            token_stats = metadata.get("prompt_metrics", {}).get("token_stats", {})
            content_metrics = metadata.get("prompt_metrics", {}).get("content_metrics", {})
            confidence_score = metadata.get("confidence_score", 0.5)

            # Expressiveness
            expressiveness = (
                0.4 * quality_metrics.get("has_exclamation", 0) +
                0.3 * quality_metrics.get("has_emoji", 0) +
                0.3 * quality_metrics.get("has_question", 0)
            )

            # Lexical diversity
            diversity = (
                0.6 * token_stats.get("basic_stats", {}).get("token_diversity", 0.5) +
                0.4 * token_stats.get("special_token_stats", {}).get("special_token_ratio", 0.0)
            )

            # Confidence adjustment
            confidence_adj = confidence_score - 0.5  # Center around 0

            # Adjust sentiment with metadata
            sentiment = 0.7 * sentiment + 0.3 * (content_metrics.get("avg_word_length", 5) / 10)

        energy_score = (
            0.4 * sentiment +
            0.3 * expressiveness +
            0.2 * diversity +
            0.1 * confidence_adj
        )
        energy_score = max(0.0, min(1.0, energy_score))

        return {
            "base_energy": energy_score,
            "sentiment": sentiment,
            "expressiveness": expressiveness,
            "lexical_diversity": diversity,
            "confidence_adj": confidence_adj
        }

    def _compute_flow(self, text: str, profile: Dict, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure syntactic complexity and interaction rhythm as conversational flow."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        syntactic_complexity = 0.5
        rhythm = 0.5
        organization = 0.5

        if metadata:
            content_metrics = metadata.get("prompt_metrics", {}).get("content_metrics", {})
            structure_metrics = metadata.get("prompt_metrics", {}).get("structure", {})
            temporal_tracking = metadata.get("relationship_context", {}).get("temporal_tracking", {})
            conversation_tracking = metadata.get("relationship_context", {}).get("conversation_tracking", {})

            # Syntactic complexity
            avg_sentence_length = content_metrics.get("avg_sentence_length", 0)
            sentence_count = content_metrics.get("sentence_count", 1)
            syntactic_complexity = min(avg_sentence_length / 20.0, 1.0) * 0.6 + min(sentence_count / 50.0, 1.0) * 0.4

            # Rhythm
            elapsed_time_ms = temporal_tracking.get("elapsed_time", 0)
            message_index = conversation_tracking.get("message_index", 1)
            rhythm = (
                0.5 * (1.0 - min(elapsed_time_ms / 5000.0, 1.0)) +  # Faster responses = higher rhythm
                0.5 * min(message_index / 50.0, 1.0)  # Later messages = sustained engagement
            )

            # Organization
            organization = (
                0.7 * (1.0 - structure_metrics.get("whitespace_ratio", 0.0)) +  # Less whitespace = more organized
                0.3 * (1.0 - min(structure_metrics.get("blank_line_count", 0) / 10.0, 1.0))
            )

        flow_score = (
            0.5 * syntactic_complexity +
            0.3 * rhythm +
            0.2 * organization
        )
        flow_score = max(0.0, min(1.0, flow_score))

        return {
            "syntactic_complexity": syntactic_complexity,
            "rhythm_score": rhythm,
            "organization": organization
        }

    def _compute_resonance(self, text: str, state: SOVLState, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure topic consistency and temperament alignment as vibe resonance."""
        topic_consistency = 0.5
        temperament_alignment = 0.5
        coherence = 0.5

        profile = state.user_profile_state.get(state.history.conversation_id, {})
        inputs = profile.get("inputs", deque(maxlen=10))
        text_words = set(re.findall(r'\w+', text.lower()))

        if inputs and text_words:
            topic_consistency = sum(
                len(text_words & set(re.findall(r'\w+', h.lower()))) /
                len(text_words | set(re.findall(r'\w+', h.lower()))) if h else 0.0
                for h in inputs
            ) / len(inputs)

        if metadata:
            reference_tracking = metadata.get("relationship_context", {}).get("reference_tracking", {})
            token_stats = metadata.get("prompt_metrics", {}).get("token_stats", {})
            structure_metrics = metadata.get("prompt_metrics", {}).get("structure", {})
            confidence_score = metadata.get("confidence_score", 0.5)

            # Adjust topic consistency
            if reference_tracking.get("references") or reference_tracking.get("parent_message_id"):
                topic_consistency += 0.15
            topic_consistency = 0.7 * topic_consistency + 0.3 * token_stats.get("pattern_stats", {}).get("bigram_diversity", 0.5)

            # Temperament alignment
            user_energy = self._compute_energy(text, metadata).get("base_energy", 0.5)
            temperament_score = self.temperament_system.get_temperament_score() if self.temperament_system else 0.5
            temperament_alignment = 1.0 - abs(temperament_score - user_energy)

            # Coherence
            coherence = (
                0.6 * confidence_score +
                0.4 * min(structure_metrics.get("indentation_levels", 1) / 5.0, 1.0)
            )

        resonance_score = (
            0.4 * topic_consistency +
            0.3 * temperament_alignment +
            0.3 * coherence
        )
        resonance_score = max(0.0, min(1.0, resonance_score))

        return {
            "topic_consistency": topic_consistency,
            "temperament_alignment": temperament_alignment,
            "coherence": coherence
        }

    def _compute_engagement(self, text: str, engagement_manager: Optional[CuriosityManager], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure engagement with engagement-driven questions and novelty."""
        question_intensity = 0.5
        novelty = 0.5
        diversity = 0.5

        if metadata:
            quality_metrics = metadata.get("prompt_metrics", {}).get("quality_metrics", {})
            token_stats = metadata.get("prompt_metrics", {}).get("token_stats", {})
            conversation_tracking = metadata.get("relationship_context", {}).get("conversation_tracking", {})
            novelty_score = metadata.get("novelty_score", 0.5)

            # Question intensity
            question_words = len(re.findall(r'\b(what|how|why|where|when|who)\b', text, re.I))
            word_count = metadata.get("prompt_metrics", {}).get("content_metrics", {}).get("word_count", 1)
            question_intensity = (
                0.6 * quality_metrics.get("has_question", 0) +
                0.4 * min(question_words / (word_count or 1), 1.0)
            )

            # Novelty
            novelty = novelty_score if engagement_manager else 0.7 * novelty_score + 0.3 * min(conversation_tracking.get("thread_depth", 1) / 10.0, 1.0)

            # Diversity
            diversity = token_stats.get("pattern_stats", {}).get("trigram_diversity", 0.5)

        engagement_score = (
            0.4 * question_intensity +
            0.3 * novelty +
            0.3 * diversity
        )
        engagement_score = max(0.0, min(1.0, engagement_score))

        return {
            "question_intensity": question_intensity,
            "novelty": novelty,
            "diversity": diversity,
            "engagement_score": engagement_score
        }

    @synchronized()
    def sculpt_vibe(
        self,
        user_input: str,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        engagement_manager: Optional[CuriosityManager] = None,
        turn_metadata: Optional[Dict[str, Any]] = None,
        short_term_memory: Optional[List[Dict[str, Any]]] = None
    ) -> VibeProfile:
        """Sculpt a vibe score that resonates with user and system energy."""
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")
            
            conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
            profile_data = state.user_profile_state.get(conversation_id, {})
            now = time.time()
            last_interaction_time = profile_data.get("last_interaction", now)
            if last_interaction_time > now:
                last_interaction_time = now
            decay = self.decay_factor ** ((now - last_interaction_time) / 86400.0)

            # Adjust vibe calculation based on short-term memory context
            if short_term_memory:
                recent_vibes = [entry.get('vibe_profile', {}).get('overall_score', 0.5) for entry in short_term_memory if 'vibe_profile' in entry]
                if recent_vibes:
                    average_recent_vibe = sum(recent_vibes) / len(recent_vibes)
                    decay *= (1 + (average_recent_vibe - 0.5))  # Example adjustment

            energy_components = self._compute_energy(user_input, turn_metadata)
            flow_components = self._compute_flow(user_input, profile_data, turn_metadata)
            resonance_components = self._compute_resonance(user_input, state, turn_metadata)
            engagement_components = self._compute_engagement(user_input, engagement_manager, turn_metadata)

            decayed_energy = energy_components["base_energy"] * decay
            decayed_flow = flow_components["rhythm_score"] * decay
            decayed_resonance = resonance_components["topic_consistency"] * decay
            decayed_engagement = engagement_components["engagement_score"] * decay
            
            overall_score = (
                self.weights["energy"] * decayed_energy +
                self.weights["flow"] * decayed_flow +
                self.weights["resonance"] * decayed_resonance +
                self.weights["engagement"] * decayed_engagement
            )
            overall_score = max(self.min_vibe, min(self.max_vibe, overall_score))

            all_dimensions = {
                **{f"energy_{k}": v for k, v in energy_components.items()},
                **{f"flow_{k}": v for k, v in flow_components.items()},
                **{f"resonance_{k}": v for k, v in resonance_components.items()},
                **{f"engagement_{k}": v for k, v in engagement_components.items()}
            }

            intensity = (decayed_energy + decayed_flow + decayed_resonance + decayed_engagement) / 4.0
            intensity = max(0.0, min(1.0, intensity))

            confidence = 0.6
            if turn_metadata:
                confidence += 0.1
                word_count = turn_metadata.get("prompt_metrics", {}).get("content_metrics", {}).get("word_count", 0)
                confidence += 0.1 * (turn_metadata.get("confidence_score", 0.5) - 0.5)
                if word_count < 3:
                    confidence -= 0.2
                elif word_count > 50:
                    confidence += 0.1
            confidence = max(0.1, min(1.0, confidence))

            # Add salient phrases (simplified)
            salient_phrases = []
            if turn_metadata and turn_metadata.get("prompt_metrics", {}).get("token_stats", {}):
                words = re.findall(r'\w+', user_input.lower())
                pos_words = {'good', 'great', 'happy', 'awesome', 'love'}
                for word in words:
                    if word in pos_words:
                        salient_phrases.append((word, {"joy": 0.9}))

            current_profile = VibeProfile(
                overall_score=overall_score,
                dimensions=all_dimensions,
                intensity=intensity,
                confidence=confidence,
                salient_phrases=salient_phrases,
                timestamp_unix=now
            )
            self.vibes.append(current_profile)

            # Add sculpting hints to metadata
            if turn_metadata is not None:
                turn_metadata["tone_boost"] = {"positive": 0.5 * energy_components["base_energy"]}
                turn_metadata["max_length"] = 50 * (0.5 + flow_components["rhythm_score"])
                if engagement_manager:
                    engagement_manager.update_pressure(engagement_components["engagement_score"] * 0.05)

            self.logger.record_event(
                event_type="vibe_sculpted",
                message="Vibe profile sculpted",
                level="info",
                additional_info={
                    "profile": vars(current_profile),
                    "conversation_id": conversation_id
                }
            )
            return current_profile

        except Exception as e:
            error_manager.record_error(
                "vibe_sculpt_error",
                f"Vibe sculpting failed: {str(e)}",
                severity=3,
                context={"user_input": user_input[:50]}
            )
            return self._get_default_vibe_profile()

    def predict_vibe_shift(self, current_vibe_profile: VibeProfile) -> bool:
        """Predict if a vibe shift is occurring based on recent trends."""
        if len(self.vibes) < 3:
            return False
        recent_overall_scores = [vp.overall_score for vp in list(self.vibes)[-3:-1]]
        if not recent_overall_scores:
            return False

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

    def get_vibe_aura(self, state: SOVLState) -> Dict[str, Any]:
        """Generate a vibe 'aura' for visualization, using VibeProfile."""
        profile_data = state.user_profile_state.get(state.history.conversation_id, {})
        
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
                trend = (self.vibes[-1].overall_score - self.vibes[0].overall_score) / (len(self.vibes) - 1) if len(self.vibes) > 1 else 0.0
                aura["trend_overall_score"] = trend
            return aura
        else:
            default_aura_profile = self._get_default_vibe_profile()
            return {
                "latest_overall_score": default_aura_profile.overall_score,
                "latest_intensity": default_aura_profile.intensity,
                "latest_confidence": default_aura_profile.confidence,
                "latest_dimensions": default_aura_profile.dimensions,
                "history_length": 0,
                "trend_overall_score": 0.0
            }