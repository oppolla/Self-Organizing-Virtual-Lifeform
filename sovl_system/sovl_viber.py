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
import copy


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
    """Sculpts conversational vibes as dynamic, empathetic fingerprints. Requires a TemperamentSystem instance."""

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        temperament_system: 'TemperamentSystem',  # Now required, not Optional
    ):
        """Initialize with config, logger, and required temperament_system."""
        if not config_manager or not logger:
            raise ValueError("config_manager and logger cannot be None")
        if temperament_system is None:
            raise ConfigurationError("temperament_system is required for VibeSculptor and cannot be None")
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
            self.repetition_threshold = vibe_config.get("repetition_threshold", 0.8)
            self.repetition_factor_min = vibe_config.get("repetition_factor_min", 0.2)
            self.repetition_factor_max = vibe_config.get("repetition_factor_max", 0.8)
            self.extremity_weight = vibe_config.get("extremity_weight", 0.5)
            self.balance_force = vibe_config.get("balance_force", 0.1)
            self.coupling_factor = vibe_config.get("coupling_factor", 0.05)
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

    def _compute_energy(self, text: str, metadata: Optional[Dict[str, Any]] = None, repetition_factor: float = 1.0, perturbation_scale: float = 0.0, seed: Optional[int] = None) -> Dict[str, float]:
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

        # Apply repetition factor
        sentiment = 0.5 + repetition_factor * (sentiment - 0.5)
        expressiveness = 0.5 + repetition_factor * (expressiveness - 0.5)
        energy_score = (
            0.4 * sentiment +
            0.3 * expressiveness +
            0.2 * diversity +
            0.1 * confidence_adj
        )
        energy_score = max(0.0, min(1.0, energy_score))
        # Apply perturbation
        if seed is not None and perturbation_scale > 0:
            import random
            random.seed(seed)
            perturbation = random.uniform(-perturbation_scale, perturbation_scale)
            energy_score = max(0.0, min(1.0, energy_score + perturbation))
        return {
            "base_energy": energy_score,
            "sentiment": sentiment,
            "expressiveness": expressiveness,
            "lexical_diversity": diversity,
            "confidence_adj": confidence_adj
        }

    def _compute_flow(self, text: str, profile: Dict, metadata: Optional[Dict[str, Any]] = None, repetition_factor: float = 1.0, perturbation_scale: float = 0.0, seed: Optional[int] = None) -> Dict[str, float]:
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

        # Apply repetition factor
        rhythm = 0.5 + repetition_factor * (rhythm - 0.5)
        flow_score = (
            0.5 * syntactic_complexity +
            0.3 * rhythm +
            0.2 * organization
        )
        flow_score = max(0.0, min(1.0, flow_score))
        # Apply perturbation
        if seed is not None and perturbation_scale > 0:
            import random
            random.seed(seed)
            perturbation = random.uniform(-perturbation_scale, perturbation_scale)
            flow_score = max(0.0, min(1.0, flow_score + perturbation))
        return {
            "syntactic_complexity": syntactic_complexity,
            "rhythm_score": rhythm,
            "organization": organization,
            "flow_score": flow_score
        }

    def _compute_resonance(self, text: str, state: SOVLState, metadata: Optional[Dict[str, Any]] = None, repetition_factor: float = 1.0, perturbation_scale: float = 0.0, seed: Optional[int] = None) -> Dict[str, float]:
        """Measure topic consistency and temperament alignment as vibe resonance. Requires temperament_system."""
        topic_consistency = 0.5
        temperament_alignment = 0.5
        coherence = 0.5

        # Use conversation_context instead of user_profile_state
        conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
        context_data = state.get_conversation_context(conversation_id)
        inputs = context_data.get("inputs", deque(maxlen=10))
        word_set_cache = context_data.get("word_set_cache", deque(maxlen=10))
        # Ensure cache is up to date with inputs
        while len(word_set_cache) < len(inputs) and len(word_set_cache) < 10:  # Respect maxlen
            try:
                word_set_cache.append(set(re.findall(r'\w+', inputs[len(word_set_cache)].lower())))
            except (IndexError, AttributeError):
                break
        while len(word_set_cache) > len(inputs) and word_set_cache:
            word_set_cache.popleft()

        text_words = set(re.findall(r'\w+', text.lower()))

        # Only use the last 3 inputs for topic consistency
        num_to_use = 3
        if not inputs or not text_words:
            topic_consistency = 0.5
        else:
            relevant_inputs = list(inputs)[-num_to_use:]
            relevant_word_sets = list(word_set_cache)[-num_to_use:]
            if not relevant_inputs or not relevant_word_sets:
                topic_consistency = 0.5
            else:
                topic_consistency = sum(
                    len(text_words & word_set) / len(text_words | word_set) if word_set else 0.0
                    for word_set in relevant_word_sets
                ) / len(relevant_word_sets)

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
            user_energy = self._compute_energy(text, metadata, repetition_factor, perturbation_scale, seed).get("base_energy", 0.5)
            temperament_score = self.temperament_system.get_temperament_score()
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
        # Apply perturbation
        if seed is not None and perturbation_scale > 0:
            import random
            random.seed(seed)
            perturbation = random.uniform(-perturbation_scale, perturbation_scale)
            resonance_score = max(0.0, min(1.0, resonance_score + perturbation))
        # Persist updated word_set_cache to conversation_context
        context_data["word_set_cache"] = word_set_cache
        state.update_conversation_context(conversation_id, context_data)
        return {
            "topic_consistency": resonance_score,
            "temperament_alignment": temperament_alignment,
            "coherence": coherence
        }

    def _compute_engagement(self, text: str, engagement_manager: Optional[CuriosityManager], metadata: Optional[Dict[str, Any]] = None, repetition_factor: float = 1.0, perturbation_scale: float = 0.0, seed: Optional[int] = None) -> Dict[str, float]:
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

        # Apply repetition factor
        question_intensity = 0.5 + repetition_factor * (question_intensity - 0.5)
        engagement_score = (
            0.4 * question_intensity +
            0.3 * novelty +
            0.3 * diversity
        )
        engagement_score = max(0.0, min(1.0, engagement_score))
        # Apply perturbation
        if seed is not None and perturbation_scale > 0:
            import random
            random.seed(seed)
            perturbation = random.uniform(-perturbation_scale, perturbation_scale)
            engagement_score = max(0.0, min(1.0, engagement_score + perturbation))
        return {
            "question_intensity": question_intensity,
            "novelty": novelty,
            "diversity": diversity,
            "engagement_score": engagement_score
        }

    def _validate_turn_metadata(self, metadata: Optional[dict]) -> bool:
        """Validate that turn_metadata has required structure and keys."""
        required_keys = ["prompt_metrics", "confidence_score", "relationship_context"]
        if not isinstance(metadata, dict):
            return False
        for key in required_keys:
            if key not in metadata:
                return False
        return True

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
        import random
        try:
            # Validate turn_metadata structure before proceeding
            if not self._validate_turn_metadata(turn_metadata):
                self.logger.record_event(
                    event_type="vibe_metadata_invalid",
                    message="Invalid or missing turn_metadata in sculpt_vibe",
                    level="error",
                    additional_info={"turn_metadata": str(turn_metadata)[:200]}
                )
                raise ConfigurationError("Invalid or missing turn_metadata in sculpt_vibe")

            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")
            
            # Avoid in-place modification of input metadata in multi-threaded environments
            if turn_metadata is not None:
                turn_metadata = copy.deepcopy(turn_metadata)

            # Extract conversation ID
            conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
            context_data = state.get_conversation_context(conversation_id)
        
            # Handle timestamps
            now = time.time()
            last_interaction_time = min(context_data.get("last_interaction_time", now), now)
            time_delta = now - last_interaction_time
        
            # Process input for repetition detection
            current_words = set(re.findall(r'\w+', user_input.lower()))
            word_set_cache = context_data.get("word_set_cache", deque(maxlen=10))
            inputs = context_data.get("inputs", deque(maxlen=10))
        
            # Calculate similarity with recent inputs
            similarities = []
            for ws in list(word_set_cache)[-3:]:
                if ws:
                    sim = len(current_words & ws) / len(current_words | ws) if (current_words and ws) else 0.0
                    similarities.append(sim)
            max_similarity = max(similarities) if similarities else 0.0
        
            # Track consecutive repetitions
            consecutive_repetitions = context_data.get("consecutive_repetitions", 0)
            consecutive_repetitions = consecutive_repetitions + 1 if max_similarity > self.repetition_threshold else 0
        
            # Calculate repetition factor
            repetition_factor = self.repetition_factor_max
            if max_similarity > self.repetition_threshold:
                similarity_diff = max_similarity - self.repetition_threshold
                threshold_range = 1.0 - self.repetition_threshold
                factor_range = self.repetition_factor_max - self.repetition_factor_min
                repetition_factor = self.repetition_factor_max - (
                    (factor_range * similarity_diff / threshold_range) * (0.9 ** consecutive_repetitions)
                )
                repetition_factor = max(self.repetition_factor_min, min(self.repetition_factor_max, repetition_factor))
        
            # Calculate decay based on previous vibe and time
            prev_score = self.vibes[-1].overall_score if self.vibes else self.default_vibe_score
            extremity = abs(prev_score - 0.5)
            reset_time = 12 * 60  # 12 minutes in seconds
            raw = min(time_delta / reset_time, 1.0)
        
            def smoothstep(x):
                return 3 * x**2 - 2 * x**3
        
            time_weight = smoothstep(raw)
            decay = self.decay_factor * (1 + self.extremity_weight * extremity * time_weight)
        
            # Adjust decay based on short-term memory
            if short_term_memory:
                recent_vibes = [
                    entry.get('vibe_profile', {}).get('overall_score', 0.5)
                    for entry in short_term_memory
                    if 'vibe_profile' in entry
                ]
                if recent_vibes:
                    average_recent_vibe = sum(recent_vibes) / len(recent_vibes)
                    decay *= (1 + (average_recent_vibe - 0.5))
        
            # Adjust decay for repetition
            if max_similarity > self.repetition_threshold:
                decay *= (1 + 0.2 * (1 - repetition_factor))
        
            # Apply repetition adjustment
            def apply_repetition(val):
                return 0.5 + repetition_factor * (val - 0.5)
        
            # Calculate perturbation and seed
            perturbation_scale = 0.05 * (1 - confidence)
            seed = (int(now * 1000) ^ (hash(user_input[:50]) & 0xFFFFFFFF)) & 0xFFFFFFFF
        
            # Compute component scores
            energy_components = self._compute_energy(user_input, turn_metadata, repetition_factor, perturbation_scale, seed)
            flow_components = self._compute_flow(user_input, context_data, turn_metadata, repetition_factor, perturbation_scale, seed)
            resonance_components = self._compute_resonance(user_input, state, turn_metadata, repetition_factor, perturbation_scale, seed)
            engagement_components = self._compute_engagement(user_input, engagement_manager, turn_metadata, repetition_factor, perturbation_scale, seed)
        
            # Apply repetition to component scores
            energy_components["base_energy"] = apply_repetition(energy_components["base_energy"])
            if "sentiment" in energy_components:
                energy_components["sentiment"] = apply_repetition(energy_components["sentiment"])
            flow_components["rhythm_score"] = apply_repetition(flow_components["rhythm_score"])
            resonance_components["topic_consistency"] = apply_repetition(resonance_components["topic_consistency"])
            engagement_components["engagement_score"] = apply_repetition(engagement_components["engagement_score"])
        
            # Apply coupling between components
            energy_components["base_energy"] *= (1 - self.coupling_factor * max(0, flow_components["rhythm_score"] - 0.5))
            flow_components["rhythm_score"] *= (1 - self.coupling_factor * max(0, resonance_components["topic_consistency"] - 0.5))
            resonance_components["topic_consistency"] *= (1 - self.coupling_factor * max(0, engagement_components["engagement_score"] - 0.5))
            engagement_components["engagement_score"] *= (1 - self.coupling_factor * max(0, energy_components["base_energy"] - 0.5))
        
            # Apply decay to component scores
            decayed_energy = energy_components["base_energy"] * decay
            decayed_flow = flow_components["rhythm_score"] * decay
            decayed_resonance = resonance_components["topic_consistency"] * decay
            decayed_engagement = engagement_components["engagement_score"] * decay
        
            # Handle extreme streak
            extreme_streak = context_data.get("extreme_streak", 0)
            if time_delta > 12 * 60:
                extreme_streak = 0
            elif prev_score > 0.8 or prev_score < 0.2:
                extreme_streak += 1
            else:
                extreme_streak = 0
        
            # Apply homeostatic balancing
            balance_adjust = 0.0
            if extreme_streak >= 3:
                balance_adjust = self.balance_force * (0.5 - prev_score)
                self.logger.record_event(
                    event_type="vibe_balancing",
                    message="Homeostatic balancing applied",
                    level="info",
                    additional_info={"prev_score": prev_score, "balance_adjust": balance_adjust, "extreme_streak": extreme_streak}
                )
        
            # Adjust confidence
            if turn_metadata:
                confidence += 0.1
                word_count = turn_metadata.get("prompt_metrics", {}).get("content_metrics", {}).get("word_count", 0)
                confidence += 0.1 * (turn_metadata.get("confidence_score", 0.5) - 0.5)
                if word_count < 3:
                    confidence -= 0.2
                elif word_count > 50:
                    confidence += 0.1
            confidence = max(0.1, min(1.0, confidence))
        
            # Calculate overall vibe score
            overall_score = (
                self.weights["energy"] * decayed_energy +
                self.weights["flow"] * decayed_flow +
                self.weights["resonance"] * decayed_resonance +
                self.weights["engagement"] * decayed_engagement
            ) + balance_adjust
            overall_score = max(self.min_vibe, min(self.max_vibe, overall_score))
        
            # Log repetition detection
            if max_similarity > self.repetition_threshold:
                self.logger.record_event(
                    event_type="repetition_detected",
                    message="Repetition detected in input",
                    level="info",
                    additional_info={
                        "similarity": max_similarity,
                        "consecutive_repetitions": consecutive_repetitions,
                        "repetition_factor": repetition_factor
                    }
                )
        
            # Update context data
            inputs.append(user_input)
            word_set_cache.append(current_words)
            context_data.update({
                "last_interaction_time": now,
                "inputs": inputs,
                "word_set_cache": word_set_cache,
                "consecutive_repetitions": consecutive_repetitions,
                "extreme_streak": extreme_streak
            })
        
            # Combine all dimensions
            all_dimensions = {
                **{f"energy_{k}": v for k, v in energy_components.items()},
                **{f"flow_{k}": v for k, v in flow_components.items()},
                **{f"resonance_{k}": v for k, v in resonance_components.items()},
                **{f"engagement_{k}": v for k, v in engagement_components.items()}
            }
        
            # Calculate intensity
            intensity = (decayed_energy + decayed_flow + decayed_resonance + decayed_engagement) / 4.0
            intensity = max(0.0, min(1.0, intensity))
        
            # Extract salient phrases
            salient_phrases = []
            if turn_metadata and turn_metadata.get("prompt_metrics", {}).get("token_stats", {}):
                words = re.findall(r'\w+', user_input.lower())
                pos_words = {'good', 'great', 'happy', 'awesome', 'love'}
                for word in words:
                    if word in pos_words:
                        salient_phrases.append((word, {"joy": 0.9}))
        
            # Create and store vibe profile
            current_profile = VibeProfile(
                overall_score=overall_score,
                dimensions=all_dimensions,
                intensity=intensity,
                confidence=confidence,
                salient_phrases=salient_phrases,
                timestamp_unix=now
            )
            self.vibes.append(current_profile)
        
            # Update turn metadata
            if turn_metadata is not None:
                turn_metadata["tone_boost"] = {"positive": 0.5 * energy_components["base_energy"]}
                turn_metadata["max_length"] = 50 * (0.5 + flow_components["rhythm_score"])
                if engagement_manager:
                    engagement_manager.update_pressure(engagement_components["engagement_score"] * 0.05)
        
            # Log vibe profile creation
            self.logger.record_event(
                event_type="vibe_sculpted",
                message="Vibe profile sculpted",
                level="info",
                additional_info={
                    "profile": vars(current_profile),
                    "conversation_id": conversation_id,
                    "repetition_factor": repetition_factor,
                    "perturbation": perturbation_scale,
                    "decay": decay,
                    "balance_adjust": balance_adjust
                }
            )
            # Persist context changes
            state.update_conversation_context(conversation_id, context_data)
            return current_profile
        except Exception as e:
            error_manager.record_error(
                "vibe_sculpt_error",
                f"Vibe sculpting failed: {str(e)}",
                severity=3,
                context={"user_input": user_input[:50]}
            )
            return self._get_default_vibe_profile()

    @synchronized()
    def predict_vibe_shift(self, current_vibe_profile: VibeProfile) -> bool:
        """Predict if a vibe shift is occurring based on recent trends. Thread-safe access to self.vibes."""
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

    @synchronized()
    def get_vibe_aura(self, state: SOVLState) -> Dict[str, Any]:
        """Generate a vibe 'aura' for visualization, using VibeProfile. Thread-safe access to self.vibes."""
        conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id") if hasattr(state, 'history') else "unknown_conv_id"
        profile_data = state.get_conversation_context(conversation_id)
        
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