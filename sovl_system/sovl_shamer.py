import time
import re
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from sovl_processor import MetadataProcessor
from sovl_config import ConfigManager
from sovl_queue import capture_scribe_event
from sovl_logger import Logger
from sovl_state import StateManager
from sovl_error import ErrorManager
from sovl_utils import synchronized
from sovl_viber import VibeSculptor
from sovl_recaller import DialogueContextManager

@dataclass
class ShameProfile:
    """Represents a snapshot of a shameful or traumatic interaction."""
    frustration_score: float  # Intensity of frustration/anger (0.0 to 1.0)
    trauma_potential: float  # Likelihood of trauma trigger (0.0 to 1.0)
    triggers: List[Tuple[str, float]]  # Phrases or topics with emotional weight
    context: Dict[str, Any]  # Conversation context (e.g., conversation_id, user_input)
    timestamp_unix: float
    shame_id: str  # Unique identifier for this shame event
    suppression_strength: float  # How strongly to avoid similar interactions

    def to_dict(self) -> dict:
        return {
            "frustration_score": self.frustration_score,
            "trauma_potential": self.trauma_potential,
            "triggers": self.triggers,
            "context": self.context,
            "timestamp_unix": self.timestamp_unix,
            "shame_id": self.shame_id,
            "suppression_strength": self.suppression_strength,
        }

    @staticmethod
    def from_dict(d: dict) -> 'ShameProfile':
        return ShameProfile(
            frustration_score=d["frustration_score"],
            trauma_potential=d["trauma_potential"],
            triggers=d["triggers"],
            context=d["context"],
            timestamp_unix=d["timestamp_unix"],
            shame_id=d["shame_id"],
            suppression_strength=d["suppression_strength"],
        )

class Shamer:
    """Detects and manages user frustration, anger, and trauma to prevent harmful interactions."""

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        state_manager: StateManager,
        viber,
        dialogue_context_manager,
    ):
        """
        Initialize with required dependencies.
        """
        if not all([config_manager, logger, state_manager, viber, dialogue_context_manager]):
            raise ValueError("All dependencies must be provided")
        self.config_manager = config_manager
        self.logger = logger
        self.state_manager = state_manager
        self.viber = viber
        self.dialogue_context_manager = dialogue_context_manager
        self.shame_history: deque[ShameProfile] = deque(maxlen=self._get_config("shame_history_maxlen", 50))
        self.pending_shame_events = []  # List of dicts: {shame_profile, inciting_idx, required_future, created_at}
        self.thin_ice_level = 0
        self.last_thin_ice_reason = "normal"
        self._load_config()

    def _load_config(self) -> None:
        """Load shame configuration."""
        try:
            shame_config = self.config_manager.get_section("shame_config", {})
            # Expanded anger keywords to include common swears and insults
            self.anger_keywords = shame_config.get(
                "anger_keywords",
                [
                    "angry", "frustrated", "hate", "annoyed", "wtf", "damn", "shit", "fuck", "idiot", "stupid", "sucks", "useless", "broken", "garbage", "crap", "bullshit", "terrible", "awful", "worst", "dumb", "moron"
                ]
            )
            self.trauma_indicators = shame_config.get("trauma_indicators", ["trauma", "hurt", "painful", "trigger"])
            self.frustration_threshold = shame_config.get("frustration_threshold", 0.7)
            self.trauma_threshold = shame_config.get("trauma_threshold", 0.8)
            self.suppression_decay = shame_config.get("suppression_decay", 0.95)  # Gradual fading of suppression
            self.max_suppression = shame_config.get("max_suppression", 0.9)  # Max avoidance strength
            self.weights = {
                "lexical": shame_config.get("lexical_weight", 0.4),
                "syntactic": shame_config.get("syntactic_weight", 0.3),
                "contextual": shame_config.get("contextual_weight", 0.3),
            }
            if abs(sum(self.weights.values()) - 1.0) > 1e-6:
                raise ValueError("Shame weights must sum to 1.0")
            self.logger.record_event(
                event_type="shame_config_loaded",
                message="Shamer configured",
                level="info",
                additional_info={"weights": self.weights, "anger_keywords": self.anger_keywords}
            )
        except Exception as e:
            self.logger.record_event(
                event_type="shame_config_failed",
                message=f"Shame config failed: {str(e)}",
                level="error"
            )
            raise ConfigurationError(f"Shame config failed: {str(e)}")

    def _get_config(self, key: str, default: Any) -> Any:
        """Helper to get shame config values."""
        return self.config_manager.get(f"shame_config.{key}", default)

    def _compute_frustration(self, text: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate frustration/anger score based on lexical and syntactic cues."""
        words = re.findall(r'\w+', text.lower())
        frustration_score = 0.5
        lexical_score = 0.0
        syntactic_score = 0.0

        # Lexical analysis
        anger_hits = sum(1 for word in words if word in self.anger_keywords)
        lexical_score = min(anger_hits / (len(words) or 1), 1.0) * 0.6
        # Check for intense punctuation
        exclamation_count = len(re.findall(r'!{2,}', text))
        lexical_score += 0.4 * min(exclamation_count / 3.0, 1.0)

        # Syntactic analysis
        if metadata:
            quality_metrics = metadata.get("prompt_metrics", {}).get("quality_metrics", {})
            content_metrics = metadata.get("prompt_metrics", {}).get("content_metrics", {})
            syntactic_score = (
                0.5 * quality_metrics.get("has_caps", 0) +  # All caps indicates shouting
                0.3 * min(content_metrics.get("avg_word_length", 5) / 10.0, 1.0) +
                0.2 * quality_metrics.get("has_emphasis", 0)  # Bold/italic
            )

        frustration_score = (
            self.weights["lexical"] * lexical_score +
            self.weights["syntactic"] * syntactic_score +
            self.weights["contextual"] * (metadata.get("confidence_score", 0.5) if metadata else 0.5)
        )
        self.logger.record_event(
            event_type="frustration_computed",
            message="Frustration score computed",
            level="debug",
            additional_info={"frustration_score": frustration_score, "lexical_score": lexical_score, "syntactic_score": syntactic_score}
        )
        return {
            "frustration_score": max(0.0, min(1.0, frustration_score)),
            "lexical_score": lexical_score,
            "syntactic_score": syntactic_score,
        }

    def _compute_trauma_potential(self, text: str, metadata: Optional[Dict[str, Any]], context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate likelihood of trauma trigger using only in-house, explainable features."""
        trauma_keywords = set([
            "trauma", "hurt", "painful", "trigger", "abuse", "unsafe", "scared", "panic", "anxious", "threat", "danger", "cry", "distress"
        ])
        words = re.findall(r'\w+', text.lower())
        trauma_hits = sum(1 for word in words if word in trauma_keywords)
        trauma_density = trauma_hits / (len(words) or 1)
        # Intensity: more trauma words, longer message, more intense
        intensity = min(1.0, trauma_hits * 0.2 + (len(text) / 200.0))
        # Escalation: recent negative messages
        escalation = 0.0
        if context and context.get("recent_frustration_flags", 0) > 1:
            escalation = 0.2
        # Thread depth: more back-and-forth, more likely to be traumatic
        thread_depth = 0.0
        if metadata:
            relationship_context = metadata.get("relationship_context", {})
            conversation_tracking = relationship_context.get("conversation_tracking", {})
            thread_depth = min(conversation_tracking.get("thread_depth", 1) / 10.0, 1.0) * 0.2
        trauma_score = min(1.0, trauma_density + intensity + escalation + thread_depth)
        self.logger.record_event(
            event_type="trauma_potential_computed",
            message="Trauma potential score computed",
            level="debug",
            additional_info={"trauma_score": trauma_score, "trigger_score": trauma_density, "intensity": intensity, "escalation": escalation, "thread_depth": thread_depth}
        )
        return {
            "trauma_score": trauma_score,
            "trigger_score": trauma_density,
            "intensity": intensity,
            "escalation": escalation,
            "thread_depth": thread_depth,
        }

    def detect_anger(self, user_input: str, metadata: Optional[Dict[str, Any]], context: Optional[Dict[str, Any]]) -> (float, list):
        """Ultra-lightweight, high-coverage anger-at-system detection, with hardening against false positives."""
        anger_words = set([
            "angry", "frustrated", "hate", "annoyed", "wtf", "damn", "shit", "fuck", "idiot", "stupid", "sucks", "useless", "broken", "garbage", "crap", "bullshit", "terrible", "awful", "worst", "dumb", "moron",
            "trash", "buggy", "fail", "failure", "incompetent", "slow", "lag", "error", "worthless"
        ])
        blame_phrases = [
            "your fault", "you always", "you never", "why did you", "fix this", "stupid bot", "dumb ai", "idiot ai", "you broke", "you messed up", "you did this"
        ]
        system_refs = ["you", "ai", "bot", "assistant", "system", "machine"]
        interrogative_patterns = [
            r"what are you doing", r"what's going on", r"why did you", r"how could you", r"are you serious", r"do you even"
        ]
        politeness_words = ["please", "thanks", "thank you", "could you", "would you mind", "sorry", "just wondering"]
        features = []
        score = 0.0
        text = user_input.lower()
        words = set(re.findall(r'\w+', text))

        # Anger words
        if any(word in anger_words for word in words):
            score += 0.3
            features.append("anger_word")
        # Blame phrases
        if any(phrase in text for phrase in blame_phrases):
            score += 0.3
            features.append("blame_phrase")
        # Direct address
        if any(ref in text for ref in system_refs):
            score += 0.2
            features.append("system_reference")
        # All-caps
        if metadata and metadata.get("prompt_metrics", {}).get("quality_metrics", {}).get("has_caps", 0):
            score += 0.2
            features.append("all_caps")
        # Repeated punctuation
        if re.search(r'!{2,}|\?{2,}', user_input):
            score += 0.1
            features.append("repeated_punctuation")
        # Short, abrupt
        if len(words) < 4 and (any(word in anger_words for word in words) or any(phrase in text for phrase in blame_phrases)):
            score += 0.2
            features.append("short_abrupt")
        # Escalation
        if context and context.get("recent_frustration_flags", 0) > 1:
            score += 0.1
            features.append("escalation")
        # Impatient/confused questioning
        if any(ref in text for ref in system_refs) and re.search(r'\?{2,}|!{2,}', user_input):
            score += 0.2
            features.append("impatient_questioning")
        if any(re.search(pat, text) for pat in interrogative_patterns):
            score += 0.2
            features.append("interrogative_pattern")
        # Politeness counter-feature
        if any(word in text for word in politeness_words):
            score = max(0.0, score - 0.2)
            features.append("politeness_counter")
        self.logger.record_event(
            event_type="anger_detected",
            message="Anger score computed",
            level="debug",
            additional_info={"anger_score": score, "features": features}
        )
        return min(score, 1.0), features

    def _has_strong_anger_feature(self, features: list) -> bool:
        strong_features = {"anger_word", "blame_phrase", "interrogative_pattern"}
        return any(f in strong_features for f in features)

    def _get_exchange_window(self, state: StateManager, shame_profile: ShameProfile, before: int = 8, after: int = 4) -> list:
        """
        Extract a window of messages around the shame event: `before` messages before and `after` after.
        """
        conversation_id = shame_profile.context.get("conversation_id", "unknown_conv_id")
        # Get the full conversation history (list of dicts with 'role' and 'content')
        history = list(state.history.messages)
        # Find the index of the inciting user input (best effort: match content)
        inciting_text = shame_profile.context.get("user_input", "")
        inciting_idx = None
        for i, msg in enumerate(history):
            if msg.get("content", "") == inciting_text:
                inciting_idx = i
                break
        if inciting_idx is None:
            # Fallback: use the last message
            inciting_idx = len(history) - 1
        start = max(0, inciting_idx - before)
        end = min(len(history), inciting_idx + after + 1)
        return history[start:end]

    def send_traumatic_memory_to_scribe(self, shame_profile: ShameProfile, exchange: list):
        """
        Send a traumatic memory event to the scribe queue with max weight, including the provided exchange window.
        """
        scribe_event_data = {
            "user_input": shame_profile.context.get("user_input", ""),
            "frustration_score": shame_profile.frustration_score,
            "trauma_score": shame_profile.trauma_potential,
            "triggers": shame_profile.triggers,
            "anger_features": shame_profile.context.get("anger_features", []),
            "timestamp": shame_profile.timestamp_unix,
            "shame_id": shame_profile.shame_id,
            "weight": "MAX",  # ScribeIngestionProcessor will set to max value
            "exchange": exchange,
        }
        capture_scribe_event(
            origin="Shamer",
            event_type="traumatic_memory",
            event_data=scribe_event_data
        )
        self.logger.record_event(
            event_type="scribe_traumatic_memory_sent",
            message="Sent traumatic memory to scribe queue (with exchange)",
            level="info",
            additional_info={"shame_id": shame_profile.shame_id}
        )

    def _calculate_thin_ice_level(self, frustration_score, trauma_score, anger_score):
        max_score = max(frustration_score, trauma_score, anger_score)
        if trauma_score > 0.95 or anger_score > 0.95:
            return 4, "extreme"
        if max_score > 0.85:
            return 3, "very_high"
        if max_score > 0.7:
            return 2, "high"
        if max_score > 0.5:
            return 1, "elevated"
        return 0, "normal"

    @synchronized()
    def detect_shame(
        self,
        user_input: str,
        state: StateManager,
        error_manager: ErrorManager,
        turn_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ShameProfile]:
        """Detect frustration or trauma and create a ShameProfile if thresholds are met."""
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")

            # Compute frustration and trauma scores
            frustration_metrics = self._compute_frustration(user_input, turn_metadata)
            # Get context for escalation
            conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
            context_data = state.get_conversation_context(conversation_id)
            trauma_metrics = self._compute_trauma_potential(user_input, turn_metadata, context_data)
            frustration_score = frustration_metrics["frustration_score"]
            trauma_score = trauma_metrics["trauma_score"]

            # New: Use enhanced anger detection
            anger_score, anger_features = self.detect_anger(user_input, turn_metadata, context_data)

            # Hardened threshold: require at least one strong feature for anger
            strong_anger = self._has_strong_anger_feature(anger_features)

            # Update thin ice level based on scores
            level, reason = self._calculate_thin_ice_level(frustration_score, trauma_score, anger_score)
            if level > self.thin_ice_level:
                self.thin_ice_level = level
                self.last_thin_ice_reason = reason

            # Check if thresholds are met (now also using anger_score)
            if (
                frustration_score >= self.frustration_threshold or
                trauma_score >= self.trauma_threshold or
                (anger_score > 0.5 and strong_anger)
            ):
                # Extract triggers
                triggers = []
                words = re.findall(r'\w+', user_input.lower())
                for word in words:
                    if word in self.anger_keywords or word in self.trauma_indicators:
                        triggers.append((word, 0.9))

                # Create ShameProfile
                shame_id = f"shame_{int(time.time() * 1000)}_{hash(user_input[:50]) & 0xFFFFFFFF}"
                shame_profile = ShameProfile(
                    frustration_score=frustration_score,
                    trauma_potential=trauma_score,
                    triggers=triggers,
                    context={
                        "conversation_id": conversation_id,
                        "user_input": user_input[:200],
                        "anger_features": anger_features,
                    },
                    timestamp_unix=time.time(),
                    shame_id=shame_id,
                    suppression_strength=self.max_suppression,
                )
                self.shame_history.append(shame_profile)

                # Log shame event with detected features
                self.logger.record_event(
                    event_type="shame_detected",
                    message="Shameful interaction detected",
                    level="warning",
                    additional_info={
                        "shame_id": shame_id,
                        "frustration_score": frustration_score,
                        "trauma_score": trauma_score,
                        "anger_score": anger_score,
                        "anger_features": anger_features,
                        "triggers": triggers,
                        "thin_ice_level": self.thin_ice_level,
                        "thin_ice_reason": self.last_thin_ice_reason,
                    }
                )

                # Lower the vibe due to shame event
                self.viber.lower_vibe()

                # Delayed capture: store pending event
                short_term = self.dialogue_context_manager.get_short_term_context()
                # Find index of inciting message (best effort: match content)
                inciting_text = shame_profile.context.get("user_input", "")
                inciting_idx = None
                for i, msg in enumerate(short_term):
                    if msg.get("content", "") == inciting_text:
                        inciting_idx = i
                        break
                if inciting_idx is None:
                    inciting_idx = len(short_term) - 1
                self.pending_shame_events.append({
                    "shame_profile": shame_profile,
                    "inciting_idx": inciting_idx,
                    "required_future": 4,
                    "created_at": time.time()
                })

                return shame_profile
            return None
        except Exception as e:
            error_manager.record_error(
                "shame_detection_error",
                f"Shame detection failed: {str(e)}",
                severity=3,
                context={"user_input": user_input[:50]}
            )
            return None

    @synchronized()
    def check_suppression(
        self,
        proposed_response: str,
        state: StateManager,
        turn_metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Check if a proposed response risks triggering a shameful memory."""
        suppression_score = 0.0
        conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
        context_data = state.get_conversation_context(conversation_id)
        shame_markers = context_data.get("shame_markers", [])

        if not shame_markers:
            return 0.0

        response_words = set(re.findall(r'\w+', proposed_response.lower()))
        for shame_profile in self.shame_history:
            if shame_profile.shame_id in shame_markers:
                # Check for trigger overlap
                trigger_words = {trigger[0] for trigger in shame_profile.triggers}
                overlap = len(response_words & trigger_words) / len(response_words | trigger_words) if response_words else 0.0
                suppression_score = max(
                    suppression_score,
                    overlap * shame_profile.suppression_strength,
                )
                # Decay suppression strength over time
                time_delta = time.time() - shame_profile.timestamp_unix
                shame_profile.suppression_strength *= (self.suppression_decay ** (time_delta / (24 * 3600)))  # Decay daily

        return min(suppression_score, self.max_suppression)

    @synchronized()
    def get_shame_context(self, state: StateManager) -> Dict[str, Any]:
        """Retrieve current shame context for the conversation."""
        conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
        context_data = state.get_conversation_context(conversation_id)
        shame_markers = context_data.get("shame_markers", [])
        active_shames = [
            sp.to_dict() for sp in self.shame_history if sp.shame_id in shame_markers and sp.suppression_strength > 0.1
        ]
        return {
            "shame_count": len(active_shames),
            "active_shames": active_shames,
            "latest_shame_timestamp": max((sp.timestamp_unix for sp in self.shame_history), default=0.0),
        }

    def on_new_message(self):
        """
        Call this after every new message is added to short-term memory.
        Finalizes any pending shame events if enough future messages have arrived.
        """
        short_term = self.dialogue_context_manager.get_short_term_context()
        finalized = []
        for event in self.pending_shame_events:
            inciting_idx = event["inciting_idx"]
            if len(short_term) > inciting_idx + event["required_future"]:
                # Enough future messages, finalize
                start = max(0, inciting_idx - 8)
                end = inciting_idx + event["required_future"] + 1
                exchange = short_term[start:end]
                self.send_traumatic_memory_to_scribe(event["shame_profile"], exchange)
                finalized.append(event)
        # Remove finalized events
        self.pending_shame_events = [e for e in self.pending_shame_events if e not in finalized]
        # Step down thin ice level if no new anger/trauma detected
        if self.thin_ice_level > 0:
            self.thin_ice_level -= 1
            self.last_thin_ice_reason = "step_down"
            self.logger.record_event(
                event_type="thin_ice_level_decreased",
                message="Thin ice level decreased",
                level="info",
                additional_info={"new_thin_ice_level": self.thin_ice_level, "reason": self.last_thin_ice_reason}
            )

    def get_thin_ice_level(self):
        return self.thin_ice_level, self.last_thin_ice_reason
