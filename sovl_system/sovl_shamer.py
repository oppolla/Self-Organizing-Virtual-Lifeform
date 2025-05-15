import time
import re
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from sovl_config import ConfigManager
from sovl_queue import capture_scribe_event
from sovl_logger import Logger
from sovl_state import StateManager
from sovl_error import ErrorManager, ConfigurationError
from sovl_utils import synchronized
from sovl_viber import VibeSculptor
from sovl_recaller import DialogueContextManager
from sovl_utils import levenshtein_distance

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
        self.last_empathy_attempt = 0
        self.empathy_cooldown = 600  # seconds (10 minutes)

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
            self.frustration_threshold = shame_config.get("frustration_threshold", 0.7)
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
            apology_config = self.config_manager.get_section("apology_config", {})
            self.apology_weights = {
                "direct": apology_config.get("direct_apology_weight", 0.4),
                "casual": apology_config.get("casual_apology_weight", 0.3),
                "defensive": apology_config.get("defensive_apology_weight", 0.2),
                "reconciliation": apology_config.get("reconciliation_weight", 0.25),
                "tentative": apology_config.get("tentative_apology_weight", 0.15),
                "politeness": apology_config.get("politeness_marker_weight", 0.05)
            }
            self.apology_thresholds = {
                "direct": apology_config.get("direct_apology_threshold", 0.4),
                "tentative": apology_config.get("tentative_apology_threshold", 0.6)
            }
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

    def is_fuzzy_anger_word(self, word, anger_words, max_distance=1):
        if len(word) < 3:
            return False
        for anger_word in anger_words:
            if abs(len(word) - len(anger_word)) <= 2 and levenshtein_distance(word, anger_word) <= max_distance:
                return True
        return False

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
        word_list = re.findall(r'\w+', text)
        words = set(word_list)

        # Early exit for clearly non-angry inputs
        if not (
            any(word in words for word in anger_words)
            or any(ref in text for ref in system_refs)
            or re.search(r'[!?]', text)
        ):
            return 0.0, ["no_anger_cues"]

        # Dynamic feature weighting for anger words (expanded)
        strong_anger = {
            "fuck", "fucking", "shit", "idiot", "bullshit", "hate", "dumbass", "moron", "useless",
            "garbage", "worthless", "bastard", "retard", "sucks", "crap", "awful", "terrible", "stupid",
            "damn", "wtf", "douche", "jackass", "asshole", "bitch", "scum", "trash"
        }
        medium_anger = {
            "angry", "frustrated", "annoyed", "annoying", "frustrating", "broken", "slow", "laggy",
            "failure", "incompetent", "buggy", "fail", "dumb", "lousy", "mediocre", "pathetic",
            "disgusting", "horrible", "pain", "problematic", "disaster", "mess", "screwup", "screwed"
        }
        weak_anger = {
            "confused", "disappointed", "problem", "issue", "error", "lag", "delay", "unhelpful", "meh",
            "lame", "bad", "not working", "doesn't work", "doesnt work", "missing", "incomplete",
            "unclear", "hard", "difficult", "trouble", "bother", "annoy", "irritate", "glitch"
        }
        if any(self.is_fuzzy_anger_word(word, strong_anger) for word in words):
            score += 0.4
            features.append("strong_anger_word")
        elif any(self.is_fuzzy_anger_word(word, medium_anger) for word in words):
            score += 0.3
            features.append("medium_anger_word")
        elif any(self.is_fuzzy_anger_word(word, weak_anger) for word in words):
            score += 0.2
            features.append("weak_anger_word")

        # Contextual negation handling
        negation_words = {"not", "never", "no"}
        window_size = 3  # Number of words to look back for negation
        negation_detected = False
        for idx, word in enumerate(word_list):
            if self.is_fuzzy_anger_word(word, anger_words):
                window_start = max(0, idx - window_size)
                if any(w in negation_words for w in word_list[window_start:idx]):
                    negation_detected = True
                    break
        if negation_detected:
            score = max(0.0, score - 0.1)
            features.append("negation_detected")
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
        if re.search(r'!{2,}|\?{2,}', text):
            score += 0.1
            features.append("repeated_punctuation")
        # Short, abrupt (adjusted)
        if (
            len(words) <= 3
            and (
                any(word in strong_anger for word in words)
                or any(phrase in text for phrase in blame_phrases)
            )
        ):
            score += 0.2
            features.append("short_abrupt")
        # Escalation
        if context and context.get("recent_frustration_flags", 0) > 1:
            score += 0.1
            features.append("escalation")
        # Impatient/confused questioning
        if any(ref in text for ref in system_refs) and re.search(r'\?{2,}|!{2,}', text):
            score += 0.2
            features.append("impatient_questioning")
        if any(re.search(pat, text) for pat in interrogative_patterns):
            score += 0.2
            features.append("interrogative_pattern")
        # Batch feature checks for efficiency
        anger_words_set = set(self.anger_keywords)
        system_refs_set = set(system_refs)
        politeness_words_set = set(politeness_words)
        strong_anger_set = strong_anger
        found_anger = False
        found_system_ref = False
        found_politeness = False
        for word in words:
            if not found_anger and (
                self.is_fuzzy_anger_word(word, strong_anger_set)
                or self.is_fuzzy_anger_word(word, medium_anger)
                or self.is_fuzzy_anger_word(word, weak_anger)
            ):
                found_anger = True
            if not found_system_ref and word in system_refs_set:
                found_system_ref = True
            if not found_politeness and word in politeness_words_set:
                found_politeness = True
        # Apply anger word scoring (dynamic weighting already handled above)
        # Apply system reference scoring
        if found_system_ref:
            score += 0.2
            features.append("system_reference")
        # Apply politeness counter-feature (adjusted)
        if found_politeness and not (words & strong_anger_set or any(phrase in text for phrase in blame_phrases)):
            score = max(0.0, score - 0.2)
            features.append("politeness_counter")
        # User history context: sustained frustration trend
        if context:
            last_inputs = context.get("last_5_inputs", [])
            anger_count = 0
            for past_input in last_inputs:
                past_words = set(re.findall(r'\w+', past_input.lower()))
                if any(self.is_fuzzy_anger_word(word, anger_words) for word in past_words):
                    anger_count += 1
            if anger_count >= 2:
                score += 0.1
                features.append("frustration_trend")
        # Boost score if recent shame profiles exist with high frustration scores
        if any(
            sp.frustration_score > 0.5 for sp in getattr(self, 'shame_history', [])
            if time.time() - sp.timestamp_unix < 3600
        ):
            score += 0.1
            features.append("recent_shame")
        # Sarcasm detection
        sarcasm_words = {"great", "awesome", "fantastic", "perfect", "genius"}
        if (
            any(word in words for word in sarcasm_words)
            and any(ref in text for ref in system_refs)
            and re.search(r'!|\?', text)
        ):
            score += 0.15
            features.append("sarcasm_detected")
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

    def _calculate_thin_ice_level(self, frustration_score, anger_score, features=None):
        features = features or []
        max_score = max(frustration_score, anger_score)
        if "sarcasm_detected" in features or anger_score > 0.95:
            return 4, "extreme"
        if max_score > 0.85 or any(sp.frustration_score > 0.5 for sp in getattr(self, 'shame_history', [])):
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
        """Detect frustration and create a ShameProfile if thresholds are met."""
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")

            # --- Apology detection and reset if in shame state ---
            if self.thin_ice_level > 0:
                apology_score, apology_features, _ = self.detect_apology(user_input)
                if apology_score > 0.5:
                    self.apply_configurable_reset(apology_score)
                    self.logger.record_event(
                        event_type="apology_detected",
                        message="Apology detected, applying configurable reset",
                        level="info",
                        additional_info={"apology_score": apology_score, "apology_features": apology_features}
                    )
                    # After reset, do not proceed with shame detection for this turn
                    return None

            # Compute frustration score
            frustration_metrics = self._compute_frustration(user_input, turn_metadata)
            conversation_id = getattr(state.history, 'conversation_id', "unknown_conv_id")
            context_data = state.get_conversation_context(conversation_id)
            frustration_score = frustration_metrics["frustration_score"]

            # Use enhanced anger detection
            anger_score, anger_features = self.detect_anger(user_input, turn_metadata, context_data)
            strong_anger = self._has_strong_anger_feature(anger_features)

            # Update thin ice level based on scores and features
            level, reason = self._calculate_thin_ice_level(frustration_score, anger_score, anger_features)
            if level > self.thin_ice_level:
                self.thin_ice_level = level
                self.last_thin_ice_reason = reason

            # Check for sustained anger in recent history (3+ angry turns)
            last_inputs = context_data.get("last_5_inputs", [])
            recent_anger_count = 0
            for past_input in last_inputs:
                past_words = set(re.findall(r'\w+', past_input.lower()))
                if any(self.is_fuzzy_anger_word(word, self.anger_keywords) for word in past_words):
                    recent_anger_count += 1

            sustained_anger = recent_anger_count >= 3

            # Only escalate if sustained anger (3+)
            if (
                (frustration_score >= self.frustration_threshold or (anger_score > 0.5 and strong_anger))
                and sustained_anger
            ):
                # Extract triggers (now includes blame phrases and interrogative patterns)
                text = user_input.lower()
                word_list = re.findall(r'\w+', text)
                words = set(word_list)
                blame_phrases = [
                    "your fault", "you always", "you never", "why did you", "fix this", "stupid bot", "dumb ai", "idiot ai", "you broke", "you messed up", "you did this"
                ]
                triggers = [(word, 0.9) for word in words if word in self.anger_keywords]
                triggers.extend((phrase, 0.9) for phrase in blame_phrases if phrase in text)
                triggers.extend((pat, 0.8) for pat in [
                    r"what are you doing", r"what's going on", r"why did you", r"how could you", r"are you serious", r"do you even"
                ] if re.search(pat, text))

                # Proactive empathy prompt with cooldown
                empathy_prompt = None
                now = time.time()
                if (
                    0.4 <= anger_score < 0.7
                    and not strong_anger
                    and (now - self.last_empathy_attempt > self.empathy_cooldown)
                ):
                    empathy_prompt = EMPATHY_SYSTEM_PROMPT_TEMPLATE.format(
                        user_input=user_input[:200],
                        anger_features=", ".join(anger_features) if anger_features else "none"
                    )
                    self.last_empathy_attempt = now

                # Create ShameProfile
                context = {
                    "conversation_id": conversation_id,
                    "user_input": user_input[:200],
                    "anger_features": anger_features,
                }
                if empathy_prompt:
                    context["empathy_prompt"] = empathy_prompt

                shame_id = f"shame_{int(time.time() * 1000)}_{hash(user_input[:50]) & 0xFFFFFFFF}"
                shame_profile = ShameProfile(
                    frustration_score=frustration_score,
                    trauma_potential=0.0,
                    triggers=triggers,
                    context=context,
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
                        "anger_score": anger_score,
                        "anger_features": anger_features,
                        "triggers": triggers,
                        "thin_ice_level": self.thin_ice_level,
                        "thin_ice_reason": self.last_thin_ice_reason,
                        "empathy_prompt": empathy_prompt,
                    }
                )

                # Lower the vibe due to shame event
                self.viber.lower_vibe()

                # Delayed capture: store pending event (only after escalation)
                short_term = self.dialogue_context_manager.get_short_term_context()
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
            # If not sustained, just set thin ice and return None (no shame, no vibe drop, no pending event)
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
        # TODO: Use anger features to weigh suppression more strongly for strong_anger_word, sarcasm_detected, etc.
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
        sarcasm_flag = any(
            "sarcasm_detected" in sp.context.get("anger_features", [])
            for sp in self.shame_history if sp.shame_id in shame_markers and sp.suppression_strength > 0.1
        )
        return {
            "shame_count": len(active_shames),
            "active_shames": active_shames,
            "latest_shame_timestamp": max((sp.timestamp_unix for sp in self.shame_history), default=0.0),
            "sarcasm_flag": sarcasm_flag,
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

    def detect_apology(self, user_input: str) -> Tuple[float, List[str], float]:
        """Detect apology in user input with expanded keywords and contextual analysis. Returns (score, features, dynamic_threshold)."""
        # Keyword sets
        direct_apologies = {
            "sorry", "apologies", "apologize", "regret", "sorrow",
            "i'm sorry", "i apologize", "my apologies", "so sorry", "really sorry", "deeply sorry",
            "i was mistaken", "i was wrong",
            "forgive me", "pardon me", "i regret that", "i'm remorseful",
            "my sincerest apologies", "truly sorry",
            "i owe you an apology", "i'm in the wrong"
        }
        casual_apologies = {
            "my bad", "oops", "my fault", "my mistake", "screwed up", "messed up", "whoops",
            "my goof", "my screw-up", "my oops", "blew it",
            "cocked up", "mucked up",
            "my error", "dropped the ball",
            "doh", "ugh"
        }
        defensive_apologies = {
            "didn't mean to", "wasn't my intention", "if i upset you", "if i came across", "no offense meant",
            "you're right", "you're correct",
            "if i offended you", "if that came out wrong", "not my intent",
            "i didn't intend that", "hope i didn't upset you",
            "my misunderstanding", "i misread that"
        }
        reconciliation_phrases = {
            "let's start over", "can we try again", "let's move on", "fresh start", "make it right",
            "nevermind i get it now",
            "let's reset", "can we backtrack", "let's clear the air",
            "i see now", "got it now", "i understand now",
            "let's make this work", "we're good now"
        }
        politeness_markers = {
            "please", "thank you", "appreciate it", "i understand", "i hear you",
            "kindly", "grateful for", "thanks for that",
            "i appreciate your patience", "thanks for clarifying"
        }
        tentative_apologies = {
            "my confusion", "i got mixed up", "i missed that",
            "that was on me", "i take it back", "i stand corrected",
            "guess i misjudged", "i see where i went wrong",
            "my apologies if so", "sorry if that seemed off"
        }

        # Build and cache the regex for multi-word phrases
        if not hasattr(self, '_apology_phrases_regex'):
            multi_word_phrases = {p for p in (direct_apologies | casual_apologies | defensive_apologies | reconciliation_phrases | tentative_apologies) if ' ' in p}
            if multi_word_phrases:
                self._apology_phrases_regex = re.compile(
                    r'\b(' + '|'.join([re.escape(phrase) for phrase in multi_word_phrases]) + r')\b'
                )
            else:
                self._apology_phrases_regex = None

        # Initialize
        score = 0.0
        features = []
        text = user_input.lower()
        words = set(re.findall(r'\w+', text))
        word_list = re.findall(r'\w+', text)

        # Fuzzy matching for single words
        def is_fuzzy_apology_word(word, apology_set, max_distance=1):
            if len(word) < 4:
                return word in apology_set
            for apology in apology_set:
                if abs(len(word) - len(apology)) <= 2 and levenshtein_distance(word, apology) <= max_distance:
                    return True
            return False

        # Exact matching for phrases
        def has_apology_phrase(text, apology_set):
            return any(phrase in text for phrase in apology_set)

        # Use the cached regex for phrase matching
        if self._apology_phrases_regex and self._apology_phrases_regex.search(text):
            score += 0.1
            features.append("phrase_match")

        # Score by category
        if has_apology_phrase(text, direct_apologies) or any(is_fuzzy_apology_word(word, direct_apologies) for word in words):
            score += 0.4
            features.append("direct_apology")
        if has_apology_phrase(text, casual_apologies) or any(is_fuzzy_apology_word(word, casual_apologies) for word in words):
            score += 0.3
            features.append("casual_apology")
        if has_apology_phrase(text, defensive_apologies) or any(is_fuzzy_apology_word(word, defensive_apologies) for word in words):
            score += 0.2
            features.append("defensive_apology")
        if has_apology_phrase(text, reconciliation_phrases):
            score += 0.25
            features.append("reconciliation_attempt")
        if any(word in politeness_markers for word in words) or has_apology_phrase(text, politeness_markers):
            # Only add score if other apology features are present
            if any(feature in features for feature in ["direct_apology", "casual_apology", "defensive_apology", "tentative_apology"]):
                score += 0.05  # Reduced weight
                features.append("politeness_marker")
        if has_apology_phrase(text, tentative_apologies) or any(is_fuzzy_apology_word(word, tentative_apologies) for word in words):
            score += 0.15  # Lower weight due to ambiguity
            features.append("tentative_apology")

        # Enhanced Negation handling
        negation_patterns = {r"\b(not|never|no)\b", r"\bnot at all\b", r"\bno way\b"}
        window_size = 3
        for idx, word in enumerate(word_list):
            if is_fuzzy_apology_word(word, direct_apologies | casual_apologies | defensive_apologies | tentative_apologies):
                window_start = max(0, idx - window_size)
                window_end = min(len(word_list), idx + window_size)
                window_text = ' '.join(word_list[window_start:window_end])
                if any(re.search(pat, window_text) for pat in negation_patterns):
                    score = max(0.0, score - 0.2)
                    features.append("negation_detected")
                    break

        # Contextual boost: recent anger
        if any(sp.frustration_score > 0.5 for sp in self.shame_history if time.time() - sp.timestamp_unix < 3600):
            score += 0.1
            features.append("recent_anger_context")

        # Syntactic features
        if re.search(r"\b(i|me|my)\b", text):
            score += 0.1
            features.append("personal_pronoun")
        if re.search(r"\.\.\.|\?", text):
            score += 0.05
            features.append("soft_punctuation")
        if len(words) < 5 and score > 0:
            score += 0.05
            features.append("short_input")

        # Emoticon and symbol support
        if re.search(r"[:;][-()/\\]|😔|🙏|😞|😣|🥺", text) and score > 0:
            score += 0.05
            features.append("regret_emoticon")

        # Sarcasm check to avoid false positives
        if any("sarcasm_detected" in sp.context.get("anger_features", []) for sp in self.shame_history if time.time() - sp.timestamp_unix < 3600):
            score = max(0.0, score - 0.15)
            features.append("sarcasm_context")

        # Dynamic threshold logic
        apology_threshold = 0.5
        if "direct_apology" in features:
            apology_threshold = 0.4  # Lower for strong apologies
        elif "tentative_apology" in features and not ("direct_apology" in features or "casual_apology" in features):
            apology_threshold = 0.6  # Higher for ambiguous cases

        # Log detection
        self.logger.record_event(
            event_type="apology_detection",
            message="Apology detection completed",
            level="debug",
            additional_info={"apology_score": score, "features": features, "apology_threshold": apology_threshold}
        )

        return min(score, 1.0), features, apology_threshold

    def apply_configurable_reset(self, apology_score: float):
        """
        Apply a configurable reset based on the apology score.
        Ensure that apologies are meaningful and effective.
        """
        # Retrieve the configured vibe drop value
        vibe_drop_value = self._get_config("vibe_drop_value", 0.25)  # Example default value

        # Calculate the vibe increase based on the apology score
        # Ensure a minimum vibe increase to make apologies feel meaningful
        min_vibe_increase = 0.05 * vibe_drop_value  # 5% of the vibe drop value
        vibe_increase = max(min_vibe_increase, vibe_drop_value * (1 - apology_score))

        # Apply the calculated vibe increase
        self.viber.raise_vibe(amount=vibe_increase)

        # Determine reset strength based on apology score
        if apology_score > 0.7:
            # Stronger reset for higher scores
            self.thin_ice_level = 0
            for shame_profile in self.shame_history:
                shame_profile.suppression_strength = min(shame_profile.suppression_strength, 0.5)  # Example baseline
        else:
            # Partial reset for minor apologies
            self.thin_ice_level = max(0, self.thin_ice_level - 1)
            for shame_profile in self.shame_history:
                shame_profile.suppression_strength *= 0.9  # Gradual reduction

        # Log the reset action
        self.logger.record_event(
            event_type="configurable_reset_applied",
            message="Configurable reset applied based on apology score",
            level="info",
            additional_info={"apology_score": apology_score, "thin_ice_level": self.thin_ice_level, "vibe_increase": vibe_increase}
        )

EMPATHY_SYSTEM_PROMPT_TEMPLATE = (
    "The user appears to be experiencing moderate frustration or upset, likely due to challenges in our recent interaction. "
    "Their most recent input was: \"{user_input}\". "
    "Detected emotional indicators include: {anger_features}. "
    "Your primary objective is to respond with deep empathy, fully acknowledging the user's emotional state and any potential missteps in the conversation. "
    "Assume responsibility for any confusion, error, or dissatisfaction, even if the cause is unclear, to demonstrate accountability and build trust. "
    "Avoid any defensive or dismissive language, as this could escalate the user's frustration. "
    "Instead, express genuine regret for their experience and convey an earnest desire to make things right. "
    "Reflect briefly on the specific emotional cues (e.g., strong language, questioning tone, or sarcasm) to show you've paid close attention to their words and feelings. "
    "Offer clear, actionable assistance, such as revisiting the previous topic, clarifying a misunderstanding, or inviting the user to provide more details about their needs. "
    "If the emotional cues suggest sarcasm or indirect frustration, acknowledge the possibility of a misread tone with humility and invite correction. "
    "Propose at least two specific ways to move forward, ensuring the user feels empowered to guide the conversation. "
    "Maintain a warm, approachable tone that prioritizes the user's comfort and signals your commitment to a positive, collaborative exchange. "
    "Frame your response as a fresh start, aiming to de-escalate tension, correct any errors, and restore a sense of mutual understanding. "
    "If prior context (e.g., recent conversation history or detected frustration trends) is available, subtly reference it to show continuity and attentiveness, but keep the focus on the present moment. "
    "Conclude with an open-ended invitation for the user to share their thoughts, reinforcing that their perspective is valued and central to resolving the issue."
)
