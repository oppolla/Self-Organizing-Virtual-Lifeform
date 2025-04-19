from typing import Optional, Dict, Any
import re
import time
import torch
from collections import deque, defaultdict
from threading import Lock
from sovl_logger import Logger
from sovl_state import SOVLState
from sovl_error import ErrorManager
from sovl_main import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized
from sovl_config import ConfigManager

class BondCalculator:
    """Calculates bonding score based on user wordprint and duration of knowing."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize bond calculator with configuration and logging.
        
        Args:
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        if not config_manager or not logger:
            raise ValueError("config_manager and logger cannot be None")
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()
        self._initialize_config()

    def _initialize_config(self) -> None:
        """Initialize bonding score configuration."""
        try:
            bond_config = self.config_manager.get_section("bond_config", {})
            self.min_bond_score = float(bond_config.get("min_bond_score", 0.0))
            self.max_bond_score = float(bond_config.get("max_bond_score", 1.0))
            self.default_bond_score = float(bond_config.get("default_bond_score", 0.5))
            self.max_interactions = int(bond_config.get("max_interactions", 100))
            self.max_session_time = float(bond_config.get("max_session_time", 3600.0))
            self.decay_rate = float(bond_config.get("decay_rate", 0.95))
            self.decay_interval = float(bond_config.get("decay_interval", 86400.0))  # 24 hours
            self.max_expected_dev = float(bond_config.get("max_expected_dev", 20.0))
            self.max_lexicon_size = int(bond_config.get("max_lexicon_size", 1000))
            self.weights = {
                "curiosity": float(bond_config.get("curiosity_weight", 0.3)),
                "stability": float(bond_config.get("stability_weight", 0.3)),
                "coherence": float(bond_config.get("coherence_weight", 0.2)),
                "personalized": float(bond_config.get("personalized_weight", 0.2))
            }
            if abs(sum(self.weights.values()) - 1.0) > 1e-6:
                raise ValueError("Bond weights must sum to 1.0")
            self.logger.record_event(
                event_type="bond_config_initialized",
                message="Bond configuration initialized",
                level="info",
                additional_info={"bond_config": bond_config}
            )
        except Exception as e:
            self.logger.record_event(
                event_type="bond_config_failed",
                message=f"Failed to initialize bond configuration: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            raise

    def _compute_wordprint_score(self, user_input: str, state: SOVLState) -> float:
        """Calculate wordprint score based on lexical signature and style consistency."""
        profile = state.user_profiles[state.conversation_id]
        # Lexical Signature
        current_words = set(re.findall(r'\w+', user_input.lower()))
        top_words = sorted(profile["user_lexicon"].items(), key=lambda x: x[1], reverse=True)[:10]
        top_words_set = set(word for word, _ in top_words)
        signature_score = (len(current_words & top_words_set) / len(current_words | top_words_set)
                          if current_words and top_words_set else 0.5)
        # Style Consistency
        word_counts = [len(re.findall(r'\w+', input)) for input in profile["recent_inputs"]]
        std_dev = torch.std(torch.tensor(word_counts, dtype=torch.float32)).item() if word_counts else 0.0
        style_score = 1 - min(std_dev / self.max_expected_dev, 1.0)
        # Combine
        wordprint_score = 0.6 * signature_score + 0.4 * style_score
        return max(0.0, min(1.0, wordprint_score))

    def _compute_knowing_score(self, state: SOVLState) -> float:
        """Calculate duration of knowing score based on interaction count and time."""
        profile = state.user_profiles[state.conversation_id]
        # Apply decay if inactive
        if time.time() - profile["last_interaction_time"] > self.decay_interval:
            profile["interaction_count"] = int(profile["interaction_count"] * self.decay_rate)
            profile["total_session_time"] *= self.decay_rate
            profile["last_interaction_time"] = time.time()
        # Interaction Familiarity
        familiarity = min(profile["interaction_count"] / self.max_interactions, 1.0)
        # Time Duration
        duration = min(profile["total_session_time"] / self.max_session_time, 1.0)
        # Combine
        knowing_score = 0.7 * familiarity + 0.3 * duration
        return max(0.0, min(1.0, knowing_score))

    def _compute_stability_score(self, state: SOVLState) -> float:
        """Placeholder for stability score (system health/errors)."""
        # Implement based on error rates or system metrics
        return 0.5  # Neutral default

    def _compute_coherence_score(self, user_input: str, state: SOVLState) -> float:
        """Placeholder for coherence score (input-response alignment)."""
        # Implement based on semantic similarity
        return 0.5  # Neutral default

    def _prune_lexicon(self, lexicon: Dict[str, int]) -> None:
        """Prune lexicon to top max_lexicon_size words by frequency."""
        if len(lexicon) > self.max_lexicon_size:
            sorted_items = sorted(lexicon.items(), key=lambda x: x[1], reverse=True)[:self.max_lexicon_size]
            lexicon.clear()
            lexicon.update(dict(sorted_items))

    @synchronized()
    def calculate_bonding_score(
        self,
        user_input: str,
        state: SOVLState,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None
    ) -> float:
        """
        Calculate bonding score using curiosity, stability, coherence, and personalized (wordprint + knowing) components.
        
        Args:
            user_input: User input string
            state: Current system state
            error_manager: Error manager instance
            context: System context
            curiosity_manager: Optional curiosity manager for novelty score
        
        Returns:
            Bonding score in [min_bond_score, max_bond_score]
        """
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")
            if not state.conversation_id:
                raise ValueError("conversation_id must be set in state")

            # Initialize profile for new conversation_id
            if state.conversation_id not in state.user_profiles:
                state.user_profiles[state.conversation_id] = {
                    "user_lexicon": defaultdict(int),
                    "interaction_count": 0,
                    "total_session_time": 0.0,
                    "recent_inputs": deque(maxlen=10),
                    "last_interaction_time": time.time()
                }
            profile = state.user_profiles[state.conversation_id]

            # Update state
            for word in re.findall(r'\w+', user_input.lower()):
                profile["user_lexicon"][word] += 1
            self._prune_lexicon(profile["user_lexicon"])
            profile["recent_inputs"].append(user_input)
            profile["interaction_count"] += 1
            profile["total_session_time"] += time.time() - state.session_start
            profile["last_interaction_time"] = time.time()

            # Calculate component scores
            curiosity_score = (curiosity_manager.get_novelty_score(user_input)
                              if curiosity_manager else 0.5)
            stability_score = self._compute_stability_score(state)
            coherence_score = self._compute_coherence_score(user_input, state)
            wordprint_score = self._compute_wordprint_score(user_input, state)
            knowing_score = self._compute_knowing_score(state)
            personalized_score = 0.5 * wordprint_score + 0.5 * knowing_score

            # Combine scores
            bond_score = (
                self.weights["curiosity"] * curiosity_score +
                self.weights["stability"] * stability_score +
                self.weights["coherence"] * coherence_score +
                self.weights["personalized"] * personalized_score
            )
            bond_score = max(self.min_bond_score, min(self.max_bond_score, bond_score))

            self.logger.record_event(
                event_type="bond_score_calculated",
                message="Bond score calculated",
                level="info",
                additional_info={
                    "bond_score": bond_score,
                    "curiosity_score": curiosity_score,
                    "stability_score": stability_score,
                    "coherence_score": coherence_score,
                    "wordprint_score": wordprint_score,
                    "knowing_score": knowing_score,
                    "conversation_id": state.conversation_id
                }
            )

            return bond_score

        except Exception as e:
            self.logger.record_event(
                event_type="bond_score_failed",
                message=f"Failed to calculate bond score: {str(e)}",
                level="error",
                additional_info={"error": str(e), "conversation_id": state.conversation_id}
            )
            error_manager.handle_data_error(
                e, {"user_input": user_input[:50]}, state.conversation_id
            )
            return self.default_bond_score
