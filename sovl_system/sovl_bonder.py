from typing import Optional, Dict
import torch
from threading import Lock
from sovl_logger import Logger
from sovl_state import StateManager, UserProfileState
from sovl_error import ErrorManager
from sovl_main import SystemContext
from sovl_curiosity import CuriosityManager
from sovl_utils import synchronized
from sovl_config import ConfigManager
import time
import hashlib
import json
import re
import traceback
import threading

class BondCalculator:
    """Calculates bonding score and manages user bond/nickname via UserProfileState."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, user_profile_state: 'UserProfileState'):
        if not config_manager or not logger or not user_profile_state:
            raise ValueError("config_manager, logger, and user_profile_state cannot be None")
        self.config_manager = config_manager
        self.logger = logger
        self.user_profile_state = user_profile_state
        self._initialize_config()
        # --- New: Bonding config parameters exposed ---
        bonding_config = self.config_manager.get_section("bonding_config", {})
        self.strong_bond_threshold = float(bonding_config.get("strong_bond_threshold", 0.8))
        self.weak_bond_threshold = float(bonding_config.get("weak_bond_threshold", 0.3))
        self.default_bond_score = float(bonding_config.get("default_bond_score", 0.5))
        self.min_bond_score = float(bonding_config.get("min_bond_score", 0.0))
        self.max_bond_score = float(bonding_config.get("max_bond_score", 1.0))
        self.weights = bonding_config.get("weights", {
            "curiosity": 0.25,
            "stability": 0.25,
            "coherence": 0.25,
            "personalized": 0.25
        })
        # Limit metadata list size for signature generation
        self.max_signature_metadata = int(bonding_config.get("max_signature_metadata", 30))
        # Batch/debounce state sync
        self._pending_state_syncs = set()
        self._last_sync_time = time.time()
        self._state_sync_interval = float(bonding_config.get("state_sync_interval", 2.0))
        self._sync_lock = threading.Lock()
        self._sync_thread = None
        self._sync_thread_stop = threading.Event()
        self._start_sync_thread()

    def _start_sync_thread(self):
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_thread_stop.clear()
            self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self._sync_thread.start()

    def _sync_worker(self):
        while not self._sync_thread_stop.is_set():
            time.sleep(self._state_sync_interval)
            with self._sync_lock:
                pending = list(self._pending_state_syncs)
                self._pending_state_syncs.clear()
            for sig_hash in pending:
                if self.state_manager and sig_hash in self.identified_users:
                    profile = self.identified_users[sig_hash]
                    def update_fn(state):
                        if hasattr(state, 'add_identified_user'):
                            state.add_identified_user(sig_hash, profile)
                        return state
                    self.state_manager.update_state_atomic(update_fn)

    def stop_sync_thread(self):
        self._sync_thread_stop.set()
        if self._sync_thread:
            self._sync_thread.join()

    def _initialize_config(self) -> None:
        """Initialize bonding score configuration."""
        try:
            bond_config = self.config_manager.get_section("bond_config", {})
            self.max_interactions = int(bond_config.get("max_interactions", 100))
            self.max_session_time = float(bond_config.get("max_session_time", 3600.0))
            self.decay_rate = float(bond_config.get("decay_rate", 0.95))
            self.decay_interval = float(bond_config.get("decay_interval", 86400.0))  # 86400.0 = 24 hours
            self.max_expected_dev = float(bond_config.get("max_expected_dev", 20.0))
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

    def _generate_signature(self, metadata_entries, extra_data: Optional[dict] = None, **kwargs) -> dict:
        """
        Generate a user signature from behavioral metadata and optional extra modalities.
        Args:
            metadata_entries: list of behavioral metadata dicts
            extra_data: dict of additional modality data (future-proof)
            kwargs: reserved for future extensions
        Returns:
            signature: dict
        """
        # Limit metadata list size
        metadata_entries = metadata_entries[-self.max_signature_metadata:]
        if not metadata_entries or not isinstance(metadata_entries, list):
            self.logger.log_warning("metadata_entries is empty or not a list in _generate_signature")
            return None

        valid_entries = []
        for idx, md in enumerate(metadata_entries):
            if not isinstance(md, dict):
                self.logger.log_warning(f"Skipping non-dict metadata entry at index {idx}")
                continue
            # Validate required keys and types
            device = md.get('device', None)
            timestamp = md.get('timestamp', None)
            content_metrics = md.get('content_metrics', {})
            if device is not None and not isinstance(device, str):
                self.logger.log_warning(f"Skipping entry at index {idx}: device is not a string")
                continue
            if timestamp is None or not isinstance(timestamp, (int, float)) or timestamp <= 0:
                self.logger.log_warning(f"Skipping entry at index {idx}: invalid timestamp {timestamp}")
                continue
            if not isinstance(content_metrics, dict):
                self.logger.log_warning(f"Skipping entry at index {idx}: content_metrics is not a dict")
                continue
            word_count = content_metrics.get('word_count', 0)
            sentiment_score = content_metrics.get('sentiment_score', 0.0)
            if (word_count is not None and not isinstance(word_count, (int, float))) or (isinstance(word_count, (int, float)) and word_count < 0):
                self.logger.log_warning(f"Skipping entry at index {idx}: invalid word_count {word_count}")
                continue
            if sentiment_score is not None and not isinstance(sentiment_score, (int, float)):
                self.logger.log_warning(f"Skipping entry at index {idx}: invalid sentiment_score {sentiment_score}")
                continue
            valid_entries.append(md)

        if not valid_entries:
            self.logger.log_warning("No valid metadata entries after validation in _generate_signature")
            return None

        # Extract features from valid entries
        devices = [md.get('device') for md in valid_entries if 'device' in md]
        device_fingerprint = devices[0] if devices else 'unknown'
        timestamps = [md.get('timestamp') for md in valid_entries if 'timestamp' in md]
        timestamps.sort()
        avg_gap = (sum(t2-t1 for t1, t2 in zip(timestamps, timestamps[1:])) / (len(timestamps)-1)) if len(timestamps) > 1 else 0.0
        activity_level = len(timestamps)
        avg_word_count = sum(md.get('content_metrics', {}).get('word_count', 0) for md in valid_entries) / len(valid_entries)
        avg_sentiment = sum(md.get('content_metrics', {}).get('sentiment_score', 0.0) for md in valid_entries) / len(valid_entries)
        # Compose signature dict
        signature = {
            'device_fingerprint': device_fingerprint,
            'avg_gap': avg_gap,
            'activity_level': activity_level,
            'avg_word_count': avg_word_count,
            'avg_sentiment': avg_sentiment
        }
        # Optionally, add extra modalities to signature (future)
        if extra_data:
            signature['extra'] = extra_data
        return signature

    def _hash_signature(self, signature):
        """Hash the signature dict to create a unique key."""
        sig_str = json.dumps(signature, sort_keys=True)
        return hashlib.sha256(sig_str.encode('utf-8')).hexdigest()

    def register_user_signature(self, metadata_entries, extra_data: Optional[dict] = None, **kwargs):
        """Register or update a user signature/profile from recent metadata, sync with central state if available."""
        signature = self._generate_signature(metadata_entries, extra_data=extra_data, **kwargs)
        if not signature:
            return None
        sig_hash = self._hash_signature(signature)
        with self.lock:
            if sig_hash not in self.identified_users:
                profile = {
                    'signature': signature,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'bond_score': self.default_bond_score,
                    'metadata_count': len(metadata_entries)
                }
                self.identified_users[sig_hash] = profile
                self.logger.record_event(
                    event_type="bond_user_signature_registered",
                    message=f"Registered new user signature: {sig_hash}",
                    level="info",
                    additional_info={'signature': signature}
                )
                # Batch state sync
                with self._sync_lock:
                    self._pending_state_syncs.add(sig_hash)
            else:
                # Only update if changed
                profile = self.identified_users[sig_hash]
                now = time.time()
                updated = False
                if abs(now - profile['last_seen']) > 1.0:
                    profile['last_seen'] = now
                    updated = True
                old_count = profile['metadata_count']
                new_count = old_count + len(metadata_entries)
                if new_count != old_count:
                    profile['metadata_count'] = new_count
                    updated = True
                if updated:
                    with self._sync_lock:
                        self._pending_state_syncs.add(sig_hash)
        return sig_hash

    def calculate_bond(self, user_id: str, metadata_entries, extra_data: Optional[dict] = None, **kwargs) -> float:
        """
        Calculate bond value using accessible metadata and optional extra modalities.
        Updates the bond score in UserProfileState.
        """
        # ... your bond calculation logic here ...
        # For demonstration, we'll just increment bond_score slightly for each call
        current_score = self.user_profile_state.get_bond_score(user_id)
        new_bond_score = min(self.max_bond_score, current_score + 0.01)
        self.user_profile_state.set_bond_score(user_id, new_bond_score)
        return new_bond_score

    def get_bond_score(self, user_id: str) -> float:
        return self.user_profile_state.get_bond_score(user_id)

    def set_bond_score(self, user_id: str, value: float) -> None:
        # Only update if changed (epsilon for float)
        current = self.user_profile_state.get_bond_score(user_id)
        if abs(current - value) > 1e-6:
            self.user_profile_state.set_bond_score(user_id, value)
            # Batch state sync
            with self._sync_lock:
                self._pending_state_syncs.add(user_id)

    def get_nickname(self, user_id: str) -> str:
        return self.user_profile_state.get_nickname(user_id)

    def set_nickname(self, user_id: str, value: str) -> None:
        # Only update if changed
        current = self.user_profile_state.get_nickname(user_id)
        if current != value:
            self.user_profile_state.set_nickname(user_id, value)
            # Batch state sync
            with self._sync_lock:
                self._pending_state_syncs.add(user_id)

    def get_all_profiles(self) -> Dict[str, Dict]:
        return self.user_profile_state.get_all_profiles()

    def assign_nickname_if_ready(self, user_id: str, early_text: str) -> None:
        profile = self.user_profile_state.get(user_id)
        if profile.get("nickname") or len(profile.get("early_interactions", [])) < self.user_profile_state.nickname_buffer_size:
            return
        try:
            from sovl_printer_unattached import Soulprinter
            system = None
            config_manager = self.config_manager
            soulprinter = Soulprinter(system, config_manager)
            nickname = soulprinter.extract_name(early_text, {})
            if not nickname or nickname.strip() == early_text.strip():
                nickname = soulprinter.extract_key_noun(early_text)
            if nickname and nickname.strip() and nickname.strip() != '[UNKNOWN]':
                self.user_profile_state.set_nickname(user_id, nickname.strip())
                self.logger.record_event(
                    event_type="nickname_assigned",
                    message="Nickname assigned to user",
                    level="info",
                    additional_info={"nickname": nickname, "user_id": user_id}
                )
        except Exception as e:
            self.logger.record_event(
                event_type="nickname_assignment_failed",
                message=f"Failed to assign nickname: {str(e)}",
                level="error",
                additional_info={"user_id": user_id}
            )

    @synchronized()
    def calculate_bonding_score(
        self,
        user_input: str,
        state: StateManager,
        error_manager: ErrorManager,
        context: SystemContext,
        curiosity_manager: Optional[CuriosityManager] = None,
        extra_data: Optional[dict] = None,
        **kwargs
    ) -> float:
        """
        Calculate bonding score using curiosity, stability, coherence, and personalized components.
        
        Args:
            user_input: User input string
            state: Current system state
            error_manager: Error manager instance
            context: System context
            curiosity_manager: Optional curiosity manager for novelty score
            extra_data: dict of additional modality data (future-proof)
            kwargs: reserved for future extensions
        
        Returns:
            Bonding score in [min_bond_score, max_bond_score]
        """
        try:
            if not isinstance(user_input, str):
                raise ValueError("user_input must be a string")
            if not state.history.conversation_id:
                raise ValueError("conversation_id must be set in state")

            # Get and update profile via UserProfileState
            conversation_id = state.history.conversation_id
            profile = state.user_profile_state.get(conversation_id)
            state.user_profile_state.update(conversation_id, user_input, state.session_start)

            # --- Nickname assignment logic moved here ---
            self.assign_nickname_if_ready(conversation_id, user_input)

            # Calculate component scores
            curiosity_score = (curiosity_manager.get_novelty_score(user_input)
                              if curiosity_manager else 0.5)
            stability_score = self._compute_stability_score(state)
            coherence_score = self._compute_coherence_score(user_input, state)
            wordprint_score = self._compute_wordprint_score(user_input, profile)
            knowing_score = self._compute_knowing_score(profile)
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
                    "conversation_id": conversation_id
                }
            )
            # Optionally, fuse extra modalities
            bond_score = self._fuse_modalities(profile, extra_data=extra_data, bond_score=bond_score, **kwargs)
            return bond_score

        except Exception as e:
            self.logger.record_event(
                event_type="bond_score_failed",
                message=f"Failed to calculate bond score: {str(e)}",
                level="error",
                additional_info={"error": str(e), "conversation_id": state.history.conversation_id}
            )
            error_manager.handle_data_error(
                e, {"user_input": user_input[:50]}, state.history.conversation_id
            )
            return self.default_bond_score

    def adjust_bond(self, user_id: str, delta: float) -> None:
        """
        Adjust the bond score for a user by a given delta, clamped to [min_bond_score, max_bond_score].
        Logs the adjustment. If user does not exist, logs a warning.
        """
        if not user_id:
            self.logger.log_warning("adjust_bond called with empty user_id", event_type="bond_adjust_warning")
            return
        with self.lock:
            profile = self.identified_users.get(user_id)
            if not profile or "bond_score" not in profile:
                self.logger.log_warning(f"adjust_bond: user_id {user_id} not found or missing bond_score", event_type="bond_adjust_warning")
                return
            old_score = profile["bond_score"]
            new_score = max(self.min_bond_score, min(self.max_bond_score, old_score + delta))
            profile["bond_score"] = new_score
            self.logger.record_event(
                event_type="bond_score_adjusted",
                message=f"Bond score for user {user_id} adjusted by {delta}",
                level="info",
                additional_info={"old_score": old_score, "new_score": new_score, "delta": delta, "user_id": user_id}
            )

    def _compute_wordprint_score(self, user_input: str, profile: dict) -> float:
        """Calculate wordprint score based on lexical signature and style consistency."""
        # Lexical Signature
        current_words = set(re.findall(r'\w+', user_input.lower()))
        top_words = sorted(profile["lexicon"].items(), key=lambda x: x[1], reverse=True)[:10]
        top_words_set = set(word for word, _ in top_words)
        signature_score = (len(current_words & top_words_set) / len(current_words | top_words_set)
                          if current_words and top_words_set else 0.5)
        # Style Consistency
        word_counts = [len(re.findall(r'\w+', input)) for input in profile["inputs"]]
        std_dev = torch.std(torch.tensor(word_counts, dtype=torch.float32)).item() if word_counts else 0.0
        style_score = 1 - min(std_dev / self.max_expected_dev, 1.0)
        # Combine
        wordprint_score = 0.6 * signature_score + 0.4 * style_score
        return max(0.0, min(1.0, wordprint_score))

    def _compute_knowing_score(self, profile: dict) -> float:
        """Calculate duration of knowing score based on interaction count and time."""
        # Apply decay if inactive
        if time.time() - profile["last_interaction"] > self.decay_interval:
            profile["interactions"] = int(profile["interactions"] * self.decay_rate)
            profile["session_time"] *= self.decay_rate
            profile["last_interaction"] = time.time()
        # Interaction Familiarity
        familiarity = min(profile["interactions"] / self.max_interactions, 1.0)
        # Time Duration
        duration = min(profile["session_time"] / self.max_session_time, 1.0)
        # Combine
        knowing_score = 0.7 * familiarity + 0.3 * duration
        return max(0.0, min(1.0, knowing_score))

    def _compute_stability_score(self, state: StateManager) -> float:
        """Placeholder for stability score (system health/errors)."""
        return 0.5  # Neutral default

    def _compute_coherence_score(self, user_input: str, state: StateManager) -> float:
        """Placeholder for coherence score (input-response alignment)."""
        return 0.5  # Neutral default

    def _fuse_modalities(self, signature: dict, extra_data: Optional[dict] = None, bond_score: float = 0.5, **kwargs) -> float:
        """
        Fuse additional modalities (e.g., facial, voice, device) into the bond score.
        Args:
            signature: user signature dict
            extra_data: dict of additional modality data. Expected keys (optional): 
                - 'face_score': float in [0,1] (confidence from facial recognition)
                - 'voice_score': float in [0,1] (confidence from voice analysis)
                - ...future modalities...
            bond_score: current bond score (from text/behavioral analysis)
            kwargs: reserved for future extensions
        Returns:
            bond_score: float (possibly modified)
        Notes:
            - If extra_data is provided but contains unrecognized or invalid keys, a warning is logged.
            - If no recognized modalities are present, the original bond_score is returned.
        """
        try:
            if extra_data is None:
                return bond_score

            # Validate and extract modality scores
            modality_scores = {}
            for key in ['face_score', 'voice_score']:
                if key in extra_data:
                    val = extra_data[key]
                    if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                        self.logger.log_warning(
                            f"extra_data['{key}'] is not a float in [0,1]: {val}",
                            event_type="bond_modality_validation_warning"
                        )
                        continue
                    modality_scores[key] = float(val)

            if not modality_scores:
                self.logger.log_warning(
                    "extra_data provided to _fuse_modalities but no recognized modalities found. Ignoring.",
                    event_type="bond_modality_unused_warning"
                )
                return bond_score

            # Compute weighted average using modality_weights config
            weights = self.modality_weights.copy() if hasattr(self, 'modality_weights') else {'text': 1.0}
            total_weight = weights.get('text', 1.0)
            fused_score = bond_score * total_weight
            for key, score in modality_scores.items():
                w = weights.get(key.replace('_score', ''), 0.0)
                fused_score += score * w
                total_weight += w

            if total_weight > 0:
                fused_score /= total_weight
            else:
                fused_score = bond_score  # fallback

            return max(self.min_bond_score, min(self.max_bond_score, fused_score))
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed in _fuse_modalities: {str(e)}",
                error_type="bond_modality_fusion_error",
                stack_trace=traceback.format_exc()
            )
            return bond_score

class BondModulator:
    """Active component to provide bond-based modulation for system interaction based on user perception."""
    def __init__(self, bond_calculator):
        self.bond_calculator = bond_calculator

    def get_bond_modulation(self, metadata_entries, extra_data: Optional[dict] = None, **kwargs):
        if not self.bond_calculator.enable_bonding:
            return self.bond_calculator.context_neutral, self.bond_calculator.default_bond_score
        sig_hash = self.bond_calculator.register_user_signature(metadata_entries, extra_data=extra_data, **kwargs)
        bond_score = self.bond_calculator.get_bond_score(sig_hash)
        if bond_score is None:
            bond_score = self.bond_calculator.default_bond_score

        # Map bond_score to a modulation context using configurable thresholds and contexts
        if bond_score > self.bond_calculator.strong_bond_threshold:
            context = self.bond_calculator.context_strong
        elif bond_score < self.bond_calculator.weak_bond_threshold:
            context = self.bond_calculator.context_weak
        else:
            context = self.bond_calculator.context_neutral

        return context, bond_score
