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
from sovl_processor import SimpleTokenizer
import queue
from collections import OrderedDict

class BondCalculator:
    """
    Calculates bonding score and manages user bond/nickname via UserProfileState.
    NOTE: The state_manager object must be thread-safe and non-blocking. This class now uses a queue-based decoupling approach for all state syncs to avoid deadlocks.
    Implements pruning for archived_users and an LRU cap for identified_users to prevent unbounded memory growth.
    state_manager must be set before the sync thread is started, either via __init__ or set_state_manager().
    """
    def __init__(self, config_manager: ConfigManager, logger: Logger, user_profile_state: 'UserProfileState', state_manager=None):
        if not config_manager or not logger or not user_profile_state:
            raise ValueError("config_manager, logger, and user_profile_state cannot be None")
        self.config_manager = config_manager
        self.logger = logger
        self.user_profile_state = user_profile_state
        self.state_manager = state_manager
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
        # Queue-based sync mechanism only
        self._sync_queue = queue.Queue()
        self._last_sync_time = time.time()
        self._state_sync_interval = float(bonding_config.get("state_sync_interval", 2.0))
        self._sync_thread = None
        self._sync_thread_stop = threading.Event()
        self._start_sync_thread()
        self.tokenizer = SimpleTokenizer()
        self.similarity_threshold = float(config_manager.get_section("bonding_config", {}).get("signature_similarity_threshold", 0.15))
        self.max_identified_users = int(bonding_config.get("max_identified_users", 10000))
        self.archived_retention_days = int(bonding_config.get("archived_retention_days", 365))
        self.identified_users = OrderedDict()  # user_id (hash) -> profile, LRU
        self.lock = Lock()
        self.archived_users = {}  # user_id (hash) -> profile
        self.signature_history_maxlen = int(self.config_manager.get_section("bonding_config", {}).get("signature_history_maxlen", 100))
        self.signature_history_maxdays = int(self.config_manager.get_section("bonding_config", {}).get("signature_history_maxdays", 180))
        self.drift_threshold = float(self.config_manager.get_section("bonding_config", {}).get("drift_threshold", 0.25))
        self.drift_consecutive = int(self.config_manager.get_section("bonding_config", {}).get("drift_consecutive", 5))
        self.archive_timeout_days = int(self.config_manager.get_section("bonding_config", {}).get("archive_timeout_days", 365))
        self.decay_lambda = float(self.config_manager.get_section("bonding_config", {}).get("decay_lambda", 0.01))

    def _start_sync_thread(self):
        if self._sync_thread is None or not self._sync_thread.is_alive():
            self._sync_thread_stop.clear()
            self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self._sync_thread.start()

    def _sync_worker(self):
        """
        Background worker for syncing identified_users to state_manager.
        Uses a queue-based approach for deadlock-free operation.
        """
        while not self._sync_thread_stop.is_set():
            if self.state_manager is None:
                self.logger.log_error(
                    error_msg="state_manager is not set in BondCalculator; skipping sync.",
                    error_type="state_manager_missing",
                )
                time.sleep(self._state_sync_interval)
                continue
            try:
                user_id = self._sync_queue.get(timeout=self._state_sync_interval)
                self._sync_user(user_id)
            except queue.Empty:
                continue

    def _sync_user(self, user_id):
        if hasattr(self, 'state_manager') and self.state_manager and user_id in self.identified_users:
            profile = self.identified_users[user_id]
            def update_fn(state):
                if hasattr(state, 'add_identified_user'):
                    state.add_identified_user(user_id, profile)
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
        Generate a user signature using a simple, robust formula.
        """
        metadata_entries = metadata_entries[-self.max_signature_metadata:]
        if not metadata_entries or not isinstance(metadata_entries, list):
            self.logger.log_warning("metadata_entries is empty or not a list in _generate_signature")
            return None
        valid_entries = []
        for idx, md in enumerate(metadata_entries):
            if not isinstance(md, dict):
                continue
            device = md.get('device', None)
            timestamp = md.get('timestamp', None)
            content_metrics = md.get('content_metrics', {})
            if device is not None and not isinstance(device, str):
                continue
            if timestamp is None or not isinstance(timestamp, (int, float)) or timestamp <= 0:
                continue
            if not isinstance(content_metrics, dict):
                continue
            word_count = content_metrics.get('word_count', 0)
            avg_sentence_length = content_metrics.get('avg_sentence_length', 0)
            if (word_count is not None and not isinstance(word_count, (int, float))) or (isinstance(word_count, (int, float)) and word_count < 0):
                continue
            if avg_sentence_length is not None and not isinstance(avg_sentence_length, (int, float)):
                continue
            # Token stats (if present)
            token_stats = md.get('token_stats', {})
            valid_entries.append({
                'device': device,
                'word_count': word_count,
                'avg_sentence_length': avg_sentence_length,
                'token_stats': token_stats,
            })
        if not valid_entries:
            self.logger.log_warning("No valid metadata entries after validation in _generate_signature")
            return None
        # Most common device
        devices = [md['device'] for md in valid_entries if md['device']]
        device_fingerprint = max(set(devices), key=devices.count) if devices else 'unknown'
        # Averages
        avg_word_count = sum(md['word_count'] for md in valid_entries) / len(valid_entries)
        avg_sentence_length = sum(md['avg_sentence_length'] for md in valid_entries) / len(valid_entries)
        # Token and bigram stats
        all_tokens = []
        all_bigrams = []
        for md in valid_entries:
            token_stats = md['token_stats']
            tokens = token_stats.get('tokens', [])
            all_tokens.extend(tokens)
            bigrams = list(zip(tokens[:-1], tokens[1:])) if len(tokens) > 1 else []
            all_bigrams.extend(bigrams)
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        token_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        total_bigrams = len(all_bigrams)
        unique_bigrams = len(set(all_bigrams))
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0
        signature = {
            'device': device_fingerprint,
            'avg_word_count': avg_word_count,
            'token_diversity': token_diversity,
            'bigram_diversity': bigram_diversity,
            'avg_sentence_length': avg_sentence_length
        }
        return signature

    def _hash_signature(self, signature):
        """Hash the signature dict to create a unique key."""
        sig_str = json.dumps(signature, sort_keys=True)
        return hashlib.sha256(sig_str.encode('utf-8')).hexdigest()

    def _signature_distance(self, sig1: dict, sig2: dict) -> float:
        """Compute Euclidean distance between two signatures (excluding device)."""
        keys = ['avg_word_count', 'token_diversity', 'bigram_diversity', 'avg_sentence_length']
        return sum((sig1.get(k, 0.0) - sig2.get(k, 0.0)) ** 2 for k in keys) ** 0.5

    def _prune_signature_history(self, profile):
        now = time.time()
        # Prune by maxlen
        profile['signature_history'] = profile['signature_history'][-self.signature_history_maxlen:]
        # Prune by maxdays
        profile['signature_history'] = [s for s in profile['signature_history'] if (now - s.get('timestamp', now)) < self.signature_history_maxdays * 86400]

    def _time_weighted_average_signature(self, signature_history):
        now = time.time()
        weights = []
        values = []
        for sig in signature_history:
            age_days = (now - sig.get('timestamp', now)) / 86400
            weight = pow(2.718, -self.decay_lambda * age_days)
            weights.append(weight)
            values.append([sig.get('avg_word_count', 0), sig.get('token_diversity', 0), sig.get('bigram_diversity', 0), sig.get('avg_sentence_length', 0)])
        if not weights or not values:
            return None
        total_weight = sum(weights)
        avg = [sum(v[i] * weights[i] for i in range(len(weights))) / total_weight for v in zip(*values)]
        return {
            'avg_word_count': avg[0],
            'token_diversity': avg[1],
            'bigram_diversity': avg[2],
            'avg_sentence_length': avg[3]
        }

    def _signature_variance(self, signature_history):
        if not signature_history:
            return 0.0
        avg = self._time_weighted_average_signature(signature_history)
        if not avg:
            return 0.0
        keys = ['avg_word_count', 'token_diversity', 'bigram_diversity', 'avg_sentence_length']
        var = 0.0
        for sig in signature_history:
            var += sum((sig.get(k, 0.0) - avg[k]) ** 2 for k in keys)
        return var / (len(signature_history) * len(keys))

    def _adaptive_similarity_threshold(self, profile):
        var = self._signature_variance(profile['signature_history'])
        base = self.similarity_threshold
        if var < 0.01:
            return max(0.05, base * 0.5)
        elif var > 0.1:
            return min(0.5, base * 2)
        return base

    def _drift_detection(self, profile, new_signature):
        avg = self._time_weighted_average_signature(profile['signature_history'])
        if not avg:
            return False
        dist = self._signature_distance(new_signature, avg)
        profile.setdefault('drift_counter', 0)
        if dist > self.drift_threshold:
            profile['drift_counter'] += 1
        else:
            profile['drift_counter'] = 0
        return profile['drift_counter'] >= self.drift_consecutive

    def _cluster_split(self, profile):
        # Simple: compare mean of last 10 vs previous mean
        history = profile['signature_history']
        if len(history) < 20:
            return False
        last10 = history[-10:]
        prev10 = history[-20:-10]
        avg_last = self._time_weighted_average_signature(last10)
        avg_prev = self._time_weighted_average_signature(prev10)
        if not avg_last or not avg_prev:
            return False
        dist = self._signature_distance(avg_last, avg_prev)
        return dist > self.drift_threshold * 2

    def _enforce_identified_users_cap(self):
        # Remove oldest users if over cap
        while len(self.identified_users) > self.max_identified_users:
            oldest_uid, oldest_profile = self.identified_users.popitem(last=False)
            oldest_profile['archived_at'] = time.time()
            self.archived_users[oldest_uid] = oldest_profile
            self.logger.record_event(
                event_type="identified_user_evicted",
                message=f"Evicted user {oldest_uid} from identified_users due to LRU cap.",
                level="info",
                additional_info={}
            )

    def _prune_archived_users(self):
        now = time.time()
        retention_seconds = self.archived_retention_days * 86400
        to_remove = [uid for uid, p in self.archived_users.items()
                     if now - p.get('archived_at', p.get('last_seen', now)) > retention_seconds]
        for uid in to_remove:
            del self.archived_users[uid]
        if to_remove:
            self.logger.record_event(
                event_type="archived_users_pruned",
                message=f"Pruned {len(to_remove)} archived users.",
                level="info",
                additional_info={}
            )

    def _archive_dormant_profiles(self):
        now = time.time()
        to_archive = [uid for uid, p in self.identified_users.items() if (now - p['last_seen']) > self.archive_timeout_days * 86400]
        for uid in to_archive:
            profile = self.identified_users.pop(uid)
            profile['archived_at'] = now
            self.archived_users[uid] = profile
            self.logger.record_event(
                event_type="profile_archived",
                message=f"Archived dormant profile: {uid}",
                level="info",
                additional_info={}
            )
        self._prune_archived_users()

    def register_user_signature(self, metadata_entries, extra_data: Optional[dict] = None, **kwargs):
        signature = self._generate_signature(metadata_entries, extra_data=extra_data, **kwargs)
        if not signature:
            return None
        signature['timestamp'] = time.time()
        sig_hash = self._hash_signature(signature)
        with self.lock:
            self._archive_dormant_profiles()
            # 1. Exact match (legacy)
            if sig_hash in self.identified_users:
                profile = self.identified_users[sig_hash]
                now = time.time()
                profile['last_seen'] = now
                profile['metadata_count'] += len(metadata_entries)
                profile.setdefault('signature_history', []).append(signature)
                self._prune_signature_history(profile)
                # Move to end to mark as recently used
                self.identified_users.move_to_end(sig_hash)
                return sig_hash
            # 2. Soft match
            best_match = None
            best_dist = float('inf')
            for uid, profile in self.identified_users.items():
                threshold = self._adaptive_similarity_threshold(profile)
                dist = self._signature_distance(signature, profile['signature'])
                if dist < best_dist:
                    best_dist = dist
                    best_match = uid
            if best_match is not None:
                profile = self.identified_users[best_match]
                threshold = self._adaptive_similarity_threshold(profile)
                if best_dist <= threshold:
                    now = time.time()
                    profile['last_seen'] = now
                    profile['metadata_count'] += len(metadata_entries)
                    profile.setdefault('signature_history', []).append(signature)
                    self._prune_signature_history(profile)
                    self.identified_users.move_to_end(best_match)
                    # Drift/cluster detection
                    if self._drift_detection(profile, signature):
                        self.logger.record_event(
                            event_type="profile_drift_detected",
                            message=f"Drift detected for user {best_match}",
                            level="info",
                            additional_info={}
                        )
                    if self._cluster_split(profile):
                        self.logger.record_event(
                            event_type="profile_cluster_split",
                            message=f"Cluster split for user {best_match}",
                            level="info",
                            additional_info={}
                        )
                        profile['archived_at'] = time.time()
                        self.archived_users[best_match] = self.identified_users.pop(best_match)
                        self._prune_archived_users()
                        # Create new profile below
                    else:
                        profile.setdefault('related_user_ids', []).append(sig_hash)
                        profile['bond_score'] = (profile['bond_score'] + self.default_bond_score) / 2
                        return best_match
            # 3. New profile
            profile = {
                'signature': signature,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'bond_score': self.default_bond_score,
                'metadata_count': len(metadata_entries),
                'signature_history': [signature],
                'related_user_ids': []
            }
            self.identified_users[sig_hash] = profile
            self._enforce_identified_users_cap()
            self.logger.record_event(
                event_type="bond_user_signature_registered",
                message=f"Registered new user signature: {sig_hash}",
                level="info",
                additional_info={'signature': signature}
            )
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
            self._sync_queue.put(user_id)

    def get_nickname(self, user_id: str) -> str:
        return self.user_profile_state.get_nickname(user_id)

    def set_nickname(self, user_id: str, value: str) -> None:
        # Only update if changed
        current = self.user_profile_state.get_nickname(user_id)
        if current != value:
            self.user_profile_state.set_nickname(user_id, value)
            self._sync_queue.put(user_id)

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
                self.logger.log_warning(
                    f"adjust_bond: user_id {user_id} not found or missing bond_score (may have been archived/evicted)",
                    event_type="bond_adjust_warning"
                )
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

    def set_state_manager(self, state_manager):
        """
        Set or update the state_manager after initialization.
        """
        self.state_manager = state_manager

class BondModulator:
    """Active component to provide bond-based modulation for system interaction based on user perception."""
    def __init__(self, bond_calculator, max_retries=3):
        self.bond_calculator = bond_calculator
        self.max_retries = max_retries

    def get_bond_modulation(self, metadata_entries, extra_data: Optional[dict] = None, **kwargs):
        """
        Attempts to get bond modulation with a retry limit. Returns safe defaults on persistent failure.
        """
        retries = 0
        last_exception = None
        while retries < self.max_retries:
            try:
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
            except Exception as e:
                last_exception = e
                if hasattr(self.bond_calculator, 'logger'):
                    self.bond_calculator.logger.log_error(
                        error_msg=f'BondModulator.get_bond_modulation failed on attempt {retries+1}: {str(e)}',
                        error_type='bond_modulation_error'
                    )
                retries += 1

        # Circuit breaker: persistent failure
        if hasattr(self.bond_calculator, 'logger'):
            self.bond_calculator.logger.log_error(
                error_msg=f'BondModulator.get_bond_modulation failed after {self.max_retries} retries. Returning safe defaults. Last error: {str(last_exception)}',
                error_type='bond_modulation_circuit_breaker'
            )
        return self.bond_calculator.context_neutral, self.bond_calculator.default_bond_score
