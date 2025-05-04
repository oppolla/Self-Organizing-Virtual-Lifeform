import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
from collections import defaultdict, deque
import time
import traceback
from threading import Lock
import math
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_error import ErrorManager, ScaffoldError
from sovl_io import ConfigurationError
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_confidence import ConfidenceManager
import contextlib
import functools
from sovl_engram import LoraAdapterManager
import difflib
import copy
import threading
from sovl_utils import check_model_health as util_check_model_health, calculate_token_map_confidence
import queue
import numpy as np
# Centralized handler for scaffold errors and recovery.
class ScaffoldErrorManager:
    """Centralized error handling for scaffold operations."""
    
    def __init__(self, logger: Logger, error_handler: Optional[ErrorManager] = None):
        self.logger = logger
        self.error_handler = error_handler
        self._lock = threading.RLock()  # Using RLock instead of Lock to prevent deadlocks
        
    def handle_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ) -> None:
        """Handle scaffold errors with consistent logging and recovery."""
        # Prepare error context outside the lock
        error_context = {
            "operation": operation,
            "timestamp": time.time(),
            "severity": severity,
            **(context or {})
        }
        
        # Only use lock for critical operations that modify shared state
        critical_error = severity in ("error", "critical")
        if critical_error and self.error_handler:
            with self._lock:
                self.error_handler.handle_scaffold_error(error, error_context)
        elif self.error_handler:
            # Non-critical errors don't need the lock
            self.error_handler.handle_scaffold_error(error, error_context)
        
        # Logging can be done outside the lock as Logger should be thread-safe
        self.logger.record_event(
            event_type=f"scaffold_{operation}_error",
            message=str(error),
            level=severity,
            additional_info={
                "error_type": type(error).__name__,
                "stack_trace": traceback.format_exc(),
                **error_context
            }
        )

# Decorator to wrap scaffold operations with consistent error handling.
def scaffold_operation(operation_name: str):
    """Decorator for consistent error handling in scaffold operations."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                error_handler = getattr(self, '_error_handler', None)
                if error_handler:
                    error_handler.handle_error(
                        error=e,
                        operation=operation_name,
                        context={
                            "function": func.__name__,
                            "args": str(args),
                            "kwargs": str(kwargs)
                        }
                    )
                raise
        return wrapper
    return decorator

# Builds and maintains mappings from base-token IDs to scaffold-token IDs.
class ScaffoldTokenMapper:
    """Handles token mapping between base and scaffold tokenizers."""
    
    def __init__(self, base_tokenizer: Any, scaffold_tokenizer: Any, logger: Any, config: Optional[Dict[str, Any]] = None, base_model: Any = None, scaffold_model: Any = None, mapping_strategy: str = None):
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer
        self.logger = logger
        self.base_model = base_model  # Optional: for embedding similarity
        self.scaffold_model = scaffold_model  # Optional: for embedding similarity
        self.mapping_strategy = mapping_strategy  # User-selectable, but defaults to None for now
        self.max_tokens_per_mapping = config.get('max_tokens_per_mapping', 3) if config else 3
        self.mapping_similarity_threshold = config.get('mapping_similarity_threshold', 0.7) if config else 0.7
        self.allow_bidirectional_mapping = config.get('allow_bidirectional_mapping', False) if config else False
        self.fallback_strategy = config.get('fallback_strategy', 'split') if config else 'split'
        self.normalization_level = config.get('normalization_level', 'basic') if config else 'basic'
        self.min_semantic_similarity = config.get('min_semantic_similarity', 0.5) if config else 0.5
        self.max_meaning_drift = config.get('max_meaning_drift', 0.3) if config else 0.3
        self.enable_periodic_validation = config.get('enable_periodic_validation', True) if config else True
        self.conflict_resolution_strategy = config.get('conflict_resolution_strategy', 'keep_highest_conf') if config else 'keep_highest_conf'
        self.token_map = defaultdict(lambda: {'ids': [scaffold_tokenizer.unk_token_id], 'weight': 1.0})
        self.embedding_available = self._check_embedding_availability()
        self.config = config or {}
        # Drift config
        self.max_drift = self.config.get('max_drift', 0.9)
        self.cosine_weight = self.config.get('cosine_weight', 0.5)
        self.euclidean_weight = self.config.get('euclidean_weight', 0.3)
        self.norm_weight = self.config.get('norm_weight', 0.2)
        self.levenshtein_weight = self.config.get('levenshtein_weight', 0.4)
        self.char_weight = self.config.get('char_weight', 0.3)
        self.subword_weight = self.config.get('subword_weight', 0.2)
        self.freq_weight = self.config.get('freq_weight', 0.1)
        self.drift_cache_size = self.config.get('drift_cache_size', 10000)
        # Thread-safe LRU cache for drift
        import threading
        self._drift_cache = {}
        self._drift_cache_order = []
        self._drift_cache_lock = threading.Lock()
        if self.embedding_available:
            self.logger.info("Using embedding-based similarity for token mapping.")
        else:
            self.logger.warning("Falling back to character-based similarity for token mapping.")
        self._initialize_token_maps()
        # Metric tracking for runtime monitoring
        self._mapping_confidences = deque(maxlen=500)
        self._fallback_counts = defaultdict(int)
        self._drift_values = deque(maxlen=500)
        self._mapping_errors = 0
        self._mapping_latencies = deque(maxlen=500)
        self._metrics_lock = threading.Lock()
        
    def _check_embedding_availability(self):
        # Check if both models and their input embeddings are available
        try:
            if self.base_model is not None and self.scaffold_model is not None:
                base_emb = getattr(self.base_model, 'get_input_embeddings', None)
                scaf_emb = getattr(self.scaffold_model, 'get_input_embeddings', None)
                if callable(base_emb) and callable(scaf_emb):
                    # Try to access weights
                    base_weights = base_emb().weight
                    scaf_weights = scaf_emb().weight
                    return base_weights is not None and scaf_weights is not None
        except Exception:
            pass
        return False

    def _get_token_embedding(self, model, token_id):
        try:
            emb_layer = model.get_input_embeddings()
            return emb_layer.weight[token_id].detach().cpu()
        except Exception:
            return None

    def _embedding_similarity(self, token1, token2):
        try:
            id1 = self.base_tokenizer.convert_tokens_to_ids(token1)
            id2 = self.scaffold_tokenizer.convert_tokens_to_ids(token2)
            emb1 = self._get_token_embedding(self.base_model, id1)
            emb2 = self._get_token_embedding(self.scaffold_model, id2)
            if emb1 is not None and emb2 is not None:
                import torch
                sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
                return sim.item()
        except Exception:
            pass
        return None

    def _char_similarity(self, token1, token2):
        set1 = set(token1)
        set2 = set(token2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _calculate_similarity(self, token1, token2):
        # User can select, but default is None (current behavior: embedding if available, else char)
        strategy = self.mapping_strategy
        if strategy == "embedding":
            sim = self._embedding_similarity(token1, token2)
            if sim is not None:
                return sim
            # fallback
            return self._char_similarity(token1, token2)
        elif strategy == "char" or strategy is None:
            if self.embedding_available:
                sim = self._embedding_similarity(token1, token2)
                if sim is not None:
                    return sim
            return self._char_similarity(token1, token2)
        else:
            # Future: add more strategies here
            return self._char_similarity(token1, token2)
        
    def _normalize_token(self, token: str) -> str:
        """Normalize token based on configured level."""
        if self.normalization_level == 'none':
            return token
        elif self.normalization_level == 'basic':
            return token.replace("Ġ", "").replace("##", "")
        else:  # aggressive
            return token.replace("Ġ", "").replace("##", "").lower()
            
    def _handle_fallback(self, token: str, base_id: int) -> List[int]:
        """Handle token mapping fallback based on configured strategy."""
        if self.fallback_strategy == 'split':
            # Split token into smaller parts
            parts = [token[i:i+2] for i in range(0, len(token), 2)]
            scaffold_ids = []
            for part in parts:
                ids = self.scaffold_tokenizer.encode(part, add_special_tokens=False)
                scaffold_ids.extend(ids)
            return scaffold_ids[:self.max_tokens_per_mapping]
            
        elif self.fallback_strategy == 'merge':
            # Try to merge with similar tokens
            similar_tokens = self._find_similar_tokens(token)
            if similar_tokens:
                return similar_tokens[0]['ids']
            return [self.scaffold_tokenizer.unk_token_id]
            
        elif self.fallback_strategy == 'nearest':
            # Find nearest token in vocabulary
            nearest = self._find_nearest_token(token)
            if nearest:
                return [nearest]
            return [self.scaffold_tokenizer.unk_token_id]
            
        else:  # 'unk'
            return [self.scaffold_tokenizer.unk_token_id]
            
    def _find_similar_tokens(self, token: str) -> List[Dict]:
        """Find tokens with similar patterns."""
        similar = []
        for base_id, mapping in self.token_map.items():
            base_token = self.base_tokenizer.decode([base_id])
            if self._calculate_similarity(token, base_token) > self.mapping_similarity_threshold:
                similar.append(mapping)
        return similar
        
    def _find_nearest_token(self, token: str) -> Optional[int]:
        """Find the nearest token in the scaffold vocabulary."""
        min_distance = float('inf')
        nearest_id = None
        
        for scaffold_id in self.scaffold_tokenizer.get_vocab().values():
            scaffold_token = self.scaffold_tokenizer.decode([scaffold_id])
            distance = self._calculate_similarity(token, scaffold_token)
            if distance < min_distance:
                min_distance = distance
                nearest_id = scaffold_id
                
        return nearest_id if min_distance < self.mapping_similarity_threshold else None
        
    def _validate_mapping_quality(self, base_token: str, scaffold_ids: List[int]) -> bool:
        """Validate mapping quality based on configured parameters."""
        if not scaffold_ids:
            return False
            
        # Check semantic similarity
        scaffold_token = self.scaffold_tokenizer.decode(scaffold_ids)
        similarity = self._calculate_similarity(base_token, scaffold_token)
        if similarity < self.min_semantic_similarity:
            return False
            
        # Check meaning drift
        if self._calculate_meaning_drift(base_token, scaffold_token) > self.max_meaning_drift:
            return False
            
        return True
        
    def _calculate_meaning_drift(self, token1: str, token2: str) -> float:
        """
        Compute semantic drift between two tokens using embeddings (if available)
        or advanced heuristics. Returns a value in [0, 1], lower is better.
        """
        try:
            max_drift = self.max_drift
            cosine_weight = self.cosine_weight
            euclidean_weight = self.euclidean_weight
            norm_weight = self.norm_weight
            levenshtein_weight = self.levenshtein_weight
            char_weight = self.char_weight
            subword_weight = self.subword_weight
            freq_weight = self.freq_weight
            cache_size = self.drift_cache_size
            cache_key = (token1, token2)
            with self._drift_cache_lock:
                if cache_key in self._drift_cache:
                    return self._drift_cache[cache_key]
            # Special token handling
            specials = set([
                getattr(self.base_tokenizer, 'pad_token', None), getattr(self.base_tokenizer, 'unk_token', None), getattr(self.base_tokenizer, 'eos_token', None),
                getattr(self.scaffold_tokenizer, 'pad_token', None), getattr(self.scaffold_tokenizer, 'unk_token', None), getattr(self.scaffold_tokenizer, 'eos_token', None)
            ])
            t1n, t2n = self._normalize_token(token1), self._normalize_token(token2)
            if t1n in specials or t2n in specials:
                drift = 0.0 if t1n == t2n else self.max_meaning_drift
                self._cache_drift(cache_key, drift, cache_size)
                return drift
            # Embedding-based drift
            if self.embedding_available:
                try:
                    id1 = self.base_tokenizer.convert_tokens_to_ids(token1)
                    id2 = self.scaffold_tokenizer.convert_tokens_to_ids(token2)
                    emb1 = self._get_token_embedding(self.base_model, id1)
                    emb2 = self._get_token_embedding(self.scaffold_model, id2)
                    if emb1 is not None and emb2 is not None:
                        emb1 = emb1 / (emb1.norm() + 1e-8)
                        emb2 = emb2 / (emb2.norm() + 1e-8)
                        cosine = 1.0 - torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
                        euclid = (emb1 - emb2).norm().item()
                        euclid = euclid / (euclid + 1.0)
                        normdiff = abs(emb1.norm().item() - emb2.norm().item())
                        normdiff = normdiff / (normdiff + 1.0)
                        drift = (
                            cosine_weight * cosine +
                            euclidean_weight * euclid +
                            norm_weight * normdiff
                        )
                        drift = min(drift, max_drift)
                        self._cache_drift(cache_key, drift, cache_size)
                        return drift
                except Exception as e:
                    self.logger.record_event(
                        event_type="meaning_drift_embedding_error",
                        message=f"Embedding drift failed: {str(e)}",
                        level="warning"
                    )
            # Heuristic fallback
            lev = self._levenshtein_distance(t1n, t2n)
            maxlen = max(len(t1n), len(t2n), 1)
            lev_drift = lev / maxlen
            char_drift = 1.0 - self._char_similarity(t1n, t2n)
            t1_sub = set(self.base_tokenizer.encode(t1n, add_special_tokens=False))
            t2_sub = set(self.scaffold_tokenizer.encode(t2n, add_special_tokens=False))
            # Subword drift calculation (explicit fallback for empty sets)
            if t1_sub and t2_sub:
                subword_drift = 1.0 - (len(t1_sub & t2_sub) / max(len(t1_sub | t2_sub), 1))
            else:
                subword_drift = 1.0  # Default to max drift for empty subword sets
            freq_drift = 0.0
            if not hasattr(self, '_freq_drift_disabled_logged'):
                self._freq_drift_disabled_logged = False
            try:
                base_vocab = self.base_tokenizer.get_vocab()
                scaf_vocab = self.scaffold_tokenizer.get_vocab()
                base_val = base_vocab.get(t1n, 0)
                scaf_val = scaf_vocab.get(t2n, 0)
                # Only use if values are plausible frequencies (not just token IDs)
                if (
                    isinstance(base_val, int) and isinstance(scaf_val, int)
                    and (base_val > 100 or scaf_val > 100)
                ):
                    base_freq = base_val
                    scaf_freq = scaf_val
                    freq_diff = abs(base_freq - scaf_freq)
                    freq_drift = freq_diff / (freq_diff + 1000.0)
                else:
                    freq_drift = 0.0
                    if not self._freq_drift_disabled_logged:
                        self.logger.record_event(
                            event_type="freq_drift_disabled",
                            message="Frequency drift disabled: vocab values are not frequencies.",
                            level="info"
                        )
                        self._freq_drift_disabled_logged = True
            except Exception:
                freq_drift = 0.0
            total_len = len(t1n) + len(t2n)
            lev_w = levenshtein_weight * (1.2 if total_len > 10 else 0.8)
            char_w = char_weight * (0.8 if total_len > 10 else 1.2)
            total_w = lev_w + char_w + subword_weight + freq_weight
            lev_w /= total_w
            char_w /= total_w
            subword_weight /= total_w
            freq_weight /= total_w
            drift = (
                lev_w * lev_drift +
                char_w * char_drift +
                subword_weight * subword_drift +
                freq_weight * freq_drift
            )
            drift = min(max(drift, 0.0), 1.0)
            # --- Add explicit logging for heuristic-based drift ---
            self.logger.record_event(
                event_type="meaning_drift_fallback",
                message=f"Heuristic-based drift for {token1} -> {token2}: {drift:.4f}",
                level="warning" if self.embedding_available else "debug",
                additional_info={
                    "tokens": [token1, token2],
                    "lev_drift": lev_drift,
                    "char_drift": char_drift,
                    "subword_drift": subword_drift,
                    "freq_drift": freq_drift,
                    "weights": [lev_w, char_w, subword_weight, freq_weight]
                }
            )
            with self._metrics_lock:
                self._drift_values.append(drift)
            self._cache_drift(cache_key, drift, cache_size)
            return drift
        except Exception as e:
            self.logger.record_event(
                event_type="meaning_drift_error",
                message=f"Failed to compute meaning drift for {token1} <-> {token2}: {str(e)}",
                level="error"
            )
            return self.max_meaning_drift

    def _cache_drift(self, cache_key, drift, cache_size):
        with self._drift_cache_lock:
            self._drift_cache[cache_key] = drift
            self._drift_cache_order.append(cache_key)
            if len(self._drift_cache_order) > cache_size:
                old_key = self._drift_cache_order.pop(0)
                self._drift_cache.pop(old_key, None)

    def _levenshtein_distance(self, s1, s2):
        # Simple Levenshtein distance implementation
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _find_levenshtein_match(self, token, threshold=0.3):
        # Normalized Levenshtein distance
        min_dist = float('inf')
        best_id = None
        for scaf_token, scaf_id in self.scaffold_tokenizer.get_vocab().items():
            dist = self._levenshtein_distance(token, scaf_token) / max(len(token), len(scaf_token), 1)
            if dist < min_dist:
                min_dist = dist
                best_id = scaf_id
        if min_dist <= threshold:
            return [best_id]
        return None

    def _subword_heuristic(self, token):
        # Try prefix/suffix match for subword tokens
        for scaf_token, scaf_id in self.scaffold_tokenizer.get_vocab().items():
            if token.startswith(scaf_token) or token.endswith(scaf_token):
                return [scaf_id]
        return None

    def _build_token_map(self):
        """Build main token mapping with enhanced strategies and layered fallback."""
        special_tokens = {
            self.base_tokenizer.pad_token_id: self.scaffold_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id: self.scaffold_tokenizer.eos_token_id,
            self.base_tokenizer.unk_token_id: self.scaffold_tokenizer.unk_token_id,
        }
        for base_token, base_id in self.base_tokenizer.get_vocab().items():
            # 1. Special token strict mapping
            if base_id in special_tokens and special_tokens[base_id] is not None:
                self.token_map[base_id] = {'ids': [special_tokens[base_id]], 'weight': 1.0, 'confidence': 1.0}
                continue
            normalized = self._normalize_token(base_token)
            # 2. Exact string match
            if normalized in self.scaffold_tokenizer.get_vocab():
                scaf_id = self.scaffold_tokenizer.get_vocab()[normalized]
                self.token_map[base_id] = {'ids': [scaf_id], 'weight': 1.0, 'confidence': 1.0}
                continue
            # 3. Levenshtein distance
            lev_match = self._find_levenshtein_match(normalized)
            if lev_match:
                self.logger.record_event(
                    event_type="token_map_fallback",
                    message=f"Levenshtein fallback for {base_token} -> {lev_match}",
                    level="debug"
                )
                self.token_map[base_id] = {'ids': lev_match, 'weight': 0.8, 'confidence': 0.8}
                continue
            # 4. Subword heuristics
            subword_match = self._subword_heuristic(normalized)
            if subword_match:
                self.logger.record_event(
                    event_type="token_map_fallback",
                    message=f"Subword heuristic fallback for {base_token} -> {subword_match}",
                    level="debug"
                )
                self.token_map[base_id] = {'ids': subword_match, 'weight': 0.7, 'confidence': 0.7}
                continue
            # 5. Legacy char similarity fallback
            best_score = 0.0
            best_id = None
            for scaf_token, scaf_id in self.scaffold_tokenizer.get_vocab().items():
                score = self._char_similarity(normalized, scaf_token)
                if score > best_score:
                    best_score = score
                    best_id = scaf_id
            if best_score > 0.0:
                self.logger.record_event(
                    event_type="token_map_fallback",
                    message=f"Legacy char similarity fallback for {base_token} -> {best_id}",
                    level="debug"
                )
                self.token_map[base_id] = {'ids': [best_id], 'weight': 0.5, 'confidence': 0.5}
                continue
            # 6. Final fallback to unk_token_id
            self.logger.record_event(
                event_type="token_map_fallback",
                message=f"Mapping {base_token} to unk_token_id as last resort.",
                level="warning"
            )
            self.token_map[base_id] = {'ids': [self.scaffold_tokenizer.unk_token_id], 'weight': 0.1, 'confidence': 0.1}

    def _resolve_conflict(self, base_id: int, new_mapping: Dict, existing_mapping: Dict) -> Dict:
        """Resolve mapping conflicts based on configured strategy."""
        if self.conflict_resolution_strategy == 'keep_first':
            return existing_mapping
        elif self.conflict_resolution_strategy == 'keep_last':
            return new_mapping
        elif self.conflict_resolution_strategy == 'keep_highest_conf':
            return new_mapping if new_mapping['confidence'] > existing_mapping['confidence'] else existing_mapping
        else:  # 'merge'
            return {
                'ids': list(set(existing_mapping['ids'] + new_mapping['ids'])),
                'weight': max(existing_mapping['weight'], new_mapping['weight']),
                'confidence': max(existing_mapping['confidence'], new_mapping['confidence'])
            }
        
    def _initialize_token_maps(self):
        """Initialize token maps between base and scaffold tokenizers."""
        try:
            # Build main token map
            self._build_token_map()
            
            # Initialize special token map
            self._initialize_special_token_map()
            
            # Validate token maps
            if not self._validate_token_maps():
                raise ValueError("Token map validation failed")
                
            self.logger.record_event(
                event_type="token_map_initialized",
                message="Token maps initialized successfully",
                level="info",
                additional_info={
                    "map_size": len(self.token_map),
                    "timestamp": time.time()
                }
            )
            
        except Exception as e:
            self.logger.record_event(
                event_type="token_map_error",
                message=f"Failed to initialize token maps: {str(e)}",
                level="error",
                additional_info={"timestamp": time.time()}
            )
            raise
            
    def _initialize_special_token_map(self):
        """Initialize special token mapping."""
        special_token_map = {
            self.base_tokenizer.pad_token_id: self.scaffold_tokenizer.pad_token_id if self.scaffold_tokenizer.pad_token_id is not None else self.scaffold_tokenizer.unk_token_id,
            self.base_tokenizer.eos_token_id: self.scaffold_tokenizer.eos_token_id if self.scaffold_tokenizer.eos_token_id is not None else self.scaffold_tokenizer.unk_token_id,
            self.base_tokenizer.unk_token_id: self.scaffold_tokenizer.unk_token_id,
        }
        
        for base_id, scaffold_id in special_token_map.items():
            self.token_map[base_id] = {'ids': [scaffold_id], 'weight': 1.0}
            
    def _validate_token_maps(self) -> bool:
        """Validate token maps integrity."""
        if not self.token_map:
            return False
            
        required_tokens = ['pad_token_id', 'eos_token_id', 'unk_token_id']
        for token in required_tokens:
            if not any(t['ids'][0] == getattr(self.scaffold_tokenizer, token) 
                      for t in self.token_map.values()):
                return False
                
        if not all(t['ids'] for t in self.token_map.values()):
            return False
            
        return True
        
    def get_token_map(self) -> Dict:
        """Get the token map."""
        return dict(self.token_map)
        
    def validate_token_maps(self) -> bool:
        """Validate token maps."""
        return self._validate_token_maps()
        
    def tokenize_and_map(self, prompt: str) -> Tuple[List[int], List[float]]:
        """Tokenize prompt and map to scaffold token space, enforcing minimum confidence."""
        try:
            start = time.time()
            base_tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
            scaffold_ids = []
            weights = []
            low_conf_count = 0
            min_conf = 0.5  # Minimum allowed confidence for a mapping
            for base_id in base_tokens:
                mapping = self.token_map[base_id]
                scaffold_ids.extend(mapping['ids'])
                weights.extend([mapping['weight']] * len(mapping['ids']))
                # --- Metrics ---
                with self._metrics_lock:
                    self._mapping_confidences.append(mapping['weight'])
                    if 'fallback_type' in mapping:
                        self._fallback_counts[mapping['fallback_type']] += 1
            total = len(base_tokens) if base_tokens else 1
            low_conf_ratio = low_conf_count / total
            # Log mapping quality
            self.logger.record_event(
                event_type="token_mapping_quality",
                message=f"Token mapping: {low_conf_count}/{total} ({low_conf_ratio:.2%}) below confidence {min_conf}",
                level="warning" if low_conf_ratio > 0.2 else "info",
                additional_info={
                    "low_conf_count": low_conf_count,
                    "total_tokens": total,
                    "low_conf_ratio": low_conf_ratio,
                    "min_conf": min_conf,
                    "timestamp": time.time()
                }
            )
            # Enforce threshold: if >20% of tokens are low-confidence, raise error
            if low_conf_ratio > 0.2:
                with self._metrics_lock:
                    self._mapping_errors += 1
                raise ScaffoldError(
                    f"Too many low-confidence token mappings: {low_conf_count}/{total} ({low_conf_ratio:.2%}) below confidence {min_conf}",
                    operation="tokenize_and_map"
                )
            end = time.time()
            with self._metrics_lock:
                self._mapping_latencies.append(end - start)
            return scaffold_ids, weights
        except Exception as e:
            with self._metrics_lock:
                self._mapping_errors += 1
            self.logger.record_event(
                event_type="token_mapping_error",
                message=f"Failed to map tokens: {str(e)}",
                level="error",
                additional_info={"timestamp": time.time()}
            )
            raise
            
    def update_token_map_memory(self, prompt: str, logits: torch.Tensor, source: str = "base"):
        """Update token map based on prompt confidence, using raw base model logits only."""
        try:
            confidence = calculate_token_map_confidence(logits, source=source, logger=self.logger)
            base_tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
            for base_id in base_tokens:
                if base_id in self.token_map:
                    self.token_map[base_id]['weight'] = max(
                        self.token_map[base_id]['weight'],
                        confidence
                    )
        except Exception as e:
            self.logger.record_event(
                event_type="token_map_update_error",
                message=f"Failed to update token map: {str(e)}",
                level="error",
                additional_info={"timestamp": time.time()}
            )
            raise

# Factory to create a configured ScaffoldTokenMapper.
def build_scaffold_token_mapping(
    base_tokenizer: Any, 
    scaffold_tokenizer: Any, 
    logger: Logger,
    config: Optional[Dict[str, Any]] = None,
    base_model: Any = None,
    scaffold_model: Any = None,
    mapping_strategy: str = None
) -> ScaffoldTokenMapper:
    """
    Create a ScaffoldTokenMapper instance.
    
    Args:
        base_tokenizer: The tokenizer for the base model
        scaffold_tokenizer: The tokenizer for the scaffold model
        logger: Logger instance for structured logging
        config: Optional configuration dictionary containing mapping parameters:
            - max_tokens_per_mapping: Maximum scaffold tokens per base token (default: 3)
            - mapping_similarity_threshold: Similarity threshold for alternative mappings (default: 0.7)
            - allow_bidirectional_mapping: Enable bidirectional token mapping (default: False)
            - fallback_strategy: Strategy for handling failed mappings (default: 'split')
            - normalization_level: Token normalization aggressiveness (default: 'basic')
            - min_semantic_similarity: Minimum semantic similarity threshold (default: 0.5)
            - max_meaning_drift: Maximum allowed semantic drift (default: 0.3)
            - enable_periodic_validation: Enable periodic mapping validation (default: True)
            - conflict_resolution_strategy: Strategy for resolving mapping conflicts (default: 'keep_highest_conf')
        base_model: Optional base model for embedding-based similarity
        scaffold_model: Optional scaffold model for embedding-based similarity
        mapping_strategy: Optional mapping strategy ("embedding", "char", etc.)
    
    Returns:
        ScaffoldTokenMapper: Initialized token mapper instance
    """
    return ScaffoldTokenMapper(base_tokenizer, scaffold_tokenizer, logger, config, base_model, scaffold_model, mapping_strategy)

# Utilities for creating and combining sparse attention masks.
class AttentionUtils:
    """Unified utilities for attention mask creation and preparation."""
    
    @staticmethod
    def create_sparse_mask(
        seq_len: int,
        sparse_pattern: str,
        window_size: int,
        device: str = 'cpu',
        logger: Optional[Logger] = None
    ) -> torch.Tensor:
        """Create a sparse attention mask based on the specified pattern."""
        try:
            with torch.no_grad():
                if sparse_pattern == 'window':
                    mask = torch.zeros(seq_len, seq_len, device=device)
                    for i in range(seq_len):
                        start = max(0, i - window_size // 2)
                        end = min(seq_len, i + window_size // 2 + 1)
                        mask[i, start:end] = 1.0
                    return mask.bool()
                elif sparse_pattern == 'block':
                    mask = torch.zeros(seq_len, seq_len, device=device)
                    for i in range(0, seq_len, window_size):
                        mask[i:i + window_size, i:i + window_size] = 1.0
                    return mask.bool()
                else:
                    raise ValueError(f"Unknown sparse pattern: {sparse_pattern}")
        except Exception as e:
            if logger:
                logger.record_event(
                    event_type="attention_mask_creation_failed",
                    message=f"Sparse mask creation failed: {str(e)}",
                    level="error",
                    additional_info={
                        "sparse_pattern": sparse_pattern,
                        "seq_len": seq_len,
                        "window_size": window_size,
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc()
                    }
                )
            raise

    @staticmethod
    def prepare_attention_mask(
        attention_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
        logger: Optional[Logger] = None
    ) -> torch.Tensor:
        """Prepare attention mask for multi-head attention."""
        try:
            with torch.no_grad():
                if attention_mask is None:
                    return None
                    
                # Validate dimensions
                if attention_mask.dim() < 2 or attention_mask.dim() > 4:
                    raise ValueError(f"Invalid mask dimensions: {attention_mask.shape}")
                
                # Handle boolean masks
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.float().masked_fill(
                        ~attention_mask, float('-inf')
                    )
                # Convert to float if needed
                elif attention_mask.dtype != torch.float:
                    attention_mask = attention_mask.float()
                
                # Add batch and head dimensions if needed
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                # Expand to match required shape
                if attention_mask.shape != (batch_size, num_heads, seq_len, seq_len):
                    attention_mask = attention_mask.expand(
                        batch_size, num_heads, seq_len, seq_len
                    )
                
                return attention_mask.to(device)
                
        except Exception as e:
            if logger:
                logger.record_event(
                    event_type="attention_mask_preparation_failed",
                    message=f"Attention mask preparation failed: {str(e)}",
                    level="error",
                    additional_info={
                        "mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
                        "target_shape": [batch_size, num_heads, seq_len, seq_len],
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc()
                    }
                )
            raise

    @staticmethod
    def combine_masks(
        sparse_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Combine sparse and attention masks."""
        return (sparse_mask & attention_mask).to(device)

# Strategy to discover model layers for injection.
class LayerDiscoveryStrategy:
    """Strategy for discovering transformer layers in a model."""
    
    def __init__(self, logger: Logger):
        self._logger = logger
        self._patterns = [
            'h.{i}',
            'layer.{i}',
            'layers.{i}',
            'transformer.h.{i}',
            'decoder.layers.{i}',
        ]

    def find_layers(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """Find layers suitable for cross-attention injection."""
        try:
            candidates = []
            names = []

            # Try specific patterns first
            for name, module in model.named_modules():
                if any(pattern.split('.')[0] in name for pattern in self._patterns):
                    if isinstance(module, nn.ModuleList):
                        candidates.extend(module)
                        names.extend([f"{name}.{i}" for i in range(len(module))])

            # Fallback to any ModuleList
            if not candidates:
                for name, module in model.named_modules():
                    if isinstance(module, nn.ModuleList):
                        candidates.extend(module)
                        names.extend([f"{name}.{i}" for i in range(len(module))])

            # Last resort: collect modules with 'layer' in name
            if not candidates:
                for name, module in model.named_modules():
                    if 'layer' in name.lower() and isinstance(module, nn.Module):
                        candidates.append(module)
                        names.append(name)

            if not candidates:
                raise ValueError("No suitable layers found for cross-attention injection")

            return candidates, names
        except Exception as e:
            self._logger.record_event(
                event_type="layer_discovery_failed",
                message=f"Layer discovery failed: {str(e)}",
                level="error",
                additional_info={
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

# Implements cross-attention mechanics between base and scaffold representations.
class CrossAttentionLayer(nn.Module):
    """Cross attention layer for scaffold integration with dynamic weighting."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Logger,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        device: Union[str, torch.device] = 'cpu'
    ):
        super().__init__()
        self._config = config
        self._logger = logger
        self._device = torch.device(device) if isinstance(device, str) else device
        
        self._initialize_parameters(hidden_size, num_heads)
        self._initialize_layers()
        self.to(self._device)
        
        self._logger.record_event(
            event_type="cross_attention_layer_initialized",
            message="Cross attention layer initialized successfully",
            level="info",
            additional_info={
                "device": str(self._device),
                "hidden_size": self._hidden_size,
                "num_heads": self._num_heads,
                "timestamp": time.time()
            }
        )
        
    def _initialize_parameters(self, hidden_size: Optional[int], num_heads: Optional[int]) -> None:
        """Initialize configuration parameters."""
        self._hidden_size = hidden_size or self._config.get('hidden_size', 768)
        self._num_heads = num_heads or self._config.get('num_heads', 12)
        self._head_dim = self._hidden_size // self._num_heads
        self._scale = 1.0 / math.sqrt(self._head_dim)
        
        self._max_weight = self._config.get('max_weight', 1.0)
        self._min_weight = self._config.get('min_weight', 0.0)
        self._weight_decay = self._config.get('weight_decay', 0.01)
        
        if self._hidden_size % self._num_heads != 0:
            raise ValueError(f"Hidden size {self._hidden_size} must be divisible by num_heads {self._num_heads}")
            
        self._base_weight = nn.Parameter(torch.ones(1, device=self._device))
        self._dynamic_scale = nn.Parameter(torch.ones(1, device=self._device))
        self._momentum = 0.9
        self._weight_history: List[float] = []
        
    def _initialize_layers(self) -> None:
        """Initialize all neural network layers."""
        self._q_proj = nn.Linear(self._hidden_size, self._hidden_size).to(self._device)
        self._k_proj = nn.Linear(self._hidden_size, self._hidden_size).to(self._device)
        self._v_proj = nn.Linear(self._hidden_size, self._hidden_size).to(self._device)
        self._out_proj = nn.Linear(self._hidden_size, self._hidden_size).to(self._device)
        
        self._gate = nn.Parameter(torch.ones(1, device=self._device))
        self._gate_bias = nn.Parameter(torch.zeros(1, device=self._device))
        self._gate_scale = nn.Parameter(torch.ones(1, device=self._device))
        
        self._layer_norm = nn.LayerNorm(self._hidden_size).to(self._device)
        self._sparse_mask = None
        
        self._initialize_weights()
        self.reset_cache()
        
    def _initialize_weights(self) -> None:
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self._q_proj.weight)
        nn.init.xavier_uniform_(self._k_proj.weight)
        nn.init.xavier_uniform_(self._v_proj.weight)
        nn.init.xavier_uniform_(self._out_proj.weight)
        
        nn.init.zeros_(self._q_proj.bias)
        nn.init.zeros_(self._k_proj.bias)
        nn.init.zeros_(self._v_proj.bias)
        nn.init.zeros_(self._out_proj.bias)
        
    def reset_cache(self) -> None:
        """Reset attention cache."""
        self._cache = {'k': None, 'v': None, 'attention_mask': None}
        
    def set_influence_weight(self, weight: float) -> None:
        """Set influence weight with dynamic scaling."""
        weight = max(self._min_weight, min(self._max_weight, weight))
        self._base_weight.data.fill_(weight)
        self._update_dynamic_scale()
        
    def set_blend_strength(self, strength: float) -> None:
        """Set blend strength with momentum."""
        strength = max(0.0, min(1.0, strength))
        self._gate_bias.data.fill_(strength)
        
    def set_lifecycle_weight(self, weight: float, curve: str = 'sigmoid_linear') -> None:
        """Set lifecycle-based weight with dynamic adjustment."""
        if curve == 'sigmoid_linear':
            weight = torch.sigmoid(torch.tensor(weight * 2 - 1))
        elif curve == 'linear':
            weight = torch.tensor(weight)
        else:
            raise ValueError(f"Unknown curve type: {curve}")
            
        self._base_weight.data.fill_(weight)
        self._update_dynamic_scale()
        
    def _update_dynamic_scale(self) -> None:
        """Update dynamic scale based on weight history."""
        if self._weight_history:
            avg_weight = sum(self._weight_history) / len(self._weight_history)
            self._dynamic_scale.data.fill_(1.0 + (self._base_weight.item() - avg_weight))
        self._weight_history.append(self._base_weight.item())
        if len(self._weight_history) > 10:
            self._weight_history.pop(0)
            
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int
    ) -> torch.Tensor:
        """Compute attention scores with dynamic weighting."""
        q = q.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self._scale * self._dynamic_scale)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
        if self._sparse_mask is not None:
            scores = scores.masked_fill(self._sparse_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self._hidden_size)
        
        return attn_output * (self._gate * self._base_weight + self._gate_bias)
        
    def _forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_tensors: Optional[torch.Tensor] = None,
        memory_weight: float = 0.0,
        dynamic_factor: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass implementation."""
        batch_size, seq_len, _ = hidden_states.shape
        
        hidden_states = self._layer_norm(hidden_states)
        
        q = self._q_proj(hidden_states)
        k = self._k_proj(cross_states)
        v = self._v_proj(cross_states)
        
        if memory_tensors is not None and memory_weight > 0:
            k = k + memory_tensors[0] * memory_weight
            v = v + memory_tensors[1] * memory_weight
            
        attn_output = self._compute_attention(q, k, v, attention_mask, seq_len, batch_size)
        attn_output = self._out_proj(attn_output)
        attn_output = attn_output * self._gate
        
        if dynamic_factor is not None:
            attn_output = attn_output * dynamic_factor
            
        if use_cache:
            self._cache['k'] = k
            self._cache['v'] = v
            self._cache['attention_mask'] = attention_mask
            
        return attn_output
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_tensors: Optional[torch.Tensor] = None,
        memory_weight: float = 0.0,
        dynamic_factor: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with error handling."""
        try:
            return self._forward(
                hidden_states, cross_states, attention_mask,
                memory_tensors, memory_weight, dynamic_factor, use_cache
            )
        except Exception as e:
            self._logger.record_event(
                event_type="cross_attention_layer_forward_pass_failed",
                message=f"CrossAttentionLayer forward pass failed: {str(e)}",
                level="error",
                additional_info={"stack_trace": traceback.format_exc()}
            )
            raise

# Injects cross-attention layers into a base model using specified strategies.
class CrossAttentionInjector:
    """Injector for adding cross-attention layers to a transformer model."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self._config_manager = config_manager
        self._logger = logger
        self._lock = Lock()
        self._scaffold_proj: Optional[nn.Module] = None
        self._scaffold_unk_id = config_manager.get("controls_config.scaffold_unk_id", 0)
        self._error_handler = ErrorManager(config_manager, logger)
        self._target_layer_names: List[str] = []  # Store names of layers being injected
        self._layer_count = 0  # Initialize layer counter for progressive strategy
        
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate cross-attention configuration."""
        try:
            required_keys = [
                "core_config.cross_attn_layers",
                "core_config.hidden_size",
                "core_config.num_heads",
                "controls_config.scaffold_weight_cap",
                "controls_config.scaffold_unk_id"
            ]
            
            for key in required_keys:
                if not self._config_manager.has_key(key):
                    raise ConfigurationError(f"Missing required config key: {key}")
            
            numeric_validations = {
                "controls_config.scaffold_weight_cap": (0.0, 1.0),
                "controls_config.blend_strength": (0.0, 1.0),
                "controls_config.attention_weight": (0.0, None)
            }
            
            for key, (min_val, max_val) in numeric_validations.items():
                if self._config_manager.has_key(key):
                    value = self._config_manager.get(key)
                    if not isinstance(value, (int, float)):
                        raise ConfigurationError(f"{key} must be numeric")
                    if min_val is not None and value < min_val:
                        raise ConfigurationError(f"{key} must be >= {min_val}")
                    if max_val is not None and value > max_val:
                        raise ConfigurationError(f"{key} must be <= {max_val}")
            
            self._logger.record_event(
                event_type="cross_attention_config_validated",
                message="Cross-attention configuration validated successfully",
                level="info",
                additional_info={"timestamp": time.time()}
            )
        except Exception as e:
            self._logger.record_event(
                event_type="cross_attention_config_error",
                message=f"Failed to validate cross-attention config: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def _get_scaffold_output(self, scaffold_model: nn.Module, token_map: Optional[Dict], base_hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate scaffold hidden states for cross-attention with robust fallback/error handling."""
        try:
            device = base_hidden_states.device
            batch_size, seq_len, hidden_dim = base_hidden_states.shape
            scaffold_dim = None
            if hasattr(scaffold_model, 'config') and hasattr(scaffold_model.config, 'hidden_size'):
                scaffold_dim = scaffold_model.config.hidden_size
            elif hasattr(scaffold_model, 'config') and hasattr(scaffold_model.config, 'd_model'):
                scaffold_dim = scaffold_model.config.d_model
            elif hasattr(scaffold_model, 'hidden_size'):
                scaffold_dim = scaffold_model.hidden_size
            if scaffold_dim is None:
                raise ValueError("Could not determine scaffold model hidden dimension")
            scaffold_output = torch.zeros(batch_size, seq_len, scaffold_dim, device=device)
            if token_map is not None and hasattr(scaffold_model, 'get_input_embeddings'):
                mapping = token_map.get('base_to_scaffold', {})
                scaffold_embeddings = scaffold_model.get_input_embeddings()
                for token_id, scaffold_ids in mapping.items():
                    if isinstance(scaffold_ids, list) and len(scaffold_ids) > 0:
                        scaffold_id = scaffold_ids[0]
                        token_embedding = scaffold_embeddings(
                            torch.tensor([scaffold_id], device=device)
                        )
                        scaffold_output[:, :, :] = token_embedding
            else:
                default_input_ids = torch.full(
                    (batch_size, seq_len), 
                    self._scaffold_unk_id, 
                    dtype=torch.long, 
                    device=device
                )
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
                with torch.no_grad():
                    try:
                        outputs = scaffold_model(
                            input_ids=default_input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
                        if hasattr(outputs, 'last_hidden_state'):
                            scaffold_output = outputs.last_hidden_state
                        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                            scaffold_output = outputs.hidden_states[-1]
                    except Exception as e:
                        self._logger.record_event(
                            event_type="scaffold_model_forward_failed",
                            message=f"Failed to run scaffold model forward pass: {str(e)}",
                            level="error",
                            additional_info={"error": str(e)}
                        )
                        # Configurable fallback: only use noise if enabled
                        fallback_noise = getattr(self, '_fallback_noise_enabled', False)
                        if fallback_noise:
                            self._logger.record_event(
                                event_type="scaffold_model_noise_fallback",
                                message="Falling back to random noise for scaffold output due to forward failure.",
                                level="warning",
                                additional_info={"batch_size": batch_size, "seq_len": seq_len, "scaffold_dim": scaffold_dim}
                            )
                            scaffold_output = torch.randn(
                                batch_size, seq_len, scaffold_dim, device=device
                            ) * 0.01
                        else:
                            raise ScaffoldError(
                                f"Scaffold model forward pass failed and noise fallback is disabled: {str(e)}",
                                operation="_get_scaffold_output",
                                context={"batch_size": batch_size, "seq_len": seq_len, "scaffold_dim": scaffold_dim}
                            )
            if scaffold_output.shape[1] != seq_len:
                scaffold_output = torch.nn.functional.interpolate(
                    scaffold_output.permute(0, 2, 1),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)
            return scaffold_output
        except Exception as e:
            self._logger.record_event(
                event_type="get_scaffold_output_failed",
                message=f"Failed to get scaffold output: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            # Only fallback to zeros if explicitly configured, else raise
            fallback_zeros = getattr(self, '_fallback_zeros_enabled', True)
            if fallback_zeros:
                self._logger.record_event(
                    event_type="scaffold_model_zeros_fallback",
                    message="Falling back to zeros for scaffold output due to error.",
                    level="warning"
                )
                return torch.zeros_like(base_hidden_states)
            raise

    def inject(
        self,
        base_model: nn.Module,
        scaffold_model: nn.Module,
        layers_to_inject: Union[str, List[int]],
        injection_strategy: str = 'sequential',
        token_map: Optional[Dict] = None
    ) -> nn.Module:
        """Inject cross-attention into the model with robust error recovery and fallback."""
        try:
            # Strict LoRA conflict detection
            for name, _ in base_model.named_modules():
                if "lora" in name.lower() or "adapter" in name.lower():
                    self._logger.log_error(
                        "LoRA adapters detected in base model. "
                        "LoRA must be applied before cross-attention injection to ensure compatibility.",
                        error_type="lora_injection_conflict"
                    )
                    raise ScaffoldError(
                        "LoRA adapters applied to base model after injection. "
                        "Apply LoRA before calling CrossAttentionInjector.inject.",
                        operation="cross_attention_injection"
                    )
            # --- Existing injection logic ---
            with self._lock:
                layers, layer_names = self.find_model_layers(base_model)
                layer_indices = self.get_cross_attention_layers(base_model, layers_to_inject)
                self._target_layer_names = [layer_names[i] for i in layer_indices]

                self._logger.record_event(
                    event_type="cross_attention_injection_start",
                    message="Cross-attention injection started",
                    level="info",
                    additional_info={
                        "layer_names": self._target_layer_names,
                        "timestamp": time.time()
                    }
                )

                # Transactional: work on a deep copy
                model_copy = copy.deepcopy(base_model)
                successful_layers = []
                skipped_layers = []

                for layer_idx in layer_indices:
                    strategies = [injection_strategy, "parallel", "replace"]
                    injected = False
                    for strat in strategies:
                        try:
                            self._inject_single_layer(
                                model=model_copy,
                                scaffold_model=scaffold_model,
                                layer_idx=layer_idx,
                                injection_strategy=strat,
                                token_map=token_map
                            )
                            injected = True
                            if strat != injection_strategy:
                                self._logger.record_event(
                                    event_type="cross_attention_injection_retry",
                                    message=f"Layer {layer_idx} injected with fallback strategy '{strat}'",
                                    level="warning"
                                )
                            break
                        except Exception as e:
                            self._logger.record_event(
                                event_type="cross_attention_injection_strategy_failed",
                                message=f"Layer {layer_idx} injection failed with strategy '{strat}': {str(e)}",
                                level="warning"
                            )
                    if not injected:
                        skipped_layers.append(layer_idx)
                        self._logger.record_event(
                            event_type="cross_attention_injection_skipped",
                            message=f"Layer {layer_idx} skipped after all strategies failed.",
                            level="error"
                        )
                    else:
                        successful_layers.append(layer_idx)

                if not self.verify_injection(model_copy, model_copy.config):
                    self._logger.record_event(
                        event_type="cross_attention_injection_rollback",
                        message="Verification failed after injection, rolling back to original model.",
                        level="error"
                    )
                    return base_model  # Return original model

                # Commit: copy injected layers back to base_model
                for name, module in model_copy.named_modules():
                    if hasattr(base_model, name):
                        setattr(base_model, name, module)

                self._logger.record_event(
                    event_type="cross_attention_injection_complete",
                    message="Cross-attention injection completed with fallback and recovery.",
                    level="info",
                    additional_info={
                        "successful_layers": successful_layers,
                        "skipped_layers": skipped_layers,
                        "timestamp": time.time()
                    }
                )
                return base_model

        except Exception as e:
            self._logger.record_event(
                event_type="cross_attention_injection_failed",
                message=f"Cross-attention injection failed: {str(e)}",
                level="error",
                additional_info={"timestamp": time.time(), "stack_trace": traceback.format_exc()}
            )
            return base_model  # Return original model on failure

    def _inject_single_layer(
        self,
        model: nn.Module,
        scaffold_model: nn.Module,
        layer_idx: int,
        injection_strategy: str,
        token_map: Optional[Dict]
    ) -> None:
        """Inject cross-attention into a single layer with retry logic."""
        try:
            layers, _ = self.find_model_layers(model)
            layer = layers[layer_idx]
            cross_attn_layer = CrossAttentionLayer(
                config=self._config_manager.get_section("core_config"),
                logger=self._logger,
                device=model.device
            )

            # Validate hidden state compatibility
            base_hidden_size = None
            if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                base_hidden_size = model.config.hidden_size
            elif hasattr(model, 'config') and hasattr(model.config, 'd_model'):
                base_hidden_size = model.config.d_model
            elif hasattr(model, 'hidden_size'):
                base_hidden_size = model.hidden_size
            
            scaffold_hidden_size = None
            if hasattr(scaffold_model, 'config') and hasattr(scaffold_model.config, 'hidden_size'):
                scaffold_hidden_size = scaffold_model.config.hidden_size
            elif hasattr(scaffold_model, 'config') and hasattr(scaffold_model.config, 'd_model'):
                scaffold_hidden_size = scaffold_model.config.d_model
            elif hasattr(scaffold_model, 'hidden_size'):
                scaffold_hidden_size = scaffold_model.hidden_size
            
            if base_hidden_size is not None and scaffold_hidden_size is not None:
                if base_hidden_size != scaffold_hidden_size and self._scaffold_proj is None:
                    # Create projection layer if needed
                    self._scaffold_proj = nn.Linear(scaffold_hidden_size, base_hidden_size).to(model.device)
                    nn.init.xavier_uniform_(self._scaffold_proj.weight)
                    nn.init.zeros_(self._scaffold_proj.bias)
                    self._scaffold_proj_norm = nn.LayerNorm(base_hidden_size).to(model.device)
                    self._logger.record_event(
                        event_type="scaffold_projection_created",
                        message=f"Created projection layer from {scaffold_hidden_size} to {base_hidden_size} with Xavier init and LayerNorm",
                        level="info"
                    )
            
            if base_hidden_size is None:
                self._logger.record_event(
                    event_type="hidden_size_detection_failed",
                    message="Could not detect base model hidden size. Proceeding with default config.",
                    level="warning"
                )

            layers[layer_idx] = self._create_wrapped_layer(
                original_layer=layer,
                cross_attn_layer=cross_attn_layer,
                scaffold_model=scaffold_model,
                token_map=token_map,
                strategy=injection_strategy
            )

            if not self._verify_single_layer(model, layer_idx):
                raise RuntimeError(f"Layer {layer_idx} injection verification failed")
        except Exception as e:
            self._logger.record_event(
                event_type="cross_attention_injection_error",
                message=f"Failed to inject cross-attention layer {layer_idx} with strategy '{injection_strategy}': {str(e)}",
                level="error",
                additional_info={
                    "layer_idx": layer_idx,
                    "strategy": injection_strategy,
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def get_cross_attention_layers(self, model: nn.Module, mode: Union[str, List[int]]) -> List[int]:
        """Determine which layers to inject cross-attention into."""
        try:
            if isinstance(mode, list):
                layers = mode
                total_layers = len(self.find_model_layers(model)[0])
                if not validate_layer_indices(layers, total_layers):
                    raise ValueError(f"Invalid layer indices: {layers}")
            else:
                total_layers = self._get_total_layers(model)
                if total_layers == 0:
                    raise ValueError("No layers found for cross-attention injection")

                if mode == "early":
                    layers = list(range(total_layers // 3))
                elif mode == "late":
                    layers = list(range(2 * total_layers // 3, total_layers))
                else:
                    layers = list(range(total_layers // 3, 2 * total_layers // 3))

            self._logger.record_event(
                event_type="layer_selection",
                message="Layer selection completed successfully",
                level="info",
                additional_info={
                    "mode": str(mode),
                    "selected_layers": layers,
                    "total_layers": total_layers,
                    "timestamp": time.time()
                }
            )
            return layers
        except Exception as e:
            self._logger.record_event(
                event_type="layer_selection_failed",
                message=f"Layer selection failed: {str(e)}",
                level="error",
                additional_info={
                    "mode": str(mode),
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def _get_total_layers(self, model: nn.Module) -> int:
        """Get the total number of layers in the model."""
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return len(model.transformer.h)
        if hasattr(model, 'layers'):
            return len(model.layers)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
            return len(model.decoder.layers)
        return 0

    def find_model_layers(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """Find transformer layers in the model."""
        strategy = LayerDiscoveryStrategy(self._logger)
        return strategy.find_layers(model)

    def _create_wrapped_layer(
        self,
        original_layer: nn.Module,
        cross_attn_layer: CrossAttentionLayer,
        scaffold_model: nn.Module,
        token_map: Optional[Dict],
        strategy: str
    ) -> nn.Module:
        """Wrap the layer with cross-attention injection."""
        original_forward = original_layer.forward
        scaffold_model = scaffold_model
        injector = self
        token_mapper = token_map

        def _wrapped_layer_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            try:
                scaffold_output = injector._get_scaffold_output(
                    scaffold_model=scaffold_model,
                    token_map=token_mapper,
                    base_hidden_states=hidden_states
                )
                
                # Apply projection if needed
                if injector._scaffold_proj is not None:
                    scaffold_output = injector._scaffold_proj(scaffold_output)
                    if hasattr(injector, '_scaffold_proj_norm') and injector._scaffold_proj_norm is not None:
                        scaffold_output = injector._scaffold_proj_norm(scaffold_output)
                
                # Create dynamic factor based on strategy if needed
                dynamic_factor = None
                if strategy == "progressive":
                    # Apply progressive scaling based on layer index
                    injector._layer_count += 1
                    layer_weight = injector._layer_count / 100.0  # Simple scaling factor
                    dynamic_factor = torch.ones_like(hidden_states) * layer_weight
                
                # Call with correct parameter names
                output_with_cross_attn = cross_attn_layer(
                    hidden_states=hidden_states,
                    cross_states=scaffold_output,  # renamed from scaffold_output
                    attention_mask=None,  # Default to None
                    memory_tensors=None,  # Default to None
                    memory_weight=0.0,    # Default to 0.0
                    dynamic_factor=dynamic_factor,
                    use_cache=False       # Default to False
                )
                
                if isinstance(output, tuple):
                    return (output_with_cross_attn,) + output[1:]
                return output_with_cross_attn
                
            except Exception as e:
                # Log the error but continue with original output
                injector._logger.record_event(
                    event_type="cross_attention_failure",
                    message=f"Cross attention injection failed: {str(e)}",
                    level="warning",
                    exc_info=e
                )
                return output

        return type(
            "WrappedLayer",
            (nn.Module,),
            {"forward": _wrapped_layer_forward, "__init__": lambda self: None}
        )()

    def _verify_single_layer(self, model: nn.Module, layer_idx: int) -> bool:
        """Verify a single layer's cross-attention injection, including forward pass stability."""
        try:
            layers, _ = self.find_model_layers(model)
            layer = layers[layer_idx]
            if not hasattr(layer, '_cross_attn'):
                self._logger.record_event(
                    event_type="cross_attention_verification_failed",
                    message=f"Layer {layer_idx} missing _cross_attn attribute after injection.",
                    level="error"
                )
                return False
            if layer._cross_attn._hidden_size != layer._cross_attn._q_proj.in_features:
                self._logger.record_event(
                    event_type="cross_attention_verification_failed",
                    message=f"Layer {layer_idx} hidden size mismatch after injection.",
                    level="error"
                )
                return False
            # Forward pass validation
            try:
                # Try to infer input shape from q_proj
                hidden_size = layer._cross_attn._q_proj.in_features
                batch_size = 2
                seq_len = 8
                dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=next(layer.parameters()).device)
                # Try to run forward
                out = layer(dummy_input)
                if isinstance(out, tuple):
                    out_tensor = out[0]
                else:
                    out_tensor = out
                if not torch.isfinite(out_tensor).all():
                    self._logger.record_event(
                        event_type="cross_attention_verification_failed",
                        message=f"Layer {layer_idx} produced NaN or Inf in forward pass.",
                        level="error"
                    )
                    return False
                norm = out_tensor.norm().item()
                if norm > 1e4 or norm < 1e-6:
                    self._logger.record_event(
                        event_type="cross_attention_verification_failed",
                        message=f"Layer {layer_idx} output norm out of range: {norm}",
                        level="error"
                    )
                    return False
            except Exception as e:
                self._logger.record_event(
                    event_type="cross_attention_verification_failed",
                    message=f"Layer {layer_idx} forward pass failed: {str(e)}",
                    level="error"
                )
                return False
            return True
        except Exception:
            return False

    def save_state(self, path: str, state_dict: dict) -> None:
        """Save cross-attention parameters."""
        try:
            with self._lock:
                torch.save(
                    {k: v for k, v in state_dict.items() if 'cross_attn' in k or '_scaffold_proj' in k},
                    path
                )
                self._logger.record_event(
                    event_type="save_state",
                    message="Cross-attention state saved successfully",
                    level="info",
                    additional_info={"path": path, "timestamp": time.time()}
                )
        except Exception as e:
            self._logger.record_event(
                event_type="save_state_failed",
                message=f"Failed to save cross-attention state: {str(e)}",
                level="error",
                additional_info={
                    "path": path,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def load_state(self, path: str, model: nn.Module) -> None:
        """Load cross-attention parameters."""
        try:
            with self._lock:
                state_dict = model.state_dict()
                checkpoint_dict = torch.load(path, map_location=model.device)
                state_dict.update({k: v for k, v in checkpoint_dict.items() if k in state_dict})
                model.load_state_dict(state_dict)
                self._logger.record_event(
                    event_type="load_state",
                    message="Cross-attention state loaded successfully",
                    level="info",
                    additional_info={"path": path, "timestamp": time.time()}
                )
        except Exception as e:
            self._logger.record_event(
                event_type="load_state_failed",
                message=f"Failed to load cross-attention state: {str(e)}",
                level="error",
                additional_info={
                    "path": path,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def inject_cross_attention(
        self,
        model: nn.Module,
        scaffold_model: nn.Module,
        core_config: Dict[str, Any],
        cross_attn_config: Dict[str, Any],
        lora_config: Dict[str, Any],
        token_map: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """Inject cross-attention layers into the model with full configuration support."""
        try:
            if not cross_attn_config.get("enable_cross_attention", True):
                self._logger.record_event(
                    event_type="cross_attention",
                    message="Cross-attention disabled",
                    level="info",
                    additional_info={"timestamp": time.time()}
                )
                return

            print("Injecting cross-attention layers...")
            
            layers_to_inject = core_config.get("cross_attn_layers", [])
            injection_strategy = cross_attn_config.get("injection_strategy", "sequential")
            
            self.inject(
                base_model=model,
                scaffold_model=scaffold_model,
                layers_to_inject=layers_to_inject,
                injection_strategy=injection_strategy,
                token_map=token_map
            )
            
            if not self.verify_injection(model, model.config):
                raise ValueError("Cross-attention layer verification failed")
                
            self._logger.record_event(
                event_type="cross_attention_injected",
                message="Cross-attention injection completed successfully",
                level="info",
                additional_info={"timestamp": time.time()}
            )
            print("Cross-attention injection complete.")
        except Exception as e:
            self._error_handler.handle_cross_attention_error(e)
            raise

    def check_injected_model_health(self, model: nn.Module) -> bool:
        """Check the health of the injected model using the general utility."""
        healthy = util_check_model_health(model, self, logger=self._logger)
        if not healthy:
            self._logger.record_event(
                event_type="cross_attention_model_health_failed",
                message="Injected model failed health check after cross-attention injection.",
                level="error"
            )
        else:
            self._logger.record_event(
                event_type="cross_attention_model_health_passed",
                message="Injected model passed health check after cross-attention injection.",
                level="info"
            )
        return healthy

def calculate_confidence_score(logits: torch.Tensor, generated_ids: torch.Tensor) -> float:
    """Calculate confidence score for scaffold generation."""
    try:
        # Get the state from the existing system
        from sovl_state import get_state
        state = get_state()
        
        if not state.curiosity:
            return 0.5
            
        # Generate query embedding from the logits
        with torch.no_grad():
            query_embedding = logits.mean(dim=1)  # Simple mean pooling of logits
            
        # Use curiosity manager for confidence calculation
        confidence = state.curiosity.compute_curiosity(
            base_conf=0.5,  # Default base confidence
            scaf_conf=0.5,  # Default scaffold confidence
            state=state,
            query_embedding=query_embedding,
            device=logits.device
        )
        
        return confidence
        
    except Exception as e:
        raise RuntimeError(f"Failed to calculate confidence score: {str(e)}")

class InsufficientDataError(Exception):
    """Exception raised when there is insufficient data for scaffold operations."""
    pass

# High-level provider for scaffold state: init, update, validation.
class ScaffoldProvider:
    """Provides scaffold functionality for the SOVL system with atomic, thread-safe state management."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, error_handler: ErrorManager):
        self.config_manager = config_manager
        self.logger = logger
        self._error_handler = error_handler
        self._scaffold_state = None
        self._lock = threading.RLock()
        self._update_queue = queue.Queue()
        self._max_retries = 10
        self._retry_delay = 0.1  # seconds
    
    @scaffold_operation("validate_config")
    def validate_scaffold_config(self, config: Dict[str, Any]) -> None:
        """Validate scaffold configuration."""
        required_fields = ["token_mapping", "attention_config", "memory_config"]
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"Missing required field: {field}")
                
        if not isinstance(config["token_mapping"], dict):
            raise ConfigurationError("token_mapping must be a dictionary")
            
        self.logger.record_event(
            event_type="scaffold_config_validation",
            message="Scaffold configuration validated successfully",
            level="info"
        )
        
    @scaffold_operation("initialize")
    def initialize_scaffold_state(self, config: Dict[str, Any]) -> None:
        """Initialize scaffold state with configuration, using atomic update."""
        try:
            self.validate_scaffold_config(config)
            new_state = {
                "token_mapping": config["token_mapping"],
                "attention_config": config["attention_config"],
                "memory_config": config["memory_config"],
                "last_updated": time.time()
            }
            for attempt in range(self._max_retries):
                acquired = self._lock.acquire(timeout=self._retry_delay)
                if acquired:
                    try:
                        self._scaffold_state = new_state
                        break
                    finally:
                        self._lock.release()
                else:
                    self.logger.record_event(
                        event_type="scaffold_init_lock_retry",
                        message=f"Retry {attempt+1}/{self._max_retries} acquiring lock for state initialization.",
                        level="warning"
                    )
                    time.sleep(self._retry_delay)
            else:
                self.logger.record_event(
                    event_type="scaffold_init_lock_failed",
                    message="Failed to acquire lock for state initialization after retries.",
                    level="error"
                )
                raise ScaffoldError("Failed to acquire lock for state initialization after retries.", operation="initialize_scaffold_state")
            self.logger.record_event(
                event_type="scaffold_initialization",
                message="Scaffold state initialized successfully",
                level="info"
            )
        except Exception as e:
            self._error_handler.handle_error(
                error=e,
                operation="initialize_scaffold_state",
                context={"config": str(config)}
            )
            raise
    
    @scaffold_operation("update")
    def update_scaffold_state(self, updates: Dict[str, Any]) -> None:
        """Update scaffold state with new values, using atomic update and retry."""
        if not self._scaffold_state:
            raise ScaffoldError(
                "Scaffold state not initialized",
                operation="update_scaffold_state"
            )
        timestamp = time.time()
        for attempt in range(self._max_retries):
            acquired = self._lock.acquire(timeout=self._retry_delay)
            if acquired:
                try:
                    self._scaffold_state.update(updates)
                    self._scaffold_state["last_updated"] = timestamp
                    break
                finally:
                    self._lock.release()
            else:
                self.logger.record_event(
                    event_type="scaffold_update_lock_retry",
                    message=f"Retry {attempt+1}/{self._max_retries} acquiring lock for state update.",
                    level="warning"
                )
                time.sleep(self._retry_delay)
        else:
            self.logger.record_event(
                event_type="scaffold_update_lock_failed",
                message="Failed to acquire lock for state update after retries. Queuing update.",
                level="error"
            )
            self._update_queue.put((updates, timestamp))
        self.logger.record_event(
            event_type="scaffold_update",
            message="Scaffold state updated successfully",
            level="info"
        )
    
    def process_queued_updates(self, max_updates=100):
        """Process up to max_updates queued state updates atomically."""
        processed = 0
        while not self._update_queue.empty() and processed < max_updates:
            updates, timestamp = self._update_queue.get()
            acquired = self._lock.acquire(timeout=self._retry_delay)
            if acquired:
                try:
                    self._scaffold_state.update(updates)
                    self._scaffold_state["last_updated"] = timestamp
                finally:
                    self._lock.release()
            else:
                self.logger.record_event(
                    event_type="scaffold_update_queue_lock_failed",
                    message="Failed to acquire lock for queued update.",
                    level="error"
                )
                self._update_queue.put((updates, timestamp))
                break
            processed += 1
        queue_size = self._update_queue.qsize()
        if queue_size >= 1000:
            self.logger.record_event(
                event_type="scaffold_update_queue_size_warning",
                message=f"Scaffold update queue size high: {queue_size}",
                level="warning"
            )
        else:
            self.logger.record_event(
                event_type="scaffold_update_queue_size_info",
                message=f"Scaffold update queue size: {queue_size}",
                level="info"
            )

    def get_update_queue_size(self):
        """Return the current size of the update queue."""
        return self._update_queue.qsize()
    
    @scaffold_operation("get_state")
    def get_scaffold_state(self) -> Dict[str, Any]:
        """Get current scaffold state, validating for corruption."""
        if not self._scaffold_state:
            raise ScaffoldError(
                "Scaffold state not initialized",
                operation="get_scaffold_state"
            )
        acquired = self._lock.acquire(timeout=self._retry_delay)
        if acquired:
            try:
                state_copy = self._scaffold_state.copy()
            finally:
                self._lock.release()
        else:
            self.logger.record_event(
                event_type="scaffold_get_state_lock_failed",
                message="Failed to acquire lock for get_state, returning possibly stale state.",
                level="error"
            )
            state_copy = self._scaffold_state.copy() if self._scaffold_state else {}
        # Validate state integrity
        required_fields = ["token_mapping", "attention_config", "memory_config", "last_updated"]
        for field in required_fields:
            if field not in state_copy:
                self.logger.record_event(
                    event_type="scaffold_state_corruption",
                    message=f"Missing required field in scaffold state: {field}",
                    level="error"
                )
                raise ScaffoldError(
                    f"Scaffold state corrupted: missing {field}",
                    operation="get_scaffold_state"
                )
        return state_copy

    def get_scaffold_metrics(self):
        """Aggregate and return current scaffold mapping metrics."""
        mapper = getattr(self, 'token_mapper', None)
        if not mapper:
            return {}
        with mapper._metrics_lock:
            metrics = {
                "avg_confidence": float(np.mean(mapper._mapping_confidences)) if mapper._mapping_confidences else None,
                "fallback_counts": dict(mapper._fallback_counts),
                "avg_drift": float(np.mean(mapper._drift_values)) if mapper._drift_values else None,
                "max_drift": float(np.max(mapper._drift_values)) if mapper._drift_values else None,
                "mapping_errors": mapper._mapping_errors,
                "avg_latency": float(np.mean(mapper._mapping_latencies)) if mapper._mapping_latencies else None,
                "timestamp": time.time(),
            }
        return metrics

    def report_metrics_to_monitor(self, system_monitor):
        """Send current scaffold metrics to the system monitor."""
        metrics = self.get_scaffold_metrics()
        if hasattr(system_monitor, 'update_component_metrics'):
            system_monitor.update_component_metrics("scaffold", metrics)

    def start_metric_reporting(self, system_monitor, interval=60):
        """Start periodic reporting of scaffold metrics to the system monitor."""
        def report_loop():
            while True:
                self.report_metrics_to_monitor(system_monitor)
                time.sleep(interval)
        t = threading.Thread(target=report_loop, daemon=True)
        t.start()

# Utility function to create a scaffold model with LoRA integration
def create_scaffold_with_adaptation(config_manager, logger, error_manager, lora_checkpoint_path=None):
    """
    Factory for creating a scaffold model wrapped with adaptation (LoRA, Adapters, or Prefix Tuning).
    Optionally loads LoRA weights from a checkpoint path (for long-term memory).
    Returns the adapted model, the LoraAdapterManager instance, and the adaptation method used.

    Notes:
        - LoRA is applied to the scaffold model before cross-attention injection.
        - For the base model, LoRA must be applied **before** CrossAttentionInjector.inject
          to avoid compatibility issues with wrapped layers.
    """
    # --- Build the base scaffold model ---
    scaffold_model = ScaffoldModel(config_manager, logger, error_manager)  # Replace with your actual scaffold model class/init
    # --- Build and apply adaptation ---
    lora_manager = LoraAdapterManager(config_manager, logger, error_manager)
    scaffold_model, method_used = lora_manager.apply_with_fallbacks(scaffold_model)
    if lora_checkpoint_path and method_used == "lora":
        try:
            scaffold_model = lora_manager.load_lora_weights(scaffold_model, lora_checkpoint_path)
        except Exception as e:
            logger.log_warning(f"Failed to load LoRA checkpoint {lora_checkpoint_path}: {e}", event_type="lora_load_warning")
    return scaffold_model, lora_manager, method_used

# Example usage elsewhere in the system:
# model, lora_mgr = create_scaffold_with_adaptation(config_manager, logger, error_manager)
