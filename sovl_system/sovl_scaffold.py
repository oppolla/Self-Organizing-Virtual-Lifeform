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
import functools
from sovl_engram import LoraAdapterManager
import copy
import threading
from sovl_utils import check_model_health as util_check_model_health, calculate_token_map_confidence
import queue
import numpy as np
import os
import pickle
import hashlib
import json

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
    
    def __init__(
        self,
        base_tokenizer: Any,
        scaffold_tokenizer: Any,
        logger: Any,
        config: Optional[Dict[str, Any]] = None,
        base_model: Any = None,
        scaffold_model: Any = None,
        mapping_strategy: str = None,
        ram_manager: Optional[RAMManager] = None,
        gpu_manager: Optional[GPUMemoryManager] = None,
        provider: Optional[Any] = None,
    ):
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer
        self.logger = logger
        self.base_model = base_model  # Optional: for embedding similarity
        self.scaffold_model = scaffold_model  # Optional: for embedding similarity
        self.mapping_strategy = mapping_strategy  # User-selectable, but defaults to None for now
        self.max_tokens_per_mapping = config.get('max_tokens_per_mapping', 3) if config else 3
        self.mapping_similarity_threshold = config.get('mapping_similarity_threshold', 0.7) if config else 0.7
        self.conflict_resolution_strategy = config.get('conflict_resolution_strategy', 'keep_highest_conf') if config else 'keep_highest_conf'
        self.token_map = defaultdict(lambda: {'ids': [scaffold_tokenizer.unk_token_id], 'weight': 1.0})
        self.embedding_available = self._check_embedding_availability()
        self.config = config or {}
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
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
        # Memory usage logging
        if self.ram_manager:
            self.logger.record_event(
                event_type="ram_memory_health",
                message="RAM health at init",
                level="info",
                additional_info=self.ram_manager.check_memory_health() if hasattr(self.ram_manager, 'check_memory_health') else {}
            )
        if self.gpu_manager:
            self.logger.record_event(
                event_type="gpu_memory_health",
                message="GPU health at init",
                level="info",
                additional_info=self.gpu_manager.get_gpu_usage() if hasattr(self.gpu_manager, 'get_gpu_usage') else {}
            )
        
        # Fallback strategy pipeline
        fallback_order = (config.get('token_mapping_fallback_order') if config else None) or [
            'levenshtein', 'subword', 'char', 'split', 'merge', 'nearest', 'unk'
        ]
        self._fallback_strategies = []
        for strat in fallback_order:
            if strat == 'levenshtein':
                self._fallback_strategies.append(LevenshteinStrategy())
            elif strat == 'subword':
                self._fallback_strategies.append(SubwordStrategy())
            elif strat == 'char':
                self._fallback_strategies.append(CharSimilarityStrategy())
            elif strat == 'split':
                self._fallback_strategies.append(SplitStrategy())
            elif strat == 'merge':
                self._fallback_strategies.append(MergeStrategy())
            elif strat == 'nearest':
                self._fallback_strategies.append(NearestStrategy())
            elif strat == 'unk':
                self._fallback_strategies.append(UnkStrategy())
        
        self.provider = provider
        self.min_token_map_confidence = self.config.get('min_token_map_confidence', 0.5)
        self.max_low_conf_ratio = self.config.get('max_low_conf_ratio', 0.2)
        self.max_fallback_ratio = self.config.get('max_fallback_ratio', 0.3)
        
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

    def _char_similarity(self, token1, token2):
        set1 = set(token1)
        set2 = set(token2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _normalize_token(self, token: str) -> str:
        """Normalize token based on configured level."""
        if self.normalization_level == 'none':
            return token
        elif self.normalization_level == 'basic':
            return token.replace("Ġ", "").replace("##", "")
        else:  # aggressive
            return token.replace("Ġ", "").replace("##", "").lower()
            
    def _validate_fallback_mapping(self, token, scaffold_ids, strategy_name, context):
        if scaffold_ids == [self.scaffold_tokenizer.unk_token_id]:
            self.logger.warning(f"Unk token mapping for '{token}' using {strategy_name}")
            return False
        if self.embedding_available and self.base_model is not None and self.scaffold_model is not None:
            base_emb = self._get_token_embedding(self.base_model, context['base_id'])
            scaf_emb = self._get_token_embedding(self.scaffold_model, scaffold_ids[0])
            if base_emb is not None and scaf_emb is not None:
                import torch.nn.functional as F
                similarity = F.cosine_similarity(base_emb, scaf_emb, dim=0).item()
                if similarity < self.mapping_similarity_threshold:
                    self.logger.warning(f"Low embedding similarity ({similarity:.3f}) for fallback mapping '{token}' using {strategy_name}")
                    return False
        return True

    def _score_mapping(self, token, scaffold_ids, context):
        # Use embedding similarity if available, else fallback to char similarity
        if self.embedding_available and self.base_model is not None and self.scaffold_model is not None:
            base_emb = self._get_token_embedding(self.base_model, context['base_id'])
            scaf_emb = self._get_token_embedding(self.scaffold_model, scaffold_ids[0])
            if base_emb is not None and scaf_emb is not None:
                import torch.nn.functional as F
                return F.cosine_similarity(base_emb, scaf_emb, dim=0).item()
        # Fallback: character similarity
        scaf_token = self.scaffold_tokenizer.decode(scaffold_ids) if hasattr(self.scaffold_tokenizer, 'decode') else str(scaffold_ids)
        return self._char_similarity(token, scaf_token)

    def _handle_fallback(self, token: str, base_id: int) -> List[int]:
        """Handle token mapping fallback using the configured strategy pipeline with validation and scoring."""
        context = {
            'self': self,
            'base_tokenizer': self.base_tokenizer,
            'scaffold_tokenizer': self.scaffold_tokenizer,
            'max_tokens_per_mapping': self.max_tokens_per_mapping,
            'base_id': base_id
        }
        candidates = []
        for strat in self._fallback_strategies:
            result = strat.try_map(token, context)
            if result and self._validate_fallback_mapping(token, result, strat.__class__.__name__, context):
                score = self._score_mapping(token, result, context)
                candidates.append({'ids': result, 'score': score, 'strategy': strat.__class__.__name__})
                self.logger.record_event(
                    event_type="token_fallback_candidate",
                    message=f"Candidate fallback mapping for '{token}' using {strat.__class__.__name__} (score={score:.3f})",
                    level="info",
                    additional_info={"token": token, "strategy": strat.__class__.__name__, "score": score, "ids": result}
                )
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            self.logger.record_event(
                event_type="token_fallback_selected",
                message=f"Selected fallback mapping for '{token}' using {best['strategy']} (score={best['score']:.3f})",
                level="info",
                additional_info={"token": token, "strategy": best['strategy'], "score": best['score'], "ids": best['ids']}
            )
            return best['ids']
        self.logger.record_event(
            event_type="token_fallback_unk",
            message=f"All fallback strategies failed for '{token}', returning <unk>",
            level="warning",
            additional_info={"token": token}
        )
        return [self.scaffold_tokenizer.unk_token_id]
        
    def _get_token_map_cache_key(self):
        import hashlib, json
        base_vocab = self.base_tokenizer.get_vocab()
        scaffold_vocab = self.scaffold_tokenizer.get_vocab()
        config_str = json.dumps(self.config, sort_keys=True)
        key_str = json.dumps(base_vocab, sort_keys=True) + json.dumps(scaffold_vocab, sort_keys=True) + config_str
        return hashlib.md5(key_str.encode()).hexdigest()

    def _try_load_token_map_cache(self):
        cache_key = self._get_token_map_cache_key()
        cache_path = f"token_map_cache_{cache_key}.pkl"
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.token_map = pickle.load(f)
                self.logger.record_event(
                    event_type="token_map_cache_loaded",
                    message=f"Loaded token_map from cache: {cache_path}",
                    level="info"
                )
                return True
            except Exception as e:
                self.logger.record_event(
                    event_type="token_map_cache_load_failed",
                    message=f"Failed to load token_map cache: {e}",
                    level="warning"
                )
        return False

    def _save_token_map_cache(self):
        cache_key = self._get_token_map_cache_key()
        cache_path = f"token_map_cache_{cache_key}.pkl"
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(dict(self.token_map), f)
            self.logger.record_event(
                event_type="token_map_cache_saved",
                message=f"Saved token_map to cache: {cache_path}",
                level="info"
            )
        except Exception as e:
            self.logger.record_event(
                event_type="token_map_cache_save_failed",
                message=f"Failed to save token_map cache: {e}",
                level="warning"
            )

    def _build_token_map(self):
        if self._try_load_token_map_cache():
            return
        special_tokens = {
            self.base_tokenizer.pad_token_id: self.scaffold_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id: self.scaffold_tokenizer.eos_token_id,
            self.base_tokenizer.unk_token_id: self.scaffold_tokenizer.unk_token_id,
        }
        base_vocab = self.base_tokenizer.get_vocab()
        total = len(base_vocab)
        for i, (base_token, base_id) in enumerate(base_vocab.items()):
            if base_id in special_tokens and special_tokens[base_id] is not None:
                self.token_map[base_id] = {'ids': [special_tokens[base_id]], 'weight': 1.0, 'confidence': 1.0, 'strategy': 'special'}
                continue
            normalized = self._normalize_token(base_token)
            if normalized in self.scaffold_tokenizer.get_vocab():
                scaf_id = self.scaffold_tokenizer.get_vocab()[normalized]
                self.token_map[base_id] = {'ids': [scaf_id], 'weight': 1.0, 'confidence': 1.0, 'strategy': 'exact'}
                continue
            # Use fallback strategy pipeline for all other cases
            best_conf = 0.0
            best_ids = [self.scaffold_tokenizer.unk_token_id]
            best_strategy = 'unk'
            for strat in self._fallback_strategies:
                result = strat.try_map(normalized, {
                    'self': self,
                    'base_tokenizer': self.base_tokenizer,
                    'scaffold_tokenizer': self.scaffold_tokenizer,
                    'max_tokens_per_mapping': self.max_tokens_per_mapping,
                    'base_id': base_id,
                })
                if result and isinstance(result, tuple):
                    ids, conf, strategy = result
                elif result:
                    ids, conf, strategy = result, 0.5, strat.__class__.__name__
                else:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_ids = ids
                    best_strategy = strategy
            self.token_map[base_id] = {'ids': best_ids, 'weight': best_conf, 'confidence': best_conf, 'strategy': best_strategy}
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                self.logger.record_event(
                    event_type="token_map_progress",
                    message=f"Token mapping progress: {i+1}/{total}",
                    level="info"
                )
        self._save_token_map_cache()

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
        try:
            start = time.time()
            base_tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
            scaffold_ids = []
            weights = []
            low_conf_count = 0
            unk_count = 0
            fallback_count = 0
            failed_tokens = []
            for idx, base_id in enumerate(base_tokens):
                mapping = self.token_map[base_id]
                scaffold_ids.extend(mapping['ids'])
                weights.extend([mapping['confidence']] * len(mapping['ids']))
                with self._metrics_lock:
                    self._mapping_confidences.append(mapping['confidence'])
                    if mapping.get('strategy') and mapping['strategy'] not in ('special', 'exact'):
                        self._fallback_counts[mapping['strategy']] += 1
                        fallback_count += 1
                if mapping['confidence'] < self.min_token_map_confidence:
                    low_conf_count += 1
                    failed_tokens.append({
                        'base_token': self.base_tokenizer.decode([base_id]) if hasattr(self.base_tokenizer, 'decode') else str(base_id),
                        'scaffold_token': self.scaffold_tokenizer.decode(mapping['ids']) if hasattr(self.scaffold_tokenizer, 'decode') else str(mapping['ids']),
                        'confidence': mapping['confidence'],
                        'strategy': mapping.get('strategy', 'unknown')
                    })
                if mapping['ids'] == [self.scaffold_tokenizer.unk_token_id]:
                    unk_count += 1
            total = len(base_tokens) if base_tokens else 1
            low_conf_ratio = low_conf_count / total
            unk_ratio = unk_count / total
            fallback_ratio = fallback_count / total
            self.logger.record_event(
                event_type="token_mapping_quality",
                message=(
                    f"Token mapping: {low_conf_count}/{total} ({low_conf_ratio:.2%}) below confidence {self.min_token_map_confidence}, "
                    f"{unk_count}/{total} ({unk_ratio:.2%}) <unk> tokens, "
                    f"{fallback_count}/{total} ({fallback_ratio:.2%}) fallback tokens"
                ),
                level="warning" if low_conf_ratio > self.max_low_conf_ratio or unk_ratio > self.max_low_conf_ratio or fallback_ratio > self.max_fallback_ratio else "info",
                additional_info={
                    "low_conf_count": low_conf_count,
                    "total_tokens": total,
                    "low_conf_ratio": low_conf_ratio,
                    "unk_count": unk_count,
                    "unk_ratio": unk_ratio,
                    "fallback_count": fallback_count,
                    "fallback_ratio": fallback_ratio,
                    "min_conf": self.min_token_map_confidence,
                    "failed_tokens": failed_tokens[:10],  # Only log first 10 for brevity
                    "timestamp": time.time()
                }
            )
            if fallback_ratio > self.max_fallback_ratio:
                self.logger.record_event(
                    event_type="token_mapping_fallback_warning",
                    message=f"High fallback ratio: {fallback_count}/{total} ({fallback_ratio:.2%}). Tokenizers may be too dissimilar.",
                    level="warning",
                    additional_info={"fallback_count": fallback_count, "total": total, "ratio": fallback_ratio}
                )
            if low_conf_ratio > self.max_low_conf_ratio or unk_ratio > self.max_low_conf_ratio:
                with self._metrics_lock:
                    self._mapping_errors += 1
                raise ScaffoldError(
                    (
                        f"Too many low-confidence or <unk> token mappings: "
                        f"{low_conf_count}/{total} ({low_conf_ratio:.2%}) below confidence {self.min_token_map_confidence}, "
                        f"{unk_count}/{total} ({unk_ratio:.2%}) <unk> tokens"
                    ),
                    operation="tokenize_and_map",
                    context={
                        "failed_tokens": failed_tokens,
                        "low_conf_ratio": low_conf_ratio,
                        "unk_ratio": unk_ratio,
                        "fallback_ratio": fallback_ratio
                    }
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
            # Synchronize with provider
            if self.provider is not None:
                self.provider.update_token_map(self.get_token_map(), source="mapper")
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
        device: Union[str, torch.device] = 'cpu',
        ram_manager: Optional[RAMManager] = None,
        gpu_manager: Optional[GPUMemoryManager] = None
    ):
        super().__init__()
        self._config = config
        self._logger = logger
        self._device = torch.device(device) if isinstance(device, str) else device
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self._initialize_parameters(hidden_size, num_heads)
        self._initialize_layers()
        self.to(self._device)
        # Memory usage logging
        if self.ram_manager:
            self._logger.record_event(
                event_type="ram_memory_health",
                message="RAM health at CrossAttentionLayer init",
                level="info",
                additional_info=self.ram_manager.check_memory_health() if hasattr(self.ram_manager, 'check_memory_health') else {}
            )
        if self.gpu_manager:
            self._logger.record_event(
                event_type="gpu_memory_health",
                message="GPU health at CrossAttentionLayer init",
                level="info",
                additional_info=self.gpu_manager.get_gpu_usage() if hasattr(self.gpu_manager, 'get_gpu_usage') else {}
            )
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
        self._attention_chunk_size = self._config.get('attention_chunk_size', 128)
        self._gpu_memory_threshold = self._config.get('gpu_memory_threshold', 0.85)
        
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
        batch_size: int,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute attention scores with dynamic weighting and memory-aware chunking."""
        chunk_size = chunk_size or self._attention_chunk_size
        q = q.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self._num_heads, self._head_dim).transpose(1, 2)
        attn_output = torch.zeros_like(q)
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end]
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / (self._scale * self._dynamic_scale)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask[:, :, i:end] == 0, float('-inf'))
            if self._sparse_mask is not None:
                scores = scores.masked_fill(self._sparse_mask[:, :, i:end] == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output[:, :, i:end] = torch.matmul(attn_weights, v)
            # Memory check and adaptive chunking
            if self.gpu_manager:
                usage = self.gpu_manager.get_gpu_usage().get('usage_percentage', 0.0)
                if usage > self._gpu_memory_threshold:
                    self._logger.record_event(
                        event_type="cross_attention_chunking",
                        message=f"GPU memory usage {usage:.2f} exceeded threshold {self._gpu_memory_threshold}, reducing chunk size.",
                        level="warning",
                        additional_info={"usage": usage, "threshold": self._gpu_memory_threshold, "old_chunk_size": chunk_size}
                    )
                    chunk_size = max(chunk_size // 2, 32)
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
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: Optional[RAMManager] = None,
        gpu_manager: Optional[GPUMemoryManager] = None,
        lora_manager: Optional[Any] = None,
        provider: Optional[Any] = None,
    ):
        self._config_manager = config_manager
        self._logger = logger
        self._lock = Lock()
        self._scaffold_proj: Optional[nn.Module] = None
        self._scaffold_unk_id = config_manager.get("controls_config.scaffold_unk_id", 0)
        self._error_handler = ErrorManager(config_manager, logger)
        self._target_layer_names: List[str] = []  # Store names of layers being injected
        self._layer_count = 0  # Initialize layer counter for progressive strategy
        self.ram_manager = ram_manager or RAMManager(config_manager, logger)
        self.gpu_manager = gpu_manager or GPUMemoryManager(config_manager, logger)
        self.lora_manager = lora_manager
        self.allow_lora_post_injection = config_manager.get('controls_config.allow_lora_post_injection', False)
        self.provider = provider
        self.current_map_version = -1
        self._validate_config()
        # Injection strategy pipeline
        strategy_order = config_manager.get('controls_config.injection_strategy_order', ['sequential', 'parallel', 'replace'])
        self._injection_strategies = []
        for strat in strategy_order:
            if strat == 'sequential':
                self._injection_strategies.append(SequentialInjectionStrategy())
            elif strat == 'parallel':
                self._injection_strategies.append(ParallelInjectionStrategy())
            elif strat == 'replace':
                self._injection_strategies.append(ReplaceInjectionStrategy())
        # Memory usage logging
        if self.ram_manager:
            self._logger.record_event(
                event_type="ram_memory_health",
                message="RAM health at CrossAttentionInjector init",
                level="info",
                additional_info=self.ram_manager.check_memory_health() if hasattr(self.ram_manager, 'check_memory_health') else {}
            )
        if self.gpu_manager:
            self._logger.record_event(
                event_type="gpu_memory_health",
                message="GPU health at CrossAttentionInjector init",
                level="info",
                additional_info=self.gpu_manager.get_gpu_usage() if hasattr(self.gpu_manager, 'get_gpu_usage') else {}
            )

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
            # Synchronize token map with provider
            if self.provider is not None:
                latest_map, version = self.provider.get_token_map()
                if version > self.current_map_version:
                    token_map = latest_map
                    self.current_map_version = version
                    self._logger.record_event(
                        event_type="cross_attention_token_map_sync",
                        message=f"Updated token map to version {version}",
                        level="info"
                    )
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
        """Inject cross-attention into the model with robust error recovery and fallback. Refactored to inject directly into base_model."""
        try:
            # Robust LoRA conflict detection
            if not self.allow_lora_post_injection:
                if not self._check_lora_compatibility(base_model):
                    self._logger.record_event(
                        event_type="lora_conflict",
                        message="LoRA detected in base model but configuration does not allow post-injection LoRA or model is incompatible.",
                        level="error"
                    )
                    raise ScaffoldError(
                        "LoRA detected in base model but configuration does not allow post-injection LoRA or model is incompatible.",
                        operation="cross_attention_injection"
                    )
            else:
                self._logger.record_event(
                    event_type="lora_post_injection_allowed",
                    message="LoRA post-injection is allowed by configuration.",
                    level="info"
                )
            with self._lock:
                layers, layer_names = self.find_model_layers(base_model)
                layer_indices = self.get_cross_attention_layers(base_model, layers_to_inject)
                self._target_layer_names = [layer_names[i] for i in layer_indices]

                self._logger.record_event(
                    event_type="cross_attention_injection_start",
                    message="Cross-attention injection started (direct in-place)",
                    level="info",
                    additional_info={
                        "layer_names": self._target_layer_names,
                        "timestamp": time.time()
                    }
                )

                successful_layers = []
                skipped_layers = []

                for layer_idx in layer_indices:
                    injected = False
                    for strategy in self._injection_strategies:
                        try:
                            strategy.inject(
                                model=base_model,
                                scaffold_model=scaffold_model,
                                layer_idx=layer_idx,
                                token_map=token_map,
                                injector=self
                            )
                            injected = True
                            break
                        except Exception as e:
                            self._logger.record_event(
                                event_type="cross_attention_injection_strategy_failed",
                                message=f"Layer {layer_idx} injection failed with strategy '{strategy.__class__.__name__}': {str(e)}",
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

                if not self.verify_injection(base_model, base_model.config):
                    self._logger.record_event(
                        event_type="cross_attention_injection_rollback",
                        message="Verification failed after injection, rolling back to original model.",
                        level="error"
                    )
                    return base_model  # Return original model

                self._logger.record_event(
                    event_type="cross_attention_injection_complete",
                    message="Cross-attention injection completed in-place.",
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
        """Inject cross-attention into a single layer with retry logic. Refactored to use per-layer projection/LayerNorm."""
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

            proj = None
            proj_norm = None
            if base_hidden_size is not None and scaffold_hidden_size is not None:
                if base_hidden_size != scaffold_hidden_size:
                    # Create per-layer projection and LayerNorm
                    proj = nn.Linear(scaffold_hidden_size, base_hidden_size).to(model.device)
                    nn.init.xavier_uniform_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    proj_norm = nn.LayerNorm(base_hidden_size).to(model.device)
                    self._logger.record_event(
                        event_type="scaffold_projection_created",
                        message=f"Created per-layer projection from {scaffold_hidden_size} to {base_hidden_size} with Xavier init and LayerNorm",
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
                strategy=injection_strategy,
                proj=proj,
                proj_norm=proj_norm
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

    def _create_wrapped_layer(
        self,
        original_layer: nn.Module,
        cross_attn_layer: CrossAttentionLayer,
        scaffold_model: nn.Module,
        token_map: Optional[Dict],
        strategy: str,
        proj=None,
        proj_norm=None
    ) -> nn.Module:
        """Wrap the layer with cross-attention injection. Accepts per-layer projection and norm."""
        original_forward = original_layer.forward
        scaffold_model = scaffold_model
        injector = self
        token_mapper = token_map

        class WrappedLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.cross_attn_layer = cross_attn_layer
                self.proj = proj
                self.proj_norm = proj_norm
            def forward(self, *args, **kwargs):
                output = original_forward(*args, **kwargs)
                hidden_states = output[0] if isinstance(output, tuple) else output
                try:
                    scaffold_output = injector._get_scaffold_output(
                        scaffold_model=scaffold_model,
                        token_map=token_mapper,
                        base_hidden_states=hidden_states
                    )
                    # Apply per-layer projection if needed
                    if self.proj is not None:
                        scaffold_output = self.proj(scaffold_output)
                        if self.proj_norm is not None:
                            scaffold_output = self.proj_norm(scaffold_output)
                    # Create dynamic factor based on strategy if needed
                    dynamic_factor = None
                    if strategy == "progressive":
                        injector._layer_count += 1
                        layer_weight = injector._layer_count / 100.0
                        dynamic_factor = torch.ones_like(hidden_states) * layer_weight
                    output_with_cross_attn = self.cross_attn_layer(
                        hidden_states=hidden_states,
                        cross_states=scaffold_output,
                        attention_mask=None,
                        memory_tensors=None,
                        memory_weight=0.0,
                        dynamic_factor=dynamic_factor,
                        use_cache=False
                    )
                    if isinstance(output, tuple):
                        return (output_with_cross_attn,) + output[1:]
                    return output_with_cross_attn
                except Exception as e:
                    injector._logger.record_event(
                        event_type="cross_attention_failure",
                        message=f"Cross attention injection failed: {str(e)}",
                        level="warning",
                        exc_info=e
                    )
                    return output
        return WrappedLayer()

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

    def _check_lora_compatibility(self, base_model):
        if self.lora_manager is None:
            self._logger.record_event(
                event_type="lora_manager_missing",
                message="No LoraAdapterManager provided; skipping LoRA compatibility check.",
                level="warning"
            )
            return True  # Assume compatible if no manager
        try:
            compatible = self.lora_manager.is_model_compatible(base_model)
            if not compatible:
                self._logger.record_event(
                    event_type="lora_incompatibility_detected",
                    message="LoRA detected but model is not compatible with LoRA configuration.",
                    level="error"
                )
            return compatible
        except Exception as e:
            self._logger.record_event(
                event_type="lora_compatibility_check_failed",
                message=f"Failed to check LoRA compatibility: {str(e)}",
                level="error"
            )
            return False

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
    
    def __init__(
            self, config_manager: ConfigManager, 
            logger: Logger, 
            error_handler: ErrorManager, 
            ram_manager: Optional[RAMManager] = None, 
            gpu_manager: Optional[GPUMemoryManager] = None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self._error_handler = error_handler
        self._scaffold_state = None
        self._lock = threading.RLock()
        self._update_queue = queue.Queue()
        self._max_retries = 10
        self._retry_delay = 0.1  # seconds
        self.ram_manager = ram_manager or RAMManager(config_manager, logger)
        self.gpu_manager = gpu_manager or GPUMemoryManager(config_manager, logger)
        self._max_update_queue_size = 1000
        self.token_map_version = 0
        self.token_map = {}
    
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
                    self._validate_updates(updates)
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
            if self._update_queue.qsize() >= self._max_update_queue_size:
                self.logger.record_event(
                    event_type="scaffold_update_queue_full",
                    message="Update queue full, dropping oldest update.",
                    level="error"
                )
                self._update_queue.get()
            self._update_queue.put((updates, timestamp))
            self.logger.record_event(
                event_type="scaffold_update_queued",
                message="Failed to acquire lock, queued update.",
                level="warning",
                additional_info={"queue_size": self._update_queue.qsize()}
            )
        self.logger.record_event(
            event_type="scaffold_update",
            message="Scaffold state updated successfully",
            level="info"
        )
    
    def process_queued_updates(self, max_updates=100):
        """Process up to max_updates queued state updates atomically."""
        processed = 0
        updates_list = []
        while not self._update_queue.empty() and processed < max_updates:
            updates, timestamp = self._update_queue.get()
            updates_list.append((updates, timestamp))
            processed += 1
        # Sort by timestamp to ensure correct order
        updates_list.sort(key=lambda x: x[1])
        for updates, timestamp in updates_list:
            acquired = self._lock.acquire(timeout=self._retry_delay)
            if acquired:
                try:
                    self._validate_updates(updates)
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
        queue_size = self._update_queue.qsize()
        if queue_size >= self._max_update_queue_size:
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

class TokenMappingStrategy:
    def try_map(self, token, context):
        raise NotImplementedError

class SplitStrategy(TokenMappingStrategy):
    def try_map(self, token, context):
        parts = [token[i:i+2] for i in range(0, len(token), 2)]
        scaffold_ids = []
        for part in parts:
            ids = context['scaffold_tokenizer'].encode(part, add_special_tokens=False)
            scaffold_ids.extend(ids)
        if scaffold_ids:
            return scaffold_ids[:context['max_tokens_per_mapping']]
        return None

class MergeStrategy(TokenMappingStrategy):
    @staticmethod
    def find_similar_tokens(token, context):
        # Example: Use character overlap as a simple similarity metric
        scaffold_tokenizer = context['scaffold_tokenizer']
        max_tokens = context.get('max_tokens_per_mapping', 3)
        results = []
        for scaf_token, scaf_id in scaffold_tokenizer.get_vocab().items():
            # Simple similarity: number of shared characters
            set1 = set(token)
            set2 = set(scaf_token)
            score = len(set1 & set2) / max(len(set1 | set2), 1)
            if score > 0.0:
                results.append({'ids': [scaf_id], 'score': score})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_tokens] if results else None

    def try_map(self, token, context):
        similar_tokens = MergeStrategy.find_similar_tokens(token, context)
        if similar_tokens:
            return similar_tokens[0]['ids']
        return None

class NearestStrategy(TokenMappingStrategy):
    @staticmethod
    def find_nearest_token(token, context):
        # Example: Use minimum edit distance (Levenshtein) as a proxy for 'nearest'
        scaffold_tokenizer = context['scaffold_tokenizer']
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
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
        min_dist = float('inf')
        best_id = None
        for scaf_token, scaf_id in scaffold_tokenizer.get_vocab().items():
            dist = levenshtein(token, scaf_token)
            if dist < min_dist:
                min_dist = dist
                best_id = scaf_id
        return best_id

    def try_map(self, token, context):
        nearest = NearestStrategy.find_nearest_token(token, context)
        if nearest is not None:
            return [nearest]
        return None

class UnkStrategy(TokenMappingStrategy):
    def try_map(self, token, context):
        return [context['scaffold_tokenizer'].unk_token_id]

class LevenshteinStrategy(TokenMappingStrategy):
    _bk_tree = None
    _bk_tree_vocab = None
    def try_map(self, token, context):
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
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
        scaffold_tokenizer = context['scaffold_tokenizer']
        vocab_keys = list(scaffold_tokenizer.get_vocab().keys())
        if (LevenshteinStrategy._bk_tree is None or LevenshteinStrategy._bk_tree_vocab != vocab_keys):
            LevenshteinStrategy._bk_tree = BKTree(levenshtein_distance, vocab_keys)
            LevenshteinStrategy._bk_tree_vocab = vocab_keys
        matches = LevenshteinStrategy._bk_tree.find(token, max_dist=2)
        if matches:
            best = min(matches, key=lambda x: (x[0]/max(len(token), len(x[1]), 1), x[0]))
            norm_dist = best[0] / max(len(token), len(best[1]), 1)
            if norm_dist <= 0.3:
                scaf_id = scaffold_tokenizer.get_vocab()[best[1]]
                confidence = 1.0 - norm_dist
                return ([scaf_id], confidence, 'levenshtein')
        return None

class SubwordStrategy(TokenMappingStrategy):
    def try_map(self, token, context):
        scaffold_tokenizer = context['scaffold_tokenizer']
        for scaf_token, scaf_id in scaffold_tokenizer.get_vocab().items():
            if token.startswith(scaf_token) or token.endswith(scaf_token):
                # Confidence: ratio of subword length to token length
                conf = len(scaf_token) / max(len(token), 1)
                return ([scaf_id], conf, 'subword')
        return None

class CharSimilarityStrategy(TokenMappingStrategy):
    @staticmethod
    def char_similarity(token1, token2):
        set1 = set(token1)
        set2 = set(token2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def try_map(self, token, context):
        best_score = 0.0
        best_id = None
        for scaf_token, scaf_id in context['scaffold_tokenizer'].get_vocab().items():
            score = CharSimilarityStrategy.char_similarity(token, scaf_token)
            if score > best_score:
                best_score = score
                best_id = scaf_id
        if best_score > 0.0:
            return ([best_id], best_score, 'char')
        return None

class InjectionStrategy:
    def inject(self, model, scaffold_model, layer_idx, token_map, injector):
        raise NotImplementedError

class SequentialInjectionStrategy(InjectionStrategy):
    def inject(self, model, scaffold_model, layer_idx, token_map, injector):
        # Use the existing _inject_single_layer logic for 'sequential'
        return injector._inject_single_layer(
            model=model,
            scaffold_model=scaffold_model,
            layer_idx=layer_idx,
            injection_strategy='sequential',
            token_map=token_map
        )

class ParallelInjectionStrategy(InjectionStrategy):
    def inject(self, model, scaffold_model, layer_idx, token_map, injector):
        # Placeholder: use the same logic as sequential for now, or implement parallel logic if available
        return injector._inject_single_layer(
            model=model,
            scaffold_model=scaffold_model,
            layer_idx=layer_idx,
            injection_strategy='parallel',
            token_map=token_map
        )

class ReplaceInjectionStrategy(InjectionStrategy):
    def inject(self, model, scaffold_model, layer_idx, token_map, injector):
        # Placeholder: use the same logic as sequential for now, or implement replace logic if available
        return injector._inject_single_layer(
            model=model,
            scaffold_model=scaffold_model,
            layer_idx=layer_idx,
            injection_strategy='replace',
            token_map=token_map
        )

    def _validate_updates(self, updates: Dict[str, Any]):
        # Use the same checks as validate_scaffold_config, but for partial updates
        # For example, check that any provided fields are valid and of correct type
        allowed_fields = {"token_mapping": dict, "attention_config": dict, "memory_config": dict}
        for key, expected_type in allowed_fields.items():
            if key in updates and not isinstance(updates[key], expected_type):
                raise ConfigurationError(f"Update field '{key}' must be of type {expected_type.__name__}")
        # Optionally, add more granular validation here
        return True

# Minimal BK-tree implementation for Levenshtein
class BKTree:
    def __init__(self, dist_fn, words):
        self.dist_fn = dist_fn
        self.tree = None
        for word in words:
            self.add(word)
    class Node:
        def __init__(self, word):
            self.word = word
            self.children = {}  # distance -> child node
    def add(self, word):
        if self.tree is None:
            self.tree = self.Node(word)
            return
        node = self.tree
        while True:
            d = self.dist_fn(word, node.word)
            if d in node.children:
                node = node.children[d]
            else:
                node.children[d] = self.Node(word)
                break
    def find(self, word, max_dist):
        results = []
        def rec(node):
            d = self.dist_fn(word, node.word)
            if d <= max_dist:
                results.append((d, node.word))
            for k in range(d - max_dist, d + max_dist + 1):
                child = node.children.get(k)
                if child:
                    rec(child)
        if self.tree:
            rec(self.tree)
        return results
