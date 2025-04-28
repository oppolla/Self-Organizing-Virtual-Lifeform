import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
from collections import defaultdict
import time
import traceback
from threading import Lock
import math
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_error import ErrorManager, ScaffoldError
from sovl_confidence import ConfidenceCalculator
from sovl_io import ConfigurationError
from sovl_curiosity import CuriosityManager
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager
import contextlib
import functools
from sovl_engram import LoraAdapterManager
import difflib
import copy
# Centralized handler for scaffold errors and recovery.
class ScaffoldErrorManager:
    """Centralized error handling for scaffold operations."""
    
    def __init__(self, logger: Logger, error_handler: Optional[ErrorManager] = None):
        self.logger = logger
        self.error_handler = error_handler
        self._lock = Lock()
        
    def handle_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error"
    ) -> None:
        """Handle scaffold errors with consistent logging and recovery."""
        with self._lock:
            error_context = {
                "operation": operation,
                "timestamp": time.time(),
                "severity": severity,
                **(context or {})
            }
            
            # Log error through both systems if available
            if self.error_handler:
                self.error_handler.handle_scaffold_error(error, error_context)
            
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
        
        # Initialize mapping strategy parameters
        self.max_tokens_per_mapping = config.get('max_tokens_per_mapping', 3) if config else 3
        self.mapping_similarity_threshold = config.get('mapping_similarity_threshold', 0.7) if config else 0.7
        self.allow_bidirectional_mapping = config.get('allow_bidirectional_mapping', False) if config else False
        self.fallback_strategy = config.get('fallback_strategy', 'split') if config else 'split'
        self.normalization_level = config.get('normalization_level', 'basic') if config else 'basic'
        
        # Initialize quality control parameters
        self.min_semantic_similarity = config.get('min_semantic_similarity', 0.5) if config else 0.5
        self.max_meaning_drift = config.get('max_meaning_drift', 0.3) if config else 0.3
        self.enable_periodic_validation = config.get('enable_periodic_validation', True) if config else True
        self.conflict_resolution_strategy = config.get('conflict_resolution_strategy', 'keep_highest_conf') if config else 'keep_highest_conf'
        
        self.token_map = defaultdict(lambda: {'ids': [scaffold_tokenizer.unk_token_id], 'weight': 1.0})
        # Check if embedding-based similarity is available
        self.embedding_available = self._check_embedding_availability()
        if self.embedding_available:
            self.logger.info("Using embedding-based similarity for token mapping.")
        else:
            self.logger.warning("Falling back to character-based similarity for token mapping.")
        self._initialize_token_maps()
        
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
        """Calculate semantic drift between tokens."""
        # This is a placeholder for a more sophisticated semantic drift calculation
        # In practice, this could use embeddings or other semantic similarity measures
        return 1.0 - self._calculate_similarity(token1, token2)
        
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
        """Tokenize prompt and map to scaffold token space."""
        try:
            base_tokens = self.base_tokenizer.encode(prompt, add_special_tokens=False)
            scaffold_ids = []
            weights = []
            
            for base_id in base_tokens:
                mapping = self.token_map[base_id]
                scaffold_ids.extend(mapping['ids'])
                weights.extend([mapping['weight']] * len(mapping['ids']))
                
            return scaffold_ids, weights
            
        except Exception as e:
            self.logger.record_event(
                event_type="token_mapping_error",
                message=f"Failed to map tokens: {str(e)}",
                level="error",
                additional_info={"timestamp": time.time()}
            )
            raise
            
    def update_token_map_memory(self, prompt: str, confidence: float):
        """Update token map based on prompt confidence."""
        try:
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
        """Create a wrapped layer based on injection strategy."""
        class WrappedLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffold, token_map, parent, strategy):
                super().__init__()
                self._base_layer = base_layer if strategy != 'replace' else None
                self._cross_attn = cross_attn
                self._scaffold = scaffold
                self._token_map = token_map or defaultdict(lambda: [parent._scaffold_unk_id])
                self._strategy = strategy
                self._parent = parent
                self._combine = (
                    nn.Linear(cross_attn._hidden_size * 2, cross_attn._hidden_size)
                    if strategy == 'parallel' else None
                )

            def forward(self, hidden_states, *args, scaffold_context=None, **kwargs):
                try:
                    if self._strategy == 'replace':
                        if scaffold_context is None:
                            return hidden_states
                        context = scaffold_context.to(hidden_states.device)
                        if self._parent._scaffold_proj is not None:
                            context = self._parent._scaffold_proj(context)
                        output = self._cross_attn(hidden_states, context, **kwargs)
                        return (output,) if isinstance(hidden_states, tuple) else output

                    base_output = self._base_layer(hidden_states, *args, **kwargs)
                    base_output = base_output[0] if isinstance(base_output, tuple) else base_output
                    
                    if scaffold_context is None:
                        return base_output
                    
                    context = scaffold_context.to(hidden_states.device)
                    if self._parent._scaffold_proj is not None:
                        context = self._parent._scaffold_proj(context)
                    cross_output = self._cross_attn(hidden_states, context, **kwargs)
                    
                    if self._strategy == 'parallel':
                        combined = torch.cat([base_output, cross_output], dim=-1)
                        output = self._combine(combined)
                    else:
                        output = cross_output
                        
                    return (output,) + base_output[1:] if isinstance(base_output, tuple) else output
                except Exception as e:
                    self._parent._logger.record_event(
                        event_type="wrapped_layer_forward_failed",
                        message=f"WrappedLayer forward failed: {str(e)}",
                        level="error",
                        additional_info={
                            "hidden_states_shape": list(hidden_states.shape),
                            "timestamp": time.time(),
                            "stack_trace": traceback.format_exc()
                        }
                    )
                    raise

        return WrappedLayer(original_layer, cross_attn_layer, scaffold_model, token_map, self, strategy)

    def _verify_single_layer(self, model: nn.Module, layer_idx: int) -> bool:
        """Verify a single layer's cross-attention injection."""
        try:
            layers, _ = self.find_model_layers(model)
            layer = layers[layer_idx]
            if not hasattr(layer, '_cross_attn'):
                return False
            if layer._cross_attn._hidden_size != layer._cross_attn._q_proj.in_features:
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
    """Provides scaffold functionality for the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, error_handler: ErrorManager):
        self.config_manager = config_manager
        self.logger = logger
        self._error_handler = error_handler
        self._scaffold_state = None
        self._lock = Lock()
        
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
        """Initialize scaffold state with configuration."""
        try:
            self.validate_scaffold_config(config)
            
            with self._lock:
                self._scaffold_state = {
                    "token_mapping": config["token_mapping"],
                    "attention_config": config["attention_config"],
                    "memory_config": config["memory_config"],
                    "last_updated": time.time()
                }
                
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
        """Update scaffold state with new values."""
        if not self._scaffold_state:
            raise ScaffoldError(
                "Scaffold state not initialized",
                operation="update_scaffold_state"
            )
            
        with self._lock:
            self._scaffold_state.update(updates)
            self._scaffold_state["last_updated"] = time.time()
            
        self.logger.record_event(
            event_type="scaffold_update",
            message="Scaffold state updated successfully",
            level="info"
        )
        
    @scaffold_operation("get_state")
    def get_scaffold_state(self) -> Dict[str, Any]:
        """Get current scaffold state."""
        if not self._scaffold_state:
            raise ScaffoldError(
                "Scaffold state not initialized",
                operation="get_scaffold_state"
            )
        return self._scaffold_state.copy()

# Utility function to create a scaffold model with LoRA integration
def create_scaffold_with_adaptation(config_manager, logger, error_manager, lora_checkpoint_path=None):
    """
    Factory for creating a scaffold model wrapped with adaptation (LoRA, Adapters, or Prefix Tuning).
    Optionally loads LoRA weights from a checkpoint path (for long-term memory).
    Returns the adapted model, the LoraAdapterManager instance, and the adaptation method used.
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
