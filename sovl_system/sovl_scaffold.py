import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Union, Dict, Any
from collections import defaultdict
import time
import traceback
from threading import Lock
import contextlib
import math
import threading
from sovl_logger import Logger
from sovl_config import ConfigManager, ConfigSchema, _SchemaValidator
from sovl_utils import NumericalGuard, safe_divide, validate_layer_indices
from sovl_error import ErrorHandler
from sovl_confidence import ConfidenceCalculator
from sovl_io import ConfigurationError

class ScaffoldTokenMapper:
    """Handles token mapping between base and scaffold tokenizers."""
    
    def __init__(self, base_tokenizer: Any, scaffold_tokenizer: Any, logger: Logger):
        """
        Initialize the ScaffoldTokenMapper with base and scaffold tokenizers.
        
        Args:
            base_tokenizer: The base model's tokenizer
            scaffold_tokenizer: The scaffold model's tokenizer
            logger: Logger instance for structured logging
        """
        self._base_tokenizer = base_tokenizer
        self._scaffold_tokenizer = scaffold_tokenizer
        self._logger = logger
        self._token_map: Dict[int, Dict[str, Union[List[int], float]]] = {}
        self._special_token_map: Dict[int, int] = {}
        self._token_map_lock = threading.Lock()
        self._special_token_map_lock = threading.Lock()
        
        self._initialize_token_maps()
        self._validate_token_maps()
        
        self._logger.record_event(
            event_type="scaffold_token_mapper_initialized",
            message="Scaffold token mapper initialized successfully",
            level="info",
            additional_info={
                "base_vocab_size": len(self._base_tokenizer),
                "scaffold_vocab_size": len(self._scaffold_tokenizer),
                "token_map_size": len(self._token_map),
                "special_token_map_size": len(self._special_token_map)
            }
        )
        
    def _initialize_token_maps(self) -> None:
        """Initialize both regular and special token mappings."""
        self._build_token_map()
        self._initialize_special_token_map()
        
    def _build_token_map(self) -> None:
        """Build the main token mapping between base and scaffold tokenizers."""
        with self._token_map_lock:
            try:
                for base_token, base_id in self._base_tokenizer.get_vocab().items():
                    normalized = self._normalize_token(base_token)
                    scaffold_ids = self._scaffold_tokenizer.encode(
                        normalized, add_special_tokens=False, max_length=3, truncation=True
                    ) or [self._scaffold_tokenizer.unk_token_id]
                    self._token_map[base_id] = {"ids": scaffold_ids, "weight": 1.0}
                    
                self._logger.record_event(
                    event_type="token_map_built",
                    message="Token map built successfully",
                    level="info",
                    additional_info={"map_size": len(self._token_map), "timestamp": time.time()}
                )
            except Exception as e:
                self._log_error("token_map_error", f"Failed to build token map: {str(e)}")
                raise
            
    def _initialize_special_token_map(self) -> None:
        """Initialize mapping for special tokens."""
        with self._special_token_map_lock:
            try:
                self._special_token_map = {
                    self._base_tokenizer.pad_token_id: self._scaffold_tokenizer.pad_token_id,
                    self._base_tokenizer.eos_token_id: (
                        self._scaffold_tokenizer.eos_token_id or self._scaffold_tokenizer.sep_token_id
                    ),
                    self._base_tokenizer.unk_token_id: self._scaffold_tokenizer.unk_token_id,
                }
                
                self._logger.record_event(
                    event_type="special_token_map_initialized",
                    message="Special token map initialized successfully",
                    level="info",
                    additional_info={
                        "map_size": len(self._special_token_map),
                        "mappings": {
                            "pad_token": self._special_token_map.get(self._base_tokenizer.pad_token_id),
                            "eos_token": self._special_token_map.get(self._base_tokenizer.eos_token_id),
                            "unk_token": self._special_token_map.get(self._base_tokenizer.unk_token_id)
                        }
                    }
                )
            except Exception as e:
                self._log_error("token_map_error", f"Failed to initialize special token map: {str(e)}")
                raise
            
    def _validate_token_maps(self) -> None:
        """Validate that token maps are properly initialized."""
        try:
            if not self._token_map:
                raise ValueError("Token map is empty")
            if not self._special_token_map:
                raise ValueError("Special token map is empty")
                
            base_vocab_size = len(self._base_tokenizer)
            scaffold_vocab_size = len(self._scaffold_tokenizer)
            
            self._validate_special_tokens(base_vocab_size, scaffold_vocab_size)
            self._validate_regular_tokens(base_vocab_size, scaffold_vocab_size)
            
            self._logger.record_event(
                event_type="token_maps_validated",
                message="Token maps validated successfully",
                level="info",
                additional_info={
                    "base_vocab_size": base_vocab_size,
                    "scaffold_vocab_size": scaffold_vocab_size,
                    "token_map_size": len(self._token_map),
                    "special_token_map_size": len(self._special_token_map),
                    "timestamp": time.time()
                }
            )
        except Exception as e:
            self._log_error("token_map_validation_failed", f"Token map validation failed: {str(e)}")
            raise
        
    def _validate_special_tokens(self, base_vocab_size: int, scaffold_vocab_size: int) -> None:
        """Validate special token mappings."""
        required_special_tokens = ['pad_token_id', 'eos_token_id', 'unk_token_id']
        for token in required_special_tokens:
            base_id = getattr(self._base_tokenizer, token, None)
            if base_id is None:
                raise ValueError(f"Base tokenizer missing {token}")
            if base_id not in self._special_token_map:
                raise ValueError(f"Special token {token} not mapped")
                
        for base_id, scaffold_id in self._special_token_map.items():
            if not (0 <= base_id < base_vocab_size):
                raise ValueError(f"Invalid base token ID in special token map: {base_id}")
            if not (0 <= scaffold_id < scaffold_vocab_size):
                raise ValueError(f"Invalid scaffold token ID in special token map: {scaffold_id}")
            
    def _validate_regular_tokens(self, base_vocab_size: int, scaffold_vocab_size: int) -> None:
        """Validate regular token mappings."""
        for base_id, mapping in self._token_map.items():
            if not (0 <= base_id < base_vocab_size):
                raise ValueError(f"Invalid base token ID in token map: {base_id}")
            for scaffold_id in mapping["ids"]:
                if not (0 <= scaffold_id < scaffold_vocab_size):
                    raise ValueError(f"Invalid scaffold token ID in token map: {scaffold_id}")
                    
    def _normalize_token(self, token: str) -> str:
        """Normalize token for mapping."""
        return token.replace("Ġ", "").replace("##", "")
        
    def _log_error(self, event_type: str, message: str) -> None:
        """Log an error with consistent formatting."""
        self._logger.record_event(
            event_type=event_type,
            message=message,
            level="error",
            additional_info={
                "error": message,
                "stack_trace": traceback.format_exc(),
                "timestamp": time.time()
            }
        )
        
    def get_token_map(self) -> Dict[int, Dict[str, Union[List[int], float]]]:
        """Get the complete token mapping."""
        return self._token_map
        
    def get_special_token_map(self) -> Dict[int, int]:
        """Get the special token mapping."""
        return self._special_token_map
        
    def tokenize_and_map(self, prompts: Union[str, List[str]], max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Tokenize prompts and map to scaffold token space."""
        try:
            prompts = [prompts] if isinstance(prompts, str) else prompts
            max_length = max_length or 128
            
            base_inputs = self._base_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            scaffold_inputs = self.map_sequence(base_inputs['input_ids'])
            
            return {
                'input_ids': scaffold_inputs,
                'attention_mask': base_inputs['attention_mask']
            }
        except Exception as e:
            self._log_error("tokenization_error", f"Tokenization and mapping failed: {str(e)}")
            raise ValueError(f"Tokenization and mapping failed: {str(e)}")
            
    def map_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Map a sequence of tokens from base to scaffold tokenizer."""
        try:
            device = sequence.device
            sequence_list = sequence.tolist()
            mapped_tokens = []
            
            for token_id in sequence_list:
                if token_id in self._special_token_map:
                    mapped_tokens.append(self._special_token_map[token_id])
                else:
                    with self._token_map_lock:
                        mapped_tokens.append(
                            self._token_map[token_id]["ids"][0] if token_id in self._token_map
                            else self._scaffold_tokenizer.unk_token_id
                        )
            
            result = torch.tensor(mapped_tokens, device=device)
            
            del sequence_list, mapped_tokens
            torch.cuda.empty_cache()
            
            return result
        except Exception as e:
            self._log_error("sequence_mapping_error", f"Error mapping sequence: {str(e)}")
            raise

    def update_token_map_memory(self, prompt: str, confidence: float) -> None:
        """Update token map memory based on prompt confidence."""
        try:
            base_inputs = self._base_tokenizer(
                prompt, return_tensors='pt'
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            base_ids = base_inputs['input_ids'][0].tolist()
            
            with self._token_map_lock:
                for base_id in base_ids:
                    if base_id in self._token_map:
                        current_weight = self._token_map[base_id]['weight']
                        self._token_map[base_id]['weight'] = min(1.0, current_weight + (confidence * 0.1))
                
            self._logger.record_event(
                event_type="token_map_updated",
                message="Token map updated successfully",
                level="info",
                additional_info={"confidence": confidence}
            )
        except Exception as e:
            self._log_error("token_map_memory_update_failed", f"Token map memory update failed: {str(e)}")
            raise

def build_scaffold_token_mapping(base_tokenizer: Any, scaffold_tokenizer: Any, logger: Logger) -> ScaffoldTokenMapper:
    """
    Create a ScaffoldTokenMapper instance.
    
    Args:
        base_tokenizer: The tokenizer for the base model
        scaffold_tokenizer: The tokenizer for the scaffold model
        logger: Logger instance for structured logging
    
    Returns:
        ScaffoldTokenMapper: Initialized token mapper instance
    """
    return ScaffoldTokenMapper(base_tokenizer, scaffold_tokenizer, logger)

class SparseMaskFactory:
    """Handles creation of sparse attention masks."""
    
    @staticmethod
    def create(
        seq_len: int,
        sparse_pattern: str,
        window_size: int,
        device: str = 'cpu',
        logger: Optional[Logger] = None
    ) -> torch.Tensor:
        """Create a sparse attention mask based on the specified pattern."""
        try:
            with NumericalGuard():
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
                    event_type="sparse_mask_creation_failed",
                    message=f"Sparse mask creation failed: {str(e)}",
                    level="error",
                    additional_info={
                        "sparse_pattern": sparse_pattern,
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc()
                    }
                )
            raise

class AttentionMaskPreparer:
    """Prepares attention masks for multi-head attention."""
    
    @staticmethod
    def prepare(
        attention_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
        logger: Optional[Logger] = None
    ) -> torch.Tensor:
        """Prepare attention mask in additive format."""
        try:
            with NumericalGuard():
                if attention_mask is None:
                    return None
                if attention_mask.dim() < 2 or attention_mask.dim() > 4:
                    raise ValueError(f"Invalid attention mask dimensions: {attention_mask.shape}")
                
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.float().masked_fill(~attention_mask, float('-inf'))
                elif attention_mask.dtype != torch.float:
                    attention_mask = attention_mask.float()
                
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                
                if attention_mask.shape != (batch_size, num_heads, seq_len, seq_len):
                    attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                
                return attention_mask.to(device)
        except Exception as e:
            if logger:
                logger.record_event(
                    event_type="attention_mask_preparation_failed",
                    message=f"Attention mask preparation failed: {str(e)}",
                    level="error",
                    additional_info={
                        "mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc()
                    }
                )
            raise

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

class CrossAttentionInjector:
    """Injector for adding cross-attention layers to a transformer model."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize the CrossAttentionInjector with configuration manager and logger.
        
        Args:
            config_manager: Configuration manager instance
            logger: Logger instance for structured logging
        """
        self._config_manager = config_manager
        self._logger = logger
        self._lock = Lock()
        self._scaffold_proj: Optional[nn.Module] = None
        self._scaffold_unk_id = config_manager.get("controls_config.scaffold_unk_id", 0)
        self._error_handler = ErrorHandler(config_manager, logger)
        
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
        """Inject cross-attention into the model."""
        try:
            with self._lock:
                layer_indices = self.get_cross_attention_layers(base_model, layers_to_inject)
                original_state = {name: param.clone() for name, param in base_model.named_parameters()}
                
                self._logger.record_event(
                    event_type="cross_attention_injection_start",
                    message="Cross-attention injection started",
                    level="info",
                    additional_info={"layers": layer_indices, "timestamp": time.time()}
                )
                
                for layer_idx in layer_indices:
                    self._inject_single_layer(
                        model=base_model,
                        scaffold_model=scaffold_model,
                        layer_idx=layer_idx,
                        injection_strategy=injection_strategy,
                        token_map=token_map
                    )
                
                if not self.verify_injection(base_model, layer_indices, base_model.config):
                    for name, param in base_model.named_parameters():
                        if name in original_state:
                            param.data.copy_(original_state[name])
                    raise RuntimeError("Cross-attention injection verification failed")
                
                self._logger.record_event(
                    event_type="cross_attention_injection_complete",
                    message="Cross-attention injection completed successfully",
                    level="info",
                    additional_info={"status": "success", "timestamp": time.time()}
                )
                
                return base_model
        except Exception as e:
            self._logger.record_event(
                event_type="cross_attention_injection_failed",
                message=f"Cross-attention injection failed: {str(e)}",
                level="error",
                additional_info={"timestamp": time.time(), "stack_trace": traceback.format_exc()}
            )
            raise

    def _inject_single_layer(
        self,
        model: nn.Module,
        scaffold_model: nn.Module,
        layer_idx: int,
        injection_strategy: str,
        token_map: Optional[Dict]
    ) -> None:
        """Inject cross-attention into a single layer."""
        try:
            layers, _ = self.find_model_layers(model)
            if layer_idx >= len(layers):
                raise ValueError(f"Layer index {layer_idx} out of bounds")
                
            original_layer = layers[layer_idx]
            cross_attn_layer = CrossAttentionLayer(
                config=self._config_manager.get_section("core_config"),
                logger=self._logger,
                device=model.device
            )
            
            layers[layer_idx] = self._create_wrapped_layer(
                original_layer=original_layer,
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
                message=f"Failed to inject cross-attention layer {layer_idx}: {str(e)}",
                level="error",
                additional_info={
                    "layer_idx": layer_idx,
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

    def verify_injection(self, model: nn.Module, expected_layers: List[int], base_config: Any) -> bool:
        """Verify that cross-attention layers were properly injected."""
        try:
            expected_layers = set(expected_layers)
            found_layers = set()

            for name, module in model.named_modules():
                if "cross_attn" in name.lower():
                    try:
                        parts = name.split('.')
                        if len(parts) >= 3 and parts[0] == 'transformer' and parts[1] == 'h':
                            layer_idx = int(parts[2])
                            found_layers.add(layer_idx)
                    except (ValueError, IndexError):
                        continue
                    
            self._logger.record_event(
                event_type="cross_attention_verification",
                message="Cross-attention verification completed",
                level="info",
                additional_info={
                    "expected_layers": list(expected_layers),
                    "found_layers": list(found_layers),
                    "timestamp": time.time()
                }
            )

            if not expected_layers.issubset(found_layers):
                missing_layers = expected_layers - found_layers
                self._logger.record_event(
                    event_type="missing_cross_attention_layers",
                    message=f"Missing cross-attention layers: {missing_layers}",
                    level="warning",
                    additional_info={
                        "missing_layers": list(missing_layers),
                        "timestamp": time.time()
                    }
                )
                return False

            for layer_idx in expected_layers:
                try:
                    layer = model.transformer.h[layer_idx]
                    if not hasattr(layer, '_cross_attn'):
                        self._logger.record_event(
                            event_type="layer_missing_cross_attention_attribute",
                            message=f"Layer {layer_idx} missing cross_attention attribute",
                            level="warning",
                            additional_info={"layer_idx": layer_idx, "timestamp": time.time()}
                        )
                        return False

                    if layer._cross_attn._hidden_size != base_config.hidden_size:
                        self._logger.record_event(
                            event_type="layer_dimension_mismatch",
                            message=f"Layer {layer_idx} dimension mismatch",
                            level="warning",
                            additional_info={"layer_idx": layer_idx, "timestamp": time.time()}
                        )
                        return False

                    if layer._cross_attn._num_heads != base_config.num_attention_heads:
                        self._logger.record_event(
                            event_type="layer_attention_heads_mismatch",
                            message=f"Layer {layer_idx} attention heads mismatch",
                            level="warning",
                            additional_info={"layer_idx": layer_idx, "timestamp": time.time()}
                        )
                        return False
                except Exception as e:
                    self._error_handler.handle_cross_attention_error(e, layer_idx)
                    return False

            return True
        except Exception as e:
            self._error_handler.handle_cross_attention_error(e)
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
            
            if not self.verify_injection(model, layers_to_inject, model.config):
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
        confidence_calculator = ConfidenceCalculator.get_instance()
        return confidence_calculator.calculate_confidence_score(
            logits=logits,
            generated_ids=generated_ids,
            state=None,
            error_manager=None,
            context=None,
            curiosity_manager=None
        )
    except Exception as e:
        raise RuntimeError(f"Failed to calculate confidence score: {str(e)}")

class InsufficientDataError(Exception):
    """Exception raised when there is insufficient data for scaffold operations."""
    pass

class ScaffoldProvider:
    """Provides scaffold-related functionality and state management."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize the ScaffoldProvider with configuration manager and logger.
        
        Args:
            config_manager: Configuration manager instance
            logger: Logger instance for structured logging
        """
        self._config_manager = config_manager
        self._logger = logger
        self._lock = Lock()
        self._error_handler = ErrorHandler(config_manager, logger)
        self._scaffold_state: Dict[str, Any] = {
            "is_initialized": False,
            "last_update": None,
            "scaffold_model": None,
            "token_map": None
        }
        self._scaffold_config = None
        self._scaffold_state_tensor = None
        self._hidden_size = None
        self._num_heads = None
        self._num_layers = None
        self._device = None
        
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate scaffold provider configuration."""
        try:
            required_keys = [
                "scaffold_config.model_path",
                "scaffold_config.model_type",
                "scaffold_config.tokenizer_path",
                "scaffold_config.quantization_mode"
            ]
            
            for key in required_keys:
                if not self._config_manager.has_key(key):
                    raise ConfigurationError(f"Missing required config key: {key}")
            
            model_type = self._config_manager.get("scaffold_config.model_type")
            if model_type not in ["gpt2", "gptj", "gpt_neox"]:
                raise ConfigurationError(f"Unsupported model type: {model_type}")
            
            quantization_mode = self._config_manager.get("scaffold_config.quantization_mode")
            if quantization_mode not in ["none", "int8", "int4"]:
                raise ConfigurationError(f"Unsupported quantization mode: {quantization_mode}")
            
            self._logger.record_event(
                event_type="scaffold_config_validated",
                message="Scaffold configuration validated successfully",
                level="info",
                additional_info={"timestamp": time.time()}
            )
        except Exception as e:
            self._logger.record_event(
                event_type="scaffold_config_error",
                message=f"Failed to validate scaffold config: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def initialize_scaffold(self) -> None:
        """Initialize the scaffold model and tokenizer."""
        try:
            with self._lock:
                if self._scaffold_state["is_initialized"]:
                    return
                
                model_path = self._config_manager.get("scaffold_config.model_path")
                model_type = self._config_manager.get("scaffold_config.model_type")
                tokenizer_path = self._config_manager.get("scaffold_config.tokenizer_path")
                quantization_mode = self._config_manager.get("scaffold_config.quantization_mode")
                
                self._scaffold_state["scaffold_model"] = self._load_model(
                    model_path=model_path,
                    model_type=model_type,
                    quantization_mode=quantization_mode
                )
                
                self._scaffold_state["token_map"] = self._build_token_map(
                    base_tokenizer=self._scaffold_state["scaffold_model"].tokenizer,
                    scaffold_tokenizer=self._load_tokenizer(tokenizer_path)
                )
                
                self._scaffold_state["is_initialized"] = True
                self._scaffold_state["last_update"] = time.time()
                
                self._logger.record_event(
                    event_type="scaffold_initialized",
                    message="Scaffold model and tokenizer initialized successfully",
                    level="info",
                    additional_info={
                        "model_type": model_type,
                        "quantization_mode": quantization_mode,
                        "timestamp": time.time()
                    }
                )
        except Exception as e:
            self._logger.record_event(
                event_type="scaffold_initialization_error",
                message=f"Failed to initialize scaffold: {str(e)}",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise

    def _load_model(self, model_path: str, model_type: str, quantization_mode: str) -> nn.Module:
        """Load scaffold model (placeholder implementation)."""
        # Note: Actual implementation would load the model based on type and quantization
        raise NotImplementedError("Model loading not implemented")

    def _load_tokenizer(self, tokenizer_path: str) -> Any:
        """Load scaffold tokenizer (placeholder implementation)."""
        # Note: Actual implementation would load the tokenizer
        raise NotImplementedError("Tokenizer loading not implemented")

    def _build_token_map(self, base_tokenizer: Any, scaffold_tokenizer: Any) -> Dict:
        """Build token mapping between base and scaffold tokenizers."""
        try:
            token_mapper = ScaffoldTokenMapper(base_tokenizer, scaffold_tokenizer, self._logger)
            return token_mapper.get_token_map()
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {"operation": "token_map_building"})
            raise

    def get_scaffold_context(
        self, 
        scaffold_inputs: Dict[str, torch.Tensor],
        scaffold_model: nn.Module
    ) -> torch.Tensor:
        """Get scaffold context from inputs."""
        try:
            with self.scaffold_context(
                self.get_scaffold_hidden_states(scaffold_inputs, scaffold_model)
            ):
                return self._scaffold_state_tensor
        except Exception as e:
            self._logger.record_event(
                event_type="scaffold_error",
                message="Failed to get scaffold context",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise
    
    def validate_scaffold_config(self) -> bool:
        """Validate scaffold configuration."""
        try:
            self._validate_config()
            return True
        except Exception as e:
            self._logger.record_event(
                event_type="scaffold_error",
                message="Failed to validate scaffold config",
                level="error",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise
    
    def initialize_scaffold_state(self, model_name: str, device: str) -> bool:
        """Initialize scaffold state."""
        try:
            from transformers import AutoConfig
            self._scaffold_config = AutoConfig.from_pretrained(model_name)
            
            required_attrs = ["hidden_size", "num_attention_heads", "num_hidden_layers", "max_position_embeddings"]
            for attr in required_attrs:
                if not hasattr(self._scaffold_config, attr):
                    raise ValueError(f"Scaffold model config missing {attr}")

            with self._lock:
                self._hidden_size = self._scaffold_config.hidden_size
                self._num_heads = self._scaffold_config.num_attention_heads
                self._num_layers = self._scaffold_config.num_hidden_layers
                self._device = device

                self._scaffold_state_tensor = torch.zeros(
                    (1, self._scaffold_config.max_position_embeddings, self._hidden_size),
                    device=self._device
                )
                
                self._logger.record_event(
                    event_type="scaffold_state_initialized",
                    message="Scaffold state initialized successfully",
                    level="info",
                    additional_info={
                        "hidden_size": self._hidden_size,
                        "num_heads": self._num_heads,
                        "num_layers": self._num_layers,
                        "max_position_embeddings": self._scaffold_config.max_position_embeddings,
                        "device": str(self._device),
                        "timestamp": time.time()
                    }
                )

            return True
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {
                "operation": "state_initialization",
                "model_name": model_name,
                "device": device
            })
            return False

    def verify_scaffold_compatibility(self, base_config: Any) -> bool:
        """Verify scaffold compatibility with base model."""
        try:
            if not self._scaffold_config:
                raise ValueError("Scaffold config not initialized")

            if self._scaffold_config.hidden_size % base_config.hidden_size != 0:
                self._logger.record_event(
                    event_type="scaffold_compatibility_error",
                    message="Incompatible hidden sizes",
                    level="error",
                    additional_info={
                        "scaffold_size": self._scaffold_config.hidden_size,
                        "base_size": base_config.hidden_size,
                        "timestamp": time.time()
                    }
                )
                return False

            if self._scaffold_config.num_attention_heads % base_config.num_attention_heads != 0:
                self._logger.record_event(
                    event_type="scaffold_compatibility_error",
                    message="Incompatible number of attention heads",
                    level="error",
                    additional_info={
                        "scaffold_heads": self._scaffold_config.num_attention_heads,
                        "base_heads": base_config.num_attention_heads,
                        "timestamp": time.time()
                    }
                )
                return False

            return True
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {
                "operation": "compatibility_verification",
                "base_config": base_config
            })
            return False

    def get_scaffold_stats(self) -> Dict:
        """Get scaffold statistics."""
        try:
            with self._lock:
                stats = {
                    "hidden_size": self._hidden_size,
                    "num_heads": self._num_heads,
                    "num_layers": self._num_layers,
                    "token_map_size": len(self._scaffold_state["token_map"]) if self._scaffold_state["token_map"] else 0,
                    "has_hidden_states": self._scaffold_state_tensor is not None,
                    "config_valid": self._validate_config() is None,
                    "device": str(self._device) if self._device else None,
                    "timestamp": time.time()
                }
                return stats
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {"operation": "stats_retrieval"})
            return {}

    def reset_scaffold_state(self) -> None:
        """Reset scaffold state."""
        with self._lock:
            self._scaffold_state["token_map"] = None
            self._scaffold_state_tensor = None
            self._logger.record_event(
                event_type="scaffold_state_reset",
                message="Scaffold state reset successfully",
                level="info",
                additional_info={"timestamp": time.time()}
            )

    def build_token_map(self, base_tokenizer: Any, scaffold_tokenizer: Any) -> Dict:
        """Build token mapping between base and scaffold models."""
        try:
            token_mapper = defaultdict(lambda: [scaffold_tokenizer.unk_token_id])
            
            for base_token, base_id in base_tokenizer.get_vocab().items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(
                    normalized, add_special_tokens=False, max_length=3, truncation=True
                ) or [scaffold_tokenizer.unk_token_id]
                token_mapper[base_id] = {'ids': scaffold_ids, 'weight': 1.0}
            
            special_token_map = {
                base_tokenizer.pad_token_id: scaffold_tokenizer.pad_token_id,
                base_tokenizer.eos_token_id: scaffold_tokenizer.eos_token_id or scaffold_tokenizer.sep_token_id,
                base_tokenizer.unk_token_id: scaffold_tokenizer.unk_token_id,
            }
            
            for base_id, scaffold_id in special_token_map.items():
                token_mapper[base_id] = {'ids': [scaffold_id], 'weight': 1.0}
            
            if not token_mapper:
                raise ValueError("Token map is empty")
                
            self._logger.record_event(
                event_type="token_map_built",
                message="Token map built successfully",
                level="info",
                additional_info={
                    "map_size": len(token_mapper),
                    "special_tokens": len(special_token_map),
                    "timestamp": time.time()
                }
            )
            
            return token_mapper
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {"operation": "token_map_building"})
            raise

    def validate_token_map(self) -> bool:
        """Validate token mapping."""
        try:
            token_map = self._scaffold_state["token_map"]
            if not token_map:
                return False
                
            required_tokens = ['pad_token_id', 'eos_token_id', 'unk_token_id']
            scaffold_tokenizer = self._scaffold_state["scaffold_model"].tokenizer
            for token in required_tokens:
                if not any(t['ids'][0] == getattr(scaffold_tokenizer, token) 
                          for t in token_map.values()):
                    return False
                    
            if not all(t['ids'] for t in token_map.values()):
                return False
                
            return True
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {"operation": "token_map_validation"})
            return False

    @contextlib.contextmanager
    def scaffold_context(self, hidden_states: torch.Tensor):
        """Context manager for scaffold operations."""
        try:
            if not self._scaffold_state["is_initialized"]:
                raise RuntimeError("ScaffoldProvider not initialized")
                
            prev_states = self._scaffold_state_tensor
            self._scaffold_state_tensor = hidden_states
            yield
            self._scaffold_state_tensor = prev_states
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {
                "operation": "scaffold_context",
                "hidden_states_shape": hidden_states.shape
            })
            raise

    def get_scaffold_hidden_states(
        self, 
        scaffold_inputs: Dict[str, torch.Tensor],
        scaffold_model: nn.Module
    ) -> torch.Tensor:
        """Get scaffold hidden states."""
        try:
            if not self._scaffold_state["is_initialized"]:
                raise RuntimeError("ScaffoldProvider not initialized")
                
            if self._scaffold_state_tensor is None:
                raise ValueError("Scaffold hidden states not initialized")
                
            if not isinstance(scaffold_inputs, dict):
                raise ValueError("scaffold_inputs must be a dictionary")
                
            if 'input_ids' not in scaffold_inputs or 'attention_mask' not in scaffold_inputs:
                raise ValueError("scaffold_inputs must contain 'input_ids' and 'attention_mask'")
                
            with torch.no_grad():
                scaffold_outputs = scaffold_model(
                    **{k: v for k, v in scaffold_inputs.items() if k in ['input_ids', 'attention_mask']},
                    output_hidden_states=True
                )
                hidden_states = (
                    scaffold_outputs.hidden_states[-1]
                    if hasattr(scaffold_outputs, 'hidden_states')
                    else scaffold_outputs.base_model_output.hidden_states[-1]
                )
                self._scaffold_state_tensor = hidden_states.detach()
                
            return self._scaffold_state_tensor
        except Exception as e:
            self._error_handler.handle_scaffold_error(e, {
                "operation": "hidden_states_retrieval",
                "inputs_shape": {k: v.shape for k, v in scaffold_inputs.items()}
            })
            raise
