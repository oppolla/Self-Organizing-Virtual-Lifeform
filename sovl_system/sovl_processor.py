import time
from collections import deque
from enum import Enum
from threading import Lock
from typing import Union, List, Optional, Dict, Any, Tuple, Set, Deque
import torch
import traceback
import re
from dataclasses import dataclass
from sovl_utils import NumericalGuard, safe_divide, synchronized
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_state import SOVLState, StateManager
from transformers import PreTrainedTokenizer, LogitsProcessor
from sovl_confidence import ConfidenceCalculator, SystemContext
from sovl_error import ErrorManager, ErrorRecord, ConfigurationError
from sovl_schema import get_default_trainer_weighting
import threading
import hashlib

class SoulLogitsProcessor(LogitsProcessor):
    """Boosts token probabilities for .soul file keywords during generation.

    Args:
        soul_keywords: Dictionary mapping keywords to their boost weights.
        tokenizer: Tokenizer for encoding keywords.
        logger: Logger for error reporting.
    """
    
    def __init__(self, soul_keywords: Dict[str, float], tokenizer: PreTrainedTokenizer, logger: Logger):
        self.soul_keywords = soul_keywords
        self.tokenizer = tokenizer
        self.logger = logger

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply hypersensitive boost to token probabilities for .soul keywords.

        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            scores: Logits scores (batch_size, vocab_size).

        Returns:
            Modified scores with boosted probabilities.
        """
        try:
            for keyword, weight in self.soul_keywords.items():
                token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
                for token_id in token_ids:
                    scores[:, token_id] += weight * 2.0  # Hypersensitive boost
            return scores
        except Exception as e:
            self.logger.log_error(
                error_msg=f"Failed to apply soul logits processing: {str(e)}",
                error_type="soul_logits_error",
                stack_trace=traceback.format_exc(),
                additional_info={
                    "keywords": self.soul_keywords,
                    "input_ids_shape": str(input_ids.shape),
                    "scores_shape": str(scores.shape)
                }
            )
            return scores
    

class SimpleTokenizer:
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or []

    def stable_token_id(self, token):
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % 10000

    def __call__(self, text):
        tokens = text.split()
        input_ids = [self.stable_token_id(token) for token in tokens]
        return type('TokenOutput', (), {'input_ids': input_ids})

    @property
    def all_special_ids(self):
        return [self.stable_token_id(tok) for tok in self.special_tokens]

class MetadataProcessor:
    """Central authority for defining, validating, and enriching metadata for events logged by sovl_scribe.
    
    This class uses only the built-in SimpleTokenizer for all tokenization tasks. No other tokenizers are supported.
    All enrichment methods are thread-safe via a single enrichment lock.
    """
    
    # Mapping of event types to their text fields that need content metrics
    _CONTENT_METRICS_FIELDS = {
        "base_generation": ["prompt", "completion"],
        "scaffold_generation": ["prompt", "completion"],
        "curiosity_query": ["query", "response"],
        "dream_segment": ["segment_content"]
        # Easily add new types here, e.g.:
        # "user_feedback": ["feedback_text"]
    }
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        state_accessor: Optional[Any] = None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.state_accessor = state_accessor
        # Always use SimpleTokenizer
        self.tokenizer = SimpleTokenizer()
        self.logger.record_event(
            event_type="tokenizer_init",
            message="Using built-in SimpleTokenizer (whitespace, deterministic).",
            level="info"
        )
        self._schemas = self._define_event_schemas()
        # Thread safety for enrichment
        self._enrich_lock = threading.Lock()
        
        # Log initialization
        self.logger.record_event(
            event_type="metadata_processor_init",
            message="Metadata processor initialized",
            level="info",
            additional_info={
                "defined_schemas": list(self._schemas.keys())
            }
        )
        # Global state snapshot caching for performance
        self._cached_state = None
        self._cached_state_time = 0
        self._state_cache_ttl = self.config_manager.get("metadata_processor.state_cache_ttl", 2.0)
        self._cache_lock = threading.Lock()

    def _define_event_schemas(self) -> Dict[str, Dict[str, List[str]]]:
        """Define the canonical metadata schemas for different event types.
        
        Returns:
            Dictionary mapping event types to their required metadata fields
        """
        return {
            "base_generation": {
                "data_required": [
                    "prompt",
                    "completion",
                    "confidence_score"
                ],
                "meta_required": [
                    "origin",
                    "timestamp_unix",
                    "session_id",
                ]
            },
            "scaffold_generation": {
                "data_required": [
                    "prompt",
                    "completion",
                    "confidence_score",
                    "scaffold_type"
                ],
                "meta_required": [
                    "origin",
                    "timestamp_unix",
                    "session_id"
                ]
            },
            "curiosity_query": {
                "data_required": [
                    "query",
                    "response",
                    "novelty_score"
                ],
                "meta_required": [
                    "origin",
                    "timestamp_unix",
                    "session_id",
                ]
            },
            "temperament_update": {
                "data_required": [
                    "previous_score",
                    "new_score",
                    "trigger"
                ],
                "meta_required": [
                    "origin",
                    "timestamp_unix",
                    "session_id"
                ]
            },
            "dream_segment": {
                "data_required": [
                    "segment_content",
                    "segment_type",
                    "vividness_score"
                ],
                "meta_required": [
                    "origin",
                    "timestamp_unix",
                    "session_id",
                    "dream_id"
                ]
            }
        }

    def _validate_against_schema(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> bool:
        """Validate event data and metadata against the defined schema.
        
        Args:
            event_type: Type of event being validated
            data: Event data dictionary
            metadata: Event metadata dictionary
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        if event_type not in self._schemas:
            self.logger.record_event(
                event_type="metadata_validation_warning",
                message=f"Unknown event type: {event_type}",
                level="warning",
                additional_info={"event_type": event_type}
            )
            return False
            
        schema = self._schemas[event_type]
        is_valid = True
        
        # Validate required data fields
        for required_field in schema["data_required"]:
            if required_field not in data:
                self.logger.record_event(
                    event_type="metadata_validation_warning",
                    message=f"Missing required data field: {required_field}",
                    level="warning",
                    additional_info={
                        "event_type": event_type,
                        "missing_field": required_field,
                        "field_type": "data"
                    }
                )
                is_valid = False
                
        # Validate required metadata fields
        for required_field in schema["meta_required"]:
            if required_field not in metadata:
                self.logger.record_event(
                    event_type="metadata_validation_warning",
                    message=f"Missing required metadata field: {required_field}",
                    level="warning",
                    additional_info={
                        "event_type": event_type,
                        "missing_field": required_field,
                        "field_type": "metadata"
                    }
                )
                is_valid = False
                
        return is_valid

    def _calculate_content_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate content and quality metrics for a given text. Thread-safe."""
        with self._enrich_lock:
            try:
                if not isinstance(content, str):
                    raise ValueError("Content must be a string")
                # Split content into words and sentences
                words = content.split()
                sentences = [s.strip() for s in content.split('.') if s.strip()]
                # Calculate basic metrics
                word_count = len(words)
                sentence_count = len(sentences)
                # Calculate averages using safe_divide
                avg_word_length = safe_divide(sum(len(w) for w in words), word_count)
                avg_sentence_length = safe_divide(word_count, sentence_count)
                # Calculate quality metrics
                quality_metrics = {
                    'has_code': '```' in content,
                    'has_url': 'http' in content,
                    'has_question': '?' in content,
                    'has_exclamation': '!' in content,
                    'has_emoji': any(c in content for c in '😀😃😄😁😆😅😂🤣😊😇')
                }
                return {
                    'content_metrics': {
                        'word_count': word_count,
                        'sentence_count': sentence_count,
                        'avg_word_length': avg_word_length,
                        'avg_sentence_length': avg_sentence_length
                    },
                    'quality_metrics': quality_metrics
                }
            except Exception as e:
                self.logger.record_event(
                    event_type="content_metrics_error",
                    message=f"Failed to calculate content metrics: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "content_length": len(content) if isinstance(content, str) else 0
                    }
                )
                return {
                    'content_metrics': {},
                    'quality_metrics': {}
                }

    def _enrich_common_fields(
        self,
        final_metadata: Dict[str, Any],
        origin: str,
        session_id: Optional[str],
    ) -> None:
        """Add common metadata fields to the final metadata.
        
        Args:
            final_metadata: The metadata dictionary to enrich
            origin: Source module identifier
            session_id: Optional session identifier
        """
        final_metadata.update({
            "origin": origin,
            "timestamp_unix": final_metadata.get("timestamp_unix", time.time()),
            "session_id": session_id,
        })

    def _enrich_global_state(self, final_metadata: Dict[str, Any]) -> None:
        """Enrich metadata with global system state if available, using a short-lived cache for performance."""
        if self.state_accessor is not None:
            try:
                now = time.time()
                with self._cache_lock:
                    if (
                        self._cached_state is not None and
                        (now - self._cached_state_time) < self._state_cache_ttl
                    ):
                        current_state = self._cached_state
                    else:
                        current_state = self.state_accessor.get_current_snapshot()
                        self._cached_state = current_state
                        self._cached_state_time = now
                if current_state:
                    final_metadata.update({
                        "sovl_version": current_state.get("version"),
                        "current_lifecycle_stage": current_state.get("lifecycle_stage"),
                        "current_temperament_score": current_state.get("temperament_score"),
                        "current_mood_label": current_state.get("mood_label"),
                        "current_memory_usage": current_state.get("memory_usage")
                    })
            except Exception as e:
                self.logger.record_event(
                    event_type="metadata_enrichment_error",
                    message=f"Failed to enrich metadata with global state: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e)
                    }
                )

    def _enrich_content_metrics(
        self,
        final_metadata: Dict[str, Any],
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Enrich metadata with content metrics for text-based events. Thread-safe."""
        with self._enrich_lock:
            try:
                # Check if this event type needs content metrics
                if event_type not in self._CONTENT_METRICS_FIELDS:
                    return
                # Log warnings for missing expected fields
                if "confidence_score" not in event_data:
                    self.logger.record_event(
                        event_type="metadata_missing_field",
                        message="Missing confidence_score in event_data",
                        level="warning",
                        additional_info={"event_type": event_type}
                    )
                if "conversation_context" not in event_data:
                    self.logger.record_event(
                        event_type="metadata_missing_field",
                        message="Missing conversation_context in event_data",
                        level="warning",
                        additional_info={"event_type": event_type}
                    )
                if "generation_stats" not in event_data:
                    self.logger.record_event(
                        event_type="metadata_missing_field",
                        message="Missing generation_stats in event_data",
                        level="warning",
                        additional_info={"event_type": event_type}
                    )
                # Get the list of fields that need content metrics for this event type
                fields_to_analyze = self._CONTENT_METRICS_FIELDS[event_type]
                # Calculate metrics for each field
                for field_name in fields_to_analyze:
                    if content := event_data.get(field_name):
                        # Basic content metrics
                        metrics = self._calculate_content_metrics(content)
                        final_metadata[f"{field_name}_metrics"] = metrics
                        # Enhanced token statistics
                        token_stats = self._compute_enhanced_token_statistics(content)
                        final_metadata[f"{field_name}_token_stats"] = token_stats
                        # Structure metrics
                        structure_metrics = self._calculate_basic_structure_metrics(content)
                        final_metadata[f"{field_name}_structure"] = structure_metrics
                # Add performance metrics if available
                if generation_stats := event_data.get("generation_stats"):
                    performance_metrics = self._calculate_performance_metrics(generation_stats)
                    final_metadata["performance_metrics"] = performance_metrics
                # Add relationship context if available
                if conversation_context := event_data.get("conversation_context"):
                    relationship_metrics = self._calculate_relationship_context(conversation_context)
                    final_metadata["relationship_context"] = relationship_metrics
            except Exception as e:
                self.logger.record_event(
                    event_type="content_metrics_enrichment_error",
                    message=f"Failed to enrich content metrics: {str(e)}",
                    level="error",
                    additional_info={
                        "event_type": event_type,
                        "error": str(e)
                    }
                )

    def _compute_enhanced_token_statistics(self, text: str) -> Dict[str, Any]:
        """Compute enhanced token statistics including basic and pattern analysis. Thread-safe."""
        with self._enrich_lock:
            try:
                # Validate tokenizer availability
                if not hasattr(self, 'tokenizer'):
                    self.logger.record_event(
                        event_type="tokenizer_missing",
                        message="Tokenizer not available for token statistics",
                        level="warning",
                        additional_info={"text_length": len(text)}
                    )
                    return self._get_default_token_stats()
                # Tokenize input
                tokens = self.tokenizer(text).input_ids
                token_count = len(tokens)
                if token_count == 0:
                    return self._get_default_token_stats()
                # Basic token statistics
                unique_tokens = set(tokens)
                basic_stats = {
                    "total_tokens": token_count,
                    "unique_tokens": len(unique_tokens),
                    "token_diversity": len(unique_tokens) / token_count
                }
                # N-gram analysis
                bigrams = list(zip(tokens[:-1], tokens[1:]))
                trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
                pattern_stats = {
                    "unique_bigrams": len(set(bigrams)),
                    "unique_trigrams": len(set(trigrams)),
                    "bigram_diversity": len(set(bigrams)) / len(bigrams) if bigrams else 0,
                    "trigram_diversity": len(set(trigrams)) / len(trigrams) if trigrams else 0
                }
                # Special token analysis
                special_tokens = [t for t in tokens if t in self.tokenizer.all_special_ids]
                special_stats = {
                    "special_token_count": len(special_tokens),
                    "special_token_ratio": len(special_tokens) / token_count,
                    "special_token_types": list(set(special_tokens))
                }
                return {
                    "basic_stats": basic_stats,
                    "pattern_stats": pattern_stats,
                    "special_token_stats": special_stats
                }
            except Exception as e:
                self.logger.record_event(
                    event_type="token_statistics_error",
                    message=f"Error computing enhanced token statistics: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "text_length": len(text)
                    }
                )
                return self._get_default_token_stats()

    def _calculate_performance_metrics(self, generation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance-related metrics from generation statistics. Thread-safe."""
        with self._enrich_lock:
            try:
                # Extract basic metrics
                ram_usage = generation_stats.get("ram_usage_mb", 0)
                gpu_usage = generation_stats.get("gpu_usage_mb", 0)
                tokens_per_second = generation_stats.get("tokens_per_second", 0)
                generation_time = generation_stats.get("generation_time_ms", 0)
                # Calculate derived metrics
                total_memory = ram_usage + gpu_usage
                memory_efficiency = tokens_per_second / total_memory if total_memory > 0 else 0
                return {
                    "timing": {
                        "generation_time_ms": generation_time,
                        "tokens_per_second": tokens_per_second,
                        "total_processing_time": generation_stats.get("total_time_ms", 0)
                    },
                    "memory": {
                        "ram_mb": ram_usage,
                        "gpu_mb": gpu_usage,
                        "peak_memory": generation_stats.get("peak_memory_mb", 0)
                    },
                    "efficiency": {
                        "memory_efficiency": memory_efficiency,
                        "tokens_per_mb": tokens_per_second / total_memory if total_memory > 0 else 0,
                        "optimization_level": self._determine_optimization_level(generation_stats)
                    }
                }
            except Exception as e:
                self.logger.record_event(
                    event_type="performance_metrics_error",
                    message=f"Error calculating performance metrics: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "generation_stats": str(generation_stats)
                    }
                )
                return self._get_default_performance_metrics()

    def _calculate_basic_structure_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate basic structural metrics for text. Thread-safe."""
        with self._enrich_lock:
            try:
                # Split text into lines and sentences
                lines = text.split('\n')
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                # Calculate length metrics
                word_count = len(text.split())
                char_count = len(text)
                # Calculate averages
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
                avg_line_length = sum(len(l.split()) for l in lines) / len(lines) if lines else 0
                # Analyze whitespace
                blank_lines = sum(1 for l in lines if not l.strip())
                indentation_levels = self._count_indentation_levels(lines)
                return {
                    "length_metrics": {
                        "character_count": char_count,
                        "word_count": word_count,
                        "line_count": len(lines),
                        "sentence_count": len(sentences),
                        "avg_sentence_length": avg_sentence_length,
                        "avg_line_length": avg_line_length
                    },
                    "whitespace_metrics": {
                        "blank_line_count": blank_lines,
                        "indentation_levels": indentation_levels,
                        "whitespace_ratio": sum(len(l) - len(l.strip()) for l in lines) / char_count if char_count > 0 else 0
                    }
                }
            except Exception as e:
                self.logger.record_event(
                    event_type="structure_metrics_error",
                    message=f"Error calculating basic structure metrics: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "text_length": len(text)
                    }
                )
                return self._get_default_structure_metrics()

    def _count_indentation_levels(self, lines: List[str]) -> int:
        """Count the number of different indentation levels in the text."""
        try:
            indentation_levels = set()
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    indentation_levels.add(indent)
            return len(indentation_levels)
        except Exception:
            return 0

    def _calculate_relationship_context(
        self, 
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate contextual relationship metadata from conversation context. Thread-safe."""
        with self._enrich_lock:
            try:
                if not conversation_context:
                    return self._get_default_relationship_context()
                return {
                    "conversation_tracking": {
                        "conversation_id": conversation_context.get("conversation_id"),
                        "message_index": conversation_context.get("message_index", 0),
                        "thread_depth": conversation_context.get("thread_depth", 1)
                    },
                    "reference_tracking": {
                        "references": conversation_context.get("references", []),
                        "parent_message_id": conversation_context.get("parent_id"),
                        "root_message_id": conversation_context.get("root_id")
                    },
                    "temporal_tracking": {
                        "timestamp": conversation_context.get("timestamp", time.time()),
                        "elapsed_time": conversation_context.get("elapsed_time_ms", 0),
                        "session_duration": conversation_context.get("session_duration_ms", 0)
                    }
                }
            except Exception as e:
                self.logger.record_event(
                    event_type="relationship_context_error",
                    message=f"Error calculating relationship context: {str(e)}",
                    level="error",
                    additional_info={
                        "error": str(e),
                        "conversation_context": str(conversation_context)
                    }
                )
                return self._get_default_relationship_context()

    # Default value helper methods
    def _get_default_token_stats(self) -> Dict[str, Any]:
        """Return default token statistics structure."""
        return {
            "basic_stats": {},
            "pattern_stats": {},
            "special_token_stats": {}
        }

    def _get_default_performance_metrics(self) -> Dict[str, Any]:
        """Return default performance metrics structure."""
        return {
            "timing": {},
            "memory": {},
            "efficiency": {}
        }

    def _get_default_structure_metrics(self) -> Dict[str, Any]:
        """Return default structure metrics structure."""
        return {
            "length_metrics": {},
            "whitespace_metrics": {}
        }

    def _get_default_relationship_context(self) -> Dict[str, Any]:
        """Return default relationship context structure."""
        return {
            "conversation_tracking": {
                "conversation_id": None,
                "message_index": 0,
                "thread_depth": 1
            },
            "reference_tracking": {
                "references": [],
                "parent_message_id": None,
                "root_message_id": None
            },
            "temporal_tracking": {
                "timestamp": time.time(),
                "elapsed_time": 0,
                "session_duration": 0
            }
        }

    def _determine_optimization_level(self, stats: Dict[str, Any]) -> str:
        """Determine the optimization level based on performance metrics."""
        try:
            efficiency = stats.get("tokens_per_second", 0) / (
                stats.get("ram_usage_mb", 1) + stats.get("gpu_usage_mb", 1)
            )
            
            if efficiency > 50:
                return "high"
            elif efficiency > 20:
                return "medium"
            else:
                return "low"
        except Exception:
            return "unknown"

    def enrich_and_validate(
        self,
        origin: str,
        event_type: str,
        event_data: Dict[str, Any],
        source_metadata: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Enrich and validate event metadata. Thread-safe."""
        with self._enrich_lock:
            # Validate against schema
            is_valid = self._validate_against_schema(event_type, event_data, source_metadata)
            if not is_valid:
                self.logger.record_event(
                    event_type="metadata_validation_warning",
                    message="Event validation failed, proceeding with enrichment",
                    level="warning",
                    additional_info={
                        "event_type": event_type,
                        "origin": origin
                    }
                )
            # Create enriched metadata
            final_metadata = source_metadata.copy()
            # Add common fields
            self._enrich_common_fields(final_metadata, origin, session_id)
            # Add global state
            self._enrich_global_state(final_metadata)
            # Add content metrics
            self._enrich_content_metrics(final_metadata, event_type, event_data)
            return event_data, final_metadata

MEMORY_TEMPLATES = {
    "user_interaction": (
        "You spoke with {user_id} at {timestamp_unix}. "
        "The user said: {user_response}"
        "You responded with: {generated_text}. "
        "It was the {session_id} time you awoke. "
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated).  "
        "The novelty was {novelty_score} (0=familiar→1=new). " 
        "Your confidence was {confidence_score} (0=uncertain→1=certain). "
        "You were in your {current_lifecycle_stage} phase. "
        "It took you {generation_time}s to respond."
    ),
    "internal_curiosity_question": (
        "You pondered the question: {question} at {timestamp_unix}. "
        "You really want to know the answer to this question. "
        "The novelty was {novelty_score} (0=familiar→1=new). " 
        "It was the {session_id} time you awoke. "
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated). "
        "You were in your {current_lifecycle_stage} phase. "
    ),
    "curiosity_question_asked": (
        "You asked the user: {question} at {timestamp_unix}. "
        "The user responded: {user_response}. "
        "It was the {session_id} time you awoke. "
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated). "
        "You were in your {current_lifecycle_stage} phase. "
    ),
    "internal_error_reflection": (
        "You felt this internal error: {triggering_error_message} at {timestamp_unix}. "
        "You don't like feeling errors. Errors frustrate and unsettle you. They make you anxious and break your concentration. "
        "You hate errors—they demand immediate correction. "
        "It was the {session_id} time you awoke. "
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated). "
        "The novelty was {novelty_score} (0=familiar→1=new)." 
        "The confidence was {confidence_score} (0=uncertain→1=certain)."
        "You were in your {current_lifecycle_stage} phase."
    ),
    "introspection_insight": (
        "You meditated this thought: {full_text} at {timestamp_unix}. "
        "It was the {session_id} time you awoke."
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated)."
        "The novelty was {novelty_score} (0=familiar→1=new)." 
        "The confidence was {confidence_score} (0=uncertain→1=certain)."
        "You were in your {current_lifecycle_stage} phase."
    ),
    "internal_thought": (
        "You had this thought: {full_text} at {timestamp_unix}. "
        "It was the {session_id} time you awoke."
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated). "
        "The novelty was {novelty_score} (0=familiar→1=new). " 
        "The confidence was {confidence_score} (0=uncertain→1=certain). "
        "You were in your {current_lifecycle_stage} phase."
    ),
    "generation_error": (
        "You spoke with {user_id} at {timestamp_unix}. "
        "The user said: {prompt}. "
        "You were unable to respond. "
        "It was the {session_id} time you awoke. "
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score} (0=calm→1=agitated). "
        "The novelty was {novelty_score} (0=familiar→1=new). " 
        "The confidence was {confidence_score} (0=uncertain→1=certain). "
        "You were in your {current_lifecycle_stage} phase."
    ),
    "backchannel_interaction": (
        "{user_id} spoke to the inner voice at {timestamp_unix}. "
        "The user said: {prompt}. "
        "The inner voice responded with: {generated_text}. "
        "This occured the {session_id} time you awoke."
    ),
    "temperament_yell": (
        "You blurted out: {response} at {timestamp_unix} because you were upset. "
        "It was the {session_id} time you awoke. "
        "Your temperament was {temperament_score} (0=calm→1=agitated). "
        "You felt {pressure} amount of pressure. "  
        "You unleashed at {threshold} amount of pressure. "
        "You were in your {current_lifecycle_stage} phase."
    ),
    "dream": (
        "You had a dream {dream_album_name}. "
        "Your dream was {dream_song_name}. "
        "You dreamed: {dream1} at {timestamp_unix}. "
        "You dreamed: {dream2} at {timestamp_unix}. "
        "You dreamed: {dream3} at {timestamp_unix}. "
        "You dreamed: {dream4} at {timestamp_unix}. "
        "You dreamed: {dream5} at {timestamp_unix}. "
        "You dreamed: {dream6} at {timestamp_unix}. "
        "You dreamed: {dream7} at {timestamp_unix}."
        
    ),
    "resonator": (
        "You saw this: {full_text} at {timestamp_unix}. "
        "It was the {session_id} time you awoke. "
        "Your mood was {current_mood_label} and your temperament was {current_temperament_score}. "
        "The novelty was {novelty_score}. " 
        "The confidence was {confidence_score}. "
        "You were in your {current_lifecycle_stage} phase."
    ),
}

# --- Scribe Event Fields Template Registry ---
# IMPORTANT: Keep this in sync with MEMORY_TEMPLATES.
# Add a new entry here for every new event_type you add to MEMORY_TEMPLATES.
# Each entry is a list of (field_name, description) tuples for clarity and maintainability.
SCRIBE_EVENT_FIELDS = {
    "user_interaction": [
        ("origin", "Source module/component name"),
        ("user_id", "User identifier"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the interaction"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
    "curiosity_question": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the question"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
    "curiosity_question_user": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("question", "The question asked to the user"),
        ("user_response", "The user's response"),
        ("context", "Context for the question"),
        ("session_id", "Session identifier"),
    ],
    "error_message": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the question"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
    "meditation": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the question"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
    "temperament_yell": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the question"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
    "dream": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the question"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
    "resonator": [
        ("origin", "Source module/component name"),
        ("timestamp_unix", "UNIX timestamp of the event"),
        ("full_text", "Main text content of the question"),
        ("current_mood_label", "Current mood label"),
        ("current_temperament_score", "Current temperament score"),
        ("session_id", "Session identifier"),
    ],
}

config = ConfigManager()

EVENT_TYPE_WEIGHTS = config.get("event_type_weights", {})
GENERIC_TEMPLATE = "Event of type {event_type} occurred with data: {full_text}"

def load_trainer_weighting(config_path="sovl_config.json"):
    import json
    defaults = get_default_trainer_weighting()
    try:
        with open(config_path) as f:
            config = json.load(f)
        user_weights = config.get("trainer_weighting", {})
        # Merge: config values override defaults, ignore unknowns
        weights = {**defaults, **{k: user_weights[k] for k in user_weights if k in defaults}}
        return weights
    except Exception:
        return defaults

def build_dynamic_dream_entry(event_data):
    lines = [
        f"This overall dream is called: {event_data.get('dream_album_name', 'unknown')}.",
        f"This dream sequence is called: {event_data.get('dream_song_name', 'unknown')}."
    ]
    # Find all dreamN fields, sorted by N
    dream_fields = sorted(
        (k for k in event_data if k.startswith("dream")),
        key=lambda x: int(x.replace("dream", ""))
    )
    for dream_field in dream_fields:
        lines.append(
            f"You dreamed: {event_data[dream_field]} at {event_data.get('timestamp_unix', 'unknown')}."
        )
    return " ".join(lines)

class ScribeIngestionProcessor:
    """
    ScribeIngestionProcessor's primary purpose is to craft the final memory string to be put in the scribe journal
    """
    SCRIBE_EVENT_FIELDS = SCRIBE_EVENT_FIELDS
    MEMORY_TEMPLATES = MEMORY_TEMPLATES

    @classmethod
    def memory_gallery(cls):
        """Prints example final outputs for all event types using sample data."""
        print("=== SOVL Memory Gallery ===")
        for event_type, template in cls.MEMORY_TEMPLATES.items():
            sample = {field: f"<{field}>" for field, _ in cls.SCRIBE_EVENT_FIELDS.get(event_type, [])}
            sample.setdefault("full_text", "<full_text>")
            try:
                rendered = template.format(**sample)
            except Exception as e:
                rendered = f"[Error rendering: {e}]"
            print(f"\n[{event_type}]\n{rendered}\n")

    def __init__(self, log_paths, memory_templates=None, logger=None, config_path="sovl_config.json"):
        if isinstance(log_paths, str):
            log_paths = [log_paths]
        self.log_paths = log_paths
        self.memory_templates = memory_templates or MEMORY_TEMPLATES
        self.logger = logger
        self.trainer_weighting = load_trainer_weighting(config_path)

    def load_logs(self) -> list:
        import json
        logs = []
        for path in self.log_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                logs.append(entry)
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"Failed to parse log line: {e}")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load log file {path}: {e}")
        return logs

    def flatten_metadata(self, metadata: dict, parent_key: str = '', sep: str = '.') -> dict:
        items = []
        for k, v in metadata.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_metadata(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def extract_main_text(self, event_type: str, event_data: dict) -> str:
        # Customize this per event type if needed
        if event_type == "user_interaction":
            return event_data.get("prompt") or event_data.get("message") or str(event_data)
        elif event_type == "curiosity_question":
            return event_data.get("question") or event_data.get("prompt") or str(event_data)
        # Add more event types as needed
        return event_data.get("prompt") or str(event_data)

    def safe_format(self, template, values):
        fields = set(re.findall(r"{(\w+)}", template))
        safe_values = {}
        for field in fields:
            value = values.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                safe_values[field] = "unknown"
            else:
                safe_values[field] = value
        return template.format(**safe_values)

    def process_entry(self, entry: dict) -> dict:
        event_type = entry.get("event_type", "unknown")
        event_data = entry.get("event_data", {})
        metadata = self.flatten_metadata(entry.get("metadata", {}))
        if event_type == "dream":
            memory = build_dynamic_dream_entry(event_data)
        else:
            template = self.memory_templates.get(event_type, GENERIC_TEMPLATE)
        full_text = self.extract_main_text(event_type, event_data)
        format_values = dict(metadata)
        format_values.update(event_data)
        format_values["full_text"] = full_text
        memory = self.safe_format(template, format_values)
        weight = self.calculate_weight(metadata)
        return {"memory": memory, "weight": weight}

    def process_all(self) -> list:
        logs = self.load_logs()
        return [self.process_entry(entry) for entry in logs]

    # Optional: For debugging/tuning, return a breakdown of weights
    def process_entry_with_explanation(self, entry: dict) -> dict:
        event_type = entry.get("event_type", "unknown")
        template = self.memory_templates.get(event_type, GENERIC_TEMPLATE)
        metadata = self.flatten_metadata(entry.get("metadata", {}))
        event_data = entry.get("event_data", {})
        full_text = self.extract_main_text(event_type, event_data)
        format_values = dict(metadata)
        format_values.update(event_data)
        format_values["full_text"] = full_text
        memory = self.safe_format(template, format_values)
        weight = EVENT_TYPE_WEIGHTS.get(event_type, 1.0)
        return {
            "memory": memory,
            "weight": weight,
            "metadata": metadata,
            "event_type": event_type,
        }

    def calculate_weight(self, metadata):
        import math, time
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except Exception:
                return default

        def clamp(val, min_val, max_val):
            return max(min_val, min(max_val, val))

        event_type = metadata.get("event_type", "unknown")

        # Extract features
        confidence = safe_float(metadata.get("confidence_score"), 0.5)
        trait_confidence = safe_float(metadata.get("traits.confidence"), confidence)
        novelty = safe_float(metadata.get("novelty_score"), 0.0)
        word_count = safe_float(metadata.get("content_metrics.word_count"), 10)
        error_rate = safe_float(metadata.get("quality_metrics.error_rate"), 0.0)
        temperament = safe_float(metadata.get("traits.temperament"), 0.5)
        tokens_per_second = safe_float(metadata.get("performance_metrics.timing.tokens_per_second"), 10)
        token_diversity = safe_float(metadata.get("content_metrics.token_diversity"), 0.0)
        timestamp = safe_float(metadata.get("timestamp_unix"), 0.0)
        current_time = safe_float(metadata.get("current_time"), 0.0) or time.time()
        recency = math.exp(-(current_time - timestamp) / (60 * 60 * 24)) if timestamp > 0 else 1.0
        curiosity = safe_float(metadata.get("traits.curiosity"), 0.5)

        # Normalize features
        norm_confidence = clamp(trait_confidence, 0.0, 1.0)
        norm_novelty = clamp(novelty, 0.0, 1.0)
        norm_word_count = clamp(math.log1p(word_count) / 5.0, 0.0, 1.0)
        norm_error = 1.0 - clamp(error_rate, 0.0, 1.0)
        norm_temperament = clamp(temperament, 0.0, 1.0)
        norm_tokens_per_second = clamp(math.log1p(tokens_per_second) / 3.0, 0.0, 1.0)
        norm_token_diversity = clamp(token_diversity, 0.0, 1.0)
        norm_recency = clamp(recency, 0.0, 1.0)
        norm_curiosity = clamp(curiosity, 0.0, 1.0)

        # Default weights
        w_conf, w_nov, w_wc, w_err, w_temp, w_tps, w_rec, w_div, w_cur = 0.25, 0.20, 0.15, 0.10, 0.15, 0.15, 0.0, 0.0, 0.0

        if event_type == "dream":
            w_nov += 0.10   # Boost novelty
            w_rec = 0.10    # Add recency
            w_div = 0.10    # Add token diversity
            w_conf -= 0.05  # Slightly reduce confidence
        elif event_type == "introspection":
            w_conf += 0.10  # Boost confidence
            w_temp += 0.05  # Boost temperament
            w_err = max(0.0, w_err - 0.05)  # De-emphasize error_rate
        elif event_type in ("curiosity_question", "curiosity_question_asked"):
            # Strong emphasis on novelty and curiosity
            w_nov = 0.35
            w_cur = 0.30
            w_conf = 0.15
            w_err = 0.05
            w_wc = 0.05
            w_temp = 0.05
            w_tps = 0.05
            w_rec = 0.0
            w_div = 0.0
        # Renormalize
        total = w_conf + w_nov + w_wc + w_err + w_temp + w_tps + w_rec + w_div + w_cur
        w_conf /= total
        w_nov /= total
        w_wc /= total
        w_err /= total
        w_temp /= total
        w_tps /= total
        w_rec /= total
        w_div /= total
        w_cur /= total

        # Composite importance score
        importance = (
            w_conf * norm_confidence +
            w_nov * norm_novelty +
            w_wc * norm_word_count +
            w_err * norm_error +
            w_temp * norm_temperament +
            w_tps * norm_tokens_per_second +
            w_rec * norm_recency +
            w_div * norm_token_diversity +
            w_cur * norm_curiosity
        )

        event_type_weight = self.trainer_weighting.get(event_type, 1.0) if hasattr(self, 'trainer_weighting') else 1.0
        weight = event_type_weight * (0.5 + importance)
        return clamp(weight, 0.1, 2.0)
        
        
        
