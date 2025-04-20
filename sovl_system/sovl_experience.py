import torch
import json
import os
from collections import deque, defaultdict
from threading import Lock
import time
import traceback
from typing import Optional, Dict, List, Tuple, Any, Union
from sovl_logger import Logger
from sovl_state import SOVLState, ConversationHistory
from sovl_utils import memory_usage, safe_divide
from sovl_config import ConfigManager
from sovl_hardware import HardwareManager
import gc
from sovl_memory import RAMManager, GPUMemoryManager

class MemoriaManager:
    """Manages the core memory system for SOVL."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize MemoriaManager with configuration and logger."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self._state = None
        self._conversation_history = None
        self._ram_manager = None
        self._gpu_manager = None
        
        # Initialize storage
        self._initialize_storage()
        
        # Initialize memory managers
        self._initialize_memory_managers()
        
        # Log initialization
        self._logger.record_event(
            event_type="memoria_manager_initialized",
            message="Memoria manager initialized",
            level="info"
        )

    def _initialize_storage(self) -> None:
        """Initialize memory storage systems."""
        with self._memory_lock:
            try:
                # Initialize conversation history
                self._conversation_history = ConversationHistory()
                
                # Initialize state
                self._state = SOVLState()
                
                # Log successful initialization
                self._logger.record_event(
                    event_type="memoria_storage_initialized",
                    message="Memoria storage initialized successfully",
                    level="info"
                )
                
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Failed to initialize memoria storage: {str(e)}",
                    error_type="storage_error",
                    stack_trace=traceback.format_exc()
                )
                raise

    def save_state(self, path_prefix: str) -> None:
        """Save current state to disk."""
        try:
            state = {
                "conversation_history": self._conversation_history.get_state(),
                "state": self._state.get_state()
            }
            
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            with open(f"{path_prefix}_memoria.json", 'w') as f:
                json.dump(state, f)
                
            self._logger.record_event(
                event_type="memoria_state_saved",
                message="Memoria state saved successfully",
                level="info"
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to save memoria state: {str(e)}",
                error_type="save_error",
                stack_trace=traceback.format_exc()
            )
            raise

    def load_state(self, path_prefix: str) -> None:
        """Load state from disk."""
        try:
            with open(f"{path_prefix}_memoria.json", 'r') as f:
                state = json.load(f)
                
            self._conversation_history.load_state(state["conversation_history"])
            self._state.load_state(state["state"])
            
            self._logger.record_event(
                event_type="memoria_state_loaded",
                message="Memoria state loaded successfully",
                level="info"
            )
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to load memoria state: {str(e)}",
                error_type="load_error",
                stack_trace=traceback.format_exc()
            )
            raise

class MetadataProcessor:
    """Handles metadata collection, analysis, and weighting for training examples."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize metadata processor with configuration and logger."""
        self.config_manager = config_manager
        self.logger = logger
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and validate metadata configuration."""
        self.config = self.config_manager.get_section("gestation_weighting")
        self.quality_weights = self.config["metadata_fields"]["quality_metrics"]
        self.content_weights = self.config["content_weights"]
        self.confidence_weights = self.config["confidence_weights"]
        self.temperament_weights = self.config["temperament_weights"]
        self.context_weights = self.config["context_weights"]
        self.timing_weights = self.config["timing_weights"]
        self.weight_bounds = self.config["weight_bounds"]
    
    def collect_metadata(self, content: str) -> Dict[str, Any]:
        """Collect metadata from content."""
        # Split content into words and sentences
        words = content.split()
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        return {
            'content_metrics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'avg_sentence_length': avg_sentence_length
            },
            'quality_metrics': {
                'has_code': '```' in content,
                'has_url': 'http' in content,
                'has_question': '?' in content,
                'has_exclamation': '!' in content,
                'has_emoji': any(c in content for c in '😀😃😄😁😆😅😂🤣😊😇')
            }
        }
    
    def calculate_weight(self, metadata: Dict[str, Any]) -> float:
        """Calculate weight for a training example based on its metadata."""
        weight = 1.0
        
        # Quality metrics weighting
        for feature, config in self.quality_weights.items():
            if config["enabled"] and metadata['quality_metrics'].get(f"has_{feature}", False):
                weight *= config["weight"]
        
        # Content metrics weighting
        content_metrics = metadata['content_metrics']
        
        # Word count balance
        word_count = content_metrics['word_count']
        if word_count > 0:
            weight *= self.content_weights["word_count_ratio_scale"]
        
        # Word length optimization
        avg_word_length = content_metrics['avg_word_length']
        word_length_range = self.content_weights["optimal_word_length_range"]
        if word_length_range["min"] <= avg_word_length <= word_length_range["max"]:
            weight *= word_length_range["weight"]
        else:
            weight *= self.content_weights["suboptimal_word_length_weight"]
        
        # Sentence structure optimization
        sentence_count = content_metrics['sentence_count']
        sentence_count_range = self.content_weights["optimal_sentence_count_range"]
        if sentence_count_range["min"] <= sentence_count <= sentence_count_range["max"]:
            weight *= sentence_count_range["weight"]
        elif sentence_count > sentence_count_range["max"]:
            weight *= self.content_weights["excessive_sentence_weight"]
        
        # Sentence length optimization
        avg_sentence_length = content_metrics['avg_sentence_length']
        sentence_length_range = self.content_weights["optimal_sentence_length_range"]
        if sentence_length_range["min"] <= avg_sentence_length <= sentence_length_range["max"]:
            weight *= sentence_length_range["weight"]
        elif avg_sentence_length > sentence_length_range["max"]:
            weight *= self.content_weights["excessive_sentence_length_weight"]
        
        # Confidence weighting
        confidence_score = metadata.get('confidence_score', 0.5)
        if confidence_score >= self.confidence_weights["high_confidence_threshold"]:
            weight *= self.confidence_weights["high_confidence_weight"]
        elif confidence_score >= self.confidence_weights["moderate_confidence_threshold"]:
            weight *= self.confidence_weights["moderate_confidence_weight"]
        elif confidence_score < self.confidence_weights["low_confidence_threshold"]:
            weight *= self.confidence_weights["low_confidence_weight"]
        
        # Temperament weighting
        temperament_score = metadata.get('temperament_score', 0.5)
        balanced_range = self.temperament_weights["balanced_range"]
        if balanced_range["min"] <= temperament_score <= balanced_range["max"]:
            weight *= balanced_range["weight"]
        else:
            weight *= self.temperament_weights["extreme_temperament_weight"]
        
        # Context weighting
        context_metrics = metadata.get('context_metrics', {})
        if context_metrics.get('is_first_message', False) or context_metrics.get('is_last_message', False):
            weight *= self.context_weights["first_last_message_weight"]
        
        # Response timing
        response_time = metadata.get('response_time')
        if response_time is not None:
            normalized_time = min(1.0, max(0.0, response_time / self.timing_weights["max_response_time"]))
            weight *= self.timing_weights["optimal_timing_weight"] + (0.5 - abs(normalized_time - 0.5))
        
        # Apply weight bounds
        return max(self.weight_bounds["min"], min(self.weight_bounds["max"], weight))
    
    def prepare_training_pairs(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Prepare training pairs with metadata from conversation history."""
        training_pairs = []
        context_window = deque(maxlen=5)
        
        for i in range(len(conversation_history) - 1):
            current_msg = conversation_history[i]
            next_msg = conversation_history[i + 1]
            
            if current_msg['role'] == 'user' and next_msg['role'] == 'assistant':
                # Collect metadata for both messages
                input_metadata = self.collect_metadata(current_msg['content'])
                output_metadata = self.collect_metadata(next_msg['content'])
                
                # Calculate response time
                response_time = None
                if current_msg.get('timestamp') and next_msg.get('timestamp'):
                    response_time = next_msg['timestamp'] - current_msg['timestamp']
                
                # Create training pair with metadata
                training_pairs.append({
                    'input': current_msg['content'],
                    'output': next_msg['content'],
                    'metadata': {
                        'input_metrics': input_metadata['content_metrics'],
                        'output_metrics': output_metadata['content_metrics'],
                        'quality_metrics': {
                            'input': input_metadata['quality_metrics'],
                            'output': output_metadata['quality_metrics']
                        },
                        'confidence_score': next_msg.get('confidence_score', 0.5),
                        'temperament_score': next_msg.get('temperament_score', 0.5),
                        'context': [m['content'] for m in context_window],
                        'context_metrics': {
                            'is_first_message': i == 0,
                            'is_last_message': i == len(conversation_history) - 2
                        },
                        'response_time': response_time
                    }
                })
                
                context_window.append(current_msg)
        
        return training_pairs


