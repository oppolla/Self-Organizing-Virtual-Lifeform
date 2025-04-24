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
from sovl_error import ErrorManager, ErrorRecord, ConfigurationError
from sovl_generation import GenerationManager
import gc

class MemoriaManager:
    """Manages the core remembering system for SOVL."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize MemoriaManager with configuration and logger."""
        self._config_manager = config_manager
        self._logger = logger
        self._memory_lock = Lock()
        self._state = None
        self._conversation_history = None
        
        # Initialize error manager
        self._error_manager = ErrorManager(
            context=None,  # Will be set by system
            state_tracker=None,  # Will be set by system
            config_manager=config_manager
        )
        
        # Set up error thresholds
        self._error_manager.set_error_threshold("storage_error", 3)
        self._error_manager.set_error_threshold("save_error", 3)
        self._error_manager.set_error_threshold("load_error", 3)
        
        # Register recovery strategies
        self._error_manager.register_recovery_strategy("storage_error", self._recover_storage)
        self._error_manager.register_recovery_strategy("save_error", self._recover_save)
        self._error_manager.register_recovery_strategy("load_error", self._recover_load)
        
        # Initialize storage
        self._initialize_storage()
        
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
                error_record = ErrorRecord(
                    error_type="storage_error",
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    severity=2,
                    context={
                        "operation": "storage_initialization",
                        "state": self._state.get_state() if self._state else None
                    }
                )
                self._error_manager.record_error(error_record)
                raise

    def _recover_storage(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for storage initialization errors."""
        try:
            # Clear existing state
            self._state = None
            self._conversation_history = None
            
            # Force garbage collection
            gc.collect()
            
            # Retry initialization
            self._initialize_storage()
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to recover from storage error: {str(e)}",
                error_type="storage_recovery_error",
                stack_trace=traceback.format_exc()
            )

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
            error_record = ErrorRecord(
                error_type="save_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=2,
                context={
                    "operation": "state_save",
                    "path_prefix": path_prefix,
                    "state": self._state.get_state() if self._state else None
                }
            )
            self._error_manager.record_error(error_record)
            raise

    def _recover_save(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for save errors."""
        try:
            # Create backup directory
            backup_dir = os.path.join(os.path.dirname(error_record.context["path_prefix"]), "backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Save to backup location
            backup_path = os.path.join(backup_dir, f"memoria_backup_{int(time.time())}.json")
            self.save_state(backup_path)
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to recover from save error: {str(e)}",
                error_type="save_recovery_error",
                stack_trace=traceback.format_exc()
            )

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
            error_record = ErrorRecord(
                error_type="load_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=2,
                context={
                    "operation": "state_load",
                    "path_prefix": path_prefix
                }
            )
            self._error_manager.record_error(error_record)
            raise

    def _recover_load(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for load errors."""
        try:
            # Try loading from backup
            backup_dir = os.path.join(os.path.dirname(error_record.context["path_prefix"]), "backup")
            if os.path.exists(backup_dir):
                backup_files = sorted(os.listdir(backup_dir), reverse=True)
                for backup_file in backup_files:
                    try:
                        backup_path = os.path.join(backup_dir, backup_file)
                        self.load_state(backup_path)
                        return
                    except Exception:
                        continue
            
            # If no backup works, reset to default state
            self._initialize_storage()
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to recover from load error: {str(e)}",
                error_type="load_recovery_error",
                stack_trace=traceback.format_exc()
            )

class MetadataProcessor:
    """Handles metadata collection, analysis, and weighting for training examples."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """Initialize metadata processor with configuration and logger."""
        self.config_manager = config_manager
        self.logger = logger
        
        # Initialize error manager
        self._error_manager = ErrorManager(
            context=None,  # Will be set by system
            state_tracker=None,  # Will be set by system
            config_manager=config_manager
        )
        
        # Set up error thresholds
        self._error_manager.set_error_threshold("config_error", 3)
        self._error_manager.set_error_threshold("weight_error", 3)
        
        # Register recovery strategies
        self._error_manager.register_recovery_strategy("config_error", self._recover_config)
        self._error_manager.register_recovery_strategy("weight_error", self._recover_weight)
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load and validate metadata configuration."""
        try:
            self.config = self.config_manager.get_section("gestation_weighting")
            if not self.config:
                raise ConfigurationError("Missing gestation_weighting configuration")
                
            self.quality_weights = self.config["metadata_fields"]["quality_metrics"]
            self.content_weights = self.config["content_weights"]
            self.confidence_weights = self.config["confidence_weights"]
            self.temperament_weights = self.config["temperament_weights"]
            self.context_weights = self.config["context_weights"]
            self.timing_weights = self.config["timing_weights"]
            self.weight_bounds = self.config["weight_bounds"]
            
        except Exception as e:
            error_record = ErrorRecord(
                error_type="config_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=2,
                context={
                    "operation": "config_load",
                    "section": "gestation_weighting"
                }
            )
            self._error_manager.record_error(error_record)
            raise

    def _recover_config(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for configuration errors."""
        try:
            # Reset to default configuration
            default_config = {
                "metadata_fields": {
                    "quality_metrics": {
                        "code": {"enabled": True, "weight": 1.2},
                        "url": {"enabled": True, "weight": 1.1},
                        "question": {"enabled": True, "weight": 1.1},
                        "exclamation": {"enabled": True, "weight": 1.1},
                        "emoji": {"enabled": True, "weight": 1.1}
                    }
                },
                "content_weights": {
                    "word_count_ratio_scale": 1.0,
                    "optimal_word_length_range": {"min": 3, "max": 8, "weight": 1.2},
                    "suboptimal_word_length_weight": 0.8,
                    "optimal_sentence_count_range": {"min": 1, "max": 5, "weight": 1.2},
                    "excessive_sentence_weight": 0.8,
                    "optimal_sentence_length_range": {"min": 5, "max": 20, "weight": 1.2},
                    "excessive_sentence_length_weight": 0.8
                },
                "confidence_weights": {
                    "high_confidence_threshold": 0.8,
                    "high_confidence_weight": 1.2,
                    "moderate_confidence_threshold": 0.5,
                    "moderate_confidence_weight": 1.0,
                    "low_confidence_threshold": 0.3,
                    "low_confidence_weight": 0.8
                },
                "temperament_weights": {
                    "balanced_range": {"min": 0.4, "max": 0.6, "weight": 1.2},
                    "extreme_temperament_weight": 0.8
                },
                "context_weights": {
                    "first_last_message_weight": 1.2
                },
                "timing_weights": {
                    "max_response_time": 300,
                    "optimal_timing_weight": 1.2
                },
                "weight_bounds": {
                    "min": 0.1,
                    "max": 2.0
                }
            }
            
            self.config = default_config
            self.quality_weights = self.config["metadata_fields"]["quality_metrics"]
            self.content_weights = self.config["content_weights"]
            self.confidence_weights = self.config["confidence_weights"]
            self.temperament_weights = self.config["temperament_weights"]
            self.context_weights = self.config["context_weights"]
            self.timing_weights = self.config["timing_weights"]
            self.weight_bounds = self.config["weight_bounds"]
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to recover from config error: {str(e)}",
                error_type="config_recovery_error",
                stack_trace=traceback.format_exc()
            )

    def calculate_weight(self, metadata: Dict[str, Any]) -> float:
        """Calculate weight for a training example based on its metadata."""
        try:
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
            
            # Apply weight bounds
            weight = max(self.weight_bounds["min"], min(self.weight_bounds["max"], weight))
            
            return weight
            
        except Exception as e:
            error_record = ErrorRecord(
                error_type="weight_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=2,
                context={
                    "operation": "weight_calculation",
                    "metadata": metadata
                }
            )
            self._error_manager.record_error(error_record)
            return self.weight_bounds["min"]  # Return minimum weight on error

    def _recover_weight(self, error_record: ErrorRecord) -> None:
        """Recovery strategy for weight calculation errors."""
        try:
            # Reset to default weight
            return self.weight_bounds["min"]
            
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to recover from weight error: {str(e)}",
                error_type="weight_recovery_error",
                stack_trace=traceback.format_exc()
            )
            return 0.1  # Absolute minimum weight as last resort

    def collect_metadata(self, content: str) -> Dict[str, Any]:
        """Collect metadata from content."""
        try:
            # Split content into words and sentences
            words = content.split()
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            
            # Calculate metrics
            word_count = len(words)
            sentence_count = len(sentences)
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            metadata = {
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
            
            return metadata
            
        except Exception as e:
            error_record = ErrorRecord(
                error_type="metadata_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=2,
                context={
                    "operation": "metadata_collection",
                    "content_length": len(content)
                }
            )
            self._error_manager.record_error(error_record)
            return {
                'content_metrics': {},
                'quality_metrics': {}
            }

    def prepare_training_pairs(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Prepare training pairs with metadata from conversation history."""
        try:
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
            
        except Exception as e:
            error_record = ErrorRecord(
                error_type="training_pair_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                severity=2,
                context={
                    "operation": "prepare_training_pairs",
                    "history_length": len(conversation_history)
                }
            )
            self._error_manager.record_error(error_record)
            return []

    class ChatTranscript:
        """
        Logs chat interactions (inputs and outputs) to a JSONL file,
        enriched with metadata. Designed to capture data originating
        from the generation process.
        """

        def __init__(
            self,
            log_file_path: str,
            logger: Logger,
            # metadata_generator: MetadataGenerator # Optional: Or pass metadata directly
            max_file_size_mb: int = 50, # Example: Configurable file size limit
            write_buffer_size: int = 10 # Example: Write every N entries
        ):
            """
            Initializes the ChatTranscript logger.

            Args:
                log_file_path: The path to the JSONL file for storing transcripts.
                logger: The main system logger for internal logging.
                max_file_size_mb: Maximum size in MB before considering rotation (logic TBD).
                write_buffer_size: Number of entries to buffer before writing to disk.
            """
            self.log_file_path = log_file_path
            self.logger = logger
            # self.metadata_generator = metadata_generator
            self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
            self.write_buffer_size = max(1, write_buffer_size) # Ensure at least 1
            self._buffer: List[Dict[str, Any]] = []
            self._file_handle = None

            # Ensure log directory exists
            try:
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                # Open in append mode, create if doesn't exist
                self._open_file()
            except OSError as e:
                self.logger.log_error(
                    f"Failed to create directory or open chat transcript log: {self.log_file_path}",
                    error_type="file_io_error",
                    stack_trace=str(e)
                )
                # Consider raising or handling more gracefully
                raise

        def _open_file(self):
            """Opens the log file in append mode."""
            try:
                # Close existing handle if necessary (e.g., during rotation)
                if self._file_handle and not self._file_handle.closed:
                    self._file_handle.close()
                self._file_handle = open(self.log_file_path, 'a', encoding='utf-8')
                self.logger.log_event(
                    "chat_transcript_opened",
                    f"Chat transcript log opened: {self.log_file_path}",
                    "info"
                )
            except IOError as e:
                self.logger.log_error(
                    f"Failed to open chat transcript file handle: {self.log_file_path}",
                    error_type="file_io_error",
                    stack_trace=str(e)
                )
                self._file_handle = None # Ensure handle is None on failure

        def log_interaction(
            self,
            generation_input: Dict[str, Any], # Input structure from sovl_generation
            generation_output: Dict[str, Any], # Output structure from sovl_generation
            metadata: Dict[str, Any] # Metadata associated with this interaction
        ) -> None:
            """
            Logs a single input/output interaction from the generation process.

            Args:
                generation_input: The data provided as input to the generation module.
                generation_output: The data produced by the generation module.
                metadata: Additional metadata relevant to this interaction (e.g.,
                          timestamps, confidence scores, system state, model config).
            """
            if not self._file_handle or self._file_handle.closed:
                self.logger.log_warning(
                    "Attempted to log chat interaction, but file handle is not open.",
                    event_type="chat_transcript_write_error"
                )
                # Optionally try to reopen
                # self._open_file()
                # if not self._file_handle: return # Exit if reopen failed
                return

            try:
                entry = self._format_entry(generation_input, generation_output, metadata)
                self._buffer.append(entry)

                if len(self._buffer) >= self.write_buffer_size:
                    self._flush_buffer()

            except Exception as e:
                self.logger.log_error(
                    "Failed to format or buffer chat transcript entry",
                    error_type="transcript_formatting_error",
                    stack_trace=str(e),
                    additional_info={
                        "input_keys": list(generation_input.keys()),
                        "output_keys": list(generation_output.keys()),
                        "metadata_keys": list(metadata.keys()),
                    }
                )

        def _format_entry(
            self,
            gen_input: Dict[str, Any],
            gen_output: Dict[str, Any],
            metadata: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Formats the log entry dictionary."""
            # Basic structure, can be customized extensively
            return {
                "timestamp_unix": time.time(),
                "interaction_id": metadata.get("interaction_id", None), # Example metadata field
                "session_id": metadata.get("session_id", None), # Example metadata field
                "input_data": gen_input,
                "output_data": gen_output,
                "metadata": metadata, # Include all provided metadata
            }

        def _write_entry(self, entry: Dict[str, Any]) -> None:
            """Writes a single formatted entry to the JSONL file."""
            if self._file_handle and not self._file_handle.closed:
                try:
                    json_string = json.dumps(entry, ensure_ascii=False)
                    self._file_handle.write(json_string + '\n')
                except (TypeError, IOError) as e:
                    self.logger.log_error(
                        f"Failed to write entry to chat transcript log: {self.log_file_path}",
                        error_type="file_io_error",
                        stack_trace=str(e),
                        additional_info={"entry_keys": list(entry.keys())}
                    )
                    # Consider closing/reopening the file or other recovery
            else:
                 self.logger.log_warning(
                    "Attempted to write chat entry, but file handle is not open.",
                    event_type="chat_transcript_write_error"
                )

        def _flush_buffer(self) -> None:
            """Writes all buffered entries to the file."""
            if not self._buffer:
                return

            if not self._file_handle or self._file_handle.closed:
                self.logger.log_warning(
                    "Attempted to flush chat transcript buffer, but file handle is not open.",
                    event_type="chat_transcript_flush_error"
                )
                # Optionally try to reopen
                return

            try:
                for entry in self._buffer:
                    self._write_entry(entry)
                self._file_handle.flush() # Ensure data is written to OS
                self._buffer = [] # Clear buffer
                # Check file size for rotation (simplified check)
                # current_size = self._file_handle.tell()
                # if current_size > self.max_file_size_bytes:
                #    self._rotate_log() # Implement rotation logic if needed
            except Exception as e:
                self.logger.log_error(
                    f"Failed during chat transcript buffer flush: {self.log_file_path}",
                    error_type="file_io_error",
                    stack_trace=str(e)
                )
                # Decide how to handle buffer on error (clear? retry?)

        def close(self) -> None:
            """Flushes buffer and closes the log file."""
            self.logger.log_event(
                "chat_transcript_closing",
                f"Closing chat transcript log: {self.log_file_path}",
                "info"
            )
            try:
                self._flush_buffer() # Ensure all data is written
                if self._file_handle and not self._file_handle.closed:
                    self._file_handle.close()
                    self._file_handle = None
            except Exception as e:
                 self.logger.log_error(
                    f"Error closing chat transcript log: {self.log_file_path}",
                    error_type="file_io_error",
                    stack_trace=str(e)
                )

        def __del__(self):
            """Ensure file is closed when object is destroyed."""
            self.close()

    # Example Usage (Conceptual - would happen elsewhere, e.g., in sovl_main or generation coordinator)
    # Assuming 'logger' is an initialized Logger instance
    # Assuming 'get_current_metadata' is a function that gathers relevant metadata

    # transcript_logger = ChatTranscript("logs/chat_transcripts.jsonl", logger)

    # ... inside the generation loop ...
    # generation_input = {"prompt": "Hello there!", "config": {...}}
    # generation_output = {"response": "General Kenobi!", "metrics": {...}}
    # current_metadata = get_current_metadata(interaction_id="xyz123", session_id="abc987")

    # transcript_logger.log_interaction(generation_input, generation_output, current_metadata)

    # ... later, on shutdown ...
    # transcript_logger.close()


    


