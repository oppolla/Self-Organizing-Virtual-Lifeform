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

    # transcript_logger.log_interaction(generation_input, generation_output, current_metadata)

    # ... later, on shutdown ...
    # transcript_logger.close()


    


