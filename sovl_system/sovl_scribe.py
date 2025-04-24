import json
import sys
import logging
import os
import threading
import queue
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sovl_config import ConfigManager
from sovl_processor import MetadataProcessor
from sovl_error import ErrorManager
from sovl_logger import Logger

class JsonlWriter:
    """
    Handles writing structured data to a rotating JSONL file.
    Manages file I/O, buffering, and rotation for the scribed output.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        error_manager: ErrorManager,
        logger: Logger
    ):
        """
        Initialize the JSONL writer.

        Args:
            config_manager: Configuration manager instance
            error_manager: Error manager instance for error handling
            logger: Logger instance for operational logging
        """
        self.config_manager = config_manager
        self.error_manager = error_manager
        self.logger = logger
        self.lock = threading.Lock()
        
        # Load configuration
        self._load_config()
        
        # Setup fallback logger
        self.fallback_logger = self._setup_fallback_logger()
        
        # Initialize file handling
        self._buffer: List[str] = []
        self._file_handle = None
        self._setup_scribe_file()

    def _load_config(self) -> None:
        """Load and validate configuration settings."""
        try:
            # Get file path
            self.scribe_file_path = self.config_manager.get(
                "scribed_config.log_path",
                "logs/sovl_scribed.jsonl"
            )
            
            # Get max file size (convert MB to bytes)
            max_mb = self.config_manager.get(
                "scribed_config.max_file_size_mb",
                50
            )
            self.max_file_size_bytes = max_mb * 1024 * 1024
            
            # Get buffer size
            self.buffer_size = max(1, self.config_manager.get(
                "scribed_config.buffer_size",
                10
            ))
            
        except Exception as e:
            self.error_manager.handle_error(
                e,
                error_type="config",
                context={
                    "config_section": "scribed_config",
                    "error_type": "config_loading_error"
                }
            )
            raise

    def _setup_fallback_logger(self) -> logging.Logger:
        """Sets up an independent logger for critical I/O errors."""
        logger = logging.getLogger('sovl.scribe.jsonl_writer')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
        return logger

    def _setup_scribe_file(self) -> None:
        """Sets up the scribe file for writing."""
        try:
            # Ensure we have a valid path
            if not self.scribe_file_path:
                self.scribe_file_path = "logs/sovl_scribed.jsonl"
            
            # Create directory if it doesn't exist
            scribe_dir = os.path.dirname(self.scribe_file_path)
            if scribe_dir:  # Only create if there's a directory component
                os.makedirs(scribe_dir, exist_ok=True)
            
            self._open_file()
        except OSError as e:
            self.fallback_logger.exception(
                f"Failed to create directory or open scribe file: {self.scribe_file_path}"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "file_setup_error"
                }
            )
            raise

    def _open_file(self) -> None:
        """Opens the scribe file in append mode."""
        try:
            if self._file_handle and not self._file_handle.closed:
                self._file_handle.close()
            self._file_handle = open(self.scribe_file_path, 'a', encoding='utf-8')
        except IOError as e:
            self.fallback_logger.exception(
                f"Failed to open scribe file: {self.scribe_file_path}"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "file_open_error"
                }
            )
            self._file_handle = None
            raise

    def write(self, json_string: str) -> bool:
        """
        Writes a JSON string to the scribe file.

        Args:
            json_string: The JSON string to write

        Returns:
            bool: True if writing was successful, False otherwise
        """
        try:
            with self.lock:
                # Add to buffer
                self._buffer.append(json_string)

                # Flush if buffer is full
                if len(self._buffer) >= self.buffer_size:
                    self._flush_buffer()

            return True
        except Exception as write_error:
            self.fallback_logger.exception(
                f"Failed to buffer scribe entry. Error: {write_error}. "
                f"Entry prefix: {json_string[:500]}..."
            )
            self.error_manager.handle_error(
                write_error,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "write_error"
                }
            )
            return False

    def _flush_buffer(self) -> None:
        """Writes all buffered entries to the file."""
        if not self._buffer:
            return

        if not self._file_handle or self._file_handle.closed:
            self.fallback_logger.warning(
                "Attempted to flush buffer, but file handle is not open."
            )
            return

        try:
            for entry in self._buffer:
                self._file_handle.write(entry + '\n')
            self._file_handle.flush()
            self._buffer = []

            # Check file size for rotation
            if self._file_handle.tell() > self.max_file_size_bytes:
                self._rotate_scribe()

        except Exception as e:
            self.fallback_logger.exception(
                "Failed during buffer flush"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "flush_error"
                }
            )

    def _rotate_scribe(self) -> None:
        """Rotates the scribe file when it reaches the size limit."""
        if not self._file_handle:
            return

        try:
            # Close current file
            self._file_handle.close()
            
            # Rename current file with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            old_path = self._file_handle.name
            new_path = f"{old_path}.{timestamp}"
            os.rename(old_path, new_path)
            
            # Reopen file
            self._open_file()
            
        except Exception as e:
            self.fallback_logger.exception(
                f"Failed to rotate scribe file: {self._file_handle.name}"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "rotation_error"
                }
            )

    def close(self) -> None:
        """Flushes buffer and closes the scribe file."""
        try:
            with self.lock:
                self._flush_buffer()
                if self._file_handle and not self._file_handle.closed:
                    self._file_handle.close()
                    self._file_handle = None
        except Exception as e:
            self.fallback_logger.exception(
                "Error closing scribe file"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "file_path": self.scribe_file_path,
                    "error_type": "close_error"
                }
            )

class Scriber:
    """
    Central scribing subsystem for SOVL, managing metadata processing and scribed output.
    
    This class serves as the definitive entry point for all scribing within the SOVL system.
    It orchestrates the process of enriching event data with metadata, formatting,
    and dispatching scribed entries to the configured output destination.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        error_manager: ErrorManager,
        metadata_processor: MetadataProcessor,
        logger: Logger,
        scribe_queue: queue.Queue,
        state_accessor: Optional[Any] = None
    ):
        """
        Initialize the Scriber.

        Args:
            config_manager: Configuration manager instance for scribing settings
            error_manager: Error manager instance for error handling
            metadata_processor: Metadata processor instance for enriching scribed data
            logger: Logger instance for operational logging
            scribe_queue: Queue instance for receiving scribed events
            state_accessor: Optional accessor for global system state
        """
        if not isinstance(config_manager, ConfigManager):
            raise TypeError("config_manager must be an instance of ConfigManager")
        if not isinstance(error_manager, ErrorManager):
            raise TypeError("error_manager must be an instance of ErrorManager")
        if not isinstance(metadata_processor, MetadataProcessor):
            raise TypeError("metadata_processor must be an instance of MetadataProcessor")
        if not isinstance(logger, Logger):
            raise TypeError("logger must be an instance of Logger")
        if not isinstance(scribe_queue, queue.Queue):
            raise TypeError("scribe_queue must be an instance of queue.Queue")

        self.config_manager = config_manager
        self.error_manager = error_manager
        self.metadata_processor = metadata_processor
        self.logger = logger
        self.state_accessor = state_accessor
        self.scribe_queue = scribe_queue

        # Setup fallback logger first (for initialization errors)
        self.fallback_logger = self._setup_fallback_logger()

        try:
            # Setup JSONL writer
            self.jsonl_writer = JsonlWriter(
                config_manager=self.config_manager,
                error_manager=self.error_manager,
                logger=self.logger
            )
            
            # Setup worker thread
            self._stop_event = threading.Event()
            self._writer_thread = threading.Thread(
                target=self._writer_worker,
                name="ScriberWriterThread",
                daemon=True
            )
            self._writer_thread.start()
            
            self.logger.info("Scriber initialized successfully with writer thread")

        except Exception as e:
            self.fallback_logger.exception(
                f"Failed to initialize Scriber: {str(e)}"
            )
            raise

    def _setup_fallback_logger(self) -> logging.Logger:
        """Sets up an independent logger for critical Scriber errors."""
        logger = logging.getLogger('sovl.scribe.fallback')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
        return logger

    def _writer_worker(self) -> None:
        """Worker thread function to process scribe queue and write to file."""
        self.logger.info("Scriber writer thread started.")
        while not self._stop_event.is_set():
            try:
                # Wait for an item for up to 1 second
                scribe_details = self.scribe_queue.get(timeout=1)
                
                try:
                    # Extract scribe details
                    origin = scribe_details.get('origin', 'unknown')
                    event_type = scribe_details.get('event_type', 'unknown')
                    event_data = scribe_details.get('event_data', {})
                    source_metadata = scribe_details.get('source_metadata', {})
                    session_id = scribe_details.get('session_id')
                    interaction_id = scribe_details.get('interaction_id')

                    # Process and enrich the scribed data
                    validated_event_data, final_metadata = self.metadata_processor.enrich_and_validate(
                        origin=origin,
                        event_type=event_type,
                        event_data=event_data,
                        source_metadata=source_metadata,
                        session_id=session_id,
                        interaction_id=interaction_id
                    )

                    # Structure the final scribe entry
                    scribe_entry = {
                        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                        "event_type": event_type,
                        "event_data": validated_event_data,
                        "metadata": final_metadata
                    }

                    # Serialize to JSON
                    formatted_scribe_string = json.dumps(scribe_entry, default=str)

                    # Write to JSONL file
                    with self.jsonl_writer.lock:
                        self.jsonl_writer._buffer.append(formatted_scribe_string)
                        if len(self.jsonl_writer._buffer) >= self.jsonl_writer.buffer_size:
                            try:
                                self.jsonl_writer._flush_buffer()
                            except Exception as flush_err:
                                self.fallback_logger.exception(
                                    f"Writer thread failed during flush: {flush_err}"
                                )

                except Exception as processing_error:
                    self.fallback_logger.exception(
                        f"Failed to process scribe entry. Error: {processing_error}. "
                        f"Entry prefix: {str(scribe_details)[:500]}..."
                    )
                    self.error_manager.handle_error(
                        processing_error,
                        error_type="processing",
                        context={
                            "origin": scribe_details.get('origin', 'unknown'),
                            "event_type": scribe_details.get('event_type', 'unknown'),
                            "error_type": "scribe_processing_error"
                        }
                    )
                finally:
                    # Mark task as done for queue.join()
                    self.scribe_queue.task_done()

            except queue.Empty:
                # Queue was empty, loop again to check stop event
                continue
            except Exception as e:
                self.fallback_logger.exception(
                    f"Unexpected error in writer thread: {e}"
                )
                # Mark task done even if error occurred to prevent blocking shutdown
                self.scribe_queue.task_done()

        # --- Shutdown sequence ---
        self.logger.info("Writer thread received stop signal. Processing remaining scribes...")
        # Process any remaining items after stop signal received
        try:
            while True: # Process all remaining items without blocking
                scribe_details = self.scribe_queue.get_nowait()
                try:
                    # ... (same processing logic as above) ...
                    pass
                finally:
                    self.scribe_queue.task_done()
        except queue.Empty:
            pass # Queue is empty
        except Exception as e:
            self.fallback_logger.exception(
                f"Error processing remaining scribes during shutdown: {e}"
            )

        # Final flush before exiting
        try:
            with self.jsonl_writer.lock:
                self.jsonl_writer._flush_buffer()
        except Exception as final_flush_err:
            self.fallback_logger.exception(
                f"Writer thread failed during final flush: {final_flush_err}"
            )

        self.logger.info("Scriber writer thread finished.")

    def shutdown(self) -> None:
        """Signals writer thread to stop, waits, and closes resources."""
        self.logger.info("Initiating Scriber shutdown...")

        # Signal the writer thread to stop
        self._stop_event.set()

        # Wait for the writer thread to finish
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=10) # Add timeout
            if self._writer_thread.is_alive():
                self.fallback_logger.error("Writer thread did not exit gracefully.")

        # Now close the JsonlWriter
        try:
            self.jsonl_writer.close()
        except Exception as e:
            self.fallback_logger.exception(
                "Error closing scribe file"
            )
            self.error_manager.handle_error(
                e,
                error_type="io",
                context={
                    "error_type": "close_error"
                }
            )

        self.logger.info("Scriber shutdown complete.")

    def __del__(self):
        """Ensure file is closed when object is destroyed."""
        self.shutdown()
