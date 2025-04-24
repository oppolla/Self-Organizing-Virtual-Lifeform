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
from sovl_queue import ScribeEntry
from sovl_io import JsonlWriter

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
                target=self._process_scribe_queue,
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

    def _process_scribe_queue(self) -> None:
        """Worker thread function to process scribe queue and write to file."""
        self.logger.info("Scriber writer thread started.")
        while not self._stop_event.is_set():
            try:
                # Wait for an item for up to 1 second
                scribe_entry_obj = self.scribe_queue.get(timeout=1)
                
                try:
                    # Extract data from ScribeEntry object using attribute access
                    origin = scribe_entry_obj.origin
                    event_type = scribe_entry_obj.event_type
                    event_data = scribe_entry_obj.event_data or {}
                    source_metadata = scribe_entry_obj.source_metadata or {}
                    session_id = scribe_entry_obj.session_id
                    timestamp = scribe_entry_obj.timestamp

                    # Process and enrich the scribed data
                    validated_event_data, final_metadata = self.metadata_processor.enrich_and_validate(
                        origin=origin,
                        event_type=event_type,
                        event_data=event_data,
                        source_metadata=source_metadata,
                        session_id=session_id,
                    )

                    # Structure the final scribe entry
                    scribe_entry = {
                        "timestamp_iso": timestamp.isoformat() if timestamp else datetime.now(timezone.utc).isoformat(),
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
                        f"Entry details: origin={getattr(scribe_entry_obj, 'origin', 'unknown')}, "
                        f"event_type={getattr(scribe_entry_obj, 'event_type', 'unknown')}"
                    )
                    self.error_manager.handle_error(
                        processing_error,
                        error_type="processing",
                        context={
                            "origin": getattr(scribe_entry_obj, 'origin', 'unknown'),
                            "event_type": getattr(scribe_entry_obj, 'event_type', 'unknown'),
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
                scribe_entry_obj = self.scribe_queue.get_nowait()
                try:
                    # Extract data from ScribeEntry object using attribute access
                    origin = scribe_entry_obj.origin
                    event_type = scribe_entry_obj.event_type
                    event_data = scribe_entry_obj.event_data or {}
                    source_metadata = scribe_entry_obj.source_metadata or {}
                    session_id = scribe_entry_obj.session_id
                    timestamp = scribe_entry_obj.timestamp

                    # Process and enrich the scribed data
                    validated_event_data, final_metadata = self.metadata_processor.enrich_and_validate(
                        origin=origin,
                        event_type=event_type,
                        event_data=event_data,
                        source_metadata=source_metadata,
                        session_id=session_id,
                    )

                    # Structure the final scribe entry
                    scribe_entry = {
                        "timestamp_iso": timestamp.isoformat() if timestamp else datetime.now(timezone.utc).isoformat(),
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
                        f"Failed to process scribe entry during shutdown. Error: {processing_error}. "
                        f"Entry details: origin={getattr(scribe_entry_obj, 'origin', 'unknown')}, "
                        f"event_type={getattr(scribe_entry_obj, 'event_type', 'unknown')}"
                    )
                    self.error_manager.handle_error(
                        processing_error,
                        error_type="processing",
                        context={
                            "origin": getattr(scribe_entry_obj, 'origin', 'unknown'),
                            "event_type": getattr(scribe_entry_obj, 'event_type', 'unknown'),
                            "error_type": "scribe_processing_error",
                            "shutdown": True
                        }
                    )
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
