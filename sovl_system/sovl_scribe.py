import json
import sys
import logging
import os
import threading
import queue
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sovl_config import ConfigManager
from sovl_processor import MetadataProcessor, ScribeIngestionProcessor
from sovl_error import ErrorManager
from sovl_logger import Logger
from sovl_queue import ScribeEntry
from sovl_io import JsonlWriter
import time

class StateAccessorInterface:
    """Interface for state accessor objects to ensure proper dependency handling."""
    def get_state(self):
        """Get the current state"""
        raise NotImplementedError("StateAccessor must implement get_state method")

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
        scribe_queue: queue.Queue = None,
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
        self.scribe_queue = scribe_queue
        self._state_accessor = None
        
        # Setup fallback logger first (for initialization errors)
        self.fallback_logger = self._setup_fallback_logger()
        
        # Placeholders for delayed initialization
        self.jsonl_writer = None
        self._writer_thread = None
        self._stop_event = threading.Event()

        self.scribe_ingestion_processor = ScribeIngestionProcessor(log_paths=[])

        # Configurable batch/flush/queue size
        self.scribe_batch_size = self.config_manager.get("scribed_config.scribe_batch_size", 20)
        self.scribe_flush_interval = self.config_manager.get("scribed_config.scribe_flush_interval", 2.0)
        self.scribe_queue_maxsize = self.config_manager.get("scribed_config.scribe_queue_maxsize", 2000)
        if scribe_queue is None:
            self.scribe_queue = queue.Queue(maxsize=self.scribe_queue_maxsize)
        else:
            self.scribe_queue = scribe_queue

        # For batch/async writes
        self._batch_buffer = []
        self._last_flush_time = time.time()
        self._batch_lock = threading.Lock()

        try:
            # Determine scribe output path from config or explicit override
            self.scribe_path = getattr(self, 'scribe_path', None) or config_manager.get("scribed_config.output_path", "scribe/sovl_scribe.jsonl")
            os.makedirs(os.path.dirname(self.scribe_path), exist_ok=True)
            
            # --- Ensure scribe journal file is created on init ---
            try:
                with open(self.scribe_path, 'a', encoding='utf-8'):
                    pass  # This will create the file if it doesn't exist
            except Exception as e:
                self.fallback_logger.error(f"Failed to create scribe journal file on init: {str(e)}")
            
            # Set the state accessor if provided during initialization
            if state_accessor is not None:
                self.set_state_accessor(state_accessor)
                
            # Log initialization
            self.logger.info(f"Scriber initialized with output path: {self.scribe_path}")

        except Exception as e:
            self.fallback_logger.exception(
                f"Failed to initialize Scriber: {str(e)}"
            )
            raise
            
    def _initialize_jsonl_writer(self) -> None:
        """Initialize the JSONL writer if not already initialized."""
        if self.jsonl_writer is None:
            self.jsonl_writer = JsonlWriter(
                self.scribe_path,
                config_manager=self.config_manager,
                error_manager=self.error_manager,
                logger=self.logger
            )
            self.logger.info(f"JSONL writer initialized for: {self.scribe_path}")
    
    def _initialize_writer_thread(self) -> None:
        """Initialize and start the writer thread if not already running."""
        if self._writer_thread is None or not self._writer_thread.is_alive():
            # Initialize the writer first
            self._initialize_jsonl_writer()
            
            # Start the thread
            self._writer_thread = threading.Thread(
                target=self._process_scribe_queue,
                name="ScriberWriterThread",
                daemon=True
            )
            self._writer_thread.start()
            self.logger.info("Scriber writer thread started")
            
    def set_state_accessor(self, state_accessor: Any) -> None:
        """
        Set or update the state accessor.
        
        Args:
            state_accessor: Object that provides access to system state
        """
        if state_accessor is not None:
            # Validate minimal state accessor interface
            if not hasattr(state_accessor, 'state_manager'):
                self.logger.warning("state_accessor provided does not have state_manager attribute")
            
        self._state_accessor = state_accessor
        self.logger.debug("State accessor set/updated in Scriber")
        
    def get_state(self):
        """Get the current state via state_accessor if available."""
        if self._state_accessor is not None:
            if hasattr(self._state_accessor, 'state_manager'):
                state_manager = getattr(self._state_accessor, 'state_manager')
                if state_manager and hasattr(state_manager, 'get_state'):
                    return state_manager.get_state()
        return None

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
        
    def scribe(self, origin: str, event_type: str, event_data: Dict[str, Any], 
               source_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Scribe an event to the queue and ensure the writer thread is running.
        
        Args:
            origin: Source of the event
            event_type: Type of event
            event_data: Event data dictionary
            source_metadata: Optional metadata
            
        Returns:
            bool: Whether scribing was successful
        """
        # Ensure writer thread is running
        self._ensure_writer_running()
        
        # Get session ID from state if available
        session_id = None
        if self._state_accessor and hasattr(self._state_accessor, 'session_id'):
            session_id = self._state_accessor.session_id
            
        # Create the scribe entry
        entry = ScribeEntry(
            origin=origin,
            event_type=event_type,
            event_data=event_data or {},
            source_metadata=source_metadata or {},
            session_id=session_id,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Add to queue
        try:
            self.scribe_queue.put(entry, timeout=0.5)
            return True
        except queue.Full:
            self.logger.warning(f"Scribe queue full, event type {event_type} from {origin} dropped")
            return False
            
    def _ensure_writer_running(self) -> None:
        """Ensure the writer thread is running."""
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._initialize_writer_thread()

    def _process_scribe_queue(self) -> None:
        """Worker thread function to process scribe queue and write to file."""
        self.logger.info("Scriber writer thread started.")
        
        # Ensure the writer is initialized
        self._initialize_jsonl_writer()
        
        metadata_cache = {}
        while not self._stop_event.is_set():
            try:
                scribe_entry_obj = self.scribe_queue.get(timeout=1)
                
                try:
                    # Extract data from ScribeEntry object using attribute access
                    origin = scribe_entry_obj.origin
                    event_type = scribe_entry_obj.event_type
                    event_data = scribe_entry_obj.event_data or {}
                    source_metadata = scribe_entry_obj.source_metadata or {}
                    session_id = scribe_entry_obj.session_id
                    timestamp = scribe_entry_obj.timestamp

                    cache_key = (origin, event_type, session_id)
                    if cache_key in metadata_cache:
                        validated_event_data, final_metadata = metadata_cache[cache_key]
                    else:
                        validated_event_data, final_metadata = self.metadata_processor.enrich_and_validate(
                            origin=origin,
                            event_type=event_type,
                            event_data=event_data,
                            source_metadata=source_metadata,
                            session_id=session_id,
                        )
                        metadata_cache[cache_key] = (validated_event_data, final_metadata)

                    # Prepare event for ingestion processor
                    event_for_ingestion = {
                        "event_type": event_type,
                        "event_data": validated_event_data,
                        "metadata": final_metadata
                    }
                    processed = self.scribe_ingestion_processor.process_entry(event_for_ingestion)

                    # Structure the final scribe entry for the log
                    scribe_entry = {
                        "memory": processed["memory"],
                        "weight": processed["weight"]
                        # Optionally: "metadata": processed["metadata"]
                    }

                    # Serialize to JSON
                    formatted_scribe_string = json.dumps(scribe_entry, default=str)

                    # Write to JSONL file
                    with self._batch_lock:
                        self._batch_buffer.append(formatted_scribe_string)
                        now = time.time()
                        if len(self._batch_buffer) >= self.scribe_batch_size or (now - self._last_flush_time) >= self.scribe_flush_interval:
                            with self.jsonl_writer.lock:
                                self.jsonl_writer._buffer.extend(self._batch_buffer)
                                self._batch_buffer = []
                                self.jsonl_writer._flush_buffer()
                            self._last_flush_time = now

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
                # On timeout, check if flush interval elapsed
                with self._batch_lock:
                    now = time.time()
                    if self._batch_buffer and (now - self._last_flush_time) >= self.scribe_flush_interval:
                        with self.jsonl_writer.lock:
                            self.jsonl_writer._buffer.extend(self._batch_buffer)
                            self._batch_buffer = []
                            self.jsonl_writer._flush_buffer()
                        self._last_flush_time = now
                continue

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

                    cache_key = (origin, event_type, session_id)
                    if cache_key in metadata_cache:
                        validated_event_data, final_metadata = metadata_cache[cache_key]
                    else:
                        validated_event_data, final_metadata = self.metadata_processor.enrich_and_validate(
                            origin=origin,
                            event_type=event_type,
                            event_data=event_data,
                            source_metadata=source_metadata,
                            session_id=session_id,
                        )
                        metadata_cache[cache_key] = (validated_event_data, final_metadata)

                    # Prepare event for ingestion processor
                    event_for_ingestion = {
                        "event_type": event_type,
                        "event_data": validated_event_data,
                        "metadata": final_metadata
                    }
                    processed = self.scribe_ingestion_processor.process_entry(event_for_ingestion)

                    # Structure the final scribe entry for the log
                    scribe_entry = {
                        "memory": processed["memory"],
                        "weight": processed["weight"]
                        # Optionally: "metadata": processed["metadata"]
                    }

                    # Serialize to JSON
                    formatted_scribe_string = json.dumps(scribe_entry, default=str)

                    # Write to JSONL file
                    with self._batch_lock:
                        self._batch_buffer.append(formatted_scribe_string)
                        now = time.time()
                        if len(self._batch_buffer) >= self.scribe_batch_size or (now - self._last_flush_time) >= self.scribe_flush_interval:
                            with self.jsonl_writer.lock:
                                self.jsonl_writer._buffer.extend(self._batch_buffer)
                                self._batch_buffer = []
                                self.jsonl_writer._flush_buffer()
                            self._last_flush_time = now

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

        # Flush any remaining batch buffer
        with self._batch_lock:
            if self._batch_buffer:
                with self.jsonl_writer.lock:
                    self.jsonl_writer._buffer.extend(self._batch_buffer)
                    self._batch_buffer = []
                    self.jsonl_writer._flush_buffer()

        self.logger.info("Scriber writer thread finished.")

    def shutdown(self) -> None:
        """Signals writer thread to stop, waits, and closes resources."""
        self.logger.info("Initiating Scriber shutdown...")

        # Signal the writer thread to stop
        self._stop_event.set()

        # Forced flush and retry before join
        for attempt in range(3):
            try:
                with self.jsonl_writer.lock:
                    self.jsonl_writer._flush_buffer()
                break
            except Exception as e:
                self.fallback_logger.error(f"Flush retry {attempt+1} failed: {e}")
                time.sleep(1)

        # Wait for the writer thread to finish
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=10)
            if self._writer_thread.is_alive():
                self.fallback_logger.error("Writer thread did not exit gracefully. Forcing shutdown and logging remaining queue items to fallback.")
                # Log remaining queue items to a shutdown fallback file
                shutdown_fallback = "scribe_shutdown_fallback.jsonl"
                while not self.scribe_queue.empty():
                    try:
                        entry = self.scribe_queue.get_nowait()
                        with open(shutdown_fallback, "a", encoding="utf-8") as f:
                            json.dump(entry.__dict__, f, default=str)
                            f.write("\n")
                        self.scribe_queue.task_done()
                    except queue.Empty:
                        break
                    except Exception as e:
                        self.fallback_logger.error(f"Failed to write shutdown fallback: {e}")

        # Retry closing the JsonlWriter up to 3 times
        for attempt in range(3):
            try:
                self.jsonl_writer.close()
                break
            except Exception as e:
                self.fallback_logger.error(f"Close retry {attempt+1} failed: {e}")
                time.sleep(1)

        self.logger.info("Scriber shutdown complete.")

    def __del__(self):
        """Ensure file is closed when object is destroyed."""
        self.shutdown()
