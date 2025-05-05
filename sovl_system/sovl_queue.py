import queue
import sys
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from sovl_logger import Logger
from sovl_error import ErrorManager
import json
import threading
import os
import time

"""
Centralized queue system for SOVL component communication.
Prevents circular dependencies by providing shared queues for producers and consumers.
"""

# Constants for queue management
MAX_QUEUE_SIZE = 2000  # Maximum number of entries in queue
WARNING_THRESHOLD = 0.8  # Warn when queue is 80% full
FALLBACK_PATH = "scribe_fallback.jsonl"
CRITICAL_EVENT_TYPES = {"checkpoint", "training_complete"}
FALLBACK_MAX_SIZE_MB = 10
FALLBACK_ROTATION_COUNT = 3

# Singleton instance
_global_scribe_queue = None
_global_queue_lock = threading.Lock()

@dataclass
class ScribeEntry:
    """Standardized structure for entries going into the scribe queue."""
    origin: str
    event_type: str
    event_data: Dict[str, Any]
    source_metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    timestamp: datetime = datetime.now()

class ScribeQueue:
    """Thread-safe queue implementation with configurable logger dependency"""
    
    def __init__(self, logger: Optional[Logger] = None, maxsize: int = MAX_QUEUE_SIZE):
        """
        Initialize a scribe queue with optional logger instance.
        
        Args:
            logger: Logger instance for operational logging
            maxsize: Maximum queue size
        """
        self._queue = queue.Queue(maxsize=maxsize)
        self._logger = logger
        self._lock = threading.Lock()
        self._fallback_lock = threading.Lock()
        
    def get_queue(self) -> queue.Queue:
        """Get the internal queue instance."""
        return self._queue
        
    def set_logger(self, logger: Logger) -> None:
        """Set or update the logger instance."""
        if not isinstance(logger, Logger):
            raise TypeError("logger must be an instance of Logger")
        self._logger = logger
        
    def capture_event(
        self,
        origin: str,
        event_type: str,
        event_data: Dict[str, Any],
        source_metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Creates a ScribeEntry and safely puts it onto the scribe queue.
        
        Args:
            origin: The source module/component name.
            event_type: A string identifying the type of event.
            event_data: Dictionary containing the core data for the event.
            source_metadata: Optional dictionary with contextual metadata.
            session_id: Optional session identifier.
            timestamp: Optional specific timestamp; defaults to now().
            
        Returns:
            bool: True if the event was successfully queued or written to fallback, False otherwise.
        """
        try:
            entry = ScribeEntry(
                origin=origin,
                event_type=event_type,
                event_data=event_data or {},
                source_metadata=source_metadata or {},
                session_id=session_id,
                timestamp=timestamp or datetime.now()
            )
            
            # Block for critical events, else use timeout
            try:
                if event_type in CRITICAL_EVENT_TYPES:
                    self._queue.put(entry, block=True)
                    if self._logger:
                        self._logger.debug(f"Successfully queued CRITICAL entry from {origin} with event type {event_type}")
                    return True
                else:
                    self._queue.put(entry, timeout=0.5)
                    if self._logger:
                        self._logger.debug(f"Successfully queued entry from {origin} with event type {event_type}")
                    return True
            except queue.Full:
                if self._logger:
                    self._logger.warning(f"Scribe queue full, writing to fallback for {origin} ({event_type})")
                try:
                    # Fallback file size/rotation logic
                    if os.path.exists(FALLBACK_PATH):
                        size_mb = os.path.getsize(FALLBACK_PATH) / (1024 * 1024)
                        if size_mb > FALLBACK_MAX_SIZE_MB:
                            # Rotate file
                            timestamp = time.strftime('%Y%m%d_%H%M%S')
                            rotated = f"{FALLBACK_PATH}.{timestamp}"
                            os.rename(FALLBACK_PATH, rotated)
                            if self._logger:
                                self._logger.warning(f"Rotated fallback file after exceeding {FALLBACK_MAX_SIZE_MB}MB.")
                            # Prune old rotated files
                            rotated_files = sorted([
                                f for f in os.listdir('.') if f.startswith('scribe_fallback.jsonl.')
                            ])
                            while len(rotated_files) > FALLBACK_ROTATION_COUNT:
                                os.remove(rotated_files.pop(0))
                    with self._fallback_lock:
                        with open(FALLBACK_PATH, "a", encoding="utf-8") as f:
                            json.dump(entry.__dict__, f, default=str)
                            f.write("\n")
                    return True
                except Exception as fallback_err:
                    if self._logger:
                        self._logger.error(f"Failed to write scribe event to fallback: {fallback_err}")
                    return False
        except Exception as e:
            if self._logger:
                self._logger.error(f"Unexpected error queuing scribe event from {origin} ({event_type}): {e}", exc_info=True)
            return False
            
    def clear(self, caller: str, confirm: bool = False) -> None:
        """
        Clear all items from the scribe queue.
        Use with caution - only in emergency situations or during shutdown.
        Requires explicit confirmation and caller name.
        """
        if not confirm:
            raise ValueError("Queue clearing requires explicit confirmation (confirm=True)")
        
        if self._logger:
            self._logger.warning(f"Clearing scribe queue by {caller} - this should only be done in emergency situations")
        
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
                
        if self._logger:
            self._logger.info("Scribe queue cleared successfully")
            
    def get_size(self) -> int:
        """
        Get the current number of items in the scribe queue.
        
        Returns:
            int: Number of items currently in the queue.
        """
        return self._queue.qsize()
        
    def check_health(self) -> Tuple[str, float]:
        """
        Check the health status of the scribe queue.
        
        Returns:
            Tuple[str, float]: Status string ("OK", "WARNING", or "FULL") and queue fill ratio
        """
        current_size = self.get_size()
        max_size = self._queue.maxsize
        fill_ratio = current_size / max_size
        
        if fill_ratio >= 1.0:
            error_msg = f"Scribe queue is full! Current size: {current_size}/{max_size}"
            if self._logger:
                self._logger.error(error_msg)
            return "FULL", fill_ratio
        elif fill_ratio >= WARNING_THRESHOLD:
            error_msg = f"Scribe queue is approaching capacity: {current_size}/{max_size} ({fill_ratio:.1%})"
            if self._logger:
                self._logger.warning(error_msg)
            return "WARNING", fill_ratio
        
        if self._logger:
            self._logger.debug(f"Scribe queue health check: {current_size}/{max_size} ({fill_ratio:.1%})")
        return "OK", fill_ratio

# Factory functions for backward compatibility
def get_scribe_queue(logger: Optional[Logger] = None, maxsize: Optional[int] = None) -> ScribeQueue:
    """
    Get the singleton scribe queue instance, initializing it if necessary.
    Optionally pass a logger on first initialization.
    
    Args:
        logger: Optional Logger instance for queue logging
        maxsize: Optional maximum queue size
        
    Returns:
        ScribeQueue: The singleton scribe queue instance
    """
    global _global_scribe_queue
    with _global_queue_lock:
        if _global_scribe_queue is None:
            qsize = maxsize if maxsize is not None else MAX_QUEUE_SIZE
            _global_scribe_queue = ScribeQueue(logger=logger, maxsize=qsize)
        elif logger is not None and _global_scribe_queue._logger is None:
            # Update logger if it wasn't set previously
            _global_scribe_queue.set_logger(logger)
        return _global_scribe_queue

def capture_scribe_event(
    origin: str,
    event_type: str,
    event_data: Dict[str, Any],
    source_metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> bool:
    """
    Creates a ScribeEntry and safely puts it onto the global scribe queue.
    This is a compatibility function for existing code.
    
    Args:
        origin: The source module/component name.
        event_type: A string identifying the type of event.
        event_data: Dictionary containing the core data for the event.
        source_metadata: Optional dictionary with contextual metadata.
        session_id: Optional session identifier.
        timestamp: Optional specific timestamp; defaults to now().
        
    Returns:
        bool: True if the event was successfully queued or written to fallback, False otherwise.
    """
    queue_instance = get_scribe_queue()
    return queue_instance.capture_event(
        origin=origin,
        event_type=event_type,
        event_data=event_data,
        source_metadata=source_metadata,
        session_id=session_id,
        timestamp=timestamp
    )

def clear_scribe_queue(caller: str, confirm: bool = False) -> None:
    """
    Clear all items from the scribe queue.
    Use with caution - only in emergency situations or during shutdown.
    Requires explicit confirmation and caller name.
    
    This is a compatibility function for existing code.
    """
    queue_instance = get_scribe_queue()
    queue_instance.clear(caller=caller, confirm=confirm)

def get_scribe_queue_size() -> int:
    """
    Get the current number of items in the scribe queue.
    
    This is a compatibility function for existing code.
    
    Returns:
        int: Number of items currently in the queue.
    """
    queue_instance = get_scribe_queue()
    return queue_instance.get_size()

def check_scribe_queue_health() -> Tuple[str, float]:
    """
    Check the health status of the scribe queue.
    
    This is a compatibility function for existing code.
    
    Returns:
        Tuple[str, float]: Status string ("OK", "WARNING", or "FULL") and queue fill ratio
    """
    queue_instance = get_scribe_queue()
    return queue_instance.check_health()
