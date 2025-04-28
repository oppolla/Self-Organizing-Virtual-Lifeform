import queue
import sys
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from sovl_logger import Logger
from sovl_error import ErrorManager
import json
import threading

"""
Centralized queue system for SOVL component communication.
Prevents circular dependencies by providing shared queues for producers and consumers.
"""

# Initialize logger for this module
logger = Logger(__name__)

# Constants for queue management
MAX_QUEUE_SIZE = 2000  # Maximum number of entries in queue
WARNING_THRESHOLD = 0.8  # Warn when queue is 80% full
FALLBACK_PATH = "scribe_fallback.jsonl"
CRITICAL_EVENT_TYPES = {"checkpoint", "training_complete"}
_fallback_lock = threading.Lock()

# Thread-safe singleton queue
_scribe_queue = None
_scribe_queue_lock = threading.Lock()

@dataclass
class ScribeEntry:
    """Standardized structure for entries going into the scribe queue."""
    origin: str
    event_type: str
    event_data: Dict[str, Any]
    source_metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    timestamp: datetime = datetime.now()

def get_scribe_queue(maxsize: Optional[int] = None) -> queue.Queue:
    """
    Get the singleton scribe queue instance, initializing it if necessary.
    Optionally set maxsize on first initialization.
    """
    global _scribe_queue
    with _scribe_queue_lock:
        if _scribe_queue is None:
            qsize = maxsize if maxsize is not None else MAX_QUEUE_SIZE
            _scribe_queue = queue.Queue(maxsize=qsize)
        return _scribe_queue

def capture_scribe_event(
    origin: str,
    event_type: str,
    event_data: Dict[str, Any],
    source_metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> bool:
    """
    Creates a ScribeEntry and safely puts it onto the scribe queue.
    This is the primary function modules should use to log events.
    It encapsulates ScribeEntry creation and queue put logic.

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
        q = get_scribe_queue()
        try:
            if event_type in CRITICAL_EVENT_TYPES:
                q.put(entry, block=True)
                logger.debug(f"Successfully queued CRITICAL entry from {origin} with event type {event_type}")
                return True
            else:
                q.put(entry, timeout=0.1)
                logger.debug(f"Successfully queued entry from {origin} with event type {event_type}")
                return True
        except queue.Full:
            logger.warning(f"Scribe queue full, writing to fallback for {origin} ({event_type})")
            try:
                with _fallback_lock:
                    with open(FALLBACK_PATH, "a", encoding="utf-8") as f:
                        json.dump(entry.__dict__, f, default=str)
                        f.write("\n")
                return True
            except Exception as fallback_err:
                logger.error(f"Failed to write scribe event to fallback: {fallback_err}")
                return False
    except Exception as e:
        logger.error(f"Unexpected error queuing scribe event from {origin} ({event_type}): {e}", exc_info=True)
        return False

def clear_scribe_queue(caller: str, confirm: bool = False) -> None:
    """
    Clear all items from the scribe queue.
    Use with caution - only in emergency situations or during shutdown.
    Requires explicit confirmation and caller name.
    """
    if not confirm:
        raise ValueError("Queue clearing requires explicit confirmation (confirm=True)")
    logger.warning(f"Clearing scribe queue by {caller} - this should only be done in emergency situations")
    q = get_scribe_queue()
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except queue.Empty:
            break
    logger.info("Scribe queue cleared successfully")

def get_scribe_queue_size() -> int:
    """
    Get the current number of items in the scribe queue.
    
    Returns:
        int: Number of items currently in the queue.
    """
    q = get_scribe_queue()
    return q.qsize()

def check_scribe_queue_health() -> Tuple[str, float]:
    """
    Check the health status of the scribe queue.
    
    Returns:
        Tuple[str, float]: Status string ("OK", "WARNING", or "FULL") and queue fill ratio
    """
    current_size = get_scribe_queue_size()
    fill_ratio = current_size / MAX_QUEUE_SIZE
    
    if fill_ratio >= 1.0:
        error_msg = f"Scribe queue is full! Current size: {current_size}/{MAX_QUEUE_SIZE}"
        logger.error(error_msg)
        return "FULL", fill_ratio
    elif fill_ratio >= WARNING_THRESHOLD:
        error_msg = f"Scribe queue is approaching capacity: {current_size}/{MAX_QUEUE_SIZE} ({fill_ratio:.1%})"
        logger.warning(error_msg)
        return "WARNING", fill_ratio
    
    logger.debug(f"Scribe queue health check: {current_size}/{MAX_QUEUE_SIZE} ({fill_ratio:.1%})")
    return "OK", fill_ratio
