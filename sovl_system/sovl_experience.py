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


