from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING, Protocol
import traceback
import time
from threading import Lock
import torch

if TYPE_CHECKING:
    from sovl_config import ConfigManager
    from sovl_logger import Logger
    from sovl_state import StateManager
    from sovl_error import ErrorManager
    from sovl_memory import RAMManager, GPUMemoryManager
    from sovl_main import SOVLSystem, SystemContext

"""
Defines core abstract interfaces, protocols, and mediation components for the SOVL system.

This module serves as a central point for establishing contracts between major
subsystems, such as the SOVLSystem and SOVLOrchestrator, and for managing
their interactions to promote loose coupling and resolve dependencies.

Key components include:
- Abstract Base Classes (ABCs):
    - SystemInterface: Defines the contract for the main SOVL system's operations.
    - OrchestratorInterface: Defines the contract for the system orchestrator.
    - StateAccessor: Defines the contract for state access and atomic updates.
- Concrete Mediation Classes:
    - SystemMediator: Manages interactions between the SOVLSystem and
      SOVLOrchestrator via their interfaces to prevent circular dependencies.
    - SOVLSystemAdapter: Adapts a SOVLSystem instance to the SystemInterface.
    - SOVLOrchestratorAdapter: Adapts an SOVLOrchestrator instance to the
      OrchestratorInterface.
- Protocols (for Structural Typing):
    - HasSessionID: Specifies an object is expected to provide a 'session_id'.
    - HasLogger: Specifies an object is expected to provide a 'logger'.

The primary architectural goal of this module is to enable a decoupled system
where components interact through these well-defined abstractions rather than
relying on direct concrete dependencies.
"""
    
class SystemInterface(ABC):
    """Core interface for the SOVL system."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current system state.
        
        Returns:
            Dict[str, Any]: The current system state.
        """
        pass
    
    @abstractmethod
    def update_state(self, state_dict: Dict[str, Any]) -> None:
        """Update the system state.
        
        Args:
            state_dict: A dictionary containing state updates.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the system, saving state and releasing resources.
        
        Raises:
            RuntimeError: If shutdown fails.
        """
        pass
    
    @abstractmethod
    def toggle_memory(self, enable: bool) -> bool:
        """Enable or disable memory features.
        
        Args:
            enable: True to enable memory, False to disable.
        
        Returns:
            bool: Current memory state after operation.
        """
        pass
    
    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage.
        
        Returns:
            Dict[str, Any]: Memory usage statistics.
        """
        pass
    
    @abstractmethod
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all system components.
        
        Returns:
            Dict[str, bool]: Component status (True for active, False for inactive).
        """
        pass
    
    @abstractmethod
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state including status of all components.
        
        Returns:
            Dict[str, Any]: Complete system state.
        """
        pass

class OrchestratorInterface(ABC):
    """
    Abstract interface for the SOVL orchestrator, defining methods for system
    coordination without direct dependency on SOVLSystem.
    """
    
    @abstractmethod
    def set_system(self, system: SystemInterface) -> None:
        """
        Set the system instance for orchestration.
        
        Args:
            system: SystemInterface implementation.
        """
        pass
    
    @abstractmethod
    def sync_state(self) -> None:
        """
        Synchronize orchestrator state with the system state.
        
        Raises:
            RuntimeError: If state synchronization fails.
        """
        pass

class SystemMediator:
    """
    Mediates interactions between SOVLOrchestrator and SOVLSystem to eliminate
    circular dependencies.
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        device: torch.device,
        state_manager: StateManager = None
    ):
        """
        Initialize the mediator with core dependencies.
        
        Args:
            config_manager: Configuration manager.
            logger: Logging manager.
            device: Device for tensor operations.
            state_manager: StateManager for atomic state updates.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.state_manager = state_manager or StateManager(
            config_manager=config_manager,
            logger=logger,
            device=device
        )
        self.error_handler = ErrorManager(logger)
        self._system: Optional[SystemInterface] = None
        self._orchestrator: Optional[OrchestratorInterface] = None
        self._lock = Lock()
        self._log_event("mediator_initialized", {
            "device": str(device),
            "config_path": config_manager.config_path
        })
    
    def register_system(self, system: SystemInterface) -> None:
        """
        Register the SOVL system with the mediator.
        
        Args:
            system: SystemInterface implementation.
        """
        with self._lock:
            try:
                self._system = system
                if self._orchestrator:
                    self._orchestrator.set_system(system)
                self._sync_state()
                self._log_event("system_registered", {
                    "state_hash": self.state_manager.load_state().state_hash
                })
            except Exception as e:
                self._log_error("System registration failed", e)
                raise
    
    def register_orchestrator(self, orchestrator: OrchestratorInterface) -> None:
        """
        Register the orchestrator with the mediator.
        
        Args:
            orchestrator: OrchestratorInterface implementation.
        """
        with self._lock:
            try:
                self._orchestrator = orchestrator
                if self._system:
                    self._orchestrator.set_system(self._system)
                self._log_event("orchestrator_registered", {})
            except Exception as e:
                self._log_error("Orchestrator registration failed", e)
                raise
    
    def sync_state(self) -> None:
        """
        Synchronize state between orchestrator and system.
        
        Raises:
            RuntimeError: If state synchronization fails.
        """
        with self._lock:
            try:
                if not self._system or not self._orchestrator:
                    return
                system_state = self._system.get_state()
                orchestrator_state = self.state_manager.load_state().to_dict()
                merged_state = self._merge_states(system_state, orchestrator_state)
                # Atomically update state using state_manager
                def update_fn(state):
                    # This assumes SOVLState has a from_dict method that mutates in place
                    state.from_dict(merged_state, self.device)
                self.state_manager.update_state_atomic(update_fn)
                self._orchestrator.sync_state()
                self._log_event("state_synchronized", {
                    "system_state_hash": self._hash_state(system_state),
                    "orchestrator_state_hash": self._hash_state(orchestrator_state)
                })
            except Exception as e:
                self._log_error("State synchronization failed", e)
                raise
    
    def shutdown(self) -> None:
        """
        Shutdown the system via the mediator.
        
        Raises:
            RuntimeError: If shutdown fails.
        """
        with self._lock:
            try:
                if self._system:
                    self._system.shutdown()
                # Optionally, perform an atomic update or just save the current state
                # If any shutdown state mutation is needed, do it atomically here
                self.state_manager.save_state(self.state_manager.load_state())
                self._log_event("system_shutdown", {})
            except Exception as e:
                self._log_error("System shutdown failed", e)
                self.error_handler.handle_generic_error(
                    error=e,
                    context="system_shutdown",
                    fallback_action=self._emergency_shutdown
                )
                raise
    
    def _merge_states(self, system_state: Dict[str, Any], orchestrator_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge system and orchestrator states, prioritizing system state for critical fields.
        
        Args:
            system_state: System state dictionary.
            orchestrator_state: Orchestrator state dictionary.
        
        Returns:
            Merged state dictionary.
        """
        merged = orchestrator_state.copy()
        critical_fields = ['history', 'dream_memory', 'seen_prompts', 'confidence_history']
        for field in critical_fields:
            if field in system_state:
                merged[field] = system_state[field]
        return merged
    
    def _hash_state(self, state: Dict[str, Any]) -> str:
        """
        Generate a hash for a state dictionary.
        
        Args:
            state: State dictionary.
        
        Returns:
            Hash string.
        """
        import hashlib
        import json
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def _log_event(self, event_type: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        try:
            if self.logger:
                self.logger.record_event(
                    event_type=event_type,
                    message=f"SystemMediator event: {event_type}",
                    level="info",
                    additional_info=additional_info or {}
                )
            else:
                print(f"[EVENT] {event_type}: {additional_info}")
        except Exception as e:
            print(f"[ERROR] Failed to log event '{event_type}': {str(e)}")
            traceback.print_exc()

    def _log_error(self, message: str, error: Exception) -> None:
        try:
            if self.logger:
                self.logger.log_error(
                    error_msg=message,
                    error_type="mediator_error",
                    stack_trace=traceback.format_exc()
                )
            else:
                print(f"[ERROR] {message}: {str(error)}")
        except Exception as e:
            print(f"[ERROR] Failed to log mediator error: {str(e)}")
            traceback.print_exc()
    
    def _emergency_shutdown(self) -> None:
        """
        Perform emergency shutdown procedures.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.close()
            self._log_event("emergency_shutdown", {"timestamp": time.time()})
        except Exception as e:
            print(f"Emergency shutdown failed: {str(e)}")

class SOVLSystemAdapter(SystemInterface):
    """
    Adapter to make SOVLSystem compatible with SystemInterface.
    """
    
    def __init__(self, sovl_system: 'SOVLSystem', state_manager: StateManager = None):
        self.sovl_system = sovl_system
        self.state_manager = state_manager or getattr(sovl_system, 'state_manager', None)
    
    def get_state(self) -> Dict[str, Any]:
        return self.sovl_system.get_state()
    
    def update_state(self, state_dict: Dict[str, Any]) -> None:
        # Atomically update state using state_manager
        if self.state_manager:
            def update_fn(state):
                # This assumes SOVLState has a from_dict method that mutates in place
                state.from_dict(state_dict, getattr(self.sovl_system, 'device', None))
            self.state_manager.update_state_atomic(update_fn)
        else:
            self.sovl_system.update_state(state_dict)
    
    def shutdown(self) -> None:
        self.sovl_system.shutdown()

    def toggle_memory(self, enable: bool) -> bool:
        """Enable or disable memory features."""
        return self.sovl_system.toggle_memory(enable)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        return self.sovl_system.get_memory_stats()

    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all system components."""
        return self.sovl_system.get_component_status()

    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state including status of all components."""
        return self.sovl_system.get_system_state()

class SOVLOrchestratorAdapter(OrchestratorInterface):
    """
    Adapter to make SOVLOrchestrator compatible with OrchestratorInterface.
    """
    
    def __init__(self, orchestrator: 'SOVLOrchestrator'):
        self._orchestrator = orchestrator
    
    def set_system(self, system: SystemInterface) -> None:
        self._orchestrator.set_system(system)
    
    def sync_state(self) -> None:
        self._orchestrator._sync_state_to_system()

class StateAccessor(ABC):
    """Interface for accessing and manipulating state.
    This interface allows components to access state without direct dependencies.
    """
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get the current state dictionary.
        
        Returns:
            Dict[str, Any]: The current state.
        """
        pass
    
    @abstractmethod
    def update_state_atomic(self, update_fn) -> bool:
        """Update state atomically using an update function.
        
        Args:
            update_fn: A function that takes the current state and returns an updated state.
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_state_version(self) -> int:
        """Get the current state version.
        
        Returns:
            int: Current state version number.
        """
        pass
    
    @abstractmethod
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate a state dictionary.
        
        Args:
            state: The state dictionary to validate.
            
        Returns:
            bool: True if state is valid, False otherwise.
        """
        pass

class HasSessionID(Protocol):
    @property
    def session_id(self) -> Optional[str]:
        ...

class HasLogger(Protocol):
    @property
    def logger(self) -> Any:
        ...

# Usage example
if __name__ == "__main__":
    from sovl_conductor import SOVLOrchestrator
    from sovl_main import SOVLSystem, SystemContext, ConfigHandler, ModelLoader, CuriosityEngine, MemoryMonitor, StateTracker, ErrorManager
    
    # Initialize dependencies
    config_path = "sovl_config.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(log_file="sovl_orchestrator_logs.jsonl")
    config_manager = ConfigManager(config_path, logger)
    context = SystemContext(config_path, str(device))
    config_handler = ConfigHandler(config_path, context.logger, context.event_dispatcher)
    state_tracker = StateTracker(context)
    error_manager = ErrorManager(context, state_tracker)
    model_loader = ModelLoader(context)
    memory_monitor = MemoryMonitor(
        config_manager=config_manager,
        logger=logger,
        ram_manager=None,  # Not needed for interface testing
        gpu_manager=None,  # Not needed for interface testing
        error_manager=error_manager
    )
    curiosity_engine = CuriosityEngine(
        config_handler=config_handler,
        model_loader=model_loader,
        state_tracker=state_tracker,
        error_manager=error_manager,
        logger=context.logger,
        device=str(device)
    )
    
    # Create system and orchestrator
    system = SOVLSystem(
        context=context,
        config_handler=config_handler,
        model_loader=model_loader,
        curiosity_engine=curiosity_engine,
        memory_monitor=memory_monitor,
        state_tracker=state_tracker,
        error_manager=error_manager
    )
    orchestrator = SOVLOrchestrator(config_path=config_path, log_file="sovl_orchestrator_logs.jsonl")
    
    # Create mediator and adapters
    mediator = SystemMediator(
        config_manager=config_manager,
        logger=logger,
        device=device
    )
    system_adapter = SOVLSystemAdapter(system)
    orchestrator_adapter = SOVLOrchestratorAdapter(orchestrator)
    
    # Register components with mediator
    mediator.register_system(system_adapter)
    mediator.register_orchestrator(orchestrator_adapter)
    
    try:
        orchestrator.run()
    finally:
        mediator.shutdown()
