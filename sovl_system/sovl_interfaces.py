from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import torch
from threading import Lock
from sovl_config import ConfigManager
from sovl_logger import LoggingManager, Logger
from sovl_state import SOVLState, StateManager
from sovl_error import ErrorHandler
import traceback
import time
from sovl_experience import MemoriaManager
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_main import SOVLSystem, SystemContext
from sovl_monitor import MemoryMonitor

class SystemInterface(ABC):
    """
    Abstract interface for the SOVL system, defining essential methods to break
    circular dependency with the orchestrator.
    """
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current system state.
        
        Returns:
            Dictionary containing the system state.
        """
        pass
    
    @abstractmethod
    def update_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Update the system state with the provided dictionary.
        
        Args:
            state_dict: Dictionary containing state updates.
        
        Raises:
            ValueError: If state update is invalid.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the system, saving state and releasing resources.
        
        Raises:
            RuntimeError: If shutdown fails.
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
        logger: LoggingManager,
        device: torch.device
    ):
        """
        Initialize the mediator with core dependencies.
        
        Args:
            config_manager: Configuration manager.
            logger: Logging manager.
            device: Device for tensor operations.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.state_manager = StateManager(
            config_manager=config_manager,
            logger=logger,
            device=device
        )
        self.error_handler = ErrorHandler(logger)
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
                self._system.update_state(merged_state)
                self.state_manager.save_state(SOVLState.from_dict(merged_state, self.device))
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
        """
        Log an event with standardized metadata.
        
        Args:
            event_type: Type of the event.
            additional_info: Additional event data.
        """
        try:
            metadata = {
                "conversation_id": self.state_manager.load_state().history.conversation_id,
                "timestamp": time.time(),
                "device": str(self.device)
            }
            if additional_info:
                metadata.update(additional_info)
            self.logger.record_event(
                event_type=f"mediator_{event_type}",
                message=f"Mediator event: {event_type}",
                level="info",
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log event: {str(e)}")
    
    def _log_error(self, message: str, error: Exception) -> None:
        """
        Log an error with standardized metadata.
        
        Args:
            message: Error message.
            error: Exception instance.
        """
        try:
            metadata = {
                "conversation_id": self.state_manager.load_state().history.conversation_id,
                "error": str(error),
                "stack_trace": traceback.format_exc()
            }
            self.logger.log_error(
                error_msg=message,
                error_type="mediator_error",
                stack_trace=traceback.format_exc(),
                additional_info=metadata
            )
        except Exception as e:
            print(f"Failed to log error: {str(e)}")
    
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
    
    def __init__(self, sovl_system: 'SOVLSystem'):
        self._system = sovl_system
    
    def get_state(self) -> Dict[str, Any]:
        return self._system.state_tracker.get_state()
    
    def update_state(self, state_dict: Dict[str, Any]) -> None:
        self._system.state_tracker.state.from_dict(state_dict, self._system.context.device)
    
    def shutdown(self) -> None:
        self._system.state_tracker.state.save_state()

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

class SystemInterface:
    """Interface for system components."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        memoria_manager: MemoriaManager,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager
    ):
        """
        Initialize system interface.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            memoria_manager: MemoriaManager instance for core memory management
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
        """
        self._config_manager = config_manager
        self._logger = logger
        self.memoria_manager = memoria_manager
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health across all memory managers."""
        try:
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            return {
                "ram_health": ram_health,
                "gpu_health": gpu_health
            }
        except Exception as e:
            self._logger.log_error(
                error_msg=f"Failed to check memory health: {str(e)}",
                error_type="memory_health_error",
                stack_trace=traceback.format_exc()
            )
            return {
                "ram_health": {"status": "error"},
                "gpu_health": {"status": "error"}
            }

# Usage example
if __name__ == "__main__":
    from sovl_conductor import SOVLOrchestrator
    from sovl_main import SOVLSystem, SystemContext, ConfigHandler, ModelLoader, CuriosityEngine, MemoryMonitor, StateTracker, ErrorManager
    
    # Initialize dependencies
    config_path = "sovl_config.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = LoggingManager(log_file="sovl_orchestrator_logs.jsonl")
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
