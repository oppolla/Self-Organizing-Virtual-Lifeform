
import json
from functools import lru_cache
from typing import Optional, Any
from pathlib import Path
from sovl_schema import (
       GestationConfig,
       OrchestratorConfig,
       LoggingConfig,
       ControlsConfig,
       MonitoringConfig,
       RAMConfig,
       GPUConfig,
       MetadataProcessorConfig,
       ScribedConfig,
       TrainerWeightingConfig,
       EventTypeWeightsConfig,
       IOConfig,
       CoreConfig,
       GenerationConfig,
       TrainingConfig,
       CuriosityConfig,
       TemperamentConfig,
       ConfidenceConfig,
       BondingConfig,
       BondConfig,
       IntrospectionConfig,
       VibeConfig,
       AspirationConfig,
       ScaffoldConfig,
       EngramLoraConfig,
       LoraConfig,
       ModelConfig,
       MemoryConfig,
       TrainingConfigSchema,
       QueueConfig,
       ErrorConfig,
       StateConfig
   )

"""
SOVL Config Manager

This module loads and manages the SOVL system configuration using schema (sovl_config_schema.py).
"""

CONFIG_PATH = Path(__file__).parent / "sovl_config.json"


class SOVLConfig:
    # Add all top-level config sections as fields
    gestation: Optional[GestationConfig] = None
    orchestrator: Optional[OrchestratorConfig] = None
    logging: Optional[LoggingConfig] = None
    controls: Optional[ControlsConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    ram: Optional[RAMConfig] = None
    gpu: Optional[GPUConfig] = None
    metadata_processor: Optional[MetadataProcessorConfig] = None
    scribed: Optional[ScribedConfig] = None
    trainer_weighting: Optional[TrainerWeightingConfig] = None
    event_type_weights: Optional[EventTypeWeightsConfig] = None
    io: Optional[IOConfig] = None
    core: Optional[CoreConfig] = None
    generation: Optional[GenerationConfig] = None
    training: Optional[TrainingConfig] = None
    curiosity: Optional[CuriosityConfig] = None
    temperament: Optional[TemperamentConfig] = None
    confidence: Optional[ConfidenceConfig] = None
    bonding: Optional[BondingConfig] = None
    bond: Optional[BondConfig] = None
    introspection: Optional[IntrospectionConfig] = None
    vibe: Optional[VibeConfig] = None
    aspiration: Optional[AspirationConfig] = None
    scaffold: Optional[ScaffoldConfig] = None
    engram_lora: Optional[EngramLoraConfig] = None
    lora: Optional[LoraConfig] = None
    model: Optional[ModelConfig] = None
    memory: Optional[MemoryConfig] = None
    training_schema: Optional[TrainingConfigSchema] = None
    queue: Optional[QueueConfig] = None
    error: Optional[ErrorConfig] = None
    state: Optional[StateConfig] = None

_config_instance: Optional[SOVLConfig] = None

def load_config(path: str = CONFIG_PATH, reload: bool = False) -> SOVLConfig:
    """
    Load and validate the SOVL config from JSON.
    Caches the config instance unless reload=True.
    """
    global _config_instance
    if _config_instance is not None and not reload:
        return _config_instance
    with open(path, "r") as f:
        data = json.load(f)
    _config_instance = SOVLConfig(**data)
    return _config_instance

def get_config() -> SOVLConfig:
    """
    Alias for load_config() for legacy code.
    """
    return load_config()

def get(key: str, default: Any = None) -> Any:
    """
    Legacy helper: get a config value by dot-separated key, e.g. 'controls_config.base_temperature'.
    """
    config = load_config()
    parts = key.split(".")
    value = config
    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value

def get_section(section: str) -> Any:
    """
    Legacy helper: get a full config section by name.
    """
    config = load_config()
    return getattr(config, section, None)

# --- Backwards compatibility shims ---

class ConfigNamespace:
    """
    Shim for legacy ConfigNamespace: wraps a config section for attribute/dict access.
    """
    def __init__(self, data):
        self._data = data
    def __getattr__(self, key):
        value = getattr(self._data, key, None)
        if isinstance(value, (dict, SOVLConfig)):
            return ConfigNamespace(value)
        return value
    def __getitem__(self, key):
        return getattr(self._data, key)
    def __setitem__(self, key, value):
        setattr(self._data, key, value)
    def __repr__(self):
        return f"ConfigNamespace({self._data})"

class ConfigManager:
    """
    Shim for legacy ConfigManager: provides get(), get_section(), and attribute access.
    Uses the singleton SOVLConfig instance under the hood.
    """
    def __init__(self, path: str = CONFIG_PATH):
        self._config = load_config(path)
    def get(self, key: str, default: Any = None) -> Any:
        return get(key, default)
    def get_section(self, section: str) -> Any:
        return get_section(section)
    def __getattr__(self, name):
        section = getattr(self._config, name, None)
        if section is not None:
            return ConfigNamespace(section)
        raise AttributeError(f"ConfigManager has no attribute '{name}'")
    def reload(self):
        global _config_instance
        _config_instance = None
        self._config = load_config()
        return self._config

# Example usage for CLI or scripts
if __name__ == "__main__":
    config = load_config()
    print(config.json(indent=2))
