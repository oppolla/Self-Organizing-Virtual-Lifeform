import json
import os
import logging
from typing import Any, Optional, Dict, List, Union, Callable
from dataclasses import dataclass
from threading import Lock
import traceback
from sovl_logger import Logger

@dataclass
class ConfigSchema:
    """Defines validation rules for configuration fields."""
    field: str
    type: type
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    range: Optional[tuple] = None

class ConfigManager:
    """Manages SOVLSystem configuration with validation, thread safety, and persistence."""
    
    # Configuration schema for all sections
    SCHEMA = [
        # core_config
        ConfigSchema("core_config.base_model_name", str, "gpt2"),
        ConfigSchema("core_config.scaffold_model_name", str, "gpt2"),
        ConfigSchema("core_config.cross_attn_layers", list, [0, 1, 2], lambda x: all(isinstance(i, int) for i in x)),
        ConfigSchema("core_config.use_dynamic_layers", bool, False),
        ConfigSchema("core_config.layer_selection_mode", str, "balanced", lambda x: x in ["balanced", "random", "fixed"]),
        ConfigSchema("core_config.custom_layers", list, [], lambda x: all(isinstance(i, int) for i in x)),
        ConfigSchema("core_config.valid_split_ratio", float, 0.2, range=(0.0, 1.0)),
        ConfigSchema("core_config.random_seed", int, 42, range=(0, 2**32)),
        ConfigSchema("core_config.quantization", str, "fp16", lambda x: x in ["fp16", "int8", "fp32"]),
        ConfigSchema("core_config.hidden_size", int, 768, range=(64, 4096)),
        # controls_config (for TemperamentSystem and SOVLProcessor)
        ConfigSchema("controls_config.temp_eager_threshold", float, 0.8, range=(0.7, 0.9)),
        ConfigSchema("controls_config.temp_sluggish_threshold", float, 0.6, range=(0.4, 0.6)),
        ConfigSchema("controls_config.temp_mood_influence", float, 0.0, range=(0.0, 1.0)),
        ConfigSchema("controls_config.temp_curiosity_boost", float, 0.5, range=(0.0, 0.5)),
        ConfigSchema("controls_config.temp_restless_drop", float, 0.1, range=(0.0, 0.5)),
        ConfigSchema("controls_config.temp_melancholy_noise", float, 0.02, range=(0.0, 0.05)),
        ConfigSchema("controls_config.conf_feedback_strength", float, 0.5, range=(0.0, 1.0)),
        ConfigSchema("controls_config.temp_smoothing_factor", float, 0.0, range=(0.0, 1.0)),
        ConfigSchema("controls_config.temperament_history_maxlen", int, 5, range=(3, 10)),
        ConfigSchema("controls_config.confidence_history_maxlen", int, 5, range=(3, 10)),
        ConfigSchema("controls_config.flat_distribution_confidence", float, 0.2, range=(0.0, 0.5)),
        ConfigSchema("controls_config.confidence_var_threshold", float, 1e-5, range=(1e-6, 1e-4)),
        ConfigSchema("controls_config.confidence_smoothing_factor", float, 0.0, range=(0.0, 1.0)),
        ConfigSchema("controls_config.max_confidence_history", int, 10, range=(5, 20)),
        # logging_config
        ConfigSchema("logging_config.log_file", str, "sovl_logs.jsonl", lambda x: x.endswith(".jsonl")),
        ConfigSchema("logging_config.max_size_mb", int, 10, range=(0, 100)),
        ConfigSchema("logging_config.compress_old", bool, False),
        ConfigSchema("logging_config.max_in_memory_logs", int, 1000, range=(100, 10000)),
    ]

    def __init__(self, config_file: str, logger: Logger):
        """
        Initialize ConfigManager with configuration file path and logger.

        Args:
            config_file: Path to configuration file
            logger: Logger instance for recording events
        """
        self.config_file = config_file
        self.logger = logger
        self.config = {}  # Flat dictionary
        self._structured_config = {}  # Structured dictionary
        self._cache = {}  # Cache for frequent access
        self.lock = Lock()
        self._load_config()

    def _load_config(self):
        """Load and validate configuration file."""
        try:
            with self.lock:
                if os.path.exists(self.config_file):
                    with open(self.config_file, "r", encoding='utf-8') as f:
                        self.config = json.load(f)
                else:
                    self.config = {}
                self._validate_config()
                self._build_structured_config()
                self._update_cache()
                self.logger.record({
                    "event": "config_load",
                    "config_file": self.config_file,
                    "timestamp": time.time()
                })
        except FileNotFoundError:
            self.logger.record({
                "error": f"Configuration file {self.config_file} not found",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            self.config = {}
        except json.JSONDecodeError as e:
            self.logger.record({
                "error": f"Error decoding JSON from {self.config_file}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            self.config = {}
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load config: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _validate_config(self):
        """Validate configuration against schema."""
        for schema in self.SCHEMA:
            value = self.get(schema.field, schema.default)
            if value is None:
                continue
            if not isinstance(value, schema.type):
                self.logger.record({
                    "warning": f"Invalid type for {schema.field}: expected {schema.type}, got {type(value)}",
                    "timestamp": time.time()
                })
                self._set_value(schema.field, schema.default)
                continue
            if schema.validator and not schema.validator(value):
                self.logger.record({
                    "warning": f"Invalid value for {schema.field}: {value}",
                    "timestamp": time.time()
                })
                self._set_value(schema.field, schema.default)
                continue
            if schema.range:
                min_val, max_val = schema.range
                if not (min_val <= value <= max_val):
                    self.logger.record({
                        "warning": f"Value for {schema.field} out of range [{min_val}, {max_val}]: {value}",
                        "timestamp": time.time()
                    })
                    self._set_value(schema.field, schema.default)

    def _build_structured_config(self):
        """Build structured config from flat config."""
        self._structured_config = {
            "core_config": {},
            "controls_config": {},
            "logging_config": {},
        }
        for schema in self.SCHEMA:
            section, field = schema.field.split('.')
            if section not in self._structured_config:
                self._structured_config[section] = {}
            self._structured_config[section][field] = self.get(schema.field, schema.default)

    def _update_cache(self):
        """Update cache with current config values."""
        self._cache = {schema.field: self.get(schema.field, schema.default) for schema in self.SCHEMA}

    def _set_value(self, key: str, value: Any):
        """Set value in flat and structured configs."""
        keys = key.split('.')
        if len(keys) != 2:
            return
        section, field = keys
        if section not in self.config:
            self.config[section] = {}
        self.config[section][field] = value
        if section in self._structured_config:
            self._structured_config[section][field] = value
        self._cache[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the configuration using a dot-separated key.

        Args:
            key: Dot-separated configuration key
            default: Default value if key is missing

        Returns:
            Configuration value or default
        """
        with self.lock:
            if key in self._cache:
                return self._cache[key]
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, default)
                else:
                    return default
            if value == {}:
                self.logger.record({
                    "warning": f"Key '{key}' is empty or missing. Using default: {default}",
                    "timestamp": time.time()
                })
                return default
            return value

    def validate_keys(self, required_keys: List[str]):
        """
        Validate that all required keys exist in the configuration.

        Args:
            required_keys: List of required configuration keys

        Raises:
            ValueError: If any required keys are missing
        """
        with self.lock:
            missing_keys = [key for key in required_keys if self.get(key, None) is None]
            if missing_keys:
                self.logger.record({
                    "error": f"Missing required configuration keys: {', '.join(missing_keys)}",
                    "timestamp": time.time()
                })
                raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section as dict.

        Args:
            section: Configuration section name

        Returns:
            Dictionary of section key-value pairs
        """
        with self.lock:
            return self._structured_config.get(section, {})

    def update(self, key: str, value: Any) -> bool:
        """
        Update a configuration value with validation.

        Args:
            key: Dot-separated configuration key
            value: New value

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            with self.lock:
                for schema in self.SCHEMA:
                    if schema.field == key:
                        if not isinstance(value, schema.type):
                            self.logger.record({
                                "error": f"Invalid type for {key}: expected {schema.type}, got {type(value)}",
                                "timestamp": time.time()
                            })
                            return False
                        if schema.validator and not schema.validator(value):
                            self.logger.record({
                                "error": f"Invalid value for {key}: {value}",
                                "timestamp": time.time()
                            })
                            return False
                        if schema.range and not (schema.range[0] <= value <= schema.range[1]):
                            self.logger.record({
                                "error": f"Value for {key} out of range {schema.range}: {value}",
                                "timestamp": time.time()
                            })
                            return False
                        break
                else:
                    self.logger.record({
                        "error": f"Unknown configuration key: {key}",
                        "timestamp": time.time()
                    })
                    return False

                self._set_value(key, value)
                self.logger.record({
                    "event": "config_update",
                    "key": key,
                    "value": value,
                    "timestamp": time.time()
                })
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update config key {key}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def update_batch(self, updates: Dict[str, Any]) -> bool:
        """
        Update multiple configuration values atomically.

        Args:
            updates: Dictionary of key-value pairs to update

        Returns:
            True if all updates succeeded, False otherwise
        """
        try:
            with self.lock:
                for key, value in updates.items():
                    if not self.update(key, value):
                        return False
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update batch config: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def save_config(self, file_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file atomically.

        Args:
            file_path: Optional path to save config (defaults to config_file)

        Returns:
            True if save succeeded, False otherwise
        """
        save_path = file_path or self.config_file
        temp_file = f"{save_path}.tmp"
        try:
            with self.lock:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
                os.replace(temp_file, save_path)
                self.logger.record({
                    "event": "config_save",
                    "file_path": save_path,
                    "timestamp": time.time()
                })
                return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to save config to {save_path}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def get_state(self) -> Dict[str, Any]:
        """
        Export current configuration state.

        Returns:
            Dictionary containing config state
        """
        with self.lock:
            return {
                "config_file": self.config_file,
                "config": self.config
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load configuration state.

        Args:
            state: Dictionary containing config state
        """
        try:
            with self.lock:
                self.config_file = state.get("config_file", self.config_file)
                self.config = state.get("config", {})
                self._validate_config()
                self._build_structured_config()
                self._update_cache()
                self.logger.record({
                    "event": "config_load_state",
                    "config_file": self.config_file,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load config state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def tune(self, **kwargs) -> bool:
        """
        Dynamically tune configuration parameters.

        Args:
            **kwargs: Parameters to update

        Returns:
            True if tuning succeeded, False otherwise
        """
        return self.update_batch(kwargs)

    if __name__ == "__main__":
        logger = Logger(LoggerConfig())
        config_manager = ConfigManager("sovl_config.json", logger)
        try:
            config_manager.validate_keys(["core_config.base_model_name", "controls_config.temp_eager_threshold"])
        except ValueError as e:
            print(e)
