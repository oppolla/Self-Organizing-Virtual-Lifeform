import json
import os
import gzip
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable, Tuple
from dataclasses import dataclass
from threading import Lock
import traceback
import time
from sovl_logger import Logger
from sovl_schema import ValidationSchema
from sovl_error import ErrorManager, ErrorRecord, ConfigurationError
import tempfile
import shutil
from sovl_utils import set_nested_dict_value, get_nested_dict_value

# ConfigSchema defines validation rules and defaults for configuration fields.
@dataclass
class ConfigSchema:
    """Defines validation rules for configuration fields."""
    field: str
    type: type
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None
    range: Optional[tuple] = None
    required: bool = False
    nullable: bool = False

# SchemaValidator checks config values against schemas and logs issues.
class SchemaValidator:
    """Handles configuration schema validation logic."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.schemas: Dict[str, ConfigSchema] = {}
        self._lock = Lock()

    def register(self, schemas: List[ConfigSchema]) -> None:
        """Register new schemas with thread safety."""
        with self._lock:
            self.schemas.update({s.field: s for s in schemas})

    def validate(self, key: str, value: Any, conversation_id: str = "init") -> tuple[bool, Any]:
        """Validate a value against its schema with thread safety and error logging."""
        with self._lock:
            schema = self.schemas.get(key)
            if not schema:
                self.logger.record({
                    "error": f"Unknown configuration key: {key}",
                    "timestamp": time.time(),
                    "conversation_id": conversation_id
                })
                return False, None

            if value is None:
                if schema.required:
                    self.logger.record({
                        "error": f"Required field {key} is missing",
                        "suggested": f"Set to default: {schema.default}",
                        "timestamp": time.time(),
                        "conversation_id": conversation_id
                    })
                    return False, schema.default
                if schema.nullable:
                    return True, value
                return False, schema.default

            if not isinstance(value, schema.type):
                self.logger.record({
                    "warning": f"Invalid type for {key}: expected {schema.type.__name__}, got {type(value).__name__}",
                    "suggested": f"Set to default: {schema.default}",
                    "timestamp": time.time(),
                    "conversation_id": conversation_id
                })
                return False, schema.default

            if schema.validator and not schema.validator(value):
                valid_options = getattr(schema.validator, '__doc__', '') or str(schema.validator)
                self.logger.record({
                    "warning": f"Invalid value for {key}: {value}",
                    "suggested": f"Valid options: {valid_options}, default: {schema.default}",
                    "timestamp": time.time(),
                    "conversation_id": conversation_id
                })
                return False, schema.default

            if schema.range and not (schema.range[0] <= value <= schema.range[1] if schema.range[1] is not None else schema.range[0] <= value):
                self.logger.record({
                    "warning": f"Value for {key} out of range {schema.range}: {value}",
                    "suggested": f"Set to default: {schema.default}",
                    "timestamp": time.time(),
                    "conversation_id": conversation_id
                })
                return False, schema.default

            return True, value

    def validate_batch(self, items: Dict[str, Any], conversation_id: str = "init") -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Validate a batch of key-value pairs. Returns (all_valid, valid_items, invalid_items_with_defaults).
        """
        all_valid = True
        valid_items = {}
        invalid_items = {}
        for key, value in items.items():
            valid, corrected = self.validate(key, value, conversation_id)
            if valid:
                valid_items[key] = value
            else:
                all_valid = False
                invalid_items[key] = corrected
        return all_valid, valid_items, invalid_items

    def required_keys_missing(self, required_keys: List[str], config: Dict[str, Any]) -> List[str]:
        """Return a list of required keys that are missing from the config."""
        return [key for key in required_keys if key not in config or config[key] is None]

    def required_section_keys_missing(self, section: str, required_keys: List[str], structured_config: Dict[str, Any]) -> List[str]:
        """Return a list of required keys missing from a section."""
        section_dict = structured_config.get(section, {})
        return [key for key in required_keys if key not in section_dict or section_dict[key] is None]

# ConfigStore maintains both flat and structured views for efficient lookups and organized access.
class ConfigStore:
    """Manages configuration storage and structure."""

    def __init__(self):
        self.flat_config: Dict[str, Any] = {}
        self.structured_config: Dict[str, Any] = {
            "core_config": {},
            "lora_config": {},
            "training_config": {"dry_run_params": {}},
            "curiosity_config": {"lifecycle_params": {}},
            "cross_attn_config": {},
            "controls_config": {"lifecycle_params": {}},
            "logging_config": {},
            "dynamic_weighting": {},
            "preprocessing": {},
            "augmentation": {},
            "hardware": {},
            "error_config": {},
            "generation_config": {},
            "data_config": {},
            "memory_config": {},
            "state_config": {},
            "temperament_config": {},
            "confidence_config": {},
            "model": {},
            "data_provider": {},
            "dream_memory_config": {},
            "scaffold_config": {},
        }
        self.cache: Dict[str, Any] = {}
        self._lock = Lock()

    def set_value(self, key: str, value: Any) -> None:
        """Set a value in flat and structured configs with thread safety."""
        with self._lock:
            self.cache[key] = value
            set_nested_dict_value(self.flat_config, key, value)
            set_nested_dict_value(self.structured_config, key, value)

    def get_value(self, key: str, default: Any) -> Any:
        """Retrieve a value from the configuration with thread safety."""
        with self._lock:
            if key in self.cache:
                return self.cache[key]
            value = get_nested_dict_value(self.flat_config, key, default)
            return value if value != {} and value is not None else default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section with thread safety."""
        with self._lock:
            return self.structured_config.get(section, {})

    def rebuild_structured(self, schemas: List[ConfigSchema]) -> None:
        """Rebuild structured config from flat config with thread safety."""
        with self._lock:
            for schema in schemas:
                set_nested_dict_value(self.structured_config, schema.field, self.get_value(schema.field, schema.default))

    def update_cache(self, schemas: List[ConfigSchema]) -> None:
        """Update cache with current config values with thread safety."""
        with self._lock:
            self.cache = {schema.field: self.get_value(schema.field, schema.default) for schema in schemas}

# FileHandler loads and saves configuration files with retry logic.
class FileHandler:
    """Handles configuration file operations."""

    def __init__(self, config_file: str, logger: Logger):
        self.config_file = config_file
        self.logger = logger
        self._lock = Lock()

    def load(self, max_retries: int = 3) -> Dict[str, Any]:
        """Load configuration file with retry logic and defensive error handling."""
        for attempt in range(max_retries):
            try:
                with self._lock:
                    if not os.path.isfile(self.config_file):
                        self.logger.record_event(
                            event_type="config_file_missing",
                            message=f"Config file not found: {self.config_file}",
                            level="error"
                        )
                        return None
                    with open(self.config_file, 'r') as f:
                        if self.config_file.endswith('.gz'):
                            with gzip.open(f, 'rt') as gz:
                                data = json.load(gz)
                        else:
                            data = json.load(f)
                return data
            except Exception as e:
                self.logger.record_event(
                    event_type="config_load_error",
                    message=f"Failed to load config (attempt {attempt+1}): {str(e)}",
                    level="error"
                )
                time.sleep(0.1)
        self.logger.record_event(
            event_type="config_load_failed",
            message=f"Failed to load config after {max_retries} attempts",
            level="critical"
        )
        return None

    def save(self, config: Dict[str, Any], file_path: Optional[str] = None, compress: bool = False, max_retries: int = 3) -> bool:
        """Save configuration to file atomically with retry logic and error handling."""
        target_file = file_path or self.config_file
        for attempt in range(max_retries):
            try:
                with self._lock:
                    temp_dir = os.path.dirname(target_file)
                    with tempfile.NamedTemporaryFile('w', dir=temp_dir, delete=False) as tmp:
                        if compress or target_file.endswith('.gz'):
                            with gzip.open(tmp, 'wt') as gz:
                                json.dump(config, gz, indent=2)
                        else:
                            json.dump(config, tmp, indent=2)
                        temp_name = tmp.name
                    os.replace(temp_name, target_file)  # Atomic rename
                self.logger.record_event(
                    event_type="config_save_success",
                    message=f"Config saved to {target_file}",
                    level="info"
                )
                return True
            except Exception as e:
                self.logger.record_event(
                    event_type="config_save_error",
                    message=f"Failed to save config (attempt {attempt+1}): {str(e)}",
                    level="error"
                )
                time.sleep(0.1)
        self.logger.record_event(
            event_type="config_save_failed",
            message=f"Failed to save config after {max_retries} attempts",
            level="critical"
        )
        return False

class ConfigNamespace:
    """Wraps a config dictionary to provide dot-access (attribute) and dict-style access."""
    def __init__(self, data):
        object.__setattr__(self, '_data', data)
    def __getattr__(self, key):
        value = self._data.get(key)
        if isinstance(value, dict):
            return ConfigNamespace(value)
        return value
    def __setattr__(self, key, value):
        self._data[key] = value
    def __getitem__(self, key):
        return self._data[key]
    def __setitem__(self, key, value):
        self._data[key] = value
    def __repr__(self):
        return f"ConfigNamespace({self._data})"

# ConfigManager orchestrates configuration lifecycle: loading, validation, updates, notifications, and persistence.
class ConfigManager:
    """Manages SOVLSystem configuration with validation, thread safety, and persistence.       
    """

    def __init__(self, config_file: str, logger: Logger):
        self.config_file = os.getenv("SOVL_CONFIG_FILE", config_file)
        self.logger = logger
        self.store = ConfigStore()
        self.lock = Lock()
        self.validator = SchemaValidator(self.logger)
        self.file_handler = FileHandler(self.config_file, self.logger)
        self.error_manager = None
        self._frozen = False
        self._last_config_hash = None
        self._initialize_error_manager()
        self._initialize_config()

    def __getattr__(self, name):
        # Provide dot-access for top-level config sections
        if name in self.store.structured_config:
            return ConfigNamespace(self.store.structured_config[name])
        raise AttributeError(f"ConfigManager has no attribute '{name}'")

    def _load_schema(self) -> List[ConfigSchema]:
        """Load schema from ValidationSchema and flatten to a list for validation."""
        schema_dict = ValidationSchema.get_schema()
        flat_schema = []
        for section, fields in schema_dict.items():
            for field, config_schema in fields.items():
                flat_schema.append(config_schema)
        return flat_schema

    def _initialize_config(self) -> None:
        """Initialize configuration from file."""
        try:
            config = self.file_handler.load()
            self.store.flat_config = config
            self._compute_config_hash()
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="config_load_error",
                context={"config_file": self.config_file}
            )
            raise ConfigurationError(f"Failed to initialize configuration: {str(e)}")

    def _compute_config_hash(self) -> str:
        try:
            config_str = json.dumps(self.store.flat_config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception as e:
            self._log_error("Config hash computation failed", {"error": str(e)})
            return ""

    def _validate_and_set_defaults(self) -> None:
        for schema in self._load_schema():
            value = self.store.get_value(schema.field, schema.default)
            is_valid, corrected_value = self.validator.validate(schema.field, value)
            if not is_valid:
                self.store.set_value(schema.field, corrected_value)
                self._log_event("config_validation", f"Set default value for {schema.field}", "warning", {
                    "field": schema.field,
                    "default_value": corrected_value
                })

    def _log_event(self, event_type: str, message: str, level: str, additional_info: Dict[str, Any] = None) -> None:
        self.logger.record_event(event_type=event_type, message=message, level=level, additional_info=additional_info or {})

    def _log_error(self, message: str, additional_info: Dict[str, Any] = None) -> None:
        self._log_event("config_error", message, "error", {
            **(additional_info or {}),
            "stack_trace": traceback.format_exc()
        })

    def freeze(self) -> None:
        with self.lock:
            self._frozen = True
            self._log_event("config_frozen", "Configuration frozen", "info", {"timestamp": time.time()})

    def unfreeze(self) -> None:
        with self.lock:
            self._frozen = False
            self._log_event("config_unfrozen", "Configuration unfrozen", "info", {"timestamp": time.time()})

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        with self.lock:
            value = self.store.get_value(key, default)
            if value == {} or value is None:
                self._log_event("config_warning", f"Key '{key}' is empty or missing. Using default: {default}", "warning", {
                    "key": key,
                    "default_value": default
                })
                return default
            return value

    def validate_keys(self, required_keys: List[str]) -> None:
        """(Refactored) Validate required keys using SchemaValidator."""
        missing_keys = self.validator.required_keys_missing(required_keys, self.store.flat_config)
        if missing_keys:
            self._log_error(f"Missing required configuration keys: {', '.join(missing_keys)}", {"keys": missing_keys})
            raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    def get_section(self, section: str) -> Dict[str, Any]:
        with self.lock:
            return self.store.get_section(section)

    def update(self, key: str, value: Any) -> bool:
        """Update a configuration value, using SchemaValidator for validation."""
        try:
            if self._frozen:
                self.error_manager.record_error(
                    error=ConfigurationError("Configuration is frozen"),
                    error_type="update_error",
                    context={"key": key, "value": value}
                )
                return False
            valid, corrected = self.validator.validate(key, value)
            if not valid:
                self.error_manager.record_error(
                    error=ConfigurationError(f"Invalid value for key {key}"),
                    error_type="validation_error",
                    context={"key": key, "value": value, "suggested": corrected}
                )
                return False
            old_value = self.store.get_value(key)
            self.store.set_value(key, value)
            self._notify_subscribers()
            return True
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="update_error",
                context={"key": key, "value": value}
            )
            return False

    def subscribe(self, callback: Callable[[], None]) -> None:
        with self.lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[], None]) -> None:
        with self.lock:
            self._subscribers.discard(callback)

    def _notify_subscribers(self) -> None:
        with self.lock:
            for callback in self._subscribers:
                try:
                    callback()
                except Exception as e:
                    self._log_error("Failed to notify subscriber of config change", {"error": str(e)})

    def update_batch(self, updates: Dict[str, Any], rollback_on_failure: bool = True) -> bool:
        """Update multiple configuration values at once, using SchemaValidator for batch validation."""
        old_values = {}
        try:
            if self._frozen:
                self.error_manager.record_error(
                    error=ConfigurationError("Configuration is frozen"),
                    error_type="update_error",
                    context={"updates": updates}
                )
                return False
            # Store old values for potential rollback
            for key in updates:
                old_values[key] = self.store.get_value(key)
            # Validate all updates first (batch)
            all_valid, valid_items, invalid_items = self.validator.validate_batch(updates)
            if not all_valid:
                for key, corrected in invalid_items.items():
                    self.error_manager.record_error(
                        error=ConfigurationError(f"Invalid value for key {key}"),
                        error_type="validation_error",
                        context={"key": key, "value": updates[key], "suggested": corrected}
                    )
                if rollback_on_failure:
                    self._rollback_updates(old_values)
                return False
            # Apply updates
            for key, value in valid_items.items():
                self.store.set_value(key, value)
            self._notify_subscribers()
            return True
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="update_error",
                context={"updates": updates}
            )
            if rollback_on_failure:
                self._rollback_updates(old_values)
            return False

    def save_config(self, file_path: Optional[str] = None, compress: bool = False, max_retries: int = 3) -> bool:
        """Save configuration to file."""
        try:
            return self.file_handler.save(
                self.store.flat_config,
                file_path or self.config_file,
                compress,
                max_retries
            )
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="config_save_error",
                context={"file_path": file_path or self.config_file}
            )
            return False

    def diff_config(self, old_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        try:
            with self.lock:
                diff = {}
                for key in set(self.store.flat_config) | set(old_config):
                    old_value = old_config.get(key)
                    new_value = self.store.flat_config.get(key)
                    if old_value != new_value:
                        diff[key] = {"old": old_value, "new": new_value}
                self._log_event("config_diff", "Configuration differences computed", "info", {
                    "changed_keys": list(diff.keys())
                })
                return diff
        except Exception as e:
            self._log_error("Config diff failed", {"error": str(e)})
            return {}

    def register_schema(self, schemas: List[ConfigSchema]) -> None:
        try:
            with self.lock:
                if self._frozen:
                    self._log_error("Cannot register schema: configuration is frozen")
                    return
                self.validator.register(schemas)
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self._load_schema() + schemas)
                self.store.update_cache(self._load_schema() + schemas)
                self._last_config_hash = self._compute_config_hash()
                self._log_event("schema_registered", f"New fields registered", "info", {
                    "new_fields": [s.field for s in schemas],
                    "config_hash": self._last_config_hash
                })
        except Exception as e:
            self._log_error("Failed to register schema", {"error": str(e)})

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "config_file": self.config_file,
                "config": self.store.flat_config,
                "frozen": self._frozen,
                "config_hash": self._last_config_hash
            }

    def load_state(self, state: Dict[str, Any]) -> None:
        try:
            with self.lock:
                self.config_file = state.get("config_file", self.config_file)
                self.store.flat_config = state.get("config", {})
                self._frozen = state.get("frozen", False)
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self._load_schema())
                self.store.update_cache(self._load_schema())
                self._last_config_hash = self._compute_config_hash()
                self._log_event("config_load_state", "Configuration state loaded", "info", {
                    "config_file": self.config_file,
                    "config_hash": self._last_config_hash
                })
        except Exception as e:
            self._log_error("Failed to load config state", {"error": str(e)})
            raise

    def tune(self, **kwargs) -> bool:
        return self.update_batch(kwargs)

    def load_profile(self, profile: str) -> bool:
        profile_file = f"{os.path.splitext(self.config_file)[0]}_{profile}.json"
        try:
            with self.lock:
                config = self.file_handler.load()
                if not config:
                    self._log_error(f"Profile file {profile_file} not found", {"profile_file": profile_file})
                    return False
                self.store.flat_config = config
                self._validate_and_set_defaults()
                self.store.rebuild_structured(self._load_schema())
                self.store.update_cache(self._load_schema())
                self._last_config_hash = self._compute_config_hash()
                self._log_event("profile_load", f"Profile {profile} loaded", "info", {
                    "profile": profile,
                    "config_file": profile_file,
                    "config_hash": self._last_config_hash
                })
                return True
        except Exception as e:
            self._log_error(f"Failed to load profile {profile}", {"error": str(e)})
            return False

    def set_global_blend(self, weight_cap: Optional[float] = None, base_temp: Optional[float] = None) -> bool:
        updates = {}
        prefix = "controls_config."

        if weight_cap is not None and 0.5 <= weight_cap <= 1.0:
            updates[f"{prefix}scaffold_weight_cap"] = weight_cap

        if base_temp is not None and 0.5 <= base_temp <= 1.5:
            updates[f"{prefix}base_temperature"] = base_temp

        return self.update_batch(updates) if updates else True

    def validate_section(self, section: str, required_keys: List[str]) -> bool:
        """(Refactored) Validate required section keys using SchemaValidator."""
        missing_keys = self.validator.required_section_keys_missing(section, required_keys, self.store.structured_config)
        if missing_keys:
            self._log_error(f"Missing required keys in section '{section}': {', '.join(missing_keys)}", {
                "section": section,
                "missing_keys": missing_keys
            })
            return False
        return True

    def tune_parameter(self, section: str, key: str, value: Any, min_value: Any = None, max_value: Any = None) -> bool:
        full_key = f"{section}.{key}"
        try:
            with self.lock:
                if min_value is not None and value < min_value:
                    self._log_error(f"Value {value} below minimum {min_value} for {full_key}", {
                        "section": section,
                        "key": key,
                        "value": value,
                        "min_value": min_value
                    })
                    return False

                if max_value is not None and value > max_value:
                    self._log_error(f"Value {value} above maximum {max_value} for {full_key}", {
                        "section": section,
                        "key": key,
                        "value": value,
                        "max_value": max_value
                    })
                    return False

                success = self.update(full_key, value)
                if success:
                    self._log_event("config_info", f"Tuned {full_key} to {value}", "info", {
                        "section": section,
                        "key": key,
                        "value": value
                    })
                return success
        except Exception as e:
            self._log_error(f"Failed to tune {full_key}", {"section": section, "key": key, "error": str(e)})
            return False

    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                if section not in self.store.structured_config:
                    self._log_error(f"Configuration section '{section}' not found", {"section": section})
                    return False

                batch_updates = {f"{section}.{key}": value for key, value in updates.items()}
                return self.update_batch(batch_updates)
        except Exception as e:
            self._log_error(f"Failed to update section '{section}'", {"section": section, "error": str(e)})
            return False

    def validate_or_raise(self, model_config: Optional[Any] = None) -> None:
        """Validate configuration and raise appropriate errors."""
        try:
            # Validate required keys
            self.validate_keys(self.REQUIRED_KEYS)

            # Validate model-specific configuration if provided
            if model_config and not self.validate_with_model(model_config):
                self.error_manager.record_error(
                    error=ConfigurationError("Model configuration validation failed"),
                    error_type="validation_error",
                    context={"model_config": str(model_config)}
                )
                raise ConfigurationError("Model configuration validation failed")

            # Validate all sections
            for section in self.store.structured_config:
                if not self.validate_section(section, self.REQUIRED_SECTION_KEYS.get(section, [])):
                    self.error_manager.record_error(
                        error=ConfigurationError(f"Section {section} validation failed"),
                        error_type="validation_error",
                        context={"section": section}
                    )
                    raise ConfigurationError(f"Section {section} validation failed")
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="validation_error",
                context={"model_config": str(model_config) if model_config else None}
            )
            raise

    def validate_with_model(self, model_config: Any) -> bool:
        try:
            self.validate_or_raise(model_config)
            return True
        except Exception as e:
            self._log_error("Configuration validation failed", {"error": str(e), "model_config": str(model_config)})
            return False

    def _initialize_memory_config(self) -> None:
        """Initialize memory configuration with new memory management structure."""
        # Initialize RAM configuration
        self.update("memory_config.ram", {
            "max_ram_mb": self.get("memory_config.max_ram_mb", 2048),
            "ram_threshold": self.get("memory_config.ram_threshold", 0.8),
            "enable_ram_compression": self.get("memory_config.enable_ram_compression", True),
            "ram_compression_ratio": self.get("memory_config.ram_compression_ratio", 0.6),
            "max_compressed_ram_mb": self.get("memory_config.max_compressed_ram_mb", 4096)
        })

        # Initialize GPU memory configuration
        self.update("memory_config.gpu", {
            "max_gpu_memory_mb": self.get("memory_config.max_gpu_memory_mb", 1024),
            "gpu_memory_threshold": self.get("memory_config.gpu_memory_threshold", 0.85),
            "enable_gpu_memory_compression": self.get("memory_config.enable_gpu_memory_compression", True),
            "gpu_compression_ratio": self.get("memory_config.gpu_compression_ratio", 0.7),
            "max_compressed_gpu_memory_mb": self.get("memory_config.max_compressed_gpu_memory_mb", 2048)
        })

        # Initialize memory manager configuration
        self.update("memory_config.manager", {
            "enable_ram_manager": self.get("memory_config.enable_ram_manager", True),
            "enable_gpu_memory_manager": self.get("memory_config.enable_gpu_memory_manager", True),
            "memory_sync_interval": self.get("memory_config.memory_sync_interval", 60),
            "enable_memory_monitoring": self.get("memory_config.enable_memory_monitoring", True),
            "memory_monitoring_interval": self.get("memory_config.memory_monitoring_interval", 5)
        })

    def _initialize_error_manager(self) -> None:
        """Initialize error manager with config-specific settings."""
        self.error_manager = ErrorManager(
            context=self,
            state_tracker=None,  # Config doesn't need state tracking
            config_manager=self,
            error_cooldown=1.0
        )
        
        # Register config-specific error thresholds
        self.error_manager.severity_thresholds.update({
            "config_load": 3,     # 3 load failures before critical
            "config_save": 3,     # 3 save failures before critical
            "validation": 5,      # 5 validation failures before critical
            "schema": 2,          # 2 schema failures before critical
            "update": 5           # 5 update failures before critical
        })
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
    def _register_recovery_strategies(self) -> None:
        """Register configuration-specific error recovery strategies."""
        self.error_manager.recovery_strategies.update({
            "config_load_error": self._recover_config_load,
            "config_save_error": self._recover_config_save,
            "validation_error": self._recover_validation,
            "schema_error": self._recover_schema,
            "update_error": self._recover_update
        })
        
    def _recover_config_load(self, record: ErrorRecord) -> None:
        """Recovery strategy for config loading errors."""
        try:
            # Reset to default configuration
            self._initialize_config()
            # Attempt to load from backup if available
            self._load_from_backup()
            self.logger.record_event(
                event_type="config_recovery",
                message="Recovered from config load error",
                level="info"
            )
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="recovery_error",
                context={"original_error": record.error_type}
            )
            
    def _recover_config_save(self, record: ErrorRecord) -> None:
        """Recovery strategy for config saving errors."""
        try:
            # Create backup of current config
            self._create_backup()
            # Retry save with compression
            self.save_config(compress=True)
            self.logger.record_event(
                event_type="config_recovery",
                message="Recovered from config save error",
                level="info"
            )
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="recovery_error",
                context={"original_error": record.error_type}
            )
            
    def _recover_validation(self, record: ErrorRecord) -> None:
        """Recovery strategy for validation errors."""
        try:
            # Reset affected section to defaults
            section = record.context.get("section")
            if section:
                self._reset_section_to_defaults(section)
            self.logger.record_event(
                event_type="config_recovery",
                message="Recovered from validation error",
                level="info"
            )
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="recovery_error",
                context={"original_error": record.error_type}
            )
            
    def _recover_schema(self, record: ErrorRecord) -> None:
        """Recovery strategy for schema errors."""
        try:
            # Reload schema
            self._load_schema()
            # Revalidate all config
            self._validate_and_set_defaults()
            self.logger.record_event(
                event_type="config_recovery",
                message="Recovered from schema error",
                level="info"
            )
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="recovery_error",
                context={"original_error": record.error_type}
            )
            
    def _recover_update(self, record: ErrorRecord) -> None:
        """Recovery strategy for update errors."""
        try:
            # Rollback the failed update
            if "updates" in record.context:
                self._rollback_updates(record.context["updates"])
            self.logger.record_event(
                event_type="config_recovery",
                message="Recovered from update error",
                level="info"
            )
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="recovery_error",
                context={"original_error": record.error_type}
            )
            
    def _load_from_backup(self) -> None:
        """Attempt to load configuration from backup file."""
        backup_file = f"{self.config_file}.backup"
        if os.path.exists(backup_file):
            try:
                self.file_handler = FileHandler(backup_file, self.logger)
                self._initialize_config()
            except Exception as e:
                self.error_manager.record_error(
                    error=e,
                    error_type="backup_load_error",
                    context={"backup_file": backup_file}
                )
                
    def _create_backup(self) -> None:
        """Create a backup of the current configuration using a temp file and atomic rename."""
        backup_file = f"{self.config_file}.backup"
        temp_dir = os.path.dirname(self.config_file)
        try:
            with open(self.config_file, 'r') as src:
                with tempfile.NamedTemporaryFile('w', dir=temp_dir, delete=False) as tmp:
                    shutil.copyfileobj(src, tmp)
                    temp_backup = tmp.name
            os.replace(temp_backup, backup_file)  # Atomic rename
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="backup_creation_error",
                context={"backup_file": backup_file}
            )
            
    def _reset_section_to_defaults(self, section: str) -> None:
        """Reset a configuration section to its default values."""
        try:
            defaults = self.validator.get_defaults_for_section(section)
            for key, value in defaults.items():
                self.store.set_value(f"{section}.{key}", value)
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="section_reset_error",
                context={"section": section}
            )
            
    def _rollback_updates(self, updates: Dict[str, Any]) -> None:
        """Rollback failed configuration updates."""
        try:
            for key, old_value in updates.items():
                self.store.set_value(key, old_value)
        except Exception as e:
            self.error_manager.record_error(
                error=e,
                error_type="rollback_error",
                context={"updates": updates}
            )

    def get_config(self) -> dict:
        """
        Return the full structured config as a dict for CLI and other consumers.
        """
        with self.lock:
            import copy
            return copy.deepcopy(self.store.structured_config)

if __name__ == "__main__":
    from sovl_logger import LoggerConfig
    logger = Logger(LoggerConfig())
    config_manager = ConfigManager("sovl_config.json", logger)
    try:
        config_manager.validate_keys(["core_config.model_name", "curiosity_config.attention_weight"])
        print("Schema validation successful")
    except ValueError as e:
        print(f"Validation error: {e}")
