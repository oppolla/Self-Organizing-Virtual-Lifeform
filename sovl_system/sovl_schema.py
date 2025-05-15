from pydantic import BaseModel, Field
from typing import Optional, List, Dict

"""
SOVL Config Schema (Pydantic-based)

This module defines the configuration schema for the SOVL system using Pydantic models.
Each config section should be represented as a Pydantic BaseModel.
Incrementally add config sections as you migrate from the old schema.
"""

# Example placeholder for a config section (remove or replace as you migrate sections)
# class CoreConfig(BaseModel):
#     base_model_name: str
#     ...

# Top-level config model placeholder (expand as you add sections)
# class SOVLConfig(BaseModel):
#     core_config: CoreConfig
#     ...

# Used by: TirednessManager, SOVLSystem (sovl_main.py)
class GestationConfig(BaseModel):
    tiredness_threshold: float = Field(0.7, ge=0.0, le=1.0)  # Threshold above which the system is considered tired
    tiredness_check_interval: int = Field(10, ge=1, le=3600)  # Interval (seconds) between tiredness checks
    tiredness_decay_k: float = Field(0.01, ge=0.0001, le=1.0)  # Decay constant for tiredness as a function of data exposure
    sleep_log_min: int = Field(10, ge=1, le=10000)  # Minimum number of log entries before sleep is considered
    gestation_countdown_seconds: int = Field(30, ge=1, le=600)  # Countdown (seconds) before gestation begins after being triggered
    tiredness_weights: Dict[str, float] = Field(default_factory=lambda: {"log": 0.4, "confidence": 0.3, "time": 0.3})  # Weights for log, confidence, and time in tiredness calculation
    min_awake_seconds: int = Field(60, ge=1, le=86400)  # Minimum time (seconds) the system must stay awake between gestation cycles
    max_awake_seconds: int = Field(7200, ge=1, le=86400)  # Maximum time (seconds) the system can stay awake before forced gestation
    post_abort_cooldown_seconds: int = Field(120, ge=1, le=86400)  # Cooldown (seconds) after a gestation abort before another can be triggered
    dream_after_gestation: bool = True  # Whether to run a dream cycle after gestation completes

# Used by: SOVLOrchestrator (sovl_conductor.py)
class OrchestratorConfig(BaseModel):
    log_max_size_mb: int = 10  # Maximum log file size in megabytes
    save_path_suffix: str = "_final.json"  # Suffix for saved state files
    enable_logging: bool = True  # Enable or disable orchestrator logging
    enable_state_saving: bool = True  # Enable or disable periodic state saving
    state_save_interval: int = 300  # Interval (seconds) between state saves
    max_backup_files: int = 5  # Maximum number of backup files to keep

# Used by: EventDispatcher, StateEventDispatcher, MemoryEventDispatcher (sovl_events.py), Logger (sovl_logger.py)
class LoggingConfig(BaseModel):
    log_file: str = "sovl_events.log"  # Path to the log file
    max_size_mb: int = 10  # Maximum log file size in megabytes before rotation
    compress_old: bool = False  # Whether to compress old log files after rotation

# Used by: EventDispatcher (sovl_events.py), possibly other modules
class ControlsConfig(BaseModel):
    monitor_deadlocks: bool = True  # Enable or disable deadlock monitoring for event locks

# Used by: SystemMonitor, MemoryMonitor, TraitsMonitor (sovl_monitor.py)
class MonitoringConfig(BaseModel):
    bond_history_maxlen: int = 30  # Max bond score history per user
    ram_critical_threshold_percent: float = 90.0  # RAM usage percent threshold for critical alert
    gpu_critical_threshold_percent: float = 95.0  # GPU usage percent threshold for critical alert
    scaffold_confidence_threshold: float = 0.6  # Scaffold mapping confidence threshold
    scaffold_fallback_rate_threshold: float = 0.2  # Scaffold fallback rate threshold
    poll_interval: float = 10.0  # Polling interval for monitor loops (seconds)
    trait_history_size: int = 100  # Number of samples to keep for trait history
    trait_variance_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        'curiosity': 0.3,
        'confidence': 0.25,
        'lifecycle': 0.2,
        'temperament': 0.35
    })  # Variance thresholds for erratic trait detection
    min_samples_for_variance: int = 10  # Minimum samples for variance calculation
    ram_critical_usage_mb: int = 8192  # RAM usage (MB) threshold for critical alert
    gpu_critical_usage_mb: int = 4096  # GPU usage (MB) threshold for critical alert

# Used by: RAMManager (sovl_memory.py)
class RAMConfig(BaseModel):
    memory_threshold: float = 0.85  # RAM usage threshold (0.0-1.0) before triggering recovery
    memory_decay_rate: float = 0.95  # Decay rate for RAM memory
    max_batch_size: int = 32  # Maximum batch size for RAM operations
    initial_batch_size: int = 8  # Initial batch size for RAM operations
    batch_size: int = 8  # Current batch size (may be updated dynamically)

# Used by: GPUMemoryManager (sovl_memory.py)
class GPUConfig(BaseModel):
    gpu_threshold: float = 0.85  # GPU usage threshold (0.0-1.0) before triggering recovery
    gpu_decay_rate: float = 0.95  # Decay rate for GPU memory
    max_gpu_memory_mb: int = 1024  # Maximum GPU memory in MB

# Used by: MetadataProcessor (sovl_processor.py)
class MetadataProcessorConfig(BaseModel):
    state_cache_ttl: float = 2.0  # Time-to-live for state cache (seconds)

# Used by: ScribeIngestionProcessor, memory/journal processing
class ScribedConfig(BaseModel):
    output_path: str = "scribe/sovl_scribe.jsonl"  # Path to scribe journal file
    max_file_size_mb: int = 50  # Max file size in MB before rotation
    buffer_size: int = 10  # Buffer size for scribe writes

# Used by: ScribeIngestionProcessor, event weighting
class TrainerWeightingConfig(BaseModel):
    # This is a mapping of every possible metadata field/event type to a float weight
    __root__: Dict[str, float] = Field(
        default_factory=lambda: {
            "internal_error_reflection": 0.2,
            "user_message": 1.0,
            "system_message": 0.5,
            "dream_memory": 0.3,
            "origin": 0.0,
            "timestamp_unix": 0.0,
            "session_id": 0.0,
            "sovl_version": 0.0,
            "current_lifecycle_stage": 0.01,
            "current_temperament_score": 0.03,
            "current_mood_label": 0.03,
            "current_memory_usage": 0.0,
            "user_id": 0.0,
            "level": 0.0,
            "stack_trace": 0.0,
            "module": 0.0,
            "generation_config": 0.0,
            "model_name": 0.0,
            "device": 0.0,
            "internal_call": 0.0,
            "request_timestamp_unix": 0.0,
            "initial_kwargs": 0.0,
            "generation_config_used": 0.0,
            "temperament_score": 0.03,
            "lifecycle_stage": 0.01,
            "novelty_score": 0.20,
            "memory_usage_mb": 0.0,
            "processing_time_ms": -0.01,
            "scaffold_index": 0.0,
            "input_length": 0.02,
            "output_length": 0.02,
            "memory_usage": 0.0,
            "generation_time": -0.01,
            "system_device": 0.0,
            "pressure": 0.0,
            "threshold": 0.0,
            "<field>_metrics.content_metrics.word_count": 0.05,
            "<field>_metrics.content_metrics.sentence_count": 0.01,
            "<field>_metrics.content_metrics.avg_word_length": 0.01,
            "<field>_metrics.content_metrics.avg_sentence_length": 0.01,
            "<field>_metrics.quality_metrics.has_code": 0.03,
            "<field>_metrics.quality_metrics.has_url": 0.01,
            "<field>_metrics.quality_metrics.has_question": 0.02,
            "<field>_metrics.quality_metrics.has_exclamation": 0.01,
            "<field>_metrics.quality_metrics.has_emoji": 0.01,
            "<field>_token_stats.basic_stats.total_tokens": 0.01,
            "<field>_token_stats.basic_stats.unique_tokens": 0.03,
            "<field>_token_stats.basic_stats.token_diversity": 0.07,
            "<field>_token_stats.pattern_stats.unique_bigrams": 0.05,
            "<field>_token_stats.pattern_stats.unique_trigrams": 0.05,
            "<field>_token_stats.pattern_stats.bigram_diversity": 0.03,
            "<field>_token_stats.pattern_stats.trigram_diversity": 0.03,
            "<field>_token_stats.special_token_stats.special_token_count": 0.0,
            "<field>_token_stats.special_token_stats.special_token_ratio": 0.0,
            "<field>_token_stats.special_token_stats.special_token_types": 0.0,
            "<field>_structure.length_metrics.character_count": 0.0,
            "<field>_structure.length_metrics.word_count": 0.01,
            "<field>_structure.length_metrics.line_count": 0.0,
            "<field>_structure.length_metrics.sentence_count": 0.0,
            "<field>_structure.length_metrics.avg_sentence_length": 0.0,
            "<field>_structure.length_metrics.avg_line_length": 0.0,
            "<field>_structure.whitespace_metrics.blank_line_count": 0.0,
            "<field>_structure.whitespace_metrics.indentation_levels": 0.0,
            "<field>_structure.whitespace_metrics.whitespace_ratio": 0.0,
            "performance_metrics.timing.generation_time_ms": -0.01,
            "performance_metrics.timing.tokens_per_second": 0.01,
            "performance_metrics.timing.total_processing_time": -0.01,
            "performance_metrics.memory.ram_mb": 0.0,
            "performance_metrics.memory.gpu_mb": 0.0,
            "performance_metrics.memory.peak_memory": 0.0,
            "performance_metrics.efficiency.memory_efficiency": 0.01,
            "performance_metrics.efficiency.tokens_per_mb": 0.01,
            "performance_metrics.efficiency.optimization_level": 0.0,
            "relationship_context.conversation_tracking.conversation_id": 0.0,
            "relationship_context.conversation_tracking.message_index": 0.0,
            "relationship_context.conversation_tracking.thread_depth": 0.0,
            "relationship_context.reference_tracking.references": 0.0,
            "relationship_context.reference_tracking.parent_message_id": 0.0,
            "relationship_context.reference_tracking.root_message_id": 0.0,
            "relationship_context.temporal_tracking.timestamp": 0.0,
            "relationship_context.temporal_tracking.elapsed_time": 0.0,
            "relationship_context.temporal_tracking.session_duration": 0.0
        }
    )

# Used by: ScribeIngestionProcessor, event weighting
class EventTypeWeightsConfig(BaseModel):
    __root__: Dict[str, float] = Field(
        default_factory=lambda: {
            "error_message": 0.2,
            "user_interaction": 1.0,
            "curiosity_question": 0.8,
            "curiosity_question_user": 2.0,
            "dream": 1.5,
            "meditation": 0.9,
            "traumatic_memory": 10.0, 
            # Add all other event types as needed
        }
    )

# Used by: JSONLLoader, load_and_split_data (sovl_io.py)
class IOConfig(BaseModel):
    field_mapping: Dict[str, str] = Field(
        default_factory=lambda: {"prompt": "prompt", "completion": "completion"}
    )  # Mapping of input fields to output fields
    required_fields: List[str] = Field(
        default_factory=lambda: ["prompt", "completion"]
    )  # Fields required in each entry
    min_string_length: int = 1  # Minimum allowed string length for fields
    max_string_length: int = 10000  # Maximum allowed string length for fields
    strict_validation: bool = False  # If True, fail on any validation error
    shuffle_data: bool = True  # Shuffle data before splitting (used in data splitting)

# Used by: JsonlWriter (sovl_io.py)
class ScribedConfig(BaseModel):
    log_path: str = "logs/sovl_scribed.jsonl"  # Path to scribe JSONL log file
    max_file_size_mb: int = 50  # Max file size in MB before rotation
    buffer_size: int = 10  # Number of entries to buffer before writing

# Used by: load_and_split_data (sovl_io.py), and possibly elsewhere
class CoreConfig(BaseModel):
    random_seed: int = 42  # Random seed for reproducibility

class ControlsConfig(BaseModel):
    scaffold_unk_id: Optional[int] = None  # Tokenizer unknown token ID
    use_token_map_memory: bool = True  # Use token map memory for scaffold
    dynamic_cross_attn_mode: Optional[Any] = None  # Dynamic cross-attention mode
    max_generation_retries: int = 3  # Max retries for generation failures
    base_batch_size: int = 1  # Default batch size for generation
    memory_threshold: float = 0.85  # Memory usage threshold for adaptive behavior
    max_cache_size: int = 1000  # Max size for embedding/tensor cache
    base_temperature: float = 0.7  # Default temperature for generation
    top_k: int = 50  # Top-k sampling for generation
    top_p: float = 0.95  # Top-p (nucleus) sampling for generation
    max_new_tokens: int = 100  # Max new tokens to generate
    enable_repetition_check: Optional[bool] = None  # Enable repetition check (if used)
    conversation_history_maxlen: Optional[int] = None  # Max conversation history length
    memory_decay_rate: Optional[float] = None  # Memory decay rate (if used)
    enable_error_listening: Optional[bool] = None  # Enable error listening (if used)
    dream_memory_weight: Optional[float] = None  # Dream memory weighting (if used)

class GenerationConfig(BaseModel):
    min_batch_size: int = 1  # Minimum batch size for generation
    max_batch_size: int = 8  # Maximum batch size for generation
    default_batch_size: int = 1  # Default batch size for generation
    mem_per_sample_mb: int = 100  # Estimated memory per sample in MB

class TrainingConfig(BaseModel):
    max_seq_length: int = 1024  # Maximum sequence length for training
    batch_size: int = 8  # Training batch size
    dry_run_params: Dict[str, Any] = Field(default_factory=dict)  # Placeholder for dry run parameters

class CuriosityConfig(BaseModel):
    enable_curiosity: bool = True  # Enable or disable curiosity system
    weight_ignorance: float = 0.7  # Weight for ignorance in curiosity score
    weight_novelty: float = 0.3  # Weight for novelty in curiosity score
    metrics_maxlen: int = 1000  # Max length for metrics/history deques
    novelty_history_maxlen: int = 1000  # Max length for novelty history
    embedding_cache_maxlen: int = 1000  # Max length for embedding cache
    embedding_cache_prune_batch: int = 100  # Batch size for pruning embedding cache
    embedding_cache_backup_enabled: bool = False  # Enable backup of pruned embeddings
    embedding_cache_backup_path: str = "embedding_cache_backup.jsonl"  # Path for embedding cache backup
    background_pruning_enabled: bool = True  # Enable background pruning of embedding cache
    similarity_early_exit_threshold: float = 0.99  # Early exit threshold for similarity checks
    adaptive_batch_min: int = 8  # Minimum adaptive batch size
    adaptive_batch_max: int = 128  # Maximum adaptive batch size

    # Curiosity pressure system
    base_pressure: float = 0.5  # Base pressure value
    max_pressure: float = 1.0  # Maximum pressure value
    min_pressure: float = 0.0  # Minimum pressure value
    decay_rate: float = 0.1  # Decay rate for pressure
    confidence_adjustment: float = 0.5  # Confidence adjustment factor
    eruption_cooldown: float = 30.0  # Cooldown (seconds) between eruptions
    pressure_threshold: float = 0.5  # Threshold for pressure eruption
    pressure_drop: float = 0.3  # Amount to drop pressure after eruption

    # Internal curiosity question buffering
    curiosity_threshold: float = 0.5  # Threshold for curiosity score to trigger question
    internal_threshold_factor: float = 0.75  # Factor for internal threshold
    max_internal_questions: int = 20  # Max number of internal questions to buffer
    internal_decay_seconds: int = 3600  # Time (seconds) to keep internal questions

    # Generation parameters (optional, may be set elsewhere)
    base_temperature: Optional[float] = None
    temperament_influence: Optional[float] = None
    max_new_tokens: Optional[int] = None
    top_k: Optional[int] = None
    novelty_threshold_response: Optional[float] = None
    novelty_threshold_spontaneous: Optional[float] = None
    min_temperature: float = 0.7
    max_temperature: float = 1.7

class TemperamentConfig(BaseModel):
    mood_influence: float = 0.3  # Influence of mood on temperament
    history_maxlen: int = 5  # Max length of mood/temperament history
    temp_eager_threshold: float = 0.7  # Threshold for eager temperament
    temp_sluggish_threshold: float = 0.3  # Threshold for sluggish temperament
    temp_mood_influence: float = 0.3  # Influence of mood on temp
    temp_restless_drop: float = 0.2  # Drop in temp for restlessness
    temp_melancholy_noise: float = 0.02  # Noise for melancholy
    conf_feedback_strength: float = 0.5  # Strength of confidence feedback
    temp_smoothing_factor: float = 0.5  # Smoothing factor for temp
    temperament_decay_rate: float = 0.9  # Decay rate for temperament
    temperament_history_maxlen: int = 5  # Max length of temperament history
    temperament_pressure_threshold: float = 0.5  # Pressure threshold for temperament
    temperament_max_pressure: float = 1.0  # Max pressure for temperament
    temperament_min_pressure: float = 0.0  # Min pressure for temperament
    temperament_pressure_drop: float = 0.2  # Pressure drop after threshold
    mood_smoothing: float = 0.8  # Smoothing for mood updates
    mood_cautious_threshold: float = -0.3  # Threshold for cautious mood label
    mood_curious_threshold: float = 0.3  # Threshold for curious mood label

    # Pressure system (TemperamentPressure)
    pressure_sensitivity: float = 0.1
    pressure_decay: float = 0.01
    pressure_high_threshold: float = 0.8
    pressure_low_threshold: float = 0.2
    pressure_eruption_cooldown: float = 30.0
    pressure_frustration_rebound: float = 0.4
    pressure_joy_rebound: float = 0.6

# Used by: ConfidenceCalculator (sovl_confidence.py)
class ConfidenceConfig(BaseModel):
    min_confidence: float = 0.0  # Minimum allowed confidence score
    max_confidence: float = 1.0  # Maximum allowed confidence score
    default_confidence: float = 0.5  # Default confidence score if calculation fails
    min_history_length: int = 3  # Minimum history length for recovery/validation

# Used by: BondCalculator, BondModulator (sovl_bonder.py)
class BondingConfig(BaseModel):
    strong_bond_threshold: float = 0.8  # Threshold for strong bond
    weak_bond_threshold: float = 0.3  # Threshold for weak bond
    default_bond_score: float = 0.5  # Default bond score
    min_bond_score: float = 0.0  # Minimum bond score
    max_bond_score: float = 1.0  # Maximum bond score
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "curiosity": 0.25,
            "stability": 0.25,
            "coherence": 0.25,
            "personalized": 0.25
        }
    )  # Weights for bond components
    max_signature_metadata: int = 30  # Max metadata entries for signature
    state_sync_interval: float = 2.0  # Interval (seconds) for state sync
    signature_similarity_threshold: float = 0.15  # Similarity threshold for signature matching
    max_identified_users: int = 10000  # Max number of identified users (LRU cap)
    archived_retention_days: int = 365  # Days to retain archived users
    signature_history_maxlen: int = 100  # Max length of signature history
    signature_history_maxdays: int = 180  # Max days for signature history retention
    drift_threshold: float = 0.25  # Threshold for drift detection
    drift_consecutive: int = 5  # Consecutive drifts before action
    archive_timeout_days: int = 365  # Days before archiving dormant profiles
    decay_lambda: float = 0.01  # Decay lambda for time-weighted averages

# Used by: BondCalculator (sovl_bonder.py)
class BondConfig(BaseModel):
    max_interactions: int = 100  # Max interactions for knowing score
    max_session_time: float = 3600.0  # Max session time for knowing score
    decay_rate: float = 0.95  # Decay rate for inactivity
    decay_interval: float = 86400.0  # Interval (seconds) for decay
    max_expected_dev: float = 20.0  # Max expected deviation for style consistency

class IntrospectionUnifiedConfig(BaseModel):
    min_topic_duration: int = 30  # Minimum seconds a topic must span to trigger introspection
    min_topic_messages: int = 3   # Minimum user messages in a topic window
    min_topic_words: int = 100    # Minimum total words in a topic window
    topic_time_window: int = 600  # Time window (seconds) for topic engagement

class IntrospectionConfig(BaseModel):
    enable: bool = True  # Master switch for all introspection
    min_curiosity_trigger: float = 0.7  # Curiosity score threshold to trigger introspection
    max_confidence_trigger: float = 0.4  # Max confidence below which introspection may trigger
    triggering_moods: List[str] = Field(default_factory=lambda: ["cautious", "melancholy"])  # Moods that can trigger introspection
    cooldown_seconds: int = 30  # Minimum seconds between introspection triggers
    base_approval_threshold: float = 0.6  # Default threshold for approving introspection results
    status_phrases: List[str] = Field(default_factory=lambda: [
        "Processing...", "Considering carefully...", "Reviewing perspectives...", "Evaluating options..."
    ])  # Status messages shown during introspection
    debug_mode: bool = False  # Enable verbose logging and debug output
    batch_size: int = 4  # Number of questions/insights to process in a batch
    topic_window_messages: int = 15  # Number of recent messages to consider for topic engagement
    time_window_seconds: int = 600  # Time window for topic engagement (seconds)
    dialogue_maxlen: int = 100  # Max number of introspection dialogues to keep in memory/history
    introspect_min_interval: float = 0.5  # Minimum interval (seconds) between introspect checks
    idle_seconds: int = 60  # Idle time (seconds) before introspection can trigger

    unified: IntrospectionUnifiedConfig = Field(default_factory=IntrospectionUnifiedConfig)  # Unified trigger system config

    # Technique-agnostic parameters
    followup_depth: int = 3  # Default number of follow-up questions for recursive introspection
    max_followup_depth: int = 4  # Maximum allowed follow-up depth
    confidence_threshold: Optional[float] = None  # Confidence required to stop follow-ups (if set)

    # Deep Study technique
    deep_study_max_depth: int = 5  # Max follow-up depth for deep study
    deep_study_min_curiosity: float = 0.8  # Minimum curiosity to trigger deep study

    # Relational technique
    relational_max_depth: int = 3  # Max follow-up depth for relational introspection
    relational_min_sentiment_threshold: float = 0.7  # Minimum sentiment to trigger relational
    relational_confidence_threshold: float = 0.8  # Confidence threshold for relational
    relational_bond_adjustment_factor: float = 0.1  # How much to adjust bond on approval

    # Creative technique
    creative_max_depth: int = 3  # Max follow-up depth for creative introspection
    creative_min_curiosity: float = 0.8  # Minimum curiosity to trigger creative
    creative_confidence_threshold: float = 0.8  # Confidence threshold for creative

    # Foresight technique
    foresight_max_depth: int = 3  # Max follow-up depth for foresight introspection
    foresight_confidence_threshold: float = 0.8  # Confidence threshold for foresight

    # Personal technique
    personal_max_depth: int = 2  # Max follow-up depth for personal introspection
    personal_confidence_threshold: float = 0.8  # Confidence threshold for personal

    # Journey technique
    journey_max_memories: int = 20  # Max number of memories to retrieve for journey introspection
    journey_confidence_threshold: float = 0.8  # Confidence threshold for journey

    # Optional: allow overriding prompt templates via config
    introspection_prompt_template: Optional[str] = None  # Custom prompt template for introspection
    followup_prompt_template: Optional[str] = None  # Custom prompt template for follow-ups

    # Optional: enable/disable specific techniques
    enable_ethical: bool = True  # Enable ethical introspection
    enable_deep_study: bool = True  # Enable deep study introspection
    enable_relational: bool = True  # Enable relational introspection
    enable_creative: bool = True  # Enable creative introspection
    enable_foresight: bool = True  # Enable foresight introspection
    enable_personal: bool = True  # Enable personal introspection
    enable_journey: bool = True  # Enable journey introspection

    # Optional: logging/async/batch tuning
    log_level: Optional[str] = None  # Log level for introspection events
    batch_flush_interval: Optional[float] = None  # Interval (seconds) for flushing batch updates
    max_pending_introspections: Optional[int] = None  # Max number of pending introspections to buffer

    # For future extensibility
    extra: Dict[str, Any] = Field(default_factory=dict)  # Extra config for future use

class VibeConfig(BaseModel):
    history_maxlen: int = 20  # Max number of vibe profiles to keep in history
    default_vibe_score: float = 0.5  # Default vibe score when initializing
    min_vibe_score: float = 0.0  # Minimum allowed vibe score
    max_vibe_score: float = 1.0  # Maximum allowed vibe score
    switch_threshold: float = 0.3  # Threshold for detecting a vibe shift
    decay_factor: float = 0.9  # Decay factor for vibe score over time
    energy_weight: float = 0.25  # Weight for energy in overall vibe calculation
    flow_weight: float = 0.25  # Weight for flow in overall vibe calculation
    resonance_weight: float = 0.25  # Weight for resonance in overall vibe calculation
    engagement_weight: float = 0.25  # Weight for engagement in overall vibe calculation
    repetition_threshold: float = 0.8  # Similarity threshold for detecting repetition
    repetition_factor_min: float = 0.2  # Minimum repetition factor
    repetition_factor_max: float = 0.8  # Maximum repetition factor
    extremity_weight: float = 0.5  # Weight for extreme streaks in decay calculation
    balance_force: float = 0.1  # Force applied for homeostatic balancing
    coupling_factor: float = 0.05  # Coupling between vibe components
    vibe_lower_amount: float = 0.3  # Amount to lower vibe on shame/anger event (default 0.3)
    vibe_lower_cooldown_turns: int = 5  # Number of turns to wait before allowing another vibe drop

# Used by: AspirationSystem, AspirationManager (sovl_striver.py)
class AspirationConfig(BaseModel):
    n_recent: int = 50  # Number of recent interactions to consider for self-assessment and doctrine generation
    days_window: int = 7  # How many days back to look for relevant long-term memories
    max_tokens: int = 1024  # Maximum number of tokens for the LLM prompt
    strong_vibe_threshold: float = 0.1  # Minimum deviation from neutral for a "strong" vibe
    moderate_vibe_threshold: float = 0.05  # Minimum deviation for a "moderate" vibe
    min_intensity_strong: float = 0.4  # Minimum intensity for a "strong" vibe
    min_intensity_moderate: float = 0.2  # Minimum intensity for a "moderate" vibe
    doctrine_fallback: str = "Be open to new experiences."  # Default doctrine if LLM generation fails

# Used by: ScaffoldTokenMapper, CrossAttentionLayer, ScaffoldProvider (sovl_scaffold.py)
class ScaffoldConfig(BaseModel):
    max_tokens_per_mapping: int = 3  # Maximum scaffold tokens per base token
    mapping_similarity_threshold: float = 0.7  # Similarity threshold for alternative mappings
    conflict_resolution_strategy: str = "keep_highest_conf"  # Strategy for resolving mapping conflicts
    min_token_map_confidence: float = 0.5  # Minimum confidence for accepting a token map
    max_low_conf_ratio: float = 0.2  # Max ratio of low-confidence mappings allowed
    max_fallback_ratio: float = 0.3  # Max ratio of fallback mappings allowed
    max_drift: float = 0.9  # Maximum allowed drift in token mapping
    cosine_weight: float = 0.5  # Weight for cosine similarity in mapping
    euclidean_weight: float = 0.3  # Weight for euclidean distance in mapping
    norm_weight: float = 0.2  # Weight for norm difference in mapping
    levenshtein_weight: float = 0.4  # Weight for Levenshtein distance in mapping
    char_weight: float = 0.3  # Weight for character similarity in mapping
    subword_weight: float = 0.2  # Weight for subword similarity in mapping
    freq_weight: float = 0.1  # Weight for frequency in mapping
    drift_cache_size: int = 10000  # Size of drift cache
    token_mapping_fallback_order: list = Field(default_factory=lambda: [
        "levenshtein", "subword", "char", "split", "merge", "nearest", "unk"
    ])  # Order of fallback strategies for token mapping
    attention_chunk_size: int = 128  # Chunk size for attention computation
    gpu_memory_threshold: float = 0.85  # GPU memory usage threshold
    token_mapping: dict = Field(default_factory=dict)  # Populated at runtime
    attention_config: dict = Field(default_factory=dict)  # Populated at runtime
    memory_config: dict = Field(default_factory=dict)  # Populated at runtime

# Used by: LoraAdapterManager (sovl_engram.py)
class EngramLoraConfig(BaseModel):
    lora_rank: int = 8  # LoRA rank (dimension of adaptation matrices)
    lora_alpha: int = 16  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.1  # LoRA dropout rate

# Used by: LoraAdapterManager (sovl_engram.py)
class LoraConfig(BaseModel):
    target_modules: list = Field(default_factory=lambda: ["q_proj", "v_proj"])  # Modules to adapt
    task_type: str = "CAUSAL_LM"  # Task type for LoRA/PEFT ("CAUSAL_LM", "SEQ_CLS", etc.)

# Used by: ModelManager (sovl_manager.py)
class ModelConfig(BaseModel):
    base_model_name: str  # Name or path of the base model
    scaffold_model_name: Optional[str] = None  # (Legacy) Name or path of a single scaffold model
    scaffold_model_names: Optional[List[str]] = None  # List of scaffold model names/paths (preferred)
    quantization_mode: str = "fp16"  # Quantization mode: "int4", "int8", or "fp16"

# Used by: DialogueContextManager, ShortTermMemory, LongTermMemory (sovl_recaller.py)
class MemoryConfig(BaseModel):
    max_short_term: int = 50  # Max messages in short-term memory
    short_term_expiry_seconds: Optional[int] = None  # Expiry for short-term memory messages (seconds)
    embedding_dim: int = 128  # Embedding dimension for memory
    db_path: str = "conversations.db"  # Path to SQLite DB for long-term memory
    long_term_retention_days: Optional[int] = None  # Retention window for long-term memory (days)
    long_term_top_k: int = 5  # Top-K results for long-term memory queries
    memory_logging_level: str = "info"  # Logging level for memory operations
    long_term_max_records: int = 10000  # Max records in long-term memory
    faiss_rebuild_threshold: int = 100  # How often to rebuild FAISS index (num new records)
    default_origin: str = "dialogue_manager"  # Default origin for messages
    default_session_id: str = "default"  # Default session ID
    default_user_id: str = "default"  # Default user ID

# Used by: ShameManager (sovl_shamer.py)
class ApologyConfig(BaseModel):
    direct_apology_weight: float = Field(0.4, ge=0.0, le=1.0)
    casual_apology_weight: float = Field(0.3, ge=0.0, le=1.0)
    defensive_apology_weight: float = Field(0.2, ge=0.0, le=1.0)
    reconciliation_weight: float = Field(0.25, ge=0.0, le=1.0)
    tentative_apology_weight: float = Field(0.15, ge=0.0, le=1.0)
    politeness_marker_weight: float = Field(0.05, ge=0.0, le=1.0)
    direct_apology_threshold: float = Field(0.4, ge=0.0, le=1.0)
    tentative_apology_threshold: float = Field(0.6, ge=0.0, le=1.0)

# Used by: TrainingConfig, TrainingWorkflowManager (sovl_trainer.py)
class TrainingConfigSchema(BaseModel):
    # Optimizer
    optimizer_type: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler_type: str = "linear"
    warmup_steps: int = 0
    total_steps: int = 100000
    cosine_min_lr: float = 1e-6
    warmup_ratio: float = 0.1

    # Memory
    batch_size: int = 1
    max_seq_length: int = 512
    use_amp: bool = False
    max_patience: int = 2

    # Training params
    max_epochs: int = 3
    validate_every_n_steps: int = 500
    checkpoint_interval: int = 5000
    checkpoint_path: str = "checkpoints/sovl_trainer"
    dropout_rate: float = 0.1
    metrics_to_track: list = Field(default_factory=lambda: ["loss", "accuracy", "confidence"])

    # Logging
    log_file: str = "training_logs.jsonl"
    max_size_mb: int = 10
    compress_old: bool = True
    max_in_memory_logs: int = 100
    rotation_count: int = 5
    max_log_age_days: int = 30
    prune_interval_hours: int = 48
    memory_threshold_mb: int = 100
    gpu_memory_threshold: float = 0.85
    error_cooldown: float = 1.0
    max_recent_errors: int = 100
    logging_verbosity: str = "info"
    error_handling_config: dict = Field(default_factory=lambda: {
        "max_history_per_error": 10,
        "critical_threshold": 5,
        "warning_threshold": 10,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "memory_recovery_attempts": 3,
        "memory_recovery_delay": 1.0
    })

# Used by: ScribeQueue, get_scribe_queue, capture_scribe_event (sovl_queue.py)
class QueueConfig(BaseModel):
    max_queue_size: int = 2000  # Maximum number of entries in the queue
    warning_threshold: float = 0.8  # Warn when queue is 80% full
    fallback_path: str = "scribe_fallback.jsonl"  # Path for fallback file if queue is full
    critical_event_types: set = {"checkpoint", "training_complete"}  # Event types that block on queue
    fallback_max_size_mb: int = 10  # Max size (MB) for fallback file before rotation
    fallback_rotation_count: int = 3  # Number of fallback file rotations to keep

# Used by: Scriber (sovl_scribe.py)
class ScribedConfig(BaseModel):
    scribe_batch_size: int = 20  # Number of entries to batch before writing to file
    scribe_flush_interval: float = 2.0  # Seconds between forced flushes to file
    scribe_queue_maxsize: int = 2000  # Max queue size for scribe events
    output_path: str = "scribe/sovl_scribe.jsonl"  # Output path for scribe JSONL file

class ErrorConfig(BaseModel):
    error_cooldown: float = 1.0  # Cooldown (seconds) between error handling attempts
    warning_threshold: float = 5.0  # Number of errors before warning threshold is triggered
    error_threshold: float = 7.0  # Number of errors before error threshold is triggered
    critical_threshold: float = 10.0  # Number of errors before critical threshold is triggered

# Used by: Logger (sovl_logger.py)
class LoggingConfig(BaseModel):
    log_file: str = "sovl_logs.jsonl"  # Path to the log file
    max_size_mb: int = 10  # Maximum log file size in megabytes before rotation
    compress_old: bool = False  # Whether to compress old log files after rotation
    max_in_memory_logs: int = 1000  # Max number of logs to keep in memory
    rotation_count: int = 5  # Number of rotated log files to keep
    max_log_age_days: int = 30  # Maximum age of logs to keep (days)
    prune_interval_hours: int = 24  # How often to prune old logs (hours)
    memory_threshold_mb: int = 100  # Memory threshold to trigger aggressive pruning
    gpu_memory_threshold: float = 0.85  # GPU memory usage threshold (0-1)
    log_level: str = "INFO"  # Log level threshold
    logging_enabled: bool = True  # Universal on/off switch for all logging
    error_cooldown: float = 1.0  # Time in seconds before an error is no longer considered recent
    max_recent_errors: int = 100  # Maximum number of recent errors to track
    error_handling_config: dict = {
        "max_history_per_error": 10,
        "critical_threshold": 5,
        "warning_threshold": 10,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "memory_recovery_attempts": 3,
        "memory_recovery_delay": 1.0
    }

# Used by: StateBase, SOVLState, StateManager, StateTracker (sovl_state.py)
class StateConfig(BaseModel):
    max_history: int = 100  # Maximum number of history entries to keep
    state_file: str = "state.json"  # Path to the state file

