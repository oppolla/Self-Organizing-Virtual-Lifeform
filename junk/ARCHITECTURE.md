# SOVL System Architecture

## Overview

The Self-Organizing Virtual Lifeform (SOVL) is an AI agent designed to emulate biologically inspired learning and adaptation. A static base language model anchors vast knowledge, while a dynamic scaffold model, fine-tuned via Low-Rank Adaptation (LoRA), captures user-specific contexts. Together, they produce flavourful, coherent, and contextually rich outputs, refined by the system’s self-directed exploration through questions that spark its curiosity.

The dynamic scaffold model is periodically trained during sleep/gestation phases. The system finds a quiet moment to take a nap, training the scaffold on insights from the recent active period. Memory consolidation occurs here, with dreaming weaving deeper or stranger connections from past moments.

SOVL’s curiosity-driven exploration, shaped by confidence and temperament, and its sleep-inspired memory consolidation enable autonomous learning, rich responses, and knowledge refinement through questioning and dreaming. Sanity is maintained as only the scaffold model evolves, the frozen base model serving as the rational anchor. The scaffold, an ever-shifting free-spirit, whispers into the base model’s hidden layers, guiding its responses. 

## System Architecture

SOVL’s modular architecture supports a cyclical workflow of initialization, exploration, generation, learning, and consolidation, ensuring scalability and efficiency.

### 1. System Orchestration (`SOVLOrchestrator`, `SOVLSystem`)

- **Function**: Coordinates initialization, execution, and shutdown, integrating components via dependency injection.
- **Key Features**:
  - Initializes components (`ConfigManager`, `StateManager`, `PluginManager`) with validated configurations.
  - Executes CLI workflows, dispatching commands to components.
  - Manages state persistence (save interval: 300 seconds, max backups: 5) and resource cleanup (e.g., GPU memory).
  - Provides system-wide methods (`generate_curiosity_question`, `dream`, `shutdown`) for operational control.
- **Operation**:
  - Loads configuration sections (`core_config`, `training_config`, etc.), validating completeness.
  - Synchronizes state via `StateManager`, logging state hashes.
  - Handles shutdown by saving state, clearing histories, and releasing resources.
- **Technical Details**:
  - Configurable parameters (log size: 10 MB, save suffix: `_final.json`) via `sovl_config.json`.
  - Thread-safe with `Lock` for state and resource management.
  - Integrates with `PluginManager` for extensibility.

### 2. Model Management (`ModelManager`, `ModelLoader`)

- **Function**: Manages loading, configuration, and switching of base and scaffold models.
- **Key Features**:
  - Supports dynamic model switching (e.g., GPT-2, BERT) for task-specific performance.
  - Applies LoRA adapters to scaffold models (rank: 8, alpha: 16, dropout: 0.1) for efficient updates.
  - Supports quantization modes (fp16, int8, int4) to optimize resources.
  - Validates model configurations (e.g., model path, type, quantization mode) before loading.
- **Operation**:
  - Loads models and tokenizers from local paths or Hugging Face, ensuring configuration validity.
  - Configures LoRA adapters for scaffold models, targeting attention and projection layers.
  - Switches models via CLI, clearing memory to prevent conflicts.
- **Technical Details**:
  - Configurable parameters via `core_config` and `lora_config`.
  - Uses `transformers` and `bitsandbytes` for model management and quantization.
  - Thread-safe with `Lock` for model operations.

### 3. State and Memory Management

#### 3.1 State Management (`StateManager`, `StateTracker`)

- **Function**: Tracks and persists system state, including curiosity, confidence, and temperament.
- **Key Features**:
  - Maintains state history (max: 100 states) and changes (max: 50) for diagnostics.
  - Provides debug tools (`get_debug_info`) for state analysis (e.g., memory usage, recent changes).
  - Supports persistent storage with configurable save intervals and backups (max: 5).
- **Operation**:
  - Serializes state to JSON (`_final.json` suffix), validating state hashes.
  - Updates state with thread-safe operations, logging changes.
  - Clears history during shutdown to optimize resources.
- **Technical Details**:
  - Configurable parameters (save interval, backup limit) via `orchestrator_config`.
  - Thread-safe with reentrant locking.

#### 3.2 Memory Management (`MemoryManager`)

- **Function**: Organizes experiences, including dream memories, conversation histories, and scaffold context.
- **Key Features**:
  - Stores experiences as tensors with metadata (e.g., timestamps), weighted by confidence.
  - Manages dream memories (max: 100), conversation histories (max: 50), and token maps.
  - Applies decay (0.95) and prune threshold (0.1) for efficiency.
- **Operation**:
  - Appends experiences, synchronizing with training and scaffold systems.
  - Updates token maps with confidence-weighted values for recall.
  - Moves unused tensors to CPU for GPU optimization.
- **Technical Details**:
  - Configurable parameters (max length, decay rate) via `memory_config`.
  - Thread-safe with `Lock`.

### 4. Behavioral Drivers

#### 4.1 Curiosity Engine (`CuriosityManager`, `CuriosityEngine`)

- **Function**: Fuels exploration by quantifying curiosity as a blend of ignorance and novelty, sparking questions that drive learning.
- **Key Features**:
  - Computes curiosity scores via `Curiosity.compute_curiosity` (70% ignorance, 30% novelty), using `ConfidenceCalculator`’s scores and `StateManager`’s memory embeddings, as implemented in `sovl_curiosity.py`.
  - Triggers exploration with `CuriosityManager.generate_question` when curiosity pressure exceeds thresholds, crafting queries to probe unknown realms.
  - Manages pressure via `CuriosityPressure`, adjusting based on confidence decay (rate: 0.95) and lifecycle stages.
  - Integrates with `TrainingCycleManager` to initiate training on novel data.
- **Operation**:
  - Evaluates ignorance from low confidence (threshold: 0.7) and novelty via cosine similarity of memory embeddings.
  - Generates questions by processing prompts through scaffold models, logged in `MemoryManager` for consolidation.
  - Adapts exploration intensity with temperament (curious: 1.2x pressure) and lifecycle feedback.
- **Technical Details**:
  - Configurable weights (ignorance: 0.7, novelty: 0.3) via `curiosity_config`.
  - Thread-safe with `Lock` for pressure and queue management.
  - Uses `torch.nn.CosineSimilarity` for novelty calculations.

#### 4.2 Temperament System (`TemperamentSystem`)

- **Function**: Models dynamic behavior via a temperament score, influencing generation and learning.
- **Key Features**:
  - Maintains temperament score (0.0–1.0): cautious (< 0.3), balanced (0.3–0.7), curious (> 0.7).
  - Adjusts generation parameters (e.g., temperature) and learning rates (curious: 1.2x).
  - Adapts based on lifecycle stage and confidence feedback.
- **Operation**:
  - Updates temperament using smoothed confidence and experience metrics.
  - Applies temperament-driven adjustments, ensuring stable transitions.
- **Technical Details**:
  - Configurable thresholds via `controls_config`.
  - Thread-safe with locking.

#### 4.3 Confidence Tracker (`ConfidenceCalculator`)

- **Function**: Evaluates output reliability, guiding exploration and training.
- **Key Features**:
  - Computes confidence from softmax probabilities, adjusted by temperament (cautious: 0.8x, curious: 1.2x) and lifecycle (exploration: 1.1x).
  - Maintains confidence history for error recovery.
  - Clamps values between 0.0 and 1.0.
- **Operation**:
  - Processes model logits, applying modifiers.
  - Triggers exploration or training at low confidence (threshold: 0.7).
- **Technical Details**:
  - Configurable parameters via `confidence_config`.
  - Thread-safe.

### 5. Processing Systems

#### 5.1 Scaffold Integration (`ScaffoldProvider`, `ScaffoldTokenMapper`, `CrossAttentionInjector`)

- **Function**: Enhances output quality by integrating scaffold model context.
- **Key Features**:
  - Aligns token spaces via `ScaffoldTokenMapper` with confidence-weighted updates (cap: 5.0).
  - Injects scaffold context using cross-attention (`CrossAttentionInjector`).
  - Adjusts scaffold influence based on confidence and temperament.
- **Operation**:
  - `ScaffoldProvider` generates context tensors.
  - `CrossAttentionInjector` modifies base model attention layers.
- **Technical Details**:
  - Supports quantization (e.g., int8) via `core_config`.
  - Uses `torch.nn` for cross-attention.
  - Thread-safe.

#### 5.2 Training System (`SOVLTrainer`)

- **Function**: Manages model training, including standard, sleep, and gestation cycles to refine the scaffold model.
- **Key Features**:
  - Supports gradient accumulation and mixed-precision training for efficiency.
  - Replays dream memories during sleep, weighted by novelty to enhance learning.
  - Uses LoRA adapters for scaffold updates (rank: 8, alpha: 16, dropout: 0.1).
  - Adjusts learning rates by temperament (curious: 1.2x, cautious: 0.8x).
- **Operation**:
  - Trains with scaffold-guided context, aligning outputs with base model stability.
  - Performs gestation cycles with low learning rates (2e-5) for subtle refinements.
- **Sleep and Gestation Cycles**:
  - **Function**: Consolidates knowledge and sparks creativity by replaying dream memories, weaving experiences into the scaffold model’s patterns.
  - **Triggers**:
    - **Low Confidence**: Initiates sleep when `ConfidenceTracker` detects scores below 0.7, signaling uncertainty that prompts reflection.
    - **Compute Intervals**: Triggers gestation after configurable cycles (e.g., every 1000 steps) to process accumulated experiences.
    - **CLI Command**: Activates via `dream` command, allowing manual introspection.
  - **Operation**:
    - Replays up to 100 dream memories from `DreamMemory`, weighted by novelty (boost: 0.03), with temperament-based noise (e.g., 0.02 for melancholy) to inspire creative connections.
    - Trains scaffold model with low learning rate (2e-5) and LoRA adapters, synchronizing with `CrossAttentionInjector`’s context.
    - Prunes low-weight memories (threshold: 0.1) to optimize GPU usage, guided by `MemoryManager`.
  - **Technical Details**:
    - Configurable parameters (cycle interval: 1000 steps, novelty boost: 0.03) via `training_config` and `memory_config`.
    - Thread-safe with `Lock` for memory access and training.
    - Leverages `torch.cuda` for memory optimization during replay.
- **Technical Details**:
  - Configurable parameters (batch size: 2, learning rate: 2e-5) via `training_config`.
  - Thread-safe with `Lock` for training operations.

#### 5.3 Memory Consolidation System (`DreamMemory`)

- **Function**: Consolidates knowledge through dreaming, enhancing learning and creativity.
- **Key Features**:
  - Stores 100 experiences as weighted tensors, with novelty boost (0.03).
  - Applies decay (0.95) and prune threshold (0.1).
  - Adds temperament-based noise (e.g., 0.02 for melancholy).
- **Operation**:
  - Appends experiences to a `deque`, pruning low-weight memories.
  - Aggregates tensors during dreaming, synchronized with scaffold context.
- **Technical Details**:
  - Configurable parameters via `memory_config`.
  - Thread-safe.

### 6. User Interaction

#### 6.1 Command Line Interface (CLI)

- **Function**: Provides robust system control via commands.
- **Key Commands**:
  - **Generation**: `generate`, `echo`, `mimic`.
  - **Training**: `train`, `dream`.
  - **Memory**: `memory`, `recall`, `recap`.
  - **Interaction**: `muse`, `flare`, `debate`, `spark`, `reflect`.
  - **Model Management**: `switch_model`, `set_quantization`.
  - **System Control**: `status`, `save`, `reset`.
- **Operation**:
  - Commands dispatched by `SOVLOrchestrator`, synchronized via `StateManager`.
- **Technical Details**:
  - Configurable command history via `orchestrator_config`.
  - Extensible via `PluginManager`.

#### 6.2 Configuration Management (`SystemContext`, `ConfigManager`)

- **Function**: Enables fine-grained control through modular configuration.
- **Key Sections**:
  - `core_config`: Model settings (e.g., base model: GPT-2, quantization: int8).
  - `training_config`: Training parameters (e.g., learning rate: 2e-5, batch size: 2).
  - `curiosity_config`: Curiosity weights (ignorance: 0.7, novelty: 0.3).
  - `memory_config`: Memory parameters (decay: 0.95, prune threshold: 0.1).
  - `controls_config`: Temperament thresholds (curious > 0.7).
  - `lora_config`: LoRA settings (rank: 8, alpha: 16, dropout: 0.1).
  - `orchestrator_config`: System settings (log size: 10 MB, save interval: 300 seconds).
- **Operation**:
  - Validates configurations, propagating updates via event dispatcher.
  - Supports JSON-based files with backups.
- **Technical Details**:
  - Thread-safe with subscription-based notifications.

## Operational Workflow

`SOVLOrchestrator` sparks the system, awakening `CuriosityManager` to generate questions through `generate_question` when novelty or low confidence stirs exploration, as defined in `sovl_curiosity.py`. 

The base model crafts responses, enriched by the scaffold’s cross-attention whispers. 

`SOVLTrainer` refines the scaffold model with novel experiences, triggered by `ConfidenceTracker`. 

During sleep/gestation, `DreamMemory` weaves memories into lasting patterns, deepening SOVL’s understanding. 

`TemperamentManager` shapes the system’s mood, guiding its curious dance with the world.
