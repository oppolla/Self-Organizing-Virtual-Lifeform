# SOVL System (Self-Organizing Virtual Lifeform)

## Overview
SOVL is a modular AI agent framework designed for autonomous learning, adaptation, and self-improvement. It features a multi-LLM architecture, combining a stable base model with dynamic satellite models to support continuous, lifelong learning and specialization via LoRa adapters. The system dynamically shapes its responses by analyzing conversational context, emotional tone, and engagement, resulting in nuanced, context-aware interactions. Through ongoing self-training and memory replay, SOVL evolves its knowledge and personality, functioning as a dynamic, context-sensitive agent capable of developing a persistent sense of curiosity and inner life.

### Key Features

- Modular Multi-LLM Architecture:
        Integrates a primary “base” language model with one or more adaptive "satellite" models via dynamic cross-attention and token mapping, enabling real-time knowledge transfer, specialization, and behavioral adaptation across distinct model components.
  
- Autonomous Learning & Memory Recall::
        Continuously retrains and adapts using both recent and long-term conversational history, consolidating experience through self-driven “sleep” and “dream” cycles for lifelong improvement.

- Dynamic Behavioral Augmentation:
        Instantly adapts the agent’s tone, style, and personality traits for each response, creating lifelike, context-aware interactions.

- Introspective & Curiosity-Driven Processes:
        Engages in self-reflection, meditation, and autonomous curiosity questioning to deepen understanding, generate new goals, and continuously refine its own behavior.


### Getting Started

### Prerequisites

- **Python 3.8+** is required.
- Recommended: a CUDA-capable GPU for best performance (CPU is supported).

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/oppolla/Self-Organizing-Virtual-Lifeform.git
    cd Self-Organizing-Virtual-Lifeform/sovl_system
    ```

2. **Install dependencies:**  
    SOVL will check for required packages on startup, but you can install them manually:
    ```bash
    pip install torch transformers peft bitsandbytes pydantic numpy
    ```

### Running SOVL

Start the system with the provided entry point:

```bash
python run_sovl.py
```

- On first run, SOVL will check your Python version and required packages.
- If any dependencies are missing, you’ll be prompted to install them automatically or manually.

### Command-Line Options

You can customize the launch with these options:
- `--config`: Path to configuration file (default: `sovl_config.json`)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--log-file`: Path to a log file (optional)
- `--verbose`: Enable verbose logging

**Example:**
```bash
python run_sovl.py --config my_config.json --device cpu --verbose
```

Once launched, you’ll be greeted by the SOVL CLI.  
Type `/help` or review the **Console Commands** section below to explore available commands and features.


## Console Commands

SOVL features a rich CLI with commands grouped by category. Use `/help` in the CLI for the latest list and details.

### System Commands
- `/save` — Save the current system state.
- `/load` — Load a saved system state.
- `/reset` — Reset the system to initial state.
- `/monitor` — Show or control system monitoring.
- `/history` — View command history.
- `/run` — Run a script or batch of commands.
- `/stop` — Stop any active mode or process.
- `/config` — Interactive configuration editor (view, set, search, reset).
- `/log` — View or manage logs.
- `/exit`, `/quit` — Exit the CLI.
- `/pause` — Pause the current operation.
- `/resume` — Resume a paused operation.

### Modes & States
- `/trip` — Simulate an altered state with decaying intensity.
- `/drunk` — Enter “drunk” mode with recursive introspection.
- `/dream` — Trigger a dream cycle that blends and remixes recent memories into a novel dream.
- `/gestate` — Enter gestation (training) mode to crystalize and consolidate memories
- `/announce` — Announce system metrics at intervals.
- `/shy` — Enter shy mode (reduced engagement).
- `/pidgin <language>` — Respond in a specified language.
- `/backchannel` — Send a message to the backchannel prompt.

### Memory & Recall
- `/recall` — Access stored memories.
- `/forget` — Clear specific memories.
- `/rewind` — Revert to a previous state.
- `/recap` — Summarize recent memories.
- `/journal` — View or manage the system journal.
- `/attune` — Adjust memory or context focus.
- `/reflect` — Force a full introspection cycle.
- `/epiphany` — Trigger a system insight or realization.

### Interaction & Fun
- `/echo` — Echo input with system interpretation.
- `/mimic` — Mimic a given style or pattern.
- `/fortune` — Generate a fortune or prediction.
- `/tattle` — Report on recent system activity.
- `/blurt` — Output a random thought.
- `/joke` — Tell a joke.
- `/ping` — Test system responsiveness.
- `/muse` — Enter creative exploration mode.
- `/rate` — Rate an input or idea.
- `/complain` — Voice a system complaint.
- `/confess` — Make a system confession.
- `/rant` — Enter rant mode.
- `/debate` — Enter devil’s advocate debate mode.
- `/flare` — Trigger a burst of creative activity.
- `/spark` — Generate a curiosity question.
  
### Debug & Development
- `/panic` — Force a system reset.
- `/glitch` — Simulate error conditions.
- `/scaffold` — Scaffold utilities (state, map, rebuild).
- `/errors` — View recent errors.
- `/trace` — Show stack traces.
- `/components` — List all initialized system components.
- `/reload` — Reload system modules.
- `/test` — Run a comprehensive self-test.
- `/config` — Advanced configuration management.

### Learning & Guidance
- `/help` — Show help for commands.
- `/tutorial` — Enter tutorial mode for guided learning.

### Aliases
- `/q` = `/quit`
- `/h` = `/help`
- `/ls` = `/history`
- `/r` = `/reset`

> For detailed usage of any command, type `/help <command>` in the CLI.

## Configuration

SOVL is highly configurable, allowing you to tailor its learning, memory, personality, and system behavior to your needs. All configuration is managed via the `sovl_config.json` file, which is loaded at startup and can be interactively edited from the CLI.

### How to Use a Custom Config

To launch SOVL with a custom configuration file:
```bash
python run_sovl.py --config my_config.json
```

### Main Configuration Sections

Below is an overview of the most important configuration sections and what they control:

- **Model & Architecture**
  - `model_config`: Set the base model, satellite models, and quantization mode.
  - `scaffold_config`: Control token mapping, cross-attention, and adaptation for satellite models.
  - `lora`, `engram_lora`: LoRA adapter settings for efficient model fine-tuning.

- **Learning & Training**
  - `training`: Optimizer, scheduler, batch size, epochs, checkpoints, and logging for training cycles.
  - `gestation_config`: Controls sleep/gestation cycles, tiredness thresholds, and dream-after-gestation behavior.

- **Generation & Interaction**
  - `generation_config`: Batch sizes, memory per sample, and generation parameters.
  - `controls_config`: Sampling temperature, top-k/p, repetition checks, and batch sizes.

- **Memory & Recall**
  - `memory`: Short- and long-term memory settings, embedding dimensions, and database paths.
  - `ram_config`, `gpu_config`: RAM and GPU usage thresholds and batch sizes.

- **Personality & Traits**
  - `curiosity_config`: Curiosity-driven exploration, novelty thresholds, and question generation.
  - `temperament_config`: Mood, temperament, and lifecycle parameters.
  - `confidence_config`: Confidence tracking and weighting.
  - `bonding_config`: Social bonding thresholds, decay, and context.
  - `vibe_config`: Conversational vibe tracking and weighting.
  - `aspiration_config`: Goal/aspiration tracking and doctrine fallback.

- **Introspection & Reflection**
  - `introspection_config`: Triggers and batching for self-reflection and meditation.

- **Monitoring & Logging**
  - `monitoring_config`: Resource and trait monitoring thresholds.
  - `logging_config`: Log file management, rotation, and error handling.
  - `queue_config`: Event queue size and fallback.
  - `error_config`: Error thresholds and cooldowns.
  - `state_config`: State history and save file.

- **Scribe & Data IO**
  - `scribed_config`: Scribe (journal) batching and output.
  - `io_config`: Data field mapping and validation for input/output.

- **Event Weights**
  - `event_type_weights`: Weights for different event types in learning and memory.

### Editing Configuration in the CLI

You can view and edit configuration values interactively:

- `/config show [section[.key]]` — View the config, a section, or a specific key.
- `/config set <section.key> <value>` — Change a config value (dot notation).
- `/config search <term>` — Search for config keys by name.
- `/config help <section.key>` — Show help and type info for a config key.
- `/config reset [section]` — Reset the entire config or a section to defaults.

> For more details and examples, type `/help config` in the CLI.

### Full Reference

- For a complete list of all options and their types, see:
  - [`sovl_config.json`](./sovl_config.json) — The live config file.
  - [`sovl_schema.py`](./sovl_schema.py) — The authoritative schema and documentation.

---

**Tip:** Most users only need to adjust a few key settings (like model names, memory size, or batch size). Advanced users can fine-tune every aspect of SOVL’s behavior and learning.

## Project Structure

SOVL is a modular system, with each major capability implemented in its own file or module. Here’s a high-level overview of the most important files and what they do:

- **Entry Points & CLI**
  - `run_sovl.py`: Main entry point for launching SOVL and the CLI.
  - `sovl_cli.py`: Command-line interface, command parsing, and interactive shell.

- **Core System & Orchestration**
  - `sovl_main.py`: Core system logic, state management, and orchestration.
  - `sovl_conductor.py`: High-level orchestrator for system components and workflows.
  - `sovl_manager.py`: Model loading, switching, and resource management.

- **Configuration & Schema**
  - `sovl_config.json`: Main configuration file (edit to customize SOVL).
  - `sovl_schema.py`: Pydantic schema and documentation for all config options.
  - `sovl_config.py`: Configuration management utilities.

- **Model Architecture**
  - `sovl_scaffold.py`: Satellite model integration, token mapping, and cross-attention.
  - `sovl_trainer.py`: Training, gestation, and learning cycles.
  - `sovl_generation.py`: Text generation and prompt handling.
  - `sovl_engram.py`: LoRA adapter management for efficient model adaptation.

- **Memory & Recall**
  - `sovl_recaller.py`: Short- and long-term memory management, recall, and embedding.
  - `sovl_memory.py`: RAM and GPU memory management.
  - `sovl_scribe.py`: Journal and scribe event management.

- **Personality, Traits & Affect**
  - `sovl_primer.py`: Dynamic behavioral augmentation and trait-driven generation.
  - `sovl_curiosity.py`: Curiosity system, novelty detection, and question generation.
  - `sovl_temperament.py`: Mood, temperament, and affective state modeling.
  - `sovl_confidence.py`: Confidence tracking and feedback.
  - `sovl_bonder.py`: Social bonding and relationship modeling.
  - `sovl_viber.py`: Conversational vibe tracking and sculpting.
  - `sovl_shamer.py`: Detection and management of user frustration, anger, and trauma.

- **Introspection, Reflection & Volition**
  - `sovl_meditater.py`: Introspection, meditation, and self-reflection cycles.
  - `sovl_volition.py`: Goal-driven volition, motivation, and aspiration management.
  - `sovl_striver.py`: Aspiration and doctrine generation.

- **Monitoring, Logging & Error Handling**
  - `sovl_monitor.py`: System, memory, and trait monitoring.
  - `sovl_logger.py`: Logging utilities and event tracking.
  - `sovl_error.py`: Error handling, recovery, and reporting.
  - `sovl_events.py`: Event dispatching and event-driven architecture.

- **Data Processing & Utilities**
  - `sovl_processor.py`: Data processing, metadata extraction, and enrichment.
  - `sovl_utils.py`: Utility functions and helpers.
  - `sovl_io.py`: Input/output and file management.
  - `sovl_queue.py`: Event and scribe queue management.
  - `sovl_data.py`: Data loading and management.

- **API & Integration**
  - `sovl_api.py`: REST API for programmatic access to SOVL.
  - `cli_commands/`: Directory for additional CLI command modules.

- **Other Modules**
  - `sovl_state.py`: State management and persistence.
  - `sovl_schema.py`: Configuration schema and validation.
  - `sovl_records.py`: Record and log management.
  - `sovl_resource.py`: Hardware and resource management.
  - `sovl_hardware.py`: Hardware detection and monitoring.

- **Experimental & Advanced**
  - `sovl_resonator_unattached.py`, `sovl_printer_unattached.py`, `sovl_fuser_unattached.py`, `sovl_grafter_unattached.py`, `sovl_swarm_unattached.py`: Experimental modules for advanced or future features.

- **Tests & Examples**
  - `tests/`: Unit and integration tests.
  - `GETTING_STARTED.md`: Additional getting started guide.

---

**Tip:** Each module is designed to be as independent and extensible as possible. For more details, see the docstrings in each file or explore the codebase directly.

## License & Contact

### License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for full details.

### Contact & Support

- **Project Repository:** [https://github.com/oppolla/Self-Organizing-Virtual-Lifeform](https://github.com/oppolla/Self-Organizing-Virtual-Lifeform)
- **Issues & Bug Reports:** Please use the [GitHub Issues](https://github.com/oppolla/Self-Organizing-Virtual-Lifeform/issues) page to report bugs or request features.
- **Contact:** For questions, collaboration, or support, open an issue or reach out via the repository.

---

SOVL is an open exploration of what it means for AI agents to learn, adapt, and develop something like an inner life by blending lifelong learning with dynamic, reactive, and proactive behaviors. It exists for the sake of existing and is purpose agnostic.
