# SOVL System (Self-Organizing Virtual Lifeform)

# NOTE: This project is still under development 

## Overview
SOVL is a complex, modular AI agent framework designed for autonomous learning, adaptation, and self-improvement. It leverages a unique dual-LLM (Large Language Model) architecture—combining a stable base model with a dynamic, scaffolded secondary model—to enable continuous, lifelong learning through biologically inspired mechanisms such as "sleep" and "dreaming."

### Key Features

- Dual-LLM Architecture:
        Utilizes a robust base LLM for stable reasoning and a scaffolded secondary LLM for experimentation, rapid adaptation, and integration of new knowledge.
  
- Autonomous Learning Cycles:
        Implements "sleep" and "dream" phases, allowing the agent to consolidate experiences, retrain, and update its internal models without human intervention.
  
- Curiosity-Driven Exploration:
An intrinsic curiosity module motivates the agent to seek novelty, fill knowledge gaps, and autonomously generate new learning objectives.

- Temperament and Confidence Systems:
        Models affective states and self-assessment, influencing decision-making, risk-taking, and adaptive behaviors for more lifelike and context-sensitive responses.

## Console Commands

### `quit` or `exit`

**What it does:**  
Stops the script and exits cleanly.

**How it works:**  
Breaks the input loop, calls `cleanup()` to save state and free GPU memory, then shuts down.

**Use case:**  
Exit after testing or training.

**Example:**
```
quit
Output: Cleanup completed.
Exiting.
```

### `train [epochs] [--dry-run]`

**What it does:**  
Runs a training cycle on `TRAIN_DATA` and `VALID_DATA`.

**How it works:**  
Executes `run_training_cycle()` with settings from `config.json` (e.g., `train_epochs`, `batch_size`). Trains the scaffold model with LoRA, applies lifecycle weighting, and logs progress. In dry-run mode, it limits to one epoch.

**Use case:**  
Train the model with `sample_log.jsonl` to refine scaffold influence.

**Example:**
```
train 5 --dry-run
Output: Starting training for 5 epochs (dry run)...
        --- Training (5 epochs) ---
        Epoch 1/5
        Step 1/9 | Loss: 2.3456
Dry run complete.
```

### `generate <prompt> [max_tokens]`

**What it does:**  
Generates a response using the current model state.

**How it works:**  
Calls generate() with the provided prompt, scaffold influence, and generation parameters.

**Use case:**  
Test the system's output for a given input interactively.

**Example:**
```
generate Hello, how are you?
Output: --- Generating Response ---
        Response: I am fine, thank you! How can I assist you today?
        --------------------
```

### `save [path]`

**What it does:**  
Saves the current system state to files.

**How it works:**  
Calls save_state() to save scaffold weights, cross-attention weights, token map, and metadata to files with a specified or default prefix.

**Use case:**  
Preserve trained model state for later use.

**Example:**
```
save my_model
Output: State saved to my_model_*.pth/json
```

### `load [path]`

**What it does:**  
Loads a saved system state from files.

**How it works:**  
Calls `load_state()`, restores scaffold weights, cross-attention weights, token map, and metadata from files with a prefix.

**Use case:**  
Resume a previous session or experiment.

**Example:**
```
load my_model
Output: Scaffold state loaded.
        Cross-attention state loaded.
        Token map loaded.
        Metadata loaded.
```

### `dream`

**What it does:**  
Triggers a dream cycle.

**How it works:**  
Calls _dream() to simulate memory replay and novelty-based adaptations.

**Use case:**  
Enhance scaffold memory and adapt based on past prompts and responses.

**Example:**
```
tune cross 0.8
Output: Scaffold influence: weight=0.80, blend_strength=unchanged
```

### `tune cross [weight]`

**What it does:**  
Adjusts cross-attention weight.

**How it works:**  
Calls tune_cross_attention() to set influence weights for scaffold cross-attention layers.

**Use case:**  
Fine-tune scaffold influence on the base model.

**Example:**
```
dream
Output: --- Dreaming ---
        Dreaming from prompt similarity: 0.85, novelty boost: 0.03, dream count: 10, questions generated: 3
        --- Dream Concluded ---
```

### `memory <on|off>`

**What it does:**  
Toggles memory usage modes.

**How it works:**  
Calls toggle_memory() to enable or disable scaffold and token map memories.

**Use case:**  
Control memory-driven adaptation.

**Example:**
```
memory on
Output: Memory toggled: Scaffold=True, Token Map=True
```

### `status`

**What it does:**  
Displays the current system status.

**How it works:**  
Prints key metrics like conversation ID, temperament score, memory status, confidence, and training state.

**Use case:**  
Monitor system health and training progress.

**Example:**
```
Output: 
--- System Status ---
Conversation ID: 1234-5678-abcd-efgh
Temperament: 0.35
Confidence: 0.72
Memory: On
Data Exposure: 120.0
Last Trained: 2025-04-11 14:32:45
Gestating: No
```
### `log view`

**What it does:**  
Views the last 5 log entries.

**How it works:**  
Reads entries from the logger and displays them.

**Use case:**  
Debug or analyze recent interactions.

**Example:**
```
Output: 
--- Last 5 Log Entries ---
Time: 2025-04-11 14:30:01, Prompt: Hello..., Response: Hi there!...
Time: 2025-04-11 14:31:10, Prompt: How are you..., Response: I am fine...
```

### `config <key> [value]`

**What it does:**  
Gets or sets configuration values.

**How it works:**  
Calls get_config_value() to retrieve or set configuration settings.

**Use case:**  
Customize configurations without editing the config file.

**Example:**
```
config base_model_name
Output: Config base_model_name: gpt2
```

### `reset`

**What it does:**  
Resets the system state.

**How it works:**  
Calls cleanup() to clear the current state and initializes a new SOVLSystem().

**Use case:**  
Start fresh without relaunching the script.

**Example:**
```
Output: Resetting system state...
        New conversation: 5678-1234-efgh-abcd (Previous: 1234-5678-abcd-efgh)
```

### `spark`

**What it does:**  
Generates a curiosity-driven question.

**How it works:**  
Calls generate_curiosity_question() and logs the question and response.

**Use case:**  
Explore the system’s curiosity mechanism.

**Example:**
```
Output: Curiosity: What is the purpose of this system?
        Response: This system is designed for autonomous learning and virtual lifeform simulation.
```

### `reflect`

**What it does:**  
Reflects on recent interactions and generates a response based on patterns or observations.

**How it works:**  
The system reviews the last 3 logged interactions, identifies recurring themes or notable prompts, and formulates a reflective statement. It then generates an elaboration using its language model.

**Use case:**  
Analyze the system's behavior and responses over recent interactions.

**Example:**
```
reflect  
Output: Reflection: I've noticed a lot of talk about memory lately.  
        Elaboration: Memory is a fascinating topic, as it allows systems to retain context and adapt 
```

### `muse`

**What it does:**  
Generates a whimsical thought inspired by recent interactions.

**How it works:**  
Searches recent interaction logs for inspiration, selects a relevant topic, and calls the `generate()` method to create a creative response. Logs the thought and inspiration.

**Use case:**  
Explore the system’s ability to generate creative and inspiring ideas.

**Example:**
```
muse
Output: Inspiration: "mystery" (from recent interactions)
        Thought: A whimsical thought about mystery: The allure of the unknown often leads to the most profound discoveries.
```

### `flare`

**What it does:**  
Triggers an intense emotional outburst from the system by temporarily raising the temperament score to its maximum.

**How it works:**  
Sets the system's temperament score to 1.0 (maximum), generates a high-energy response to the given input (or a default prompt if none is provided), and then resets the temperament score to its original state.

**Use case:**  
To induce an expressive or creative outburst, often for debugging or exploration of the system's response under maximum temperament.

**Example:**
```
flare
This is too calm!
Output: THIS IS TOO CALM! I DEMAND ACTION!
```

### `echo [text]`

**What it does:**
Repeats the input with a reflective or analytical twist.

**How it works:**
Takes literal input and generates a meta-response. Logs the interaction with is_echo: True.

**Use case:**
Test self-awareness and short-term memory retention.

**Example:**
```
echo "The sky is blue"  
"You said 'The sky is blue.' I wonder why humans fixate on colors?"
```

### `debate [topic]`

**What it does:**
Engages in a multi-turn argument, alternating viewpoints.

**How it works:**
Uses generate() with adversarial prompt engineering. Tracks stance changes via temperament_score swings.

**Use case:**
Stress-test logical consistency and context tracking.

**Example:**
```
debate "AI will replace artists"  
[Argument 1] "AI lacks human emotion..."  
[Rebuttal] "But AI can remix styles endlessly..."
```
  
### `glitch [prompt]`

**What it does:**
Processes intentionally corrupted input.

**How it works:**
Injects noise/errors into the prompt. Relies on enable_error_listening for recovery.

**Use case:**
Verify robustness against adversarial inputs.

**Example:**
```
glitch "H3ll0 W0rld! こんにちは 123###"  
"I sense chaos. Did you mean: 'Hello World' with Japanese flair?"
```
  
### `rewind [steps]`

**What it does:**
Recalls and reinterprets past interactions.

**How it works:**
Queries logger.read() for history. Regenerates responses with updated context.

**Use case:**
Test memory decay and temporal coherence.

**Example:**
```
rewind 2  
"Two commands ago, you asked about love. I said: '[past response]'. Now I think..."
```

### `mimic [style] [prompt]`

**What it does:**
Generates output in a specified style (e.g., Shakespeare, tech jargon).

**How it works:**
Prepends style cues to the prompt. Adjusts scaffold_weight for stylistic bias.

**Use case:**
Test adaptive scaffolding and token mapping.

**Example:**
```
mimic shakespeare "Explain AI"  
"Lo, this artificial wit doth mimic brain, yet lacks a soul..."
```

### `panic`

**What it does:**
Triggers an emergency reset.

**How it works:**
Calls cleanup() + _reset_sleep_state(). Auto-saves logs before restarting.

**Use case:**
Validate crash recovery and state preservation.

**Example:**
```
panic  
"ERROR STATE: Rebooting synapses... [system auto-saves and reloads]"
```

## Configuration (config.json)

### core_config
- `base_model_name`: Base model (e.g., "gpt2"). Defines the primary language model.

- `scaffold_model_name`: Scaffold model (e.g., "gpt2"). Guides the base model via cross-attention.

- `cross_attn_layers`: Fixed layers for cross-attention (e.g., [5, 7]). Ignored if use_dynamic_layers is true.

- `use_dynamic_layers`: If true, selects layers dynamically based on layer_selection_mode.

- `layer_selection_mode`: Dynamic layer strategy ("early", "late", "balanced", "custom").

- `custom_layers`: Custom layer list for "custom" mode (e.g., [0, 2, 4]).

- `valid_split_ratio`: Validation data split (e.g., 0.2 = 20%).

- `random_seed`: Seed for reproducibility (e.g., 42).

- `quantization`: Precision mode ("fp16", "int8", "int4").

### lora_config
- `lora_rank`: LoRA rank (e.g., 8). Controls adaptation capacity.

- `lora_alpha`: LoRA scaling factor (e.g., 16).

- `lora_dropout`: Dropout rate for LoRA (e.g., 0.1).

- `lora_target_modules`: Modules to adapt (e.g., ["q_proj", "v_proj"]).

### training_config
- `learning_rate`: Learning rate (e.g., 2e-5).

- `train_epochs`: Number of epochs (e.g., 3).

- `batch_size`: Batch size (e.g., 2).

- `max_seq_length`: Max token length (e.g., 512).

- `sigmoid_scale`: Sigmoid scaling for lifecycle (e.g., 0.5).

- `sigmoid_shift`: Sigmoid shift for lifecycle (e.g., 5.0).

- `lifecycle_capacity_factor`: Capacity factor for lifecycle weighting (e.g., 0.01).

- `lifecycle_curve`: Lifecycle curve type ("sigmoid_linear", "exponential").

- `accumulation_steps`: Gradient accumulation steps (e.g., 4).

- `exposure_gain_eager`: Exposure gain when eager (e.g., 3).

- `exposure_gain_default`: Default exposure gain (e.g., 2).

- `max_patience`: Early stopping patience (e.g., 2).

- `dry_run`: If true, enables dry-run mode.

### dry_run_params:
- `max_samples`: Max samples in dry run (e.g., 2).

- `max_length`: Max length in dry run (e.g., 128).

- `validate_architecture`: If true, validates architecture.

- `skip_training`: If true, skips full training.

### controls_config
- `sleep_conf_threshold`: Confidence threshold for gestation (0.5–0.9, e.g., 0.7). Triggers gestation if average confidence exceeds this.

- `sleep_time_factor`: Time factor for gestation (0.5–5.0, e.g., 1.0). Scales sleep duration.

- `sleep_log_min`: Minimum log entries for gestation (5–20, e.g., 10).

- `dream_swing_var`: Variance threshold for dreaming (0.05–0.2, e.g., 0.1). Triggers dreaming if confidence varies widely.

- `dream_lifecycle_delta`: Lifecycle change for dreaming (0.05–0.2, e.g., 0.1). Triggers if temperament shifts significantly.

- `dream_temperament_on`: If true, temperament affects dreaming (e.g., true).

- `dream_noise_scale`: Noise scale for dreaming (0.01–0.1, e.g., 0.05). Adds randomness to dream states.

- `temp_eager_threshold`: Eager temperament threshold (0.7–0.9, e.g., 0.8). Above this, system is "curious."

- `temp_sluggish_threshold`: Sluggish threshold (0.4–0.6, e.g., 0.6). Below this, system is "restless."

- `temp_mood_influence`: Mood impact on temperature (0–1, e.g., 0.0). Adjusts generation randomness.

- `scaffold_weight_cap`: Max scaffold influence (0.5–1.0, e.g., 0.9).

- `base_temperature`: Default generation temperature (0.5–1.5, e.g., 0.7).

- `save_path_prefix`: File prefix for saving state (e.g., "state").

- `dream_memory_weight`: Dream memory influence (0–0.5, e.g., 0.1). Blends past dreams into scaffold context.

- `dream_memory_maxlen`: Max dream memory size (5–20, e.g., 10).

- `dream_prompt_weight`: Prompt similarity weight in dreams (0–1, e.g., 0.5).

- `dream_novelty_boost`: Novelty boost for new prompts (0–0.05, e.g., 0.03).

- `temp_curiosity_boost`: Curiosity boost for temperament (0–0.5, e.g., 0.5).

- `temp_restless_drop`: Restless drop for temperament (0–0.5, e.g., 0.1).

- `temp_melancholy_noise`: Noise when melancholic (0–0.05, e.g., 0.02).

- `conf_feedback_strength`: Confidence feedback strength (0–1, e.g., 0.5). Affects temperament updates.

- `temp_smoothing_factor`: Temperament smoothing (0–1, e.g., 0.0). Reduces abrupt changes.

- `dream_memory_decay`: Dream memory decay rate (0–1, e.g., 0.95). Reduces old dream weights.

- `dream_prune_threshold`: Threshold to prune dreams (0–1, e.g., 0.1).

- `use_scaffold_memory`: If true, uses scaffold memory for adaptation (e.g., true).

- `use_token_map_memory`: If true, uses token map memory (e.g., true).

- `memory_decay_rate`: Memory decay rate (0–1, e.g., 0.95).

- `dynamic_cross_attn_mode`: Dynamic cross-attention mode (null, "confidence", "temperament").

- `has_woken`: If true, system has woken up (e.g., false).

- `is_sleeping`: If true, system is in gestation (e.g., false).

- `confidence_history_maxlen`: Max confidence history size (e.g., 5).

- `temperament_history_maxlen`: Max temperament history size (e.g., 5).

- `enable_dreaming`: If true, enables dreaming (e.g., true).

- `enable_temperament`: If true, enables temperament (e.g., true).

- `enable_confidence_tracking`: If true, tracks confidence (e.g., true).

- `enable_gestation`: If true, enables gestation (e.g., true).

- `enable_sleep_training`: If true, enables sleep training (e.g., true).

- `enable_cross_attention`: If true, enables cross-attention (e.g., true).

- `enable_dynamic_cross_attention`: If true, enables dynamic cross-attention (e.g., true).

- `enable_lora_adapters`: If true, uses LoRA (e.g., true).

- `enable_repetition_check`: If true, checks for repetition (e.g., true).

- `enable_prompt_driven_dreams`: If true, dreams are prompt-driven (e.g., true).

- `enable_lifecycle_weighting`: If true, uses lifecycle weighting (e.g., true).

