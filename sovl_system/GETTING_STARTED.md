# Step-by-Step Guide to Running SOVL System

## Prerequisites

1. **Python Environment**:
   - Python 3.8 or higher
   - Virtual environment recommended (but not required)

2. **Dependencies**:
   ```bash
   pip install torch transformers peft bitsandbytes
   ```

3. **Hardware Requirements**:
   - For GPU usage: NVIDIA GPU with CUDA support
   - Minimum 16GB RAM recommended
   - Sufficient disk space for model storage and checkpoints

## Step 1: Prepare the Configuration File

Create a `sovl_config.json` file with the following structure:

```json
{
  "core_config": {
    "base_model_name": "SmolLM2-360M",
    "base_model_path": null,
    "scaffold_model_name": "SmolLM2-135M",
    "scaffold_model_path": null,
    "cross_attn_layers": [],
    "use_dynamic_layers": false,
    "layer_selection_mode": "balanced",
    "custom_layers": null,
    "valid_split_ratio": 0.2,
    "random_seed": 42,
    "quantization": "fp16",
    "hidden_size": 768,
    "num_heads": 12,
    "gradient_checkpointing": true,
    "initializer_range": 0.02,
    "migration_mode": true
  },
  "training_config": {
    "learning_rate": 0.0001,
    "grad_accum_steps": 4,
    "max_grad_norm": 1.0,
    "batch_size": 32,
    "epochs": 10,
    "warmup_steps": 1000,
    "weight_decay": 0.01
  },
  "memory_config": {
    "memory_threshold": 0.8,
    "memory_decay_rate": 0.95,
    "max_memory_mb": 16000,
    "checkpoint_interval": 1
  },
  "state_config": {
    "state_save_interval": 300,
    "max_backup_files": 5
  }
}
```

## Step 2: Running the System

The system can be run in several modes:

### Basic Run
```bash
python run_sovl.py --config sovl_config.json --device cuda
```

### Available Command Line Arguments:
- `--config`: Path to configuration file (required)
- `--device`: Device to use ("cuda" or "cpu")
- `--mode`: Operation mode ("train", "generate", "dream", "muse", "flare", "debate", "spark", "reflect")
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--train-data`: Path to training data file
- `--valid-data`: Path to validation data file
- `--test`: Run in test mode
- `--verbose`: Enable verbose logging
- `--monitor-interval`: Monitoring update interval in seconds
- `--checkpoint-interval`: Checkpoint interval in epochs
- `--resume-from-checkpoint`: Path to checkpoint file to resume from
- `--validate-every`: Run validation every N epochs
- `--max-patience`: Max epochs without validation improvement
- `--max-checkpoints`: Maximum number of checkpoints to keep
- `--warmup-steps`: Number of warmup steps for learning rate
- `--weight-decay`: Weight decay for optimizer
- `--grad-accum-steps`: Number of gradient accumulation steps
- `--max-grad-norm`: Maximum gradient norm for clipping

## Step 3: System Operation Modes

### Training Mode
```bash
python run_sovl.py --config sovl_config.json --mode train --epochs 10
```

### Generation Mode
```bash
python run_sovl.py --config sovl_config.json --mode generate
```

### Dream Mode
```bash
python run_sovl.py --config sovl_config.json --mode dream
```

## Step 4: Monitoring and Interaction

The system provides several ways to monitor and interact with the running instance:

### Log Files
- Logs are automatically created in the `output` directory
- Format: `sovl_run_YYYYMMDD_HHMMSS.log`
- Log rotation is enabled (max 5 files, 10MB each)
- In-memory logs are kept (max 1000 entries)
- Old logs are automatically compressed

### CLI Commands
Once the system is running, you can use these commands:

#### System Control
- `quit`/`exit`: Gracefully shut down the system
- `save`: Save current system state
- `load`: Load a saved system state
- `reset`: Reset the system to initial state
- `status`: Check system health (memory, training, etc.)
- `help`: Show available commands
- `monitor`: Control monitoring systems
  - `monitor system [start|stop|status]`: Control system monitoring
  - `monitor traits [start|stop|status]`: Control traits monitoring

#### Training and Generation
- `train [epochs]`: Start training (e.g., `train 10`)
- `dream`: Run a dream cycle to consolidate memories
- `generate`: Generate text based on current state
- `echo`: Echo back input with system interpretation
- `mimic`: Mimic a given style or pattern
- `muse`: Enter creative exploration mode
- `flare`: Trigger a burst of creative activity
- `debate`: Engage in internal debate
- `spark`: Generate new ideas or connections
- `reflect`: Enter deep reflection mode

#### Memory Management
- `memory`: View memory usage and statistics
- `recall`: Access stored memories
- `forget`: Clear specific memories
- `recap`: Get a summary of recent memories

#### System Management
- `log view [n]`: Show recent logs
- `config get/set`: View or modify configurations
- `panic`: Force a system reset if errors occur
- `glitch`: Simulate error conditions for testing
- `tune`: Fine-tune system parameters
- `rewind`: Revert to a previous state
- `history`: View command history

### Monitoring Features
The system includes comprehensive monitoring capabilities:

1. **Resource Monitoring**:
   - Memory usage tracking
   - GPU utilization
   - CPU load
   - Disk space monitoring

2. **Training Monitoring**:
   - Loss tracking
   - Learning rate changes
   - Gradient statistics
   - Validation metrics

3. **State Monitoring**:
   - System state changes
   - Component health
   - Error tracking
   - Performance metrics

4. **Traits Monitoring**:
   - Behavior patterns
   - Learning progress
   - Adaptation metrics
   - Personality traits

5. **Memory Monitoring**:
   - Memory usage patterns
   - Recall success rates
   - Memory consolidation
   - Forgetting patterns

## Step 5: Troubleshooting

### Common Issues and Solutions

1. **Model Loading Errors**:
   - Verify the model path in `sovl_config.json`
   - Ensure the model is compatible with `transformers`
   - Check for sufficient disk space
   - Verify CUDA installation if using GPU
   - Check model quantization settings

2. **Memory Issues**:
   - Adjust `max_memory_mb` in memory_config
   - Monitor memory usage with `memory` command
   - Consider reducing batch size
   - Enable gradient checkpointing
   - Use gradient accumulation
   - Check for memory leaks with `monitor` command

3. **Training Issues**:
   - Adjust learning rate and warmup steps
   - Modify gradient accumulation steps
   - Check gradient norm clipping
   - Verify batch size compatibility
   - Monitor loss patterns
   - Check validation metrics

4. **State Management Issues**:
   - Verify state save interval
   - Check backup file limits
   - Monitor state transitions
   - Verify component serialization
   - Check state restoration

5. **Monitoring Issues**:
   - Check log rotation settings
   - Verify monitoring intervals
   - Check resource thresholds
   - Monitor trait changes
   - Verify error tracking

### Error Logging
- Check the latest log file in the `output` directory
- Use `log view` command to see recent errors
- Monitor system status with `status` command
- Check component health with `monitor` command
- Review error context in logs

### Recovery Procedures
1. **Graceful Shutdown**:
   - Use `quit` or `exit` command
   - Wait for state to be saved
   - Check for clean shutdown in logs

2. **State Recovery**:
   - Use `load` command to restore state
   - Verify component initialization
   - Check system health after recovery

3. **Error Recovery**:
   - Use `panic` command if needed
   - Check error context in logs
   - Verify system state after recovery
   - Monitor for recurring errors

4. **Resource Recovery**:
   - Monitor memory usage
   - Check GPU utilization
   - Verify disk space
   - Monitor CPU load

## Step 6: Best Practices for Testing

1. **Start Small**:
   - Begin with CPU mode to verify basic functionality
   - Use small batch sizes initially
   - Test with minimal epochs
   - Verify basic commands first

2. **Monitor Resources**:
   - Watch memory usage during training
   - Check GPU utilization
   - Monitor disk space for checkpoints
   - Track CPU load patterns

3. **Regular Checkpoints**:
   - Set appropriate checkpoint intervals
   - Keep backup files for recovery
   - Test checkpoint loading functionality
   - Verify state restoration

4. **Error Handling**:
   - Test graceful shutdown
   - Verify error recovery mechanisms
   - Check log rotation and cleanup
   - Monitor error patterns

5. **System Testing**:
   - Test all operation modes
   - Verify command functionality
   - Check monitoring features
   - Test state management
   - Verify memory operations



