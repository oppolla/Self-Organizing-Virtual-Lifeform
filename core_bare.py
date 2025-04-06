from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType  # Import PEFT components
import copy
import time
import random
from train_data import TRAIN_DATA

# VRAM Monitor
def print_memory_stats(label=""):
    """Prints current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"\n--- Memory Stats ({label}) ---")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(torch.cuda.memory_summary(abbreviated=True))
    else:
        print("(CPU mode - no GPU memory stats)")

# --- Configuration (Bare Bones + LoRA) ---
BASE_MODEL_NAME = "gpt2"  # ~117M params (Frozen)
SCAFFOLD_MODEL_NAME = "gpt2"  # ~117M params (LoRA Fine-tuned)
CROSS_ATTN_LAYERS = [5, 7]  # Indices for GPT-2 layers (0-11)
VALID_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

# LoRA Configuration
LORA_RANK = 8
LORA_ALPHA = 16  # Typically 2*LORA_RANK
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]  # Adjust based on model architecture if needed

# Training Config
LEARNING_RATE = 3e-4  # Higher LR common for LoRA
TRAIN_EPOCHS = 3  # Number of epochs to train on the mini-dataset
BATCH_SIZE = 1  # Keep batch size small due to potential memory constraints
MAX_SEQ_LENGTH = 128  # Max sequence length for training/inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Train Data Validation Split
random.seed(RANDOM_SEED)
random.shuffle(TRAIN_DATA)
split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
print(f"Dataset split: {len(TRAIN_DATA)} train, {len(VALID_DATA)} validation")

# --- Simplified Cross-Attention Module ---
class SimpleCrossAttentionFuser(nn.Module):
    """
    Minimalist Fuser: Applies gated cross-attention.
    Assumes base_dim == scaffold_dim.
    """
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.influence_weight = 1.0

    def set_influence_weight(self, weight):
        """Set influence weight (0-1 scale)"""
        self.influence_weight = max(0.0, min(1.0, weight))

    def forward(self, base_hidden_state, scaffold_context):
        pooled_scaffold_context = scaffold_context.mean(dim=1, keepdim=True)
        attn_output, _ = self.cross_attention(
            query=base_hidden_state,
            key=pooled_scaffold_context,
            value=pooled_scaffold_context
        )
        gate_values = self.gate(base_hidden_state)
        fused_state = base_hidden_state + gate_values * (attn_output * self.influence_weight)
        fused_state = self.layer_norm(fused_state)
        return fused_state

# --- Bare Bones System with Learning ---
class BareBonesDMAO_Learn:
    def __init__(self):
        # --- Load Base Model (Frozen) ---
        print(f"Loading base model: {BASE_MODEL_NAME}")
        self.base_config = AutoConfig.from_pretrained(BASE_MODEL_NAME)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, config=self.base_config
        ).to(DEVICE)
        print_memory_stats("After base model load")
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False
        print(f"Base model '{BASE_MODEL_NAME}' loaded and frozen.")

        # --- Load Scaffold Model ---
        print(f"Loading scaffold model: {SCAFFOLD_MODEL_NAME}")
        self.scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
            SCAFFOLD_MODEL_NAME, config=self.scaffold_config
        )
        print(f"Scaffold model '{SCAFFOLD_MODEL_NAME}' loaded.")
        print_memory_stats("After scaffold model load")

        # --- Apply LoRA to Scaffold Model ---
        print("Applying LoRA adapters to scaffold model...")
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.scaffold_model = get_peft_model(scaffold_model_raw, lora_config)
        self.scaffold_model.to(DEVICE)
        print("LoRA adapters applied. Trainable scaffold parameters:")
        self.scaffold_model.print_trainable_parameters()

        # --- Load Tokenizers ---
        print(f"Loading base tokenizer from: {BASE_MODEL_NAME}")
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"Base tokenizer loaded (Vocab size: {self.base_tokenizer.vocab_size}).")

        print(f"Loading scaffold tokenizer from: {SCAFFOLD_MODEL_NAME}")
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)
        print(f"Scaffold tokenizer loaded (Vocab size: {self.scaffold_tokenizer.vocab_size}).")

        # --- Handle Padding Tokens ---
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            print(f"Base tokenizer pad token set to EOS token: '{self.base_tokenizer.eos_token}' (ID: {self.base_tokenizer.eos_token_id})")

        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
            print(f"Scaffold tokenizer pad token set to EOS token: '{self.scaffold_tokenizer.eos_token}' (ID: {self.scaffold_tokenizer.eos_token_id})")

        self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
        self.scaffold_model.config.pad_token_id = self.scaffold_tokenizer.pad_token_id

        try:
            if hasattr(self.scaffold_model, 'base_model') and hasattr(self.scaffold_model.base_model, 'model') and hasattr(self.scaffold_model.base_model.model, 'config'):
                self.scaffold_model.base_model.model.config.pad_token_id = self.scaffold_tokenizer.pad_token_id
            elif hasattr(self.scaffold_model, 'model') and hasattr(self.scaffold_model.model, 'config'):
                self.scaffold_model.model.config.pad_token_id = self.scaffold_tokenizer.pad_token_id
        except AttributeError:
            print("Could not set pad_token_id on underlying scaffold model config.")

                # --- Build Token Mapping ---
        def build_token_map(base_tokenizer, scaffold_tokenizer):
            """Build mapping with support for multi-token sequences"""
            base_vocab = base_tokenizer.get_vocab()
            scaffold_vocab = scaffold_tokenizer.get_vocab()
            token_map = {}

            for base_token, base_id in base_vocab.items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(
                    normalized, 
                    add_special_tokens=False,
                    max_length=3,  # Limit expansion length
                    truncation=True
                ) or [scaffold_tokenizer.unk_token_id]
                token_map[base_id] = scaffold_ids

            return token_map

        self.token_map = build_token_map(self.base_tokenizer, self.scaffold_tokenizer)

        # Special token mapping
        self.special_token_map = {
            self.base_tokenizer.pad_token_id: self.scaffold_tokenizer.pad_token_id,
            self.base_tokenizer.eos_token_id: self.scaffold_tokenizer.eos_token_id or self.scaffold_tokenizer.sep_token_id,
            self.base_tokenizer.unk_token_id: self.scaffold_tokenizer.unk_token_id,
        }
        self.scaffold_unk_id = self.scaffold_tokenizer.unk_token_id

        # --- Inject Cross-Attention ---
        print("Injecting cross-attention layers...")
        self._insert_cross_attention()
        print("Cross-attention injection complete.")

        # Temporary storage for scaffold context
        self._temp_scaffold_context: Optional[torch.Tensor] = None

        # --- Setup Optimizer (placeholder) ---
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.best_valid_loss = float('inf')
        self.patience = 0
        self.max_patience = 2
        print("Initialization complete. Optimizer needs setup before training.")

    def set_scaffold_influence(self, weight):
        """Set the influence weight for all cross-attention layers (0-1 scale)"""
        base_layers = self._get_model_layers(self.base_model)
        for layer_idx in CROSS_ATTN_LAYERS:
            if layer_idx < len(base_layers):
                modified_layer = base_layers[layer_idx]
                if hasattr(modified_layer, 'cross_attn'):
                    modified_layer.cross_attn.set_influence_weight(weight)

    def _get_model_layers(self, model):
        """Helper to get the main list of transformer layers"""
        actual_model = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
            return actual_model.transformer.h
        elif hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
            return actual_model.model.layers
        elif hasattr(actual_model, 'layers'):
            return actual_model.layers
        elif hasattr(actual_model, 'decoder') and hasattr(actual_model.decoder, 'layers'):
            return actual_model.decoder.layers
        else:
            raise ValueError(f"Cannot determine layer structure for model: {actual_model.__class__.__name__}")

    def _insert_cross_attention(self):
        """Injects the simplified cross-attention fuser into specified base model layers."""
        base_layers = self._get_model_layers(self.base_model)
        num_base_layers = len(base_layers)
        hidden_dim = self.base_config.hidden_size
        num_heads = self.base_config.num_attention_heads

        if self.scaffold_config.hidden_size != hidden_dim:
            print(f"Warning: Scaffold hidden size != base hidden size. Add projection if needed.")

        print(f"Injecting CrossAttentionFuser at layers: {CROSS_ATTN_LAYERS}")

        for layer_idx in CROSS_ATTN_LAYERS:
            if layer_idx >= num_base_layers:
                print(f"Warning: Layer index {layer_idx} out of bounds ({num_base_layers} layers). Skipping.")
                continue

            original_layer = base_layers[layer_idx]
            cross_attn_fuser = SimpleCrossAttentionFuser(
                hidden_dim=hidden_dim,
                num_heads=num_heads
            ).to(DEVICE)
            for param in cross_attn_fuser.parameters():
                param.requires_grad = False

            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn_module, parent_system):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn_module
                    self._parent_system = parent_system

                def forward(self, hidden_states, **kwargs):
                    outputs = self.orig_layer(hidden_states, **kwargs)
                    base_hidden_state_output = outputs[0] if isinstance(outputs, tuple) else outputs
                    scaffold_context = getattr(self._parent_system, '_temp_scaffold_context', None)

                    if scaffold_context is not None:
                        scaffold_context = scaffold_context.to(base_hidden_state_output.device)
                        fused_hidden_state = self.cross_attn(base_hidden_state_output, scaffold_context)
                        final_outputs = (fused_hidden_state,) + outputs[1:] if isinstance(outputs, tuple) else fused_hidden_state
                        return final_outputs
                    else:
                        return outputs

            base_layers[layer_idx] = ModifiedLayer(original_layer, cross_attn_fuser, self)
            print(f"Successfully injected wrapper into layer {layer_idx}")

    def setup_optimizer(self, num_training_steps):
        """Sets up the optimizer and scheduler for LoRA training."""
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.scaffold_model.parameters()),
            lr=LEARNING_RATE
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        print("Optimizer and scheduler set up.")

    def map_sequence(self, base_input_ids):
        """Handle multi-token expansions with efficient padding"""
        batch_size = base_input_ids.size(0)
        max_expanded_len = MAX_SEQ_LENGTH * 3  # Allow reasonable expansion
        
        # Initialize tensor with padding tokens
        mapped_ids = torch.full(
            (batch_size, max_expanded_len), 
            self.scaffold_tokenizer.pad_token_id,
            dtype=torch.long,
            device=DEVICE
        )
        
        # Build sequences for each item in batch
        for batch_idx in range(batch_size):
            position = 0
            for base_id in base_input_ids[batch_idx]:
                mapped_tokens = self.special_token_map.get(
                    base_id.item(),
                    self.token_map.get(base_id.item(), [self.scaffold_unk_id])
                )
                
                # Add tokens until we reach max length
                for token in mapped_tokens:
                    if position >= max_expanded_len:
                        break
                    mapped_ids[batch_idx, position] = token
                    position += 1

        # Truncate to MAX_SEQ_LENGTH while preserving batch dimension
        return mapped_ids[:, :MAX_SEQ_LENGTH]

    def train_step(self, batch):
        """Performs a single training step."""
        if not self.optimizer:
            raise RuntimeError("Optimizer not set up. Call setup_optimizer first.")
        print_memory_stats("Train step start")

        self.scaffold_model.train()
        self.base_model.eval()

        prompts = [item['prompt'] for item in batch]
        completions = [item['completion'] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        base_tokenizer_output = self.base_tokenizer(
            full_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        base_input_ids = base_tokenizer_output.input_ids.to(DEVICE)
        base_attention_mask = base_tokenizer_output.attention_mask.to(DEVICE)

        prompts_base = self.base_tokenizer(
            prompts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        scaffold_input_ids = self.map_sequence(prompts_base.input_ids)
        scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()

        scaffold_inputs = {
            'input_ids': scaffold_input_ids,
            'attention_mask': scaffold_attention_mask
        }

        labels = base_input_ids.clone()
        labels[labels == self.base_tokenizer.pad_token_id] = -100
        prompt_lengths = [len(self.base_tokenizer(p).input_ids) for p in prompts]
        for i, prompt_len in enumerate(prompt_lengths):
            actual_prompt_len_in_batch = min(prompt_len, MAX_SEQ_LENGTH)
            labels[i, :actual_prompt_len_in_batch] = -100

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            scaffold_core_model = self.scaffold_model.base_model.transformer if hasattr(self.scaffold_model.base_model, 'transformer') else self.scaffold_model.base_model.model
            scaffold_outputs = scaffold_core_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            scaffold_hidden_states = scaffold_outputs.hidden_states[-1]
            print_memory_stats("After forward pass")

            self._temp_scaffold_context = scaffold_hidden_states

            outputs = self.base_model(
                input_ids=base_input_ids,
                attention_mask=base_attention_mask,
            )
            base_logits = outputs.logits

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(base_logits.view(-1, base_logits.size(-1)), labels.view(-1))

        accumulation_steps = 4
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss encountered. Skipping batch.")
            self._temp_scaffold_context = None
            return None

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        if hasattr(self, 'global_step'):
            if (self.global_step + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        else:
            print("Warning: global_step not defined. Running without accumulation logic.")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        print_memory_stats("After optimizer step")
        self._temp_scaffold_context = None
        return loss.item()

    def run_training_cycle(self, train_data, valid_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE):
        """Modified training loop with validation"""
        num_training_steps = (len(train_data) // batch_size) * epochs
        if num_training_steps == 0:
            print("Not enough data or epochs for training.")
            return

        self.setup_optimizer(num_training_steps)
        print(f"\n--- Starting Training ({epochs} epochs) ---")
        start_train_time = time.time()
        global_step = 0

        for epoch in range(epochs):
            print_memory_stats(f"Epoch {epoch + 1} start")
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_loss = 0
            steps_in_epoch = 0
            random.shuffle(train_data)

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i: i + batch_size]
                if not batch:
                    continue

                step_loss = self.train_step(batch)
                if step_loss is not None:
                    epoch_loss += step_loss
                    steps_in_epoch += 1
                    global_step += 1
                    if global_step % 1 == 0:
                        print(f"  Step {global_step}/{num_training_steps} | Loss: {step_loss:.4f}")
                else:
                    print(f"  Step {global_step}/{num_training_steps} | Skipped")

            valid_loss = self.validate_epoch(valid_data)
            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
            print(f"Epoch {epoch + 1} Stats:")
            print(f"  Train Loss: {avg_epoch_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print_memory_stats(f"Epoch {epoch + 1} end")

            if not hasattr(self, 'best_valid_loss'):
                self.best_valid_loss = float('inf')
                self.patience = 0
                self.max_patience = 2
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = 0
            else:
                self.patience += 1
                print(f"Patience: {self.patience}/{self.max_patience}")
                if self.patience >= self.max_patience:
                    print("Early stopping triggered.")
                    break

            if (epoch + 1) % 1 == 0:
                self.evaluate_generation_quality(num_samples=2)

        end_train_time = time.time()
        print(f"--- Training Finished ({end_train_time - start_train_time:.2f}s) ---")

    def has_repetition(self, output_ids, n=3):
        """Check for n-gram repetition in output_ids."""
        output_ids = output_ids.tolist()
        for i in range(len(output_ids) - n):
            if all(output_ids[i + j] == output_ids[i + j + n] for j in range(n)):
                return True
        return False

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, scaffold_weight=None, **kwargs):
        """Generates text with optional scaffold influence control"""
        print_memory_stats("Pre-generation")
        if scaffold_weight is not None:
            self.set_scaffold_influence(scaffold_weight)

        start_time = time.time()
        base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to(DEVICE)
        input_ids = base_inputs['input_ids']
        input_length = input_ids.shape[1]

        scaffold_base_inputs = self.base_tokenizer(
            prompt, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(DEVICE)
        scaffold_input_ids = self.map_sequence(scaffold_base_inputs.input_ids)
        scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()

        scaffold_inputs = {
            'input_ids': scaffold_input_ids,
            'attention_mask': scaffold_attention_mask
        }

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            scaffold_outputs = self.scaffold_model(
                **scaffold_inputs,
                output_hidden_states=True
            )
            actual_outputs = scaffold_outputs.hidden_states if hasattr(scaffold_outputs, 'hidden_states') else scaffold_outputs.base_model_output.hidden_states
            scaffold_hidden_states = actual_outputs[-1]

        self._temp_scaffold_context = scaffold_hidden_states

        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == 'cuda' else torch.bfloat16):
            outputs = self.base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.base_tokenizer.pad_token_id,
                eos_token_id=self.base_tokenizer.eos_token_id,
                **kwargs
            )

        self._temp_scaffold_context = None
        print_memory_stats("Post-generation")
        generated_ids = outputs[0][input_length:]
        if self.has_repetition(generated_ids, n=3):
            print("Warning: Repetition detected in output. Truncating at first repeat.")
            for i in range(len(generated_ids) - 3):
                if all(generated_ids[i + j] == generated_ids[i + j + 3] for j in range(3)):
                    generated_ids = generated_ids[:i + 3]
                    break
        response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)

        end_time = time.time()
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        return response

    @torch.no_grad()
    def validate_epoch(self, valid_data):
        """Validation loss calculation"""
        self.scaffold_model.eval()
        total_loss, batches = 0, 0

        for i in range(0, len(valid_data), BATCH_SIZE):
            batch = valid_data[i:i + BATCH_SIZE]
            if not batch:
                continue

            prompts = [item['prompt'] for item in batch]
            completions = [item['completion'] for item in batch]
            full_texts = [p + c for p, c in zip(prompts, completions)]

            prompts_base = self.base_tokenizer(
                prompts,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            scaffold_input_ids = self.map_sequence(prompts_base.input_ids)
            scaffold_attention_mask = (scaffold_input_ids != self.scaffold_tokenizer.pad_token_id).int()

            scaffold_inputs = {
                'input_ids': scaffold_input_ids,
                'attention_mask': scaffold_attention_mask
            }

            scaffold_outputs = self.scaffold_model.base_model.transformer(**scaffold_inputs)
            self._temp_scaffold_context = scaffold_outputs.last_hidden_state

            base_inputs = self.base_tokenizer(
                full_texts,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(DEVICE)
            labels = base_inputs.input_ids.clone()
            labels[labels == self.base_tokenizer.pad_token_id] = -100

            outputs = self.base_model(**base_inputs)
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)),
                                   labels.view(-1), ignore_index=-100)

            total_loss += loss.item()
            batches += 1
            self._temp_scaffold_context = None

        return total_loss / batches if batches > 0 else 0

    @torch.no_grad()
    def evaluate_generation_quality(self, num_samples=3):
        """Generate sample responses"""
        samples = random.sample(VALID_DATA, num_samples)
        print("\n=== Generation Evaluation ===")

        for example in samples:
            print(f"\nPrompt: {example['prompt']}")
            print(f"Expected: {example['completion']}")
            for weight in [0.0, 0.5, 1.0]:
                response = self.generate(example['prompt'], scaffold_weight=weight,
                                         max_new_tokens=60, temperature=0.7)
                print(f"w={weight}: {response}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nInitializing Bare Bones DMAO System with Learning...")
    try:
        dmao_system = BareBonesDMAO_Learn()
        print("\nSystem Ready.")
        print("Commands: 'quit', 'exit', 'train', or enter a prompt.")

        while True:
            user_cmd = input("\nEnter command or prompt: ")
            cmd = user_cmd.lower().strip()

            if cmd in ['quit', 'exit']:
                break
            elif cmd == 'train':
                dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
            elif not user_cmd:
                continue
            else:
                prompt = user_cmd
                gen_params = {
                    'temperature': 0.7,
                    'top_k': 50,
                    'do_sample': True
                }
                print("\n--- Generating Response ---")
                response = dmao_system.generate(prompt, max_new_tokens=60, **gen_params)
                print("\nResponse:")
                print(response)
                print("-" * 20)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        del dmao_system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nExiting.")

# DOCUMENTATION
# After training:
# response = system.generate("How to make coffee?", scaffold_weight=0.7)

# To completely disable scaffold influence:
# response = system.generate("Explain quantum physics", scaffold_weight=0.0)
