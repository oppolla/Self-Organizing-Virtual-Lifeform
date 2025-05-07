# Issue Log

## Lack of Atomicity in State Saving/Loading

- **Affected File:** `sovl_system/sovl_state.py`
- **Affected Lines:** ~1087-1119 (`save_state`), ~1120-1182 (`load_state`)
- **Function:** `StateManager.save_state`, `StateManager.load_state`
- **Summary:** The `save_state` function writes state to two separate files (`_state.json` and `_tensors.pt`) sequentially. If an interruption (e.g., crash, power loss) occurs between writing the JSON file and successfully writing the tensor file, the persisted state becomes inconsistent. The `load_state` function attempts to handle a missing or corrupted tensor file by logging an error and continuing without the tensor data.
- **Impact:** Loading this inconsistent state can lead to unexpected runtime errors, incorrect behavior, or crashes later when the code expects the tensor data (referenced in the JSON) to be present. This can break any functionality that relies on correctly loading persisted state.
- **Severity:** High (Potential Show-Stopper)
- **Suggested Solution:** Implement atomic saving. Write both JSON and tensor data to temporary files first. Only after both temporary files are successfully written, rename them to their final target filenames (e.g., `_state.json`, `_tensors.pt`). This ensures that either the complete state is saved or no state (or only the previous complete state) is present. Additionally, make `load_state` either fail explicitly if the tensor file is missing when the JSON file exists, or ensure that all subsequent code paths correctly handle the possibility of missing tensor data after a partial load.

## Potential for Deadlock in Multi-threaded Operations (Revised)

- **Affected Files:** 
    - `sovl_system/sovl_trainer.py` (TrainingWorkflowManager)
    - `sovl_system/sovl_dreamer.py` (Dreamer)
    - `sovl_system/sovl_main.py` (SOVLSystem, for independent dream cycle invocation)
- **Affected Functions:** 
    - `TrainingWorkflowManager.run_gestation_cycle`
    - `TrainingWorkflowManager._get_resource_lock`
    - `SOVLSystem.run_dream_cycle_with_abort` (which calls `Dreamer.run_dream_cycle` in a thread)
- **Summary:** 
    The `TrainingWorkflowManager.run_gestation_cycle` acquires multiple `RLock` resource locks (e.g., for 'state', 'dreamer', 'model_manager'). While the specific previously hypothesized producer-consumer deadlock involving `MetadataProcessor` directly acquiring the 'state' lock during scribe processing seems unlikely (as the `MetadataProcessor` instance used by `Scriber` is not initialized with a direct `state_accessor` that would use these locks), a general risk remains due to complex lock interactions in a multi-threaded environment.
    If other system components (e.g., an independently triggered dream cycle via `SOVLSystem.run_dream_cycle_with_abort`, or other background tasks) attempt to acquire the same conceptual resources or use related locks in a different order than `run_gestation_cycle`, deadlocks are still possible. For example, if gestation holds lock A and tries for lock B, while another thread holds lock B and tries for lock A.
    The `TrainingWorkflowManager` also has its own `run_dream_cycle` method that acquires the 'dreamer' lock; if this were invoked concurrently with gestation's use of the 'dreamer' lock, contention would occur.
- **Impact:** A deadlock would cause involved threads to hang, potentially halting training, dreaming, or other core system functions, making the application unresponsive.
- **Severity:** Medium (was High). Downgraded because the most direct producer-consumer deadlock path via `MetadataProcessor` seems unlikely. However, general multi-threading with multiple locks still warrants caution.
- **Suggested Solution:** 
    1. **Comprehensive Lock Audit:** Perform a full audit of all lock acquisition and release patterns across all threads and components to ensure a strict global order for any shared locks, or to verify that separate locks for separate conceptual resources are truly independent.
    2. **Minimize Lock Scope:** Keep critical sections protected by locks as short as possible.
    3. **State Access Patterns:** Clarify and ensure that components like `MetadataProcessor`, if they need state, access it in a non-blocking way or via snapshots if they are part of a processing pipeline that could be blocked by a producer holding state locks.
    4. **Test Concurrency:** Implement stress tests specifically designed to provoke race conditions and deadlocks.

## Potential OOM Risk due to Aggressive Manual Memory Management

- **Affected File:** `sovl_system/sovl_trainer.py`
- **Affected Lines:** Throughout `TrainingWorkflowManager.run_gestation_cycle` (approx. lines 310-532)
- **Function:** `TrainingWorkflowManager.run_gestation_cycle`
- **Summary:** The `run_gestation_cycle` method contains numerous explicit calls to `del` on variables (optimizer, outputs, loss), `gc.collect()`, and `torch.cuda.empty_cache()`. This pattern is repeated multiple times within the function: before and within the main loop processing scaffolds, and in finally blocks.
- **Impact:** While explicit memory management is sometimes required for GPU resources, its pervasive and repeated use here suggests the system may be operating very close to its memory limits. If these manual efforts are insufficient or mask underlying memory leaks, the application could crash due to Out-of-Memory (OOM) errors, especially under heavy load or with larger models/data. An OOM error would halt the gestation/training process.
- **Severity:** High (Potential Show-Stopper)
- **Suggested Solution:** 
    1. **Profiling:** Profile memory usage (especially GPU memory) during the gestation cycle to identify true bottlenecks and periods of high consumption.
    2. **Optimize Memory Usage:** Instead of relying heavily on manual `gc.collect()` and `del`, investigate if tensor operations can be made more memory-efficient (e.g., using in-place operations where safe, ensuring tensors are moved to CPU when not needed on GPU, reducing batch sizes if necessary based on available memory).
    3. **Review Object Lifecycles:** Ensure objects (especially PyTorch models, tensors, optimizers) are naturally going out of scope and being reclaimed by Python's garbage collector when no longer needed, reducing the necessity for manual `del`.
    4. **Context Managers:** Use context managers (`with ...:`) for resources that need deterministic cleanup, if applicable, rather than manual `del` in `finally` blocks where possible.

## ScaffoldTokenMapper: Initialization Performance Issue

- **Affected File:** `sovl_system/sovl_scaffold.py`
- **Affected Lines:** Within `ScaffoldTokenMapper._build_token_map` (approx. lines 469-529), specifically the "legacy char similarity fallback" loop.
- **Function:** `ScaffoldTokenMapper._build_token_map`
- **Summary:** The `_build_token_map` method, which runs during `ScaffoldTokenMapper` initialization, has a fallback step for character-based similarity. This step involves iterating through the entire scaffold tokenizer vocabulary for each base tokenizer token that hasn't been mapped by prior strategies. If vocabularies are large (e.g., 30k-50k+ tokens), this nested loop (effectively `O(V_base * V_scaffold * L)`) can be extremely computationally expensive.
- **Impact:** This can lead to excessively long system startup times if many tokens require this fallback, potentially making the system seem unresponsive or unusable if initialization takes minutes or even hours. This is a show-stopper for practical use if tokenizers are significantly different.
- **Severity:** High (Potential Show-Stopper for startup/usability)
- **Suggested Solution:** 
    1. **Optimize Similarity Search:** Replace the naive O(V_scaffold) lookup with more efficient methods for finding similar tokens, such as pre-computed data structures (e.g., BK-trees for Levenshtein distance if applicable, or locality-sensitive hashing for character n-grams).
    2. **Limit Fallback Scope:** Consider restricting the character similarity fallback to only a subset of tokens or only if other, faster heuristics (like subword matching) fail and embeddings are unavailable.
    3. **Caching:** If `_build_token_map` must be slow, cache the resulting `token_map` to disk after the first successful initialization to avoid re-computing it on every startup, assuming tokenizers don't change.
    4. **Progress Reporting:** For such a potentially long operation, provide clear progress reporting to the user.

## ScaffoldTokenMapper: Mapping Quality and Error Trigger

- **Affected File:** `sovl_system/sovl_scaffold.py`
- **Affected Lines:** `ScaffoldTokenMapper._build_token_map` (approx. lines 469-529) for how 'weight' (used as confidence) is assigned, and `ScaffoldTokenMapper.tokenize_and_map` (approx. lines 613-667) for the error trigger.
- **Function:** `ScaffoldTokenMapper._build_token_map`, `ScaffoldTokenMapper.tokenize_and_map`
- **Summary:** During `_build_token_map`, mappings are assigned a 'weight' based on the fallback strategy used (e.g., 1.0 for exact match, 0.5 for char similarity, 0.1 for UNK). The `tokenize_and_map` function later uses this 'weight' as a confidence score. If more than 20% of the tokens in a given prompt map with a confidence below a fixed threshold (implicitly, if their 'weight' reflects a less reliable strategy), a `ScaffoldError` is raised.
- **Impact:** If the base and scaffold tokenizers are significantly different, many tokens might rely on low-'weight' fallback strategies. This could cause `tokenize_and_map` to frequently throw `ScaffoldError` for many legitimate prompts, rendering the scaffold functionality unusable. The static 'weight' per strategy may not accurately capture the true quality of all individual mappings made by that strategy.
- **Severity:** High (Potential Show-Stopper for core scaffold functionality)
- **Suggested Solution:** 
    1. **Dynamic Confidence Scoring:** Instead of static 'weights' per strategy, implement a more nuanced confidence scoring mechanism for each individual token mapping, perhaps based on the actual similarity score achieved (e.g., Levenshtein score, embedding similarity score if used).
    2. **Configurable Threshold:** Make the 20% low-confidence ratio and the minimum confidence threshold itself configurable, allowing users to tune it based on their specific tokenizers and tolerance for mapping uncertainty.
    3. **Review Fallback Necessity:** If fallbacks are frequently resulting in low overall confidence, it might indicate that the chosen base and scaffold tokenizers are too dissimilar for effective mapping, potentially requiring a design reconsideration for tokenizer choice or a more advanced mapping technique.
    4. **Granular Error Reporting:** When `ScaffoldError` is raised, provide more detailed information about which tokens failed to map with sufficient confidence.

## CrossAttentionInjector: Fragile Commit Step after Injection

- **Affected File:** `sovl_system/sovl_scaffold.py`
- **Affected Lines:** Within `CrossAttentionInjector.inject` (approx. lines 1348-1351).
- **Function:** `CrossAttentionInjector.inject`
- **Summary:** After modifying a deep copy (`model_copy`) of the `base_model` by injecting layers, the `inject` method attempts to commit these changes back to the original `base_model`. It does this using `setattr(base_model, name, module)` while iterating through `model_copy.named_modules()`. This appears to perform a shallow copy of the (potentially new or wrapped) modules from the copy back to the original.
- **Impact:** Shallow copying modules between model instances, especially after deep copying and modification, is fragile. It can lead to unexpected parameter sharing, incorrect computation graphs if internal references within modules are not handled correctly, or runtime errors if the `model_copy` is garbage collected later. This could cause the model to behave incorrectly or crash. **Show-stopper** risk.
- **Severity:** High.
- **Suggested Solution:** 
    1. **Direct Modification:** Consider performing the injection directly on the `base_model` within the main `try-except` block of the `inject` method. If injection fails for a layer or verification fails, the function can still return the original (partially modified or unmodified) `base_model` or raise an exception, avoiding the risky commit step. 
    2. **Careful State Dict Transfer:** If modification must happen on a copy, instead of using `setattr`, use PyTorch's recommended state dictionary manipulation (`load_state_dict`) to transfer the *parameters* from the modified copy back to the original model structure, ensuring the computation graph and module instances of the original model are preserved correctly.

## CrossAttentionInjector: Shared Projection Layer Across Injections

- **Affected File:** `sovl_system/sovl_scaffold.py`
- **Affected Lines:** Within `CrossAttentionInjector._inject_single_layer` (approx. lines 1420-1430).
- **Function:** `CrossAttentionInjector._inject_single_layer`
- **Summary:** If the hidden sizes of the base and scaffold models differ, `_inject_single_layer` creates a linear projection layer (`nn.Linear`) and potentially a LayerNorm, storing them as instance variables (`self._scaffold_proj`, `self._scaffold_proj_norm`). Because these are instance variables, the *same* projection layer instance (with the same trained weights) appears to be reused if cross-attention is injected into multiple layers of the base model during a single `inject` call.
- **Impact:** Sharing projection weights across different layers of a transformer is architecturally unusual. Each layer typically processes information differently. Forcing them to use the same projection from the scaffold might hinder the model's ability to learn effectively, constrain the representations passed between layers, and potentially degrade overall performance or learning capacity. While not necessarily a crash, it could significantly impair the intended functionality.
- **Severity:** Medium to High (Potential functionality impairment).
- **Suggested Solution:** 
    1. **Layer-Specific Projections:** If projection is needed, create a *new* instance of the projection layer (`nn.Linear`, `nn.LayerNorm`) for *each* `CrossAttentionLayer` that is injected. Ensure these new projection layers are correctly integrated into the model structure and their parameters are registered for training/saving/loading.
    2. **Configuration:** Make the use of shared vs. independent projection layers a configurable option if there's a specific reason for the shared design.
