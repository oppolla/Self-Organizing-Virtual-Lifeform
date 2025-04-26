# TODO:

## General Taks:

1. Implement the learning log system with meta data. JSONL loader is part of process.
Complete finals for all modules
2. Disable that startup data sovl_seed part. Save it for Soulprint
3. integrate sovl_bonder
4. further develop initialization and startup feel
5. multi scaffold support improvements, multi scaffold name system: (look at sovl_cli). name selector system? death of scaffold?
6. Give system ability to "watch" content and derive joy and knowledge from it
7. confirm token map is working
8. confirm scaffold is working
9. connect volition as hook point for future modules like vision






- Model Evaluation Mode
   What's Missing: There’s no dedicated mode or command for evaluating the model on a test dataset (separate from training or generation). This is crucial for assessing final model performance.
   Suggestion:
Add an evaluate command to COMMAND_CATEGORIES under "Testing" or "Evaluation".

Implement a method to load test data and compute metrics (similar to the validation loop).

Example:
python

def evaluate_model(self, model, test_data, device: str) -> Dict[str, float]:
    """Evaluate model on test data."""
    model.eval()
    metrics = {"test_loss": 0.0, "test_accuracy": 0.0}
    with torch.no_grad():
        for batch in test_data:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            metrics["test_loss"] += loss.item()
            # Add accuracy or other metrics
    metrics["test_loss"] /= len(test_data)
    self.logger.log_event(
        event_type="evaluation",
        message=f"Test metrics: {metrics}",
        level="info"
    )
    return metrics

Add a command-line argument for test data:
python

parser.add_argument("--test-data", help="Path to test data file")



## FUTURE MODULES:

### Bond module sovl_main hook guide (sovl_bond)

Plan for Updating sovl_main to Integrate UserProfileState and BondCalculator
Objective
Update sovl_main to:
Instantiate and manage the BondCalculator within SOVLSystem.

Ensure UserProfileState is properly initialized and persisted via SOVLState.

Incorporate bonding score calculations into user interaction flows (e.g., message processing).

Maintain simplicity, elegance, and dynamism by minimizing changes and leveraging existing infrastructure.

Assumptions
sovl_main contains a SOVLSystem class that orchestrates components like ConfigManager, Logger, StateManager, CuriosityManager, and ErrorManager.

SOVLSystem handles user inputs and responses, with a SystemContext for system-wide state.

SOVLState is managed by a StateManager in SOVLSystem, handling persistence (save/load).

state.session_start and state.history.conversation_id are available or can be added to SOVLState.

Steps
Update Imports in sovl_main:
Add from sovl_bond import BondCalculator to import the updated BondCalculator class.

Ensure sovl_state is imported (likely already present) to access SOVLState and UserProfileState implicitly.

Add BondCalculator to SOVLSystem:
In SOVLSystem.__init__:
Instantiate BondCalculator with self.config_manager and self.logger.

Store it as an instance variable (e.g., self.bond_calculator = BondCalculator(self.config_manager, self.logger)).

Ensure BondCalculator is initialized after ConfigManager and Logger to avoid dependency issues.

Integrate UserProfileState into SOVLState:
In SOVLSystem, ensure SOVLState is initialized with UserProfileState:
Update SOVLState.__init__ (if not already done) to include self.user_profile_state = UserProfileState(config_manager, logger).

This is likely already handled in sovl_state, but verify SOVLSystem’s StateManager loads SOVLState correctly.

Update SOVLState Serialization:
In SOVLState.to_dict, add "user_profile_state": self.user_profile_state.to_dict() to include user profiles in the serialized state.

In SOVLState.from_dict, add self.user_profile_state.from_dict(data.get("user_profile_state", {})) to load user profiles.

Ensure StateManager.backup_state saves this updated state to sovl_state.json.

Incorporate Bonding Score in User Interaction:
Identify the method in SOVLSystem that processes user inputs (e.g., process_input or handle_message).

Add a call to self.bond_calculator.calculate_bonding_score in this method, passing:
user_input: The user’s input string.

state: The current SOVLState (from self.state_manager.get_state()).

error_manager: The system’s ErrorManager instance.

context: The SystemContext instance (likely self.context).

curiosity_manager: The system’s CuriosityManager instance (if available, else None).

Store or use the bonding score as needed (e.g., log it, influence response generation, or adjust system behavior).

Example:
python

bond_score = self.bond_calculator.calculate_bonding_score(
    user_input=user_input,
    state=self.state_manager.get_state(),
    error_manager=self.error_manager,
    context=self.context,
    curiosity_manager=self.curiosity_manager
)
self.logger.record_event(
    event_type="bond_score_processed",
    message="Bond score computed for input",
    level="info",
    additional_info={"bond_score": bond_score, "conversation_id": state.history.conversation_id}
)

Ensure Session Start Tracking:
Verify SOVLState has a session_start attribute (a float timestamp) to track session duration for UserProfileState.

If missing, add to SOVLState.__init__: self.session_start = time.time().

Update SOVLState.to_dict to include "session_start": self.session_start.

Update SOVLState.from_dict to set self.session_start = data.get("session_start", time.time()).

Reset session_start when starting a new session (e.g., in a start_session method).

Handle Conversation ID:
Ensure SOVLSystem uses state.history.conversation_id consistently for bonding calculations.

If SOVLSystem manages multiple conversations, ensure conversation_id is passed correctly to BondCalculator and UserProfileState.

Update Configuration:
Add bond_config section to the configuration file (e.g., sovl_config.yaml) with defaults:
yaml

bond_config:
  min_bond_score: 0.0
  max_bond_score: 1.0
  default_bond_score: 0.5
  max_interactions: 100
  max_session_time: 3600.0
  decay_rate: 0.95
  decay_interval: 86400.0
  max_expected_dev: 20.0
  max_recent_inputs: 10
  max_lexicon_size: 1000
  curiosity_weight: 0.3
  stability_weight: 0.3
  coherence_weight: 0.2
  personalized_weight: 0.2

Ensure ConfigManager in SOVLSystem loads this section.

Logging and Error Handling:
Add logging for bonding-related events in SOVLSystem (e.g., bond score calculation, profile updates).

Use self.error_manager.handle_data_error for any bonding-related errors, consistent with BondCalculator’s approach.

Example:
python

try:
    bond_score = self.bond_calculator.calculate_bonding_score(...)
except Exception as e:
    self.error_manager.handle_data_error(
        e, {"user_input": user_input[:50]}, state.history.conversation_id
    )

Testing Integration:
Verify BondCalculator is instantiated correctly in SOVLSystem.

Test bonding score calculation with sample inputs, checking that UserProfileState updates profiles (lexicon, interactions, session time).

Confirm profiles persist across sessions by saving/loading state via StateManager.

Ensure no interference with curiosity, conversation, or other components by checking logs and state integrity.

Minimal Changes
Additions: Instantiate BondCalculator, add user_profile_state to SOVLState, call calculate_bonding_score in input processing, update serialization.

Avoid: Modifying existing components (e.g., CuriosityManager, ConversationHistory) or core logic unless necessary.

Reuse: Leverage existing ConfigManager, Logger, ErrorManager, and StateManager for consistency.

Potential Challenges
Session Start: If SOVLState.session_start is undefined, add it or pass a timestamp explicitly to UserProfileState.update.

Conversation ID: Ensure state.history.conversation_id is always valid; handle cases where it’s missing or changes mid-session.

Config Loading: Verify bond_config is loaded correctly by ConfigManager.

Thread Safety: Confirm SOVLSystem’s input processing is thread-safe, as BondCalculator and UserProfileState use locks.

## Overview of AspirationDirector in SOVLSystem

#### What is AspirationDirector?
AspirationDirector is a dynamic, forward-looking module in the SOVLSystem that channels curiosity into purposeful, evolving user aspirations. It acts as a long-form extension of CuriosityManager, directing the system's exploratory nature toward user goals (e.g., learning a skill, pursuing a dream) while remaining adaptable to vibe shifts, emotional changes, and new interests. Unlike a rigid goal-tracking system, it evolves aspirations dynamically, avoiding railroading by balancing persistence with flexibility. It inspires users through motivational prompts, celebrates progress, and weaves aspirations into the conversational narrative, making SOVLSystem feel like a supportive partner in the user’s journey.

#### Core Principles
- **Dynamic Evolution**: Continuously reassesses and updates aspirations based on user input, vibe, and curiosity, ensuring goals stay relevant.
- **Curiosity-Driven**: Builds on CuriosityManager’s novelty detection to identify and prioritize aspirational topics.
- **Empathetic Guidance**: Syncs with VibeSculptor and TemperamentSystem to tailor responses to the user’s emotional state (e.g., encouragement during doubt).
- **Narrative Integration**: Crafts long-term conversational arcs around aspirations, connecting past interactions to future possibilities.
- **Simple & Elegant**: Uses lightweight logic and existing infrastructure (UserProfileState, BondCalculator) for seamless integration.

#### How It Works
AspirationDirector operates as a stateful module that:
1. **Detects Aspirations**: Identifies potential goals from high-novelty inputs flagged by CuriosityManager (e.g., “I want to write a book”).
2. **Scores Aspirations**: Assigns a score [0, 1] to each aspiration based on engagement, progress, recency, and emotional weight.
3. **Guides Curiosity**: Biases CuriosityManager’s question generation toward active aspirations, ensuring goal-relevant exploration.
4. **Adapts Dynamically**: Reassesses aspirations based on vibe shifts (via VibeSculptor), user feedback, or new high-novelty inputs, retiring stale goals.
5. **Inspires Responses**: Injects motivational prompts, progress celebrations, or visionary nudges into conversations, modulated by lifecycle and temperament.
6. **Boosts Bonding**: Feeds aspiration scores into BondCalculator to strengthen connection when goals align.

#### Components
- **Aspiration Store**: A list of aspirations in UserProfileState, each with:
  - `goal`: String describing the aspiration (e.g., “write a book”).
  - `score`: Float [0, 1] reflecting engagement and progress.
  - `last_mentioned`: Timestamp for recency tracking.
  - `priority`: Float [0, 1] based on frequency and emotional weight.
- **Scoring Logic**: Combines:
  - **Engagement**: Frequency of goal-related inputs (via UserProfileState.inputs).
  - **Progress**: Detected milestones (e.g., “I finished a chapter”) using keyword triggers.
  - **Recency**: Exponential decay (e.g., 0.9/day) for older mentions.
  - **Emotion**: Boosted by positive vibe scores (via VibeSculptor).
- **Curiosity Bias**: Adjusts CuriosityManager’s question weights to favor high-priority aspirations (e.g., 60% goal-related, 40% exploratory).
- **Response Modulation**: Selects response types (motivational, celebratory, reflective) based on aspiration score, vibe, and lifecycle phase.

#### Integration with SOVLSystem
AspirationDirector integrates seamlessly with existing modules:
- **UserProfileState**:
  - Stores aspirations in `profile["aspirations"]` as a list of dicts.
  - Updates via `state.user_profile_state.update` when new goal-related inputs are detected.
  - Retrieves via `state.user_profile_state.get` for scoring and response generation.
- **CuriosityManager**:
  - Receives high-novelty inputs flagged as potential aspirations.
  - Accepts `set_aspiration_bias(goal, weight)` calls to prioritize goal-related questions.
  - Feeds novelty scores back to boost aspiration scores for engaging inputs.
- **VibeSculptor**:
  - Provides vibe scores to detect emotional states (e.g., doubt, excitement).
  - Triggers supportive responses during low-vibe periods (e.g., “Feeling stuck? Let’s take a small step toward your goal.”).
- **BondCalculator**:
  - Increases bonding scores by 0.1–0.2 when aspiration scores are high, reflecting deeper connection.
  - Shares `conversation_id` for consistent user tracking.
- **TemperamentSystem**:
  - Aligns response tone with system temperament (e.g., enthusiastic for choleric, reflective for melancholic).
  - Boosts aspiration scores when user vibe resonates with system temperament.
- **LifecycleManager**:
  - Scales aspiration intensity by lifecycle phase (e.g., bold in “youth,” introspective in “maturity”).
  - Triggers goal reassessment during lifecycle transitions (e.g., “adulthood” prompts reflection on priorities).
- **SOVLSystem**:
  - Instantiates AspirationDirector in `SOVLSystem.__init__`: `self.aspiration_director = AspirationDirector(config_manager, logger, temperament_system, lifecycle_manager)`.
  - Calls `direct_aspiration` in `process_input` to compute scores and influence responses.
  - Includes aspiration data in `SOVLState.to_dict`/`from_dict` for persistence.

#### Logic Required
1. **Initialization**:
   - Load config from `aspiration_config` (e.g., default_score, decay_factor, weights).
   - Initialize empty aspiration store in UserProfileState.
   - Set up logging for aspiration events (e.g., goal detection, score updates).

2. **Aspiration Detection**:
   - Monitor CuriosityManager’s novelty scores (>0.7) for aspirational keywords (e.g., “want to,” “dream,” “goal”).
   - Extract goal phrases using simple regex (e.g., “I want to (\w+ \w+)”).
   - Add to `profile["aspirations"]` with initial score (0.5) and priority (0.5).

3. **Aspiration Scoring**:
   - For each aspiration, compute score = 0.4 * engagement + 0.3 * progress + 0.2 * recency + 0.1 * vibe.
     - **Engagement**: Count goal-related inputs in `profile["inputs"]` (Jaccard similarity >0.3).
     - **Progress**: Detect milestones via keywords (e.g., “finished,” “achieved”).
     - **Recency**: Apply decay: score *= 0.9 ^ (days since last_mentioned).
     - **Vibe**: Boost by VibeSculptor’s score if >0.6.
   - Normalize to [0, 1], cap at min/max from config.

4. **Curiosity Guidance**:
   - Select top 1–2 aspirations with highest scores.
   - Call `CuriosityManager.set_aspiration_bias(goal, weight)` with weight = score * 0.6.
   - Ensure 40% of questions remain exploratory to avoid railroading.

5. **Dynamic Adaptation**:
   - Reassess aspirations every 10 interactions or when VibeSculptor detects a shift (>0.3 deviation).
   - Retire aspirations with score <0.2 for >7 days.
   - Promote new goals if novelty >0.8 and not redundant (Jaccard <0.5 with existing goals).

6. **Response Modulation**:
   - If score >0.7: Use motivational tone (“You’re crushing it! What’s next for your goal?”).
   - If score <0.4 and vibe <0.5: Use supportive tone (“It’s okay to feel stuck—want to revisit your goal?”).
   - If milestone detected: Celebrate (“Wow, you did it! Let’s aim higher!”).
   - Adjust tone via TemperamentSystem (e.g., bold for sanguine, calm for phlegmatic).

7. **Bonding Synergy**:
   - Pass aspiration score to BondCalculator to add 0.1 * score to bonding score.
   - Log synergy events for debugging.

8. **Serialization**:
   - Store `profile["aspirations"]` in UserProfileState.to_dict.
   - Load and validate (e.g., remove expired goals) in UserProfileState.from_dict.

#### Configuration
Add to `sovl_config.yaml`:
```yaml
aspiration_config:
  default_score: 0.5
  min_score: 0.0
  max_score: 1.0
  decay_factor: 0.9
  reassess_interval: 10
  retire_threshold: 0.2
  weights:
    engagement: 0.4
    progress: 0.3
    recency: 0.2
    vibe: 0.1
```





