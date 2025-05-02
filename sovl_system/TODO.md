# TODO:

## General Taks:

1. complete training queue metadata tag weight customization system. maybe exposed parameters that adjust the weighting of each system like dreams higher than error messages
2. Disable that startup data sovl_seed part. Save it for Soulprint
3. 
4. 
5. multi scaffold support improvements, multi scaffold name system: (look at sovl_cli). name selector system? death of scaffold?
6. 
7. 
8. complete sovl_state atomic refactors





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





