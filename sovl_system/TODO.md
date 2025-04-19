# TODO:

## General Taks:

- Migrate events module
- further develop initialization and startup feel

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



## Module interaction clean progress

### Bond module sovl_main hook guide
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


