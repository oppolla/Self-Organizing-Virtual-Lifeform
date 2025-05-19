# TODO:

## General Taks:

1. Build a moral code system where it makes entries for its personal 
2. Confirm default system prompt is part of full system prompt
3. 
4. System should know unix time at initialization or always. maybe part of the recall sysyem where it knows its own monitoring in real time. can always reference it. 
5. multi scaffold support improvements, multi scaffold name system: (look at sovl_cli). name selector system? death of scaffold?
6. Disable that startup data sovl_seed part. Save it for Soulprint
7. Consider an "offline" mode for maintainence
8. Helios structure - sun like
9. Procedure system must be a process it uses to cone up with a consistantly formatted procedure list. save it with special metadata tag?




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


