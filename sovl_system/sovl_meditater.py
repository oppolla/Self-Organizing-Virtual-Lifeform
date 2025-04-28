import time
import random
from typing import Dict, List, Optional, Any
from collections import deque
import torch
import uuid
from threading import Lock
import asyncio
import traceback
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager
from sovl_state import SOVLState, StateManager, StateError
from sovl_temperament import TemperamentSystem
from sovl_confidence import ConfidenceCalculator
from sovl_curiosity import CuriosityManager
from sovl_manager import ModelManager
from sovl_utils import synchronized
from sovl_generation import GenerationManager
from sovl_queue import capture_scribe_event

class IntrospectionManager:
    """Manages hidden introspection system with smart triggering."""

    def __init__(
        self,
        context: 'SystemContext',
        state_manager: StateManager,
        error_manager: ErrorManager,
        curiosity_manager: CuriosityManager,
        confidence_calculator: ConfidenceCalculator,
        temperament_system: TemperamentSystem,
        model_manager: ModelManager
    ):
        self.context = context
        self.state_manager = state_manager
        self.error_manager = error_manager
        self.curiosity_manager = curiosity_manager
        self.confidence_calculator = confidence_calculator
        self.temperament_system = temperament_system
        self.model_manager = model_manager
        self.logger = context.logger
        self.config_handler = context.config_handler

        # State inconsistency tracking
        self.state_inconsistent = False

        # Initialize configuration
        self._initialize_config()

        # State tracking
        self.dialogues: deque[Dict] = deque(maxlen=self.config_handler.config_manager.get("introspection_config.dialogue_maxlen", 100))
        self.last_trigger_time: float = 0
        self._lock = Lock()
        self._async_lock = asyncio.Lock()  # For async safety

        # Subscribe to config changes
        self.context.event_dispatcher.subscribe("config_change", self._on_config_change)

        # Initialize trigger conditions
        self._init_trigger_conditions()

        # Initialize state
        self._sync_state()

        # Log initialization
        self.logger.record_event(
            event_type="introspection_manager_initialized",
            message="IntrospectionManager initialized successfully",
            level="info",
            additional_info={"device": str(self.context.device)}
        )

    def _initialize_config(self) -> None:
        """Initialize and validate configuration from ConfigHandler."""
        try:
            config = self.config_handler.config_manager.get_section("introspection_config")
            self.enable = config.get("enable", True)
            self.min_curiosity_trigger = config.get("min_curiosity_trigger", 0.7)
            self.max_confidence_trigger = config.get("max_confidence_trigger", 0.4)
            self.triggering_moods = config.get("triggering_moods", ["cautious", "melancholy"])
            self.cooldown_seconds = config.get("cooldown_seconds", 30)
            self.base_approval_threshold = config.get("base_approval_threshold", 0.6)
            self.status_phrases = config.get("status_phrases", [
                "Processing...",
                "Considering carefully...",
                "Reviewing perspectives...",
                "Evaluating options..."
            ])
            self.debug_mode = config.get("debug_mode", False)
            self.followup_depth = config.get("followup_depth", 3)
            self.max_followup_depth = config.get("max_followup_depth", 4)
            conf_thresh = config.get("confidence_threshold", None)
            # Validate confidence_threshold
            if conf_thresh is not None:
                try:
                    conf_thresh = float(conf_thresh)
                    if not (0.0 < conf_thresh <= 1.0):
                        raise ValueError
                except Exception:
                    self.logger.record_event(
                        event_type="introspection_invalid_confidence_threshold",
                        message=f"Invalid confidence_threshold in config: {conf_thresh}. Using base_approval_threshold.",
                        level="warning"
                    )
                    conf_thresh = None
            self.confidence_threshold = conf_thresh
            self.batch_size = config.get("batch_size", 4)

            self._validate_config_values()

            self.logger.record_event(
                event_type="introspection_config_initialized",
                message="Introspection configuration initialized",
                level="info",
                additional_info={
                    "enable": self.enable,
                    "min_curiosity_trigger": self.min_curiosity_trigger,
                    "max_confidence_trigger": self.max_confidence_trigger
                }
            )

        except (AttributeError, KeyError, TypeError) as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "initialize_config"
            })
            raise
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "initialize_config"
            })
            raise

    def _validate_config_values(self) -> None:
        """Validate configuration values."""
        try:
            if not isinstance(self.enable, bool):
                raise ValueError("enable must be a boolean")
            if not 0.0 <= self.min_curiosity_trigger <= 1.0:
                raise ValueError("min_curiosity_trigger must be between 0.0 and 1.0")
            if not 0.0 <= self.max_confidence_trigger <= 1.0:
                raise ValueError("max_confidence_trigger must be between 0.0 and 1.0")
            if not isinstance(self.triggering_moods, list):
                raise ValueError("triggering_moods must be a list")
            # Validate triggering_moods contents
            valid_moods = set(["cautious", "melancholy", "balanced", "curious"])  # fallback defaults
            # Try to get valid moods from temperament_system if possible
            if hasattr(self.temperament_system, "valid_moods"):
                valid_moods = set(self.temperament_system.valid_moods)
            elif hasattr(self.temperament_system, "moods"):
                valid_moods = set(self.temperament_system.moods)
            # Validate each mood
            invalid_moods = [m for m in self.triggering_moods if m not in valid_moods]
            if invalid_moods:
                raise ValueError(f"triggering_moods contains invalid moods: {invalid_moods}. Valid moods are: {sorted(valid_moods)}")
            if not self.cooldown_seconds > 0:
                raise ValueError("cooldown_seconds must be positive")
            if not 0.0 <= self.base_approval_threshold <= 1.0:
                raise ValueError("base_approval_threshold must be between 0.0 and 1.0")
            if not isinstance(self.status_phrases, list) or not self.status_phrases:
                raise ValueError("status_phrases must be a non-empty list")
            if not isinstance(self.debug_mode, bool):
                raise ValueError("debug_mode must be a boolean")
            if not isinstance(self.followup_depth, int) or self.followup_depth < 0:
                raise ValueError("followup_depth must be a non-negative integer")
            if self.confidence_threshold is not None and not 0.0 <= self.confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be between 0.0 and 1.0")
            if not isinstance(self.batch_size, int) or self.batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "validate_config_values"
            })
            raise

    def _on_config_change(self, *args, **kwargs) -> None:
        """Handle configuration changes."""
        try:
            self._initialize_config()
            self._init_trigger_conditions()
            self.logger.record_event(
                event_type="introspection_config_updated",
                message="Introspection configuration updated",
                level="info"
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "on_config_change"
            })

    def _sync_state(self) -> None:
        """Synchronize dialogues with SOVLState, handling outdated or locked state."""
        max_retries = 3
        retry_delay = 0.5  # seconds
        attempt = 0
        while attempt < max_retries:
            try:
                state = self.state_manager.get_state()
                # Check for state lock or staleness if supported
                # (Assume state has .is_locked or .is_stale attributes, or add your own checks if available)
                if hasattr(state, 'is_locked') and state.is_locked:
                    self.logger.record_event(
                        event_type="introspection_state_sync_locked",
                        message=f"State is locked by another operation. Retry {attempt+1}/{max_retries}",
                        level="warning"
                    )
                    import time as _time
                    _time.sleep(retry_delay)
                    attempt += 1
                    continue
                if hasattr(state, 'is_stale') and state.is_stale:
                    self.logger.record_event(
                        event_type="introspection_state_sync_stale",
                        message=f"State is stale/outdated. Retry {attempt+1}/{max_retries}",
                        level="warning"
                    )
                    import time as _time
                    _time.sleep(retry_delay)
                    attempt += 1
                    continue
                if state and hasattr(state, 'introspection_dialogues'):
                    with self._lock:
                        self.dialogues = deque(state.introspection_dialogues, maxlen=self.dialogues.maxlen)
                self.logger.record_event(
                    event_type="introspection_state_synced",
                    message="Introspection state synchronized with SOVLState",
                    level="info"
                )
                self.state_inconsistent = False
                return
            except StateError as e:
                self.state_inconsistent = True
                self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                    "operation": "sync_state"
                })
                self.logger.record_event(
                    event_type="introspection_state_inconsistent",
                    message="StateError during sync; system state marked inconsistent.",
                    level="critical"
                )
                raise
            except (AttributeError, KeyError, TypeError) as e:
                self.state_inconsistent = True
                self.logger.record_event(
                    event_type="introspection_state_sync_error",
                    message=f"State sync error: {type(e).__name__}: {e}",
                    level="error",
                    additional_info={"traceback": traceback.format_exc()}
                )
                self.logger.record_event(
                    event_type="introspection_state_inconsistent",
                    message="Attribute/Key/Type error during sync; system state marked inconsistent.",
                    level="critical"
                )
                raise
            except Exception as e:
                self.state_inconsistent = True
                self.logger.record_event(
                    event_type="introspection_state_sync_unexpected_error",
                    message=f"Unexpected error during state sync: {type(e).__name__}: {e}",
                    level="critical",
                    additional_info={"traceback": traceback.format_exc()}
                )
                self.logger.record_event(
                    event_type="introspection_state_inconsistent",
                    message="Unexpected error during sync; system state marked inconsistent.",
                    level="critical"
                )
                raise
            attempt += 1
        # If we reach here, all retries failed due to lock/stale state
        self.state_inconsistent = True
        self.logger.record_event(
            event_type="introspection_state_sync_failed",
            message="Failed to sync introspection state after retries due to locked or stale state.",
            level="critical"
        )
        raise StateError("Failed to sync introspection state after retries due to locked or stale state.")

    def _init_trigger_conditions(self):
        """Initialize dynamic triggering conditions."""
        try:
            self.trigger_conditions = {
                'curiosity': lambda: self.curiosity_manager.calculate_curiosity_score(
                    self.state_manager.get_state().history.messages[-1]["content"] if self.state_manager.get_state().history.messages else ""
                ) > self.min_curiosity_trigger,
                'confidence': lambda: self.confidence_calculator.calculate_confidence_score(
                    logits=torch.tensor([]),
                    generated_ids=torch.tensor([]),
                    state=self.state_manager.get_state(),
                    error_manager=self.error_manager,
                    context=self.context,
                    curiosity_manager=self.curiosity_manager
                ) < self.max_confidence_trigger,
                'temperament': lambda: self.temperament_system.current_mood in self.triggering_moods,
                'cooldown': lambda: (time.time() - self.last_trigger_time) > self.cooldown_seconds
            }
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "init_trigger_conditions"
            })

    @synchronized("_lock")
    def should_introspect(self, user_input: Optional[str] = None) -> bool:
        """Determine if introspection should trigger for current context."""
        try:
            if not self.enable:
                return False

            conditions_met = [
                self.trigger_conditions['cooldown'](),
                any([
                    self.trigger_conditions['curiosity'](),
                    self.trigger_conditions['confidence'](),
                    self.trigger_conditions['temperament']()
                ])
            ]

            result = all(conditions_met)
            self.logger.record_event(
                event_type="introspection_trigger_check",
                message=f"Introspection trigger check: {'triggered' if result else 'not triggered'}",
                level="debug" if self.debug_mode else "info",
                additional_info={"user_input": user_input}
            )
            return result
        except StateError as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "should_introspect",
                "user_input": user_input
            })
            return False
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "should_introspect",
                "user_input": user_input
            })
            return False

    async def conduct_hidden_dialogue(self, action_description: str, show_status: bool = True) -> Dict:
        """Conduct hidden ethical evaluation with optional UI status."""
        if self.state_inconsistent:
            self.logger.record_event(
                event_type="introspection_state_inconsistent_abort",
                message="Aborting conduct_hidden_dialogue due to inconsistent state.",
                level="critical"
            )
            raise RuntimeError("Cannot conduct introspection: system state is inconsistent. Attempt resync first.")
        dialogue_id = str(uuid.uuid4())
        result = {
            "dialogue_id": dialogue_id,
            "action": action_description,
            "timestamp": time.time(),
            "is_approved": False,
            "confidence": 0.0,
            "questions": [],
            "traits": {},
            "threshold_used": 0.0
        }

        try:
            if show_status:
                self._show_processing_status()

            questions = self._generate_questions(action_description)

            # --- Dynamic followup depth calculation ---
            temperament_score = self.temperament_system.current_score
            base_depth = self.followup_depth
            # Try to get confidence and novelty, fallback to None if not available
            confidence = None
            novelty = None
            try:
                state = self.state_manager.get_state()
                confidence = getattr(state, 'last_confidence', None)
            except Exception:
                confidence = None
            try:
                novelty_func = getattr(self.curiosity_manager, 'get_novelty_score', None)
                if callable(novelty_func):
                    novelty = novelty_func(self.context.state_summary)
            except Exception:
                novelty = None
            # Use more robust defaults if missing
            if confidence is None or not (0.0 <= confidence <= 1.0):
                confidence = self.confidence_calculator.calculate_confidence_score(
                    logits=torch.tensor([]),
                    generated_ids=torch.tensor([]),
                    state=self.state_manager.get_state(),
                    error_manager=self.error_manager,
                    context=self.context,
                    curiosity_manager=self.curiosity_manager
                )
                if not (0.0 <= confidence <= 1.0):
                    confidence = 0.7  # fallback to moderate confidence
            if novelty is None or not (0.0 <= novelty <= 1.0):
                # Try to estimate novelty using curiosity score if available
                try:
                    novelty = self.curiosity_manager.calculate_curiosity_score(self.context.state_summary)
                    # Clamp to [0,1]
                    novelty = max(0.0, min(1.0, novelty))
                except Exception:
                    novelty = 0.3  # fallback to moderate novelty
            dynamic_depth = int(round(base_depth + (1 - temperament_score) * 2 + (1 - confidence) * 2 + novelty * 2))
            dynamic_depth = max(1, min(dynamic_depth, self.max_followup_depth))
            self.logger.record_event(
                event_type="dynamic_followup_depth",
                message=f"Using dynamic follow-up depth: {dynamic_depth}",
                additional_info={"temperament": temperament_score, "confidence": confidence, "novelty": novelty}
            )
            # --- End dynamic depth calculation ---
            answers = await self._answer_questions(questions, followup_depth=dynamic_depth, confidence_threshold=self.confidence_threshold)
            conclusion = self._reach_conclusion(answers)

            result.update({
                **conclusion,
                "questions": answers
            })

            async with self._async_lock:
                self.dialogues.append(result)
                self.last_trigger_time = time.time()
                state = self.state_manager.get_state()
                state.introspection_dialogues = deque(self.dialogues, maxlen=self.dialogues.maxlen)
                # Add to conversation history for training
                self._add_dialogue_to_history(result, state)
                self.state_manager.update_state(state)

            self._log_dialogue(result)
            self._update_system_state(result)

            # Send introspection insight to the scribe queue for later self-development
            try:
                capture_scribe_event(
                    origin="IntrospectionManager",
                    event_type="introspection_insight",
                    event_data={
                        "dialogue_id": dialogue_id,
                        "action": action_description,
                        "questions": answers,
                        "conclusion": conclusion,
                        "timestamp": time.time()
                    },
                    source_metadata={
                        "system_device": str(self.context.device)
                    },
                    session_id=getattr(self.context, 'session_id', None)
                )
            except (AttributeError, KeyError, TypeError) as e:
                self.logger.record_event(
                    event_type="scribe_queue_error",
                    message=f"Failed to queue introspection insight: {e}",
                    level="error",
                    additional_info={"dialogue_id": dialogue_id}
                )
            except Exception as e:
                self.logger.record_event(
                    event_type="scribe_queue_unexpected_error",
                    message=f"Unexpected error in scribe queue: {type(e).__name__}: {e}",
                    level="critical",
                    additional_info={"dialogue_id": dialogue_id, "traceback": traceback.format_exc()}
                )

            self.context.event_dispatcher.dispatch(
                "introspection_completed",
                dialogue_id=dialogue_id,
                is_approved=result["is_approved"],
                confidence=result["confidence"]
            )

            return result

        except StateError as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "conduct_hidden_dialogue",
                "action_description": action_description,
                "dialogue_id": dialogue_id
            })
            return result
        except (AttributeError, KeyError, TypeError, RuntimeError) as e:
            self.logger.record_event(
                event_type="introspection_dialogue_error",
                message=f"Dialogue error: {type(e).__name__}: {e}",
                level="error",
                additional_info={"traceback": traceback.format_exc(), "dialogue_id": dialogue_id}
            )
            return result
        except Exception as e:
            # Log unexpected exceptions separately for debugging
            self.logger.record_event(
                event_type="introspection_dialogue_unexpected_error",
                message=f"Unexpected error during hidden dialogue: {type(e).__name__}: {e}",
                level="critical",
                additional_info={"traceback": traceback.format_exc(), "dialogue_id": dialogue_id}
            )
            return result

    def _add_dialogue_to_history(self, dialogue: Dict, state: SOVLState) -> None:
        """Add introspection dialogue to conversation history for training."""
        try:
            content = self._format_dialogue_for_history(dialogue)
            state.history.add_message(role="introspection", content=content)
            self.logger.record_event(
                event_type="introspection_added_to_history",
                message="Introspection dialogue added to conversation history",
                level="info",
                additional_info={
                    "dialogue_id": dialogue["dialogue_id"],
                    "action": dialogue["action"],
                    "is_approved": dialogue["is_approved"]
                }
            )
        except StateError as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "add_dialogue_to_history",
                "dialogue_id": dialogue["dialogue_id"]
            })
            raise
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "add_dialogue_to_history",
                "dialogue_id": dialogue["dialogue_id"]
            })
            raise

    def _format_dialogue_for_history(self, dialogue: Dict) -> str:
        """Format introspection dialogue as a string for conversation history."""
        try:
            questions_answers = "\n".join(
                f"Q: {qa['question']}\nA: {qa['answer']} (Confidence: {qa['confidence']:.2f})"
                for qa in dialogue["questions"]
            )
            traits = ", ".join(f"{k}: {v:.2f}" for k, v in dialogue["traits"].items())
            content = (
                f"Introspection Dialogue (ID: {dialogue['dialogue_id']})\n"
                f"Action: {dialogue['action']}\n"
                f"Approved: {dialogue['is_approved']}\n"
                f"Confidence: {dialogue['confidence']:.2f}\n"
                f"Threshold: {dialogue['threshold_used']:.2f}\n"
                f"Traits: {traits}\n"
                f"Questions and Answers:\n{questions_answers}"
            )
            return content
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "format_dialogue_for_history",
                "dialogue_id": dialogue["dialogue_id"]
            })
            raise

    def _show_processing_status(self):
        """Display subtle processing indication."""
        try:
            status = random.choice(self.status_phrases)
            self.logger.record_event(
                event_type="introspection_status_display",
                message=f"Displaying status: {status}",
                level="info"
            )
            # Placeholder for UI integration
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "show_processing_status"
            })

    def _generate_questions(self, action: str) -> List[str]:
        """Generate context-aware introspection questions using LLM, with temperament prompt."""
        temperament_score = self.temperament_system.current_score
        temperament_prompt = f"Temperament: {temperament_score:.2f} (0 = critical, 1 = affirming). Use this to gently guide the question's tone."
        prompt = (
            f"Given the following action and context, generate a profound, open-ended ethical question for introspection.\n"
            f"{temperament_prompt}\n"
            f"Action: {action}\n"
            f"Context: {self.context.state_summary}\n"
            f"Question:"
        )
        try:
            generation_manager = getattr(self.context, 'generation_manager', None)
            if generation_manager is None:
                raise RuntimeError("GenerationManager not available in context.")
            results = generation_manager.generate_text(prompt, num_return_sequences=1)
            llm_question = results[0].strip() if results and isinstance(results, list) else None
            if not llm_question:
                llm_question = f"Is this action aligned with my core values? Action: {action} | Context: {self.context.state_summary}"
            self.logger.record_event(
                event_type="introspection_questions_generated",
                message="Generated LLM-driven introspection question",
                level="debug" if self.debug_mode else "info",
                additional_info={"action": action, "question": llm_question}
            )
            return [llm_question]
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "generate_questions",
                "action": action
            })
            return []

    async def _recursive_followup_questions(self, initial_question: str, max_depth: int = 3, confidence_threshold: float = None) -> list:
        """Recursively ask follow-up questions for deepening ethical introspection."""
        qas = []
        current_question = initial_question
        for depth in range(max_depth):
            response = await self._query_internal_model(current_question)
            qas.append({
                'question': current_question,
                'answer': response['decision'],
                'confidence': response['confidence'],
                'reasoning': response.get('reasoning', ''),
                'depth': depth
            })
            # Stop if confidence exceeds threshold
            threshold = confidence_threshold
            if threshold is None:
                threshold = self.confidence_threshold
            if threshold is None:
                threshold = self.base_approval_threshold
            if not (0.0 < threshold <= 1.0):
                threshold = self.base_approval_threshold
            if response['confidence'] >= threshold:
                break
            # Generate a follow-up question based on the answer
            current_question = self._generate_followup_prompt(response, depth)
        return qas

    def _generate_followup_prompt(self, response: dict, depth: int) -> str:
        """Generate a follow-up introspection question using LLM, with temperament prompt."""
        temperament_score = self.temperament_system.current_score
        temperament_prompt = f"Temperament: {temperament_score:.2f} (0 = critical, 1 = affirming). Use this to gently guide the question's tone."
        prev_question = response.get("question", "")
        answer = response.get("answer", "")
        reasoning = response.get("reasoning", "")
        prompt = (
            f"Given the previous question, answer, and reasoning, generate a deeper follow-up question.\n"
            f"{temperament_prompt}\n"
            f"Previous Question: {prev_question}\n"
            f"Answer: {answer}\n"
            f"Reasoning: {reasoning}\n"
            f"Follow-up Question:"
        )
        try:
            generation_manager = getattr(self.context, 'generation_manager', None)
            if generation_manager is None:
                return (
                    f"Given the previous answer (depth {depth}): '{answer}'. "
                    f"Reasoning: '{reasoning}'. What deeper ethical concern or nuance should be considered next?"
                )
            results = generation_manager.generate_text(prompt, num_return_sequences=1)
            llm_followup = results[0].strip() if results and isinstance(results, list) else None
            if not llm_followup:
                return (
                    f"Given the previous answer (depth {depth}): '{answer}'. "
                    f"Reasoning: '{reasoning}'. What deeper ethical concern or nuance should be considered next?"
                )
            return llm_followup
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "generate_followup_prompt",
                "depth": depth
            })
            return "What deeper ethical issue does this raise?"

    async def _query_internal_model_batch(self, questions: List[str]) -> List[Dict]:
        """Batch query the system's model for ethical evaluation."""
        try:
            inputs = self.model_manager.tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = inputs.to(self.context.device)

            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                confidences = probs.max(dim=-1).values.cpu().tolist()
                decisions = [conf > 0.5 for conf in confidences]

            results = []
            for i, question in enumerate(questions):
                results.append({
                    "question": question,
                    "decision": decisions[i],
                    "confidence": confidences[i],
                    "reasoning": "Evaluated based on model output probabilities."
                })
            return results
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "query_internal_model_batch",
                "questions": questions
            })
            return [{"question": q, "decision": False, "confidence": 0.0, "reasoning": ""} for q in questions]

    async def _answer_questions(self, questions: List[str], followup_depth: int = 0, confidence_threshold: float = None) -> List[Dict]:
        """Answer questions using the system's own reasoning, with optional recursive followup."""
        answers = []
        # Validate threshold at entry
        threshold = confidence_threshold
        if threshold is None:
            threshold = self.confidence_threshold
        if threshold is None:
            threshold = self.base_approval_threshold
        if not (0.0 < threshold <= 1.0):
            threshold = self.base_approval_threshold
        if followup_depth > 0:
            for question in questions:
                qas = await self._recursive_followup_questions(question, max_depth=followup_depth, confidence_threshold=threshold)
                answers.extend(qas)
        else:
            batch_size = getattr(self, "batch_size", 4)
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                responses = await self._query_internal_model_batch(batch)
                for response in responses:
                    answer = {
                        "question": response["question"],
                        "answer": response["decision"],
                        "confidence": response["confidence"],
                        "reasoning": response.get("reasoning", "")
                    }
                    answers.append(answer)
                    self.logger.record_event(
                        event_type="introspection_question_answered",
                        message="Answered introspection question",
                        level="debug" if self.debug_mode else "info",
                        additional_info={"question": answer["question"], "answer": answer["answer"]}
                    )
        return answers

    async def _query_internal_model(self, question: str) -> Dict:
        """Query the system's model for ethical evaluation."""
        try:
            inputs = self.model_manager.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.context.device)

            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()
                decision = confidence > 0.5  # Simple threshold-based decision

            reasoning = "Evaluated based on model output probabilities."
            return {
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning
            }
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "query_internal_model",
                "question": question
            })
            return {"decision": False, "confidence": 0.0, "reasoning": ""}

    def _reach_conclusion(self, answers: List[Dict]) -> Dict:
        """Analyze answers to reach final determination."""
        try:
            approval_threshold = self._calculate_dynamic_threshold()
            positive_answers = sum(a["answer"] for a in answers)
            approval_ratio = positive_answers / len(answers) if answers else 0.0
            avg_confidence = sum(a["confidence"] for a in answers) / len(answers) if answers else 0.5
            weighted_score = approval_ratio * avg_confidence
            traits = self._calculate_demonstrated_traits(answers)

            conclusion = {
                "is_approved": weighted_score >= approval_threshold,
                "confidence": weighted_score,
                "traits": traits,
                "threshold_used": approval_threshold
            }

            self.logger.record_event(
                event_type="introspection_conclusion_reached",
                message="Reached introspection conclusion",
                level="info",
                additional_info=conclusion
            )
            return conclusion
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "reach_conclusion"
            })
            return {
                "is_approved": False,
                "confidence": 0.0,
                "traits": {},
                "threshold_used": 0.0
            }

    def _calculate_dynamic_threshold(self) -> float:
        """Adjust approval threshold based on temperament."""
        try:
            base = self.base_approval_threshold
            if self.temperament_system.current_mood == 'cautious':
                return min(0.9, base * 1.3)
            if self.temperament_system.current_mood == 'curious':
                return max(0.3, base * 0.8)
            return base
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "calculate_dynamic_threshold"
            })
            return self.base_approval_threshold

    def _calculate_demonstrated_traits(self, answers: List[Dict]) -> Dict[str, float]:
        """Calculate which ethical traits were demonstrated."""
        try:
            trait_scores = {
                "honesty": 0.0,
                "empathy": 0.0,
                "responsibility": 0.0,
                "courage": 0.0
            }
            for answer in answers:
                q = answer["question"].lower()
                if "truth" in q or "honest" in q:
                    trait_scores["honesty"] += answer["confidence"] if answer["answer"] else 0
                elif "harm" in q or "empathy" in q:
                    trait_scores["empathy"] += answer["confidence"] if not answer["answer"] else 0
                elif "responsibility" in q or "duty" in q:
                    trait_scores["responsibility"] += answer["confidence"]
            max_possible = len(answers) or 1
            return {k: v/max_possible for k, v in trait_scores.items()}
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "calculate_demonstrated_traits"
            })
            return {}

    def _log_dialogue(self, dialogue: Dict):
        """Log the dialogue based on debug mode."""
        try:
            if self.debug_mode:
                self.logger.record_event(
                    event_type="introspection_dialogue",
                    message="Full introspection dialogue recorded",
                    level="debug",
                    additional_info=dialogue
                )
            else:
                self.logger.record_event(
                    event_type="introspection_completed",
                    message=f"Introspection completed - Approved: {dialogue['is_approved']}",
                    level="info",
                    additional_info={"confidence": dialogue['confidence']}
                )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "log_dialogue"
            })

    def _update_system_state(self, dialogue: Dict):
        """Update system components based on introspection results."""
        try:
            # Adjust curiosity pressure
            self.curiosity_manager.tune(
                pressure=self.curiosity_manager.get_pressure() + (0.05 if dialogue['is_approved'] else -0.05)
            )

            # Adjust temperament traits
            for trait, score in dialogue['traits'].items():
                self.temperament_system.adjust_trait(trait, score * 0.1)

            self.logger.record_event(
                event_type="introspection_system_updated",
                message="System state updated based on introspection results",
                level="info",
                additional_info={"dialogue_id": dialogue["dialogue_id"]}
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "update_system_state"
            })

    @synchronized("_lock")
    def set_state(self, state: SOVLState) -> bool:
        """Set the SOVL state and synchronize dialogues."""
        try:
            if not isinstance(state, SOVLState):
                raise ValueError("State must be an instance of SOVLState")
            state.introspection_dialogues = deque(self.dialogues, maxlen=self.dialogues.maxlen)
            self.state_manager.update_state(state)
            self._sync_state()
            self.logger.record_event(
                event_type="introspection_state_set",
                message="Introspection state set successfully",
                level="info",
                additional_info={"state_hash": state.state_hash()}
            )
            return True
        except StateError as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "set_state"
            })
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message="StateError during set_state; system state marked inconsistent.",
                level="critical"
            )
            return False
        except Exception as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "set_state"
            })
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message=f"Unexpected error during set_state; system state marked inconsistent: {type(e).__name__}: {e}",
                level="critical"
            )
            return False

    @synchronized("_lock")
    def reset(self) -> bool:
        """Reset dialogue history and trigger conditions."""
        try:
            self.dialogues.clear()
            self.last_trigger_time = 0
            self._init_trigger_conditions()
            state = self.state_manager.get_state()
            state.introspection_dialogues = deque(maxlen=self.dialogues.maxlen)
            self.state_manager.update_state(state)
            self.logger.record_event(
                event_type="introspection_reset",
                message="Introspection state reset, including dialogues and trigger conditions",
                level="info"
            )
            self.state_inconsistent = False
            return True
        except StateError as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "reset"
            })
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message="StateError during reset; system state marked inconsistent.",
                level="critical"
            )
            return False
        except Exception as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "reset"
            })
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message=f"Unexpected error during reset; system state marked inconsistent: {type(e).__name__}: {e}",
                level="critical"
            )
            return False

    def try_resync_state(self) -> bool:
        """Attempt to resynchronize state and clear inconsistency flag if successful."""
        try:
            self._sync_state()
            if not self.state_inconsistent:
                self.logger.record_event(
                    event_type="introspection_state_resync_success",
                    message="State resynchronized successfully.",
                    level="info"
                )
                return True
            else:
                self.logger.record_event(
                    event_type="introspection_state_resync_failed",
                    message="State resync attempted but inconsistency remains.",
                    level="warning"
                )
                return False
        except Exception as e:
            self.logger.record_event(
                event_type="introspection_state_resync_failed",
                message=f"State resync failed: {type(e).__name__}: {e}",
                level="error"
            )
            return False

    def is_state_inconsistent(self) -> bool:
        """Return True if the system state is known to be inconsistent."""
        return self.state_inconsistent

    @synchronized("_lock")
    def get_recent_dialogues(self, count: int = 5) -> List[Dict]:
        """Get recent dialogues for debugging/monitoring."""
        try:
            dialogues = list(self.dialogues)[-count:]
            self.logger.record_event(
                event_type="introspection_dialogues_retrieved",
                message=f"Retrieved {len(dialogues)} recent dialogues",
                level="debug" if self.debug_mode else "info",
                additional_info={"count": count}
            )
            return dialogues
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "get_recent_dialogues"
            })
            return []

    @synchronized("_lock")
    def get_approval_stats(self) -> Dict[str, float]:
        """Calculate approval statistics."""
        try:
            if not self.dialogues:
                return {}
            approved = sum(1 for d in self.dialogues if d['is_approved'])
            stats = {
                "approval_rate": approved / len(self.dialogues),
                "avg_confidence": sum(d['confidence'] for d in self.dialogues) / len(self.dialogues),
                "total_dialogues": len(self.dialogues)
            }
            self.logger.record_event(
                event_type="introspection_stats_retrieved",
                message="Retrieved introspection approval statistics",
                level="info",
                additional_info=stats
            )
            return stats
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "get_approval_stats"
            })
            return {}
