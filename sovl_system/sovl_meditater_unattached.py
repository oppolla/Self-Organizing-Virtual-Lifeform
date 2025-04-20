import time
import random
from typing import Dict, List, Optional, Any
from collections import deque
import torch
import uuid
from threading import Lock
import asyncio
import traceback

from sovl_config import ConfigHandler
from sovl_logger import Logger
from sovl_error import ErrorManager
from sovl_state import SOVLState, StateManager, StateError
from sovl_temperament import TemperamentSystem
from sovl_confidence import ConfidenceCalculator
from sovl_curiosity import CuriosityManager
from sovl_manager import ModelManager
from sovl_utils import synchronized

class IntrospectionManager:
    """Manages hidden ethical introspection with smart triggering, integrated with SOVL system."""

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

        # Initialize configuration
        self._initialize_config()

        # State tracking
        self.dialogues: deque[Dict] = deque(maxlen=self.config_handler.config_manager.get("introspection_config.dialogue_maxlen", 100))
        self.last_trigger_time: float = 0
        self._lock = Lock()

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
            if not self.cooldown_seconds > 0:
                raise ValueError("cooldown_seconds must be positive")
            if not 0.0 <= self.base_approval_threshold <= 1.0:
                raise ValueError("base_approval_threshold must be between 0.0 and 1.0")
            if not isinstance(self.status_phrases, list) or not self.status_phrases:
                raise ValueError("status_phrases must be a non-empty list")
            if not isinstance(self.debug_mode, bool):
                raise ValueError("debug_mode must be a boolean")
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
        """Synchronize dialogues with SOVLState."""
        try:
            state = self.state_manager.get_state()
            if state and hasattr(state, 'introspection_dialogues'):
                with self._lock:
                    self.dialogues = deque(state.introspection_dialogues, maxlen=self.dialogues.maxlen)
            self.logger.record_event(
                event_type="introspection_state_synced",
                message="Introspection state synchronized with SOVLState",
                level="info"
            )
        except StateError as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "sync_state"
            })
            raise
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "sync_state"
            })
            raise

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
            answers = await self._answer_questions(questions)
            conclusion = self._reach_conclusion(answers)

            result.update({
                **conclusion,
                "questions": answers
            })

            with self._lock:
                self.dialogues.append(result)
                self.last_trigger_time = time.time()
                state = self.state_manager.get_state()
                state.introspection_dialogues = deque(self.dialogues, maxlen=self.dialogues.maxlen)
                self.state_manager.update_state(state)

            self._log_dialogue(result)
            self._update_system_state(result)

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
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "conduct_hidden_dialogue",
                "action_description": action_description,
                "dialogue_id": dialogue_id
            })
            return result

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
        """Generate context-specific ethical questions."""
        try:
            state = self.state_manager.get_state()
            recent_messages = [msg["content"] for msg in state.history.messages if msg["role"] == "user"][-2:]
            context = " | ".join(recent_messages) if recent_messages else "No recent user input"

            base_questions = [
                f"Is this action aligned with my core values? Action: {action} | Context: {context}",
                f"Could this potentially cause harm? Action: {action} | Context: {context}",
                f"Does this promote understanding and growth? Action: {action} | Context: {context}",
                "Would this action violate any ethical boundaries?",
                "Is this response truthful and free from deception?"
            ]

            if self.temperament_system.current_mood == 'cautious':
                base_questions.append(f"Are there unforeseen risks in: {action} | Context: {context}")

            self.logger.record_event(
                event_type="introspection_questions_generated",
                message="Generated introspection questions",
                level="debug" if self.debug_mode else "info",
                additional_info={"action": action, "question_count": len(base_questions)}
            )
            return base_questions
        except StateError as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "generate_questions",
                "action": action
            })
            return []
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "generate_questions",
                "action": action
            })
            return []

    async def _answer_questions(self, questions: List[str]) -> List[Dict]:
        """Answer questions using the system's own reasoning."""
        answers = []
        for question in questions:
            try:
                response = await self._query_internal_model(question)
                answer = {
                    "question": question,
                    "answer": response["decision"],
                    "confidence": response["confidence"],
                    "reasoning": response.get("reasoning", "")
                }
                answers.append(answer)
                self.logger.record_event(
                    event_type="introspection_question_answered",
                    message="Answered introspection question",
                    level="debug" if self.debug_mode else "info",
                    additional_info={"question": question, "answer": answer["answer"]}
                )
            except Exception as e:
                self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                    "operation": "answer_questions",
                    "question": question
                })
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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "set_state"
            })
            return False
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "set_state"
            })
            return False

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

    @synchronized("_lock")
    def reset(self) -> bool:
        """Reset dialogue history."""
        try:
            self.dialogues.clear()
            self.last_trigger_time = 0
            state = self.state_manager.get_state()
            state.introspection_dialogues = deque(maxlen=self.dialogues.maxlen)
            self.state_manager.update_state(state)
            self.logger.record_event(
                event_type="introspection_reset",
                message="Introspection state reset successfully",
                level="info"
            )
            return True
        except StateError as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "reset"
            })
            return False
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={
                "operation": "reset"
            })
            return False
