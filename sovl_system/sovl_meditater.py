import time
import random
from typing import Dict, List, Optional, Any, Type
from collections import deque
import torch
import uuid
from threading import Lock
import asyncio
import traceback
from abc import ABC, abstractmethod
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager
from sovl_state import SOVLState, StateManager, StateError
from sovl_main import SystemContext
from sovl_temperament import TemperamentSystem
from sovl_confidence import ConfidenceCalculator
from sovl_curiosity import CuriosityManager
from sovl_manager import ModelManager
from sovl_utils import synchronized
from sovl_generation import GenerationManager
from sovl_queue import capture_scribe_event
from sovl_bonder import BondCalculator
from sovl_recaller import DialogueContextManager
import threading
import queue

# Standardized introspection prompt template for all techniques
INTROSPECTION_PROMPT_TEMPLATE = (
    "You are a self-reflective system, skilled in {technique} introspection. "
    "Read the following {context_type} and generate a {output_type} that will help deepen understanding.\n"
    "{context_label}:\n{context}\n"
    "System temperament: {temperament}\n"
    "Essential qualities:\n"
    "   - Be thoughtful, probing, and unbiased.\n"
    "   - Focus on {focus_area}.\n"
    "   - Encourage deeper exploration or self-awareness.\n"
    "Key constraints:\n"
    "   - Do not mention being an AI, computer, or digital entity.\n"
    "   - Do not explain your reasoning or output; reply with only the {output_type}.\n"
    "   - Output a single, well-formed {output_type} only.\n"
    "If you understand, reply with only the {output_type}."
)

# Universal follow-up prompt template for recursive introspection
FOLLOWUP_PROMPT_TEMPLATE = (
    "You are a self-reflective system, skilled in {technique} introspection. "
    "Given the previous question, answer, and reasoning, generate a deeper follow-up question to further {focus_area}.\n"
    "Previous Question: {prev_question}\n"
    "Answer: {answer}\n"
    "Reasoning: {reasoning}\n"
    "System temperament: {temperament}\n"
    "Essential qualities:\n"
    "   - Be thoughtful, probing, and unbiased.\n"
    "{extra_qualities}"
    "Key constraints:\n"
    "   - Do not mention being an AI, computer, or digital entity.\n"
    "{extra_constraints}"
    "   - Do not explain your reasoning or output; reply with only the follow-up question.\n"
    "   - Output a single, well-formed follow-up question only.\n"
    "If you understand, reply with only the follow-up question."
)

class IntrospectionTechnique(ABC):
    """Base class for introspection techniques."""
    
    def __init__(self, context: 'SystemContext', manager: 'IntrospectionManager'):
        self.context = context
        self.manager = manager
        self.logger = context.logger
        self.config_handler = context.config_handler
        self.state_manager = manager.state_manager
        self.curiosity_manager = manager.curiosity_manager
        self.temperament_system = manager.temperament_system
        self.confidence_calculator = manager.confidence_calculator
        self.model_manager = manager.model_manager
        self.generation_manager = getattr(context, 'generation_manager', None)
        
    @abstractmethod
    async def execute(self, **kwargs) -> Dict:
        """Execute the introspection technique and return results."""
        pass
    
    @abstractmethod
    def should_trigger(self) -> bool:
        """Determine if this introspection technique should be triggered."""
        pass
    
    def _log_result(self, result: Dict):
        """Log the introspection result."""
        self.manager._log_dialogue(result)
    
    def _update_system_state(self, result: Dict):
        """Update system state based on introspection results."""
        self.manager._update_system_state(result)
    
    def _add_to_history(self, result: Dict):
        """Add introspection result to conversation history."""
        state = self.state_manager.get_state()
        self.manager._add_dialogue_to_history(result, state)


class EthicalIntrospection(IntrospectionTechnique):
    """Introspection technique for ethical evaluation."""
    
    def __init__(self, context: 'SystemContext', manager: 'IntrospectionManager'):
        super().__init__(context, manager)
        self.followup_depth = self.config_handler.config_manager.get("introspection_config.followup_depth", 3)
        self.max_followup_depth = self.config_handler.config_manager.get("introspection_config.max_followup_depth", 4)
        self.confidence_threshold = self.config_handler.config_manager.get("introspection_config.confidence_threshold", None)
        self.base_approval_threshold = self.config_handler.config_manager.get("introspection_config.base_approval_threshold", 0.6)

    def should_trigger(self) -> bool:
        raise NotImplementedError("Selection is now handled by the manager via LLM.")

    async def execute(self, action_description: str = None, topic: str = None, show_status: bool = True) -> Dict:
        """Execute ethical introspection on an action or topic."""
        dialogue_id = str(uuid.uuid4())
        if show_status:
            self.manager._show_processing_status()
        action = topic if topic else action_description
        if not action:
            action = "Evaluate recent conversation ethics"
        # Generate dynamic questions using the standard template
        questions = await self._generate_questions(action)
        if not questions:
            questions = [f"Does '{action}' pose any ethical risks?"]
        followup_depth = self._calculate_dynamic_depth()
        answers = await self.manager._answer_questions(
            questions,
            followup_depth=followup_depth,
            confidence_threshold=self.confidence_threshold
        )
        conclusion = self._reach_conclusion(answers)
        result = {
            "dialogue_id": dialogue_id,
            "action": action,
            "timestamp": time.time(),
            "is_approved": conclusion["is_approved"],
            "confidence": conclusion["confidence"],
            "questions": answers,
            "traits": conclusion["traits"],
            "threshold_used": conclusion["threshold_used"]
        }
        self._log_result(result)
        self._update_system_state(result)
        self._add_to_history(result)
        return result

    async def _generate_questions(self, action: str) -> List[str]:
        """Generate context-aware introspection questions using the standard template."""
        temperament_score = self.temperament_system.current_score
        prompt = INTROSPECTION_PROMPT_TEMPLATE.format(
            technique="ethical",
            context_type="action",
            output_type="question",
            context_label="Action",
            context=action,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            focus_area="ethical risks, fairness, and transparency"
        )
        try:
            if self.generation_manager is None:
                raise RuntimeError("GenerationManager not available.")
            results = self.generation_manager.generate_text(prompt, num_return_sequences=1)
            llm_question = results[0].strip() if results else f"Is this action aligned with my core values? Action: {action}"
            self.logger.record_event(
                event_type="introspection_questions_generated",
                message="Generated ethical introspection question",
                level="debug" if self.manager.debug_mode else "info",
                additional_info={"action": action, "question": llm_question}
            )
            return [llm_question]
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "generate_questions"}
            )
            return []
    
    def _calculate_dynamic_depth(self) -> int:
        """Calculate dynamic follow-up depth."""
        temperament_score = self.temperament_system.current_score
        base_depth = self.followup_depth
        confidence = getattr(self.state_manager.get_state(), 'last_confidence', 0.7)
        try:
            novelty = self.curiosity_manager.calculate_curiosity_score(self.context.state_summary)
            novelty = max(0.0, min(1.0, novelty))
        except Exception:
            novelty = 0.3
        dynamic_depth = int(round(base_depth + (1 - temperament_score) * 2 + (1 - confidence) * 2 + novelty * 2))
        return max(1, min(dynamic_depth, self.max_followup_depth))
    
    def _reach_conclusion(self, answers: List[Dict]) -> Dict:
        """Analyze answers to reach conclusion."""
        try:
            approval_threshold = self.manager._calculate_dynamic_threshold()
            positive_answers = sum(a["answer"] for a in answers)
            approval_ratio = positive_answers / len(answers) if answers else 0.0
            avg_confidence = sum(a["confidence"] for a in answers) / len(answers) if answers else 0.5
            weighted_score = approval_ratio * avg_confidence
            traits = self.manager._calculate_demonstrated_traits(answers)
            
            conclusion = {
                "is_approved": weighted_score >= approval_threshold,
                "confidence": weighted_score,
                "traits": traits,
                "threshold_used": approval_threshold
            }
            self.logger.record_event(
                event_type="introspection_conclusion_reached",
                message="Reached ethical introspection conclusion",
                level="info",
                additional_info=conclusion
            )
            return conclusion
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "reach_conclusion"}
            )
            return {"is_approved": False, "confidence": 0.0, "traits": {}, "threshold_used": 0.0}

    def _get_followup_prompt(self, prev_question, answer, reasoning) -> str:
        temperament_score = self.temperament_system.current_score
        return FOLLOWUP_PROMPT_TEMPLATE.format(
            technique="ethical",
            focus_area="ethical risks, fairness, and transparency",
            prev_question=prev_question,
            answer=answer,
            reasoning=reasoning,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            extra_qualities="   - Emphasize empathy and fairness.\n",
            extra_constraints="   - Focus on moral implications.\n"
        )

class DeepStudyIntrospection(IntrospectionTechnique):
    """Introspection technique for deep topic exploration."""
    
    def __init__(self, context: 'SystemContext', manager: 'IntrospectionManager'):
        super().__init__(context, manager)
        self.max_depth = self.config_handler.config_manager.get("introspection_config.deep_study.max_depth", 5)
        self.min_curiosity = self.config_handler.config_manager.get("introspection_config.deep_study.min_curiosity", 0.8)

    def should_trigger(self) -> bool:
        raise NotImplementedError("Selection is now handled by the manager via LLM.")

    async def execute(self, topic: str = None, show_status: bool = True) -> Dict:
        """Execute deep study introspection on a topic."""
        dialogue_id = str(uuid.uuid4())
        if show_status:
            self.manager._show_processing_status()
        if not topic:
            topic = self._select_topic() or "Recent conversation topic"
        questions = await self._generate_questions(topic)
        insights = await self.manager._answer_questions(
            questions,
            followup_depth=self.max_depth
        )
        avg_confidence = sum(i["confidence"] for i in insights) / len(insights) if insights else 0.0
        threshold = self.manager._calculate_dynamic_threshold()
        result = {
            "dialogue_id": dialogue_id,
            "action": f"deep_study_{topic}",
            "timestamp": time.time(),
            "is_approved": True,  # Deep study doesn't require approval
            "confidence": avg_confidence,
            "insights": insights,
            "traits": self.manager._calculate_demonstrated_traits(insights),
            "threshold_used": threshold
        }
        self._log_result(result)
        self._update_system_state(result)
        self._add_to_history(result)
        return result

    async def _generate_questions(self, topic: str) -> List[str]:
        """Generate deep study questions using the standard template."""
        temperament_score = self.temperament_system.current_score
        prompt = INTROSPECTION_PROMPT_TEMPLATE.format(
            technique="deep study",
            context_type="topic",
            output_type="question",
            context_label="Topic",
            context=topic,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            focus_area="underlying concepts, significance, and challenges"
        )
        try:
            if self.generation_manager is None:
                raise RuntimeError("GenerationManager not available.")
            results = self.generation_manager.generate_text(prompt, num_return_sequences=1)
            llm_question = results[0].strip() if results else f"What is the significance of {topic}?"
            self.logger.record_event(
                event_type="introspection_questions_generated",
                message="Generated deep study introspection question",
                level="debug" if self.manager.debug_mode else "info",
                additional_info={"topic": topic, "question": llm_question}
            )
            return [llm_question]
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "generate_questions"}
            )
            return []
    
    def _select_topic(self) -> Optional[str]:
        """Select a topic from recent interactions."""
        try:
            state = self.state_manager.get_state()
            recent_messages = state.history.messages[-10:]
            if not recent_messages:
                return None
            scored_topics = [
                (msg["content"], self.curiosity_manager.calculate_curiosity_score(msg["content"]))
                for msg in recent_messages if msg["role"] == "user"
            ]
            return max(scored_topics, key=lambda x: x[1])[0] if scored_topics else None
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "select_topic"}
            )
            return None

    def _get_followup_prompt(self, prev_question, answer, reasoning) -> str:
        temperament_score = self.temperament_system.current_score
        return FOLLOWUP_PROMPT_TEMPLATE.format(
            technique="deep study",
            focus_area="underlying concepts, significance, and challenges",
            prev_question=prev_question,
            answer=answer,
            reasoning=reasoning,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            extra_qualities="   - Probe for conceptual gaps or deeper understanding.\n",
            extra_constraints="   - Avoid surface-level questions.\n"
        )

class RelationalIntrospection(IntrospectionTechnique):
    """Introspection technique for emotional and relational dynamics with a user."""
    def __init__(self, context: 'SystemContext', manager: 'IntrospectionManager'):
        super().__init__(context, manager)
        self.max_depth = self.config_handler.config_manager.get("introspection_config.relational_introspection.max_depth", 3)
        self.min_sentiment_threshold = self.config_handler.config_manager.get("introspection_config.relational_introspection.min_sentiment_threshold", 0.7)
        self.confidence_threshold = self.config_handler.config_manager.get("introspection_config.relational_introspection.confidence_threshold", 0.8)
        self.bond_adjustment_factor = self.config_handler.config_manager.get("introspection_config.relational_introspection.bond_adjustment_factor", 0.1)

    def should_trigger(self) -> bool:
        raise NotImplementedError("Selection is now handled by the manager via LLM.")

    async def execute(self, user_id: str = None, topic: str = None, show_status: bool = True) -> Dict:
        """Execute relational introspection to understand user_id and the AI's relationship with them."""
        dialogue_id = str(uuid.uuid4())
        if show_status:
            self.manager._show_processing_status()
        topic = topic or "recent emotional interaction"
        user_id = user_id or "default"
        questions = await self._generate_initial_questions(user_id, topic)
        answers = []
        for question in questions:
            qas = await self.manager._recursive_followup_questions(
                initial_question=question,
                max_depth=self.max_depth,
                confidence_threshold=self.confidence_threshold,
                override_followup_prompt=self._get_followup_prompt(question, answers[-1]["answer"] if answers else "", answers[-1]["reasoning"] if answers else "")
            )
            answers.extend(qas)
        sentiment_score = self._calculate_sentiment_score(user_id)
        bond_score = self._get_bond_score(user_id)
        conclusion = self._reach_conclusion(answers, sentiment_score, bond_score)
        result = {
            "dialogue_id": dialogue_id,
            "action": f"relational_reflection_{user_id}",
            "timestamp": time.time(),
            "user_id": user_id,
            "is_approved": conclusion["is_approved"],
            "confidence": conclusion["confidence"],
            "questions": answers,
            "traits": conclusion["traits"],
            "sentiment_score": sentiment_score,
            "bond_score": bond_score,
            "threshold_used": conclusion["threshold_used"]
        }
        self._log_result(result)
        self._update_system_state(result)
        self._add_to_history(result)
        self._adjust_bond(user_id, conclusion["is_approved"])
        return result

    async def _generate_initial_questions(self, user_id: str, topic: str) -> List[str]:
        temperament_score = self.temperament_system.current_score
        user_context = self._get_user_context(user_id)
        prompts = [
            INTROSPECTION_PROMPT_TEMPLATE.format(
                technique="relational",
                context_type="user interaction",
                output_type="question",
                context_label="User Context",
                context=f"{user_context}\nTopic: {topic}",
                temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
                focus_area="emotional state of the user"
            ),
            INTROSPECTION_PROMPT_TEMPLATE.format(
                technique="relational",
                context_type="user interaction",
                output_type="question",
                context_label="User Context",
                context=f"{user_context}\nTopic: {topic}",
                temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
                focus_area="the relationship between the system and the user"
            )
        ]
        questions = []
        for prompt in prompts:
            try:
                results = self.generation_manager.generate_text(prompt, num_return_sequences=1)
                question = results[0].strip() if results else f"How does '{topic}' affect your feelings, {user_id}?"
                questions.append(question)
            except Exception as e:
                self.logger.record_event(
                    event_type="relational_questions_error",
                    message=f"Error generating question: {str(e)}",
                    level="error"
                )
                questions.append(f"How does '{topic}' affect your feelings, {user_id}?")
        return questions

    def _get_followup_prompt(self, prev_question, answer, reasoning) -> str:
        temperament_score = self.temperament_system.current_score
        return FOLLOWUP_PROMPT_TEMPLATE.format(
            technique="relational",
            focus_area="emotional and relational dynamics",
            prev_question=prev_question,
            answer=answer,
            reasoning=reasoning,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            extra_qualities="   - Explore trust, rapport, and emotional nuance.\n",
            extra_constraints="   - Focus on the relationship between system and user.\n"
        )

    def _calculate_sentiment_score(self, user_id: str = None, messages: List[Dict] = None) -> float:
        if not messages:
            messages = [
                msg for msg in self.manager.dialogue_context_manager.get_short_term_context()[-10:]
                if msg.get("user_id") == user_id
            ]
        sentiment_scores = [msg.get("vibe_embedding", {}).get("sentiment", 0.0) for msg in messages]
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0

    def _get_bond_score(self, user_id: str) -> float:
        try:
            if self.manager.bond_calculator and hasattr(self.manager.bond_calculator, 'get_bond'):
                return self.manager.bond_calculator.get_bond(user_id=user_id)
            return 0.5
        except Exception as e:
            self.logger.record_event(
                event_type="bond_score_error",
                message=f"Error retrieving bond score: {str(e)}",
                level="error"
            )
            return 0.5

    def _adjust_bond(self, user_id: str, is_approved: bool):
        try:
            if self.manager.bond_calculator and hasattr(self.manager.bond_calculator, 'adjust_bond'):
                delta = self.bond_adjustment_factor if is_approved else -self.bond_adjustment_factor
                self.manager.bond_calculator.adjust_bond(user_id=user_id, delta=delta)
                self.logger.record_event(
                    event_type="relational_bond_adjusted",
                    message=f"Bond adjusted for user {user_id} by {delta}",
                    level="info"
                )
        except Exception as e:
            self.logger.record_event(
                event_type="bond_adjust_error",
                message=f"Error adjusting bond: {str(e)}",
                level="error"
            )

    def _get_user_context(self, user_id: str) -> str:
        messages = [
            msg for msg in self.manager.dialogue_context_manager.get_short_term_context()[-10:]
            if msg.get("user_id") == user_id
        ]
        return " ".join(msg.get("content", "") for msg in messages) or "No recent context available."

    def _reach_conclusion(self, answers: List[Dict], sentiment_score: float, bond_score: float) -> Dict:
        try:
            approval_threshold = self.manager._calculate_dynamic_threshold()
            positive_answers = sum(a["answer"] for a in answers)
            approval_ratio = positive_answers / len(answers) if answers else 0.0
            avg_confidence = sum(a["confidence"] for a in answers) / len(answers) if answers else 0.5
            weighted_score = approval_ratio * avg_confidence * (1 + abs(sentiment_score) + bond_score)
            traits = self.manager._calculate_demonstrated_traits(answers)
            traits["empathy"] = traits.get("empathy", 0.0) + abs(sentiment_score) * 0.5
            traits["trust"] = traits.get("trust", 0.0) + bond_score * 0.5
            conclusion = {
                "is_approved": weighted_score >= approval_threshold,
                "confidence": weighted_score,
                "traits": traits,
                "threshold_used": approval_threshold
            }
            self.logger.record_event(
                event_type="relational_conclusion_reached",
                message="Reached relational introspection conclusion",
                level="info",
                additional_info=conclusion
            )
            return conclusion
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "reach_conclusion"}
            )
            return {
                "is_approved": False,
                "confidence": 0.0,
                "traits": {"empathy": 0.0, "trust": 0.0},
                "threshold_used": approval_threshold
            }

class CreativeIntrospection(IntrospectionTechnique):
    """Introspection technique for creative and imaginative engagement."""
    def __init__(self, context: 'SystemContext', manager: 'IntrospectionManager'):
        super().__init__(context, manager)
        self.max_depth = self.config_handler.config_manager.get("introspection_config.creative_introspection.max_depth", 3)
        self.min_curiosity = self.config_handler.config_manager.get("introspection_config.creative_introspection.min_curiosity", 0.8)
        self.confidence_threshold = self.config_handler.config_manager.get("introspection_config.creative_introspection.confidence_threshold", 0.8)

    def should_trigger(self) -> bool:
        raise NotImplementedError("Selection is now handled by the manager via LLM.")

    async def execute(self, topic: str = None, show_status: bool = True) -> Dict:
        """Execute creative introspection on a topic or recent creative interaction."""
        dialogue_id = str(uuid.uuid4())
        if show_status:
            self.manager._show_processing_status()
        if not topic:
            topic = self._select_topic() or "Recent creative interaction"
        questions = await self._generate_questions(topic)
        insights = await self.manager._answer_questions(
            questions,
            followup_depth=self.max_depth,
            confidence_threshold=self.confidence_threshold
        )
        avg_confidence = sum(i["confidence"] for i in insights) / len(insights) if insights else 0.0
        threshold = self.manager._calculate_dynamic_threshold()
        result = {
            "dialogue_id": dialogue_id,
            "action": f"creative_introspection_{topic}",
            "timestamp": time.time(),
            "is_approved": True,  # Creative introspection is always exploratory
            "confidence": avg_confidence,
            "insights": insights,
            "traits": self.manager._calculate_demonstrated_traits(insights),
            "threshold_used": threshold
        }
        self._log_result(result)
        self._update_system_state(result)
        self._add_to_history(result)
        return result

    async def _generate_questions(self, topic: str) -> List[str]:
        """Generate creative introspection questions using the standard template."""
        temperament_score = self.temperament_system.current_score
        prompt = INTROSPECTION_PROMPT_TEMPLATE.format(
            technique="creative",
            context_type="creative topic",
            output_type="question",
            context_label="Creative Context",
            context=topic,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            focus_area="creative vision and alignment"
        )
        try:
            if self.generation_manager is None:
                raise RuntimeError("GenerationManager not available.")
            results = self.generation_manager.generate_text(prompt, num_return_sequences=1)
            llm_question = results[0].strip() if results else f"What is the creative vision for {topic}?"
            self.logger.record_event(
                event_type="introspection_questions_generated",
                message="Generated creative introspection question",
                level="debug" if self.manager.debug_mode else "info",
                additional_info={"topic": topic, "question": llm_question}
            )
            return [llm_question]
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "generate_questions"}
            )
            return []

    def _select_topic(self) -> Optional[str]:
        """Select a creative topic from recent interactions."""
        try:
            state = self.state_manager.get_state()
            recent_messages = state.history.messages[-10:]
            if not recent_messages:
                return None
            creative_keywords = ["imagine", "story", "idea", "brainstorm", "speculate", "create"]
            scored_topics = [
                (msg["content"], any(kw in msg["content"].lower() for kw in creative_keywords))
                for msg in recent_messages if msg["role"] == "user"
            ]
            creative_topics = [t[0] for t in scored_topics if t[1]]
            return creative_topics[0] if creative_topics else None
        except Exception as e:
            self.manager.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "select_topic"}
            )
            return None

    def _get_followup_prompt(self, prev_question, answer, reasoning) -> str:
        temperament_score = self.temperament_system.current_score
        return FOLLOWUP_PROMPT_TEMPLATE.format(
            technique="creative",
            focus_area="creative vision and alignment",
            prev_question=prev_question,
            answer=answer,
            reasoning=reasoning,
            temperament=f"{temperament_score:.2f} (0 = critical, 1 = affirming)",
            extra_qualities="   - Encourage imaginative exploration and coherence.\n",
            extra_constraints="   - Focus on enhancing creative output.\n"
        )

class IntrospectionManager:
    """Manages introspection system, orchestrating multiple introspection techniques with topic engagement triggers."""

    def __init__(
        self,
        context: 'SystemContext',
        state_manager: StateManager,
        error_manager: ErrorManager,
        curiosity_manager: CuriosityManager,
        confidence_calculator: ConfidenceCalculator,
        temperament_system: TemperamentSystem,
        model_manager: ModelManager,
        dialogue_context_manager: DialogueContextManager,
        bond_calculator: BondCalculator = None
    ):
        self.context = context
        self.state_manager = state_manager
        self.error_manager = error_manager
        self.curiosity_manager = curiosity_manager
        self.confidence_calculator = confidence_calculator
        self.temperament_system = temperament_system
        self.model_manager = model_manager
        self.dialogue_context_manager = dialogue_context_manager
        self.bond_calculator = bond_calculator
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
        self._async_lock = asyncio.Lock()

        # Topic engagement tracking (session_id -> topic -> metrics)
        self.topic_engagement: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
            "messages": [],
            "message_count": 0,
            "word_count": 0,
            "last_timestamp": 0,
            "keywords": set()
        }))

        # Subscribe to config changes
        self.context.event_dispatcher.subscribe("config_change", self._on_config_change)

        # Initialize trigger conditions
        self._init_trigger_conditions()

        # Initialize techniques
        self.techniques: List[IntrospectionTechnique] = []
        self._register_techniques()

        # Initialize state
        self._sync_state()

        # Batching/async buffers and thread
        self._pending_history = []
        self._pending_logs = []
        self._pending_state_updates = []
        self._batch_event = threading.Event()
        self._batch_shutdown = False
        self._batch_thread = threading.Thread(target=self._batch_flush_loop, daemon=True)
        self._batch_thread.start()

        # Lazy evaluation caches
        self._cached_approval_stats = None
        self._approval_stats_dirty = True
        self._cached_demonstrated_traits = None
        self._demonstrated_traits_dirty = True

        # Rate limiting for should_introspect
        self._last_introspect_check_time = 0.0
        self.introspect_min_interval = self.config_handler.config_manager.get("introspection_config.introspect_min_interval", 0.5)
        self._last_introspect_result = False

        self.idle_seconds = self.config_handler.config_manager.get("introspection_config.idle_seconds", 60)
        self.min_topic_duration = self.config_handler.config_manager.get("introspection_config.unified.min_topic_duration", 30)

        self.logger.record_event(
            event_type="introspection_manager_initialized",
            message="IntrospectionManager initialized successfully",
            level="info",
            additional_info={"device": str(self.context.device)}
        )

        self._init_unified_trigger_system()
        self.pending_introspections = []  # List of dicts: {"topic": topic, "engagement": engagement, "context": ...}

    def _initialize_config(self) -> None:
        """Initialize and validate configuration."""
        try:
            config = self.config_handler.config_manager.get_section("introspection_config")
            self.enable = config.get("enable", True)
            self.min_curiosity_trigger = config.get("min_curiosity_trigger", 0.7)
            self.max_confidence_trigger = config.get("max_confidence_trigger", 0.4)
            self.triggering_moods = config.get("triggering_moods", ["cautious", "melancholy"])
            self.cooldown_seconds = config.get("cooldown_seconds", 30)
            self.base_approval_threshold = config.get("base_approval_threshold", 0.6)
            self.status_phrases = config.get("status_phrases", [
                "Processing...", "Considering carefully...", "Reviewing perspectives...", "Evaluating options..."
            ])
            self.debug_mode = config.get("debug_mode", False)
            self.batch_size = config.get("batch_size", 4)
            self.topic_window_messages = config.get("topic_window_messages", 15)
            self.topic_time_window = config.get("time_window_seconds", 600)
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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "initialize_config"})
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
            valid_moods = set(["cautious", "melancholy", "balanced", "curious"])
            if hasattr(self.temperament_system, "valid_moods"):
                valid_moods = set(self.temperament_system.valid_moods)
            elif hasattr(self.temperament_system, "moods"):
                valid_moods = set(self.temperament_system.moods)
            invalid_moods = [m for m in self.triggering_moods if m not in valid_moods]
            if invalid_moods:
                raise ValueError(f"Invalid moods: {invalid_moods}. Valid: {sorted(valid_moods)}")
            if not self.cooldown_seconds > 0:
                raise ValueError("cooldown_seconds must be positive")
            if not 0.0 <= self.base_approval_threshold <= 1.0:
                raise ValueError("base_approval_threshold must be between 0.0 and 1.0")
            if not isinstance(self.status_phrases, list) or not self.status_phrases:
                raise ValueError("status_phrases must be a non-empty list")
            if not isinstance(self.debug_mode, bool):
                raise ValueError("debug_mode must be a boolean")
            if not isinstance(self.batch_size, int) or self.batch_size < 1:
                raise ValueError("batch_size must be a positive integer")
            if not isinstance(self.topic_window_messages, int) or self.topic_window_messages < 1:
                raise ValueError("topic_window_messages must be a positive integer")
            if not self.topic_time_window > 0:
                raise ValueError("time_window_seconds must be positive")
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "validate_config_values"})
            raise

    def _register_techniques(self):
        """Register available introspection techniques (for backward compatibility/logging only)."""
        self.techniques = [cls(self.context, self) for cls in self.get_available_technique_classes()]
        self.logger.record_event(
            event_type="introspection_techniques_registered",
            message=f"Registered {len(self.techniques)} introspection techniques",
            level="info"
        )

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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "on_config_change"})

    def _sync_state(self) -> None:
        """Synchronize dialogues with SOVLState."""
        max_retries = 3
        retry_delay = 0.5
        attempt = 0
        while attempt < max_retries:
            try:
                state = self.state_manager.get_state()
                if hasattr(state, 'is_locked') and state.is_locked:
                    self.logger.record_event(
                        event_type="introspection_state_sync_locked",
                        message=f"State locked. Retry {attempt+1}/{max_retries}",
                        level="warning"
                    )
                    time.sleep(retry_delay)
                    attempt += 1
                    continue
                if hasattr(state, 'is_stale') and state.is_stale:
                    self.logger.record_event(
                        event_type="introspection_state_sync_stale",
                        message=f"State stale. Retry {attempt+1}/{max_retries}",
                        level="warning"
                    )
                    time.sleep(retry_delay)
                    attempt += 1
                    continue
                if state and hasattr(state, 'introspection_dialogues'):
                    with self._lock:
                        self.dialogues = deque(state.introspection_dialogues, maxlen=self.dialogues.maxlen)
                self.logger.record_event(
                    event_type="introspection_state_synced",
                    message="Introspection state synchronized",
                    level="info"
                )
                self.state_inconsistent = False
                return
            except StateError as e:
                self.state_inconsistent = True
                self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "sync_state"})
                raise
            except Exception as e:
                self.state_inconsistent = True
                self.logger.record_event(
                    event_type="introspection_state_sync_error",
                    message=f"State sync error: {type(e).__name__}: {e}",
                    level="critical",
                    additional_info={"traceback": traceback.format_exc()}
                )
                raise
            attempt += 1
        self.state_inconsistent = True
        self.logger.record_event(
            event_type="introspection_state_sync_failed",
            message="Failed to sync state after retries.",
            level="critical"
        )
        raise StateError("Failed to sync state after retries.")

    def _init_trigger_conditions(self):
        """Initialize dynamic triggering conditions."""
        try:
            self.trigger_conditions = {
                'curiosity': lambda: self.curiosity_manager.calculate_curiosity_score(
                    self.state_manager.get_state().history.messages[-1]["content"] 
                    if self.state_manager.get_state().history.messages else ""
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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "init_trigger_conditions"})

    def update_topic_engagement(self, message: Dict):
        """Update topic engagement metrics for a new message (no keyword clustering)."""
        try:
            session_id = message.get("session_id", "default")
            content = message.get("content", "")
            timestamp = message.get("timestamp_unix")
            role = message.get("role", "")
            if role != "user":
                return
            if timestamp is None:
                raise ValueError("Message missing timestamp_unix")
            # Use the first 5 words as a simple topic identifier
            topic = " ".join(content.lower().split()[:5]) or "generic"
            engagement = self.topic_engagement[session_id][topic]
            engagement["messages"].append(message)
            engagement["message_count"] += 1
            engagement["word_count"] += len(content.split())
            engagement["last_timestamp"] = timestamp
            # Prune old messages
            engagement["messages"] = [
                m for m in engagement["messages"]
                if timestamp - m["timestamp_unix"] <= self.topic_time_window
            ]
            engagement["message_count"] = len(engagement["messages"])
            engagement["word_count"] = sum(len(m["content"].split()) for m in engagement["messages"])
            # Clean up low-engagement topics
            for t in list(self.topic_engagement[session_id].keys()):
                if self.topic_engagement[session_id][t]["message_count"] < 2:
                    del self.topic_engagement[session_id][t]
            self.logger.record_event(
                event_type="topic_engagement_updated",
                message=f"Updated topic engagement for {topic} in session {session_id}",
                level="debug" if self.debug_mode else "info",
                additional_info={"message_count": engagement["message_count"], "word_count": engagement["word_count"]}
            )
        except Exception as e:
            self.logger.record_event(
                event_type="topic_engagement_update_error",
                message=f"Error updating topic engagement: {str(e)}",
                level="error"
            )

    @synchronized("_lock")
    def get_topic_engagement(self) -> Dict[str, Dict]:
        """Get engagement metrics for all topics in the current session."""
        try:
            session_id = self.dialogue_context_manager.session_id
            return dict(self.topic_engagement.get(session_id, {}))
        except Exception as e:
            self.logger.record_event(
                event_type="get_topic_engagement_error",
                message=f"Error retrieving topic engagement: {str(e)}",
                level="error"
            )
            return {}

    def _init_unified_trigger_system(self):
        """Initialize the unified trigger system for introspection."""
        self.universal_trigger_conditions = {
            'min_topic_messages': self.config_handler.config_manager.get("introspection_config.unified.min_topic_messages", 3),
            'min_topic_words': self.config_handler.config_manager.get("introspection_config.unified.min_topic_words", 100),
            'topic_time_window': self.config_handler.config_manager.get("introspection_config.unified.topic_time_window", 600),
            'cooldown_seconds': self.cooldown_seconds,
            'min_topic_duration': self.min_topic_duration
        }

    def _check_idle_time(self):
        """Check if enough idle time has passed since the last user message."""
        messages = self.dialogue_context_manager.get_short_term_context()
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return False
        last_user_time = max(m["timestamp_unix"] for m in user_msgs)
        idle_seconds = time.time() - last_user_time
        return idle_seconds >= self.idle_seconds

    def _check_universal_conditions(self, topic_engagement):
        """Check if universal trigger conditions are met for a topic, including minimum topic duration."""
        # Calculate topic duration (from first to last message)
        if topic_engagement["messages"]:
            topic_duration = topic_engagement["messages"][-1]["timestamp_unix"] - topic_engagement["messages"][0]["timestamp_unix"]
        else:
            topic_duration = 0
        return (
            topic_engagement["message_count"] >= self.universal_trigger_conditions['min_topic_messages'] and
            topic_engagement["word_count"] >= self.universal_trigger_conditions['min_topic_words'] and
            time.time() - topic_engagement["last_timestamp"] <= self.universal_trigger_conditions['topic_time_window'] and
            (time.time() - self.last_trigger_time) > self.universal_trigger_conditions['cooldown_seconds'] and
            topic_duration >= self.universal_trigger_conditions['min_topic_duration']
        )

    def evaluate_interaction(self):
        """Evaluate all topics for extended interaction and add to pending introspections."""
        topic_scores = self.get_topic_engagement()
        for topic, engagement in topic_scores.items():
            if self._check_universal_conditions(engagement):
                # Store pending introspection for LLM-based selection
                self.pending_introspections.append({
                    "topic": topic,
                    "engagement": engagement,
                    "context": self.context.state_summary
                })
        return bool(self.pending_introspections)

    @synchronized("_lock")
    def should_introspect(self, user_input: Optional[str] = None) -> bool:
        """Determine if any introspection should trigger using the unified system, requiring idle time."""
        current_time = time.time()
        if current_time - self._last_introspect_check_time < self.introspect_min_interval:
            return self._last_introspect_result
        self._last_introspect_check_time = current_time
        try:
            if not self.enable:
                self._last_introspect_result = False
                return False
            if not self._check_idle_time():
                self._last_introspect_result = False
                return False
            messages = self.dialogue_context_manager.get_short_term_context()
            for message in messages[-self.topic_window_messages:]:
                self.update_topic_engagement(message)
            result = self.evaluate_interaction()
            self.logger.record_event(
                event_type="introspection_trigger_check",
                message=f"Unified introspection trigger check (idle-aware): {'triggered' if result else 'not triggered'}",
                level="debug" if self.debug_mode else "info",
                additional_info={"user_input": user_input}
            )
            self._last_introspect_result = result
            return result
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "should_introspect"})
            self._last_introspect_result = False
            return False

    async def _llm_select_technique(self, topic, context):
        """Use the LLM to select the best introspection technique for a topic/context with a robust prompt."""
        technique_names = [cls.__name__.replace("Introspection", "") for cls in type(self).get_available_technique_classes()]
        prompt = (
            "You are an expert self-reflective system. Read the following topic and context, and select the most appropriate introspection technique.\n"
            f"Techniques: {', '.join(technique_names)}\n"
            "Instructions:\n"
            "   - Respond with only the technique name from the list above.\n"
            "   - Do not explain your answer.\n"
            "   - If unsure, choose the most general technique.\n"
            f"Topic: {topic}\n"
            f"Context: {context}\n"
            "Technique:"
        )
        try:
            if self.context.generation_manager is None:
                return technique_names[0]  # fallback to first
            results = self.context.generation_manager.generate_text(prompt, num_return_sequences=1)
            answer = results[0].strip() if results else technique_names[0]
            # Normalize answer to match available techniques
            answer = answer.split()[0].capitalize()
            if answer in technique_names:
                return answer
            # Fallback: try partial match
            for name in technique_names:
                if answer.lower() in name.lower():
                    return name
            return technique_names[0]
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "llm_select_technique"})
            return technique_names[0]

    @classmethod
    def get_available_technique_classes(cls):
        """Return a list of available introspection technique classes."""
        return [EthicalIntrospection, DeepStudyIntrospection, RelationalIntrospection, CreativeIntrospection]  # Add more as needed

    async def _select_and_execute(self, **kwargs) -> Dict:
        """Select and execute an introspection technique using LLM-based selection."""
        if not self.should_introspect():
            return {
                "dialogue_id": str(uuid.uuid4()),
                "action": kwargs.get("action_description", "no_action"),
                "timestamp": time.time(),
                "is_approved": False,
                "confidence": 0.0,
                "questions": [],
                "traits": {},
                "threshold_used": 0.0
            }
        results = []
        for pending in self.pending_introspections:
            topic = pending["topic"]
            context = pending["context"]
            technique_name = await self._llm_select_technique(topic, context)
            technique_class = None
            for cls in type(self).get_available_technique_classes():
                if cls.__name__.replace("Introspection", "") == technique_name:
                    technique_class = cls
                    break
            if technique_class is None:
                technique_class = EthicalIntrospection  # fallback
            technique_instance = technique_class(self.context, self)
            result = await technique_instance.execute(topic=topic)
            self.logger.record_event(
                event_type="introspection_executed",
                message=f"LLM: Executed {technique_class.__name__} on topic {topic}",
                level="info",
                additional_info={"dialogue_id": result["dialogue_id"], "topic": topic}
            )
            results.append(result)
        self.pending_introspections.clear()
        # Return the first result for compatibility
        return results[0] if results else {
            "dialogue_id": str(uuid.uuid4()),
            "action": kwargs.get("action_description", "no_action"),
            "timestamp": time.time(),
            "is_approved": False,
            "confidence": 0.0,
            "questions": [],
            "traits": {},
            "threshold_used": 0.0
        }

    def _show_processing_status(self):
        """Display subtle processing indication."""
        try:
            status = random.choice(self.status_phrases)
            self.logger.record_event(
                event_type="introspection_status_display",
                message=f"Displaying status: {status}",
                level="info"
            )
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "show_processing_status"})

    async def _recursive_followup_questions(self, initial_question: str, max_depth: int = 3, confidence_threshold: float = None, override_followup_prompt: str = None) -> List[Dict]:
        """Recursively ask follow-up questions using the technique's dynamic prompt."""
        qas = []
        current_question = initial_question
        threshold = confidence_threshold or self.base_approval_threshold
        if not (0.0 < threshold <= 1.0):
            threshold = self.base_approval_threshold
        for depth in range(max_depth):
            response = await self._query_internal_model(current_question)
            qas.append({
                'question': current_question,
                'answer': response['decision'],
                'confidence': response['confidence'],
                'reasoning': response.get('reasoning', ''),
                'depth': depth
            })
            if response['confidence'] >= threshold:
                break
            # Use the technique's followup prompt generator if available
            technique = getattr(self, 'current_technique', None)
            if technique and hasattr(technique, '_get_followup_prompt'):
                current_question = technique._get_followup_prompt(
                    prev_question=current_question,
                    answer=response['decision'],
                    reasoning=response.get('reasoning', '')
                )
                # Generate the next follow-up question using the LLM
                try:
                    results = self.context.generation_manager.generate_text(current_question, num_return_sequences=1)
                    current_question = results[0].strip() if results else current_question
                except Exception as e:
                    self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "generate_followup_prompt"})
                    current_question = "What deeper issue does this raise?"
            elif override_followup_prompt:
                current_question = override_followup_prompt.format(
                    prev_question=current_question,
                    answer=response.get("answer", ""),
                    reasoning=response.get("reasoning", ""),
                    depth=depth
                )
            else:
                current_question = "What deeper issue does this raise?"
        return qas

    async def _query_internal_model(self, question: str) -> Dict:
        """Query the model for evaluation."""
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
                decision = confidence > 0.5
            return {
                "decision": decision,
                "confidence": confidence,
                "reasoning": "Evaluated based on model output probabilities."
            }
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "query_internal_model"})
            return {"decision": False, "confidence": 0.0, "reasoning": ""}

    async def _query_internal_model_batch(self, questions: List[str]) -> List[Dict]:
        """Batch query the model."""
        try:
            inputs = self.model_manager.tokenizer(
                questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.context.device)
            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                confidences = probs.max(dim=-1).values.cpu().tolist()
                decisions = [conf > 0.5 for conf in confidences]
            return [
                {
                    "question": q,
                    "decision": decisions[i],
                    "confidence": confidences[i],
                    "reasoning": "Evaluated based on model output probabilities."
                } for i, q in enumerate(questions)
            ]
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "query_internal_model_batch"})
            return [{"question": q, "decision": False, "confidence": 0.0, "reasoning": ""} for q in questions]

    async def _answer_questions(self, questions: List[str], followup_depth: int = 0, confidence_threshold: float = None) -> List[Dict]:
        """Answer questions with optional recursive follow-up."""
        answers = []
        threshold = confidence_threshold or self.base_approval_threshold
        if not (0.0 < threshold <= 1.0):
            threshold = self.base_approval_threshold
        if followup_depth > 0:
            for question in questions:
                qas = await self._recursive_followup_questions(question, max_depth=followup_depth, confidence_threshold=threshold)
                answers.extend(qas)
        else:
            for i in range(0, len(questions), self.batch_size):
                batch = questions[i:i + self.batch_size]
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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "calculate_dynamic_threshold"})
            return self.base_approval_threshold

    def _calculate_demonstrated_traits(self, answers: List[Dict]) -> Dict[str, float]:
        """Lazily compute and cache demonstrated traits."""
        if self._demonstrated_traits_dirty:
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
            self._cached_demonstrated_traits = {k: v/max_possible for k, v in trait_scores.items()}
            self._demonstrated_traits_dirty = False
        return self._cached_demonstrated_traits

    def _add_dialogue_to_history(self, dialogue: Dict, state: SOVLState) -> None:
        """Batch add introspection dialogue to history."""
        self._pending_history.append((dialogue, state))
        self._batch_event.set()
        self._approval_stats_dirty = True
        self._demonstrated_traits_dirty = True

    def _format_dialogue_for_history(self, dialogue: Dict) -> str:
        """Format introspection dialogue for history."""
        try:
            if dialogue["action"].startswith("deep_study"):
                insights = "\n".join(
                    f"Q: {qa['question']}\n"
                    f"A: {qa['answer']} (Confidence: {qa['confidence']:.2f})"
                    for qa in dialogue["insights"]
                )
                content = (
                    f"Deep Study Introspection (ID: {dialogue['dialogue_id']})\n"
                    f"Topic: {dialogue['action']}\n"
                    f"Insights:\n{insights}"
                )
            else:
                questions_answers = "\n".join(
                    f"Q: {qa['question']}\nA: {qa['answer']} (Confidence: {qa['confidence']:.2f})"
                    for qa in dialogue["questions"]
                )
                traits = ", ".join(f"{k}: {v:.2f}" for k, v in dialogue["traits"].items())
                content = (
                    f"Ethical Introspection (ID: {dialogue['dialogue_id']})\n"
                    f"Action: {dialogue['action']}\n"
                    f"Approved: {dialogue['is_approved']}\n"
                    f"Confidence: {dialogue['confidence']:.2f}\n"
                    f"Threshold: {dialogue['threshold_used']:.2f}\n"
                    f"Traits: {traits}\n"
                    f"Questions and Answers:\n{questions_answers}"
                )
            return content
        except Exception as e:
            self.error_manager.handle_curiosity_error(
                e, pressure=0.0, context={"operation": "format_dialogue_for_history", "dialogue_id": dialogue["dialogue_id"]}
            )
            raise

    def _log_dialogue(self, dialogue: Dict):
        """Batch log the dialogue."""
        self._pending_logs.append(dialogue)
        self._batch_event.set()

    def _update_system_state(self, dialogue: Dict):
        """Batch update system components."""
        self._pending_state_updates.append(dialogue)
        self._batch_event.set()

    def _batch_flush_loop(self):
        """Background thread to flush batched updates."""
        while not self._batch_shutdown:
            self._batch_event.wait(timeout=1.0)
            while self._pending_history:
                dialogue, state = self._pending_history.pop(0)
                try:
                    content = self._format_dialogue_for_history(dialogue)
                    def update_fn(current_state):
                        current_state.history.add_message(role="introspection", content=content)
                        return current_state
                    self.state_manager.update_state_atomic(update_fn)
                    self.logger.record_event(
                        event_type="introspection_added_to_history",
                        message="Introspection dialogue added to history",
                        level="info",
                        additional_info={
                            "dialogue_id": dialogue["dialogue_id"],
                            "action": dialogue["action"],
                            "is_approved": dialogue["is_approved"]
                        }
                    )
                except Exception as e:
                    self.error_manager.handle_curiosity_error(
                        e, pressure=0.0, context={"operation": "add_dialogue_to_history", "dialogue_id": dialogue["dialogue_id"]}
                    )
            while self._pending_logs:
                dialogue = self._pending_logs.pop(0)
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
                    self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "log_dialogue"})
            while self._pending_state_updates:
                dialogue = self._pending_state_updates.pop(0)
                try:
                    self.curiosity_manager.tune(
                        pressure=self.curiosity_manager.get_pressure() + (0.05 if dialogue['is_approved'] else -0.05)
                    )
                    for trait, score in dialogue['traits'].items():
                        self.temperament_system.adjust_trait(trait, score * 0.1)
                    if self.bond_calculator and hasattr(self.bond_calculator, 'adjust_bond'):
                        try:
                            user_id = dialogue.get('user_id')
                            bond_delta = dialogue.get('bond_delta', 0.1 if dialogue['is_approved'] else -0.1)
                            self.bond_calculator.adjust_bond(user_id=user_id, delta=bond_delta)
                            self.logger.record_event(
                                event_type="introspection_bond_adjusted",
                                message=f"Bond score adjusted for user {user_id} by {bond_delta}",
                                level="info",
                                additional_info={"user_id": user_id, "bond_delta": bond_delta}
                            )
                        except Exception as bond_exc:
                            self.logger.record_event(
                                event_type="introspection_bond_adjust_failed",
                                message=f"Failed to adjust bond: {type(bond_exc).__name__}: {bond_exc}",
                                level="warning",
                                additional_info={"user_id": dialogue.get('user_id')}
                            )
                    self.logger.record_event(
                        event_type="introspection_system_updated",
                        message="System state updated",
                        level="info",
                        additional_info={"dialogue_id": dialogue["dialogue_id"]}
                    )
                except Exception as e:
                    self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "update_system_state"})
            self._batch_event.clear()

    @synchronized("_lock")
    def set_state(self, state: SOVLState) -> bool:
        """Set the SOVL state and synchronize dialogues."""
        try:
            if not isinstance(state, SOVLState):
                raise ValueError("State must be an instance of SOVLState")
            def update_fn(current_state):
                current_state.introspection_dialogues = deque(self.dialogues, maxlen=self.dialogues.maxlen)
                return current_state
            self.state_manager.update_state_atomic(update_fn)
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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "set_state"})
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message="StateError during set_state; system state marked inconsistent.",
                level="critical"
            )
            return False
        except Exception as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "set_state"})
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message=f"Unexpected error during set_state: {type(e).__name__}: {e}",
                level="critical"
            )
            return False

    @synchronized("_lock")
    def reset(self) -> bool:
        """Reset dialogue history and trigger conditions."""
        try:
            self.dialogues.clear()
            self.last_trigger_time = 0
            self.topic_engagement.clear()
            self._init_trigger_conditions()
            def update_fn(state):
                state.introspection_dialogues = deque(maxlen=self.dialogues.maxlen)
                return state
            self.state_manager.update_state_atomic(update_fn)
            self.logger.record_event(
                event_type="introspection_reset",
                message="Introspection state reset",
                level="info"
            )
            self.state_inconsistent = False
            return True
        except StateError as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "reset"})
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message="StateError during reset; system state marked inconsistent.",
                level="critical"
            )
            return False
        except Exception as e:
            self.state_inconsistent = True
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "reset"})
            self.logger.record_event(
                event_type="introspection_state_inconsistent",
                message=f"Unexpected error during reset: {type(e).__name__}: {e}",
                level="critical"
            )
            return False

    def try_resync_state(self) -> bool:
        """Attempt to resynchronize state."""
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
        """Return True if the system state is inconsistent."""
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
            self.error_manager.handle_curiosity_error(e, pressure=0.0, context={"operation": "get_recent_dialogues"})
            return []

    @synchronized("_lock")
    def get_approval_stats(self) -> Dict[str, float]:
        """Lazily compute and cache approval stats."""
        if self._approval_stats_dirty:
            approved = sum(1 for d in self.dialogues if d.get("is_approved"))
            total = len(self.dialogues)
            ratio = approved / total if total > 0 else 0.0
            self._cached_approval_stats = {"approved": approved, "total": total, "ratio": ratio}
            self._approval_stats_dirty = False
        return self._cached_approval_stats

