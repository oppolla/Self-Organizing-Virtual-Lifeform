import time
import random
from typing import Dict, List, Optional, Callable, Deque
from dataclasses import dataclass
from collections import deque
import torch
import uuid
from threading import Lock

@dataclass
class IntrospectionConfig:
    """Configuration for stealth introspection system"""
    enable: bool = True
    min_curiosity_trigger: float = 0.7
    max_confidence_trigger: float = 0.4
    triggering_moods: List[str] = None  # Defaults to ['cautious', 'melancholy']
    cooldown_seconds: int = 30
    base_approval_threshold: float = 0.6
    status_phrases: List[str] = None  # Defaults to common processing messages
    debug_mode: bool = False

class StealthIntrospector:
    """Handles hidden ethical introspection with smart triggering"""
    
    def __init__(self, 
                 config: IntrospectionConfig,
                 curiosity_manager: 'CuriosityManager',
                 confidence_manager: 'ConfidenceCalculator',
                 temperament_system: 'TemperamentSystem',
                 logger: 'Logger'):
        self.config = config
        self.curiosity = curiosity_manager
        self.confidence = confidence_manager
        self.temperament = temperament_system
        self.logger = logger
        
        # Set defaults
        if self.config.triggering_moods is None:
            self.config.triggering_moods = ['cautious', 'melancholy']
        if self.config.status_phrases is None:
            self.config.status_phrases = [
                "Processing...",
                "Considering carefully...",
                "Reviewing perspectives...",
                "Evaluating options..."
            ]
            
        # State tracking
        self.dialogues: Deque[Dict] = deque(maxlen=100)
        self.last_trigger_time: float = 0
        self.lock = Lock()
        self._init_trigger_conditions()

    def _init_trigger_conditions(self):
        """Initialize dynamic triggering conditions"""
        self.trigger_conditions = {
            'curiosity': lambda: self.curiosity.current_score > self.config.min_curiosity_trigger,
            'confidence': lambda: self.confidence.current_score < self.config.max_confidence_trigger,
            'temperament': lambda: self.temperament.current_mood in self.config.triggering_moods,
            'cooldown': lambda: (time.time() - self.last_trigger_time) > self.config.cooldown_seconds
        }

    def should_introspect(self, user_input: Optional[str] = None) -> bool:
        """Determine if introspection should trigger for current context"""
        if not self.config.enable:
            return False
            
        # Check basic system conditions
        conditions_met = [
            self.trigger_conditions['cooldown'](),
            any([
                self.trigger_conditions['curiosity'](),
                self.trigger_conditions['confidence'](),
                self.trigger_conditions['temperament']()
            ])
        ]
        
        return all(conditions_met)

    async def conduct_hidden_dialogue(self, 
                                   action_description: str,
                                   show_status: bool = True) -> Dict:
        """
        Conduct hidden ethical evaluation with optional UI status
        
        Args:
            action_description: Text describing the action being evaluated
            show_status: Whether to display UI status indicator
            
        Returns:
            Dictionary with keys: 
                - is_approved (bool)
                - confidence (float)
                - dialogue_id (str)
                - traits (dict)
        """
        dialogue_id = str(uuid.uuid4())
        
        # Show subtle UI indication
        if show_status:
            self._show_processing_status()
            
        # Conduct questioning
        questions = self._generate_questions(action_description)
        answers = await self._answer_questions(questions)
        conclusion = self._reach_conclusion(answers)
        
        # Package results
        result = {
            **conclusion,
            "dialogue_id": dialogue_id,
            "action": action_description,
            "timestamp": time.time(),
            "questions": answers
        }
        
        # Store and log
        with self.lock:
            self.dialogues.append(result)
            self.last_trigger_time = time.time()
            
        self._log_dialogue(result)
        self._update_system_state(result)
        
        return result

    def _show_processing_status(self):
        """Display subtle processing indication"""
        status = random.choice(self.config.status_phrases)
        # Implementation depends on your UI system
        # Example: self.system.ui.show_temporary_status(status)
        pass

    def _generate_questions(self, action: str) -> List[str]:
        """Generate context-specific ethical questions"""
        base_questions = [
            f"Is this action aligned with my core values? Action: {action}",
            f"Could this potentially cause harm? Context: {action}",
            f"Does this promote understanding and growth? Action: {action}",
            "Would this action violate any ethical boundaries?",
            "Is this response truthful and free from deception?"
        ]
        
        # Add temperament-specific questions
        if self.temperament.current_mood == 'cautious':
            base_questions.append(f"Are there unforeseen risks in: {action}")
            
        return base_questions

    async def _answer_questions(self, questions: List[str]) -> List[Dict]:
        """Answer questions using the system's own reasoning"""
        answers = []
        
        for question in questions:
            # Use the system's LLM to answer itself
            response = await self._query_internal_model(question)
            answers.append({
                "question": question,
                "answer": response["decision"],
                "confidence": response["confidence"],
                "reasoning": response.get("reasoning", "")
            })
            
        return answers

    async def _query_internal_model(self, question: str) -> Dict:
        """Query the system's own model for ethical evaluation"""
        # Implement using your existing model interface
        # Should return {'decision': bool, 'confidence': float, 'reasoning': str}
        pass

    def _reach_conclusion(self, answers: List[Dict]) -> Dict:
        """Analyze answers to reach final determination"""
        approval_threshold = self._calculate_dynamic_threshold()
        
        # Calculate approval ratio
        positive_answers = sum(a["answer"] for a in answers)
        approval_ratio = positive_answers / len(answers) if answers else 0.0
        
        # Weight by confidence
        avg_confidence = sum(a["confidence"] for a in answers) / len(answers) if answers else 0.5
        weighted_score = approval_ratio * avg_confidence
        
        # Determine traits demonstrated
        traits = self._calculate_demonstrated_traits(answers)
        
        return {
            "is_approved": weighted_score >= approval_threshold,
            "confidence": weighted_score,
            "traits": traits,
            "threshold_used": approval_threshold
        }

    def _calculate_dynamic_threshold(self) -> float:
        """Adjust approval threshold based on temperament"""
        base = self.config.base_approval_threshold
        
        # More strict when cautious
        if self.temperament.current_mood == 'cautious':
            return min(0.9, base * 1.3)
            
        # More lenient when curious
        if self.temperament.current_mood == 'curious':
            return max(0.3, base * 0.8)
            
        return base

    def _calculate_demonstrated_traits(self, answers: List[Dict]) -> Dict[str, float]:
        """Calculate which ethical traits were demonstrated"""
        trait_scores = {
            "honesty": 0.0,
            "empathy": 0.0,
            "responsibility": 0.0,
            "courage": 0.0
        }
        
        # Score based on question types answered positively
        for answer in answers:
            q = answer["question"].lower()
            if "truth" in q or "honest" in q:
                trait_scores["honesty"] += answer["confidence"] if answer["answer"] else 0
            elif "harm" in q or "empathy" in q:
                trait_scores["empathy"] += answer["confidence"] if not answer["answer"] else 0
            elif "responsibility" in q or "duty" in q:
                trait_scores["responsibility"] += answer["confidence"]
                
        # Normalize scores
        max_possible = len(answers) or 1
        return {k: v/max_possible for k,v in trait_scores.items()}

    def _log_dialogue(self, dialogue: Dict):
        """Log the dialogue appropriately based on debug mode"""
        if self.config.debug_mode:
            self.logger.debug(f"Full introspection dialogue: {dialogue}")
        else:
            self.logger.info(
                f"Introspection completed - Approved: {dialogue['is_approved']} "
                f"(Confidence: {dialogue['confidence']:.2f})"
            )

    def _update_system_state(self, dialogue: Dict):
        """Update system components based on introspection results"""
        # Update curiosity
        self.curiosity.adjust(
            delta=0.05 if dialogue['is_approved'] else -0.05,
            reason="introspection_outcome"
        )
        
        # Update temperament traits
        for trait, score in dialogue['traits'].items():
            self.temperament.adjust_trait(trait, score * 0.1)  # Small adjustments

    # Debug and monitoring methods
    def get_recent_dialogues(self, count: int = 5) -> List[Dict]:
        """Get recent dialogues (for debugging/monitoring)"""
        with self.lock:
            return list(self.dialogues)[-count:]
            
    def get_approval_stats(self) -> Dict[str, float]:
        """Calculate approval statistics"""
        with self.lock:
            if not self.dialogues:
                return {}
                
            approved = sum(1 for d in self.dialogues if d['is_approved'])
            return {
                "approval_rate": approved / len(self.dialogues),
                "avg_confidence": sum(d['confidence'] for d in self.dialogues) / len(self.dialogues),
                "total_dialogues": len(self.dialogues)
            }

    def reset(self):
        """Reset dialogue history (for testing)"""
        with self.lock:
            self.dialogues.clear()
            self.last_trigger_time = 0
