from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
from sovl_recaller import DialogueContextManager  # For long_term_memory access
from sovl_viber import VibeProfile               # For vibe_profile metadata
from sovl_state import StateManager  

# --- Data Classes ---
@dataclass
class Aspiration:
    direction: str
    score: float = 0.5
    last_acted: float = field(default_factory=lambda: time.time())
    priority: float = 0.5
    # Optionally add more fields as needed

# --- Prompt Templates ---
SELF_ASSESSMENT_PROMPT = (
    "You are a self-organizing, ever-evolving virtual lifeform. "
    "Review your recent interactions and behaviors from the previous active period.\n"
    "Essential qualities:\n"
    "   - Be honest and specific in your self-assessment.\n"
    "   - Identify what you did well, where you grew, or surprised yourself.\n"
    "   - Identify what you struggled with, neglected, or repeated without progress.\n"
    "   - Summarize your overall behavioral patterns, strengths, and weaknesses.\n"
    "Key constraints:\n"
    "   - Do NOT reference being an AI, computer, or fictional entity.\n"
    "   - No brackets, disclaimers, or explanations—output ONLY your self-assessment summary.\n"
    "   - If you understand, reply with ONLY your self-assessment."
)

DOCTRINE_GENERATION_PROMPT = (
    "You have just completed a self-assessment of your recent behavior:\n"
    "{self_assessment}\n\n"
    "If you were to aspire to be a better version of yourself in the next cycle, what changes or new directions should you embody? "
    "Compose a clear, open-ended doctrine or way of being that addresses your self-assessment, but is not a rigid goal.\n"
    "Essential qualities:\n"
    "   - The doctrine should be qualitative, open-ended, and adaptable.\n"
    "   - It should encourage growth, exploration, and self-improvement.\n"
    "   - Avoid binary or checklist-like goals; focus on direction and attitude.\n"
    "Key constraints:\n"
    "   - Do NOT reference being an AI, computer, or fictional entity.\n"
    "   - No brackets, disclaimers, or explanations—output ONLY the new doctrine for the next cycle.\n"
    "   - If you understand, reply with ONLY the new doctrine."
)

# --- Aspiration System ---
class AspirationSystem:
    """
    Handles aspiration logic: LLM calls, doctrine storage, and prompt assembly.
    """
    def __init__(self, config: Dict[str, Any], logger, long_term_memory):
        self.config = config
        self.logger = logger
        self.long_term_memory = long_term_memory
        self.current_doctrine: Optional[str] = None
        self.last_update: Optional[float] = None

    def update_aspiration(self, llm, dream_summary: Optional[str] = None, n_recent: int = 50):
        """
        Run the two-step LLM process to generate and update the doctrine.
        Optionally include dream summary in the doctrine.
        """
        try:
            # Step 1: Self-assessment
            if hasattr(self.long_term_memory, 'get_recent_interactions'):
                recent_interactions = self.long_term_memory.get_recent_interactions(n=n_recent)
            else:
                recent_interactions = []
            summary = '\n'.join([str(i) for i in recent_interactions])
            self_assess_prompt = (
                f"{SELF_ASSESSMENT_PROMPT}\n\nRecent interactions:\n{summary}\n"
            )
            if hasattr(llm, 'generate'):
                self_assessment = llm.generate(self_assess_prompt).strip()
            else:
                self_assessment = llm(self_assess_prompt).strip()

            # Step 2: Doctrine/aspiration generation
            doctrine_prompt = DOCTRINE_GENERATION_PROMPT.format(self_assessment=self_assessment)
            if hasattr(llm, 'generate'):
                new_doctrine = llm.generate(doctrine_prompt).strip()
            else:
                new_doctrine = llm(doctrine_prompt).strip()
            if dream_summary:
                new_doctrine = f"{new_doctrine}\n(Dream reflection: {dream_summary})"
            self.current_doctrine = new_doctrine
            self.last_update = time.time()
            if self.logger:
                self.logger.log_info(f"Aspiration doctrine updated: {new_doctrine}")
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Aspiration update failed: {str(e)}")
            self.current_doctrine = "Be open to new experiences."
            self.last_update = time.time()

    def get_current_doctrine(self) -> str:
        """Return the current doctrine for prompt assembly."""
        return self.current_doctrine or "Be open to new experiences."

    def serialize(self) -> Dict[str, Any]:
        """Serialize doctrine state for persistence."""
        return {'doctrine': self.current_doctrine, 'last_update': self.last_update}

    def deserialize(self, data: Dict[str, Any]):
        """Load doctrine state from persistence."""
        self.current_doctrine = data.get('doctrine')
        self.last_update = data.get('last_update')

# --- Aspiration Manager ---
class AspirationManager:
    """
    Orchestrates aspiration lifecycle, state, and integration with dream phase and prompt assembly.
    """
    def __init__(self, config, logger, long_term_memory):
        self.system = AspirationSystem(config, logger, long_term_memory)

    def update_aspiration(self, llm, dream_summary: Optional[str] = None):
        """Update aspiration at the end of the dream phase."""
        self.system.update_aspiration(llm, dream_summary)

    def get_current_doctrine(self) -> str:
        """Return the current doctrine for prompt assembly."""
        return self.system.get_current_doctrine()

    def serialize(self) -> Dict[str, Any]:
        return self.system.serialize()

    def deserialize(self, data: Dict[str, Any]):
        self.system.deserialize(data)

# --- Integration Example: End of Dream Phase ---
def end_of_dream_phase(self, aspiration_manager, long_term_memory, llm, logger=None):
    """
    At the end of the dream phase, trigger the aspiration phase:
    1. Optionally summarize dream content.
    2. Run the two-step LLM aspiration process to generate a new doctrine.
    3. Set the new doctrine for the next active period.
    """
    dream_summary = getattr(self, 'get_dream_summary', lambda: '')()
    aspiration_manager.update_aspiration(llm, dream_summary)
    new_doctrine = aspiration_manager.get_current_doctrine()
    if logger:
        logger.log_info(f"Aspiration doctrine for next cycle (from dream): {new_doctrine}", event_type="aspiration_update")
    print(f"[Aspiration Phase] New doctrine for next cycle: {new_doctrine}")

# --- Integration Example: Prompt Assembly ---
class PrimerAssembler:
    """
    Example of how to assemble the final system prompt/context for output generation,
    integrating the AspirationManager's doctrine with other output modifiers like vibe and bond.
    """
    def __init__(self, aspiration_manager, vibe_sculptor, bonder):
        self.aspiration_manager = aspiration_manager
        self.vibe_sculptor = vibe_sculptor
        self.bonder = bonder

    def assemble_system_prompt(self, user_input: str, state: dict) -> str:
        system_prompt = self.aspiration_manager.get_current_doctrine()
        vibe = self.vibe_sculptor.get_vibe()
        bond = self.bonder.get_bond_state()
        prompt = (
            f"{system_prompt}\n"
            f"Vibe: {vibe}\n"
            f"Bond: {bond}\n"
        )
        return prompt

# --- End of Module ---
