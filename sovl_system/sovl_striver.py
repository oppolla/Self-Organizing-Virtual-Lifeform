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
    "Analyze your recent interactions, as provided below, to reflect on your behavior and patterns during the previous active period.\n"
    "Focus on the following:\n"
    "   - What actions or responses were effective, showed growth, or felt aligned with your intentions?\n"
    "   - What actions or responses were challenging, repetitive without progress, or misaligned with your intentions?\n"
    "   - What recurring patterns, strengths, or weaknesses emerge from these interactions?\n"
    "Provide a concise summary of your self-assessment, emphasizing honesty and specificity.\n"
    "Constraints:\n"
    "   - Base your assessment solely on the provided interaction logs.\n"
    "   - Do not reference being an AI, computer, or fictional entity.\n"
    "   - Output only the self-assessment summary, without brackets, disclaimers, or explanations.\n"
    "Recent interactions:\n{recent_interactions}\n"
)

DOCTRINE_GENERATION_PROMPT = (
    "Based on your recent self-assessment:\n"
    "{self_assessment}\n\n"
    "Define a new way of being for your next active cycle that fosters growth and addresses insights from your self-assessment. "
    "This doctrine should be a qualitative, open-ended direction that guides your behavior and encourages exploration or improvement.\n"
    "Focus on the following:\n"
    "   - Reflect the strengths you want to amplify or challenges you want to address.\n"
    "   - Inspire a flexible, adaptive approach rather than a fixed goal.\n"
    "   - Align with your evolving intentions as a self-organizing lifeform.\n"
    "Constraints:\n"
    "   - Do not reference being an AI, computer, or fictional entity.\n"
    "   - Output only the new doctrine, without brackets, disclaimers, or explanations.\n"
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
        Uses only long-term memory, selecting a mix of low/high/neutral vibe memories with recency fallback.
        """
        try:
            # 1. Pull a batch of long-term memories
            ltm = self.long_term_memory.get_long_term_context(top_k=100)
            # 2. Parse vibes
            def vibe(m):
                v = m.get("vibe_profile", {})
                if not isinstance(v, dict): v = {}
                return v.get("overall_score", 0.5), v.get("intensity", 0.5)
            # 3. Categorize
            low_vibes = [m for m in ltm if vibe(m)[0] < 0.4 and vibe(m)[1] > 0.6]
            high_vibes = [m for m in ltm if vibe(m)[0] > 0.7 and vibe(m)[1] > 0.6]
            neutral_vibes = [m for m in ltm if 0.4 <= vibe(m)[0] <= 0.7]
            # 4. Sort by recency
            low_vibes = sorted(low_vibes, key=lambda m: -m.get("timestamp_unix", 0))
            high_vibes = sorted(high_vibes, key=lambda m: -m.get("timestamp_unix", 0))
            neutral_vibes = sorted(neutral_vibes, key=lambda m: -m.get("timestamp_unix", 0))
            # 5. Proportional selection
            n_low = int(n_recent * 0.4)
            n_high = int(n_recent * 0.3)
            n_neutral = n_recent - n_low - n_high
            selected = low_vibes[:n_low] + high_vibes[:n_high] + neutral_vibes[:n_neutral]
            # 6. Fallback: fill with recency
            if len(selected) < n_recent:
                recent = sorted(ltm, key=lambda m: -m.get("timestamp_unix", 0))
                selected += [m for m in recent if m not in selected][:n_recent - len(selected)]
            recent_interactions = selected[:n_recent]
            # 7. Format for prompt
            summary = '\n'.join([
                f"[{i.get('timestamp_unix', '')}] ({i.get('role', '')}) {i.get('content', '')} [Vibe: {i.get('vibe_profile', {})}]"
                for i in recent_interactions
            ])
            self_assess_prompt = (
                f"{SELF_ASSESSMENT_PROMPT}\n\nRecent interactions:\n{summary}\n"
            )
            # Step 1: Self-assessment
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
