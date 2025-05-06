"""
Module: sovl_procedure_unattached.py

This is a prototype location for the Procedural/Skill Memory subsystem.
When ready, classes/functions here should be moved into:
 - sovl_processor.py     (ProceduralDefinitionProcessor)
 - sovl_scribe.py        (wire into Scriber._process_scribe_queue)
 - sovl_trainer.py       (call train_procedures in run_gestation_cycle)
 - sovl_recaller.py      (detection + capture_scribe_event in add_message)
"""

import re
import json
from typing import Dict, Any, List, Optional

# Placeholder imports; replace with your actual module paths
# from sovl_recaller import DialogueContextManager
# from sovl_queue import capture_scribe_event
# from sovl_logger import Logger

class ProceduralDefinitionProcessor:
    """
    Prototype: gathers 'procedure_definition' events.
    In final system, move this class to sovl_processor.py and import in scribe and trainer.
    """
    def __init__(self, dialogue_manager: 'DialogueContextManager', logger: 'Logger'):
        self.dcm = dialogue_manager
        self.logger = logger
        self._pending: List[Dict[str, Any]] = []

    def process(self, entry: Dict[str, Any]):
        """
        Called by Scriber thread when event_type == 'procedure_definition'.
        In final system, wire into Scriber._process_scribe_queue.
        """
        proc = entry.get("event_data", {})
        if not proc.get("name") or not isinstance(proc.get("steps"), list):
            self.logger.log_error(f"Invalid procedure payload: {proc}", error_type="ProceduralError")
            return
        self._pending.append(proc)
        self.logger.record_event(
            event_type="procedural_definition_received",
            message=f"Queued procedure {proc['name']}",
            level="debug"
        )

    def train_procedures(self):
        """
        Called by TrainingWorkflowManager at start of gestation.
        In final system, call in run_gestation_cycle before LORA loops.
        """
        for proc in self._pending:
            try:
                self.dcm.add_procedure(
                    proc["name"],
                    proc.get("description", ""),
                    proc["steps"]
                )
                self.logger.record_event(
                    event_type="procedural_memory_autogestated",
                    message=f"Autogestated procedure {proc['name']}",
                    level="info"
                )
            except Exception as e:
                self.logger.log_error(f"Failed to autogestate {proc['name']}: {e}", error_type="ProceduralError")
        self._pending.clear()


class ProcedureDetector:
    """
    Prototype for detection heuristics in dialogue layer.
    Move methods to DialogueContextManager in sovl_recaller.py.
    """
    @staticmethod
    def looks_like_numbered_list(text: str) -> bool:
        lines = text.splitlines()
        markers = sum(1 for L in lines if re.match(r'^\s*(?:\d+\.\s+|Step\s+\d+)', L))
        return markers >= 2

    @staticmethod
    def has_procedure_keywords(text: str) -> bool:
        t = text.lower()
        KEYWORDS = ["step", "then", "click", "go to", "after that"]
        return any(k in t for k in KEYWORDS)

    @staticmethod
    def looks_like_enumeration(text: str) -> bool:
        """
        Detects common list/step formats: numeric, lettered, roman, or bullet.
        Returns True if at least two lines match any pattern.
        """
        patterns = [
            r'^\\s*\\d+\\.\\s+',   # numeric enumerations
            r'^\\s*[a-zA-Z]\\)\\s+',  # lettered lists like 'a)'
            r'^\\s*[IVX]+\\.\\s+',    # roman numerals
            r'^\\s*[-*+]\\s+'           # bullets '-', '*', '+'
        ]
        lines = text.splitlines()
        matches = sum(any(re.match(p, L) for p in patterns) for L in lines)
        return matches >= 2

    @staticmethod
    def score_candidate(text: str) -> float:
        """
        Composite confidence score [0.0-1.0] for how likely 'text' defines a multi-step procedure.
        Combines multiple signals (enumeration, keywords, position).
        """
        score = 0.0
        # strong signal: enumeration patterns
        if ProcedureDetector.looks_like_enumeration(text):
            score += 0.4
        # medium signal: trigger keywords
        if ProcedureDetector.has_procedure_keywords(text):
            score += 0.3
        # light signal: starts with a step indicator
        first_line = text.strip().splitlines()[0] if text.strip() else ''
        if re.match(r'^(?:\d+\.|Step)', first_line):
            score += 0.3
        return min(1.0, score)

    @staticmethod
    def classify_candidate(text: str, llm) -> bool:
        """
        Optional LLM classifier: returns True if LLM thinks text defines a procedure.
        Stub: llm should have .generate(prompt) method.
        Move to sovl_recaller.py if used.
        """
        prompt = (
            "You are a classifier. Reply YES if the text below "
            "defines a procedure (multi-step), otherwise NO.\n\n" + text
        )
        ans = llm.generate(prompt)
        return ans.strip().upper().startswith("YES")

    @staticmethod
    def parse_procedure(text: str, llm) -> Optional[Dict[str, Any]]:
        """
        LLM parser: returns dict with name, description, steps list.
        Move to sovl_recaller.py or dedicated parser module.
        """
        prompt = (
            "You are a parser. Extract procedure as JSON with keys 'name','description','steps'. "
            "Return null if no procedure.\n\n" + text
        )
        resp = llm.generate(prompt)
        try:
            data = json.loads(resp)
            if {"name","description","steps"}.issubset(data):
                return data
        except:
            pass
        return None

    @staticmethod
    def confirm_procedure(parsed: Dict[str, Any], ask_user) -> bool:
        """
        Ask user to confirm via ask_user callback. Stub returns True.
        In final, hook into user prompt.
        """
        return True


def capture_procedure(content: str, role: str, session_id: str, llm, dcm, logger):
    """
    Prototype: call from DialogueContextManager.add_message.
    When detection and parsing succeed, emit scribe event.
    Replace with actual capture_scribe_event in final system.
    """
    # use composite score to decide
    if ProcedureDetector.score_candidate(content) >= 0.5:
        # optional classifier refinement
        if ProcedureDetector.classify_candidate(content, llm):
            parsed = ProcedureDetector.parse_procedure(content, llm)
            if parsed and ProcedureDetector.confirm_procedure(parsed, None):
                entry = {
                    "event_type": "procedure_definition",
                    "event_data": parsed,
                    "metadata": {"role": role},
                    "session_id": session_id
                }
                # TODO: in final system, call:
                # capture_scribe_event(origin="dialogue", **entry)


# Simple test harness
if __name__ == "__main__":
    sample = """1. Go to settings
2. Click 'Forgot password'
3. Enter your email
4. Follow link"""
    print("Numbered list?", ProcedureDetector.looks_like_numbered_list(sample))
    print("Has keywords?", ProcedureDetector.has_procedure_keywords(sample))
    class DummyLLM:
        def generate(self, prompt): return "YES"
    llm = DummyLLM()
    parsed = ProcedureDetector.parse_procedure(sample, llm)
    print("Parsed:", parsed)
