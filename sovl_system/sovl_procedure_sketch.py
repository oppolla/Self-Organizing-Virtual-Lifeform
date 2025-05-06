"""
Module: sovl_procedure_sketch.py

This sketch outlines the system for detecting, defining, storing,
and logging new procedures within the SOVL architecture.

Core Idea:
1. DialogueContextManager (DCM) in sovl_recaller.py is central. It processes
   incoming messages and uses ProcedureDetector logic to identify potential procedures.
2. If a procedure is detected, parsed, and confirmed, DCM *immediately* stores
   it in its internal, persistent procedural memory.
3. After internal storage, DCM calls the Scriber to log a "procedure_defined"
   (or similar) event. This event, enriched by MetadataProcessor, enters the
   main scribe journal for auditing and potential LoRA learning about the *event*
   of skill acquisition.
4. The procedure definition itself is *not* "trained" via sovl_trainer.py's
   gestation cycle. It's made available operationally by DCM.

Key Modules Involved:
 - sovl_recaller.py (DialogueContextManager): Owns procedure detection,
   internal storage, and initiates scribing.
 - sovl_scribe.py (Scriber): Receives the "procedure_defined" event,
   enriches it via MetadataProcessor, and logs it to the scribe journal.
 - sovl_processor.py (MetadataProcessor): Enriches metadata for the
   "procedure_defined" event. Also (ScribeIngestionProcessor) defines
   how this event is templated for LoRA if it's part of training data.
"""

import re
import json
from typing import Dict, Any, List, Optional

# --- Placeholder Imports ---
# These would be actual imports in the final system.

# From sovl_recaller.py (or its dependencies)
# class DialogueContextManager:
#     def __init__(self, ..., scriber_instance, logger_instance, llm_instance):
#         self.scriber = scriber_instance
#         self.logger = logger_instance
#         self.llm = llm_instance
#         # This internal_procedures_store is CRITICAL. It holds the operational procedures.
#         # It MUST be persisted (e.g., to a database or file) across sessions if procedures
#         # are to be remembered long-term, independent of the scribe journal.
#         # The recommended approach is a dedicated SQLite table, see details below.
#         self.internal_procedures_store = {} 
#
#     def add_procedure_to_internal_store(self, name: str, description: str, steps: List[str], metadata: Optional[Dict[str, Any]] = None):
#         # Logic to add to self.internal_procedures_store, including robust persistence
#         # (e.g., writing to the dedicated SQLite `procedures` table described below).
#         self.internal_procedures_store[name] = { # This in-memory dict can act as a cache
#             "description": description,
#             "steps": steps,
#             "metadata": metadata or {},
#             "defined_at": time.time() # or a proper ISO timestamp
#         }
#         # Actual DB write operation would happen here.
#         self.logger.info(f"Procedure '{name}' added to DCM's internal operational memory.")
#
#     def get_procedure_from_internal_store(self, name: str) -> Optional[Dict[str, Any]]:
#         # Logic to retrieve from self.internal_procedures_store (cache) or fetch from
#         # the dedicated SQLite `procedures` table if not in cache.
#         # Remember to deserialize JSON fields (steps, metadata) when reading from DB.
#         pass
#
#     def process_incoming_message_for_procedure(self, content: str, role: str, session_id: str):
#         # This conceptual method within DialogueContextManager is the primary entry point
#         # for processing any incoming message that might contain a procedure definition.
#         # It would orchestrate calls to ProcedureDetector methods, handle LLM interactions
#         # for parsing/classification, manage user confirmation (if needed via UI callbacks),
#         # and upon successful validation, call `add_procedure_to_internal_store` immediately,
#         # followed by initiating the scribing of a `procedure_defined` event.
#         # The `capture_procedure_in_dcm` function below simulates this detailed orchestration.
#         pass

# From sovl_scribe.py
# class Scriber:
#     def scribe(self, origin: str, event_type: str, event_data: Dict[str, Any], source_metadata: Optional[Dict[str, Any]] = None):
#         # Actual scribing logic that queues the event for processing,
#         # enrichment by MetadataProcessor, and logging.
#         print(f"SCRIBE_EVENT: origin={origin}, type={event_type}, data={event_data}, meta={source_metadata}")
#         pass

# From sovl_logger.py
# class Logger:
#     def info(self, message: str): print(f"LOG INFO: {message}")
#     def error(self, message: str): print(f"LOG ERROR: {message}")
#     def record_event(self, event_type: str, message: str, level: str, additional_info: Optional[Dict[str, Any]] = None):
#        print(f"LOG EVENT ({level}): {event_type} - {message} - {additional_info}")


# --- Procedural Memory Storage (within DialogueContextManager) ---
#
# Procedures require a dedicated, persistent store managed by DialogueContextManager.
# Reusing the LongTermMemory (conversational vector store) for primary procedure storage
# is not recommended due to schema mismatch, different retrieval needs (exact name vs. semantic),
# and potential for data confusion.
#
# **Recommended Approach: Dedicated SQLite Table**
# - Within the same SQLite database file that LongTermMemory uses (e.g., `conversations.db`),
#   DialogueContextManager should create and manage a new table, e.g., `procedures`.
#
# **Proposed `procedures` Table Schema:**
#   - `name` (TEXT, PRIMARY KEY, UNIQUE NOT NULL): Unique procedure name.
#   - `description` (TEXT): Human-readable description of the procedure.
#   - `steps` (TEXT NOT NULL): JSON string array of procedure steps (e.g., json.dumps(["step1", "step2"])).
#   - `metadata` (TEXT): JSON string object for additional metadata (e.g., source, version, UI hints).
#                        Example: json.dumps({"source": "user_defined", "version": 1, "created_by": "user_xyz"})
#   - `created_at` (REAL or TEXT NOT NULL): Timestamp of creation (e.g., Unix epoch or ISO 8601).
#   - `updated_at` (REAL or TEXT NOT NULL): Timestamp of last update.
#   - `embedding` (BLOB, OPTIONAL): If storing embeddings of the description directly for semantic search assistance.
#
# **DCM Interaction:**
#   - **Initialization:** DCM ensures the `procedures` table exists (`CREATE TABLE IF NOT EXISTS`).
#   - **Adding (`add_procedure_to_internal_store`):** Serializes `steps` and `metadata` to JSON,
#     then performs an `INSERT OR REPLACE INTO procedures ...` SQL operation.
#     The `self.internal_procedures_store` dict can act as a write-through cache or load-on-demand cache.
#   - **Retrieving (`get_procedure_from_internal_store`):** Fetches by `name` from the table, deserializes JSON fields.
#     Caches frequently accessed procedures in memory if desired.
#   - **Listing/Updating/Deleting:** Standard SQL operations for these functionalities.
#
#   - **Deleting Procedures:**
#     - **User Intent:** DCM needs to recognize user commands like "delete procedure [name]" or "forget how to [task]".
#     - **Confirmation:** CRITICAL to confirm with the user before deletion (e.g., "Are you sure you want to delete '{procedure_name}'?").
#     - **DCM Method:** A `delete_procedure(self, name: str) -> bool` method in DCM.
#       - Executes `DELETE FROM procedures WHERE name = ?;`.
#       - Removes the procedure from any in-memory cache.
#       - Returns success/failure.
#     - **Feedback:** Inform the user of the outcome.
#     - **Scribe Event:** Log a `procedure_deleted` event via `scriber.scribe()` for auditing and potential LoRA learning
#       (e.g., `event_data={"name": procedure_name}, source_metadata={"deleted_by": "user"}`).
#
# **Semantic Search for Procedures (Optional Enhancement):**
#   - While the SQLite table provides exact name lookup, for semantic discovery:
#     1. When a procedure is added/updated, its `description` (and/or `name`) can be embedded.
#     2. This embedding, along with the procedure `name` as an identifier, can be added to
#        the *existing* FAISS index of `LongTermMemory` (perhaps with a special marker/namespace).
#        If a procedure is deleted, its corresponding entry in the FAISS index should also be removed if possible
#        (FAISS index removal can sometimes be non-trivial, depending on the index type and library version).
#     3. A semantic query would hit FAISS, retrieve the `name`, and then DCM would fetch the full
#        procedure details from the `procedures` SQLite table using the exact name.
#     This keeps the `procedures` table as the source of truth, with FAISS as an auxiliary search index.
#
# --- End of Procedural Memory Storage Section ---


# --- Procedure Detection Logic ---
# In the final system, these static methods would be part of, or utilized by,
# the DialogueContextManager in sovl_recaller.py.

class ProcedureDetector:
    """
    Contains heuristics and LLM-based methods to detect, parse,
    and confirm procedures from text. These methods form a multi-stage
    validation process to reliably identify and structure procedures.
    These methods are intended to be integrated into DialogueContextManager.
    """
    @staticmethod
    def looks_like_numbered_list(text: str) -> bool:
        lines = text.splitlines()
        markers = sum(1 for L in lines if re.match(r'^\\s*(?:\\d+\\.\\s+|Step\\s+\\d+)', L))
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
        if re.match(r'^(?:\\d+\\.|Step)', first_line):
            score += 0.3
        return min(1.0, score)

    @staticmethod
    def classify_candidate(text: str, llm_instance: Any) -> bool:
        """
        Optional LLM classifier, acting as a key semantic vetting stage.
        In final system, llm_instance would be the actual LLM interface from DCM.
        The prompt used here is illustrative; actual prompts would likely involve more
        sophisticated prompt engineering and could be managed via a templating system.
        A production system might also explore using confidence scores from the LLM
        if available, though this sketch uses a simpler YES/NO determination.
        """
        prompt = (
            "You are a classifier. Reply YES if the text below "
            "defines a procedure (multi-step), otherwise NO.\\n\\n" + text
        )
        ans = llm_instance.generate(prompt) # Assumes llm_instance has a .generate() method
        return ans.strip().upper().startswith("YES")

    @staticmethod
    def parse_procedure(text: str, llm_instance: Any) -> Optional[Dict[str, Any]]:
        """
        LLM parser, acting as a structural and content vetting stage.
        In final system, llm_instance would be the actual LLM interface from DCM.
        Should return a dict with 'name', 'description', 'steps'.
        The prompt used here is illustrative; actual prompts would likely involve more
        sophisticated prompt engineering, potentially with few-shot examples, and could
        be managed via a templating system for robustness and maintainability.
        A production system might also explore using confidence scores for parsing quality
        if available, though this sketch focuses on successful parsing or failure.
        """
        prompt = (
            "You are a parser. Extract procedure as JSON with keys 'name','description','steps'. "
            "The 'name' should be a concise verb-noun phrase. 'description' explains its purpose. "
            "'steps' is a list of strings. Return null if no clear procedure.\\n\\n" + text
        )
        resp = llm_instance.generate(prompt)
        try:
            data = json.loads(resp)
            if isinstance(data, dict) and {"name","description","steps"}.issubset(data) and isinstance(data["steps"], list):
                return data
        except:
            pass # Error in parsing or LLM returned non-JSON
        return None

    @staticmethod
    def confirm_procedure(parsed_procedure: Dict[str, Any], user_interface_callback: Any) -> bool:
        """
        Ask user for confirmation via a user_interface_callback.
        Stub returns True. In a real system, this would involve actual user interaction.
        The callback mechanism (e.g., how DCM prompts the user and receives a response)
        is a significant UI/UX design consideration for the main application.
        """
        # Example:
        # if user_interface_callback:
        #     return user_interface_callback(f"Found procedure '{parsed_procedure['name']}'. Confirm?")
        return True


# --- Conceptual Flow within DialogueContextManager (sovl_recaller.py) ---

def capture_procedure_in_dcm(
    content: str,
    role: str,
    session_id: str,
    dcm_instance: Any, # Represents the DialogueContextManager instance
    # llm_instance and user_interface_callback would be accessed via dcm_instance
):
    """
    This function simulates the logic that would reside within
    DialogueContextManager.add_message() or a similar processing method in sovl_recaller.py.

    It handles procedure detection, parsing, confirmation (the vetting stages),
    internal storage, and then initiates scribing of the "procedure_defined" event.
    """
    logger = dcm_instance.logger # Assumes DCM has a logger
    llm = dcm_instance.llm     # Assumes DCM has an LLM interface
    # user_interface_callback would also be managed by DCM

    # Stage 1: Initial Heuristic Vetting (Fast Filter)
    if ProcedureDetector.score_candidate(content) < 0.5: # Confidence threshold
        logger.info("Content did not meet initial score threshold for being a procedure.")
        return

    # Stage 2: LLM Semantic Vetting (Classification)
    if not ProcedureDetector.classify_candidate(content, llm):
        logger.info("LLM classified content as not a procedure.")
        return

    # Stage 3: LLM Structural & Content Vetting (Parsing)
    parsed_procedure = ProcedureDetector.parse_procedure(content, llm)
    if not parsed_procedure:
        logger.info("Failed to parse a structured procedure from content.")
        return

    # Stage 4: User Confirmation Vetting (Final Approval)
    # In a real system, dcm_instance would handle interaction with the user.
    # For the sketch, we assume confirmation.
    if not ProcedureDetector.confirm_procedure(parsed_procedure, None): # Pass a real callback in practice
        logger.info(f"User did not confirm procedure '{parsed_procedure['name']}'.")
        return

    logger.info(f"Procedure '{parsed_procedure['name']}' detected, parsed, and confirmed through all vetting stages.")

    # 5. Immediate Internal Storage within DialogueContextManager
    # DCM is responsible for its own persistent store of procedures.
    try:
        dcm_instance.add_procedure_to_internal_store(
            name=parsed_procedure["name"],
            description=parsed_procedure.get("description", ""),
            steps=parsed_procedure["steps"],
            metadata={"source": "dialogue_detection", "role": role, "session_id": session_id}
        )
        logger.info(f"Procedure '{parsed_procedure['name']}' added to DCM's internal operational memory.")
    except Exception as e:
        logger.error(f"Error adding procedure '{parsed_procedure['name']}' to DCM internal store: {e}")
        return # Don't scribe if internal storage failed

    # 6. Scribe the "Procedure Defined" Event for Auditing and Broader Learning
    # This event goes through the full Scriber pipeline. `Scriber` will call
    # `MetadataProcessor` to enrich this event with comprehensive system context before logging.
    # The LoRA can learn *about* this event of skill acquisition from the scribe journal.
    event_data_for_scribe = {
        "name": parsed_procedure["name"],
        "description_snippet": parsed_procedure.get("description", "")[:100] + "..." if parsed_procedure.get("description", "") else "", # Snippet for brevity
        "steps_count": len(parsed_procedure["steps"]),
        # Avoid logging full steps in every scribe event if they are very long or numerous,
        # as DCM already has the full definition in its persistent internal_procedures_store.
        # Log key identifiers, summaries, or metadata about the steps if needed for the event log.
        # The primary definition lives in DCM.
    }
    source_metadata_for_scribe = {
        "detection_method": "heuristic_and_llm",
        "confirmation_status": "confirmed_by_user", # Or "auto_confirmed" or "user_edited_and_confirmed"
        "original_text_length": len(content),
        "role": role, # Role of the message author (e.g. "user")
        "session_id": session_id # session_id is often added by Scriber or MetadataProcessor too, but can be passed
    }

    try:
        # Assumes dcm_instance has a reference to the scriber_instance
        dcm_instance.scriber.scribe(
            origin="dialogue_manager.procedure_learning",
            event_type="procedure_defined", # This new event_type needs a template in ScribeIngestionProcessor
            event_data=event_data_for_scribe,
            source_metadata=source_metadata_for_scribe
        )
        logger.info(f"Event 'procedure_defined' for '{parsed_procedure['name']}' sent to Scriber.")
    except Exception as e:
        logger.error(f"Error sending 'procedure_defined' event to Scriber for '{parsed_procedure['name']}': {e}")


# --- Test Harness / Example Usage ---
if __name__ == "__main__":

    # --- Dummy/Mock Components for testing the sketch ---
    class DummyLLM:
        def generate(self, prompt: str) -> str:
            if "classifier" in prompt:
                return "YES"
            if "parser" in prompt:
                return json.dumps({
                    "name": "Make PBJ Sandwich",
                    "description": "How to make a peanut butter and jelly sandwich.",
                    "steps": ["Get two slices of bread.", "Spread peanut butter on one slice.", "Spread jelly on the other slice.", "Put the slices together."]
                })
            return ""

    class DummyLogger:
        def info(self, message: str): print(f"SKETCH LOG INFO: {message}")
        def error(self, message: str): print(f"SKETCH LOG ERROR: {message}")
        def record_event(self, event_type: str, message: str, level: str, additional_info: Optional[Dict[str, Any]] = None):
            print(f"SKETCH LOG EVENT ({level}): {event_type} - {message} - {additional_info}")


    class DummyScriber:
         def scribe(self, origin: str, event_type: str, event_data: Dict[str, Any], source_metadata: Optional[Dict[str, Any]] = None):
            print(f"SKETCH SCRIBE_EVENT: origin={origin}, type={event_type}, data={json.dumps(event_data)}, meta={json.dumps(source_metadata)}")

    class DummyDialogueContextManager:
        def __init__(self):
            self.llm = DummyLLM()
            self.logger = DummyLogger()
            self.scriber = DummyScriber()
            self.internal_procedures_store = {}

        def add_procedure_to_internal_store(self, name: str, description: str, steps: List[str], metadata: Optional[Dict[str, Any]] = None):
            self.internal_procedures_store[name] = {
                "description": description,
                "steps": steps,
                "metadata": metadata or {},
                "defined_at": "dummy_timestamp"
            }
            self.logger.info(f"(DCM) Procedure '{name}' added to internal store.")

    # --- Example ---
    print("\\n--- Running Procedure Sketch Example ---")
    sample_text_input = """
    Here's how to make a great cup of tea:
    1. Boil water.
    2. Pour water into a cup with a tea bag.
    3. Let it steep for 3-5 minutes.
    4. Remove tea bag and enjoy.
    """
    user_role = "user"
    current_session_id = "session_123"

    mock_dcm = DummyDialogueContextManager()

    # Simulate DCM processing this input
    capture_procedure_in_dcm(
        content=sample_text_input,
        role=user_role,
        session_id=current_session_id,
        dcm_instance=mock_dcm
    )

    print(f"\\nProcedures now in DCM's internal store: {json.dumps(mock_dcm.internal_procedures_store, indent=2)}")

    print("\\n--- Sketch Example Complete ---")

    # Further considerations for the actual system:
    # - Robust persistence strategy for DCM.internal_procedures_store (e.g., SQLite DB, dedicated file store).
    # - Error handling, backoff, and retry logic for scribing calls to ensure event logs are not lost.
    # - The exact structure of `event_data` and `source_metadata` for the `procedure_defined` event needs careful design.
    # - Definition of the new template for `procedure_defined` in ScribeIngestionProcessor (sovl_processor.py)
    #   to control how this event is represented for potential LoRA learning.
    # - Comprehensive design of the `ProcedureExecutor` within DCM for running these stored procedures (step tracking, context, etc.).
    # - Versioning of procedures if they can be updated.
    # - Security considerations if procedures can execute actions.
