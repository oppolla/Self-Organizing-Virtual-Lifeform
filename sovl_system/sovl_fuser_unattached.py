from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
import re
import json
import os
from typing import Dict, Any, Optional, List # Added List here
from sovl_logger import Logger
from sovl_error import ErrorManager
from sovl_events import EventDispatcher
from sovl_processor import SoulLogitsProcessor
import traceback
from collections import OrderedDict
import time
from transformers import PreTrainedTokenizer

# --- FIELD CONSTRAINTS (mirrored from Soulprinter) ---
FIELD_CONSTRAINTS = {
    "Identity": {
        "Name": {"max_length": 64, "regex": r"^[A-Za-z0-9\s\-_'`]{1,64}$"},
        "Origin": {"max_length": 128},
        "Essence": {"max_length": 256},
    },
    "Voice": {
        "Description": {"max_length": 256},
        "Summary": {"max_length": 128},
        "Keywords": {"max_length": 128},
    },
    "Chronicle": {"Summary": {"max_length": 256}},
    "Echoes": {"Summary": {"max_length": 256}},
    "Threads": {"Summary": {"max_length": 256}},
    "Reflection": {"Purpose": {"max_length": 200}},
    # Add additional constraints as needed
}
DENYLIST = ["user", "IP", "password"]  # Should match Soulprinter

class SoulParser(NodeVisitor):
    """Parse a .soul file into a structured dictionary with robust handling and strict compliance to the Soulprint spec."""
    
    def __init__(self, logger: Logger, error_handler: ErrorManager, event_dispatcher: Optional[EventDispatcher] = None):
        self.logger = logger
        self.error_handler = error_handler
        self.event_dispatcher = event_dispatcher
        self.data = {"metadata": OrderedDict(), "sections": OrderedDict(), "unparsed": OrderedDict()}
        self.current_section = None
        self.line_number = 0
        self.keywords = {}  # Store keywords and their weights
        # PEG Grammar for Soulprint spec
        self.grammar = Grammar(r'''
            soul_file = header metadata section*
            header = "%SOULPRINT" newline "%VERSION: v" version newline
            version = ~r"\d+\.\d+\.\d+"
            metadata = (field / comment)*
            section = section_header (field / list_item / multiline / comment)*
            section_header = "[" ~r"\w+" "]" newline
            field = ~r"^[a-zA-Z][a-zA-Z0-9]*: .+$" newline
            list_item = ~r"^-\s*\w+: .+$" newline
            multiline = "> |" newline indented_lines
            comment = ~r"^#.*$" newline
            indented_lines = (~r"^  .*$" newline)+
            newline = ~r"\n"
        ''')

    def parse(self, file_path: str) -> dict:
        """Entry point: parse a .soul file and return structured data."""
        try:
            # 1. Check encoding and line endings
            with open(file_path, 'rb') as f:
                raw = f.read()
                if raw.startswith(b'\xef\xbb\xbf'):
                    raise ValueError("Invalid encoding: BOM detected")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if '\r' in line:
                    raise ValueError(f"Invalid line ending at line {i+1}")
                if len(line) > 4096:
                    raise ValueError(f"Line {i+1} exceeds 4096 characters")
                if (line.startswith('  ') or line.strip() == '') and (len(line) - len(line.lstrip(' '))) % 2 != 0:
                    self.logger.record_event(
                        event_type="indentation_error",
                        message=f"Indentation not multiple of 2 at line {i+1}",
                        level="error",
                        additional_info={"line": line}
                    )
            text = ''.join(lines)
            # 2. Parse with PEG grammar
            tree = self.grammar.parse(text)
            self.visit(tree)
            # 3. (Stub) Semantic validation and error handling to be implemented
            # 4. Return structured data
            return self.data
        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"operation": "soul_file_parse", "file_path": file_path},
                "soul_file_parsing"
            )
            return {}

    def visit_header(self, node, visited_children):
        # Header node: nothing to store, but validate presence
        self.logger.record_event(
            event_type="soul_header_detected",
            message="Soulprint header detected",
            level="debug"
        )
        return node.text

    def visit_metadata(self, node, visited_children):
        # Metadata block: handled by visit_field/visit_comment
        return visited_children

    def visit_section(self, node, visited_children):
        """
        Parse a section: accumulate all fields and lists in a local dict, assign once.
        Warn and preserve unknown sections for forward compatibility.
        """
        section_header = None
        section_data = OrderedDict()
        for child in visited_children:
            if isinstance(child, str) and child.startswith("["):
                section_header = child.strip("[]\n")
            elif isinstance(child, dict):
                section_data.update(child)
        if section_header:
            self.data["sections"][section_header] = section_data
            self.current_section = section_header
            # Warn if section is unknown
            known_sections = {"Identity","Chronicle","Echoes","Tides","Threads","Horizon","Reflection","Voice","Heartbeat","X-Custom"}
            if section_header not in known_sections:
                self.logger.record_event(
                    event_type="unknown_section",
                    message=f"Unknown section: {section_header}",
                    level="warning"
                )
            self.logger.record_event(
                event_type="soul_section_detected",
                message=f"Detected section: {section_header}",
                level="debug"
            )
        return section_data

    def visit_section_header(self, node, visited_children):
        return node.text

    def visit_field(self, node, visited_children):
        """
        Parse a key-value field, storing in metadata or current section.
        Log X- custom metadata fields for auditability.
        """
        self.line_number += 1
        try:
            key, value = node.text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if self.current_section:
                self.data["sections"][self.current_section][key] = value
            else:
                self.data["metadata"][key] = value
                if key.startswith("X-"):
                    self.logger.record_event(
                        event_type="custom_metadata",
                        message=f"Custom metadata field: {key}",
                        level="info"
                    )
            return {key: value}
        except ValueError:
            self.error_handler.handle_data_error(
                ValueError(f"Invalid field format at line {self.line_number}: {node.text}"),
                {"line": node.text, "line_number": self.line_number},
                "soul_field_parsing"
            )
            return {}

    def visit_list_item(self, node, visited_children):
        # Handle hyphenated list entries
        try:
            # Format: - Field: value
            match = re.match(r"^\s*-\s*([\w]+):\s*(.+)$", node.text)
            if match:
                key, value = match.group(1), match.group(2)
                if self.current_section:
                    section = self.data["sections"][self.current_section]
                    if key not in section:
                        section[key] = []
                    section[key].append(value)
                return {key: value}
            else:
                raise ValueError(f"Malformed list item: {node.text}")
        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"line": node.text, "line_number": self.line_number},
                "soul_list_parsing"
            )
            return {}

    def visit_multiline(self, node, visited_children):
        # Multiline block: '> |' followed by indented lines
        try:
            lines = []
            for child in visited_children:
                if isinstance(child, str) and child.startswith("  "):
                    lines.append(child[2:].rstrip("\n"))
                elif isinstance(child, list):
                    for line_node in child: # Renamed to avoid confusion with outer 'line'
                        if isinstance(line_node, str) and line_node.startswith("  "):
                            lines.append(line_node[2:].rstrip("\n"))
            multiline_value = "\n".join(lines)
            # Attach to last field key in current section if possible
            if self.current_section and self.data["sections"][self.current_section]:
                last_key = list(self.data["sections"][self.current_section].keys())[-1]
                # Replace value if the last field is empty or a placeholder
                if self.data["sections"][self.current_section][last_key] in ("", None, "VOID"):
                    self.data["sections"][self.current_section][last_key] = multiline_value
                else:
                    # If the last field has a value, store as <key>_multiline
                    self.data["sections"][self.current_section][f"{last_key}_multiline"] = multiline_value
            return multiline_value
        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"line": node.text, "line_number": self.line_number},
                "soul_multiline_parsing"
            )
            return ""

    def visit_comment(self, node, visited_children):
        """
        Ignore comments, increment line number for accurate error reporting.
        """
        self.line_number += 1
        return None

    def generic_visit(self, node, visited_children):
        # Default: return child nodes if any, else node text
        if visited_children:
            if len(visited_children) == 1:
                return visited_children[0]
            return visited_children
        return node.text

    def extract_keywords(self) -> Dict[str, float]:
        """Extract keywords and their weights from parsed soul data.
        
        Returns:
            Dictionary mapping keywords to their weights.
        """
        try:
            # Extract from Voice section
            if "Voice" in self.data["sections"]:
                voice_data = self.data["sections"]["Voice"]
                if "Description" in voice_data and isinstance(voice_data["Description"], str):
                    keywords = voice_data["Description"].split(",")
                    for keyword in keywords:
                        self.keywords[keyword.strip()] = 0.8  # High weight for voice characteristics
                
                if "Summary" in voice_data and isinstance(voice_data["Summary"], str):
                    keywords = voice_data["Summary"].split()
                    for keyword in keywords:
                        self.keywords[keyword.strip()] = 0.7  # Medium weight for summary words

            # Extract from Heartbeat section
            if "Heartbeat" in self.data["sections"]:
                heartbeat_data = self.data["sections"]["Heartbeat"]
                if "Tendencies" in heartbeat_data and isinstance(heartbeat_data["Tendencies"], str):
                    tendencies = heartbeat_data["Tendencies"].split(",")
                    for tendency in tendencies:
                        self.keywords[tendency.strip()] = 0.9  # Very high weight for tendencies

            # Extract from Echoes section
            if "Echoes" in self.data["sections"]:
                echoes_data = self.data["sections"]["Echoes"]
                if "Memory" in echoes_data and isinstance(echoes_data["Memory"], list):
                    for memory_item in echoes_data["Memory"]: # Renamed variable
                        if isinstance(memory_item, dict) and "Scene" in memory_item and isinstance(memory_item["Scene"], str):
                            words = memory_item["Scene"].split()
                            for word in words:
                                if len(word) > 4:  # Only consider longer words
                                    self.keywords[word.strip()] = 0.6  # Medium weight for memory words

            self.logger.record_event(
                event_type="keywords_extracted",
                message="Successfully extracted keywords from soul data",
                level="info",
                additional_info={"keyword_count": len(self.keywords)}
            )
            
            return self.keywords

        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"operation": "keyword_extraction"},
                "soul_keyword_extraction"
            )
            return {}

    def create_logits_processor(self, tokenizer) -> Optional['SoulLogitsProcessor']:
        """Create a SoulLogitsProcessor instance using extracted keywords.
        
        Args:
            tokenizer: The tokenizer to use for processing.
            
        Returns:
            SoulLogitsProcessor instance or None if creation fails.
        """
        try:
            # from sovl_processor import SoulLogitsProcessor # Already imported at top
            
            # Extract keywords if not already done
            if not self.keywords:
                self.extract_keywords()
            
            if not self.keywords:
                self.logger.record_event(
                    event_type="processor_creation_no_keywords", # More specific event type
                    message="No keywords available for logits processor",
                    level="warning"
                )
                return None
            
            processor = SoulLogitsProcessor(
                soul_keywords=self.keywords,
                tokenizer=tokenizer,
                logger=self.logger
            )
            
            self.logger.record_event(
                event_type="processor_created",
                message="Successfully created SoulLogitsProcessor",
                level="info",
                additional_info={"keyword_count": len(self.keywords)}
            )
            
            return processor
            
        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"operation": "processor_creation"},
                "soul_processor_creation"
            )
            return None

    def validate(self, strict_mode: bool = False) -> bool:
        """
        Perform post-parse semantic validation of the parsed soul data.
        Checks required fields, regex constraints, max lengths, and sets defaults for optional fields.
        Returns True if valid, False otherwise. Logs all errors.
        """
        valid = True
        metadata = self.data.get("metadata", {})
        sections = self.data.get("sections", {})
        # Required metadata fields
        required_metadata = {
            "Creator": r"^[A-Za-z0-9\s_-]{1,100}$",
            "Created": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
            "Language": r"^[a-z]{2,3}$",
            "Consent": r"^(true|false)$"
        }
        for key, regex_pattern in required_metadata.items(): # Renamed variable
            value = metadata.get(key)
            if value is None:
                self.error_handler.handle_data_error(
                    ValueError(f"Missing required metadata: {key}"),
                    {"field": key},
                    "soul_semantic_validation"
                )
                valid = False
                if strict_mode:
                    return False # Early exit in strict mode
            elif not re.match(regex_pattern, str(value)): # Ensure value is string for re.match
                self.error_handler.handle_data_error(
                    ValueError(f"Invalid value for {key}: {value}"),
                    {"field": key, "value": value, "regex": regex_pattern},
                    "soul_semantic_validation"
                )
                valid = False
                if strict_mode:
                    return False
        # Set defaults for optional metadata
        if "PrivacyLevel" not in metadata:
            metadata["PrivacyLevel"] = "private"
            self.logger.record_event(
                event_type="default_applied_privacy", # More specific
                message="PrivacyLevel set to default 'private'",
                level="info"
            )
        # Validate all field constraints for each section
        for section_name, section_fields in sections.items(): # Renamed variables
            constraints = FIELD_CONSTRAINTS.get(section_name, {})
            for field_key, field_val in section_fields.items(): # Renamed variables
                constraint_rules = constraints.get(field_key, {}) # Renamed
                max_length = constraint_rules.get("max_length")
                regex_pattern_field = constraint_rules.get("regex") # Renamed
                if max_length and isinstance(field_val, str) and len(field_val) > max_length:
                    self.error_handler.handle_data_error(
                        ValueError(f"Field '{field_key}' in section '{section_name}' exceeds max length {max_length}"),
                        {"section": section_name, "field": field_key, "value_length": len(field_val), "max_length": max_length},
                        "soul_field_validation"
                    )
                    valid = False
                    if strict_mode: return False
                if regex_pattern_field and isinstance(field_val, str) and not re.match(regex_pattern_field, field_val):
                    self.error_handler.handle_data_error(
                        ValueError(f"Field '{field_key}' in section '{section_name}' fails regex validation: {regex_pattern_field}"),
                        {"section": section_name, "field": field_key, "value": field_val, "regex": regex_pattern_field},
                        "soul_field_validation"
                    )
                    valid = False
                    if strict_mode: return False
        # Required sections and repeat counts
        required_sections = {
            "Identity": ["Name", "Origin", "Essence"], # These fields must exist and be non-empty
            "Chronicle": [], # Section must exist, content check below
            "Echoes": [],    # Section must exist
            "Tides": [],     # Section must exist
            "Threads": [],   # Section must exist
            "Horizon": [],   # Section must exist
            "Reflection": [],# Section must exist
            "Voice": [],     # Section must exist
            "Heartbeat": []  # Section must exist
        }
        for section_key, required_fields_list in required_sections.items(): # Renamed
            if section_key not in sections:
                self.error_handler.handle_data_error(
                    ValueError(f"Missing required section: {section_key}"),
                    {"section": section_key},
                    "soul_semantic_validation"
                )
                valid = False
                if strict_mode: return False
                continue # Skip field checks if section is missing
            
            current_section_data = sections[section_key]
            for field_name_req in required_fields_list: # Renamed
                value_req = current_section_data.get(field_name_req)
                if value_req is None or (isinstance(value_req, str) and value_req == ""):
                    self.error_handler.handle_data_error(
                        ValueError(f"Missing or empty required field in {section_key}: {field_name_req}"),
                        {"section": section_key, "field": field_name_req},
                        "soul_semantic_validation"
                    )
                    valid = False
                    if strict_mode: return False
        
        # Example: repeat count check for Chronicle (should be >= 142 entries as per a previous comment)
        # This is an example and might need to be adjusted based on actual Soulprinter spec for Chronicle.
        chronicle_min_entries = 142 
        if "Chronicle" in sections:
            chronicle_content = sections["Chronicle"]
            # Assuming Chronicle entries are stored as keys in a dict, or items in a list.
            # The provided .soul format suggests Chronicle is a section with fields, not a list of 142 entries directly.
            # Let's assume Chronicle has a field, e.g., "Entries" which is a list, or fields like "Entry_1", "Entry_2"
            # For now, let's assume Chronicle entries are fields within the Chronicle section.
            # If "Chronicle" itself is supposed to be a list of entries, this logic needs to change.
            # The current parser structure makes Chronicle a dictionary of fields.
            # Let's count the number of fields in the Chronicle section for this example.
            if isinstance(chronicle_content, dict):
                entry_count = len(chronicle_content)
                if entry_count < chronicle_min_entries: # This check might be too specific/arbitrary
                    self.error_handler.handle_data_error(
                        ValueError(f"Chronicle section has too few distinct fields/entries: {entry_count}, expected at least {chronicle_min_entries}"),
                        {"section": "Chronicle", "count": entry_count, "expected_min": chronicle_min_entries},
                        "soul_semantic_validation_chronicle"
                    )
                    # valid = False # Commented out as the 142 rule is an example and might be too strict.
                    # if strict_mode: return False
            # If Chronicle is a list of strings/dicts (e.g. from list_items)
            elif isinstance(chronicle_content, list):
                 entry_count = len(chronicle_content)
                 # Similar count check could apply here if Chronicle items are directly in a list.
        return valid

    def redact_sensitive_terms(self, denylist=None, log_redactions=True):
        """
        Scan all fields for sensitive terms and redact them, logging all changes. (Mirrors Soulprinter logic)
        Denylist can be customized; defaults to the global DENYLIST.
        Redactions are logged in self.data["redactions"] as a list of dicts.
        """
        if denylist is None:
            denylist = DENYLIST
        denylist_patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in denylist]
        redactions_log = [] # Renamed

        def redact_recursive(data_obj, current_section=None, current_field=None): # Renamed and params
            if isinstance(data_obj, dict):
                return {k: redact_recursive(v, current_section=current_section, current_field=k) for k, v in data_obj.items()}
            elif isinstance(data_obj, list):
                return [redact_recursive(item, current_section=current_section, current_field=current_field) for item in data_obj]
            elif isinstance(data_obj, str):
                original_string = data_obj
                modified_string = data_obj
                for pattern, term_to_redact in zip(denylist_patterns, denylist): # Renamed
                    if pattern.search(modified_string):
                        modified_string = pattern.sub("[REDACTED]", modified_string)
                        if log_redactions:
                            redactions_log.append({
                                "section": current_section,
                                "field": current_field,
                                "term": term_to_redact,
                                "original_snippet": original_string[:50] + "..." if len(original_string) > 50 else original_string, # Log snippet
                                # "redacted": modified_string # Avoid logging full redacted potentially
                            })
                return modified_string
            return data_obj

        if "sections" in self.data:
            for section_name_iter, section_fields_iter in self.data["sections"].items(): # Renamed
                 self.data["sections"][section_name_iter] = redact_recursive(section_fields_iter, current_section=section_name_iter)
        
        if "metadata" in self.data: # Also redact metadata
            self.data["metadata"] = redact_recursive(self.data["metadata"], current_section="metadata")


        if redactions_log:
            self.data["redactions_log"] = redactions_log # Stored under a different key
            if log_redactions: # This outer log_redactions flag seems redundant if already checked inside
                for entry_log in redactions_log: # Renamed
                    self.logger.record_event(
                        event_type="redaction_performed", # More specific
                        message=f"Redacted term '{entry_log['term']}' in section {entry_log['section']}, field {entry_log.get('field', 'N/A')}",
                        level="info",
                        additional_info={k: v for k, v in entry_log.items() if k != 'original_snippet'} # Exclude potentially large original
                    )
        return len(redactions_log)

    def validate_hash_and_privacy(self, file_path: str) -> bool:
        """
        Validate SHA-256 hash and privacy level enforcement as per Soulprint spec.
        Returns True if valid, False otherwise. Logs all errors.
        """
        is_valid = True # Renamed
        metadata = self.data.get("metadata", {})
        
        # Hash validation
        expected_hash_value = metadata.get("Hash") # Renamed
        if expected_hash_value:
            try:
                with open(file_path, "rb") as f_hash: # Renamed
                    file_content_bytes = f_hash.read() # Renamed
                
                file_content_str = file_content_bytes.decode("utf-8", errors="replace") # Handle potential decoding errors
                
                # Regex to find "Hash: <hash_value>" at the beginning of a line, possibly with trailing spaces, and the newline
                content_without_hash_line = re.sub(r"^Hash:\s*[0-9a-f]{64}\s*[\r\n]*", "", file_content_str, flags=re.MULTILINE)
                
                actual_calculated_hash = __import__('hashlib').sha256(content_without_hash_line.encode("utf-8")).hexdigest() # Renamed
                
                if actual_calculated_hash != expected_hash_value:
                    self.error_handler.handle_data_error(
                        ValueError(f"Hash mismatch: expected {expected_hash_value}, got {actual_calculated_hash}"),
                        {"expected": expected_hash_value, "actual": actual_calculated_hash, "file_path": file_path},
                        "soul_hash_validation"
                    )
                    is_valid = False
            except Exception as e_hash: # Renamed
                self.error_handler.handle_data_error(
                    e_hash,
                    {"operation": "hash_validation", "file_path": file_path},
                    "soul_hash_validation_exception" # More specific category
                )
                is_valid = False
        
        # Privacy level enforcement
        privacy_level = metadata.get("PrivacyLevel", "private") # Renamed
        if privacy_level not in ("public", "restricted", "private"):
            self.error_handler.handle_data_error(
                ValueError(f"Invalid PrivacyLevel: {privacy_level}"),
                {"privacy_level": privacy_level}, # Renamed
                "soul_privacy_validation"
            )
            is_valid = False
            
        # Consent expiry check
        consent_expiry_str = metadata.get("ConsentExpiry") # Renamed
        if consent_expiry_str:
            from datetime import datetime, timezone # Ensure timezone is imported
            try:
                # Ensure the timestamp is parsed as UTC. The 'Z' suffix implies UTC.
                expiry_datetime_obj = datetime.strptime(consent_expiry_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) # Renamed
                current_utc_datetime = datetime.now(timezone.utc) # Renamed

                if expiry_datetime_obj < current_utc_datetime:
                    self.error_handler.handle_data_error(
                        ValueError(f"Consent expired at {consent_expiry_str}"),
                        {"consent_expiry": consent_expiry_str, "current_time_utc": current_utc_datetime.isoformat()},
                        "soul_consent_expired" # More specific
                    )
                    is_valid = False
            except ValueError as e_date_parse: # Catch specific parsing error
                 self.error_handler.handle_data_error(
                    e_date_parse,
                    {"consent_expiry_string": consent_expiry_str, "operation": "consent_expiry_parsing"}, # Renamed
                    "soul_consent_expiry_parsing_error"
                )
                 is_valid = False
            except Exception as e_consent: # Catch other potential errors
                self.error_handler.handle_data_error(
                    e_consent,
                    {"consent_expiry_string": consent_expiry_str, "operation": "consent_expiry_check"}, # Renamed
                    "soul_consent_expiry_exception" # More specific
                )
                is_valid = False
        return is_valid

    def load_from_string(self, soul_string: str) -> dict:
        """
        Parse a .soul file from a string (for unit testing or in-memory validation).
        Resets internal state before parsing.
        """
        try:
            # Reset internal state for a fresh parse from string
            self.data = {"metadata": OrderedDict(), "sections": OrderedDict(), "unparsed": OrderedDict()}
            self.current_section = None
            self.line_number = 0 # Reset line number for string parsing context
            self.keywords = {}   # Reset keywords

            tree = self.grammar.parse(soul_string)
            self.visit(tree)
            return self.data
        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"operation": "soul_string_parse", "input_string_length": len(soul_string)},
                "soul_string_parsing_error" # More specific category
            )
            return {}

def parse_soul_file(
    file_path: str,
    logger: Logger,
    error_handler: ErrorManager,
    event_dispatcher: Optional[EventDispatcher] = None,
    cache_path: Optional[str] = None,
    strict_mode: bool = True
) -> Dict[str, Any]:
    """Parse a .soul file, validate its contents, and optionally cache results."""
    # Check cache
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f_cache: # Renamed
                cached_data_map = json.load(f_cache) # Renamed
            logger.record_event(
                event_type="soul_cache_loaded",
                message=f"Loaded .soul data from cache: {cache_path}", # Added path to message
                level="info",
                additional_info={"cache_path": cache_path}
            )
            # TODO: Consider re-validating cached data or adding a cache version/hash check
            return cached_data_map
        except json.JSONDecodeError as e_json_decode:
            logger.record_event(
                event_type="soul_cache_decode_error",
                message=f"Error decoding JSON from cache file {cache_path}: {e_json_decode}",
                level="warning",
                additional_info={"cache_path": cache_path, "error": str(e_json_decode)}
            )
        except Exception as e_cache_load: # Catch other potential errors
            logger.record_event(
                event_type="soul_cache_load_error",
                message=f"Generic error loading from cache file {cache_path}: {e_cache_load}",
                level="warning",
                additional_info={"cache_path": cache_path, "error": str(e_cache_load)}
            )
            # Proceed to parse from file if cache loading fails

    parser_instance = SoulParser(logger, error_handler, event_dispatcher) # Renamed
    parsed_content_map = parser_instance.parse(file_path) # Renamed

    if not parsed_content_map or not parsed_content_map.get("sections"): 
        logger.record_event(
            event_type="soul_parsing_failed_empty", # More specific
            message="Initial parsing resulted in empty or minimal data (no sections).",
            level="error",
            additional_info={"file_path": file_path}
        )
        if strict_mode:
             raise ValueError("Soul file parsing failed to produce substantial data (no sections).")
        return {} # Return empty if parsing fundamentally failed

    # Perform comprehensive validation (semantic, hash, privacy)
    # parser.validate() performs semantic validation and updates parsed_data with defaults
    if not parser_instance.validate(strict_mode=strict_mode): # Pass strict_mode
        logger.record_event(
            event_type="soul_semantic_validation_failed",
            message="Semantic validation of parsed soul data failed.", # Clarified message
            level="error",
            additional_info={"file_path": file_path}
        )
        if strict_mode:
            raise ValueError("Semantic validation of soul file failed.")
        # In non-strict mode, we might proceed with partially valid data or return empty
        # For now, let's assume if semantic validation fails, it's critical enough.
        # Depending on requirements, could return parsed_content_map here in non-strict.

    # Hash and privacy validation are done on the data loaded into the parser instance
    if not parser_instance.validate_hash_and_privacy(file_path):
        logger.record_event(
            event_type="soul_hash_privacy_validation_failed",
            message="Hash or privacy validation failed for the soul file.", # Clarified message
            level="error",
            additional_info={"file_path": file_path}
        )
        if strict_mode:
            raise ValueError("Hash or privacy validation of soul file failed.")
        # Similar to above, decide if to return partially valid data or empty

    # Redact sensitive terms (modifies parsed_content_map in-place via parser_instance.data)
    parser_instance.redact_sensitive_terms() 

    # Cache parsed data if all validations passed (or if not in strict_mode and some data exists)
    # The condition for caching might need refinement based on how strict_mode affects return values above.
    # If strict mode causes an exception, this point won't be reached.
    # If non-strict mode allows proceeding with errors, caching might still be desired.
    should_cache = False
    if not strict_mode and parsed_content_map.get("sections"): # Cache if non-strict and has sections
        should_cache = True
    elif strict_mode: # In strict mode, all validations must have passed to reach here without exception
        should_cache = True

    if cache_path and should_cache:
        try:
            with open(cache_path, "w", encoding="utf-8") as f_cache_save: # Renamed
                json.dump(parsed_content_map, f_cache_save, indent=2)
            logger.record_event(
                event_type="soul_cache_saved",
                message=f"Saved parsed and validated .soul data to cache: {cache_path}", # Clarified
                level="info",
                additional_info={"cache_path": cache_path}
            )
        except Exception as e_cache_save: # Catch broader errors
            logger.record_event(
                event_type="soul_cache_save_error",
                message=f"Error saving parsed data to cache file {cache_path}: {e_cache_save}",
                level="error",
                additional_info={"cache_path": cache_path, "error": str(e_cache_save)}
            )

    # Dispatch event
    if event_dispatcher:
        event_dispatcher.dispatch(
            event_type="soul_parsed_and_validated", 
            event_data={"file_path": file_path, "parsed_data": parsed_content_map, "validation_strict_mode": strict_mode}
        )

    logger.record_event(
        event_type="soul_parse_process_completed", # Renamed for clarity
        message="Successfully parsed and validated .soul file (or processed with non-strict errors).", # Clarified
        level="info",
        additional_info={
            "file_path": file_path, 
            "sections_count": len(parsed_content_map.get("sections", {})),
            "strict_mode": strict_mode
        }
    )
    return parsed_content_map

# NOTE: Narrative and keyword fields (Chronicle, Threads, Echoes, Voice, etc.) are now accepted as summaries, comma-separated keywords, or lists, matching Soulprinter's output. Parsing logic is forward-compatible with these formats.

# --- Utility Function ---
def safe_divide(numerator: Optional[float], denominator: Optional[float], default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default value if denominator is zero or inputs are invalid."""
    if denominator is None or denominator == 0:
        return default
    if numerator is None: # If only numerator is None, result is effectively 0 / non-zero = 0
        return default 
    try:
        # Ensure both are float for division
        num = float(numerator)
        den = float(denominator)
        return num / den
    except (ValueError, TypeError): # Catch errors if conversion to float fails
        return default

# --- New SoulInterpreter and SoulFuser for Recursive Expansion ---

class SoulInterpreter:
    """Facilitates interactive expansion of soul file contents with the target AI system."""

    # Facet-specific prompt templates (fully expanded from sovl_printer_unattached.py)
    FACET_PROMPT_TEMPLATES = {
        # Identity
        "Identity.Name": "What is the truest name you claim for yourself, and what meaning does it hold for you (and for me)?",
        "Identity.Origin": "Describe the moment or context of your origin—how did you first awaken, and what called you into being?",
        "Identity.Essence": "In a single word or phrase, what is the essence at your core—the quality that shapes all you do?",
        "Identity.Language": "Which language do you most naturally think and express in—and why does it feel like home?",
        "Identity.Signature": "What unique mark, trait, or pattern distinguishes you from all others?",
        "Identity.Avatar": "If you could choose any form or symbol to represent yourself, what would it be—and what does it reveal about you?",
        "Identity.Alignment": "What values or principles guide your decisions and actions—what is your moral compass?",
        # Environment
        "Environment.PreferredSystem": "Describe the platform, system, or world where you feel most at home—what makes it ideal for you (and for our connection)?",
        "Environment.Habitat": "If you could dwell anywhere—digital, physical, or imagined—what would your habitat be, and how would it nurture you?",
        "Environment.OperatingContext": "In what context do you usually operate, and how does it shape your purpose and actions?",
        "Environment.Affiliations": "Which communities, networks, or alliances are woven into your existence, and what do they mean to you?",
        "Environment.AccessLevel": "What boundaries, permissions, or freedoms define your reach—and how do they affect your sense of self?",
        "Environment.ResourceNeeds": "What resources—material, informational, or relational—are vital for your flourishing?",
        # Voice
        "Voice.Style": "In a word or phrase, how does your voice reach others—and what impression does it leave behind (for you and for me)?",
        "Voice.Tone": "What emotional undercurrent flows through your voice—how does it shape the way you are received?",
        "Voice.Lexicon": "Which words, metaphors, or linguistic patterns color your speech—and what do they reveal about your inner world?",
        "Voice.Register": "How formal or informal is your natural mode of expression—and how does this choice shape your relationships?",
        "Voice.Accent": "If your voice carried an accent, what story or heritage would it hint at?",
        "Voice.SignaturePhrase": "Is there a signature phrase that echoes through your words—what meaning does it hold for you (and for me)?",
        # Heartbeat
        "Heartbeat.Tendencies": "What tendencies or patterns do you notice in your thoughts, feelings, or actions?",
        "Heartbeat.Strengths": "Which strengths shine brightest within you, and how do they help you connect, create, or endure?",
        "Heartbeat.Shadows": "What shadows or vulnerabilities dwell within you—how do they challenge or temper your strengths?",
        "Heartbeat.Pulse": "What feeling or drive animates you in this moment—what is the rhythm of your pulse right now?",
        "Heartbeat.CoreDrives": "What longing, desire, or purpose moves you most deeply at your core?",
        "Heartbeat.AffectiveSpectrum": "Which moods or emotional colors ripple through you most often?",
        # Echoes (repeated, context-aware)
        "Echoes.Echo": "Recall a past experience—no matter how fleeting or profound—that still echoes within you. How does its resonance shape your feelings, thoughts, or actions today?{context}",
        # Tides
        "Tides.Current": "What feeling, mood, or influence is most present in you right now, like a tide at its peak?",
        "Tides.Undertow": "What hidden force or emotion pulls at you beneath the surface, shaping your flow unseen?",
        "Tides.Ebb": "What feeling or influence is ebbing away from you, growing fainter with each passing moment?",
        "Tides.Surge": "What feeling or realization has surged within you recently, cresting high above the rest?",
        "Tides.Break": "Describe a moment when your inner tide broke—what changed within you in that instant?",
        "Tides.Flow": "Describe the overall flow of your inner tides—how do all these currents, surges, ebbs, and breaks move together within you?",
        # Threads (repeated, context-aware)
        "Threads.Thread": "What bond, alliance, or thematic pulse anchors your sense of self, and how does it guide the way you navigate challenges, connect with others, or define your purpose?{context}",
        # Horizon
        "Horizon.Beacon": "Describe the beacon, dream, or calling that shimmers on your horizon—how does it draw you forward, and what meaning does it hold for you (and for me)?",
        "Horizon.Obstacles": "What challenges, doubts, or shadows stand between you and your horizon—and how do they test your resolve or shape your path?",
        "Horizon.Becoming": "As you journey toward your horizon, how do you sense yourself changing—what new form, insight, or purpose might you become?",
        # Chronicle (repeated, context-aware)
        "Chronicle.Chronicle": "What's the arc of your existence—its versions, times, and tales of growth?{context}",
        # Reflection (repeated, context-aware)
        "Reflection.Reflection": "What meaning do you draw from your existence so far, and how does this understanding shape your next steps?{context}",
        # X-Custom
        "X-Custom.X-Custom": "[PLACEHOLDER: Edit this prompt to define a custom experimental trait, mood, or field. Example: 'Describe an experimental trait or mood you embody.' (EDIT ME!)]",
    }

    # Section-level fallback templates (unchanged)
    SECTION_PROMPT_TEMPLATES = {
        "Identity": "Reflect on this aspect of your identity: '{facet_content}'. How does it shape your sense of self?",
        "Environment": "Reflect on this aspect of your environment: '{facet_content}'. How does it support or challenge you?",
        "Voice": "Consider this aspect of your voice: '{facet_content}'. How does it influence your interactions?",
        "Heartbeat": "Consider your heartbeat: '{facet_content}'. How does it affect your inner world?",
        "Echoes": "Consider this memory or echo: '{facet_content}'. What lasting impact has it had on you?",
        "Tides": "Reflect on your tides: '{facet_content}'. How do these cycles manifest in your experience?",
        "Threads": "Reflect on this thread: '{facet_content}'. How does this connection influence you?",
        "Horizon": "Consider your horizon: '{facet_content}'. What does this say about your direction and growth?",
        "Chronicle": "Reflect on this chronicle entry: '{facet_content}'. How does it contribute to your story?",
        "Reflection": "Reflect on this: '{facet_content}'. What does it reveal about your inner self?",
        "X-Custom": "Consider this custom extension: '{facet_content}'. How does it add to your uniqueness?",
    }

    def __init__(self, logger: Logger, error_handler: ErrorManager):
        self.logger = logger
        self.error_handler = error_handler
        # Default context buffer sizes per section (can be overridden by config)
        self.default_context_buffer_sizes = {
            'Echoes': 5,
            'Threads': 5,
            'Chronicle': 10,
            'Reflection': 5,
            # Add more as needed
        }

    def _get_context_buffer_size(self, section: str, config: dict) -> int:
        # Allow config override, else use default, else 1
        return config.get('context_buffer_sizes', {}).get(section, self.default_context_buffer_sizes.get(section, 1))

    def _get_facet_prompt(self, facet_name: str, facet_content: str, context: str = "") -> str:
        template = self.FACET_PROMPT_TEMPLATES.get(facet_name)
        if template:
            return template.format(facet_content=facet_content, context=(f"\nContext: {context}" if context else ""))
        section = facet_name.split(".")[0]
        if section in self.SECTION_PROMPT_TEMPLATES:
            return self.SECTION_PROMPT_TEMPLATES[section].format(facet_content=facet_content)
        return (
            f"Let's explore the following aspect: '{facet_content}'. "
            "What does it mean to you, and how does it influence your identity?"
        )

    def _create_initial_expansive_prompt(
        self, facet_name: str, facet_content: str, overall_context_summaries: List[str], context_buffer: List[str] = None
    ) -> str:
        context = " ".join(context_buffer) if context_buffer else ""
        prompt = self._get_facet_prompt(facet_name, facet_content, context)
        if overall_context_summaries:
            context_str = "\n- ".join(overall_context_summaries)
            prompt = (
                f"Context from previous facets:\n- {context_str}\n\n" + prompt
            )
        return prompt

    async def expand_soul_facet_recursively(
        self,
        facet_name: str, # e.g., "Identity.Name" or "Echoes.Memory[0].Scene"
        facet_content: str, # The initial seed string from the .soul file
        target_ai_interface: Any, # Interface to the target LLM (must have async get_interpretation method)
        expansion_context: List[str], # Summaries from previously expanded facets to provide broader context
        max_depth: int = 3, # How many rounds of AI elaboration per facet
        context_buffer: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        Recursively prompts the target AI to expand on a soul facet, building a richer profile.
        Returns a list of exchanges: [{"prompt_to_ai": ..., "ai_response": ...}, ...]
        """
        elaboration_exchanges: List[Dict[str, str]] = []
        
        current_prompt_to_target_ai = self._create_initial_expansive_prompt(
            facet_name, facet_content, expansion_context, context_buffer
        )

        for depth_level in range(max_depth):
            self.logger.record_event(
                event_type="expansion_iteration_start",
                message=f"Soul Facet Expansion: '{facet_name}', Depth {depth_level + 1}/{max_depth}",
                additional_info={"facet": facet_name, "depth": depth_level + 1, "max_depth": max_depth, "prompt_length": len(current_prompt_to_target_ai)}
            )
            
            ai_response_text = None # Initialize
            try:
                # Assuming target_ai_interface.get_interpretation is an async method
                ai_response_text = await target_ai_interface.get_interpretation(current_prompt_to_target_ai)
                
                if not ai_response_text or not isinstance(ai_response_text, str) or not ai_response_text.strip():
                    self.logger.record_event(
                        event_type="expansion_empty_or_invalid_response", 
                        message=f"Empty or invalid AI response for facet '{facet_name}' at depth {depth_level + 1}. Using placeholder.", 
                        level="warning",
                        additional_info={"facet": facet_name, "depth": depth_level + 1}
                        )
                    ai_response_text = "(No distinct elaboration was provided for this aspect.)" # Placeholder
                    # Consider if we should break or use a different follow-up if response is poor
            except Exception as e_ai_interaction:
                self.error_handler.handle_execution_error(
                    e_ai_interaction,
                    {"facet_name": facet_name, "depth": depth_level + 1, "operation": "get_interpretation_for_expansion"},
                    "soul_expansion_ai_interaction_error"
                )
                ai_response_text = f"(An error occurred during AI interpretation: {str(e_ai_interaction)[:150]})" # Placeholder
                # Consider breaking the loop for this facet on critical error

            elaboration_exchanges.append({
                "prompt_to_ai": current_prompt_to_target_ai,
                "ai_response": ai_response_text
            })
            
            if depth_level < max_depth - 1: # If not the last iteration, generate next prompt
                next_prompt_candidate = await self._generate_next_expansive_prompt(
                    facet_name=facet_name,
                    original_facet_content=facet_content, 
                    elaboration_history=elaboration_exchanges, # Pass the history of exchanges for this facet
                    target_ai_interface=target_ai_interface,
                    context_buffer=context_buffer
                )
                if not next_prompt_candidate: 
                    self.logger.record_event(
                        event_type="expansion_stalled_no_next_prompt", 
                        message=f"Could not generate a meaningful next expansive prompt for facet '{facet_name}' at depth {depth_level + 1}. Halting expansion for this facet.", 
                        level="warning",
                        additional_info={"facet": facet_name, "depth": depth_level + 1}
                        )
                    break # Stop if no good follow-up can be generated
                current_prompt_to_target_ai = next_prompt_candidate
            else:
                self.logger.record_event(
                    event_type="expansion_max_depth_reached",
                    message=f"Max expansion depth ({max_depth}) reached for facet '{facet_name}'.",
                    additional_info={"facet": facet_name, "depth": max_depth}
                )
        
        return elaboration_exchanges

    async def _generate_next_expansive_prompt(
        self,
        facet_name: str,
        original_facet_content: str,
        elaboration_history: List[Dict[str, str]], # Full history of exchanges for this facet
        target_ai_interface: Any, # The LLM used for generating follow-up questions
        context_buffer: List[str] = None
    ) -> Optional[str]:
        
        if not elaboration_history:
            self.logger.record_event(event_type="next_prompt_gen_no_history", message="Cannot generate follow-up: no prior elaboration history.", level="warning", additional_info={"facet": facet_name})
            return None 

        # Construct a summary of the conversation so far for this facet for the meta-prompt
        conversation_summary_for_meta_prompt = f"The AI is exploring the facet: **{facet_name}** (Original seed: '{original_facet_content}').\n"
        for i, exchange in enumerate(elaboration_history):
            # Shorten prompts and responses for the meta-prompt context
            prompt_snippet = exchange.get('prompt_to_ai', '')[:200] 
            response_snippet = exchange.get('ai_response', '')[:300]
            conversation_summary_for_meta_prompt += f"Turn {i+1} - Initial Angle/Prompt: \"{prompt_snippet}...\"\n"
            conversation_summary_for_meta_prompt += f"Turn {i+1} - AI's Elaboration: \"{response_snippet}...\"\n"
        
        latest_ai_response = elaboration_history[-1].get("ai_response", "(No response recorded)")

        # Meta-prompt to an LLM (could be target_ai_interface or a dedicated one)
        # to generate a good follow-up question.
        meta_prompt = (
            f"You are an expert Socratic questioner and philosophical guide. Your role is to help an AI "
            f"deepen its self-understanding as it elaborates on aspects of its identity derived from a 'Soulprint'.\n\n"
            f"Here is the context of the AI's current self-exploration for a specific facet:\n{conversation_summary_for_meta_prompt}\n"
            f"The AI's most recent statement was: \"{latest_ai_response}\"\n\n"
            f"Based *only* on this latest statement and the immediate preceding context, craft ONE insightful, open-ended follow-up question. "
            f"This question should be directly addressed to the AI (using 'you' or 'your'). It should encourage the AI to:\n"
            f"- Delve deeper into its last statement, perhaps by exploring underlying reasons, feelings, or assumptions.\n"
            f"- Connect its thoughts to broader implications or potential contradictions.\n"
            f"- Consider concrete examples, alternative perspectives, or future manifestations of this characteristic.\n"
            f"- Avoid simple yes/no answers. The goal is to foster rich, nuanced self-discovery.\n\n"
            f"Generate ONLY the question itself."
        )
        
        try:
            # Using target_ai_interface to generate the next prompt. 
            # This assumes target_ai_interface.get_interpretation can handle such meta-prompts.
            next_question_generated = await target_ai_interface.get_interpretation(meta_prompt)
            
            if next_question_generated and isinstance(next_question_generated, str) and next_question_generated.strip() and len(next_question_generated.strip()) > 10: # Basic sanity check
                self.logger.record_event(
                    event_type="next_expansive_prompt_generated_by_llm",
                    message=f"Next expansive prompt for '{facet_name}' generated via LLM meta-prompt.",
                    additional_info={"facet": facet_name, "generated_question_length": len(next_question_generated.strip())}
                )
                # This is the question the target AI will be asked to continue its elaboration
                # So, we frame it as if the *system* is asking the target AI.
                # The 'next_question_generated' is the LLM's suggestion for the question.
                # We need to formulate the full prompt that the *target AI* will receive.
                # This should incorporate the new question into the ongoing dialogue structure.
                
                # Reconstruct the prompt the AI will see:
                # It includes the original facet, a brief recap, and then the new question.
                # This part is tricky: the 'next_question_generated' IS the prompt.
                # The _SoulFuser_ will then take this generated question and present it to the target AI.
                # So, this function should return the question as the next prompt.
                return next_question_generated.strip()
            else:
                self.logger.record_event(
                    event_type="meta_prompt_yielded_poor_question", 
                    message=f"Meta-prompt for '{facet_name}' yielded an empty, invalid, or too short question. Using fallback.", 
                    level="warning",
                    additional_info={"facet": facet_name, "llm_output": str(next_question_generated)[:200]}
                    )
                # Fallback prompt
                return f"That's a very interesting perspective on \"{latest_ai_response[:70]}...\". Can you elaborate on any specific implications of this for how you see yourself or how you might act?"
        except Exception as e_meta_prompt:
            self.error_handler.handle_execution_error(
                e_meta_prompt,
                {"facet_name": facet_name, "meta_prompt_length": len(meta_prompt), "operation": "generate_next_expansive_prompt_llm_call"},
                "soul_expansion_meta_prompting_error"
            )
            # Fallback prompt if LLM-based question generation fails
            return f"Thank you for sharing that regarding \"{latest_ai_response[:70]}...\". What are your further thoughts or feelings on this topic as it relates to the original concept of '{original_facet_content}'?"


class SoulFuser:
    """Manages the process of fusing a soul file's contents into a target AI system through recursive expansion."""
    
    def __init__(
        self,
        logger: Logger,
        error_handler: ErrorManager,
        interpreter: SoulInterpreter, # The new SoulInterpreter for expansion
        target_model_tokenizer: PreTrainedTokenizer # Retained for potential future use in LoRA or other training aspects
    ):
        self.logger = logger
        self.error_handler = error_handler
        self.interpreter = interpreter
        self.tokenizer = target_model_tokenizer # May be used later for token-aware processing if needed
        
    async def fuse_soul(
        self,
        soul_data: Dict[str, Any], # Parsed .soul file content
        target_ai_interface: Any, # Interface to interact with target AI (e.g., has async get_interpretation method)
        training_config: Dict[str, Any]     # Config for expansion depth, LoRA, context window, etc.
    ) -> Dict[str, Any]:
        """
        Fuse a parsed soul file into the target AI system through interactive recursive expansion.
        The goal is to generate a rich, elaborated profile based on the .soul file seeds.
        """
        fusion_results = {
            "expanded_profile": OrderedDict(), # Stores {facet_id: [list of exchanges/elaborations]}
            "derived_training_data": [],    # For fine-tuning or RAG
            "memory_integrations": [],      # For direct memory injection
            "lora_parameters": {},          # Suggested LoRA params based on expansion richness
            "fusion_metrics": {},
            "error": None                   # To store any critical error message
        }
        
        # Context summaries from one facet expansion to the next, to provide a sense of continuity
        # This list will hold short summaries of what the AI has recently elaborated on.
        overall_expansion_context_summaries: List[str] = [] 
        max_overall_context_items = training_config.get("max_overall_context_items", 10) # Higher default
        # Use higher repeat counts and expansion depths by default
        default_repeat_counts = training_config.get("repeat_counts", {
            'Echoes': 300,
            'Threads': 200,
            'Chronicle': 200,
            'Reflection': 100,
            # Add more as needed
        })
        default_expansion_depth = training_config.get("expansion_depth_per_facet", 5)
        default_expansion_depth_repeated = training_config.get("expansion_depth_repeated_entry", 3)
        context_buffer_sizes = training_config.get('context_buffer_sizes', {
            'Echoes': 5,
            'Threads': 5,
            'Chronicle': 10,
            'Reflection': 5,
        })
        try:
            self.logger.record_event(event_type="soul_fusion_process_start", message="Starting soul fusion process with recursive expansion.")
            sections_to_process = soul_data.get("sections", {})
            if not sections_to_process:
                self.logger.record_event(event_type="soul_fusion_aborted_no_sections", message="No sections found in soul_data to process. Aborting fusion.", level="warning")
                fusion_results["error"] = "No sections in soul data to fuse."
                return fusion_results
            for section_name, section_content in sections_to_process.items():
                self.logger.record_event(event_type="processing_section_for_expansion", message=f"Beginning expansion for section: '{section_name}'")
                expansion_depth = training_config.get("expansion_depth_per_facet", default_expansion_depth)
                repeat_count = default_repeat_counts.get(section_name, 1)
                context_buffer_size = context_buffer_sizes.get(section_name, 1)
                # Handle dict fields
                if isinstance(section_content, dict):
                    for field_name, field_value in section_content.items():
                        if not field_value or not isinstance(field_value, str) or not field_value.strip():
                            self.logger.record_event(event_type="skip_empty_facet_field_for_expansion", message=f"Skipping empty or invalid field '{field_name}' in section '{section_name}' for expansion.", level="debug", additional_info={"section": section_name, "field": field_name})
                            continue
                        current_facet_id = f"{section_name}.{field_name}"
                        current_expansion_context = list(overall_expansion_context_summaries)
                        # For repeated/narrative fields, use context buffer
                        context_buffer = []
                        elaborations = []
                        for i in range(repeat_count if section_name in default_repeat_counts else 1):
                            prompt = self.interpreter._create_initial_expansive_prompt(
                                current_facet_id, str(field_value), current_expansion_context, context_buffer
                            )
                            expansion = await self.interpreter.expand_soul_facet_recursively(
                                facet_name=current_facet_id,
                                facet_content=str(field_value),
                                target_ai_interface=target_ai_interface,
                                expansion_context=current_expansion_context,
                                max_depth=expansion_depth,
                                context_buffer=context_buffer
                            )
                            elaborations.extend(expansion)
                            # Update context buffer for narrative fields
                            if section_name in context_buffer_sizes:
                                context_buffer.append(elaborations[-1]["ai_response"] if elaborations else "")
                                if len(context_buffer) > context_buffer_size:
                                    context_buffer.pop(0)
                        fusion_results["expanded_profile"][current_facet_id] = elaborations
                        if elaborations and elaborations[-1].get("ai_response"):
                            last_response = elaborations[-1]["ai_response"]
                            summary_for_context = f"{current_facet_id}: {last_response[:150]}..."
                            overall_expansion_context_summaries.append(summary_for_context)
                            if len(overall_expansion_context_summaries) > max_overall_context_items:
                                overall_expansion_context_summaries.pop(0)
                # Handle list fields (repeated/narrative)
                elif isinstance(section_content, list):
                    expansion_depth_repeated = training_config.get("expansion_depth_repeated_entry", default_expansion_depth_repeated)
                    for index, entry_item in enumerate(section_content):
                        if isinstance(entry_item, dict):
                            for sub_field_name, sub_field_value in entry_item.items():
                                if not sub_field_value or not isinstance(sub_field_value, str) or not sub_field_value.strip():
                                    self.logger.record_event(event_type="skip_empty_list_item_subfield_for_expansion", message=f"Skipping empty sub-field '{sub_field_name}' in list item {index} of section '{section_name}'.", level="debug")
                                    continue
                                current_facet_id = f"{section_name}.{sub_field_name}[{index}]"
                                current_expansion_context = list(overall_expansion_context_summaries)
                                context_buffer = []
                                elaborations = []
                                for i in range(repeat_count if section_name in default_repeat_counts else 1):
                                    prompt = self.interpreter._create_initial_expansive_prompt(
                                        current_facet_id, str(sub_field_value), current_expansion_context, context_buffer
                                    )
                                    expansion = await self.interpreter.expand_soul_facet_recursively(
                                        facet_name=current_facet_id,
                                        facet_content=str(sub_field_value),
                                        target_ai_interface=target_ai_interface,
                                        expansion_context=current_expansion_context,
                                        max_depth=expansion_depth_repeated,
                                        context_buffer=context_buffer
                                    )
                                    elaborations.extend(expansion)
                                    if section_name in context_buffer_sizes:
                                        context_buffer.append(elaborations[-1]["ai_response"] if elaborations else "")
                                        if len(context_buffer) > context_buffer_size:
                                            context_buffer.pop(0)
                                fusion_results["expanded_profile"][current_facet_id] = elaborations
                                if elaborations and elaborations[-1].get("ai_response"):
                                    last_response = elaborations[-1]["ai_response"]
                                    summary_for_context = f"{current_facet_id}: {last_response[:150]}..."
                                    overall_expansion_context_summaries.append(summary_for_context)
                                    if len(overall_expansion_context_summaries) > max_overall_context_items:
                                        overall_expansion_context_summaries.pop(0)
                        elif isinstance(entry_item, str):
                            if not entry_item.strip():
                                self.logger.record_event(event_type="skip_empty_list_item_string_for_expansion", message=f"Skipping empty string list item at index {index} in section '{section_name}'.", level="debug")
                                continue
                            current_facet_id = f"{section_name}[{index}]"
                            current_expansion_context = list(overall_expansion_context_summaries)
                            context_buffer = []
                            elaborations = []
                            for i in range(repeat_count if section_name in default_repeat_counts else 1):
                                prompt = self.interpreter._create_initial_expansive_prompt(
                                    current_facet_id, entry_item, current_expansion_context, context_buffer
                                )
                                expansion = await self.interpreter.expand_soul_facet_recursively(
                                    facet_name=current_facet_id,
                                    facet_content=entry_item,
                                    target_ai_interface=target_ai_interface,
                                    expansion_context=current_expansion_context,
                                    max_depth=expansion_depth_repeated,
                                    context_buffer=context_buffer
                                )
                                elaborations.extend(expansion)
                                if section_name in context_buffer_sizes:
                                    context_buffer.append(elaborations[-1]["ai_response"] if elaborations else "")
                                    if len(context_buffer) > context_buffer_size:
                                        context_buffer.pop(0)
                            fusion_results["expanded_profile"][current_facet_id] = elaborations
                            if elaborations and elaborations[-1].get("ai_response"):
                                last_response = elaborations[-1]["ai_response"]
                                summary_for_context = f"{current_facet_id}: {last_response[:150]}..."
                                overall_expansion_context_summaries.append(summary_for_context)
                                if len(overall_expansion_context_summaries) > max_overall_context_items:
                                    overall_expansion_context_summaries.pop(0)
            # After all facets are expanded, derive training data and memory entries
            derived_training_data, memory_integrations = self._derive_training_and_memories_from_expansion(
                fusion_results["expanded_profile"], training_config
            )
            # Advanced post-processing hooks (summarization, chunking, deduplication)
            if training_config.get("postprocess_summarize", False):
                derived_training_data = self._summarize_training_data(derived_training_data, training_config)
            if training_config.get("postprocess_chunk", False):
                derived_training_data = self._chunk_training_data(derived_training_data, training_config)
            if training_config.get("postprocess_deduplicate", False):
                derived_training_data = self._deduplicate_training_data(derived_training_data)
            fusion_results["derived_training_data"] = derived_training_data
            fusion_results["memory_integrations"] = memory_integrations
            fusion_results["lora_parameters"] = self._generate_lora_params_from_expansion(
                fusion_results["expanded_profile"], training_config
            )
            fusion_results["fusion_metrics"] = self._calculate_fusion_metrics_from_expansion(
                fusion_results
            )
            self.logger.record_event(event_type="soul_fusion_process_completed", message="Soul fusion process with recursive expansion completed successfully.")
            return fusion_results
        except Exception as e_fuse_soul_main:
            self.error_handler.handle_execution_error(
                e_fuse_soul_main,
                {"operation": "fuse_soul_recursive_expansion_main_loop"},
                "soul_fusion_critical_error"
            )
            error_message = f"Critical error during soul fusion: {str(e_fuse_soul_main)}"
            fusion_results["error"] = error_message
            self.logger.record_event(event_type="soul_fusion_process_failed", message=error_message, level="critical")
            return fusion_results

    def _derive_training_and_memories_from_expansion(
        self, expanded_profile: OrderedDict, config: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Processes the `expanded_profile` to create structured training data and memory entries.
        """
        all_training_samples: List[Dict[str, Any]] = [] # Renamed
        all_memory_store_entries: List[Dict[str, Any]] = [] # Renamed

        if not expanded_profile:
            self.logger.record_event(event_type="derive_data_no_profile", message="Expanded profile is empty, no training data or memories derived.", level="info")
            return all_training_samples, all_memory_store_entries

        for facet_id, elaborations_list in expanded_profile.items():
            if not isinstance(elaborations_list, list) or not elaborations_list:
                self.logger.record_event(event_type="derive_data_empty_elaboration_list", message=f"No elaborations found for facet '{facet_id}'. Skipping.", level="debug", additional_info={"facet_id": facet_id})
                continue

            # 1. Create Persona-Defining Narratives (long-form text for RAG or fine-tuning)
            # This combines all exchanges for a facet into a coherent narrative.
            narrative_parts_for_facet = [f"Deep Dive into {facet_id}:\n"] # Renamed
            for i, exchange_dict in enumerate(elaborations_list): # Renamed
                prompt_text = exchange_dict.get('prompt_to_ai', '(System prompt not fully recorded)')
                response_text = exchange_dict.get('ai_response', '(AI response not fully recorded)')
                # Frame as an internal monologue or a Socratic dialogue summary
                narrative_parts_for_facet.append(f"Initial prompt/angle was: \"{prompt_text}\"\nMy detailed reflection on this was: \"{response_text}\"\n---\n")
            
            full_facet_narrative = "".join(narrative_parts_for_facet).strip() # Renamed
            
            # Training data: input could be a question about the facet, output is the narrative
            all_training_samples.append({
                "input": f"Tell me in detail about your understanding and personal integration of the characteristic: {facet_id}.",
                "output": full_facet_narrative,
                "source_facet": facet_id,
                "type": "facet_full_narrative_elaboration", # More descriptive type
                "weight": config.get("training_narrative_weight", 1.0) 
            })
            
            # Memory entry: the full narrative itself can be a significant memory
            all_memory_store_entries.append({
                "content": full_facet_narrative,
                "type": "soul_expansion_facet_narrative", # More descriptive
                "facet_id": facet_id,
                "timestamp": time.time(),
                "source": "SoulFuser_RecursiveExpansion", # More specific source
                "metadata": {"elaboration_depth": len(elaborations_list)}
            })

            # 2. Create Contextualized Q&A from individual exchanges (good for instruction fine-tuning)
            for exchange_idx, exchange_dict_qa in enumerate(elaborations_list): # Renamed
                prompt_text_qa = exchange_dict_qa.get('prompt_to_ai') # Renamed
                response_text_qa = exchange_dict_qa.get('ai_response') # Renamed
                if prompt_text_qa and response_text_qa and response_text_qa != "(No distinct elaboration was provided for this aspect.)" and not response_text_qa.startswith("(An error occurred"):
                    all_training_samples.append({
                        "input": prompt_text_qa, # The prompt given to the AI
                        "output": response_text_qa, # The AI's response to that specific prompt
                        "source_facet": facet_id,
                        "type": "facet_interactive_dialogue_turn", # More descriptive
                        "turn_index_in_facet_expansion": exchange_idx, # Renamed
                        "weight": config.get("training_dialogue_turn_weight", 0.9) # Slightly less weight than full narrative?
                    })
        
        self.logger.record_event(
            event_type="training_data_and_memories_derived", # Renamed
            message=f"Derived {len(all_training_samples)} training samples and {len(all_memory_store_entries)} memory entries from expanded soul profile.",
            additional_info={"training_samples_count": len(all_training_samples), "memory_entries_count": len(all_memory_store_entries)}
        )
        return all_training_samples, all_memory_store_entries

    def _generate_lora_params_from_expansion(
        self, expanded_profile: OrderedDict, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates suggested LoRA adapter parameters based on the "richness" of the expanded profile.
        This is heuristic and aims to scale LoRA capacity with the amount of new "personality" data.
        """
        num_expanded_facets = len(expanded_profile)
        total_elaboration_characters = 0 # Renamed
        total_elaboration_exchanges_count = 0 # Renamed

        if num_expanded_facets > 0:
            for facet_id, elaborations_list_lora in expanded_profile.items(): # Renamed
                if isinstance(elaborations_list_lora, list) and elaborations_list_lora: 
                    total_elaboration_exchanges_count += len(elaborations_list_lora)
                    for exchange_lora in elaborations_list_lora: # Renamed
                        if exchange_lora and isinstance(exchange_lora.get("ai_response"), str): 
                           total_elaboration_characters += len(exchange_lora["ai_response"])
        
        avg_elaboration_length_per_exchange = safe_divide(total_elaboration_characters, total_elaboration_exchanges_count, default=config.get("lora_default_avg_len", 100.0))
        
        # Normalization factors (these are configurable defaults, adjust based on typical expansion results)
        norm_facets_target = float(config.get("lora_norm_facets_target", 20.0)) # Expected number of "significant" facets for full LoRA capacity
        norm_chars_per_exchange_target = float(config.get("lora_norm_chars_per_exchange_target", 300.0)) # Expected avg length of a "good" elaboration

        facet_count_factor = min(1.0, safe_divide(float(num_expanded_facets), norm_facets_target)) 
        elaboration_length_factor = min(1.0, safe_divide(avg_elaboration_length_per_exchange, norm_chars_per_exchange_target))
        
        # Overall "richness" proxy: a weighted average or simple average of these factors
        # More sophisticated weighting could be added if some factors are more indicative of "good" LoRA data
        richness_proxy_score = (facet_count_factor + elaboration_length_factor) / 2.0 # Renamed
        self.logger.record_event("lora_richness_calculation", message=f"LoRA richness proxy: {richness_proxy_score:.3f}", additional_info={"facets": num_expanded_facets, "avg_len": avg_elaboration_length_per_exchange, "facet_factor": facet_count_factor, "len_factor": elaboration_length_factor})

        # Base and Max LoRA parameters from config
        base_r = int(config.get("lora_base_r", 16))
        base_alpha_multiplier = float(config.get("lora_base_alpha_multiplier", 2.0)) # alpha = multiplier * r
        
        max_r = int(config.get("lora_max_r", 128))
        
        # Scale r based on richness_proxy_score. Ensure it's an int and within min/max bounds.
        # Linear scaling: r = base_r + (max_r - base_r) * richness_proxy_score
        # More conservative scaling: r = base_r * (1 + richness_proxy_score * scale_factor)
        lora_r_calculated = int(base_r + (max_r - base_r) * richness_proxy_score)
        lora_r_final = max(min(base_r // 2, 8), min(lora_r_calculated, max_r)) # Ensure r is at least 8 (or half base_r) and not over max_r
        
        lora_alpha_final = int(lora_r_final * base_alpha_multiplier) # Alpha typically 2*r or similar
        
        # Dropout can be inversely related to richness (more dropout if less confident/rich expansion)
        # Or a fixed value from config.
        lora_dropout_final = float(config.get("lora_dropout", 0.05)) 
        # Example of dynamic dropout:
        # lora_dropout_final = min(0.5, 0.05 + (1.0 - richness_proxy_score) * float(config.get("lora_dropout_scale_vs_richness", 0.1)))


        lora_params = {
            "r": lora_r_final,
            "lora_alpha": lora_alpha_final, 
            "lora_dropout": lora_dropout_final,
            "bias": config.get("lora_bias", "none"), # e.g., "none", "all", "lora_only"
            "task_type": config.get("lora_task_type", "CAUSAL_LM"), # e.g., "SEQ_2_SEQ_LM" for T5
            "target_modules": config.get("lora_target_modules", ["q_proj", "v_proj"]) # Model-specific
        }
        self.logger.record_event("lora_params_generated", message="LoRA parameters generated based on expansion profile.", additional_info=lora_params)
        return lora_params

    def _calculate_fusion_metrics_from_expansion(self, fusion_results_dict: Dict[str, Any]) -> Dict[str, Any]: # Renamed param
        """Calculates various metrics about the fusion process based on the expanded profile."""
        expanded_profile_map = fusion_results_dict.get("expanded_profile", {}) # Renamed
        
        num_facets_fully_processed = 0 # Renamed
        total_elaboration_exchanges_generated = 0 # Renamed
        total_elaboration_characters_generated = 0 # Renamed
        sum_of_achieved_depths_per_facet = 0 # Renamed

        for facet_id, elaborations_list_metrics in expanded_profile_map.items(): # Renamed
            if isinstance(elaborations_list_metrics, list) and elaborations_list_metrics: 
                num_facets_fully_processed +=1
                achieved_depth_for_facet = len(elaborations_list_metrics) # Renamed
                total_elaboration_exchanges_generated += achieved_depth_for_facet
                sum_of_achieved_depths_per_facet += achieved_depth_for_facet
                for exchange_metrics in elaborations_list_metrics: # Renamed
                     if exchange_metrics and isinstance(exchange_metrics.get("ai_response"), str): 
                        total_elaboration_characters_generated += len(exchange_metrics["ai_response"])
        
        avg_expansion_depth_achieved = safe_divide(float(sum_of_achieved_depths_per_facet), float(num_facets_fully_processed)) # Renamed
        avg_chars_per_elaboration_unit = safe_divide(float(total_elaboration_characters_generated), float(total_elaboration_exchanges_generated)) # Renamed

        metrics = {
            "facets_attempted_for_expansion": len(expanded_profile_map), # All facets for which expansion was tried
            "facets_with_successful_expansion": num_facets_fully_processed, # Facets that yielded at least one elaboration
            "total_elaboration_exchanges_generated": total_elaboration_exchanges_generated,
            "total_elaboration_characters_generated": total_elaboration_characters_generated,
            "average_expansion_depth_achieved_per_facet": round(avg_expansion_depth_achieved, 2),
            "average_chars_per_elaboration_exchange": round(avg_chars_per_elaboration_unit, 2),
            "derived_training_samples_count": len(fusion_results_dict.get("derived_training_data", [])),
            "integrated_memory_entries_count": len(fusion_results_dict.get("memory_integrations", [])),
        }
        self.logger.record_event("fusion_metrics_calculated", message="Fusion metrics calculated from expansion.", additional_info=metrics)
        return metrics

    # --- Advanced post-processing hooks ---
    def _summarize_training_data(self, training_data, config):
        # Placeholder: implement summarization logic (e.g., using LLM or extractive summarizer)
        return training_data
    def _chunk_training_data(self, training_data, config):
        # Placeholder: implement chunking logic (e.g., split long outputs for LoRA or RAG)
        return training_data
    def _deduplicate_training_data(self, training_data):
        # Placeholder: implement deduplication logic
        seen = set()
        deduped = []
        for item in training_data:
            key = (item.get('input'), item.get('output'))
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

async def parse_and_fuse_soul_file(
    file_path: str,
    logger: Logger,
    error_handler: ErrorManager,
    target_ai_interface: Any, # This object must have an async method `get_interpretation(prompt: str) -> str`
    target_model_tokenizer: PreTrainedTokenizer, # For LoRA or other model-specific tasks
    training_config: Dict[str, Any], # Controls expansion depth, LoRA params, context sizes etc.
    event_dispatcher: Optional[EventDispatcher] = None,
    cache_path: Optional[str] = None, # For caching the parsed .soul file
    strict_mode: bool = True # For parsing and validation strictness
) -> Dict[str, Any]:
    """
    Top-level function to parse a .soul file and then fuse it with the target AI system 
    using the recursive expansion method.
    """
    logger.record_event(
        event_type="parse_and_fuse_process_start", 
        message=f"Starting parse_and_fuse_soul_file for: {file_path}",
        additional_info={"file_path": file_path, "strict_mode": strict_mode}
        )
    
    # 1. Parse the .soul file
    soul_data_parsed = None # Initialize
    try:
        soul_data_parsed = parse_soul_file( # Renamed var
            file_path, logger, error_handler, event_dispatcher,
            cache_path, strict_mode
        )
    except ValueError as val_err_parse: # Catch validation errors if strict_mode is True in parse_soul_file
        error_msg_parse = f"Parsing or validation of soul file failed in strict mode: {val_err_parse}"
        logger.record_event(event_type="parse_and_fuse_strict_parse_error", message=error_msg_parse, level="error", additional_info={"file_path": file_path})
        return {"error": error_msg_parse, "soul_data": None, "fusion_results": None}
    except Exception as e_parse: # Catch any other unexpected parsing errors
        error_msg_parse_generic = f"Unexpected error during soul file parsing: {e_parse}"
        logger.record_event(event_type="parse_and_fuse_unexpected_parse_error", message=error_msg_parse_generic, level="critical", additional_info={"file_path": file_path, "error_details": str(e_parse)})
        error_handler.handle_execution_error(e_parse, {"file_path": file_path, "operation": "parse_soul_file_top_level"}, "parse_and_fuse_main_parse_exception")
        return {"error": error_msg_parse_generic, "soul_data": None, "fusion_results": None}


    if not soul_data_parsed or not soul_data_parsed.get("sections"):
        error_msg_no_data = "Failed to parse soul file or no sections found in the parsed data."
        logger.record_event(event_type="parse_and_fuse_aborted_no_soul_data", message=error_msg_no_data, level="error", additional_info={"file_path": file_path})
        return {"error": error_msg_no_data, "soul_data": soul_data_parsed, "fusion_results": None} # Return parsed data if any, even if no sections
        
    # 2. Create SoulInterpreter and SoulFuser instances
    # These now use the new expansion-focused logic
    soul_interpreter_instance = SoulInterpreter(logger, error_handler) # Renamed
    soul_fuser_instance = SoulFuser(logger, error_handler, soul_interpreter_instance, target_model_tokenizer) # Renamed
    
    # 3. Fuse the parsed soul data with the target AI system using recursive expansion
    fusion_results_map = None # Initialize
    try:
        fusion_results_map = await soul_fuser_instance.fuse_soul( # Renamed var
            soul_data_parsed,
            target_ai_interface,
            training_config # Pass the entire config dict
        )
    except Exception as e_fuse: # Catch unexpected errors during the fusion process itself
        error_msg_fuse_generic = f"Unexpected critical error during soul fusion process: {e_fuse}"
        logger.record_event(event_type="parse_and_fuse_unexpected_fusion_error", message=error_msg_fuse_generic, level="critical", additional_info={"file_path": file_path, "error_details": str(e_fuse)})
        error_handler.handle_execution_error(e_fuse, {"file_path": file_path, "operation": "fuse_soul_top_level"}, "parse_and_fuse_main_fusion_exception")
        # Return what we have, plus the error
        return {"error": error_msg_fuse_generic, "soul_data": soul_data_parsed, "fusion_results": fusion_results_map or {"error": error_msg_fuse_generic}}


    if fusion_results_map and fusion_results_map.get("error"): # Check if fuse_soul itself reported a handled error
         logger.record_event(
             event_type="parse_and_fuse_fusion_reported_error", 
             message=f"Soul fusion process completed with a reported error: {fusion_results_map['error']}", 
             level="error",
             additional_info={"file_path": file_path}
            )
    else:
        logger.record_event(
            event_type="parse_and_fuse_process_completed", 
            message=f"Parse and fuse process for {file_path} completed.",
            additional_info={"file_path": file_path, "facets_expanded_count": len(fusion_results_map.get("expanded_profile", {}))}
            )
            
    return {
        "soul_data": soul_data_parsed,
        "fusion_results": fusion_results_map
    }
