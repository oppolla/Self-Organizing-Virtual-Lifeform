from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
import re
import json
import os
from typing import Dict, Any, Optional
from sovl_logger import Logger
from sovl_error import ErrorHandler
from sovl_events import EventDispatcher
from sovl_processor import SoulLogitsProcessor
import traceback
from collections import OrderedDict

class SoulParser(NodeVisitor):
    """Parse a .soul file into a structured dictionary with robust handling and strict compliance to the Soulprint spec."""
    
    def __init__(self, logger: Logger, error_handler: ErrorHandler, event_dispatcher: Optional[EventDispatcher] = None):
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
                    for line in child:
                        if isinstance(line, str) and line.startswith("  "):
                            lines.append(line[2:].rstrip("\n"))
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
                if "Description" in voice_data:
                    keywords = voice_data["Description"].split(",")
                    for keyword in keywords:
                        self.keywords[keyword.strip()] = 0.8  # High weight for voice characteristics
                
                if "Summary" in voice_data:
                    keywords = voice_data["Summary"].split()
                    for keyword in keywords:
                        self.keywords[keyword.strip()] = 0.7  # Medium weight for summary words

            # Extract from Heartbeat section
            if "Heartbeat" in self.data["sections"]:
                heartbeat_data = self.data["sections"]["Heartbeat"]
                if "Tendencies" in heartbeat_data:
                    tendencies = heartbeat_data["Tendencies"].split(",")
                    for tendency in tendencies:
                        self.keywords[tendency.strip()] = 0.9  # Very high weight for tendencies

            # Extract from Echoes section
            if "Echoes" in self.data["sections"]:
                echoes_data = self.data["sections"]["Echoes"]
                if "Memory" in echoes_data:
                    for memory in echoes_data["Memory"]:
                        if isinstance(memory, dict) and "Scene" in memory:
                            words = memory["Scene"].split()
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
            from sovl_processor import SoulLogitsProcessor
            
            # Extract keywords if not already done
            if not self.keywords:
                self.extract_keywords()
            
            if not self.keywords:
                self.logger.record_event(
                    event_type="processor_creation",
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
        Checks required fields, regex constraints, repeat counts, and sets defaults for optional fields.
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
        for key, regex in required_metadata.items():
            value = metadata.get(key)
            if value is None:
                self.error_handler.handle_data_error(
                    ValueError(f"Missing required metadata: {key}"),
                    {"field": key},
                    "soul_semantic_validation"
                )
                valid = False
                if strict_mode:
                    return False
            elif not re.match(regex, value):
                self.error_handler.handle_data_error(
                    ValueError(f"Invalid value for {key}: {value}"),
                    {"field": key, "value": value},
                    "soul_semantic_validation"
                )
                valid = False
                if strict_mode:
                    return False
        # Set defaults for optional metadata
        if "PrivacyLevel" not in metadata:
            metadata["PrivacyLevel"] = "private"
            self.logger.record_event(
                event_type="default_applied",
                message="PrivacyLevel set to private",
                level="info"
            )
        # Required sections and repeat counts
        required_sections = {
            "Identity": ["Name", "Origin", "Essence"],
            "Chronicle": [],
            "Echoes": [],
            "Tides": [],
            "Threads": [],
            "Horizon": [],
            "Reflection": [],
            "Voice": [],
            "Heartbeat": []
        }
        for section, required_fields in required_sections.items():
            if section not in sections:
                self.error_handler.handle_data_error(
                    ValueError(f"Missing required section: {section}"),
                    {"section": section},
                    "soul_semantic_validation"
                )
                valid = False
                if strict_mode:
                    return False
            else:
                for field in required_fields:
                    value = sections[section].get(field)
                    if value is None or value == "":
                        self.error_handler.handle_data_error(
                            ValueError(f"Missing required field in {section}: {field}"),
                            {"section": section, "field": field},
                            "soul_semantic_validation"
                        )
                        valid = False
                        if strict_mode:
                            return False
        # Example: repeat count check for Chronicle (should be 142 entries)
        if "Chronicle" in sections:
            chronicle = sections["Chronicle"]
            if isinstance(chronicle, dict):
                count = len(chronicle)
            elif isinstance(chronicle, list):
                count = len(chronicle)
            else:
                count = 0
            if count < 142:
                self.error_handler.handle_data_error(
                    ValueError(f"Chronicle has too few entries: {count}"),
                    {"section": "Chronicle", "count": count},
                    "soul_semantic_validation"
                )
                valid = False
        return valid

    def redact_sensitive_terms(self, denylist=None, log_redactions=True):
        """
        Scan narrative fields for sensitive terms and redact them, logging all changes.
        Denylist can be customized; defaults to ["user", "IP", "password"].
        Redactions are logged in self.data["redactions"] as a list of dicts.
        """
        if denylist is None:
            denylist = ["user", "IP", "password"]
        # Compile denylist regexes once for efficiency
        denylist_patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in denylist]
        redactions = []
        def redact_in_obj(obj, section=None):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, list, dict)):
                        obj[k] = redact_in_obj(v, section=section)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    obj[i] = redact_in_obj(item, section=section)
            elif isinstance(obj, str):
                redacted = obj
                for pattern, term in zip(denylist_patterns, denylist):
                    if pattern.search(redacted):
                        redacted = pattern.sub("[REDACTED]", redacted)
                        if log_redactions:
                            redactions.append({
                                "section": section,
                                "term": term,
                                "original": obj if log_redactions else None,
                                "redacted": redacted if log_redactions else None
                            })
                return redacted
            return obj
        for section, fields in self.data.get("sections", {}).items():
            self.data["sections"][section] = redact_in_obj(fields, section=section)
        if redactions:
            self.data["redactions"] = redactions
            if log_redactions:
                for entry in redactions:
                    self.logger.record_event(
                        event_type="redaction",
                        message=f"Redacted term '{entry['term']}' in section {entry['section']}",
                        level="info",
                        additional_info={k: v for k, v in entry.items() if k != 'original' and k != 'redacted'}
                    )
        return len(redactions)

    def validate_hash_and_privacy(self, file_path: str) -> bool:
        """
        Validate SHA-256 hash and privacy level enforcement as per Soulprint spec.
        Returns True if valid, False otherwise. Logs all errors.
        """
        valid = True
        metadata = self.data.get("metadata", {})
        # Hash validation
        expected_hash = metadata.get("Hash")
        if expected_hash:
            try:
                with open(file_path, "rb") as f:
                    content = f.read()
                # Remove the Hash field line from content for hash calculation
                content_str = content.decode("utf-8")
                content_wo_hash = re.sub(r"^Hash: [0-9a-f]{64}\s*$", "", content_str, flags=re.MULTILINE)
                actual_hash = __import__('hashlib').sha256(content_wo_hash.encode("utf-8")).hexdigest()
                if actual_hash != expected_hash:
                    self.error_handler.handle_data_error(
                        ValueError(f"Hash mismatch: expected {expected_hash}, got {actual_hash}"),
                        {"expected": expected_hash, "actual": actual_hash},
                        "soul_hash_validation"
                    )
                    valid = False
            except Exception as e:
                self.error_handler.handle_data_error(
                    e,
                    {"operation": "hash_validation", "file_path": file_path},
                    "soul_hash_validation"
                )
                valid = False
        # Privacy level enforcement
        privacy = metadata.get("PrivacyLevel", "private")
        if privacy not in ("public", "restricted", "private"):
            self.error_handler.handle_data_error(
                ValueError(f"Invalid PrivacyLevel: {privacy}"),
                {"privacy": privacy},
                "soul_privacy_validation"
            )
            valid = False
        # Consent expiry check
        consent_expiry = metadata.get("ConsentExpiry")
        if consent_expiry:
            from datetime import datetime
            try:
                expiry = datetime.strptime(consent_expiry, "%Y-%m-%dT%H:%M:%SZ")
                now = datetime.utcnow()
                if expiry < now:
                    self.error_handler.handle_data_error(
                        ValueError(f"Consent expired at {consent_expiry}"),
                        {"consent_expiry": consent_expiry},
                        "soul_consent_expiry"
                    )
                    valid = False
            except Exception as e:
                self.error_handler.handle_data_error(
                    e,
                    {"consent_expiry": consent_expiry},
                    "soul_consent_expiry"
                )
                valid = False
        return valid

    def load_from_string(self, soul_string: str) -> dict:
        """
        Parse a .soul file from a string (for unit testing or in-memory validation).
        """
        try:
            tree = self.grammar.parse(soul_string)
            self.visit(tree)
            return self.data
        except Exception as e:
            self.error_handler.handle_data_error(
                e,
                {"operation": "soul_string_parse"},
                "soul_file_parsing"
            )
            return {}

def parse_soul_file(
    file_path: str,
    logger: Logger,
    error_handler: ErrorHandler,
    event_dispatcher: Optional[EventDispatcher] = None,
    cache_path: Optional[str] = None,
    strict_mode: bool = True
) -> Dict[str, Any]:
    """Parse a .soul file, validate its contents, and optionally cache results."""
    # Check cache
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        logger.record_event(
            event_type="soul_cache_loaded",
            message="Loaded .soul data from cache",
            level="info",
            additional_info={"cache_path": cache_path}
        )
        return cached_data

    parser = SoulParser(logger, error_handler, event_dispatcher)
    parsed_data = parser.parse(file_path)

    # Comprehensive validation
    validation_rules = {
        "metadata": {
            "Consent": lambda x: x == "true",
            "Version": lambda x: re.match(r"^\d+\.\d+\.\d+$", x),
            "Creator": lambda x: isinstance(x, str) and len(x) <= 100,
            "Created": lambda x: re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", x) if x else True
        },
        "sections": {
            "Identity": {
                "Name": lambda x: re.match(r"^[A-Za-z0-9_-]{1,50}$", x),
                "Essence": lambda x: isinstance(x, str) and len(x) <= 200
            },
            "Voice": {
                "Summary": lambda x: isinstance(x, str) and len(x) <= 100,
                "Metadata": lambda x: re.match(r"^\w+:\s*[\d.]+$", x) if x else True
            },
            "Echoes": {
                "Memory": lambda x: len(x) == 142 and all(
                    isinstance(m, str) and float(d["Resonance"]) >= 0.1 and float(d["Resonance"]) <= 1.0
                    for m, d in zip(x, parsed_data["sections"]["Echoes"].get("Memory", []))
                )
            },
            "Reflection": {
                "Purpose": lambda x: isinstance(x, str) and len(x) <= 200
            }
        }
    }

    for category, rules in validation_rules.items():
        for key, rule in rules.items():
            if category == "metadata":
                value = parsed_data["metadata"].get(key)
                if value is None and key in ["Consent", "Version", "Creator"]:
                    error_handler.handle_data_error(
                        ValueError(f"Missing required metadata: {key}"),
                        {"file_path": file_path, "key": key},
                        "soul_validation"
                    )
                    if strict_mode:
                        raise ValueError(f"Missing required metadata: {key}")
                elif value and not rule(value):
                    error_handler.handle_data_error(
                        ValueError(f"Invalid {key}: {value}"),
                        {"file_path": file_path, "key": key, "value": value},
                        "soul_validation"
                    )
                    if strict_mode:
                        raise ValueError(f"Invalid {key}: {value}")
            elif category == "sections":
                if key not in parsed_data["sections"] and key in ["Identity", "Voice", "Reflection"]:
                    error_handler.handle_data_error(
                        ValueError(f"Missing required section: {key}"),
                        {"file_path": file_path, "section": key},
                        "soul_validation"
                    )
                    if strict_mode:
                        raise ValueError(f"Missing required section: {key}")
                else:
                    for subkey, subrule in rules[key].items():
                        value = parsed_data["sections"][key].get(subkey)
                        if value is None and subkey in ["Name", "Summary", "Purpose"]:
                            error_handler.handle_data_error(
                                ValueError(f"Missing required field in {key}: {subkey}"),
                                {"file_path": file_path, "section": key, "field": subkey},
                                "soul_validation"
                            )
                            if strict_mode:
                                raise ValueError(f"Missing required field in {key}: {subkey}")
                        elif value and not subrule(value):
                            error_handler.handle_data_error(
                                ValueError(f"Invalid {subkey} in {key}: {value}"),
                                {"file_path": file_path, "section": key, "field": subkey, "value": value},
                                "soul_validation"
                            )
                            if strict_mode:
                                raise ValueError(f"Invalid {subkey} in {key}: {value}")

    # Set defaults for optional fields
    parsed_data["metadata"].setdefault("PrivacyLevel", "private")
    parsed_data["sections"].setdefault("X-Custom", {})

    # Cache parsed data
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, indent=2)
        logger.record_event(
            event_type="soul_cache_saved",
            message="Saved .soul data to cache",
            level="info",
            additional_info={"cache_path": cache_path}
        )

    # Dispatch event
    if event_dispatcher:
        event_dispatcher.dispatch(
            event_type="soul_parsed",
            event_data={"file_path": file_path, "parsed_data": parsed_data}
        )

    logger.record_event(
        event_type="soul_parsed",
        message="Successfully parsed and validated .soul file",
        level="info",
        additional_info={"file_path": file_path, "sections": list(parsed_data["sections"].keys())}
    )
    return parsed_data
