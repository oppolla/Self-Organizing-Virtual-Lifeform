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

class SoulParser(NodeVisitor):
    """Parse a .soul file into a structured dictionary with robust handling."""
    
    def __init__(self, logger: Logger, error_handler: ErrorHandler, event_dispatcher: Optional[EventDispatcher] = None):
        self.logger = logger
        self.error_handler = error_handler
        self.event_dispatcher = event_dispatcher
        self.data = {"metadata": {}, "sections": {}, "unparsed": {}}
        self.current_section = None
        self.line_number = 0
        self.keywords = {}  # Store keywords and their weights

    def visit_section(self, node, visited_children):
        self.current_section = node.text.strip("[]")
        self.data["sections"][self.current_section] = {}
        self.logger.record_event(
            event_type="soul_section_detected",
            message=f"Detected section: {self.current_section}",
            level="debug",
            additional_info={"line_number": self.line_number}
        )

    def visit_field(self, node, visited_children):
        self.line_number += 1
        try:
            key, value = node.text.split(":", 1)
            key = key.strip()
            value = value.strip()
            if self.current_section:
                self.data["sections"][self.current_section][key] = value
            else:
                self.data["metadata"][key] = value
        except ValueError:
            self.error_handler.handle_data_error(
                ValueError(f"Invalid field format at line {self.line_number}: {node.text}"),
                {"line": node.text, "line_number": self.line_number},
                "soul_field_parsing"
            )
            self.data["unparsed"][self.line_number] = node.text

    def visit_list_item(self, node, visited_children):
        self.line_number += 1
        match = re.match(r"-\s*(\w+):\s*(.+)", node.text.strip())
        if match and self.current_section:
            key, value = match.groups()
            if key not in self.data["sections"][self.current_section]:
                self.data["sections"][self.current_section][key] = []
            self.data["sections"][self.current_section][key].append(value)
        else:
            self.error_handler.handle_data_error(
                ValueError(f"Invalid list item at line {self.line_number}: {node.text}"),
                {"line": node.text, "line_number": self.line_number},
                "soul_list_parsing"
            )
            self.data["unparsed"][self.line_number] = node.text

    def visit_multiline(self, node, visited_children):
        self.line_number += 1
        lines = node.text.strip().split("\n")[1:]  # Skip "> |"
        value = "\n".join(line.strip() for line in lines)
        key = "Multiline"  # Default key; can be overridden by context
        if self.current_section:
            self.data["sections"][self.current_section][key] = value
        else:
            self.data["metadata"][key] = value

    def generic_visit(self, node, visited_children):
        if node.expr_name == "comment":
            self.line_number += 1
        return node

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

    grammar = Grammar(
        r"""
        soul_file = header metadata section*
        header = "%SOULPRINT\n%VERSION: v" version "\n"
        version = ~r"\d+\.\d+\.\d+"
        metadata = (field / multiline / comment)*
        section = section_header (field / list_item / multiline / comment)*
        section_header = "[" ~r"\w+" "]" "\n"
        field = ~r"^\s*\w+:\s*.+$" "\n"
        list_item = ~r"^\s*-\s*\w+:\s*.+$" "\n"
        multiline = ~r"^\s*> \|\n((?:.*?(?:\n|$))*)" "\n"
        comment = ~r"^\s*#.*$" "\n"
        """
    )

    try:
        # Stream file reading for large files
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        parser = SoulParser(logger, error_handler, event_dispatcher)
        tree = grammar.parse(text)
        parsed_data = parser.visit(tree)

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

    except Exception as e:
        error_handler.handle_data_error(
            e,
            {"file_path": file_path, "stack_trace": traceback.format_exc()},
            "soul_parsing"
        )
        logger.log_error(
            error_msg=str(e),
            error_type="soul_parsing_error",
            stack_trace=traceback.format_exc(),
            additional_info={"file_path": file_path}
        )
        raise
