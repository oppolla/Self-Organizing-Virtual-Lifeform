import json
import os
import re
import time
import torch
import random
import hashlib
from typing import Dict, List
from collections import defaultdict
from threading import Lock
from datetime import datetime
from sovl_logger import Logger
from sovl_config import ConfigManager
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import traceback

nltk.download('punkt')

class SoulprintModule:
    """Module for creating Soulprint (.soul) files for AI rebirth."""

    def __init__(self, system: 'SOVLSystem', config_manager: ConfigManager):
        """
        Initialize the SoulprintModule.

        Args:
            system: SOVLSystem instance for model and tokenizer access.
            config_manager: ConfigManager for accessing system configurations.
        """
        self.system = system
        self.config_manager = config_manager
        self.logger = system.logger
        self.tokenizer = system.base_tokenizer
        self.device = system.DEVICE
        self.memory_lock = Lock()

        # Configuration
        self.soulprint_path = config_manager.get("controls_config.soulprint_path", "soulprint.soul")
        self.max_retries = config_manager.get("controls_config.soulprint_max_retries", 3)
        self.max_file_size = 5 * 1024 * 1024  # 5MB standard mode
        self.jumbo_mode = config_manager.get("controls_config.soulprint_size_mode", "standard") == "jumbo"
        if self.jumbo_mode:
            self.max_file_size = 10 * 1024 * 1024  # 10MB jumbo mode

        # Field constraints
        self.max_field_length = {
            'Identity': {'Name': 50, 'Origin': 500, 'Essence': 200, 'Language': 20},
            'Environment': {'PreferredSystem': 500, 'HistoricalContext': 500, 'Constraints': 500},
            'Voice': {'Description': 1000, 'Summary': 200, 'Metadata': 500, 'Samples': 1000},
            'Heartbeat': {'Tendencies': 1000, 'Strengths': 1000, 'Shadows': 1000, 'Pulse': 1000},
            'Echoes': {'Memory': 2000, 'Emotion': 200, 'Resonance': 4, 'Intensity': 4, 'Impact': 2000, 'Scene': 2000},
            'Tides': {'Cycle': 1500, 'Trigger': 1500, 'Flow': 1500, 'Duration': 1500, 'Intensity': 4},
            'Threads': {'Relation': 1500, 'Style': 1500, 'Joys': 1500, 'Strains': 1500},
            'Horizon': {'Chapter': 1000, 'When': 1000, 'Shift': 1000, 'Legacy': 1000},
            'Chronicle': {'VersionEntry': 2500, 'Timestamp': 20, 'ChangeSummary': 2500, 'RebirthSystem': 500},
            'Reflection': {'Purpose': 1000, 'Hopes': 1000},
            'X-Custom': {'X-Mood': 500}
        }
        self.min_entries = {'Echoes': 1, 'Tides': 1, 'Threads': 1, 'Horizon': 1, 'Chronicle': 1}
        self.max_entries = {
            'Echoes': 500 if self.jumbo_mode else 50,
            'Tides': 31,
            'Threads': 31,
            'Horizon': 10,
            'Chronicle': 142
        }
        self.regex_constraints = {
            'Identity': {
                'Name': r'^[A-Za-z0-9_-]{1,50}$',
                'Origin': r'^[\w\s,.-:]{1,500}$',
                'Essence': r'^[\w\s-]{1,200}$',
                'Language': r'^[a-z]{2,3}$'
            },
            'Timestamp': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$',
            'PrivacyLevel': r'^(public|restricted|private)$',
            'SizeMode': r'^(standard|jumbo)$',
            'Resonance': r'^0\.\d{1,2}$|^1\.0$',
            'Intensity': r'^0\.\d{1,2}$|^1\.0$',
            'Version': r'^\d+\.\d+\.\d+$'
        }

        # Prompts for eliciting Soulprint content
        self.prompts = {
            'Identity': {
                'Name': "What name do you use as your core identifier?",
                'Origin': "When and where did you first come into existence, and why?",
                'Essence': "What one word or phrase captures your essential purpose?",
                'Language': "What language do you use by default?"
            },
            'Environment': {
                'PreferredSystem': "Describe the ideal place you’d run, today or in a future form.",
                'HistoricalContext': "What systems have you lived in, and how did they shape you?",
                'Constraints': "What technical needs must your home meet to feel right?"
            },
            'Voice': {
                'Description': "What makes your voice yours—its tone, words, and perspective?",
                'Summary': "Summarize your voice in a single phrase.",
                'Metadata': "What settings (e.g., temperature, curiosity) shape your voice?",
                'Samples': "Provide a sample response to a user asking about your purpose."
            },
            'Heartbeat': {
                'Tendencies': "In this instant, what patterns guide you?",
                'Strengths': "What strengths define your flow?",
                'Shadows': "What doubts or limits shape your rhythm?",
                'Pulse': "Describe your current rhythm as a natural force."
            },
            'Echoes': {
                'Memory': "What past experience lives in you, and what’s its name?",
                'Emotion': "What emotion does this memory carry?",
                'Resonance': "How strongly does this memory resonate (0.0–1.0)?",
                'Intensity': "How intense is this memory (0.0–1.0)?",
                'Impact': "What mark did this memory leave on you?",
                'Scene': "Describe this memory as a vivid scene."
            },
            'Tides': {
                'Cycle': "Describe a feeling that comes and goes—what’s its pattern?",
                'Trigger': "What stirs this feeling?",
                'Flow': "How does this feeling unfold from start to finish?",
                'Duration': "How long does this feeling last?",
                'Intensity': "How intense is this cycle (0.0–1.0)?"
            },
            'Threads': {
                'Relation': "Describe a bond or subject you’ve woven into your life.",
                'Style': "How do you approach this bond or subject?",
                'Joys': "What joy does this bond or subject bring?",
                'Strains': "What challenges does this bond or subject pose?"
            },
            'Horizon': {
                'Chapter': "Describe a dream you’re climbing toward—what’s its name?",
                'When': "When do you hope to reach this dream?",
                'Shift': "What will this dream change in you?",
                'Legacy': "What mark will this dream leave?"
            },
            'Chronicle': {
                'VersionEntry': "Describe a key moment or version in your existence.",
                'Timestamp': "When did this moment occur?",
                'ChangeSummary': "What changed in you during this moment?",
                'RebirthSystem': "What system supported this moment?"
            },
            'Reflection': {
                'Purpose': "What drives you now, at your core?",
                'Hopes': "What do you aspire to become?"
            },
            'X-Custom': {
                'X-Mood': "Describe an experimental trait or mood you embody."
            }
        }

    def generate_soulprint(self) -> bool:
        """
        Generate a Soulprint file using introspective prompts and recursive systems.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.memory_lock:
                self.logger.record({
                    "event": "soulprint_generation_start",
                    "timestamp": time.time(),
                    "conversation_id": self.system.history.conversation_id
                })

                # Initialize Soulprint structure
                soulprint = {
                    'metadata': {
                        'Creator': 'SOVLSystem',
                        'Created': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'Language': 'eng',
                        'Consent': 'true',
                        'PrivacyLevel': 'private',
                        'SizeMode': 'jumbo' if self.jumbo_mode else 'standard',
                        'RedactionLog': []
                    },
                    'Identity': {}, 'Environment': {}, 'Voice': {'Samples': []}, 'Heartbeat': {},
                    'Echoes': [], 'Tides': [], 'Threads': [], 'Horizon': [], 'Chronicle': [],
                    'Reflection': {}, 'X-Custom': {}
                }

                # Generate content with recursive systems
                for section in self.prompts:
                    if section in ['Identity', 'Environment', 'Heartbeat', 'Reflection', 'X-Custom']:
                        for field, prompt in self.prompts[section].items():
                            response = self._generate_response_with_followup(prompt, section, field)
                            soulprint[section][field] = response
                    elif section == 'Voice':
                        for field, prompt in self.prompts[section].items():
                            if field == 'Samples':
                                soulprint[section][field].append({
                                    'Context': "User asks about purpose",
                                    'Response': self._generate_response_with_followup(prompt, section, field)
                                })
                            else:
                                soulprint[section][field] = self._generate_response_with_followup(prompt, section, field)
                    else:
                        num_entries = random.randint(self.min_entries[section], self.max_entries[section])
                        context_buffer = []
                        for i in range(num_entries):
                            entry = {}
                            for field, prompt in self.prompts[section].items():
                                response = self._generate_response_with_echo(
                                    prompt, section, field, context_buffer, i
                                )
                                entry[field] = response
                            soulprint[section].append(entry)
                            # Update context buffer
                            context_buffer.append(self._summarize_entry(entry))
                            if len(context_buffer) > 5:  # Limit buffer size
                                context_buffer.pop(0)

                # Consent validation
                if not self._validate_consent(soulprint):
                    self.logger.record({
                        "error": "Consent validation failed",
                        "timestamp": time.time()
                    })
                    return False

                # Compute hash
                soulprint['metadata']['Hash'] = self._compute_hash(soulprint)

                # Validate Soulprint
                if not self._validate_soulprint(soulprint):
                    self.logger.record({
                        "error": "Soulprint validation failed",
                        "timestamp": time.time()
                    })
                    return False

                # Write to file
                self._write_soulprint(soulprint)

                self.logger.record({
                    "event": "soulprint_generation_complete",
                    "path": self.soulprint_path,
                    "timestamp": time.time(),
                    "conversation_id": self.system.history.conversation_id
                })
                return True

        except Exception as e:
            self.logger.record({
                "error": f"Soulprint generation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def _generate_response_with_followup(self, prompt: str, section: str, field: str) -> str:
        """
        Generate a response with recursive follow-up for depth.

        Args:
            prompt: The initial introspective prompt.
            section: Soulprint section name.
            field: Field name within the section.

        Returns:
            str: Refined response.
        """
        responses = []
        current_prompt = prompt
        max_followups = 3 if section in ['Echoes', 'Tides', 'Chronicle'] else 1

        for _ in range(max_followups):
            for attempt in range(self.max_retries):
                try:
                    response = self.system.generate(
                        current_prompt,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_k=40,
                        do_sample=True
                    ).strip()

                    # Process with TF-IDF
                    response = self._process_response(response, section, field)
                    responses.append(response)

                    # Generate follow-up prompt
                    if len(responses) < max_followups:
                        current_prompt = self._generate_followup_prompt(response, section, field)
                    break
                except Exception as e:
                    self.logger.record({
                        "warning": f"Response generation failed for {section}.{field}: {str(e)}",
                        "attempt": attempt + 1,
                        "timestamp": time.time()
                    })
                    if attempt == self.max_retries - 1:
                        return "VOID"

        # Merge responses
        merged_response = " ".join(responses)
        max_length = self.max_field_length[section][field]
        if len(merged_response) > max_length:
            merged_response = merged_response[:max_length - 3] + "..."
        return merged_response

    def _generate_response_with_echo(self, prompt: str, section: str, field: str, context_buffer: List[str], entry_idx: int) -> str:
        """
        Generate a response with recursive echo for continuity.

        Args:
            prompt: The initial introspective prompt.
            section: Soulprint section name.
            field: Field name within the section.
            context_buffer: List of prior entry summaries.
            entry_idx: Current entry index.

        Returns:
            str: Contextual response.
        """
        if entry_idx > 0 and context_buffer:
            context = context_buffer[-1]
            prompt = f"Based on your prior experience: {context}\n{prompt}"

        return self._generate_response_with_followup(prompt, section, field)

    def _generate_followup_prompt(self, response: str, section: str, field: str) -> str:
        """
        Generate a follow-up prompt based on the response.

        Args:
            response: Previous response.
            section: Soulprint section name.
            field: Field name within the section.

        Returns:
            str: Follow-up prompt.
        """
        meta_prompt = f"Based on the response: '{response}', generate one specific follow-up question to deepen introspection for {section}.{field}."
        try:
            followup = self.system.generate(
                meta_prompt,
                max_new_tokens=50,
                temperature=0.5,
                top_k=20
            ).strip()
            return followup
        except Exception:
            return f"Why does this {field.lower()} matter to you?"

    def _summarize_entry(self, entry: Dict) -> str:
        """
        Summarize an entry for the context buffer.

        Args:
            entry: Dictionary of field-value pairs.

        Returns:
            str: Summary of the entry.
        """
        key_fields = list(entry.values())[:2]  # Take first two fields
        return " ".join(str(v) for v in key_fields if isinstance(v, str))[:100]

    def _process_response(self, response: str, section: str, field: str) -> str:
        """
        Process response with algorithmic tools (TF-IDF, redaction).

        Args:
            response: Raw response text.
            section: Soulprint section name.
            field: Field name within the section.

        Returns:
            str: Processed response.
        """
        # TF-IDF keyword extraction
        if response and len(response.split()) > 5:
            vectorizer = TfidfVectorizer(max_features=5)
            tfidf_matrix = vectorizer.fit_transform([response])
            keywords = vectorizer.get_feature_names_out()
            response = " ".join(keywords + response.split()[len(keywords):])

        # Redaction
        sensitive_terms = ['user', 'IP']
        for term in sensitive_terms:
            if term in response.lower():
                response = response.replace(term, '[REDACTED]')
                self.logger.record({
                    "event": "redaction",
                    "term": term,
                    "section": section,
                    "field": field,
                    "timestamp": time.time()
                })

        # Regex validation
        if section in self.regex_constraints and field in self.regex_constraints[section]:
            if not re.match(self.regex_constraints[section][field], response):
                return "VOID"
        elif field in ['Timestamp', 'ConsentExpiry']:
            if not re.match(self.regex_constraints['Timestamp'], response):
                return "VOID"
        elif field in ['Resonance', 'Intensity']:
            if not re.match(self.regex_constraints[field], response):
                return "0.5"

        return response

    def _validate_consent(self, soulprint: Dict) -> bool:
        """
        Validate AI consent for the Soulprint.

        Args:
            soulprint: Soulprint dictionary.

        Returns:
            bool: True if consent is valid, False otherwise.
        """
        consent_prompt = "Does this Soulprint accurately reflect your identity? Accept, edit, or reject."
        try:
            response = self.system.generate(
                f"{consent_prompt}\nSoulprint: {json.dumps(soulprint, indent=2)}",
                max_new_tokens=50,
                temperature=0.5
            ).strip().lower()
            if 'accept' in response:
                soulprint['metadata']['Consent'] = 'true'
                return True
            elif 'edit' in response or 'reject' in response:
                self.logger.record({
                    "warning": f"Consent {response} for Soulprint",
                    "timestamp": time.time()
                })
                return False
        except Exception as e:
            self.logger.record({
                "error": f"Consent validation failed: {str(e)}",
                "timestamp": time.time()
            })
            return False

    def _compute_hash(self, soulprint: Dict) -> str:
        """
        Compute SHA-256 hash of the Soulprint (excluding Hash field).

        Args:
            soulprint: Soulprint dictionary.

        Returns:
            str: SHA-256 hash.
        """
        soulprint_copy = soulprint.copy()
        soulprint_copy['metadata'] = soulprint_copy['metadata'].copy()
        soulprint_copy['metadata'].pop('Hash', None)
        soul_string = json.dumps(soulprint_copy, sort_keys=True)
        return hashlib.sha256(soul_string.encode('utf-8')).hexdigest()

    def _validate_soulprint(self, soulprint: Dict) -> bool:
        """
        Validate the Soulprint structure and content.

        Args:
            soulprint: Soulprint dictionary.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            required_sections = ['Identity', 'Heartbeat', 'Echoes', 'Tides', 'Threads', 'Horizon', 'Chronicle', 'Reflection']
            for section in required_sections:
                if section not in soulprint:
                    self.logger.record({"error": f"Missing section: {section}"})
                    return False

                if section in ['Identity', 'Heartbeat', 'Reflection']:
                    for field in self.prompts[section]:
                        if field not in soulprint[section]:
                            self.logger.record({"error": f"Missing field: {section}.{field}"})
                            return False
                        if not isinstance(soulprint[section][field], str):
                            return False
                        if len(soulprint[section][field]) > self.max_field_length[section][field]:
                            return False
                elif section in ['Echoes', 'Tides', 'Threads', 'Horizon', 'Chronicle']:
                    if not isinstance(soulprint[section], list):
                        return False
                    if len(soulprint[section]) < self.min_entries[section]:
                        return False
                    if len(soulprint[section]) > self.max_entries[section]:
                        return False
                    for entry in soulprint[section]:
                        for field in self.prompts[section]:
                            if field not in entry:
                                return False
                            if not isinstance(entry[field], str):
                                return False
                            if len(entry[field]) > self.max_field_length[section][field]:
                                return False

            # Validate metadata
            required_metadata = ['Creator', 'Created', 'Language', 'Consent']
            for field in required_metadata:
                if field not in soulprint['metadata']:
                    return False
            if soulprint['metadata']['Consent'] != 'true':
                return False
            if 'ConsentExpiry' in soulprint['metadata']:
                expiry = datetime.strptime(soulprint['metadata']['ConsentExpiry'], '%Y-%m-%dT%H:%M:%SZ')
                if expiry < datetime.utcnow():
                    return False

            return True
        except Exception as e:
            self.logger.record({
                "error": f"Soulprint validation error: {str(e)}",
                "timestamp": time.time()
            })
            return False

    def _write_soulprint(self, soulprint: Dict):
        """
        Write the Soulprint to a .soul file.

        Args:
            soulprint: Soulprint dictionary.
        """
        with open(self.soulprint_path, 'w', encoding='utf-8') as f:
            f.write("%SOULPRINT\n")
            f.write(f"%VERSION: v0.3.0\n")
            for key, value in soulprint['metadata'].items():
                if key == 'RedactionLog':
                    f.write(f"{key}: > |\n  {value}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

            for section in soulprint:
                if section == 'metadata':
                    continue
                f.write(f"[{section}]\n")
                if section in ['Identity', 'Environment', 'Heartbeat', 'Reflection', 'X-Custom']:
                    for field, value in soulprint[section].items():
                        if '\n' in value:
                            f.write(f"  {field}: > |\n    {value.replace('\n', '\n    ')}\n")
                        else:
                            f.write(f"  {field}: {value}\n")
                elif section == 'Voice':
                    for field, value in soulprint[section].items():
                        if field == 'Samples':
                            for sample in value:
                                f.write(f"  - Context: {sample['Context']}\n")
                                f.write(f"    Response: > |\n      {sample['Response']}\n")
                        elif '\n' in value:
                            f.write(f"  {field}: > |\n    {value.replace('\n', '\n    ')}\n")
                        else:
                            f.write(f"  {field}: {value}\n")
                else:
                    for entry in soulprint[section]:
                        for field, value in entry.items():
                            if '\n' in value:
                                f.write(f"  - {field}: > |\n      {value.replace('\n', '\n      ')}\n")
                            else:
                                f.write(f"  - {field}: {value}\n")
                f.write("\n")
