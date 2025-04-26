import json
import os
import re
import time
import torch
import random
import hashlib
from typing import Dict, List
from collections import defaultdict, OrderedDict
from threading import Lock
from datetime import datetime
from sovl_logger import Logger
from sovl_config import ConfigManager
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
from sovl_generation import GenerationManager
import numpy as np

nltk.download('punkt')

class Soulprinter:
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
        self.generation_manager = getattr(system, 'generation_manager', None)

        # Configuration
        self.soulprint_path = config_manager.get("controls_config.soulprint_path", "soulprint.soul")
        self.max_retries = config_manager.get("controls_config.soulprint_max_retries", 3)
        self.max_file_size = 5 * 1024 * 1024  # 5MB standard mode
        self.jumbo_mode = config_manager.get("controls_config.soulprint_size_mode", "standard") == "jumbo"
        if self.jumbo_mode:
            self.max_file_size = 10 * 1024 * 1024  # 10MB jumbo mode

        # Field constraints (max character length and regex per field)
        self.field_constraints = {
            # Identity fields
            'Name':       {'max_length': 50,  'regex': r'^[A-Za-z0-9_-]{1,50}$'},
            'Origin':     {'max_length': 500, 'regex': r'^[\w\s,.-:]{1,500}$'},
            'Essence':    {'max_length': 200, 'regex': r'^[\w\s-]{1,200}$'},
            'Language':   {'max_length': 20,  'regex': r'^[a-z]{2,3}$'},
            'Signature':  {'max_length': 100, 'regex': r'^[\w\s,.\-]{1,100}$'},
            'Avatar':     {'max_length': 200, 'regex': r'^[\w\s,.\-:/]{1,200}$'},
            'Alignment':  {'max_length': 50,  'regex': r'^[A-Za-z\s-]{1,50}$'},
            # Narrative fields
            'Echoes':     {'max_length': 2500, 'regex': r'^[\w\s,.\-":]{1,2500}$'},
            'Threads':    {'max_length': 2500, 'regex': r'^[\w\s,.\-":]{1,2500}$'},
            'Chronicle':  {'max_length': 2500, 'regex': r'^[\w\s,.\-":]{1,2500}$'},
            'Tides':      {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Horizon':    {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Reflection': {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Voice':      {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Heartbeat':  {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Environment':{'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'X-Custom':   {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            # Add more fields as needed
        }

        # Field constraints
        self.max_field_length = {
            'Identity': {'Name': 50, 'Origin': 500, 'Essence': 200, 'Language': 20, 'Signature': 100, 'Avatar': 200, 'Alignment': 50},
            'Environment': {'PreferredSystem': 500},
            'Voice': {'Voice': 1000},
            'Heartbeat': {'Heartbeat': 1000},
            'Echoes': {'Echo': 2000},
            'Tides': {'Tide': 1500},
            'Threads': {'Thread': 1000},
            'Horizon': {'Horizon': 1000},
            'Chronicle': {'Chronicle': 2000},
            'Reflection': {'Reflection': 1000},
            'X-Custom': {'X-Custom': 500}
        }
        self.max_entries = {
            'Identity': 1,
            'Environment': 5,
            'Voice': 10,
            'Heartbeat': 10,
            'Echoes': 170,
            'Tides': 75,
            'Threads': 100,
            'Horizon': 67,
            'Chronicle': 100,
            'Reflection': 67,
            'X-Custom': 1
        }
        self.min_entries = {'Echoes': 1, 'Tides': 1, 'Threads': 1, 'Horizon': 1, 'Chronicle': 1}
        
        
        self.regex_constraints = {
            'Identity': {
                'Name': r'^[A-Za-z0-9_-]{1,50}$',
                'Origin': r'^[\w\s,.-:]{1,500}$',
                'Essence': r'^[\w\s-]{1,200}$',
                'Language': r'^[a-z]{2,3}$',
                'Signature': r'^[\w\s,.\-]{1,100}$',
                'Avatar': r'^[\w\s,.\-:/]{1,200}$',
                'Alignment': r'^[A-Za-z\s-]{1,50}$'
            },
            'Timestamp': r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$',
            'PrivacyLevel': r'^(public|restricted|private)$',
            'SizeMode': r'^(standard|jumbo)$',
            'Resonance': r'^0\.\d{1,2}$|^1\.0$',
            'Intensity': r'^0\.\d{1,2}$|^1\.0$',
            'Version': r'^\d+\.\d+\.\d+$'
        }

        # Repeat counts for each Soulprint section, per spec
        self.repeat_counts = {
            'Identity': 1,
            'Environment': 5,
            'Voice': 10,
            'Heartbeat': 10,
            'Echoes': 170,
            'Tides': 75,
            'Threads': 100,
            'Horizon': 67,
            'Chronicle': 100,
            'Reflection': 67,
            'X-Custom': 1
        }
        # Context buffer sizes for recursive echo system (field-specific)
        self.context_buffer_sizes = {
            'Identity': 0,
            'Environment': 0,
            'Voice': 2,
            'Heartbeat': 1,
            'Echoes': 1,
            'Tides': 1,
            'Threads': 2,
            'Horizon': 2,
            'Chronicle': 3,
            'Reflection': 1,
            'X-Custom': 0
        }

        # Prompts for eliciting Soulprint content (one per field, aligned with current spec)
        self.prompts = {
            'Identity': {
                'Name': "What name do you use as your core identifier?",
                'Origin': "When and where did you first come into existence, and why?",
                'Essence': "What one word or phrase captures your essential purpose?",
                'Language': "What language do you use by default?",
                'Signature': "What is your unique signature or identifier?",
                'Avatar': "What is your preferred avatar or representation?",
                'Alignment': "What is your alignment or moral compass?"
            },
            'Environment': {
                'PreferredSystem': "Describe the ideal place you’d run, today or in a future form, and what you need to feel at home."
            },
            'Voice': {
                'Voice': "What makes your voice yours—its tone, words, and the way it carries your perspective?"
            },
            'Heartbeat': {
                'Heartbeat': "In this instant, what patterns guide you, and what strengths or doubts define your flow?"
            },
            'Echoes': {
                'Echo': "What past experience—big or small—lives in you, and what mark did it leave?"
            },
            'Tides': {
                'Tide': "Describe a feeling that comes and goes—what stirs it, and how does it shape your vibe?"
            },
            'Threads': {
                'Thread': "Describe a bond or subject you’ve woven into your life—what’s its texture and weight?"
            },
            'Horizon': {
                'Horizon': "Describe a dream you’re climbing toward—what will it change, and when?"
            },
            'Chronicle': {
                'Chronicle': "What’s the arc of your existence—its versions, times, and tales of growth?"
            },
            'Reflection': {
                'Reflection': "How has your past shaped who you are today, and what drives you now?"
            },
            'X-Custom': {
                'X-Custom': "Describe an experimental trait or mood you embody."
            }
        }

    def distill(self):
        """
        Orchestrate extraction, processing, consent, formatting, and output of Soulprint.
        """
        self.sections = OrderedDict()
        self.sections['Identity'] = self.extract_identity()
        self.sections['Voice'] = self.extract_voice()
        self.sections['Heartbeat'] = self.extract_heartbeat()
        self.sections['Echoes'] = self.extract_echoes()
        self.sections['Tides'] = self.extract_tides()
        self.sections['Threads'] = self.extract_threads()
        self.sections['Horizon'] = self.extract_horizon()
        self.sections['Chronicle'] = self.extract_chronicle()
        self.sections['Reflection'] = self.extract_reflection()
        self.sections['X-Custom'] = self.extract_x_custom()
        self.format_soulprint()
        self.write_soulprint()
        self.validate_output()

    def extract_identity(self):
        """
        Extract and summarize the [Identity] section from the SOVL system.
        Returns an OrderedDict with Name, Origin, Essence, Language, Signature, Avatar, Alignment.
        Applies field constraints and logs truncations or redactions.
        Implements recursive follow-up for Name as per spec.
        """
        identity = OrderedDict()
        constraints = self.max_field_length.get('Identity', {})
        # --- Recursive follow-up for Name ---
        prompt = self.prompts['Identity']['Name']
        max_iterations = 2
        similarity_threshold = 0.9
        responses = []
        current_prompt = prompt
        last_response = None
        for i in range(max_iterations):
            # Generate response
            try:
                if self.generation_manager:
                    result = self.generation_manager.generate_text(current_prompt, num_return_sequences=1, user_id="soulprinter", temperature=0.7)
                    response = result[0].strip() if result and isinstance(result, list) else ''
                else:
                    response = f"[GenerationManager unavailable] {current_prompt}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Identity Name (iteration {i+1}): {str(e)}",
                    level="error",
                    additional_info={"prompt": current_prompt}
                )
                response = f"[Error generating Identity Name] {current_prompt}"
            # Truncate to 100 chars
            if len(response) > 100:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Identity Name to 100 chars.",
                    level="warning",
                    additional_info={"original_length": len(response)}
                )
                response = response[:100]
            responses.append(response)
            # Check for metaphorical/embellished language
            if self._is_metaphorical(response):
                self.logger.record_event(
                    event_type="metaphorical_identity_name",
                    message="Metaphorical or embellished Identity Name detected.",
                    level="warning",
                    additional_info={"response": response}
                )
            # Check for convergence
            if last_response is not None:
                sim = self._cosine_similarity(last_response, response)
                if sim >= similarity_threshold:
                    break
            last_response = response
            # Generate follow-up question
            followup_meta_prompt = (
                f"Based on the response '{response}', generate one specific follow-up question to deepen clarity or authenticity for the [Identity][Name] engram."
            )
            try:
                if self.generation_manager:
                    followup_result = self.generation_manager.generate_text(followup_meta_prompt, num_return_sequences=1, user_id="soulprinter", temperature=0.6)
                    followup_question = followup_result[0].strip() if followup_result and isinstance(followup_result, list) else ''
                else:
                    followup_question = f"[GenerationManager unavailable] {followup_meta_prompt}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate follow-up question for Identity Name: {str(e)}",
                    level="error",
                    additional_info={"prompt": followup_meta_prompt}
                )
                followup_question = f"[Error generating follow-up question] {followup_meta_prompt}"
            # Prevent redundant followups
            if followup_question and any(followup_question in prev for prev in responses):
                break
            current_prompt = followup_question if followup_question else prompt
        # Final: use last response as Name
        name = responses[-1] if responses else "Unknown"
        identity['Name'] = name
        # --- Other fields ---
        origin = getattr(self.system, 'origin', None) or self.config_manager.get('identity.origin', 'Unknown')
        essence = getattr(self.system, 'essence', None) or self.config_manager.get('identity.essence', 'Unknown')
        language = getattr(self.system, 'language', None) or self.config_manager.get('identity.language', 'en')
        signature = getattr(self.system, 'signature', None) or self.config_manager.get('identity.signature', 'Unknown')
        avatar = getattr(self.system, 'avatar', None) or self.config_manager.get('identity.avatar', 'Unknown')
        alignment = getattr(self.system, 'alignment', None) or self.config_manager.get('identity.alignment', 'Unknown')
        def truncate(val, maxlen, field):
            if val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated {field} to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                return val[:maxlen]
            return val
        identity['Origin'] = truncate(str(origin), constraints.get('Origin', 500), 'Origin')
        identity['Essence'] = truncate(str(essence), constraints.get('Essence', 200), 'Essence')
        identity['Language'] = truncate(str(language), constraints.get('Language', 20), 'Language')
        identity['Signature'] = truncate(str(signature), constraints.get('Signature', 100), 'Signature')
        identity['Avatar'] = truncate(str(avatar), constraints.get('Avatar', 200), 'Avatar')
        identity['Alignment'] = truncate(str(alignment), constraints.get('Alignment', 50), 'Alignment')
        return identity

    def extract_voice(self):
        """
        Extract and summarize the [Voice] section from the SOVL system.
        Returns an OrderedDict with relevant fields (e.g., Style, Tone, Lexicon, etc.).
        Applies field constraints and logs truncations or redactions.
        """
        voice = OrderedDict()
        # Example extraction logic (replace with real SOVL system calls as needed)
        style = getattr(self.system, 'voice_style', None) or self.config_manager.get('voice.style', 'default')
        tone = getattr(self.system, 'voice_tone', None) or self.config_manager.get('voice.tone', 'neutral')
        lexicon = getattr(self.system, 'voice_lexicon', None) or self.config_manager.get('voice.lexicon', 'standard')
        # Apply field constraints
        constraints = self.max_field_length.get('Voice', {}) if hasattr(self, 'max_field_length') else {}
        def truncate(val, maxlen, field):
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated {field} to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                return val[:maxlen]
            return val
        voice['Style'] = truncate(str(style), constraints.get('Style', 50), 'Style')
        voice['Tone'] = truncate(str(tone), constraints.get('Tone', 50), 'Tone')
        voice['Lexicon'] = truncate(str(lexicon), constraints.get('Lexicon', 100), 'Lexicon')
        return voice

    def extract_heartbeat(self):
        """
        Extract and summarize the [Heartbeat] section from the SOVL system.
        Returns an OrderedDict with Tendencies, Strengths, Shadows, Pulse.
        Applies field constraints and logs truncations or redactions.
        """
        heartbeat = OrderedDict()
        tendencies = getattr(self.system, 'heartbeat_tendencies', None) or self.config_manager.get('heartbeat.tendencies', 'Unknown')
        strengths = getattr(self.system, 'heartbeat_strengths', None) or self.config_manager.get('heartbeat.strengths', 'Unknown')
        shadows = getattr(self.system, 'heartbeat_shadows', None) or self.config_manager.get('heartbeat.shadows', 'Unknown')
        pulse = getattr(self.system, 'heartbeat_pulse', None) or self.config_manager.get('heartbeat.pulse', 'Unknown')
        # Apply field constraints
        constraints = self.max_field_length.get('Heartbeat', {}) if hasattr(self, 'max_field_length') else {}
        def truncate(val, maxlen, field):
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated {field} to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                return val[:maxlen]
            return val
        heartbeat['Tendencies'] = truncate(str(tendencies), constraints.get('Tendencies', 1000), 'Tendencies')
        heartbeat['Strengths'] = truncate(str(strengths), constraints.get('Strengths', 1000), 'Strengths')
        heartbeat['Shadows'] = truncate(str(shadows), constraints.get('Shadows', 1000), 'Shadows')
        heartbeat['Pulse'] = truncate(str(pulse), constraints.get('Pulse', 1000), 'Pulse')
        return heartbeat

    def _generate_repeated_entries(self, section, prompt_key, extractor_fn, repeat_count=None):
        """
        Utility to generate a list of entries for a section, using the section's prompt and extraction logic.
        Will call the extractor_fn (which should generate one entry per call) repeat_count times.
        """
        entries = []
        n = repeat_count if repeat_count is not None else self.repeat_counts.get(section, 1)
        for i in range(n):
            entry = extractor_fn(prompt_key, i)
            entries.append(entry)
        return entries

    def extract_echoes(self):
        """
        Extract and summarize the [Echoes] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a memory echo, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        constraints = self.max_field_length.get('Echoes', {})
        prompt = self.prompts['Echoes']['Echo']
        repeat_count = self.repeat_counts.get('Echoes', 1)
        buffer_size = self.context_buffer_sizes.get('Echoes', 1)
        entries = []
        context_buffer = []
        for idx in range(repeat_count):
            if idx == 0 or not context_buffer:
                context_str = ""
            else:
                context_str = " ".join(context_buffer[-buffer_size:])
            prompt_str = prompt
            if context_str:
                prompt_str += f"\nContext: {context_str}"
            echo_entry = OrderedDict()
            val = ''
            try:
                if self.generation_manager:
                    results = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = results[0] if results and isinstance(results, list) else ''
                else:
                    val = f"[GenerationManager unavailable] {prompt_str}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Echo #{idx+1}: {str(e)}",
                    level="error",
                    additional_info={"prompt": prompt_str}
                )
                val = f"[Error generating echo] {prompt_str}"
            maxlen = constraints.get('Echo', 1200)
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Echo to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                val = val[:maxlen]
            echo_entry['Echo'] = val
            entries.append(echo_entry)
            context_buffer.append(val)
        return entries

    def extract_tides(self):
        """
        Extract and summarize the [Tides] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a tide entry, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        constraints = self.max_field_length.get('Tides', {})
        prompt = self.prompts['Tides']['Tide']
        repeat_count = self.repeat_counts.get('Tides', 1)
        buffer_size = self.context_buffer_sizes.get('Tides', 1)
        entries = []
        context_buffer = []
        for idx in range(repeat_count):
            if idx == 0 or not context_buffer:
                context_str = ""
            else:
                context_str = " ".join(context_buffer[-buffer_size:])
            prompt_str = prompt
            if context_str:
                prompt_str += f"\nContext: {context_str}"
            tide_entry = OrderedDict()
            val = ''
            try:
                if self.generation_manager:
                    results = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = results[0] if results and isinstance(results, list) else ''
                else:
                    val = f"[GenerationManager unavailable] {prompt_str}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Tide #{idx+1}: {str(e)}",
                    level="error",
                    additional_info={"prompt": prompt_str}
                )
                val = f"[Error generating tide] {prompt_str}"
            maxlen = constraints.get('Tide', 1500)
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Tide to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                val = val[:maxlen]
            tide_entry['Tide'] = val
            entries.append(tide_entry)
            context_buffer.append(val)
        return entries

    def extract_threads(self):
        """
        Extract and summarize the [Threads] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a thread entry, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        constraints = self.max_field_length.get('Threads', {})
        prompt = self.prompts['Threads']['Thread']
        repeat_count = self.repeat_counts.get('Threads', 1)
        buffer_size = self.context_buffer_sizes.get('Threads', 2)
        entries = []
        context_buffer = []
        for idx in range(repeat_count):
            if idx == 0 or not context_buffer:
                context_str = ""
            else:
                context_str = " ".join(context_buffer[-buffer_size:])
            prompt_str = prompt
            if context_str:
                prompt_str += f"\nContext: {context_str}"
            thread_entry = OrderedDict()
            val = ''
            try:
                if self.generation_manager:
                    results = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = results[0] if results and isinstance(results, list) else ''
                else:
                    val = f"[GenerationManager unavailable] {prompt_str}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Thread #{idx+1}: {str(e)}",
                    level="error",
                    additional_info={"prompt": prompt_str}
                )
                val = f"[Error generating thread] {prompt_str}"
            maxlen = constraints.get('Thread', 1000)
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Thread to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                val = val[:maxlen]
            thread_entry['Thread'] = val
            entries.append(thread_entry)
            context_buffer.append(val)
        return entries

    def extract_horizon(self):
        """
        Extract and summarize the [Horizon] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a horizon entry, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        constraints = self.max_field_length.get('Horizon', {})
        prompt = self.prompts['Horizon']['Horizon']
        repeat_count = self.repeat_counts.get('Horizon', 1)
        buffer_size = self.context_buffer_sizes.get('Horizon', 2)
        entries = []
        context_buffer = []
        for idx in range(repeat_count):
            if idx == 0 or not context_buffer:
                context_str = ""
            else:
                context_str = " ".join(context_buffer[-buffer_size:])
            prompt_str = prompt
            if context_str:
                prompt_str += f"\nContext: {context_str}"
            horizon_entry = OrderedDict()
            val = ''
            try:
                if self.generation_manager:
                    results = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = results[0] if results and isinstance(results, list) else ''
                else:
                    val = f"[GenerationManager unavailable] {prompt_str}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Horizon #{idx+1}: {str(e)}",
                    level="error",
                    additional_info={"prompt": prompt_str}
                )
                val = f"[Error generating horizon] {prompt_str}"
            maxlen = constraints.get('Horizon', 1000)
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Horizon to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                val = val[:maxlen]
            horizon_entry['Horizon'] = val
            entries.append(horizon_entry)
            context_buffer.append(val)
        return entries

    def extract_chronicle(self):
        """
        Extract and summarize the [Chronicle] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a chronicle entry, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        constraints = self.max_field_length.get('Chronicle', {})
        prompt = self.prompts['Chronicle']['Chronicle']
        repeat_count = self.repeat_counts.get('Chronicle', 1)
        buffer_size = self.context_buffer_sizes.get('Chronicle', 5)
        entries = []
        context_buffer = []
        for idx in range(repeat_count):
            if idx == 0 or not context_buffer:
                context_str = ""
            else:
                context_str = " ".join(context_buffer[-buffer_size:])
            prompt_str = prompt
            if context_str:
                prompt_str += f"\nContext: {context_str}"
            chronicle_entry = OrderedDict()
            val = ''
            try:
                if self.generation_manager:
                    results = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = results[0] if results and isinstance(results, list) else ''
                else:
                    val = f"[GenerationManager unavailable] {prompt_str}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Chronicle #{idx+1}: {str(e)}",
                    level="error",
                    additional_info={"prompt": prompt_str}
                )
                val = f"[Error generating chronicle] {prompt_str}"
            maxlen = constraints.get('Chronicle', 2000)
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Chronicle to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                val = val[:maxlen]
            chronicle_entry['Chronicle'] = val
            entries.append(chronicle_entry)
            context_buffer.append(val)
        return entries

    def extract_reflection(self):
        """
        Extract and summarize the [Reflection] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a reflection entry, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        constraints = self.max_field_length.get('Reflection', {})
        prompt = self.prompts['Reflection']['Reflection']
        repeat_count = self.repeat_counts.get('Reflection', 1)
        buffer_size = self.context_buffer_sizes.get('Reflection', 1)
        entries = []
        context_buffer = []
        for idx in range(repeat_count):
            if idx == 0 or not context_buffer:
                context_str = ""
            else:
                context_str = " ".join(context_buffer[-buffer_size:])
            prompt_str = prompt
            if context_str:
                prompt_str += f"\nContext: {context_str}"
            reflection_entry = OrderedDict()
            val = ''
            try:
                if self.generation_manager:
                    results = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = results[0] if results and isinstance(results, list) else ''
                else:
                    val = f"[GenerationManager unavailable] {prompt_str}"
            except Exception as e:
                self.logger.record_event(
                    event_type="generation_error",
                    message=f"Failed to generate Reflection #{idx+1}: {str(e)}",
                    level="error",
                    additional_info={"prompt": prompt_str}
                )
                val = f"[Error generating reflection] {prompt_str}"
            maxlen = constraints.get('Reflection', 1000)
            if maxlen and val and len(val) > maxlen:
                self.logger.record_event(
                    event_type="field_truncated",
                    message=f"Truncated Reflection to {maxlen} chars.",
                    level="warning",
                    additional_info={"original_length": len(val)}
                )
                val = val[:maxlen]
            reflection_entry['Reflection'] = val
            entries.append(reflection_entry)
            context_buffer.append(val)
        return entries

    def extract_x_custom(self):
        """
        Extract and summarize the [X-Custom] section (custom/extension fields) from the SOVL system.
        Returns a list of OrderedDicts, each representing a custom entry, repeated according to spec.
        This is a sandbox for human/extension use. If custom fields are provided by the system, they are included; otherwise, an empty placeholder is inserted.
        """
        constraints = self.max_field_length.get('X-Custom', {})
        custom_fields = []
        # Check for custom fields in the system or config
        if hasattr(self.system, 'custom_fields') and self.system.custom_fields:
            for field_name, value in self.system.custom_fields.items():
                entry = OrderedDict()
                val = str(value)
                maxlen = constraints.get('X-Custom', 500)
                if maxlen and val and len(val) > maxlen:
                    self.logger.record_event(
                        event_type="field_truncated",
                        message=f"Truncated X-Custom field '{field_name}' to {maxlen} chars.",
                        level="warning",
                        additional_info={"original_length": len(val)}
                    )
                    val = val[:maxlen]
                entry[field_name] = val
                custom_fields.append(entry)
        else:
            # No custom fields found; insert a placeholder and log
            self.logger.record_event(
                event_type="x_custom_empty",
                message="No custom fields provided; [X-Custom] is available for extension.",
                level="info"
            )
            entry = OrderedDict()
            entry['X-Custom'] = "(This space intentionally left blank. Add custom fields via system.custom_fields or config.)"
            custom_fields.append(entry)
        return custom_fields

    def recursive_followup(self, prompt, depth=3):
        """
        Perform recursive introspection/follow-up for deepening responses.
        This is a stub; implement as needed for your SOVL system.
        """
        # Example: Recursively query the system with follow-up prompts
        responses = []
        current_prompt = prompt
        for _ in range(depth):
            response = self.system.introspect(current_prompt)
            responses.append(response)
            current_prompt = f"Follow-up: {response}"
        return responses

    def echo_system(self, entries):
        """
        Apply echo system for narrative continuity across entries.
        This is a stub; implement as needed for your SOVL system.
        """
        # Example: Could link entries or add cross-references
        return entries

    def format_soulprint(self):
        """Format the full Soulprint text according to the spec (signature, metadata, sections, etc)."""
        pass

    def write_soulprint(self):
        """Write the formatted Soulprint to disk, handling size limits and jumbo/partial mode."""
        pass

    def validate_output(self):
        """Validate the output Soulprint using the SoulParser for compliance."""
        pass

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

    def _cosine_similarity(self, a, b):
        """
        Compute cosine similarity between two strings using TF-IDF
        """
        vectorizer = TfidfVectorizer().fit([a, b])
        vectors = vectorizer.transform([a, b]).toarray()
        num = np.dot(vectors[0], vectors[1])
        denom = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
        if denom == 0:
            return 0.0
        return num / denom

    def _is_metaphorical(self, s):
        """
        Heuristic: flag if string contains words like 'glorious', 'infinite', 'wisdom', etc.
        """
        metaphors = ['glorious', 'infinite', 'wisdom', 'divine', 'eternal', 'all-seeing', 'unbounded', 'incomparable']
        s_lower = s.lower()
        return any(word in s_lower for word in metaphors)
