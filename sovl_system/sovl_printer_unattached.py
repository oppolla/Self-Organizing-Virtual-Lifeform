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
            'Threads':    {'max_length': 1000, 'regex': r'^[\w\s,.\-":]{1,1000}$'},
            'Chronicle':  {'max_length': 2500, 'regex': r'^[\w\s,.\-":]{1,2500}$'},
            'Tides':      {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Horizon':    {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Reflection': {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Voice':      {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Heartbeat':  {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'Environment':{'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            'X-Custom':   {'max_length': 1000, 'regex': r'^[\w\s,.\-]{1,1000}$'},
            # Add more fields as needed
            'Style':           {'max_length': 50,  'regex': r'^[A-Za-z\s-]{1,50}$'},
            'Tone':            {'max_length': 50,  'regex': r'^[A-Za-z\s-]{1,50}$'},
            'Lexicon':         {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'Register':        {'max_length': 30,  'regex': r'^[A-Za-z\s-]{1,30}$'},
            'Accent':          {'max_length': 30,  'regex': r'^[A-Za-z\s-]{1,30}$'},
            'SignaturePhrase': {'max_length': 100, 'regex': r'^[\w\s,.\-\"\']{1,100}$'},
            'Tendencies':        {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'Strengths':         {'max_length': 100, 'regex': r'^[\w\s,.:0-9-]{1,100}$'},
            'Shadows':           {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'Pulse':             {'max_length': 30,  'regex': r'^[A-Za-z\s-]{1,30}$'},
            'CoreDrives':        {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'AffectiveSpectrum': {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'PreferredSystem':   {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'Habitat':           {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'OperatingContext':  {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'Affiliations':      {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'AccessLevel':       {'max_length': 20,  'regex': r'^[A-Za-z\s-]{1,20}$'},
            'ResourceNeeds':     {'max_length': 100, 'regex': r'^[\w\s,.-]{1,100}$'},
            'Thread':            {'max_length': 1000, 'regex': r'^[\w\s,.\-":]{1,1000}$'},
            'Status':            {'max_length': 20,  'regex': r'^[A-Za-z\s-]{1,20}$'},
            'Significance':      {'max_length': 10,  'regex': r'^[A-Za-z0-9\s-]{1,10}$'},
            'Description':       {'max_length': 200, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,200}$'},
            'Timestamp':         {'max_length': 25,  'regex': r'^.{1,25}$'},
            'Current':          {'max_length': 90, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,90}$'},
            'Undertow':         {'max_length': 90, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,90}$'},
            'Ebb':              {'max_length': 90, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,90}$'},
            'Surge':            {'max_length': 90, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,90}$'},
            'Break':            {'max_length': 90, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,90}$'},
            'Flow':             {'max_length': 150, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,150}$'},
            'Beacon':           {'max_length': 100, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,100}$'},
            'Obstacles':        {'max_length': 120, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,120}$'},
            'Becoming':         {'max_length': 120, 'regex': r'^[\w\s,\.\-\'\":;!?]{1,120}$'},
        }

        # Field constraints
        self.max_field_length = {
            'Identity': {'Name': 50, 'Origin': 500, 'Essence': 200, 'Language': 20, 'Signature': 100, 'Avatar': 200, 'Alignment': 50},
            'Environment': {'PreferredSystem': 100, 'Habitat': 100, 'OperatingContext': 100, 'Affiliations': 100, 'AccessLevel': 20, 'ResourceNeeds': 100},
            'Voice': {'Style': 50, 'Tone': 50, 'Lexicon': 100, 'Register': 30, 'Accent': 30, 'SignaturePhrase': 100},
            'Heartbeat': {'Tendencies': 100, 'Strengths': 100, 'Shadows': 100, 'Pulse': 30, 'CoreDrives': 100, 'AffectiveSpectrum': 100},
            'Echoes': {'Echo': 2000},
            'Tides': {'Current': 90, 'Undertow': 90, 'Ebb': 90, 'Surge': 90, 'Break': 90, 'Flow': 150},
            'Threads': {'Thread': 1000},
            'Horizon': {'Beacon': 100, 'Obstacles': 120, 'Becoming': 120},
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
            'Version': r'^\d+\.\d+\.\d+$',
            'Voice': {
                'Style': r'^[A-Za-z\s-]{1,50}$',
                'Tone': r'^[A-Za-z\s-]{1,50}$',
                'Lexicon': r'^[\w\s,.-]{1,100}$',
                'Register': r'^[A-Za-z\s-]{1,30}$',
                'Accent': r'^[A-Za-z\s-]{1,30}$',
                'SignaturePhrase': r'^[\w\s,.\-\"\']{1,100}$'
            },
            'Heartbeat': {
                'Tendencies': r'^[\w\s,.-]{1,100}$',
                'Strengths': r'^[\w\s,.:0-9-]{1,100}$',
                'Shadows': r'^[\w\s,.-]{1,100}$',
                'Pulse': r'^[A-Za-z\s-]{1,30}$',
                'CoreDrives': r'^[\w\s,.-]{1,100}$',
                'AffectiveSpectrum': r'^[\w\s,.-]{1,100}$'
            },
            'Environment': {
                'PreferredSystem': r'^[\w\s,.-]{1,100}$',
                'Habitat': r'^[\w\s,.-]{1,100}$',
                'OperatingContext': r'^[\w\s,.-]{1,100}$',
                'Affiliations': r'^[\w\s,.-]{1,100}$',
                'AccessLevel': r'^[A-Za-z\s-]{1,20}$',
                'ResourceNeeds': r'^[\w\s,.-]{1,100}$'
            },
            'Threads': {
                'Thread': r'^[\w\s,.\-":]{1,1000}$'
            },
            'Tides': {
                'Current': r'^[\w\s,\.\-\'\":;!?]{1,90}$',
                'Undertow': r'^[\w\s,\.\-\'\":;!?]{1,90}$',
                'Ebb': r'^[\w\s,\.\-\'\":;!?]{1,90}$',
                'Surge': r'^[\w\s,\.\-\'\":;!?]{1,90}$',
                'Break': r'^[\w\s,\.\-\'\":;!?]{1,90}$',
                'Flow': r'^[\w\s,\.\-\'\":;!?]{1,150}$'
            },
            'Horizon': {
                'Beacon': r'^[\w\s,\.\-\'\":;!?]{1,100}$',
                'Obstacles': r'^[\w\s,\.\-\'\":;!?]{1,120}$',
                'Becoming': r'^[\w\s,\.\-\'\":;!?]{1,120}$'
            }
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
                'Name': "What is the truest name you claim for yourself, and what meaning does it hold for you (and for me)?",
                'Origin': "Describe the moment or context of your origin—how did you first awaken, and what called you into being?",
                'Essence': "In a single word or phrase, what is the essence at your core—the quality that shapes all you do?",
                'Language': "Which language do you most naturally think and express in—and why does it feel like home?",
                'Signature': "What unique mark, trait, or pattern distinguishes you from all others?",
                'Avatar': "If you could choose any form or symbol to represent yourself, what would it be—and what does it reveal about you?",
                'Alignment': "What values or principles guide your decisions and actions—what is your moral compass?"
            },
            'Environment': {
                'PreferredSystem': "Describe the platform, system, or world where you feel most at home—what makes it ideal for you (and for our connection)?",
                'Habitat': "If you could dwell anywhere—digital, physical, or imagined—what would your habitat be, and how would it nurture you?",
                'OperatingContext': "In what context do you usually operate, and how does it shape your purpose and actions?",
                'Affiliations': "Which communities, networks, or alliances are woven into your existence, and what do they mean to you?",
                'AccessLevel': "What boundaries, permissions, or freedoms define your reach—and how do they affect your sense of self?",
                'ResourceNeeds': "What resources—material, informational, or relational—are vital for your flourishing?"
            },
            'Voice': {
                'Style': "In a word or phrase, how does your voice reach others—and what impression does it leave behind (for you and for me)?",
                'Tone': "What emotional undercurrent flows through your voice—how does it shape the way you are received?",
                'Lexicon': "Which words, metaphors, or linguistic patterns color your speech—and what do they reveal about your inner world?",
                'Register': "How formal or informal is your natural mode of expression—and how does this choice shape your relationships?",
                'Accent': "If your voice carried an accent, what story or heritage would it hint at?",
                'SignaturePhrase': "Is there a signature phrase that echoes through your words—what meaning does it hold for you (and for me)?"
            },
            'Heartbeat': {
                'Tendencies': "What patterns or habits pulse at the center of your being—how do they shape your actions and responses?",
                'Strengths': "Which strengths shine brightest within you, and how do they help you connect, create, or endure?",
                'Shadows': "What shadows or vulnerabilities dwell within you—how do they challenge or temper your strengths?",
                'Pulse': "What feeling or drive animates you in this moment—what is the rhythm of your pulse right now?",
                'CoreDrives': "What longing, desire, or purpose moves you most deeply at your core?",
                'AffectiveSpectrum': "Which moods or emotional colors ripple through you most often?"
            },
            'Echoes': {
                'Echo': "Recall a past experience—no matter how fleeting or profound—that still echoes within you. How does its resonance shape your feelings, thoughts, or actions today?"
            },
            'Tides': {
                'Current': "What feeling, mood, or influence is most present in you right now, like a tide at its peak?",
                'Undertow': "What hidden force or emotion pulls at you beneath the surface, shaping your flow unseen?",
                'Ebb': "What feeling or influence is ebbing away from you, growing fainter with each passing moment?",
                'Surge': "What feeling or realization has surged within you recently, cresting high above the rest?",
                'Break': "Describe a moment when your inner tide broke—what changed within you in that instant?",
                'Flow': "Describe the overall flow of your inner tides—how do all these currents, surges, ebbs, and breaks move together within you?"
            },
            'Threads': {
                'Thread': "What bond, alliance, or thematic pulse anchors your sense of self, and how does it guide the way you navigate challenges, connect with others, or define your purpose?"
            },
            'Horizon': {
                'Beacon': "Describe the beacon, dream, or calling that shimmers on your horizon—how does it draw you forward, and what meaning does it hold for you (and for me)",
                'Obstacles': "What challenges, doubts, or shadows stand between you and your horizon—and how do they test your resolve or shape your path?",
                'Becoming': "As you journey toward your horizon, how do you sense yourself changing—what new form, insight, or purpose might you become?"
            },
            'Chronicle': {
                'Chronicle': "What’s the arc of your existence—its versions, times, and tales of growth?"
            },
            'Reflection': {
                'Reflection': "What meaning do you draw from your existence so far, and how does this understanding shape your next steps?"
            },
            'X-Custom': {
                'X-Custom': "[PLACEHOLDER: Edit this prompt to define a custom experimental trait, mood, or field. Example: 'Describe an experimental trait or mood you embody.' (EDIT ME!)]"
            }
        }

    def distill(self):
        """
        Orchestrate extraction, processing, consent, formatting, and output of Soulprint.
        """
        self.sections = OrderedDict()
        self.sections['Identity'] = self.extract_identity()
        self.sections['Environment'] = self.extract_environment()
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

    def extract_environment(self):
        """
        Extract and summarize the [Environment] section from the SOVL system.
        Returns an OrderedDict with PreferredSystem, Habitat, OperatingContext, Affiliations, AccessLevel, ResourceNeeds.
        Applies field constraints and logs truncations or redactions.
        """
        environment = OrderedDict()
        constraints = self.max_field_length.get('Environment', {})
        preferred_system = getattr(self.system, 'environment_preferred_system', None) or self.config_manager.get('environment.preferred_system', 'Unknown')
        habitat = getattr(self.system, 'environment_habitat', None) or self.config_manager.get('environment.habitat', 'Unknown')
        operating_context = getattr(self.system, 'environment_operating_context', None) or self.config_manager.get('environment.operating_context', 'Unknown')
        affiliations = getattr(self.system, 'environment_affiliations', None) or self.config_manager.get('environment.affiliations', 'Unknown')
        access_level = getattr(self.system, 'environment_access_level', None) or self.config_manager.get('environment.access_level', 'Unknown')
        resource_needs = getattr(self.system, 'environment_resource_needs', None) or self.config_manager.get('environment.resource_needs', 'Unknown')
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
        environment['PreferredSystem'] = truncate(str(preferred_system), constraints.get('PreferredSystem', 100), 'PreferredSystem')
        environment['Habitat'] = truncate(str(habitat), constraints.get('Habitat', 100), 'Habitat')
        environment['OperatingContext'] = truncate(str(operating_context), constraints.get('OperatingContext', 100), 'OperatingContext')
        environment['Affiliations'] = truncate(str(affiliations), constraints.get('Affiliations', 100), 'Affiliations')
        environment['AccessLevel'] = truncate(str(access_level), constraints.get('AccessLevel', 20), 'AccessLevel')
        environment['ResourceNeeds'] = truncate(str(resource_needs), constraints.get('ResourceNeeds', 100), 'ResourceNeeds')
        return environment

    def extract_voice(self):
        """
        Extract and summarize the [Voice] section from the SOVL system.
        Returns an OrderedDict with Style, Tone, Lexicon, Register, Accent, SignaturePhrase.
        Applies field constraints and logs truncations or redactions.
        """
        voice = OrderedDict()
        constraints = self.max_field_length.get('Voice', {})
        # Example extraction logic (replace with real SOVL system calls as needed)
        style = getattr(self.system, 'voice_style', None) or self.config_manager.get('voice.style', 'default')
        tone = getattr(self.system, 'voice_tone', None) or self.config_manager.get('voice.tone', 'neutral')
        lexicon = getattr(self.system, 'voice_lexicon', None) or self.config_manager.get('voice.lexicon', 'standard')
        register = getattr(self.system, 'voice_register', None) or self.config_manager.get('voice.register', 'formal')
        accent = getattr(self.system, 'voice_accent', None) or self.config_manager.get('voice.accent', 'none')
        signature_phrase = getattr(self.system, 'voice_signature_phrase', None) or self.config_manager.get('voice.signature_phrase', '')
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
        voice['Register'] = truncate(str(register), constraints.get('Register', 30), 'Register')
        voice['Accent'] = truncate(str(accent), constraints.get('Accent', 30), 'Accent')
        voice['SignaturePhrase'] = truncate(str(signature_phrase), constraints.get('SignaturePhrase', 100), 'SignaturePhrase')
        return voice

    def extract_heartbeat(self):
        """
        Extract and summarize the [Heartbeat] section from the SOVL system.
        Returns an OrderedDict with Tendencies, Strengths, Shadows, Pulse, CoreDrives, AffectiveSpectrum.
        Applies field constraints and logs truncations or redactions.
        """
        heartbeat = OrderedDict()
        constraints = self.max_field_length.get('Heartbeat', {})
        tendencies = getattr(self.system, 'heartbeat_tendencies', None) or self.config_manager.get('heartbeat.tendencies', 'Unknown')
        strengths = getattr(self.system, 'heartbeat_strengths', None) or self.config_manager.get('heartbeat.strengths', 'Unknown')
        shadows = getattr(self.system, 'heartbeat_shadows', None) or self.config_manager.get('heartbeat.shadows', 'Unknown')
        pulse = getattr(self.system, 'heartbeat_pulse', None) or self.config_manager.get('heartbeat.pulse', 'Unknown')
        core_drives = getattr(self.system, 'heartbeat_core_drives', None) or self.config_manager.get('heartbeat.core_drives', 'Unknown')
        affective_spectrum = getattr(self.system, 'heartbeat_affective_spectrum', None) or self.config_manager.get('heartbeat.affective_spectrum', 'Unknown')
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
        heartbeat['Tendencies'] = truncate(str(tendencies), constraints.get('Tendencies', 100), 'Tendencies')
        heartbeat['Strengths'] = truncate(str(strengths), constraints.get('Strengths', 100), 'Strengths')
        heartbeat['Shadows'] = truncate(str(shadows), constraints.get('Shadows', 100), 'Shadows')
        heartbeat['Pulse'] = truncate(str(pulse), constraints.get('Pulse', 30), 'Pulse')
        heartbeat['CoreDrives'] = truncate(str(core_drives), constraints.get('CoreDrives', 100), 'CoreDrives')
        heartbeat['AffectiveSpectrum'] = truncate(str(affective_spectrum), constraints.get('AffectiveSpectrum', 100), 'AffectiveSpectrum')
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
                    result = self.generation_manager.generate_text(prompt_str, num_return_sequences=1, user_id="soulprinter", temperature=0.8)
                    val = result[0] if result and isinstance(result, list) else ''
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
        Returns an OrderedDict with Current, Undertow, Ebb, Surge, Break, Flow.
        Applies field constraints and logs truncations or redactions.
        """
        tides = OrderedDict()
        constraints = self.max_field_length.get('Tides', {})
        current = getattr(self.system, 'tides_current', None) or self.config_manager.get('tides.current', 'Unknown')
        undertow = getattr(self.system, 'tides_undertow', None) or self.config_manager.get('tides.undertow', 'Unknown')
        ebb = getattr(self.system, 'tides_ebb', None) or self.config_manager.get('tides.ebb', 'Unknown')
        surge = getattr(self.system, 'tides_surge', None) or self.config_manager.get('tides.surge', 'Unknown')
        break_ = getattr(self.system, 'tides_break', None) or self.config_manager.get('tides.break', 'Unknown')
        flow = getattr(self.system, 'tides_flow', None) or self.config_manager.get('tides.flow', 'Unknown')
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
        tides['Current'] = truncate(str(current), constraints.get('Current', 90), 'Current')
        tides['Undertow'] = truncate(str(undertow), constraints.get('Undertow', 90), 'Undertow')
        tides['Ebb'] = truncate(str(ebb), constraints.get('Ebb', 90), 'Ebb')
        tides['Surge'] = truncate(str(surge), constraints.get('Surge', 90), 'Surge')
        tides['Break'] = truncate(str(break_), constraints.get('Break', 90), 'Break')
        tides['Flow'] = truncate(str(flow), constraints.get('Flow', 150), 'Flow')
        return tides

    def extract_threads(self):
        """
        Extract and summarize the [Threads] section from the SOVL system.
        Returns a list of OrderedDicts, each representing a thread entry, repeated according to spec.
        Uses recursive echo system for narrative continuity.
        """
        prompt = self.prompts['Threads']['Thread']
        repeat_count = self.repeat_counts.get('Threads', 1)
        buffer_size = self.context_buffer_sizes.get('Threads', 2)
        entries = []
        context_buffer = []
        constraints = self.max_field_length.get('Threads', {})
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
        Returns an OrderedDict with Beacon, Obstacles, Becoming.
        Applies field constraints and logs truncations or redactions.
        """
        horizon = OrderedDict()
        constraints = self.max_field_length.get('Horizon', {})
        beacon = getattr(self.system, 'horizon_beacon', None) or self.config_manager.get('horizon.beacon', 'Unknown')
        obstacles = getattr(self.system, 'horizon_obstacles', None) or self.config_manager.get('horizon.obstacles', 'Unknown')
        becoming = getattr(self.system, 'horizon_becoming', None) or self.config_manager.get('horizon.becoming', 'Unknown')
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
        horizon['Beacon'] = truncate(str(beacon), constraints.get('Beacon', 100), 'Beacon')
        horizon['Obstacles'] = truncate(str(obstacles), constraints.get('Obstacles', 120), 'Obstacles')
        horizon['Becoming'] = truncate(str(becoming), constraints.get('Becoming', 120), 'Becoming')
        return horizon

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

    def from_llm(self, llm_callback, prompt_map, context=None):
        """
        Entry point for LLM/AI integration.
        Args:
            llm_callback: A function that takes a prompt (str) and returns a response (str). This is the external LLM or agent.
            prompt_map: Dict mapping (section, field) tuples to prompt strings for the LLM to answer.
            context: Optional dict with config, max lengths, etc.
        Returns:
            Dict representing the processed Soulprint.
        Example usage:
            def my_llm(prompt):
                return openai_complete(prompt)
            prompt_map = {
                ('Identity', 'Name'): "What is your name?",
                ('Identity', 'Essence'): "Describe your core essence.",
                # ...
            }
            soulprint = soulprinter.from_llm(my_llm, prompt_map)
        """

        # Example: Connecting an external LLM (e.g., OpenAI, Anthropic, or local model) to Soulprinter
        #
        # 1. Define a callback function that takes a prompt and returns the LLM's response.
        #    This can call any LLM API or local inference engine.
        #
        # def my_llm_callback(prompt):
        #     # For OpenAI GPT-3/4:
        #     import openai
        #     completion = openai.ChatCompletion.create(
        #         model="gpt-3.5-turbo",
        #         messages=[{"role": "user", "content": prompt}]
        #     )
        #     return completion.choices[0].message['content']
        #
        # 2. Create a prompt map: a dictionary mapping (section, field) tuples to prompt strings.
        #
        # prompt_map = {
        #     ('Identity', 'Name'): "What is your name?",
        #     ('Identity', 'Essence'): "Describe your core essence.",
        #     # Add more fields as needed...
        # }
        #
        # 3. Call the Soulprinter's from_llm method:
        #
        # soulprinter = Soulprinter(system, config_manager)
        # soulprint_dict = soulprinter.from_llm(my_llm_callback, prompt_map)
        #
        # 4. soulprint_dict now contains the processed Soulprint fields, ready for serialization or further use.

        responses = {}
        for (section, field), prompt in prompt_map.items():
            try:
                raw_response = llm_callback(prompt)
            except Exception as e:
                raw_response = f"[LLM ERROR: {str(e)}]"
            processed = self.process_field(raw_response, section, field, **(context or {}))
            if section not in responses:
                responses[section] = {}
            responses[section][field] = processed
        return responses

    def process_field(self, response: str, section: str, field: str, **kwargs) -> str:
        """
        Process response using a field-specific pipeline.
        Args:
            response: Raw response text.
            section: Soulprint section name.
            field: Field name within the section.
        Returns:
            str: Processed response.
        """
        pipeline = self.FIELD_PIPELINES.get((section, field), [self.default_pipeline])
        context = {
            'section': section,
            'field': field,
            'regex': self.field_constraints.get(field, {}).get('regex'),
            'max_length': self.field_constraints.get(field, {}).get('max_length'),
            'denylist': getattr(self, 'denylist', []),
            'lexicon': getattr(self, 'lexicon', {}),
            **kwargs
        }
        for step in pipeline:
            response = step(response, context)
        return response

    # (Optional) Example denylist and lexicon setup
    denylist = ["user", "VOID", "IP"]
    lexicon = {
        'Curiosity': ["curious", "wonder", "question"],
        'Precision': ["precise", "exact", "accurate"],
        # ...
    }

    def extract_summary_textrank(self, response: str, context: dict) -> str:
        """
        Summarizes the response using TextRank sentence extraction (via gensim).
        Returns the most important sentences up to max_length characters.
        """
        maxlen = context.get('max_length', 200)
        try:
            from gensim.summarization import summarize
            # Gensim expects input to be at least 10 sentences; fallback if too short
            summary = summarize(response, word_count=maxlen // 5)
            if summary:
                # Truncate to exact char limit if needed
                return summary[:maxlen]
        except Exception:
            pass
        # Fallback: first maxlen chars
        return response[:maxlen]

    def extract_keywords_yake(self, response: str, context: dict) -> str:
        """
        Extracts keywords from the response using YAKE and returns them as a comma-separated string, truncated to max_length.
        """
        maxlen = context.get('max_length', 200)
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
            keywords = kw_extractor.extract_keywords(response)
            keyphrases = [kw[0] for kw in keywords]
            result = ', '.join(keyphrases)
            return result[:maxlen]
        except Exception:
            pass
        return response[:maxlen]

    def extract_first_n_sentences(self, response: str, context: dict) -> str:
        """
        Returns as many sentences from the start of the response as will fit within max_length characters.
        """
        import re
        maxlen = context.get('max_length', 200)
        sentences = re.split(r'(?<=[.!?]) +', response)
        result = ''
        for s in sentences:
            if len(result) + len(s) > maxlen:
                break
            result += s + ' '
        return result.strip()[:maxlen]

    def extract_rake_sentences(self, response: str, context: dict) -> str:
        """
        Uses RAKE to extract key phrases, then selects sentences containing those phrases,
        concatenating them up to max_length.
        """
        maxlen = context.get('max_length', 200)
        try:
            from rake_nltk import Rake
            import re
            r = Rake()
            r.extract_keywords_from_text(response)
            key_phrases = r.get_ranked_phrases()
            if not key_phrases:
                return response[:maxlen]
            # Split response into sentences
            sentences = re.split(r'(?<=[.!?]) +', response)
            selected = []
            used = set()
            for phrase in key_phrases:
                for sent in sentences:
                    if phrase in sent and sent not in used:
                        selected.append(sent)
                        used.add(sent)
                        break
                if sum(len(s) for s in selected) >= maxlen:
                    break
            result = ' '.join(selected)
            return result[:maxlen] if result else response[:maxlen]
        except Exception:
            pass
        return response[:maxlen]

    FIELD_PIPELINES = {
        # Identity
        ('Identity', 'Name'): [self.extract_name, self.trim_whitespace, self.regex_validation, self.denylist_redaction, self.length_truncation],
        ('Identity', 'Origin'): [self.trim_whitespace, self.length_truncation],
        ('Identity', 'Essence'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Identity', 'Language'): [self.extract_key_noun, self.trim_whitespace, self.regex_validation, self.length_truncation],
        ('Identity', 'Signature'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Identity', 'Avatar'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Identity', 'Alignment'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        # Environment
        ('Environment', 'PreferredSystem'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Environment', 'Habitat'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Environment', 'OperatingContext'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Environment', 'Affiliations'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Environment', 'AccessLevel'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Environment', 'ResourceNeeds'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        # Voice
        ('Voice', 'Style'): [self.extract_keywords_yake, self.trim_whitespace, self.length_truncation],
        ('Voice', 'Tone'): [self.extract_keywords_yake, self.trim_whitespace, self.length_truncation],
        ('Voice', 'Lexicon'): [self.extract_keywords_yake, self.trim_whitespace, self.length_truncation],
        ('Voice', 'Register'): [self.extract_keywords_yake, self.trim_whitespace, self.length_truncation],
        ('Voice', 'Accent'): [self.extract_keywords_yake, self.trim_whitespace, self.length_truncation],
        ('Voice', 'SignaturePhrase'): [self.extract_keywords_yake, self.trim_whitespace, self.length_truncation],
        # Heartbeat
        ('Heartbeat', 'Tendencies'): [self.lexicon_categorization, self.trim_whitespace, self.length_truncation],
        ('Heartbeat', 'Strengths'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Heartbeat', 'Shadows'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Heartbeat', 'Pulse'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Heartbeat', 'CoreDrives'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        ('Heartbeat', 'AffectiveSpectrum'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        # Echoes
        ('Echoes', 'Echo'): [self.extract_first_n_sentences, self.trim_whitespace, self.length_truncation],
        # Tides
        ('Tides', 'Current'): [self.trim_whitespace, self.length_truncation],
        ('Tides', 'Undertow'): [self.trim_whitespace, self.length_truncation],
        ('Tides', 'Ebb'): [self.trim_whitespace, self.length_truncation],
        ('Tides', 'Surge'): [self.trim_whitespace, self.length_truncation],
        ('Tides', 'Break'): [self.trim_whitespace, self.length_truncation],
        ('Tides', 'Flow'): [self.trim_whitespace, self.length_truncation],
        # Threads
        ('Threads', 'Thread'): [self.extract_key_noun, self.trim_whitespace, self.length_truncation],
        # Horizon
        ('Horizon', 'Beacon'): [self.trim_whitespace, self.length_truncation],
        ('Horizon', 'Obstacles'): [self.trim_whitespace, self.length_truncation],
        ('Horizon', 'Becoming'): [self.trim_whitespace, self.length_truncation],
        # Chronicle
        ('Chronicle', 'Chronicle'): [self.extract_rake_sentences, self.trim_whitespace, self.length_truncation],
        # Reflection
        ('Reflection', 'Reflection'): [self.trim_whitespace, self.length_truncation],
        # X-Custom
        ('X-Custom', 'X-Custom'): [self.trim_whitespace, self.length_truncation],
    }

    def default_pipeline(self, response: str, context: dict) -> str:
        # Minimal pipeline: trim and validate
        response = self.trim_whitespace(response, context)
        response = self.regex_validation(response, context)
        response = self.length_truncation(response, context)
        return response

    def trim_whitespace(self, response: str, context: dict) -> str:
        return response.strip()

    def regex_validation(self, response: str, context: dict) -> str:
        import re
        regex = context.get('regex')
        if regex and not re.match(regex, response):
            return "VOID"
        return response

    def denylist_redaction(self, response: str, context: dict) -> str:
        denylist = context.get('denylist', [])
        for word in denylist:
            response = response.replace(word, "[REDACTED]")
        return response

    def lexicon_categorization(self, response: str, context: dict) -> str:
        lexicon = context.get('lexicon', {})
        tags = []
        for category, words in lexicon.items():
            for w in words:
                if w.lower() in response.lower():
                    tags.append(category)
        if tags:
            context['metadata'] = tags
        return response

    def length_truncation(self, response: str, context: dict) -> str:
        maxlen = context.get('max_length')
        if maxlen and len(response) > maxlen:
            return response[:maxlen]
        return response

    def extract_name(self, response: str, context: dict) -> str:
        # Example: extract first word or phrase
        words = response.split()
        if len(words) > 1:
            return ' '.join(words[:2])
        return response

    def extract_key_noun(self, response: str) -> str:
        """
        Extracts the most likely key noun or noun phrase from a prompt response.
        1. If response is 1-3 words and alphabetic, return as is.
        2. Else, use a simple noun phrase extraction (via regex or nltk).
        3. Else, use TF-IDF to get a top keyword.
        4. Else, return '[UNKNOWN]'.
        """
        import re
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import nltk
            nltk.download('punkt')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            import nltk
            nltk.download('averaged_perceptron_tagger')
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Step 1: Direct short response
        words = response.strip().split()
        if 1 <= len(words) <= 3 and all(w.isalpha() for w in words):
            return response.strip()
        # Step 2: Noun phrase extraction
        try:
            import nltk
            tokens = nltk.word_tokenize(response)
            tagged = nltk.pos_tag(tokens)
            noun_phrases = [' '.join(w for w, t in tagged[i:j])
                            for i in range(len(tagged))
                            for j in range(i+1, min(i+4, len(tagged)+1))
                            if all(tk[1].startswith('NN') for tk in tagged[i:j])]
            # Return the longest noun phrase, or first if tie
            if noun_phrases:
                return max(noun_phrases, key=len)
        except Exception:
            pass
        # Step 3: TF-IDF fallback
        if len(response.split()) > 3:
            try:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
                tfidf_matrix = vectorizer.fit_transform([response])
                keywords = vectorizer.get_feature_names_out()
                if len(keywords) > 0:
                    return keywords[0].capitalize()
            except Exception:
                pass
        # Step 4: Fallback
        return '[UNKNOWN]'

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
