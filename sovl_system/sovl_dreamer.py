import json
import random
import time
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Any
from functools import wraps
from copy import deepcopy
import hashlib
from sovl_io import JSONLLoader
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager, ScaffoldError
from sovl_queue import get_scribe_queue, capture_scribe_event
from sovl_processor import ScribeIngestionProcessor
from sovl_schema import ConfigSchema
import tracemalloc

"""
Handles all dream generation, dream structure, narration, and scribe journal integration for SOVL
"""

class DreamNarrationStrategy:
    """Base class for dream narration strategies."""
    def narrate(self, dream_event: Dict[str, Any], noise_level: float) -> str:
        raise NotImplementedError

class SurrealNarrationStrategy(DreamNarrationStrategy):
    """Surreal narration blending two memory fragments with dream-like phrases."""
    def get_llm_dreamlike_phrase(self, generation_manager):
        prompt = (
            "Invent a short, surreal, poetic phrase (max 8 words) that could appear in a dream. "
            "Do not reference music, songs, or lyrics. Be mysterious and evocative."
        )
        return generation_manager.generate_text(prompt, num_return_sequences=1)[0].strip()

    def narrate(self, dream_event: Dict[str, Any], noise_level: float, generation_manager=None) -> str:
        """
        Compose a short, surreal, dreamlike narration blending two memory fragments
        """
        memory1 = dream_event['event_data'].get('memory1', '')
        memory2 = dream_event['event_data'].get('memory2', '')
        if memory1 and memory2 and memory1 != memory2:
            prompt = (
                "Essential qualities:\n"
                "  - Compose a short, surreal, dreamlike narration (max 2 sentences) that blends the following two dream fragments.\n"
                "  - The narration should be vivid, mysterious, and evocative of a dream.\n"
                "  - You may optionally end with a brief, surreal phrase.\n"
                "Fragments:\n"
                f"  - Fragment 1: {memory1}\n"
                f"  - Fragment 2: {memory2}\n"
                "Key constraints:\n"
                "  - Output only your narration, no explanations or commentary.\n"
                "  - Do not reference music, songs, or lyrics directly.\n"
                "  - Do not refer to the specific key, scale, or tempo.\n"
                "  - Do not include dialogue or meta-commentary.\n"
            )
        else:
            prompt = (
                "Essential qualities:\n"
                "  - Compose a short, surreal, dreamlike narration (max 2 sentences) inspired by the following dream fragment.\n"
                "  - The narration should be vivid, mysterious, and evocative of a dream.\n"
                "  - You may optionally end with a brief, surreal phrase.\n"
                "Fragment:\n"
                f"  - Fragment: {memory1 or memory2}\n"
                "Key constraints:\n"
                "  - Output only your narration, no explanations or commentary.\n"
                "  - Do not reference music, songs, or lyrics directly.\n"
                "  - Do not refer to the specific key, scale, or tempo.\n"
                "  - Do not include dialogue or meta-commentary.\n"
            )
        narration = generation_manager.generate_text(prompt, num_return_sequences=1)[0].strip()
        return narration

class DreamEventSelector:
    """Handles event extraction and selection for dream generation."""
    def __init__(self, config_manager: ConfigManager, scribe_path: str, error_manager: ErrorManager):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = Logger.get_instance()
        self.error_manager = error_manager
        self.max_dreams = config_manager.get("dreamer_config.max_events_per_cycle")
        self.novelty_weight = config_manager.get("dreamer_config.novelty_weight")
        self.confidence_weight = config_manager.get("dreamer_config.confidence_weight")
        self.selection_strategy = config_manager.get("dreamer_config.selection_strategy")

    def extract_last_active_period(self) -> List[Dict]:
        """Extract events since the last 'wake' event, streaming to reduce memory usage."""
        try:
            loader = JSONLLoader(self.config_manager, self.logger, self.error_manager)
            events = []
            for entry in loader.stream_jsonl(self.scribe_path):
                if entry.get("event_type") == "wake":
                    events = []
                else:
                    events.append(entry)
            return events
        except Exception as e:
            error_type = f"dreamer_extract_last_active_period_error"
            context = {"function": "extract_last_active_period"}
            self.error_manager.record_error(
                error=ScaffoldError(
                    message=f"Error in extract_last_active_period: {str(e)}",
                    operation="extract_last_active_period",
                    context=context
                ),
                error_type=error_type,
                context=context,
                stack_trace=traceback.format_exc()
            )
            Logger.get_instance().record_event(
                event_type=error_type,
                message=f"Error in extract_last_active_period: {str(e)}",
                level="error",
                additional_info=context,
                stack_trace=traceback.format_exc()
            )
        return []

    def score_and_select_dreams(self, events: List[Dict]) -> List[Dict]:
        """Select events based on weight, with a pinch of randomization."""
        try:
            # Extract weights, prefer event['weight'], fallback to event['event_data']['weight']
            weighted_events = []
            for event in events:
                weight = event.get('weight')
                if weight is None:
                    # Try event_data['weight']
                    event_data = event.get('event_data', {})
                    weight = event_data.get('weight', 1.0)
                weighted_events.append((weight, event))
            # Sort by weight descending
            weighted_events.sort(reverse=True, key=lambda x: x[0])
            # Top-N pool (2x max_dreams or all if fewer)
            pool_size = min(len(weighted_events), max(self.max_dreams * 2, self.max_dreams))
            candidate_pool = weighted_events[:pool_size]
            # Prepare for weighted random selection
            weights = [max(0.01, w) for w, _ in candidate_pool]  # Avoid zero weights
            events_only = [e for _, e in candidate_pool]
            # Randomly select up to max_dreams, weighted by weight, no repeats
            selected = []
            selected_indices = set()
            attempts = 0
            while len(selected) < self.max_dreams and attempts < pool_size * 3:
                idx = random.choices(range(len(events_only)), weights=weights, k=1)[0]
                if idx not in selected_indices:
                    selected.append(events_only[idx])
                    selected_indices.add(idx)
                attempts += 1
            # If not enough unique, fill with more from pool
            if len(selected) < self.max_dreams:
                for i, e in enumerate(events_only):
                    if i not in selected_indices:
                        selected.append(e)
                    if len(selected) == self.max_dreams:
                        break
            return selected
        except Exception as e:
            error_type = f"dreamer_score_and_select_dreams_error"
            context = {"function": "score_and_select_dreams"}
            self.error_manager.record_error(
                error=ScaffoldError(
                    message=f"Error in score_and_select_dreams: {str(e)}",
                    operation="score_and_select_dreams",
                    context=context
                ),
                error_type=error_type,
                context=context,
                stack_trace=traceback.format_exc()
            )
            Logger.get_instance().record_event(
                event_type=error_type,
                message=f"Error in score_and_select_dreams: {str(e)}",
                level="error",
                additional_info=context,
                stack_trace=traceback.format_exc()
            )
            return []

class DreamGenerator:
    """Generates dream events with narration and noise."""
    def __init__(
        self,
        config_manager: ConfigManager,
        error_manager: ErrorManager,
        narration_strategy: DreamNarrationStrategy = None
    ):
        self.config_manager = config_manager
        self.logger = Logger.get_instance()
        self.error_manager = error_manager
        self.narration_strategy = narration_strategy or SurrealNarrationStrategy()
        self.noise_level = config_manager.get("dreamer_config.noise_level")

    def apply_dream_noise(self, dream_event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random noise to dream event data and metadata."""
        dream_event = deepcopy(dream_event)
        ed = dream_event["event_data"]
        for key, value in ed.items():
            if isinstance(value, str) and random.random() < self.noise_level:
                words = value.split()
                if words:
                    random.shuffle(words)
                    if random.random() < self.noise_level:
                        words.insert(
                            random.randint(0, len(words)),
                            random.choice(["???", "dream", "echo", "phantom", "mist", "fragment"])
                        )
                    ed[key] = " ".join(words)
        meta = dream_event["metadata"]
        if "novelty" in meta:
            meta["novelty"] += random.uniform(-0.1, 0.1) * self.noise_level
        if "confidence" in meta:
            meta["confidence"] += random.uniform(-0.1, 0.1) * self.noise_level
        return dream_event

    def generate_dream_events(self, dream_candidates: List[Dict], generation_manager=None) -> List[Dict]:
        """Generate dream events with playoff-style memory pairings and narration, using only the top half of memories."""
        dreams = []
        now = datetime.now().isoformat()
        # Extract memory strings from each candidate
        memories = []
        for event in dream_candidates:
            mem = event.get('memory')
            if mem is None:
                mem = event.get('event_data', {}).get('memory')
            if mem is not None:
                memories.append(mem)
        n = len(memories)
        # Only use the top half (round up if odd)
        half = n // 2 if n % 2 == 0 else (n // 2) + 1
        top_half = memories[:half]
        # Playoff-style pairing within top_half
        pairs = []
        for i in range((half + 1) // 2):
            j = half - 1 - i
            pairs.append((top_half[i], top_half[j]))
        for idx, (mem1, mem2) in enumerate(pairs):
            event_data = {'memory1': mem1, 'memory2': mem2}
            dream_event = {
                "timestamp_iso": now,
                "event_type": "dream",
                "event_data": event_data,
                "metadata": {},
                "dreamed_from": "dream_pairing"
            }
            dream_event = self.apply_dream_noise(dream_event)
            narration = self.narration_strategy.narrate(dream_event, self.noise_level, generation_manager=generation_manager)
            dream_event["narration"] = narration
            dreams.append(dream_event)
        return dreams

class DreamAlbumGenerator:
    """Generates dream albums with musical structure and narration."""
    MAJOR_KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    MINOR_KEYS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    SCALES = ["major", "minor"]
    MAJOR_CHAIN = {
        "I":   {"I": 8, "I7": 2, "IV": 5, "V": 6, "V7": 4, "vi": 2},
        "I7":  {"IV": 8, "V": 4, "V7": 4},
        "ii":  {"V": 6, "V7": 6, "ii7": 2, "IV": 2},
        "ii7": {"V": 8, "V7": 4},
        "iii": {"vi": 6, "IV": 2, "iii7": 2},
        "iii7": {"vi": 6, "IV": 2},
        "IV":  {"I": 6, "I7": 2, "V": 6, "V7": 4, "ii": 2, "IV7": 2},
        "IV7": {"I": 6, "V": 4, "V7": 4},
        "V":   {"I": 8, "vi": 2, "V7": 4},
        "V7":  {"I": 10, "vi": 2},
        "vi":  {"IV": 4, "ii": 2, "V": 4, "V7": 2, "vi7": 2},
        "vi7": {"IV": 4, "ii": 2, "V": 4, "V7": 2},
        "vii°": {"I": 6, "I7": 2, "V": 2, "V7": 2},
    }
    MINOR_CHAIN = {
        "i":   {"i": 8, "i7": 2, "iv": 5, "V": 6, "V7": 4, "VI": 2},
        "i7":  {"iv": 8, "V": 4, "V7": 4},
        "ii°": {"V": 6, "V7": 6, "iiø7": 2, "iv": 2},
        "iiø7": {"V": 8, "V7": 4},
        "III": {"VI": 6, "iv": 2, "III7": 2},
        "III7": {"VI": 6, "iv": 2},
        "iv":  {"i": 6, "i7": 2, "V": 6, "V7": 4, "ii°": 2, "iv7": 2},
        "iv7": {"i": 6, "V": 4, "V7": 4},
        "V":   {"i": 8, "VI": 2, "V7": 4},
        "V7":  {"i": 10, "VI": 2},
        "VI":  {"iv": 4, "ii°": 2, "V": 4, "V7": 2, "VI7": 2},
        "VI7": {"iv": 4, "ii°": 2, "V": 4, "V7": 2},
        "vii°": {"i": 6, "i7": 2, "V": 2, "V7": 2},
    }
    # Dynamic chords per bar options and weights
    chords_per_bar_options = [1, 2, 3, 4, 6]
    chords_per_bar_weights = [0.1, 0.3, 0.2, 0.3, 0.1]

    def __init__(
        self,
        config_manager: ConfigManager,
        scribe_path: str,
        error_manager: ErrorManager,
        scribe_event_fn,
        state_manager=None
    ):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = Logger.get_instance()
        self.error_manager = error_manager
        self.scribe_event_fn = scribe_event_fn
        self.noise_level = config_manager.get("dreamer_config.noise_level")
        self.chord_functions = config_manager.get("dreamer_config.chord_functions")
        self.num_songs = config_manager.get("dreamer_config.num_songs_per_album")
        # Emotional color mapping for chords
        self.chord_emotional_colors = {
            "I": "home, resolution, peace",
            "I7": "nostalgic, bluesy, relaxed resolution",
            "ii": "gentle motion, hope, anticipation",
            "ii7": "yearning, gentle tension, movement",
            "iii": "longing, bittersweet, wistful",
            "iii7": "wistful, searching, unresolved",
            "IV": "uplift, openness, warmth",
            "IV7": "expansive, soulful, bright tension",
            "V": "tension, drive, expectation",
            "V7": "heightened tension, drama, anticipation",
            "vi": "melancholy, nostalgia, comfort",
            "vi7": "soft sadness, gentle closure",
            "vii°": "suspense, instability, anxiety",
            "i": "darkness, gravity, mystery",
            "i7": "deep reflection, somber warmth",
            "ii°": "fragility, uncertainty, tension",
            "iiø7": "fragile, mysterious, unresolved tension",
            "III": "triumph, surprise, brightness",
            "III7": "boldness, surprise, bright tension",
            "iv": "depth, reflection, sorrow",
            "iv7": "deep sorrow, soulful longing",
            "VI": "drama, distance, yearning",
            "VI7": "distant longing, dramatic color",
            "bVII": "freedom, escape, rebellion",
            # Add more as needed if new chords are introduced
        }
        self.state_manager = state_manager

    def _clean_title(self, text: str, max_length: int = 48) -> str:
        """Clean and truncate text for titles."""
        clean = text.strip().replace('\n', ' ').replace('\r', ' ')
        return clean[:max_length] + ('...' if len(clean) > max_length else '')

    def load_memories(self) -> List[str]:
        """Load memories from scribe journal."""
        palette = []
        try:
            with open(self.scribe_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if "memory" in entry:
                        palette.append(entry["memory"])
        except Exception as e:
            raise ScaffoldError(
                message=f"Failed to load scribe journal: {str(e)}",
                operation="load_scribe_journal",
                context={"scribe_path": self.scribe_path}
            )
        return palette

    def generate_section_narration(self, section, song, motif_memory, memories, generation_manager, previous_section=None):
        section_lyric_so_far = ""
        for bar in section.progression.progression:
            chord = bar["roman"]
            chord_info = section.key.get_chord(chord)
            chord_notes = chord_info["notes"]
            chord_quality = chord_info["quality"]
            # Get emotional color for this chord, fallback to chord_functions, else generic
            chord_desc = self.chord_emotional_colors.get(
                chord,
                self.chord_functions.get(chord, "mystery, ambiguity, dreamlike uncertainty")
            )
            prompt = (
                "Essential qualities:\n"
                f"  - You are composing a surreal dream inspired by the lyrics music. "
                f"  - The current section is in the {section.key.scale_type} scale, key {section.key.tonic}. "
                f"  - The tempo is {song.tempo} bpm.\n"
                f"  - The previous lines in this section are: '{section_lyric_so_far.strip()}'. "
                f"  - This bar corresponds to the '{chord}' chord ({'-'.join(chord_notes)}, {chord_quality}), which creates a feeling of {chord_desc}. "
                f"  - Here is a memory fragment to use: '{motif_memory}'. "
                "   - Write the next poetic line (under 15 words) that continues the section, blending the memory with the chord's feeling."
                "   - End your line with a / to indicate a stanza break, as in standard lyrics."
                "Key constraints:\n"
                "   - Output only your lyric line, no explanations or commentary.\n"
                "   - Do not ever reference songs or music directly.\n"
                "   - Do not refer to the specific key or scale or tempo.\n"
                "   - Do not include dialogue or meta-commentary.\n"
                "   - Must use first- or second-person perspective.\n"
            )
            next_line = generation_manager.generate_text(prompt, num_return_sequences=1)[0]
            section_lyric_so_far += next_line.strip() + "\n"
        return section_lyric_so_far.strip()

    def run_dream_album_cycle(self, generation_manager, memories: Optional[List[str]] = None, state_manager=None, abort_flag=None) -> Optional[Dict]:
        if state_manager is None:
            state_manager = self.state_manager
        logger = Logger.get_instance()
        logger.info("Starting dream album cycle generation...")
        import time
        start_time = time.perf_counter()
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        try:
            if memories is None:
                logger.debug("No memories provided, loading from scribe journal...")
                memories = self.load_memories()
            if not memories:
                logger.error("No memories available for dream album, aborting.")
                raise ScaffoldError(
                    message="No memories available for dream album",
                    operation="validate_memories",
                    context={}
                )
            motif = memories[0]
            album_title = self._clean_title(motif, max_length=48)
            album = []
            logger.info(f"Dream album title: {album_title}")
            total_songs = self.num_songs
            for song_idx in range(self.num_songs):
                if abort_flag and abort_flag[0]:
                    logger.info("Dream cycle aborted by user during album generation.")
                    break
                logger.info(f"Generating song {song_idx+1}/{self.num_songs}...")
                song = self.Song(self)  # Markov structure
                song_sections = []
                score_lines = [f"Key: {song.key} | Tempo: {song.get_tempo_desc()} | Time Sig: {song.time_signature}"]
                section_motifs = {}
                section_narrations = []
                for section in song.overall_sections:
                    logger.debug(f"Generating section '{section.name}' for song {song_idx+1}...")
                    if section.name not in section_motifs:
                        section_motifs[section.name] = memories[(song_idx + ord(section.name[0])) % len(memories)]
                    motif_memory = section_motifs[section.name]
                    # Generate narration fragments for the section
                    narration = self.generate_section_narration(section, song, motif_memory, memories, generation_manager)
                    section_narrations.append(narration)
                prog_roman = [" ".join(bar) for bar in section.progression.progression]
                prog_in_key = [" ".join(section.key.get_chord(c) for c in bar) for bar in section.progression.progression]
                bars = len(section.progression.progression)
                repeats = section.repeats
                label = f"Section {section.name} ({section.key.tonic}, {song.get_tempo()})"
                musical_details = {
                    "key": str(section.key),
                    "scale": section.key.scale_type,
                    "tempo": song.get_tempo_desc(),
                    "progression_roman": prog_roman,
                    "progression_in_key": prog_in_key,
                    "bars": bars,
                    "repeats": repeats
                }
                score_lines.append(f"{label}: ║: {' | '.join(prog_roman)} :║ x{repeats}")
                song_sections.append({
                    "label": label,
                    "narration": narration,
                    "musical_details": musical_details
                })
                song_title = self._clean_title(list(section_motifs.values())[0], max_length=48)
                # Update dreaming progress after each song
                if state_manager and hasattr(state_manager, 'set_dreaming_progress'):
                    state_manager.set_dreaming_progress((song_idx + 1) / total_songs)
            poetic_score = "\n".join(score_lines)
            # Each dreamN is a full section narration
            event_data = {
                f"dream{i+1}": section_narrations[i] for i in range(len(section_narrations))
            }
            event_data["musical_key"] = str(song.key)
            event_data["timestamp_unix"] = int(time.time())
            song_entry = {
                "dream_album_name": album_title,
                "dream_song_name": song_title,
                "musical_key": str(song.key),
                "timestamp_unix": int(time.time()),
            }
            # Add dreamN fields
            for i in range(len(section_narrations)):
                song_entry[f"dream{i+1}"] = section_narrations[i]
            album.append(song_entry)
            # Process with ScribeIngestionProcessor
            metadata = {"motif": motif, "album_title": album_title}
            processed = ScribeIngestionProcessor(log_paths=[self.scribe_path], logger=self.logger).process_entry({
                "event_type": "dream",
                "event_data": event_data,
                "metadata": metadata
            })
            self.scribe_event_fn(
                origin="dreamer",
                event_type="dream",
                event_data={"memory": processed["memory"]},
                source_metadata=metadata,
                session_id=None,
                timestamp=datetime.now()
            )
            logger.info(f"Dream album generated with {self.num_songs} songs.")
            return {
                "dream_album_name": album_title,
                "motif": motif,
                "songs": album
            }
        except Exception as e:
            logger.error(f"Exception during dream album cycle: {e}", exc_info=True)
            raise
        finally:
            end_time = time.perf_counter()
            end_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            elapsed = end_time - start_time
            stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            top_stats = stats[:5]  # Top 5 memory consumers
            logger.info(f"Dream album cycle completed in {elapsed:.2f} seconds.")
            logger.info("Top memory usage during cycle:")
            for stat in top_stats:
                logger.info(stat)
            if state_manager and hasattr(state_manager, 'set_dreaming_progress'):
                state_manager.set_dreaming_progress(1.0)

    def clear_scribe_memory_cache(self):
        """Clear the cached scribe memories (call if journal is updated)."""
        if hasattr(self, "_cached_scribe_memories"):
            del self._cached_scribe_memories

    def _get_scribe_memories(self):
        """Load all memory fragments from the scribe journal (cache for efficiency)."""
        if not hasattr(self, "_cached_scribe_memories"):
            self._cached_scribe_memories = []
            try:
                with open(self.scribe_path, "r") as f:
                    for line in f:
                        entry = json.loads(line)
                        if "memory" in entry:
                            self._cached_scribe_memories.append(entry["memory"])
            except Exception as e:
                Logger.get_instance().log_warning(f"Could not read scribe journal for noise: {e}")
        return self._cached_scribe_memories

    def _deterministic_noise_fragment(self, context: str, memories: list) -> str:
        """Select a short phrase (2–5 words) from scribe memories deterministically based on context."""
        if not memories:
            return ""
        h = int(hashlib.sha256(context.encode()).hexdigest(), 16)
        memory = memories[h % len(memories)]
        words = memory.split()
        if not words:
            return ""
        if len(words) <= 2:
            return " ".join(words)
        start = h % max(1, len(words) - 2)
        length = 2 + (h % 4)  # 2–5 words
        end = min(len(words), start + length)
        return " ".join(words[start:end])

    def apply_dream_noise(self, dream_event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply deterministic, scribe-journal-based noise to dream event narration."""
        dream_event = deepcopy(dream_event)
        ed = dream_event["event_data"]
        memories = self._get_scribe_memories()
        for key, value in ed.items():
            if isinstance(value, str) and random.random() < self.noise_level and memories:
                words = value.split()
                if words:
                    context = f"{key}:{value}"
                    noise_phrase = self._deterministic_noise_fragment(context, memories)
                    if noise_phrase:
                        # Try to insert after a comma or period if present, else random position
                        insert_pos = None
                        for i, w in enumerate(words):
                            if w.endswith((',', '.')):
                                insert_pos = i + 1
                                break
                        if insert_pos is None:
                            insert_pos = random.randint(0, len(words))
                        words.insert(insert_pos, noise_phrase)
                        ed[key] = " ".join(words)
        return dream_event

    def _generate_key(self):
        scale = random.choice(self.SCALES)
        key = random.choice(self.MAJOR_KEYS if scale == "major" else self.MINOR_KEYS)
        return scale, key

    def _modulate_key(self, scale, key, modulation_type):
        major_keys = self.MAJOR_KEYS
        minor_keys = self.MINOR_KEYS
        idx = (major_keys if scale == "major" else minor_keys).index(key)
        if modulation_type == "parallel":
            new_scale = "minor" if scale == "major" else "major"
            new_key = key
        elif modulation_type == "relative":
            if scale == "major":
                new_scale = "minor"
                new_idx = (idx + 9) % 12  # Down 3 semitones
                new_key = minor_keys[new_idx]
            else:
                new_scale = "major"
                new_idx = (idx + 3) % 12  # Up 3 semitones
                new_key = major_keys[new_idx]
        elif modulation_type == "neighbour":
            # Circle of fifths neighbour (±1)
            new_scale = scale
            new_key = (major_keys if scale == "major" else minor_keys)[(idx + random.choice([-1, 1])) % 12]
        elif modulation_type == "near":
            # Circle of fifths ±2 or ±3
            new_scale = scale
            new_key = (major_keys if scale == "major" else minor_keys)[(idx + random.choice([-3, -2, 2, 3])) % 12]
        elif modulation_type == "foreign":
            new_scale = random.choice(self.SCALES)
            new_key = random.choice(major_keys if new_scale == "major" else minor_keys)
        elif modulation_type == "truck_driver":
            # Shift up or down by 1 or 2 semitones
            shift = random.choice([-2, -1, 1, 2])
            new_scale = scale
            new_key = (major_keys if scale == "major" else minor_keys)[(idx + shift) % 12]
        else:
            new_scale, new_key = scale, key
        return new_scale, new_key, modulation_type

    def _assign_section_keys(self, section_names, home_scale, home_key):
        section_key_map = {}
        for name in set(section_names):
            if random.randint(1, 8) == 1:
                modulation_type = random.choice(["parallel", "relative", "neighbour", "near", "foreign"])
                scale, key, mod_type = self._modulate_key(home_scale, home_key, modulation_type)
                section_key_map[name] = (scale, key, mod_type)
            else:
                section_key_map[name] = (home_scale, home_key, "home")
        return section_key_map

    def _generate_sections(self, song):
        section_count = random.randint(3, 7)
        base_names = ["A", "B", "C", "D", "E", "F", "G"]
        section_names = ["A"]
        for _ in range(1, section_count):
            available = base_names[:min(1 + len(section_names), len(base_names))]
            if len(available) == 1:
                next_name = "A"
            else:
                weights = [0.4] + [0.6 / (len(available) - 1)] * (len(available) - 1)
                next_name = random.choices(available, weights=weights)[0]
            section_names.append(next_name)
        # Assign keys to sections, modulating as needed
        section_key_map = {}
        home_key = song.key
        for name in set(section_names):
            if name == "A":
                section_key_map[name] = (home_key, "home")
            else:
                # 1 in 8 chance to modulate
                if random.randint(1, 8) == 1:
                    mod_type = random.choice(["parallel", "relative_minor", "relative_major", "neighbour_up", "neighbour_down", "foreign"])
                    mod_key = home_key.modulate(mod_type)
                    section_key_map[name] = (mod_key, mod_type)
                else:
                    section_key_map[name] = (home_key, "home")
        # Create sections
        sections = [self.Section(name, section_key_map[name][0], self, scale=section_key_map[name][0].scale_type, modulation_type=section_key_map[name][1]) for name in section_names]
        # Insert pivot chords at modulation points
        for i in range(1, len(sections)):
            prev_section = sections[i-1]
            curr_section = sections[i]
            if prev_section.key.tonic != curr_section.key.tonic or prev_section.key.scale_type != curr_section.key.scale_type:
                pivots = prev_section.key.suggest_pivot_chords(curr_section.key)
                if pivots:
                    pivot_chord = random.choice(pivots)
                    # Append pivot chord to end of previous section's progression
                    prev_section.progression.progression.append({
                        "roman": pivot_chord,
                        **prev_section.key.get_chord(pivot_chord),
                        "is_pivot": True
                    })
        return sections

    def _generate_progression(self, section_name, scale="major", key="C", bar_count=None, section=None):
        # All logic is derived from the section's actual key/scale
        if section is not None:
            scale_type = section.key.scale_type
        else:
            scale_type = scale
        chain = self.MAJOR_CHAIN if scale_type == "major" else self.MINOR_CHAIN
        possible_starts = list(chain.keys())
        # Weighted random choice for starting chord: favor tonic and predominant chords
        if scale_type == "major":
            weights = [10 if chord == "I" else 5 if chord in ("vi", "IV") else 2 if chord in ("V", "ii") else 1 for chord in possible_starts]
        else:
            weights = [10 if chord == "i" else 5 if chord in ("VI", "iv") else 2 if chord in ("V", "ii°") else 1 for chord in possible_starts]
        start_chord = random.choices(possible_starts, weights=weights, k=1)[0]
        # Determine bar count if not provided
        if bar_count is None:
            bar_count = random.choices([2, 4, 6], weights=[0.5, 0.4, 0.1])[0]
        # Choose chords_per_bar dynamically per section
        chords_per_bar = random.choices(self.chords_per_bar_options, weights=self.chords_per_bar_weights)[0]
        total_chords = bar_count * chords_per_bar
        current = start_chord
        progression = [current]
        for _ in range(total_chords - 1):
            next_chords = chain.get(current, {})
            if not next_chords:
                current = start_chord  # fallback to section's start chord
            else:
                chords, weights_ = zip(*next_chords.items())
                current = random.choices(chords, weights=weights_)[0]
            progression.append(current)
        # Split into bars
        return [progression[i:i+chords_per_bar] for i in range(0, total_chords, chords_per_bar)]

    class Song:
        """Represents a song with musical structure."""
        def __init__(self, album_generator):
            # Home key and scale as Key object
            self.key = Key(album_generator._generate_key(), album_generator._generate_scale())
            self.time_signature = album_generator._generate_time_signature()
            self.tempo = album_generator._generate_bpm()
            self.sections = album_generator._generate_sections(self)
            self.overall_sections = self.sections
        def get_tempo(self):
            return self.tempo
        def get_tempo_desc(self):
            return f"{self.tempo} bpm"
    class Section:
        """Represents a section of a song (A, B, etc.)."""
        def __init__(self, name, key, album_generator, scale="major", modulation_type="home"):
            self.name = name
            # Key object for this section (modulated if needed)
            self.key = key if isinstance(key, Key) else Key(key, scale)
            self.scale = self.key.scale_type
            self.modulation_type = modulation_type
            self.progression = album_generator.Progression(album_generator, self.key, self.name)
            self.repeats = random.choice([1, 2, 4])
    class Progression:
        """Represents a chord progression for a section."""
        def __init__(self, album_generator, key, section_name):
            # Use Markov chain logic to generate Roman numerals, passing section for key/scale context
            roman_prog = album_generator._generate_progression(section_name, key.scale_type, key.tonic, section=self)
            # For each chord, get notes and quality from Key
            self.progression = [
                {
                    "roman": roman,
                    **key.get_chord(roman)
                }
                for bar in roman_prog for roman in bar
            ]
            self.key = key

class Dreamer:
    """Dream system for SOVL: generates and logs dream events and albums."""
    def __init__(
        self,
        config_manager: ConfigManager,
        scribe_path: str,
        scribe_event_fn=capture_scribe_event,
        error_manager: ErrorManager = None,
        state_manager=None
    ):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = Logger.get_instance()
        self.scribe_event_fn = scribe_event_fn
        self.error_manager = error_manager or ErrorManager(
            context=self,
            state_tracker=None,
            config_manager=config_manager,
            error_cooldown=1.0
        )
        self.event_selector = DreamEventSelector(config_manager, scribe_path, self.error_manager)
        self.dream_generator = DreamGenerator(config_manager, self.error_manager, SurrealNarrationStrategy())
        self.album_generator = DreamAlbumGenerator(config_manager, scribe_path, self.error_manager, scribe_event_fn, state_manager=state_manager)
        self.scribe_ingestion_processor = ScribeIngestionProcessor(log_paths=[scribe_path], logger=self.logger)
        self.state_manager = state_manager
        # Register recovery strategy
        self.error_manager.recovery_strategies["dreamer_queue_error"] = self._recover_queue_failure

    def run_dream_cycle(self, generation_manager=None, state_manager=None, abort_flag=None) -> None:
        if state_manager is None:
            state_manager = self.state_manager
        if state_manager and hasattr(state_manager, 'set_dreaming_progress'):
            state_manager.set_dreaming_progress(0.0)
        events = self.event_selector.extract_last_active_period()
        candidates = self.event_selector.score_and_select_dreams(events)
        dreams = self.dream_generator.generate_dream_events(candidates, generation_manager=generation_manager)
        for idx, dream in enumerate(dreams):
            if abort_flag and abort_flag[0]:
                self.logger.info("Dream cycle aborted by user during dream event generation.")
                break
            self.log_dreams([dream])
            if state_manager and hasattr(state_manager, 'set_dreaming_progress'):
                state_manager.set_dreaming_progress((idx + 1) / len(dreams))
        if state_manager and hasattr(state_manager, 'set_dreaming_progress'):
            state_manager.set_dreaming_progress(1.0)
        if state_manager and hasattr(state_manager, 'set_mode'):
            state_manager.set_mode('online')

    def log_dreams(self, dreams: List[Dict]) -> None:
        """Log dream events to scribe queue."""
        for dream in dreams:
            event_data = {
                "dream1": dream.get("narration", ""),
                "musical_key": dream.get("musical_key", "unknown"),
                "timestamp_unix": int(time.time())
            }
            source_metadata = {
                "dreamed_from": dream.get("dreamed_from", "unknown"),
                "timestamp_iso": dream.get("timestamp_iso", ""),
                **dream.get("metadata", {})
            }
            processed = self.scribe_ingestion_processor.process_entry({
                "event_type": "dream",
                "event_data": event_data,
                "metadata": source_metadata
            })
            self.scribe_event_fn(
                origin="dreamer",
                event_type="dream",
                event_data={"memory": processed["memory"]},
                source_metadata=source_metadata,
                session_id=dream.get("session_id"),
                timestamp=datetime.fromisoformat(dream.get("timestamp_iso", datetime.now().isoformat()))
            )
            self.logger.record_event(
                event_type="dream_log",
                message=f"Dream event logged: {processed['memory']}",
                level="info"
            )

    def run_dream_album_cycle(self, generation_manager, memories: Optional[List[str]] = None, state_manager=None, abort_flag=None) -> Optional[Dict]:
        return self.album_generator.run_dream_album_cycle(generation_manager, memories, state_manager, abort_flag)

    def _recover_queue_failure(self, record):
        """Recovery strategy for scribe queue failures."""
        try:
            self.logger.record_event(
                event_type="dreamer_recovery",
                message="Retrying scribe queue after failure",
                level="info",
                additional_info={"error_type": record.error_type}
            )
            queue_instance = get_scribe_queue()
            queue_instance.clear(caller="dreamer_recovery", confirm=True)
            self.scribe_event_fn(**record.context["entry"])
        except Exception as e:
            self.error_manager.record_error(
                error=ScaffoldError(
                    message=f"Recovery failed: {str(e)}",
                    operation="queue_recovery",
                    context={"original_error": record.error_type}
                ),
                error_type="dreamer_recovery_error",
                context={"original_error": record.error_type}
            )

    @classmethod
    def cli_run_dream(
        cls,
        config_path: str,
        scribe_path: str,
        generation_manager: Optional[Any] = None,
        **kwargs
    ) -> Optional[Dict]:
        """
        CLI-friendly entry point to generate a dream cycle.
        """
        logger = Logger.get_instance()
        logger.info("[CLI] Starting dream cycle generation...")
        try:
            config_manager = ConfigManager(config_path)
            error_manager = ErrorManager(context=None, state_tracker=None, config_manager=config_manager)
            state_manager = kwargs.get('state_manager', None)
            abort_flag = kwargs.get('abort_flag', None)
            dreamer = cls(
                config_manager=config_manager,
                scribe_path=scribe_path,
                error_manager=error_manager,
                state_manager=state_manager
            )
            if generation_manager is None:
                try:
                    from sovl_generate import GenerationManager
                    generation_manager = GenerationManager(config_manager=config_manager, logger=logger)
                except ImportError:
                    logger.error("Could not import GenerationManager. Please provide a generation_manager.")
                    return None
            result = dreamer.run_dream_cycle(generation_manager=generation_manager, state_manager=state_manager, abort_flag=abort_flag)
            logger.info("[CLI] Dream cycle generation complete.")
            return result
        except Exception as e:
            logger.error(f"[CLI] Exception during dream cycle generation: {e}", exc_info=True)
            return None

class Key:
    """
    Represents a musical key and scale, with methods for chord and modulation logic.
    Usage:
        key = Key("C", "major")
        print(key.notes)  # ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        print(key.get_chord("IV"))  # {'notes': ['F', 'A', 'C'], 'quality': '', 'roman': 'IV'}
        mod_key = key.modulate("relative_minor")
        print(mod_key.tonic, mod_key.scale_type)  # 'A', 'minor'
    """
    MAJOR_INTERVALS = [2, 2, 1, 2, 2, 2, 1]
    MINOR_INTERVALS = [2, 1, 2, 2, 1, 2, 2]
    NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    NOTES_FLAT =  ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    ROMAN_TO_DEGREE = {
        "I": 0, "ii": 1, "iii": 2, "IV": 3, "V": 4, "vi": 5, "vii": 6,
        "i": 0, "ii°": 1, "III": 2, "iv": 3, "V": 4, "VI": 5, "vii°": 6
    }
    CHORD_QUALITIES = {
        "major": ["", "m", "m", "", "", "m", "dim"],
        "minor": ["m", "dim", "", "m", "", "", ""],
    }
    _scale_cache = {}
    _chord_cache = {}
    def __init__(self, tonic, scale_type, accidental_style=None):
        """
        tonic: root note (e.g., 'C', 'F#')
        scale_type: 'major' or 'minor'
        accidental_style: 'sharps', 'flats', or None for auto
        """
        if scale_type not in ("major", "minor"):
            raise ValueError(f"Invalid scale_type: {scale_type}")
        if tonic not in self.NOTES_SHARP and tonic not in self.NOTES_FLAT:
            raise ValueError(f"Invalid tonic: {tonic}")
        self.tonic = tonic
        self.scale_type = scale_type
        self.accidental_style = accidental_style
        self.notes = self._get_scale_notes()
    def _get_scale_notes(self):
        """Return the list of notes in the key/scale, using caching."""
        cache_key = (self.tonic, self.scale_type, self.accidental_style)
        if cache_key in self._scale_cache:
            return self._scale_cache[cache_key]
        if self.accidental_style == "sharps":
            notes = self.NOTES_SHARP
        elif self.accidental_style == "flats":
            notes = self.NOTES_FLAT
        else:
            notes = self.NOTES_SHARP if "#" in self.tonic or self.tonic == "F#" else self.NOTES_FLAT
        idx = notes.index(self.tonic)
        intervals = self.MAJOR_INTERVALS if self.scale_type == "major" else self.MINOR_INTERVALS
        scale = [notes[idx]]
        for step in intervals:
            idx = (idx + step) % 12
            scale.append(notes[idx])
        result = scale[:-1]  # 7 notes
        self._scale_cache[cache_key] = result
        return result
    def get_chord(self, roman):
        """Return chord notes, quality, and roman for a given Roman numeral (supports 7th chords)."""
        cache_key = (self.tonic, self.scale_type, roman)
        if cache_key in self._chord_cache:
            return self._chord_cache[cache_key]
        base_roman = roman.replace("7", "").replace("°", "").replace("ø", "")
        if base_roman not in self.ROMAN_TO_DEGREE:
            raise ValueError(f"Invalid Roman numeral: {roman}")
        degree = self.ROMAN_TO_DEGREE[base_roman]
        root = self.notes[degree]
        third = self.notes[(degree + 2) % 7]
        fifth = self.notes[(degree + 4) % 7]
        notes = [root, third, fifth]
        quality = self._get_chord_quality(roman, degree)
        if "7" in roman:
            seventh = self.notes[(degree + 6) % 7]
            notes.append(seventh)
        result = {"notes": notes, "quality": quality, "roman": roman}
        self._chord_cache[cache_key] = result
        return result
    def _get_chord_quality(self, roman, degree):
        """Return the chord quality string for a given Roman numeral and degree."""
        if "7" in roman:
            if "M" in roman:
                return "M7"
            elif "ø" in roman:
                return "m7b5"
            elif "°" in roman:
                return "dim7"
            elif roman.islower():
                return "m7"
            else:
                return "7"
        if self.scale_type == "major":
            return self.CHORD_QUALITIES["major"][degree]
        else:
            return self.CHORD_QUALITIES["minor"][degree]
    def get_scale_type(self):
        """Return the scale type ('major' or 'minor')."""
        return self.scale_type
    def get_scale_degree(self, note):
        """Return the scale degree (1-based) of a note in the key, or None if not present."""
        try:
            return self.notes.index(note) + 1
        except ValueError:
            return None
    def modulate(self, modulation_type):
        """Return a new Key object modulated as specified."""
        idx = self.NOTES_SHARP.index(self.tonic) if self.tonic in self.NOTES_SHARP else self.NOTES_FLAT.index(self.tonic)
        if modulation_type == "parallel":
            return Key(self.tonic, "minor" if self.scale_type == "major" else "major", self.accidental_style)
        elif modulation_type == "relative_minor" and self.scale_type == "major":
            new_idx = (idx + 9) % 12  # Down 3 semitones
            return Key(self.NOTES_SHARP[new_idx], "minor", self.accidental_style)
        elif modulation_type == "relative_major" and self.scale_type == "minor":
            new_idx = (idx + 3) % 12  # Up 3 semitones
            return Key(self.NOTES_SHARP[new_idx], "major", self.accidental_style)
        elif modulation_type == "neighbour_up":
            new_idx = (idx + 7) % 12
            return Key(self.NOTES_SHARP[new_idx], self.scale_type, self.accidental_style)
        elif modulation_type == "neighbour_down":
            new_idx = (idx - 7) % 12
            return Key(self.NOTES_SHARP[new_idx], self.scale_type, self.accidental_style)
        elif modulation_type == "foreign":
            import random
            return Key(random.choice(self.NOTES_SHARP), self.scale_type, self.accidental_style)
        else:
            return self  # fallback
    def suggest_pivot_chords(self, other_key):
        """Suggest possible pivot chords for modulation to another key (triads and 7ths)."""
        pivots = []
        # Consider both triads and 7th chords
        roman_numerals = list(self.ROMAN_TO_DEGREE.keys())
        for roman in roman_numerals:
            try:
                chord_self = set(self.get_chord(roman)['notes'])
                chord_other = set(other_key.get_chord(roman)['notes'])
                # Check for exact match (triad or 7th)
                if chord_self == chord_other and len(chord_self) >= 3:
                    pivots.append(roman)
                # Allow subset match (e.g., triad in one, 7th in other)
                elif chord_self.issubset(chord_other) or chord_other.issubset(chord_self):
                    pivots.append(roman)
            except Exception:
                continue
        # Optionally, sort pivots to prefer tonic, predominant, dominant
        def pivot_priority(roman):
            if roman in ("I", "i"): return 0
            if roman in ("IV", "iv", "ii", "ii°"): return 1
            if roman in ("V", "V7", "v", "vii°"): return 2
            return 3
        pivots = sorted(set(pivots), key=pivot_priority)
        return pivots
    def __str__(self):
        return f"Key({self.tonic} {self.scale_type})"
    def __repr__(self):
        return f"Key(tonic={self.tonic!r}, scale_type={self.scale_type!r}, notes={self.notes!r})"