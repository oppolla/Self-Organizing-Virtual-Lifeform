import json
import random
import time
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Any
from functools import wraps
from copy import deepcopy

from sovl_io import JSONLLoader
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager, ScaffoldError
from sovl_queue import get_scribe_queue, capture_scribe_event
from sovl_processor import ScribeIngestionProcessor
from sovl_schema import ConfigSchema

"""
Handles all dream generation, dream structure, narration, and scribe journal integration for SOVL
"""

# Configuration schema for dream_memory_config
DREAM_MEMORY_CONFIG_SCHEMA = [
    ConfigSchema(
        field="dream_memory_config.max_events_per_cycle",
        type=int,
        default=5,
        range=(1, 100),
        required=True
    ),
    ConfigSchema(
        field="dream_memory_config.novelty_weight",
        type=float,
        default=1.0,
        range=(0.0, 10.0),
        required=True
    ),
    ConfigSchema(
        field="dream_memory_config.confidence_weight",
        type=float,
        default=0.0,
        range=(0.0, 10.0),
        required=True
    ),
    ConfigSchema(
        field="dream_memory_config.selection_strategy",
        type=str,
        default="top",
        validator=lambda x: x in ["top", "random"],
        required=True
    ),
    ConfigSchema(
        field="dream_memory_config.noise_level",
        type=float,
        default=0.2,
        range=(0.0, 1.0),
        required=True
    ),
    ConfigSchema(
        field="dream_memory_config.num_songs_per_album",
        type=int,
        default=3,
        range=(1, 10),
        required=True
    ),
    ConfigSchema(
        field="dream_memory_config.chord_functions",
        type=dict,
        default={
            "I": "home and safety",
            "IV": "expansion and openness",
            "V": "tension and anticipation",
            "vi": "nostalgia and longing",
            "ii": "subtle unease",
            "vii°": "instability and surrealism"
        },
        required=True
    )
]

# Error handling decorator
def handle_dreamer_errors(logger: Logger, error_manager: ErrorManager):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = f"dreamer_{func.__name__}_error"
                context = {"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                error_manager.record_error(
                    error=ScaffoldError(
                        message=f"Error in {func.__name__}: {str(e)}",
                        operation=func.__name__,
                        context=context
                    ),
                    error_type=error_type,
                    context=context,
                    stack_trace=traceback.format_exc()
                )
                logger.record_event(
                    event_type=error_type,
                    message=f"Error in {func.__name__}: {str(e)}",
                    level="error",
                    additional_info=context,
                    stack_trace=traceback.format_exc()
                )
                return None
        return wrapper
    return decorator

class DreamNarrationStrategy:
    """Base class for dream narration strategies."""
    def narrate(self, dream_event: Dict[str, Any], noise_level: float) -> str:
        raise NotImplementedError

class SurrealNarrationStrategy(DreamNarrationStrategy):
    """Surreal narration blending prompt and response with dream-like phrases."""
    def narrate(self, dream_event: Dict[str, Any], noise_level: float) -> str:
        prompt = dream_event["event_data"].get("prompt", "")
        response = dream_event["event_data"].get("response", "")
        parts = [prompt, response]
        random.shuffle(parts)
        narration = f"In the midst of swirling thoughts, a dream emerged: {parts[0]} ... Suddenly, {parts[1]} ... The boundaries of meaning blurred."
        if random.random() < noise_level:
            narration += f" {random.choice(['A phantom word echoed.', 'Mist enveloped the memory.', 'Fragments danced in the void.'])}"
        return narration.strip()

class DreamEventSelector:
    """Handles event extraction and selection for dream generation."""
    def __init__(self, config_manager: ConfigManager, scribe_path: str, error_manager: ErrorManager):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = Logger.get_instance()
        self.error_manager = error_manager
        self.max_dreams = config_manager.get("dream_memory_config.max_events_per_cycle")
        self.novelty_weight = config_manager.get("dream_memory_config.novelty_weight")
        self.confidence_weight = config_manager.get("dream_memory_config.confidence_weight")
        self.selection_strategy = config_manager.get("dream_memory_config.selection_strategy")

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
        """Score events by novelty/confidence and select candidates."""
        try:
            scored = []
            for event in events:
                meta = event.get("metadata", {})
                novelty = meta.get("novelty", 0.0)
                confidence = meta.get("confidence", 1.0)
                score = self.novelty_weight * novelty - self.confidence_weight * confidence
                scored.append((score, event))
            scored.sort(reverse=True, key=lambda x: x[0])
            if self.selection_strategy == "top":
                return [e for _, e in scored[:self.max_dreams]]
            elif self.selection_strategy == "random":
                return [e for _, e in random.sample(scored, min(self.max_dreams, len(scored)))]
            return [e for _, e in scored[:self.max_dreams]]
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
        self.noise_level = config_manager.get("dream_memory_config.noise_level")

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

    def generate_dream_events(self, dream_candidates: List[Dict]) -> List[Dict]:
        """Generate dream events with narration."""
        dreams = []
        now = datetime.now().isoformat()
        for event in dream_candidates:
            dream_event = {
                "timestamp_iso": now,
                "event_type": "dream",
                "event_data": event.get("event_data", {}),
                "metadata": event.get("metadata", {}),
                "dreamed_from": event.get("event_type", "unknown")
            }
            dream_event = self.apply_dream_noise(dream_event)
            narration = self.narration_strategy.narrate(dream_event, self.noise_level)
            dream_event["narration"] = narration
            dreams.append(dream_event)
        return dreams

class DreamAlbumGenerator:
    """Generates dream albums with musical structure and narration."""
    class Song:
        """Represents a song with musical structure."""
        def __init__(self, album_generator):
            self.key = album_generator._generate_key()
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
        def __init__(self, name, key, album_generator):
            self.name = name
            self.key = key
            self.progression = album_generator.Progression(album_generator)
            self.repeats = random.choice([1, 2, 4])
    class Progression:
        """Represents a chord progression for a section."""
        def __init__(self, album_generator):
            self.progression = album_generator._generate_progression()
    def __init__(
        self,
        config_manager: ConfigManager,
        scribe_path: str,
        error_manager: ErrorManager,
        scribe_event_fn
    ):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = Logger.get_instance()
        self.error_manager = error_manager
        self.scribe_event_fn = scribe_event_fn
        self.noise_level = config_manager.get("dream_memory_config.noise_level")
        self.chord_functions = config_manager.get("dream_memory_config.chord_functions")
        self.num_songs = config_manager.get("dream_memory_config.num_songs_per_album")

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

    def generate_section_narration(
        self,
        section,
        song: Song,
        motif_memory: str,
        memories: List[str],
        generation_manager
    ) -> str:
        """Generate narration for a song section, batching LLM calls."""
        prompts = []
        for bar in section.progression.progression:
            for chord in bar:
                memory = motif_memory if random.random() < 0.5 else random.choice(memories)
                chord_desc = self.chord_functions.get(chord, "mystery and transformation")
                prompt = (
                    "Essential qualities:\n"
                    f"  - You are composing a surreal dream inspired by the lyrics music. "
                    f"  - The current section is in the {section.key.scale} scale, key {section.key}. Let the musicality guide your writing."
                    f"  - This moment corresponds to the '{chord}' chord, which creates a feeling of {chord_desc}. "
                    f"  - The tempo is {song.tempo} bpm. Let this tempo influence the energy and rhythm of your lyric.\n"
                    f"  - Here is a memory fragment to use: '{memory}'. "
                     "  - Format your words as if they are song lyrics. Follow lyric conventions closely. "
                     "  - Let the musical indicators like key and tempo and modulation to guide your journey and shape your flow. "
                    f"  - Write 50 word single, vivid, poetic sentence that blends the memory with the chord's feeling."
                    "Key constraints:\n"
                     "   - Output only your lyric, no explanations or commentary.\n"
                     "   - Do not ever reference songs or music directly.\n"
                     "   - Do not refer to the specific key or scale or tempo. Only allow it to guide the tone of your lyrics\n"
                     "   - Do not include dialogue or meta-commentary.\n"
                     "   - Keep it to a single poetic sentence, under 50 words.\n"
                     "   - Must use first- or second-person perspective.\n"
                )
                prompts.append(prompt)
        try:
            fragments = generation_manager.generate_text(prompts, num_return_sequences=1)
        except Exception as e:
            fragments = [motif_memory] * len(prompts)
        # Apply noise to fragments
        fragments = [
            self.apply_dream_noise({"event_data": {"prompt": f}, "metadata": {}})["event_data"]["prompt"]
            for f in fragments
        ]
        return " ".join(fragments)

    def run_dream_album_cycle(self, generation_manager, memories: Optional[List[str]] = None) -> Optional[Dict]:
        """Generate a dream album with songs and sections."""
        if memories is None:
            memories = self.load_memories()
        if not memories:
            raise ScaffoldError(
                message="No memories available for dream album",
                operation="validate_memories",
                context={}
            )
        motif = memories[0]
        album_title = self._clean_title(motif, max_length=48)
        album = []
        for song_idx in range(self.num_songs):
            song = self.Song(self)  # Markov structure
            song_sections = []
            score_lines = [f"Key: {song.key} | Tempo: {song.get_tempo_desc()} | Time Sig: {song.time_signature}"]
            section_motifs = {}
            section_narrations = []
            for section in song.overall_sections:
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
                label = f"Section {section.name} ({section.key}, {song.get_tempo()})"
                musical_details = {
                    "key": str(section.key),
                    "scale": section.key.scale,
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
            poetic_score = "\n".join(score_lines)
            # Each dreamN is a full section narration
            event_data = {
                f"dream{i+1}": section_narrations[i] for i in range(min(len(section_narrations), 12))
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
            for i in range(min(len(section_narrations), 12)):
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
        self.logger.record_event(
            event_type="dream_album_generated",
            message=f"Dream album generated with {self.num_songs} songs",
            level="info",
            additional_info={"album_title": album_title}
        )
        return {
            "dream_album_name": album_title,
            "motif": motif,
            "songs": album
        }

    def apply_dream_noise(self, dream_event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise to dream event (used in narration)."""
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
        return dream_event

    def _generate_key(self):
        # Placeholder: return a random key
        return random.choice(["C", "G", "D", "A", "E", "F", "Bb"])

    def _generate_time_signature(self):
        return random.choice(["4/4", "3/4"])

    def _generate_bpm(self):
        return random.randint(60, 140)

    def _generate_sections(self, song):
        section_names = ["A", "B", "C"]
        return [self.Section(name, song.key, self) for name in section_names]

    def _generate_progression(self):
        # Placeholder: return a list of lists of chords
        chords = ["I", "IV", "V", "vi", "ii"]
        return [[random.choice(chords) for _ in range(4)] for _ in range(2)]

class Dreamer:
    """Dream system for SOVL: generates and logs dream events and albums."""
    def __init__(
        self,
        config_manager: ConfigManager,
        scribe_path: str,
        scribe_event_fn=capture_scribe_event,
        error_manager: ErrorManager = None
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
        self.config_manager.register_schema(DREAM_MEMORY_CONFIG_SCHEMA)
        self.event_selector = DreamEventSelector(config_manager, scribe_path, self.error_manager)
        self.dream_generator = DreamGenerator(config_manager, self.error_manager, SurrealNarrationStrategy())
        self.album_generator = DreamAlbumGenerator(config_manager, scribe_path, self.error_manager, scribe_event_fn)
        self.scribe_ingestion_processor = ScribeIngestionProcessor(log_paths=[scribe_path], logger=self.logger)
        # Register recovery strategy
        self.error_manager.recovery_strategies["dreamer_queue_error"] = self._recover_queue_failure

    def run_dream_cycle(self) -> None:
        """Extract, select, generate, and log dream events."""
        events = self.event_selector.extract_last_active_period()
        candidates = self.event_selector.score_and_select_dreams(events)
        dreams = self.dream_generator.generate_dream_events(candidates)
        self.log_dreams(dreams)

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

    def run_dream_album_cycle(self, generation_manager, memories: Optional[List[str]] = None) -> Optional[Dict]:
        """Generate and log a dream album."""
        return self.album_generator.run_dream_album_cycle(generation_manager, memories)

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