import json
import random
import traceback
from datetime import datetime
from sovl_io import JSONLLoader
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager
from sovl_queue import capture_scribe_event
import string
from formula import *  # For Markov music structure
import music_def as mus
from sovl_processor import ScribeIngestionProcessor, MetadataProcessor
import time

class Dreamer:
    """
    Dream system for SOVL: selects, generates, and logs dream events from the last active period.
    Each dream is a surreal narration, with optional dream noise for creativity.
    """
    def __init__(self, config_manager, scribe_path, logger, metadata_processor, scribe_event_fn, error_manager=None):
        self.config_manager = config_manager
        self.scribe_path = scribe_path
        self.logger = logger
        self.metadata_processor = metadata_processor
        self.scribe_event_fn = scribe_event_fn  # Function to log a scribe event (e.g., capture_scribe_event)
        self.error_manager = error_manager
        # Configurable dream parameters
        self.max_dreams = config_manager.get("dream_max_events_per_cycle", 5)
        self.novelty_weight = config_manager.get("dream_novelty_weight", 1.0)
        self.confidence_weight = config_manager.get("dream_confidence_weight", 0.0)
        self.selection_strategy = config_manager.get("dream_selection_strategy", "top")
        self.dream_noise_level = config_manager.get("dream_noise_level", 0.2)
        # Add ScribeIngestionProcessor for final memory/weight formatting
        self.scribe_ingestion_processor = ScribeIngestionProcessor(log_paths=[], logger=self.logger)

    def extract_last_active_period(self):
        """
        Extract events from the last active period (since last 'wake' event).
        Returns: List of scribe log event dicts.
        """
        events = []
        try:
            loader = JSONLLoader(self.config_manager, self.logger, self.error_manager) if self.error_manager else JSONLLoader(self.config_manager, self.logger, None)

            # First pass: Find the index of the last 'wake' event
            last_wake_idx = None
            current_idx = 0
            for entry in loader.stream_jsonl(self.scribe_path):
                if entry.get("event_type") == "wake":
                    last_wake_idx = current_idx
                current_idx += 1

            # Second pass: Collect events from last_wake_idx (or start) using streaming
            current_idx = 0
            for entry in loader.stream_jsonl(self.scribe_path):
                if last_wake_idx is None or current_idx > last_wake_idx:
                    events.append(entry)
                current_idx += 1

        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    f"Failed to stream scribe log with JSONLLoader: {str(e)}",
                    error_type="scribe_stream_error",
                    stack_trace=traceback.format_exc()
                )

        return events

    def score_and_select_dreams(self, events):
        """
        Score and select dream candidates based on novelty/confidence.
        Returns: List of selected dream event dicts.
        """
        scored = []
        for event in events:
            meta = event.get("metadata", {})
            novelty = meta.get("novelty", 0)
            confidence = meta.get("confidence", 1)
            score = self.novelty_weight * novelty - self.confidence_weight * confidence
            scored.append((score, event))
        scored.sort(reverse=True, key=lambda x: x[0])
        if self.selection_strategy == "top":
            selected = [e for (_, e) in scored[:self.max_dreams]]
        elif self.selection_strategy == "random":
            selected = [e for (_, e) in random.sample(scored, min(self.max_dreams, len(scored)))]
        else:
            selected = [e for (_, e) in scored[:self.max_dreams]]
        return selected

    def add_dream_noise(self, dream_event):
        noise_level = self.dream_noise_level
        ed = dream_event["event_data"].copy()
        # Shuffle words or insert random tokens in text fields
        for key, value in ed.items():
            if isinstance(value, str) and random.random() < noise_level:
                words = value.split()
                if words:
                    random.shuffle(words)
                    if random.random() < noise_level:
                        words.insert(
                            random.randint(0, len(words)),
                            random.choice(["???", "dream", "echo", "phantom", "mist", "fragment"])
                        )
                    ed[key] = " ".join(words)
        # Mutate metadata
        meta = dream_event["metadata"].copy()
        if "novelty" in meta:
            meta["novelty"] += random.uniform(-0.1, 0.1) * noise_level
        if "confidence" in meta:
            meta["confidence"] += random.uniform(-0.1, 0.1) * noise_level
        dream_event["event_data"] = ed
        dream_event["metadata"] = meta
        return dream_event

    def narrate_dream(self, dream_event):
        """
        Generate a surreal narrative for the dream event.
        """
        prompt = dream_event["event_data"].get("prompt", "")
        response = dream_event["event_data"].get("response", "")
        # Simple surreal narration: merge, shuffle, and add dream-like phrases
        parts = [prompt, response]
        random.shuffle(parts)
        narration = f"In the midst of swirling thoughts, a dream emerged: {parts[0]} ... Suddenly, {parts[1]} ... The boundaries of meaning blurred."
        if random.random() < self.dream_noise_level:
            narration += f" {random.choice(['A phantom word echoed.', 'Mist enveloped the memory.', 'Fragments danced in the void.'])}"
        return narration.strip()

    def generate_dream_events(self, dream_candidates):
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
            dream_event = self.add_dream_noise(dream_event)
            narration = self.narrate_dream(dream_event)
            # Only the narration is logged as the main dream content
            dream_event["narration"] = narration
            dreams.append(dream_event)
        return dreams

    def log_dreams(self, dreams):
        for dream in dreams:
            try:
                event_data = {
                    "narration": dream["narration"],
                    "prompt": dream["event_data"].get("prompt"),
                    "response": dream["event_data"].get("response"),
                    "novelty_score": dream["metadata"].get("novelty"),
                    "confidence_score": dream["metadata"].get("confidence"),
                }
                source_metadata = {
                    "dreamed_from": dream.get("dreamed_from"),
                    "timestamp_iso": dream.get("timestamp_iso"),
                    **dream.get("metadata", {})
                }
                self.scribe_event_fn(
                    origin="dreamer",
                    event_type="dream",
                    event_data=event_data,
                    source_metadata=source_metadata,
                    session_id=dream.get("session_id"),
                    timestamp=None  # or parse from dream["timestamp_iso"] if needed
                )
                self.logger.info(f"Dream event logged: {dream['narration']}")
            except Exception as e:
                self.logger.log_error(f"Failed to log dream event: {e}")

    def run_dream_cycle(self):
        """
        Main entry: extract, select, generate, and log dreams.
        """
        events = self.extract_last_active_period()
        dream_candidates = self.score_and_select_dreams(events)
        dreams = self.generate_dream_events(dream_candidates)
        self.log_dreams(dreams)

    def run_dream_album_cycle(self, memories=None):
        """
        Generate a dream album: multiple songs, each with Markov structure, each section remixing scribe memories.
        Each section now includes musical details: key, scale, tempo, progression, and repeats.
        memories: list of memory strings (if None, will extract from scribe log)
        Returns: album dict with song/section/narration/musical structure
        """
        logger = getattr(self, 'logger', None)
        # 1. Load all memories if not provided
        if memories is None:
            events = self.extract_last_active_period()
            memories = [e.get("event_data", {}).get("prompt", "") for e in events if e.get("event_data", {}).get("prompt")]
        if not memories:
            if logger:
                logger.log_error("No memories available for dream album.")
            return None
        # 2. Configurable number of songs per album
        num_songs = self.config_manager.get("dream_num_songs_per_album", 3)
        album = []
        motif = random.choice(memories) if memories else ""
        for song_idx in range(num_songs):
            song = Song()  # Markov structure
            song_sections = []
            score_lines = [f"Key: {song.key} | Tempo: {song.get_tempo_desc()} | Time Sig: {song.time_signature}"]
            section_memories = []  # Collect memories used for deterministic title
            for section in song.overall_sections:
                if random.random() < 0.3:
                    memory = motif
                else:
                    memory = random.choice(memories)
                section_memories.append(memory)
                dream_event = {
                    "event_data": {"prompt": memory, "response": ""},
                    "metadata": {}
                }
                dream_event = self.add_dream_noise(dream_event)
                narration = self.narrate_dream(dream_event)
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
            # Deterministic song title: use first memory used, cleaned and truncated
            raw_title = section_memories[0] if section_memories else f"Untitled {song_idx+1}"
            clean_title = raw_title.strip().replace('\n', ' ').replace('\r', ' ')
            if len(clean_title) > 48:
                clean_title = clean_title[:48] + '...'
            song_title = clean_title
            poetic_score = "\n".join(score_lines)
            album.append({
                "title": song_title,
                "sections": song_sections,
                "poetic_score": poetic_score
            })
        # Optionally log or return the album
        if logger:
            logger.info(f"Dream album generated with {num_songs} songs.")
        # Process and queue each song as a training sample
        if self.scribe_event_fn and self.metadata_processor and self.scribe_ingestion_processor:
            for idx, song in enumerate(album):
                # 1. Randomly select number of sections (3–7)
                num_sections = random.randint(3, 7)
                section_narratives = []
                for i in range(num_sections):
                    narration = song["sections"][i]["narration"] if i < len(song["sections"]) else ""
                    # 2. Truncate each section to 40 words
                    words = narration.split()
                    if len(words) > 40:
                        narration = " ".join(words[:40]) + "..."
                    section_narratives.append(narration)
                # 3. Build event_data for the template (let template handle missing slots)
                event_data = {f"dream{i+1}": section_narratives[i] for i in range(len(section_narratives))}
                event_data["musical_key"] = song["sections"][0]["musical_details"]["key"]
                event_data["timestamp_unix"] = int(time.time())
                # 4. Process as before
                validated_event_data, final_metadata = self.metadata_processor.enrich_and_validate(
                    origin="dreamer",
                    event_type="dream",
                    event_data=event_data,
                    source_metadata={},
                    session_id=None
                )
                event_for_ingestion = {
                    "event_type": "dream",
                    "event_data": validated_event_data,
                    "metadata": final_metadata
                }
                processed = self.scribe_ingestion_processor.process_entry(event_for_ingestion)
                # 5. (Optional) Truncate total memory string if >512 words
                memory_words = processed["memory"].split()
                if len(memory_words) > 512:
                    processed["memory"] = " ".join(memory_words[:512]) + "..."
                # 6. Send only memory/weight to queue
                capture_scribe_event(
                    origin="dreamer",
                    event_type="dream",
                    event_data={"memory": processed["memory"], "weight": processed["weight"]},
                    source_metadata=None,
                    session_id=None,
                    timestamp=None
                )
        return {
            "album_title": f"Dream Album: '{motif[:32]}...'",
            "motif": motif,
            "songs": album
        }
