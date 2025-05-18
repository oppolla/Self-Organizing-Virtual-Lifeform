import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random
import math
from sovl_logger import Logger
from sovl_error import ErrorManager
from sovl_config import ConfigManager
from sovl_state import StateManager, SOVLState

class Chronos:
    """
    Provides temporal context for LLM prompts, enabling the system to express awareness of the passage of time in a human-like way.
    Designed to be injected into system prompts alongside modules like viber and shamer.
    Easily expandable for more advanced temporal reasoning.
    """
    def __init__(self, config_manager: ConfigManager, state_manager: StateManager, logger: Logger, config: Optional[dict] = None):
        """
        Args:
            config_manager: ConfigManager instance
            state_manager: StateManager instance
            logger: Logger instance
            config: Optional dict with configuration, e.g.:
                {
                    "min_gap_seconds": 60,  # Only mention time if > 1 minute gap
                    "max_history": 10,     # How many past events to consider
                    "mention_start": True, # Whether to mention conversation start time
                    "max_memories": 3,     # Max number of long-term memories to reference
                    "memory_decay_lambda": 0.1, # Decay rate for memory salience
                }
        """
        self.config = config or {
            "min_gap_seconds": 60,
            "max_history": 10,
            "mention_start": True,
            "max_memories": 3,
            "memory_decay_lambda": 0.1,
        }
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.logger = logger
        self.error_manager = ErrorManager(context=self, state_tracker=None, config_manager=self.config_manager)

    def humanize_time_delta(self, seconds: float) -> str:
        """
        Convert a time delta in seconds to a human-friendly string.
        """
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds // 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds // 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds < 2592000:
            weeks = int(seconds // 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        else:
            months = int(seconds // 2592000)
            return f"{months} month{'s' if months != 1 else ''} ago"

    def absolute_time_str(self, timestamp: float) -> str:
        """
        Return an absolute date string for a timestamp (e.g., 'on March 3rd, 2024').
        """
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('on %B %d, %Y')

    def memory_salience(self, age_seconds: float) -> float:
        """
        Exponential decay for memory salience. Returns value in [0,1].
        """
        decay_lambda = self.config.get("memory_decay_lambda", 0.1)
        age_days = age_seconds / 86400
        return math.exp(-decay_lambda * age_days)

    def faded_reference_phrase(self, salience: float, event_phrase: str) -> str:
        """
        Always returns a phrase, never forgetful, but softens language as salience drops.
        """
        if salience > 0.7:
            return f"You mentioned {event_phrase} recently."
        elif salience > 0.3:
            return f"A while back, you mentioned {event_phrase}."
        elif salience > 0.1:
            return f"Earlier in our conversations, you brought up {event_phrase}."
        else:
            return f"It's been a long time, but I remember you said {event_phrase}."

    def get_temporal_context_prompt(self, history: List[Dict], now: Optional[float] = None) -> str:
        """
        Generate a dynamic temporal context string based on conversation history
        for the 'Temporal Awareness' section.
        Args:
            history: List of message dicts, each with at least 'timestamp' (UNIX time float) and 'role'.
            now: Optional UNIX timestamp for 'current' time (defaults to time.time()).
        Returns:
            A string describing the dynamic temporal context.
        """
        now = now or time.time()
        awareness_parts = []
        if not history:
            awareness_parts.append("This is the start of the conversation.")
        else:
            if self.config.get("mention_start", True):
                start_time = history[0].get("timestamp", now)
                conv_age = now - start_time
                if conv_age > 86400:  # More than a day old
                    start_str = self.humanize_time_delta(conv_age)
                    awareness_parts.append(f"This conversation began {start_str}.")
            
            last_event = history[-1]
            last_time = last_event.get("timestamp", now)
            delta = now - last_time
            if delta >= self.config["min_gap_seconds"]:
                time_str = self.humanize_time_delta(delta)
                if last_event["role"] == "user":
                    awareness_parts.append(f"The last user message was {time_str}.")
                else:
                    awareness_parts.append(f"The last system message was {time_str}.")
            
            pattern_data = self.pattern_recognition_summary(history)
            if pattern_data.get("recent_message_hours"):
                awareness_parts.append(f"Recent message hours: {pattern_data['recent_message_hours']}")
        
        return " ".join(awareness_parts) if awareness_parts else "No specific short-term temporal observations."

    def is_high_quality_memory(self, mem: Dict, min_words: int = 5) -> bool:
        """
        Returns True if the memory content is above a minimum word count (default: 5 words).
        Can be extended for more advanced quality checks.
        """
        content = mem.get('content', '')
        return len(content.split()) > min_words

    def get_long_term_temporal_references(self, memories: Optional[List[Dict]], topic: Optional[str], now: Optional[float] = None, fuzzy: bool = True, prefer_absolute: bool = False) -> str:
        """
        Generates a string summarizing relevant long-term memories for the 'Long-term Memory' section.
        Only references high-quality memories (content length > 5 words by default).
        Returns 'None available.' if no relevant memories are found or inputs are insufficient.
        """
        if not memories or not topic:
            return "None available for the current topic."

        now = now or time.time()
        relevant_phrases = []
        min_words = self.config.get('min_memory_words', 5)
        for mem in memories:
            if topic.lower() in mem.get('content', '').lower() and self.is_high_quality_memory(mem, min_words):
                delta = now - mem['timestamp']
                event_dt = datetime.fromtimestamp(mem['timestamp'])
                salience = self.memory_salience(delta)
                # Provide structured time metadata for the LLM to phrase naturally
                if prefer_absolute and delta > 30 * 86400:
                    time_info = f"on {event_dt.strftime('%B %d, %Y')}"
                else:
                    days_ago = int(delta // 86400)
                    time_info = f"{days_ago} days ago (timestamp: {event_dt.strftime('%Y-%m-%d')})"
                snippet = mem.get('content', '')
                if len(snippet) > 60:
                    snippet = snippet[:57] + '...'
                phrase = self.faded_reference_phrase(salience, f'"{snippet}" {time_info}')
                relevant_phrases.append(phrase)
                if len(relevant_phrases) >= self.config.get('max_memories', 3):
                    break
        return " ".join(relevant_phrases) if relevant_phrases else "No specific long-term memories found for this topic."

    def pattern_recognition_summary(self, history: List[Dict]) -> dict:
        """
        Returns structured timing data for the last few messages, for the LLM to interpret.
        """
        if not history:
            return {}
        hours = []
        for msg in history[-5:]:
            ts = msg.get('timestamp')
            if ts:
                dt = datetime.fromtimestamp(ts)
                hours.append(dt.hour)
        return {"recent_message_hours": hours}

    def update_active_temporal_prompt_in_state(self, history: List[Dict], now: Optional[float] = None, long_term_memories: Optional[List[Dict]] = None, topic: Optional[str] = None, fuzzy: bool = True, prefer_absolute: bool = False):
        """
        Constructs the full structured temporal block and updates it in SOVLState.
        """
        now_val = now or time.time()

        awareness_content = self.get_temporal_context_prompt(history, now_val)
        long_term_content = self.get_long_term_temporal_references(long_term_memories, topic, now_val, fuzzy, prefer_absolute)

        guidelines = (
            "Bring a natural sense of time to the conversation, like a human reflecting on when things happened. "
            "Feel free to weave in temporal cues to make responses coherent and engaging, using phrases like 'just now,' 'a few days ago,' or 'some time back' to match the context. "
            "When mentioning past events or memories, aim for a conversational tone, softening references to older events to feel more distant or nostalgic. "
            "Draw on the provided time metadata to craft your own phrasing, keeping it fluid and natural. "
            "If you notice patterns in message timing—like chats often happening late at night or early in the morning—you might subtly nod to them in a way that feels organic, as a friend might. "
            "Let the context guide how much or how little you emphasize time."
        )
        guardrails = (
            "Highlight time only when it adds clarity, connection, or depth to the conversation. "
            "Prefer conversational phrases over precise timestamps unless the user asks for specifics. "
            "Aim to keep time references natural and unobtrusive, using fuzzy terms like 'recently' or 'a while ago' when details are vague."
        )
        rule = (
            "Include temporal references only when they enrich the response’s clarity, context, or relevance. "
            "Let time cues enhance the conversation naturally, without becoming the focus, unless the user specifically asks about timing or it’s central to their query."
        )

        final_temporal_block = (
            f"- **Temporal Awareness:** {awareness_content}\\n"
            f"- **Long-term Memory:** {long_term_content}\\n"
            f"- **Guidelines:** {guidelines}\\n"
            f"- **Guardrails:** {guardrails}\\n"
            f"- **Rule:** {rule}"
        )

        def update_fn(state_clone: SOVLState) -> SOVLState:
            state_clone.active_temporal_prompt = final_temporal_block
            self.logger.record_event("chronos_state_update", "Updated active_temporal_prompt in SOVLState with structured block.")
            return state_clone

        success = self.state_manager.update_state_atomic(update_fn)
        if not success:
            self.logger.log_error("Failed to update active_temporal_prompt in SOVLState.")

    def inject_temporal_context(self, base_prompt: str, history: List[Dict], now: Optional[float] = None, long_term_memories: Optional[List[Dict]] = None, topic: Optional[str] = None, fuzzy: bool = True, prefer_absolute: bool = False) -> str:
        """
        Inject the temporal context string (short and long term) into the system prompt.
        Args:
            base_prompt: The base system prompt string.
            history: Conversation history as above.
            now: Optional UNIX timestamp for 'current' time.
            long_term_memories: Optional list of long-term memory dicts.
            topic: Optional topic string for long-term memory search.
            fuzzy: If True, use fuzzy time phrases for older events.
            prefer_absolute: If True, use absolute date for older events.
        Returns:
            The system prompt with temporal context prepended (if relevant).
        """
        temporal_context = self.get_temporal_context_prompt(history, now)
        long_term_context = ""
        if long_term_memories and topic:
            long_term_context = self.get_long_term_temporal_references(long_term_memories, topic, now, fuzzy, prefer_absolute)
        context_parts = [temporal_context]
        if long_term_context:
            context_parts.append(long_term_context)
        context_str = "\n".join([part for part in context_parts if part])
        if context_str:
            return f"{context_str}\n{base_prompt}"
        return base_prompt
