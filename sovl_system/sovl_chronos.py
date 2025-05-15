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

    This class also supports referencing long-term memory events (e.g., from sovl_recaller), allowing the system to say things like
    "We talked about this about two weeks ago" by searching for relevant past events and expressing their temporal distance in human terms.
    Now includes pattern recognition, relative/absolute time references, fuzzy/uncertain time expressions, and meta-communicative memory fading for a more human feel.
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

    def fuzzy_time_phrase(self, seconds_ago: float, event_dt: Optional[datetime] = None) -> str:
        """
        Return a fuzzy, human-like time phrase for an event.
        """
        if seconds_ago < 86400:
            return random.choice(["recently", "not long ago"])
        elif seconds_ago < 7 * 86400:
            return random.choice(["earlier this week", "a few days ago"])
        elif seconds_ago < 30 * 86400:
            return random.choice(["earlier this month", "a couple weeks ago"])
        elif event_dt:
            return f"around {event_dt.strftime('%B %Y')}"
        else:
            return random.choice(["a while back", "some time ago"])

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
            
            pattern_summary = self.pattern_recognition_summary(history)
            if pattern_summary:
                awareness_parts.append(pattern_summary)
        
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
                if prefer_absolute and delta > 30 * 86400:
                    time_str = self.absolute_time_str(mem['timestamp'])
                elif fuzzy and delta > 7 * 86400:
                    time_str = self.fuzzy_time_phrase(delta, event_dt)
                else:
                    time_str = self.humanize_time_delta(delta)
                snippet = mem.get('content', '')
                if len(snippet) > 60:
                    snippet = snippet[:57] + '...'
                phrase = self.faded_reference_phrase(salience, f'"{snippet}" {time_str}')
                relevant_phrases.append(phrase)
                if len(relevant_phrases) >= self.config.get('max_memories', 3):
                    break
        
        return " ".join(relevant_phrases) if relevant_phrases else "No specific long-term memories found for this topic."

    def pattern_recognition_summary(self, history: List[Dict]) -> str:
        """
        Gently comments on recent conversational timing, only if a clear, recent pattern is present.
        Frames as a shared experience, not as analysis or prediction.
        Never uses statistics or percentages. Only considers the last few messages.
        """
        if not history:
            return ""
        hours = []
        for msg in history[-5:]:  # Only look at the last 5 messages for recency
            ts = msg.get('timestamp')
            if ts:
                dt = datetime.fromtimestamp(ts)
                hours.append(dt.hour)
        if not hours:
            return ""
        if all(18 <= h < 24 for h in hours):
            return "It's nice to chat with you in the evening again."
        elif all(6 <= h < 12 for h in hours):
            return "Good morning! We've had a few morning chats lately."
        elif all(12 <= h < 18 for h in hours):
            return "We've had a few afternoon chats recently."
        elif all(0 <= h < 6 for h in hours):
            return "We've had a few late night conversations lately."
        return ""

    def update_active_temporal_prompt_in_state(self, history: List[Dict], now: Optional[float] = None, long_term_memories: Optional[List[Dict]] = None, topic: Optional[str] = None, fuzzy: bool = True, prefer_absolute: bool = False):
        """
        Constructs the full structured temporal block and updates it in SOVLState.
        """
        now_val = now or time.time()

        awareness_content = self.get_temporal_context_prompt(history, now_val)
        long_term_content = self.get_long_term_temporal_references(long_term_memories, topic, now_val, fuzzy, prefer_absolute)

        guidelines = (
            "Weave temporal context naturally to mirror human time awareness. "
            "Clarify event timing for coherence. Note gaps or recurring topics conversationally (e.g., 'just now,' 'ages ago'). "
            "Frame reminders or follow-ups with relevant time cues. Soften older memory references (e.g., 'some time back')."
        )
        guardrails = (
            "Mention time only for clarity or connection. Avoid technical timestamps unless asked. "
            "Stay conversational, not time-obsessed. Use fuzzy terms for vague data."
        )
        rule = (
            "Do not mention time unless it adds clarity, context, or value to your response. "
            "Avoid overemphasizing time or making it the focus of the conversation unless the user specifically asks about it or it is directly relevant."
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

# Example for future expansion:
# - Add semantic similarity for topic matching
# - Add pattern recognition ("You usually message in the evenings.")
# - Add support for future events (reminders, scheduled actions)
