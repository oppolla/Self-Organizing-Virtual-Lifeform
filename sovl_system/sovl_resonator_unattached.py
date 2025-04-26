import time
from typing import Callable, Optional
from sovl_queue import get_scribe_queue, ScribeEntry

class SOVLResonator:
    """
    The SOVL Resonator processes perceptual events (e.g., from camera, sensors),
    computes affective and cognitive responses (joy, curiosity, etc.), and updates
    internal state that can influence other modules (curiosity, temperament, etc.).
    Optionally logs to a scribe and notifies callbacks. Can also nudge a curiosity module.
    """
    def __init__(self, logger: Optional[object] = None, scribe: Optional[object] = None, curiosity: Optional[object] = None):
        self.logger = logger
        self.scribe = scribe
        self.curiosity = curiosity
        self.state = {
            "joy": 0.0,
            "curiosity": 0.0,
            "last_event": None,
            "last_update": time.time()
        }
        self.callbacks = []
        # Integrate SOVL queue system
        self.scribe_queue = get_scribe_queue()

    def resonate(self, event: dict) -> dict:
        """
        Process a perceptual event, update internal resonance state, and nudge curiosity if available.
        Args:
            event: Dict with at least a 'description' key.
        Returns:
            Updated resonance state.
        """
        desc = event.get("description", "")
        # Simple affect logic (customize as needed)
        if "cat" in desc:
            self.state["joy"] += 0.2
            self.state["curiosity"] += 0.1
        elif "novel" in desc or "new" in desc:
            self.state["curiosity"] += 0.2
        else:
            self.state["curiosity"] += 0.01  # baseline curiosity

        # Clamp values
        self.state["joy"] = min(max(self.state["joy"], 0.0), 1.0)
        self.state["curiosity"] = min(max(self.state["curiosity"], 0.0), 1.0)
        self.state["last_event"] = desc
        self.state["last_update"] = time.time()

        # Log to scribe if available
        if self.scribe:
            self.scribe.log_event({
                "type": "resonance",
                "joy": self.state["joy"],
                "curiosity": self.state["curiosity"],
                "event": desc,
                "timestamp": self.state["last_update"]
            })

        # --- Enqueue event to SOVL scribe queue ---
        entry = ScribeEntry(
            origin="SOVLResonator",
            event_type="resonance_event",
            event_data={
                "state": self.state.copy(),
                "event": event
            }
        )
        try:
            self.scribe_queue.put(entry, timeout=1)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Failed to enqueue resonance event: {e}")

        # Nudge curiosity system if available
        if self.curiosity:
            if hasattr(self.curiosity, 'nudge_curiosity'):
                self.curiosity.nudge_curiosity(self.state["curiosity"])
            elif hasattr(self.curiosity, 'update_pressure'):
                self.curiosity.update_pressure(self.state["curiosity"])

        # Notify callbacks
        for cb in self.callbacks:
            try:
                cb(self.state.copy())
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Callback error in SOVLResonator: {e}")

        # Optionally log resonance
        if self.logger:
            self.logger.record_event(
                event_type="resonance_update",
                message=f"Resonance updated: joy={self.state['joy']}, curiosity={self.state['curiosity']}",
                additional_info={"event": desc, "timestamp": self.state["last_update"]}
            )

        return self.state.copy()

    def get_state(self) -> dict:
        """
        Return the current resonance state.
        """
        return self.state.copy()

    def register_callback(self, callback: Callable[[dict], None]):
        """
        Register a callback to be notified on resonance state updates.
        """
        self.callbacks.append(callback)
