from typing import Dict, Any, List
from threading import Thread, Event, Lock
from collections import deque
from sovl_memory import RAMManager, GPUMemoryManager
from sovl_trainer import TrainingCycleManager
from sovl_curiosity import CuriosityManager
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_events import MemoryEventDispatcher, MemoryEventTypes
from sovl_state import StateManager, UserProfileState
from sovl_error import ErrorManager
from sovl_queue import check_scribe_queue_health
from sovl_bonder import BondCalculator, BondModulator  # Add import
from sovl_io import count_jsonl_entries
import time
import traceback
import curses
import statistics
from datetime import datetime
import math

def safe_divide(numerator, denominator, default=0.0):
    """Safely divides two numbers, returning a default value if denominator is zero or inputs are invalid."""
    if denominator == 0 or denominator is None or numerator is None:
        return default
    try:
        result = numerator / denominator
        return result
    except (TypeError, ZeroDivisionError):
        return default

class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager,
        error_manager: ErrorManager,
        user_profile_state: UserProfileState = None,
        bond_calculator: BondCalculator = None
    ):
        """
        Initialize system monitor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
            error_manager: ErrorManager instance for error handling
            user_profile_state: UserProfileState instance for user profile management
            bond_calculator: BondCalculator instance for bond score calculation
        """
        self._config_manager = config_manager
        self._logger = logger
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self._error_manager = error_manager
        self.bond_calculator = bond_calculator
        self._bond_score_history = {}  # user_id -> deque of bond scores
        self._bond_history_maxlen = self._config_manager.get_setting(
            'monitoring', 'bond_history_maxlen', default=30
        )
        
        # Load thresholds from config
        self._ram_critical_threshold = self._config_manager.get_setting(
            'monitoring', 'ram_critical_threshold_percent', default=90.0
        )
        self._gpu_critical_threshold = self._config_manager.get_setting(
            'monitoring', 'gpu_critical_threshold_percent', default=95.0
        )
        
        self._component_metrics = {}  # Store latest metrics for each component
        self._scaffold_confidence_threshold = self._config_manager.get_setting(
            'monitoring', 'scaffold_confidence_threshold', default=0.6
        )
        self._scaffold_fallback_rate_threshold = self._config_manager.get_setting(
            'monitoring', 'scaffold_fallback_rate_threshold', default=0.2
        )
        
        self.user_profile_state = user_profile_state
        
    def get_metrics(self) -> dict:
        """
        Return a dictionary of all current system metrics for UI/CLI/monitoring use.
        This is the standard interface for retrieving system health and status.
        """
        return self._collect_metrics()

    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect system metrics. For internal use only; use get_metrics() for external access.
        """
        try:
            # Get memory stats from managers
            ram_stats = self.ram_manager.check_memory_health()
            gpu_stats = self.gpu_manager.check_memory_health()
            
            # Get queue health status
            queue_status, queue_fill_ratio = check_scribe_queue_health()
            
            # Get scribe journal entry count
            scribe_journal_entry_count = self.get_scribe_journal_entry_count()
            
            metrics = {
                "ram_stats": ram_stats,
                "gpu_stats": gpu_stats,
                "queue_stats": {
                    "status": queue_status,
                    "fill_ratio": queue_fill_ratio
                },
                "scribe_journal_entry_count": scribe_journal_entry_count
            }
            
            # Check for concerning metrics using configured thresholds
            if ram_stats.get("usage_percent", 0) > self._ram_critical_threshold:
                self._error_manager.handle_error(
                    error_type="memory",
                    error_message=f"RAM usage critically high (>{self._ram_critical_threshold}%)",
                    context={"ram_stats": ram_stats}
                )
                
            if gpu_stats.get("usage_percent", 0) > self._gpu_critical_threshold:
                self._error_manager.handle_error(
                    error_type="memory",
                    error_message=f"GPU memory usage critically high (>{self._gpu_critical_threshold}%)",
                    context={"gpu_stats": gpu_stats}
                )
            
            # Check queue health
            if queue_status in ["WARNING", "FULL"]:
                self._error_manager.handle_error(
                    error_type="queue",
                    error_message=f"Scribe queue {queue_status.lower()}: {queue_fill_ratio:.1%} full",
                    context={"queue_stats": metrics["queue_stats"]}
                )
                
            return metrics
            
        except Exception as e:
            self._error_manager.handle_error(
                error_type="monitoring",
                error_message=f"Failed to collect metrics: {str(e)}",
                context={"stack_trace": traceback.format_exc()}
            )
            return {
                "ram_stats": {"status": "error"},
                "gpu_stats": {"status": "error"},
                "queue_stats": {"status": "error"},
                "scribe_journal_entry_count": None
            }

    def record_bond_score(self, user_id, bond_score):
        """Record a bond score for tracking (per user/session)."""
        if user_id is None:
            return
        if user_id not in self._bond_score_history:
            self._bond_score_history[user_id] = deque(maxlen=self._bond_history_maxlen)
        self._bond_score_history[user_id].append((time.time(), bond_score))

    def get_latest_bond_score(self, user_id):
        """Get the latest bond score for a user/session."""
        if self.user_profile_state and user_id:
            return self.user_profile_state.get_bond_score(user_id)
        return None

    def get_all_user_profiles(self):
        """Get all user profiles."""
        if self.user_profile_state:
            return self.user_profile_state.get_all_profiles()
        return {}

    def get_bond_stats(self, user_id=None):
        """Return bond stats for a user, or all users if user_id is None."""
        stats = {}
        users = [user_id] if user_id else list(self._bond_score_history.keys())
        for uid in users:
            history = self._bond_score_history.get(uid, [])
            scores = [score for ts, score in history]
            if scores:
                stats[uid] = {
                    'latest': scores[-1],
                    'mean': statistics.mean(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'stddev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'count': len(scores),
                }
            else:
                stats[uid] = {'latest': self.get_latest_bond_score(uid)}
        return stats

    def update_component_metrics(self, component_name: str, metrics: dict):
        """Update metrics for a component and check thresholds for alerts."""
        self._component_metrics[component_name] = metrics
        if component_name == "scaffold":
            avg_conf = metrics.get("avg_confidence", 1.0)
            fallback_counts = metrics.get("fallback_counts", {})
            total_fallbacks = sum(fallback_counts.values())
            unk_fallbacks = fallback_counts.get("unk", 0)
            fallback_rate = unk_fallbacks / total_fallbacks if total_fallbacks > 0 else 0.0
            if avg_conf < self._scaffold_confidence_threshold:
                self._logger.log_warning(
                    f"Scaffold mapping confidence degraded: {avg_conf:.3f}",
                    additional_info=metrics
                )
            if fallback_rate > self._scaffold_fallback_rate_threshold:
                self._logger.log_warning(
                    f"Scaffold fallback rate high: {fallback_rate:.2%}",
                    additional_info=metrics
                )
        self._logger.log_info(f"Updated metrics for {component_name}", additional_info=metrics)

    def set_state_manager(self, state_manager):
        self.state_manager = state_manager

    def get_gestation_status(self) -> dict:
        mode = None
        progress = None
        if hasattr(self, 'state_manager') and self.state_manager:
            mode = self.state_manager.get_mode()
            if mode == 'gestating':
                progress = self.state_manager.get_gestation_progress()
        return {
            'mode': mode,
            'progress': progress,
        }

    def get_gestation_message(self, dot_count=1):
        status = self.get_gestation_status()
        mode = status.get('mode', 'unknown')
        progress = status.get('progress', None)
        percent = int(progress * 100) if progress is not None else None
        dots = '.' * dot_count
        if percent is not None:
            return f"Gestating {percent}% {dots:<3}"
        else:
            return f"Gestating {dots:<3}"

    def get_dreaming_status(self) -> dict:
        mode = None
        progress = None
        if hasattr(self, 'state_manager') and self.state_manager:
            mode = self.state_manager.get_mode()
            if mode == 'dreaming':
                progress = self.state_manager.get_dreaming_progress()
        return {
            'mode': mode,
            'progress': progress,
        }

    def get_dreaming_message(self, dot_count=1):
        status = self.get_dreaming_status()
        mode = status.get('mode', 'unknown')
        progress = status.get('progress', None)
        percent = int(progress * 100) if progress is not None else None
        dots = '.' * dot_count
        if percent is not None:
            return f"Dreaming {percent}% {dots:<3}"
        else:
            return f"Dreaming {dots:<3}"

    def get_scribe_journal_entry_count(self):
        """Return the number of entries in the scribe journal JSONL file."""
        path = self._config_manager.get('scribed_config.output_path', 'scribe/sovl_scribe.jsonl')
        return count_jsonl_entries(path)

class MemoryMonitor:
    """Monitors system memory usage."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        ram_manager: RAMManager,
        gpu_manager: GPUMemoryManager,
        error_manager: ErrorManager
    ):
        """
        Initialize the memory monitor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            ram_manager: RAMManager instance for RAM memory management
            gpu_manager: GPUMemoryManager instance for GPU memory management
            error_manager: ErrorManager instance for error handling
        """
        self._config_manager = config_manager
        self._logger = logger
        self.ram_manager = ram_manager
        self.gpu_manager = gpu_manager
        self._error_manager = error_manager
        
        # Load critical thresholds from config
        self._ram_critical_mb = self._config_manager.get_setting(
            'monitoring', 'ram_critical_usage_mb', default=8192  # 8GB default
        )
        self._gpu_critical_mb = self._config_manager.get_setting(
            'monitoring', 'gpu_critical_usage_mb', default=4096  # 4GB default
        )
        
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health across all memory managers."""
        try:
            ram_health = self.ram_manager.check_memory_health()
            gpu_health = self.gpu_manager.check_memory_health()
            
            # Check for memory issues using configured thresholds
            if ram_health.get("usage_mb", 0) > self._ram_critical_mb:
                self._error_manager.handle_memory_error(
                    Exception(f"Critical RAM usage detected (> {self._ram_critical_mb}MB)"),
                    ram_health.get("usage_mb", 0)
                )
                
            if gpu_health.get("usage_mb", 0) > self._gpu_critical_mb:
                self._error_manager.handle_memory_error(
                    Exception(f"Critical GPU memory usage detected (> {self._gpu_critical_mb}MB)"),
                    gpu_health.get("usage_mb", 0)
                )
                
            return {
                "ram_health": ram_health,
                "gpu_health": gpu_health
            }
            
        except Exception as e:
            self._error_manager.handle_memory_error(e, 0)
            return {
                "ram_health": {"status": "error"},
                "gpu_health": {"status": "error"}
            }

class TraitsMonitor:
    """Monitors SOVL system traits in real-time."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        state_manager: StateManager,
        curiosity_manager: CuriosityManager,
        training_manager: TrainingCycleManager,
        error_manager: ErrorManager,
        user_profile_state: UserProfileState = None,
        bond_calculator: BondCalculator = None,
        bond_modulator: BondModulator = None,
        update_interval: float = None
    ):
        """
        Initialize the traits monitor.
        
        Args:
            config_manager: Config manager for fetching configuration values
            logger: Logger instance for logging events
            state_manager: StateManager for atomic SOVLState access
            curiosity_manager: CuriosityManager for curiosity metrics
            training_manager: TrainingCycleManager for lifecycle metrics
            error_manager: ErrorManager instance for error handling
            user_profile_state: UserProfileState instance for user profile management
            bond_calculator: BondCalculator instance for bond score calculation
            bond_modulator: BondModulator instance for bond score modulation
            update_interval: Optional override for update interval (defaults to config value)
        """
        self._config_manager = config_manager
        self._logger = logger
        self._state_manager = state_manager
        self._curiosity_manager = curiosity_manager
        self._training_manager = training_manager
        self._error_manager = error_manager
        self.bond_calculator = bond_calculator
        self.bond_modulator = bond_modulator
        
        # Configurable polling interval for monitor loop
        self._update_interval = update_interval if update_interval is not None else config_manager.get_setting(
            'monitoring', 'poll_interval', default=10.0
        )
        
        # Load config values
        self._history_size = self._config_manager.get_setting(
            'monitoring', 'trait_history_size', default=100
        )
        self._variance_thresholds = self._config_manager.get_setting(
            'monitoring', 'trait_variance_thresholds', default={
                'curiosity': 0.3,
                'confidence': 0.25,
                'lifecycle': 0.2,
                'temperament': 0.35
            }
        )
        self._min_samples_for_variance = self._config_manager.get_setting(
            'monitoring', 'min_samples_for_variance', default=10
        )

        # Use a unified trait history dict for all traits
        self._trait_histories = {
            'curiosity': deque(maxlen=self._history_size),
            'confidence': deque(maxlen=self._history_size),
            'lifecycle': deque(maxlen=self._history_size),
            'temperament': deque(maxlen=self._history_size),
            'bond_score': deque(maxlen=self._history_size),
        }
        self._latest_traits = {}
        self._stop_event = Event()
        self._monitor_thread = None
        self._display_lock = Lock()
        self._screen = None
        
        self.user_profile_state = user_profile_state
    
    def start(self):
        """Start the traits monitor."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._logger.log_info("TraitsMonitor already running.")
            return
            
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop_wrapper, daemon=True)
        self._monitor_thread.start()
        self._logger.log_info("TraitsMonitor started.")
        
    def stop(self):
        """Stop the traits monitor."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._logger.log_info("TraitsMonitor already stopped.")
            return
            
        self._stop_event.set()
        self._monitor_thread.join(timeout=self._update_interval * 2)
        if self._monitor_thread.is_alive():
            self._logger.log_warning("TraitsMonitor thread did not exit cleanly.")
        self._monitor_thread = None
        self._logger.log_info("TraitsMonitor stopped.")

    def _monitor_loop_wrapper(self):
        """Wraps the main loop with robust curses initialization and cleanup."""
        try:
            # Initialize curses screen within the thread
            self._screen = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            if curses.has_colors():
                curses.init_pair(1, curses.COLOR_RED, -1)  # Red for erratic behavior
                curses.init_pair(2, curses.COLOR_GREEN, -1)  # Green for normal behavior
            else:
                curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
                curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)

            curses.curs_set(0)  # Hide cursor
            self._screen.nodelay(1)  # Non-blocking input

            self._monitor_loop()

        except curses.error as e:
            self._logger.log_error(
                error_msg=f"Failed to initialize curses: {str(e)}. TraitsMonitor UI disabled.",
                error_type="curses_init_error"
            )
            self._screen = None
            while not self._stop_event.is_set():
                self._stop_event.wait(self._update_interval)

        except Exception as e:
            self._logger.log_error(
                error_msg=f"Unexpected error during TraitsMonitor setup: {str(e)}",
                error_type="monitor_setup_error",
                stack_trace=traceback.format_exc()
            )
        finally:
            if self._screen is not None:
                with self._display_lock:
                    self._screen.nodelay(0)
                    curses.curs_set(1)
                    curses.echo()
                    curses.nocbreak()
                    curses.endwin()
                self._screen = None
                self._logger.log_info("Curses screen cleaned up.")

    def _monitor_loop(self):
        show_user_list = False
        scroll_offset = 0
        while not self._stop_event.is_set():
            try:
                traits = self._get_current_traits()
                if self._screen:
                    if show_user_list:
                        self._display_user_list(scroll_offset)
                    else:
                        self._update_display(traits)
                    try:
                        ch = self._screen.getch()
                        if ch == ord('q'):
                            self._stop_event.set()
                            break
                        elif ch == ord('u'):
                            show_user_list = not show_user_list
                            scroll_offset = 0
                        elif show_user_list and ch == curses.KEY_DOWN:
                            scroll_offset += 1
                        elif show_user_list and ch == curses.KEY_UP:
                            scroll_offset = max(0, scroll_offset - 1)
                    except curses.error:
                        pass
                self._stop_event.wait(self._update_interval)
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Error in monitor loop: {str(e)}",
                    error_type="monitor_loop_error",
                    stack_trace=traceback.format_exc()
                )
                self._stop_event.wait(1.0)

    def _get_current_user_id(self):
        state = self._state_manager.get_state()
        if hasattr(state, "history") and hasattr(state.history, "conversation_id"):
            return state.history.conversation_id
        return None

    def _get_current_traits(self) -> Dict[str, float]:
        state = self._state_manager.get_state()
        traits = {
            'curiosity': self._curiosity_manager.get_curiosity_score(),
            'confidence': state.get_confidence_level(),
            'lifecycle': self._training_manager.get_lifecycle_phase(),
            'temperament': state.get_temperament_score(),
        }
        training_state = state._training_state
        traits.update({
            'data_exposure': training_state.data_exposure,
            'sleep_confidence': safe_divide(
                training_state.sleep_confidence_sum,
                training_state.sleep_confidence_count,
                default=0.0
            ),
            'data_quality': training_state.data_quality_metrics.get('avg_quality', 0.0),
            'avg_input_length': training_state.data_quality_metrics.get('avg_input_length', 0.0),
            'avg_output_length': training_state.data_quality_metrics.get('avg_output_length', 0.0),
            'message_count': getattr(training_state, 'message_count', 0)
        })
        user_id = self._get_current_user_id()
        bond_score = None
        nickname = ""
        interactions = 0
        if self.user_profile_state and user_id:
            bond_score = self.user_profile_state.get_bond_score(user_id)
            nickname = self.user_profile_state.get_nickname(user_id)
            profile = self.user_profile_state.get(user_id)
            interactions = profile.get("interactions", 0)
        traits['bond_score'] = bond_score
        traits['nickname'] = nickname
        traits['user_id'] = user_id
        traits['interactions'] = interactions
        for trait_name in self._trait_histories:
            if trait_name in traits:
                self._trait_histories[trait_name].append(traits[trait_name])
        self._latest_traits = traits.copy()
        for trait_name, history in self._trait_histories.items():
            self._is_trait_erratic(trait_name, history)
        return traits
    
    def _is_trait_erratic(self, trait_name: str, history: deque) -> bool:
        """Check if a trait is showing erratic behavior based on variance."""
        if len(history) < self._min_samples_for_variance:
            return False
            
        try:
            # Calculate variance on the most recent samples
            samples = list(history)[-self._min_samples_for_variance:]
            if len(samples) < 2:  # Need at least 2 points for variance
                return False
                
            variance = statistics.variance(samples)
            threshold = self._variance_thresholds.get(trait_name, 0.3)
            return variance > threshold
            
        except statistics.StatisticsError:
            # Handle cases with insufficient data for variance calculation
            return False
        except Exception as e:
            self._error_manager.handle_error(
                error_type="traits",
                error_message=f"Error calculating variance for {trait_name}: {str(e)}",
                context={"trait_name": trait_name, "history_sample": list(history)[-10:]}
            )
            return False  # Treat calculation errors as non-erratic for safety
    
    def _update_display(self, traits: Dict[str, float]):
        if self._screen is None:
            return
        with self._display_lock:
            try:
                self._screen.clear()
                self._screen.addstr(0, 0, "SOVL Traits Monitor", curses.A_BOLD)
                self._screen.addstr(1, 0, "=" * 50)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._screen.addstr(2, 0, f"Last Update: {current_time}")
                row = 4
                self._screen.addstr(row, 0, "Core Traits:", curses.A_BOLD)
                row += 1
                for trait_name in ['curiosity', 'confidence', 'lifecycle', 'temperament']:
                    value = traits.get(trait_name, float('nan'))
                    history = self._trait_histories[trait_name]
                    is_erratic = self._is_trait_erratic(trait_name, history)
                    color_pair = 1 if is_erratic else 2
                    color_attr = curses.color_pair(color_pair) if curses.has_colors() else curses.A_NORMAL
                    self._screen.addstr(row, 2, f"{trait_name.capitalize()}: ")
                    value_str = f"{value:.3f}" if not math.isnan(value) else "N/A"
                    self._screen.addstr(row, 15, value_str, color_attr)
                    self._screen.addstr(row, 25, "ERRATIC" if is_erratic else "NORMAL", color_attr)
                    row += 1
                row += 1
                self._screen.addstr(row, 0, "Training Metrics:", curses.A_BOLD)
                row += 1
                training_metrics = [
                    ('Data Exposure', 'data_exposure'),
                    ('Sleep Confidence', 'sleep_confidence'),
                    ('Data Quality', 'data_quality'),
                    ('Avg Input Length', 'avg_input_length'),
                    ('Avg Output Length', 'avg_output_length')
                ]
                for label, key in training_metrics:
                    value = traits.get(key, float('nan'))
                    self._screen.addstr(row, 2, f"{label}: ")
                    value_str = f"{value:.2f}" if not math.isnan(value) else "N/A"
                    self._screen.addstr(row, 20, value_str)
                    row += 1
                row += 1
                self._screen.addstr(row, 0, "Conversation Stats:", curses.A_BOLD)
                row += 1
                message_count = traits.get('message_count', 'N/A')
                self._screen.addstr(row, 2, f"Messages: {message_count}")
                row += 1
                self._screen.addstr(row, 0, "Current User Info:", curses.A_BOLD)
                row += 1
                self._screen.addstr(row, 2, f"User ID: {traits.get('user_id', 'N/A')}")
                row += 1
                self._screen.addstr(row, 2, f"Nickname: {traits.get('nickname', '(None)')}")
                row += 1
                self._screen.addstr(row, 2, f"Bond Score: {traits.get('bond_score', 'N/A')}")
                row += 1
                self._screen.addstr(row, 2, f"Interactions: {traits.get('interactions', 'N/A')}")
                row += 2
                max_y, max_x = self._screen.getmaxyx()
                if row < max_y:
                    self._screen.addstr(row, 0, "Press 'u' for user list, 'q' to quit")
                self._screen.refresh()
            except curses.error as e:
                self._logger.log_warning(f"Curses display error: {e}")
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Unexpected error in _update_display: {str(e)}",
                    error_type="display_error",
                    stack_trace=traceback.format_exc()
                )

    def _display_user_list(self, scroll_offset=0):
        if self._screen is None or not self.user_profile_state:
            return
        with self._display_lock:
            try:
                self._screen.clear()
                self._screen.addstr(0, 0, "Known Users (press 'u' to return, 'q' to quit)", curses.A_BOLD)
                self._screen.addstr(1, 0, "=" * 50)
                profiles = list(self.user_profile_state.get_all_profiles().items())
                max_y, max_x = self._screen.getmaxyx()
                header = "User ID        Nickname        Bond    Interactions"
                self._screen.addstr(2, 0, header)
                row = 3
                for i in range(scroll_offset, min(len(profiles), scroll_offset + max_y - 4)):
                    user_id, profile = profiles[i]
                    nickname = profile.get("nickname", "")
                    bond_score = profile.get("bond_score", 0.5)
                    interactions = profile.get("interactions", 0)
                    self._screen.addstr(row, 0, f"{user_id[:12]:<14}{nickname[:12]:<14}{bond_score:>6.2f}{interactions:>12}")
                    row += 1
                self._screen.refresh()
            except curses.error as e:
                self._logger.log_warning(f"Curses display error: {e}")
            except Exception as e:
                self._logger.log_error(
                    error_msg=f"Unexpected error in _display_user_list: {str(e)}",
                    error_type="display_error",
                    stack_trace=traceback.format_exc()
                )

    def get_latest_trait(self, trait_name: str):
        """Get the latest value for a given trait."""
        return self._latest_traits.get(trait_name)

    def get_trait_history(self, trait_name: str):
        """Get the history for a given trait."""
        return list(self._trait_histories.get(trait_name, []))
