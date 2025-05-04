import torch
import json
import os
from collections import deque
from threading import Lock
import time
import numpy as np
import sqlite3
import faiss
from typing import Optional, Dict, List, Any, Callable
from sovl_memory import RAMManager
from sovl_logger import Logger
from sovl_error import ErrorManager, ConfigurationError
from sovl_config import ConfigManager
import hashlib
import threading

class ShortTermMemory:
    """
    Handles in-memory, per-session short-term conversation history.
    """
    def __init__(self, max_short_term: int = 50, logger: Optional[Logger] = None, config_manager: Optional[ConfigManager] = None, expiry_seconds: Optional[int] = None, logging_level: str = "info"):
        # Use config_manager for max_short_term, expiry_seconds, and logging_level if provided
        if config_manager is not None:
            try:
                max_short_term = config_manager.get("memory.max_short_term", max_short_term)
                expiry_seconds = config_manager.get("memory.short_term_expiry_seconds", expiry_seconds)
                logging_level = config_manager.get("memory.memory_logging_level", logging_level)
            except Exception as e:
                if logger:
                    logger.log_error(f"ShortTermMemory failed to get config values: {e}", error_type="ShortTermMemoryError")
        self.max_short_term = max_short_term
        self.expiry_seconds = expiry_seconds
        self.memory = []  # List of dicts, each with a timestamp
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.logger = logger or Logger.get_instance()
        self.logging_level = logging_level
        try:
            self.logger.record_event(
                event_type="short_term_memory_init",
                message=f"ShortTermMemory initialized with max_short_term={max_short_term}, expiry_seconds={expiry_seconds}",
                level=logging_level
            )
        except Exception as e:
            self.logger.log_error(f"ShortTermMemory.__init__ failed: {e}", error_type="ShortTermMemoryError")

    def add(self, msg: Dict[str, Any]):
        start_wait = time.perf_counter()
        with self._write_lock:
            wait_time = time.perf_counter() - start_wait
            self.logger.record_event(
                event_type="short_term_memory_write_lock_acquired",
                message=f"Write lock acquired in {wait_time:.6f} seconds.",
                level="debug"
            )
            start_hold = time.perf_counter()
            msg_with_time = dict(msg)
            msg_with_time["_timestamp"] = time.time()
            self.memory.append(msg_with_time)
            # Prune by count
            if len(self.memory) > self.max_short_term:
                removed = self.memory.pop(0)
                self.logger.record_event(
                    event_type="short_term_memory_prune",
                    message="ShortTermMemory pruned oldest message due to max size.",
                    level=self.logging_level,
                    additional_info={"removed": removed}
                )
            # Prune by expiry
            if self.expiry_seconds is not None:
                now = time.time()
                before = len(self.memory)
                self.memory = [m for m in self.memory if now - m["_timestamp"] <= self.expiry_seconds]
                after = len(self.memory)
                if before != after:
                    self.logger.record_event(
                        event_type="short_term_memory_expiry_prune",
                        message=f"ShortTermMemory pruned {before - after} expired messages.",
                        level=self.logging_level
                    )
            hold_time = time.perf_counter() - start_hold
            self.logger.record_event(
                event_type="short_term_memory_write_lock_held",
                message=f"Write lock held for {hold_time:.6f} seconds.",
                level="debug"
            )
        # Logging outside lock
        self.logger.record_event(
            event_type="short_term_memory_add",
            message="Message added to ShortTermMemory.",
            level=self.logging_level,
            additional_info={"msg": msg}
        )

    def get(self) -> List[Dict[str, Any]]:
        start_wait = time.perf_counter()
        with self._read_lock:
            wait_time = time.perf_counter() - start_wait
            self.logger.record_event(
                event_type="short_term_memory_read_lock_acquired",
                message=f"Read lock acquired in {wait_time:.6f} seconds.",
                level="debug"
            )
            now = time.time()
            if self.expiry_seconds is not None:
                filtered = [m for m in self.memory if now - m["_timestamp"] <= self.expiry_seconds]
            else:
                filtered = list(self.memory)
            self.logger.record_event(
                event_type="short_term_memory_get",
                message="ShortTermMemory context retrieved.",
                level=self.logging_level,
                additional_info={"size": len(filtered)}
            )
            # Remove _timestamp before returning
            result = [{k: v for k, v in m.items() if k != "_timestamp"} for m in filtered]
        self.logger.record_event(
            event_type="short_term_memory_get",
            message="ShortTermMemory context retrieved.",
            level=self.logging_level,
            additional_info={"size": len(result)}
        )
        return result

    def clear(self):
        start_wait = time.perf_counter()
        with self._write_lock:
            wait_time = time.perf_counter() - start_wait
            self.logger.record_event(
                event_type="short_term_memory_write_lock_acquired",
                message=f"Write lock acquired in {wait_time:.6f} seconds.",
                level="debug"
            )
            self.memory = []
        self.logger.record_event(
            event_type="short_term_memory_clear",
            message="ShortTermMemory cleared.",
            level=self.logging_level
        )

    def to_dict(self):
        """Serialize short-term memory for persistence."""
        with self._read_lock:
            return list(self.memory)

    def from_dict(self, memory_list):
        """Restore short-term memory from saved state."""
        with self._write_lock:
            self.memory = list(memory_list)

    def remove_last(self):
        """Remove the most recently added message (for transactional rollback)."""
        with self._write_lock:
            if self.memory:
                removed = self.memory.pop()
                self.logger.record_event(
                    event_type="short_term_memory_rollback",
                    message="Rolled back last message from ShortTermMemory.",
                    level="warning",
                    additional_info={"removed": removed}
                )

class LongTermMemory:
    """
    Handles persistent, vector-searchable conversation storage using SQLite and FAISS.
    Uses a single thread-safe SQLite connection per instance (or shared if provided),
    with check_same_thread=False and a timeout. All DB operations use self._db_conn
    and are protected by _read_lock and _write_lock. Connection is cleaned up on error or deletion.
    """
    def __init__(self, db_path: str, embedding_dim: int, session_id: str, logger: Optional[Logger] = None, config_manager: Optional[ConfigManager] = None, retention_days: Optional[int] = None, top_k: int = 5, logging_level: str = "info", max_records: int = 10000, db_conn: Optional[sqlite3.Connection] = None):
        # Use config_manager for db_path, embedding_dim, retention_days, top_k, logging_level, max_records if provided
        if config_manager is not None:
            try:
                db_path = config_manager.get("memory.db_path", db_path)
                embedding_dim = config_manager.get("memory.embedding_dim", embedding_dim)
                retention_days = config_manager.get("memory.long_term_retention_days", retention_days)
                top_k = config_manager.get("memory.long_term_top_k", top_k)
                logging_level = config_manager.get("memory.memory_logging_level", logging_level)
                max_records = config_manager.get("memory.long_term_max_records", max_records)
            except Exception as e:
                if logger:
                    logger.log_error(f"LongTermMemory failed to get config values: {e}", error_type="LongTermMemoryError")
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.session_id = session_id
        self.retention_days = retention_days
        self.top_k = top_k
        self.max_records = max_records
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self.logger = logger or Logger.get_instance()
        self.logging_level = logging_level
        self.message_timestamps = {}
        try:
            if db_conn is not None:
                self._db_conn = db_conn
            else:
                self._db_conn = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False,
                    timeout=10.0
                )
            self._init_database()
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.message_ids = []
            # Throttle FAISS index rebuilds
            self._faiss_pending_changes = 0
            self._faiss_rebuild_threshold = config_manager.get("memory.faiss_rebuild_threshold", 100) if config_manager else 100
            self.rebuild_faiss_index()
            self.logger.record_event(
                event_type="long_term_memory_init",
                message=f"LongTermMemory initialized with db_path={db_path}, embedding_dim={embedding_dim}, session_id={session_id}, retention_days={retention_days}, top_k={top_k}, max_records={max_records}",
                level=logging_level
            )
        except Exception as e:
            self._cleanup_connection()
            self.logger.log_error(f"LongTermMemory.__init__ failed: {e}", error_type="LongTermMemoryError")
            raise

    def _cleanup_connection(self):
        try:
            if hasattr(self, '_db_conn') and self._db_conn:
                self._db_conn.close()
                self._db_conn = None
        except Exception as e:
            self.logger.log_error(f"Failed to close database connection: {e}", error_type="LongTermMemoryError")

    def __del__(self):
        self._cleanup_connection()

    def _init_database(self):
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT,
                    content TEXT,
                    embedding BLOB,
                    timestamp REAL,
                    user_id TEXT,
                    session_id TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON conversations (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON conversations (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations (timestamp)")
            self._db_conn.commit()
            cursor.execute("PRAGMA table_info(conversations)")
            columns = {row[1] for row in cursor.fetchall()}
            required = {"id", "role", "content", "embedding", "timestamp", "user_id", "session_id"}
            if not required.issubset(columns):
                raise ConfigurationError(f"Conversations table schema invalid. Missing columns: {required - columns}")
            self.logger.record_event(
                event_type="long_term_memory_db_init",
                message="Conversations table and indexes ensured.",
                level="info"
            )
        except Exception as e:
            self._cleanup_connection()
            self.logger.log_error(f"LongTermMemory._init_database failed: {e}", error_type="LongTermMemoryError")
            raise ConfigurationError(f"Failed to initialize conversations table: {e}")

    def rebuild_faiss_index(self):
        try:
            with self._read_lock:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT id, embedding, timestamp FROM conversations WHERE session_id = ? ORDER BY id ASC", (self.session_id,))
                rows = cursor.fetchall()
                if rows:
                    embeddings = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    self.faiss_index.add(embeddings)
                    self.message_ids = [row[0] for row in rows]
                    self.message_timestamps = {row[0]: row[2] for row in rows}
                else:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    self.message_ids = []
                    self.message_timestamps = {}
        except Exception as e:
            self._cleanup_connection()
            self.logger.log_error(f"LongTermMemory.rebuild_faiss_index failed: {e}", error_type="LongTermMemoryError")
            raise

    def add(self, msg: Dict[str, Any]):
        start_wait = time.perf_counter()
        with self._write_lock:
            wait_time = time.perf_counter() - start_wait
            self.logger.record_event(
                event_type="long_term_memory_write_lock_acquired",
                message=f"Write lock acquired in {wait_time:.6f} seconds.",
                level="debug"
            )
            self.faiss_index.add(msg["embedding"].reshape(1, -1))
        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (role, content, embedding, timestamp, user_id, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                (msg["role"], msg["content"], msg["embedding"].tobytes(), msg["timestamp"], msg["user_id"], self.session_id)
            )
            msg_id = cursor.lastrowid
            self._db_conn.commit()
            with self._write_lock:
                self.message_ids.append(msg_id)
                self.message_timestamps[msg_id] = msg["timestamp"]
                # Throttle FAISS index rebuilds
                self._faiss_pending_changes += 1
                if self._faiss_pending_changes >= self._faiss_rebuild_threshold:
                    self.rebuild_faiss_index()
                    self._faiss_pending_changes = 0
            self.logger.record_event(
                event_type="long_term_memory_add",
                message="Message added to LongTermMemory.",
                level=self.logging_level,
                additional_info={"msg": msg, "msg_id": msg_id}
            )
        except Exception as e:
            self._cleanup_connection()
            self.logger.log_error(f"LongTermMemory.add failed: {e}", error_type="LongTermMemoryError")
            raise

    def query(self, query_embedding: np.ndarray, user_id: Optional[str] = None, top_k: int = 5, min_timestamp: Optional[float] = None, short_term_memory: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        start_time = time.perf_counter()
        with self._read_lock:
            if len(self.message_ids) == 0:
                self.logger.record_event(
                    event_type="long_term_memory_query_empty",
                    message="No rows found for LongTermMemory query.",
                    level=self.logging_level
                )
                return []
            filtered_ids = self.message_ids
            if min_timestamp is not None:
                filtered_ids = [mid for mid in self.message_ids if self.message_timestamps.get(mid, 0) >= min_timestamp]
            if not filtered_ids:
                self.logger.record_event(
                    event_type="long_term_memory_query_empty",
                    message="No rows after timestamp filtering.",
                    level=self.logging_level
                )
                return []
            k = min(top_k, len(filtered_ids))
            if min_timestamp is not None:
                cursor = self._db_conn.cursor()
                placeholders = ','.join('?' for _ in filtered_ids)
                cursor.execute(f"SELECT id, embedding FROM conversations WHERE id IN ({placeholders})", filtered_ids)
                rows = cursor.fetchall()
                if not rows:
                    return []
                embeddings = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
                temp_index = faiss.IndexFlatL2(self.embedding_dim)
                temp_index.add(embeddings)
                D, I = temp_index.search(query_embedding.reshape(1, -1), k)
                result_ids = [rows[i][0] for i in I[0]]
            else:
                D, I = self.faiss_index.search(query_embedding.reshape(1, -1), k)
                result_ids = [filtered_ids[i] for i in I[0]]
        try:
            cursor = self._db_conn.cursor()
            placeholders = ','.join('?' for _ in result_ids)
            where = f"WHERE id IN ({placeholders}) AND session_id = ?"
            params = result_ids + [self.session_id]
            if user_id:
                where += " AND user_id = ?"
                params.append(user_id)
            cursor.execute(f"SELECT id, role, content, embedding, timestamp, user_id FROM conversations {where}", params)
            rows = cursor.fetchall()
            id_to_row = {row[0]: row for row in rows}
            results = [
                {
                    "id": id_to_row[rid][0],
                    "role": id_to_row[rid][1],
                    "content": id_to_row[rid][2],
                    "timestamp": id_to_row[rid][4],
                    "user_id": id_to_row[rid][5],
                    "session_id": self.session_id
                }
                for rid in result_ids if rid in id_to_row
            ]
            query_time = time.perf_counter() - start_time
            self.logger.record_event(
                event_type="long_term_memory_query",
                message=f"LongTermMemory query executed in {query_time:.6f} seconds.",
                level=self.logging_level,
                additional_info={"num_results": len(results)}
            )
            return results
        except Exception as e:
            self._cleanup_connection()
            self.logger.log_error(f"LongTermMemory.query failed: {e}", error_type="LongTermMemoryError")
            raise

    def clear(self, user_id: Optional[str] = None):
        start_wait = time.perf_counter()
        with self._write_lock:
            wait_time = time.perf_counter() - start_wait
            self.logger.record_event(
                event_type="long_term_memory_write_lock_acquired",
                message=f"Write lock acquired in {wait_time:.6f} seconds.",
                level="debug"
            )
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.message_ids = []
            self.message_timestamps = {}
            # Always rebuild index and reset counter on clear
            self.rebuild_faiss_index()
            self._faiss_pending_changes = 0
        try:
            cursor = self._db_conn.cursor()
            if user_id:
                cursor.execute("DELETE FROM conversations WHERE user_id = ? AND session_id = ?", (user_id, self.session_id))
            else:
                cursor.execute("DELETE FROM conversations WHERE session_id = ?", (self.session_id,))
            self._db_conn.commit()
            self.logger.record_event(
                event_type="long_term_memory_clear",
                message="LongTermMemory cleared.",
                level=self.logging_level,
                additional_info={"user_id": user_id}
            )
        except Exception as e:
            self._cleanup_connection()
            self.logger.log_error(f"LongTermMemory.clear failed: {e}", error_type="LongTermMemoryError")
            raise

class MemoryPressureError(Exception):
    """Raised when a message cannot be added due to high RAM usage, even after cleanup attempts."""
    pass

class DialogueContextManager:
    """
    Orchestrates short-term and long-term memory for a session, with optional RAM health checks.
    Uses a shared SQLite connection for all LongTermMemory instances for the same db_path.
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        max_short_term: int = 50,
        db_path: str = "conversations.db",
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        logger: Optional[object] = None,
        session_id: str = "default",
        config_manager: Optional[object] = None,
        short_term_expiry_seconds: Optional[int] = None,
        long_term_retention_days: Optional[int] = None,
        long_term_top_k: int = 5,
        memory_logging_level: str = "info",
        model_manager: Optional[object] = None
    ):
        self.config_manager = config_manager
        # Use config_manager for all memory parameters if provided
        if config_manager is not None:
            try:
                embedding_dim = config_manager.get("memory.embedding_dim", embedding_dim)
                max_short_term = config_manager.get("memory.max_short_term", max_short_term)
                db_path = config_manager.get("memory.db_path", db_path)
                short_term_expiry_seconds = config_manager.get("memory.short_term_expiry_seconds", short_term_expiry_seconds)
                long_term_retention_days = config_manager.get("memory.long_term_retention_days", long_term_retention_days)
                long_term_top_k = config_manager.get("memory.long_term_top_k", long_term_top_k)
                memory_logging_level = config_manager.get("memory.memory_logging_level", memory_logging_level)
            except Exception as e:
                if logger:
                    logger.log_error(f"DialogueContextManager failed to get config values: {e}", error_type="DialogueContextManagerError")
        self.embedding_dim = embedding_dim
        self.max_short_term = max_short_term
        self.db_path = db_path
        self.logger = logger or Logger.get_instance()
        self.session_id = session_id
        self.error_manager = ErrorManager.get_instance() if hasattr(ErrorManager, 'get_instance') else ErrorManager()
        self.short_term = ShortTermMemory(
            max_short_term=self.max_short_term,
            logger=self.logger,
            config_manager=config_manager,
            expiry_seconds=short_term_expiry_seconds,
            logging_level=memory_logging_level
        )
        # Shared DB connection for all LongTermMemory instances for this db_path
        self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        self.long_term = LongTermMemory(
            db_path=self.db_path,
            embedding_dim=self.embedding_dim,
            session_id=self.session_id,
            logger=self.logger,
            config_manager=config_manager,
            retention_days=long_term_retention_days,
            top_k=long_term_top_k,
            logging_level=memory_logging_level,
            db_conn=self._db_conn
        )
        self.model_manager = model_manager
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn
        else:
            self.embedding_fn = self._embedding_from_base_model_with_fallback
        self.ram_manager = None
        if config_manager is not None and hasattr(RAMManager, 'check_memory_health'):
            try:
                self.ram_manager = RAMManager(config_manager, self.logger)
            except Exception as e:
                self.logger.log_error(f"DialogueContextManager: RAMManager init failed: {e}", error_type="RAMManagerInitError")

    def __del__(self):
        if hasattr(self, '_db_conn') and self._db_conn:
            try:
                self._db_conn.close()
            except Exception as e:
                self.logger.log_error(f"Failed to close database connection: {e}", error_type="DialogueContextManagerError")

    def _embedding_from_base_model_with_fallback(self, text: str) -> np.ndarray:
        # Try to use base model for embedding, fallback to hash-based if unavailable
        try:
            if self.model_manager is not None and hasattr(self.model_manager, 'base_model') and hasattr(self.model_manager, 'base_tokenizer'):
                tokenizer = self.model_manager.base_tokenizer
                model = self.model_manager.base_model
                import torch
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                last_hidden = outputs.last_hidden_state  # (batch, seq, hidden)
                pooled = last_hidden.mean(dim=1).squeeze().cpu().numpy()
                if pooled.shape[0] > self.embedding_dim:
                    pooled = pooled[:self.embedding_dim]
                elif pooled.shape[0] < self.embedding_dim:
                    pooled = np.pad(pooled, (0, self.embedding_dim - pooled.shape[0]))
                return pooled.astype(np.float32)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Base model embedding failed: {e}", error_type="EmbeddingError")
        # Fallback: hash-based deterministic embedding
        return self._hash_based_embedding(text)

    def _hash_based_embedding(self, text: str) -> np.ndarray:
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        needed = self.embedding_dim * 4  # 4 bytes per float32
        full_bytes = (hash_bytes * ((needed // len(hash_bytes)) + 1))[:needed]
        arr = np.frombuffer(full_bytes, dtype=np.uint8).astype(np.float32)
        arr = arr[:self.embedding_dim]
        arr = arr / np.linalg.norm(arr) if np.linalg.norm(arr) > 0 else arr
        return arr

    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        return (
            isinstance(embedding, np.ndarray)
            and embedding.shape == (self.embedding_dim,)
            and embedding.dtype == np.float32
        )

    def add_message(self, role: str, content: str, user_id: str = "default"):
        """
        Transactionally add a message to both short-term and long-term memory.
        If long-term add fails, rollback short-term add.
        Raises MemoryPressureError if RAM usage is critically high even after cleanup.
        """
        try:
            if self.ram_manager:
                try:
                    ram_health = self.ram_manager.check_memory_health()
                    if ram_health.get('usage_percentage', 0) > 0.90:
                        before = len(self.short_term.memory)
                        self.short_term.clear()  # Attempt to free memory
                        after = len(self.short_term.memory)
                        self.logger.record_event(
                            event_type="ram_usage_high_cleanup",
                            message=f"RAM usage high, pruned short-term memory from {before} to {after}.",
                            level="warning"
                        )
                        # Re-check RAM
                        ram_health = self.ram_manager.check_memory_health()
                        if ram_health.get('usage_percentage', 0) > 0.90:
                            self.logger.record_event(
                                event_type="ram_usage_still_high",
                                message="RAM usage still high after cleanup, skipping message.",
                                level="error"
                            )
                            raise MemoryPressureError("RAM usage critically high, message not added.")
                except Exception as e:
                    self.logger.record_event(
                        event_type="ram_health_check_failed",
                        message=f"RAM health check failed: {e}",
                        level="warning"
                    )
            embedding = self.embedding_fn(content)
            if not self._validate_embedding(embedding):
                if self.logger:
                    self.logger.log_error("Invalid embedding generated. Skipping message.", error_type="EmbeddingError")
                return
            timestamp = time.time()
            msg = {
                "role": role,
                "content": content,
                "embedding": embedding,
                "timestamp": timestamp,
                "user_id": user_id,
                "session_id": self.session_id
            }
            self.short_term.add(msg)
            try:
                self.long_term.add(msg)
            except Exception as e:
                self.short_term.remove_last()
                if self.logger:
                    self.logger.log_error(f"LongTermMemory.add failed, rolled back ShortTermMemory: {e}", error_type="MemoryConsistencyError")
                raise
        except MemoryPressureError:
            raise
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.add_message failed: {e}", error_type="MemoryConsistencyError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="MemoryConsistencyError")

    def get_short_term_context(self) -> List[Dict[str, Any]]:
        try:
            return self.short_term.get()
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.get_short_term_context failed: {e}", error_type="DialogueContextManagerError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="DialogueContextManagerError")
            return []

    def get_long_term_context(self, user_id: Optional[str] = None, query_embedding: Optional[np.ndarray] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            if query_embedding is None:
                stm = self.short_term.get()
                if stm:
                    query_embedding = stm[-1]["embedding"]
                else:
                    return []
            return self.long_term.query(query_embedding, user_id=user_id, top_k=top_k)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.get_long_term_context failed: {e}", error_type="DialogueContextManagerError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="DialogueContextManagerError")
            return []

    def clear_short_term_memory(self):
        try:
            self.short_term.clear()
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.clear_short_term_memory failed: {e}", error_type="DialogueContextManagerError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="DialogueContextManagerError")

    def clear_long_term_memory(self, user_id: Optional[str] = None):
        try:
            self.long_term.clear(user_id=user_id)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.clear_long_term_memory failed: {e}", error_type="DialogueContextManagerError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="DialogueContextManagerError")

    def forward(self, input_message: str, role: str = "user", user_id: str = "default") -> np.ndarray:
        try:
            self.add_message(role, input_message, user_id)
            stm = self.short_term.get()
            # Only use messages with valid embeddings
            valid_stm = [msg for msg in stm if self._validate_embedding(msg.get("embedding"))]
            short_term_emb = np.stack([msg["embedding"] for msg in valid_stm]) if valid_stm else np.empty((0, self.embedding_dim), dtype=np.float32)
            query_emb = short_term_emb[-1] if short_term_emb.shape[0] > 0 else self.embedding_fn(input_message)
            long_term_msgs = self.get_long_term_context(user_id, query_emb)
            valid_ltm = [msg for msg in long_term_msgs if self._validate_embedding(msg.get("embedding"))]
            if len(valid_ltm) < len(long_term_msgs):
                self.logger.record_event(
                    event_type="long_term_memory_invalid_embedding",
                    message=f"Skipped {len(long_term_msgs) - len(valid_ltm)} long-term messages with invalid embeddings.",
                    level="warning"
                )
            long_term_emb = np.stack([msg["embedding"] for msg in valid_ltm]) if valid_ltm else np.empty((0, self.embedding_dim), dtype=np.float32)
            if short_term_emb.shape[0] > 0 and long_term_emb.shape[0] > 0:
                combined = np.concatenate([short_term_emb, long_term_emb], axis=0)
            elif short_term_emb.shape[0] > 0:
                combined = short_term_emb
            elif long_term_emb.shape[0] > 0:
                combined = long_term_emb
            else:
                combined = np.empty((0, self.embedding_dim), dtype=np.float32)
            return combined
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.forward failed: {e}", error_type="DialogueContextManagerError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="DialogueContextManagerError")
            return np.empty((0, self.embedding_dim), dtype=np.float32)

# Example usage (for testing)
if __name__ == "__main__":
    memory_module = DialogueContextManager(embedding_dim=128, max_short_term=50, db_path="convo.db", session_id="test_session")
    memory_module.add_message("user", "I love sci-fi books!", user_id="user1")
    memory_module.add_message("assistant", "Cool! What's your favorite?", user_id="user1")
    memory_module.add_message("user", "Dune is epic!", user_id="user1")
    print("Short-Term Context:")
    for msg in memory_module.get_short_term_context():
        print(f"{msg['role']}: {msg['content']}")
    print("\nLong-Term Context (user1):")
    long_term = memory_module.get_long_term_context(user_id="user1", top_k=2)
    for msg in long_term:
        print(f"{msg['role']}: {msg['content']}")
    embeddings = memory_module.forward("What's another good book?", user_id="user1")
    print("\nCombined Embeddings Shape:", embeddings.shape)
    memory_module.clear_long_term_memory()
    import os
    os.remove("convo.db")
