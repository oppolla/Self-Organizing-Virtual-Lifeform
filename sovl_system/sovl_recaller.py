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
        self._lock = Lock()
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
        try:
            with self._lock:
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
                self.logger.record_event(
                    event_type="short_term_memory_add",
                    message="Message added to ShortTermMemory.",
                    level=self.logging_level,
                    additional_info={"msg": msg}
                )
        except Exception as e:
            self.logger.log_error(f"ShortTermMemory.add failed: {e}", error_type="ShortTermMemoryError")

    def get(self) -> List[Dict[str, Any]]:
        try:
            with self._lock:
                now = time.time()
                # Filter expired messages
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
                return [{k: v for k, v in m.items() if k != "_timestamp"} for m in filtered]
        except Exception as e:
            self.logger.log_error(f"ShortTermMemory.get failed: {e}", error_type="ShortTermMemoryError")
            return []

    def clear(self):
        try:
            with self._lock:
                self.memory = []
                self.logger.record_event(
                    event_type="short_term_memory_clear",
                    message="ShortTermMemory cleared.",
                    level=self.logging_level
                )
        except Exception as e:
            self.logger.log_error(f"ShortTermMemory.clear failed: {e}", error_type="ShortTermMemoryError")

    def to_dict(self):
        """Serialize short-term memory for persistence."""
        with self._lock:
            return list(self.memory)

    def from_dict(self, memory_list):
        """Restore short-term memory from saved state."""
        with self._lock:
            self.memory = list(memory_list)

class LongTermMemory:
    """
    Handles persistent, vector-searchable conversation storage using SQLite and FAISS.
    """
    def __init__(self, db_path: str, embedding_dim: int, session_id: str, logger: Optional[Logger] = None, config_manager: Optional[ConfigManager] = None, retention_days: Optional[int] = None, top_k: int = 5, logging_level: str = "info"):
        # Use config_manager for db_path, embedding_dim, retention_days, top_k, logging_level if provided
        if config_manager is not None:
            try:
                db_path = config_manager.get("memory.db_path", db_path)
                embedding_dim = config_manager.get("memory.embedding_dim", embedding_dim)
                retention_days = config_manager.get("memory.long_term_retention_days", retention_days)
                top_k = config_manager.get("memory.long_term_top_k", top_k)
                logging_level = config_manager.get("memory.memory_logging_level", logging_level)
            except Exception as e:
                if logger:
                    logger.log_error(f"LongTermMemory failed to get config values: {e}", error_type="LongTermMemoryError")
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.session_id = session_id
        self.retention_days = retention_days
        self.top_k = top_k
        self._lock = Lock()
        self.logger = logger or Logger.get_instance()
        self.logging_level = logging_level
        try:
            self._init_database()
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.message_ids = []
            self.logger.record_event(
                event_type="long_term_memory_init",
                message=f"LongTermMemory initialized with db_path={db_path}, embedding_dim={embedding_dim}, session_id={session_id}, retention_days={retention_days}, top_k={top_k}",
                level=logging_level
            )
        except Exception as e:
            self.logger.log_error(f"LongTermMemory.__init__ failed: {e}", error_type="LongTermMemoryError")

    # Example: prune old records by retention_days (to be called after add or periodically)
    def prune_expired(self):
        if self.retention_days is not None:
            try:
                cutoff = time.time() - self.retention_days * 86400
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff,))
                    conn.commit()
                self.logger.record_event(
                    event_type="long_term_memory_prune",
                    message="Pruned expired long-term memory records.",
                    level=self.logging_level
                )
            except Exception as e:
                self.logger.log_error(f"LongTermMemory.prune_expired failed: {e}", error_type="LongTermMemoryError")

    # You may want to call this inside add() or as a periodic maintenance task

    def add(self, msg: Dict[str, Any]):
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO conversations (role, content, embedding, timestamp, user_id, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (msg["role"], msg["content"], msg["embedding"].tobytes(), msg["timestamp"], msg["user_id"], self.session_id)
                    )
                    msg_id = cursor.lastrowid
                    conn.commit()
                self.faiss_index.add(msg["embedding"].reshape(1, -1))
                self.message_ids.append(msg_id)
                self.logger.record_event(
                    event_type="long_term_memory_add",
                    message="Message added to LongTermMemory.",
                    level=self.logging_level,
                    additional_info={"msg": msg, "msg_id": msg_id}
                )
        except Exception as e:
            self.logger.log_error(f"LongTermMemory.add failed: {e}", error_type="LongTermMemoryError")

    def query(self, query_embedding: np.ndarray, user_id: Optional[str] = None, top_k: int = 5, short_term_memory: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    where = "WHERE session_id = ?"
                    params = [self.session_id]
                    if user_id:
                        where += " AND user_id = ?"
                        params.append(user_id)
                    cursor.execute(f"SELECT id, role, content, embedding, timestamp, user_id FROM conversations {where}", params)
                    rows = cursor.fetchall()
                    if not rows:
                        self.logger.record_event(
                            event_type="long_term_memory_query_empty",
                            message="No rows found for LongTermMemory query.",
                            level=self.logging_level
                        )
                        return []
                    embeddings = np.stack([np.frombuffer(row[3], dtype=np.float32) for row in rows])
                    faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                    faiss_index.add(embeddings)
                    D, I = faiss_index.search(query_embedding.reshape(1, -1), min(top_k, len(rows)))
                    results = [
                        {
                            "id": rows[i][0],
                            "role": rows[i][1],
                            "content": rows[i][2],
                            "timestamp": rows[i][4],
                            "user_id": rows[i][5],
                            "session_id": self.session_id
                        }
                        for i in I[0]
                    ]
                    self.logger.record_event(
                        event_type="long_term_memory_query",
                        message="LongTermMemory query executed.",
                        level=self.logging_level,
                        additional_info={"num_results": len(results)}
                    )
                    return results
        except Exception as e:
            self.logger.log_error(f"LongTermMemory.query failed: {e}", error_type="LongTermMemoryError")
            return []

    def clear(self, user_id: Optional[str] = None):
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    if user_id:
                        cursor.execute("DELETE FROM conversations WHERE user_id = ? AND session_id = ?", (user_id, self.session_id))
                    else:
                        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (self.session_id,))
                    conn.commit()
                self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                self.message_ids = []
                self.logger.record_event(
                    event_type="long_term_memory_clear",
                    message="LongTermMemory cleared.",
                    level=self.logging_level,
                    additional_info={"user_id": user_id}
                )
        except Exception as e:
            self.logger.log_error(f"LongTermMemory.clear failed: {e}", error_type="LongTermMemoryError")

class DialogueContextManager:
    """
    Orchestrates short-term and long-term memory for a session, with optional RAM health checks.
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
        memory_logging_level: str = "info"
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
        self.long_term = LongTermMemory(
            db_path=self.db_path,
            embedding_dim=self.embedding_dim,
            session_id=self.session_id,
            logger=self.logger,
            config_manager=config_manager,
            retention_days=long_term_retention_days,
            top_k=long_term_top_k,
            logging_level=memory_logging_level
        )
        self.embedding_fn = embedding_fn or self._default_embedding_fn
        self.ram_manager = None
        if config_manager is not None and hasattr(RAMManager, 'check_memory_health'):
            try:
                self.ram_manager = RAMManager(config_manager, self.logger)
            except Exception as e:
                self.logger.log_error(f"DialogueContextManager: RAMManager init failed: {e}", error_type="RAMManagerInitError")

    def _default_embedding_fn(self, content: str) -> np.ndarray:
        return np.random.rand(self.embedding_dim).astype(np.float32)

    def add_message(self, role: str, content: str, user_id: str = "default"):
        try:
            if self.ram_manager:
                try:
                    ram_health = self.ram_manager.check_memory_health()
                    if ram_health.get('usage_percentage', 0) > 0.90:
                        if self.logger:
                            self.logger.record_event(
                                event_type="ram_usage_high",
                                message=f"RAM usage high ({ram_health['usage_percentage']*100:.1f}%), skipping add_message.",
                                level="warning"
                            )
                        return  # Skip adding message if RAM is critically high
                except Exception as e:
                    if self.logger:
                        self.logger.record_event(
                            event_type="ram_health_check_failed",
                            message=f"RAM health check failed: {e}",
                            level="warning"
                        )
            embedding = self.embedding_fn(content)
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
            self.long_term.add(msg)
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"DialogueContextManager.add_message failed: {e}", error_type="DialogueContextManagerError")
            if self.error_manager:
                self.error_manager.record_error(e, error_type="DialogueContextManagerError")

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
            short_term_emb = np.stack([msg["embedding"] for msg in stm]) if stm else np.empty((0, self.embedding_dim), dtype=np.float32)
            query_emb = short_term_emb[-1] if short_term_emb.shape[0] > 0 else self._default_embedding_fn(input_message)
            long_term_msgs = self.get_long_term_context(user_id, query_emb)
            long_term_emb = np.stack([query_emb for _ in long_term_msgs]) if long_term_msgs else np.empty((0, self.embedding_dim), dtype=np.float32)
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
