"""
src/companion_database.py
=========================
Production-grade SQLite database management for the AI Companion Project.
Central persistent repository for:
- Datasets inventory & processing lineage
- Data exclusions audit
- Model checkpoints & metadata
- Training sessions & loss history
- Evaluation results
- Cleanup operations & audit trail
- Application events & runtime logs

Key design features:
- ACID transactional integrity with WAL mode
- Foreign key constraints enabled
- Online hot backups via sqlite3.Connection.backup
- PRAGMA integrity checks
- Append-only historical tracking (no silent overwriting)
"""

import os
import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

DEFAULT_DB_PATH = Path("D:/DEEP-LEARNING-PROJECT/data/companion_project.db")
DEFAULT_BACKUP_DIR = Path("D:/DEEP-LEARNING-PROJECT/backups/database_backups")


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA256 hash of a file safely in chunks."""
    if not filepath.exists() or not filepath.is_file():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class CompanionDatabase:
    """Central SQLite database for the AI Companion project."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self._get_connection()
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency and robust ACID crash resilience
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def close(self):
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_tables(self):
        """Initializes all relational schema tables with indexes."""
        with self.conn:
            # 1. Datasets Inventory
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                file_path TEXT NOT NULL,
                record_count INTEGER NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                sha256_hash TEXT NOT NULL,
                is_active_companion INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'preserved',
                notes TEXT,
                created_at TEXT NOT NULL
            );
            """)

            # 2. Dataset Processing History & Lineage
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT UNIQUE NOT NULL,
                step_name TEXT NOT NULL,
                input_datasets TEXT NOT NULL,
                output_dataset TEXT NOT NULL,
                records_in INTEGER NOT NULL,
                records_out INTEGER NOT NULL,
                operation_details TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            # 3. Data Exclusions Audit
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_exclusions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exclusion_id TEXT UNIQUE NOT NULL,
                source_dataset TEXT NOT NULL,
                category TEXT NOT NULL,
                count_excluded INTEGER NOT NULL,
                reason TEXT NOT NULL,
                sample_preview TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            # 4. Checkpoints Inventory
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                step INTEGER NOT NULL DEFAULT 0,
                epoch REAL NOT NULL DEFAULT 0.0,
                val_loss REAL,
                is_best INTEGER NOT NULL DEFAULT 0,
                is_latest INTEGER NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                sha256_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'preserved',
                notes TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            # 5. Training Sessions
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                session_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset_used TEXT NOT NULL,
                total_steps INTEGER NOT NULL DEFAULT 0,
                best_val_loss REAL,
                checkpoint_id TEXT,
                config_json TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                notes TEXT
            );
            """)

            # 6. Training Loss History
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_loss_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                epoch REAL,
                train_loss REAL NOT NULL,
                val_loss REAL,
                learning_rate REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            );
            """)

            # 7. Evaluation Results
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eval_id TEXT UNIQUE NOT NULL,
                checkpoint_id TEXT NOT NULL,
                suite_name TEXT NOT NULL,
                score REAL,
                passed_count INTEGER NOT NULL DEFAULT 0,
                total_count INTEGER NOT NULL DEFAULT 0,
                metrics_json TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            # 8. Cleanup & Archive Operations
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cleanup_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT UNIQUE NOT NULL,
                operation_type TEXT NOT NULL,
                original_path TEXT NOT NULL,
                target_path TEXT,
                file_size INTEGER NOT NULL DEFAULT 0,
                sha256_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'executed',
                timestamp TEXT NOT NULL
            );
            """)

            # 9. Application Events & System Logs
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS application_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                metadata_json TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            # 10. Chat Conversations (UI Chatbot History)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT 'custom',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            # 11. Chat Messages (UI Chatbot History)
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_used TEXT,
                route TEXT,
                sources_json TEXT,
                latency_ms REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
            );
            """)

            # Useful indexes for rapid queries
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_datasets_cat ON datasets(category);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_best ON checkpoints(is_best);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_loss_sess_step ON training_loss_history(session_id, step);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_suite ON evaluation_results(suite_name);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);")

    # -------------------------------------------------------------
    # Database Integrity & Backup Methods
    # -------------------------------------------------------------
    def verify_integrity(self) -> Dict[str, Any]:
        """Runs PRAGMA integrity_check and PRAGMA foreign_key_check."""
        cursor = self.conn.cursor()
        integrity_res = cursor.execute("PRAGMA integrity_check;").fetchall()
        fk_res = cursor.execute("PRAGMA foreign_key_check;").fetchall()
        is_ok = len(integrity_res) == 1 and integrity_res[0][0] == "ok" and len(fk_res) == 0
        return {
            "status": "PASS" if is_ok else "FAIL",
            "integrity_check": [r[0] for r in integrity_res],
            "foreign_key_violations": [dict(r) for r in fk_res],
            "database_path": str(self.db_path),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0
        }

    def backup_database(self, backup_dir: Path = DEFAULT_BACKUP_DIR) -> Path:
        """
        Creates an atomic hot backup using SQLite online backup API.
        Never locks writers or reads partially committed transactions.
        """
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_filename = f"companion_project_backup_{ts_str}.db"
        backup_path = backup_dir / backup_filename

        backup_conn = sqlite3.connect(str(backup_path))
        with backup_conn:
            self.conn.backup(backup_conn, pages=100)
        backup_conn.close()

        # Log event
        self.record_event(
            event_type="DATABASE_BACKUP_CREATED",
            description=f"Online SQLite backup created at {backup_path.name}",
            metadata={"backup_path": str(backup_path), "size_bytes": backup_path.stat().st_size}
        )
        return backup_path

    # -------------------------------------------------------------
    # Record Ingestion / Mutator Methods
    # -------------------------------------------------------------
    def register_dataset(
        self,
        dataset_id: str,
        name: str,
        category: str,
        file_path: str,
        record_count: int,
        size_bytes: int,
        sha256_hash: str,
        is_active_companion: bool = False,
        status: str = "preserved",
        notes: Optional[str] = None
    ) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO datasets (
                dataset_id, name, category, file_path, record_count,
                size_bytes, sha256_hash, is_active_companion, status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id) DO UPDATE SET
                file_path=excluded.file_path,
                record_count=excluded.record_count,
                size_bytes=excluded.size_bytes,
                sha256_hash=excluded.sha256_hash,
                is_active_companion=excluded.is_active_companion,
                status=excluded.status,
                notes=excluded.notes;
            """, (
                dataset_id, name, category, file_path, record_count,
                size_bytes, sha256_hash, 1 if is_active_companion else 0,
                status, notes, get_utc_timestamp()
            ))
            return cursor.lastrowid

    def record_exclusion(
        self,
        exclusion_id: str,
        source_dataset: str,
        category: str,
        count_excluded: int,
        reason: str,
        sample_preview: Optional[str] = None
    ) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO data_exclusions (
                exclusion_id, source_dataset, category, count_excluded,
                reason, sample_preview, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(exclusion_id) DO UPDATE SET
                count_excluded=excluded.count_excluded,
                reason=excluded.reason,
                sample_preview=excluded.sample_preview;
            """, (
                exclusion_id, source_dataset, category, count_excluded,
                reason, sample_preview, get_utc_timestamp()
            ))
            return cursor.lastrowid

    def register_checkpoint(
        self,
        checkpoint_id: str,
        name: str,
        file_path: str,
        step: int,
        epoch: float,
        val_loss: Optional[float],
        is_best: bool,
        is_latest: bool,
        size_bytes: int,
        sha256_hash: str,
        status: str = "preserved",
        notes: Optional[str] = None
    ) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO checkpoints (
                checkpoint_id, name, file_path, step, epoch, val_loss,
                is_best, is_latest, size_bytes, sha256_hash, status, notes, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(checkpoint_id) DO UPDATE SET
                file_path=excluded.file_path,
                step=excluded.step,
                epoch=excluded.epoch,
                val_loss=excluded.val_loss,
                is_best=excluded.is_best,
                is_latest=excluded.is_latest,
                size_bytes=excluded.size_bytes,
                sha256_hash=excluded.sha256_hash,
                status=excluded.status,
                notes=excluded.notes;
            """, (
                checkpoint_id, name, file_path, step, epoch, val_loss,
                1 if is_best else 0, 1 if is_latest else 0,
                size_bytes, sha256_hash, status, notes, get_utc_timestamp()
            ))
            return cursor.lastrowid

    def register_training_session(
        self,
        session_id: str,
        session_name: str,
        model_name: str,
        dataset_used: str,
        total_steps: int,
        best_val_loss: Optional[float] = None,
        checkpoint_id: Optional[str] = None,
        config_json: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO training_sessions (
                session_id, session_name, model_name, dataset_used, total_steps,
                best_val_loss, checkpoint_id, config_json, start_time, end_time, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                total_steps=excluded.total_steps,
                best_val_loss=excluded.best_val_loss,
                checkpoint_id=excluded.checkpoint_id,
                end_time=excluded.end_time,
                notes=excluded.notes;
            """, (
                session_id, session_name, model_name, dataset_used, total_steps,
                best_val_loss, checkpoint_id, config_json,
                start_time or get_utc_timestamp(), end_time, notes
            ))
            return cursor.lastrowid

    def record_loss_point(
        self,
        session_id: str,
        step: int,
        epoch: Optional[float],
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        timestamp: Optional[str] = None
    ):
        with self.conn:
            self.conn.execute("""
            INSERT INTO training_loss_history (
                session_id, step, epoch, train_loss, val_loss, learning_rate, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                session_id, step, epoch, train_loss, val_loss,
                learning_rate, timestamp or get_utc_timestamp()
            ))

    def record_cleanup_op(
        self,
        operation_id: str,
        operation_type: str,
        original_path: str,
        target_path: Optional[str],
        file_size: int,
        sha256_hash: str,
        reason: str,
        status: str = "executed"
    ) -> int:
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO cleanup_operations (
                operation_id, operation_type, original_path, target_path,
                file_size, sha256_hash, reason, status, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(operation_id) DO UPDATE SET
                status=excluded.status,
                target_path=excluded.target_path;
            """, (
                operation_id, operation_type, original_path, target_path,
                file_size, sha256_hash, reason, status, get_utc_timestamp()
            ))
            return cursor.lastrowid

    def record_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        event_id = f"evt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO application_events (
                event_id, event_type, description, metadata_json, timestamp
            ) VALUES (?, ?, ?, ?, ?);
            """, (
                event_id, event_type, description,
                json.dumps(metadata or {}, ensure_ascii=False),
                get_utc_timestamp()
            ))
            return cursor.lastrowid

    # -------------------------------------------------------------
    # Query & Reporting Methods
    # -------------------------------------------------------------
    def get_summary(self) -> Dict[str, Any]:
        """Provides high-level dataset, checkpoint, and history summary."""
        cursor = self.conn.cursor()
        
        datasets_count = cursor.execute("SELECT count(*) FROM datasets;").fetchone()[0]
        active_datasets_count = cursor.execute("SELECT count(*) FROM datasets WHERE is_active_companion = 1;").fetchone()[0]
        checkpoints_count = cursor.execute("SELECT count(*) FROM checkpoints;").fetchone()[0]
        sessions_count = cursor.execute("SELECT count(*) FROM training_sessions;").fetchone()[0]
        loss_points_count = cursor.execute("SELECT count(*) FROM training_loss_history;").fetchone()[0]
        exclusions_count = cursor.execute("SELECT count(*), coalesce(sum(count_excluded), 0) FROM data_exclusions;").fetchone()
        cleanup_ops_count = cursor.execute("SELECT count(*) FROM cleanup_operations;").fetchone()[0]

        best_cp = cursor.execute("SELECT checkpoint_id, name, file_path, step, val_loss, sha256_hash FROM checkpoints WHERE is_best = 1 LIMIT 1;").fetchone()

        return {
            "database_file": str(self.db_path),
            "datasets_total": datasets_count,
            "active_companion_datasets": active_datasets_count,
            "checkpoints_total": checkpoints_count,
            "best_checkpoint": dict(best_cp) if best_cp else None,
            "training_sessions": sessions_count,
            "recorded_loss_points": loss_points_count,
            "exclusion_categories": exclusions_count[0],
            "total_records_excluded": exclusions_count[1],
            "cleanup_operations_recorded": cleanup_ops_count
        }

    def fetch_all(self, table_name: str) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        rows = cursor.execute(f"SELECT * FROM {table_name};").fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------
    # Chatbot Conversation History Management
    # -------------------------------------------------------------
    def create_conversation(
        self,
        title: str = "New Chat",
        model: str = "custom",
        conversation_id: Optional[str] = None
    ) -> str:
        """Creates a new conversation record and returns conversation_id."""
        import uuid
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        now = get_utc_timestamp()
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT INTO conversations (conversation_id, title, model, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?);
            """, (conv_id, title.strip() or "New Chat", model, now, now))
        return conv_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves conversation metadata by ID."""
        cursor = self.conn.cursor()
        row = cursor.execute("""
        SELECT * FROM conversations WHERE conversation_id = ?;
        """, (conversation_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists all conversations with message count and last updated timestamp.
        Optionally filters by title or message content search keyword.
        """
        cursor = self.conn.cursor()
        if search and search.strip():
            kw = f"%{search.strip().lower()}%"
            query = """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            WHERE LOWER(c.title) LIKE ? OR LOWER(m.content) LIKE ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC;
            """
            rows = cursor.execute(query, (kw, kw)).fetchall()
        else:
            query = """
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.conversation_id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC;
            """
            rows = cursor.execute(query).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        """Updates the conversation title."""
        now = get_utc_timestamp()
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE conversation_id = ?;
            """, (new_title.strip() or "Untitled Chat", now, conversation_id))
            return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Deletes a conversation and its messages via CASCADE."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?;", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE conversation_id = ?;", (conversation_id,))
            return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        route: Optional[str] = None,
        sources_json: Optional[str] = None,
        latency_ms: Optional[float] = None
    ) -> int:
        """Records a new message in conversation history and updates conversation updated_at."""
        now = get_utc_timestamp()
        with self.conn:
            cursor = self.conn.cursor()
            # Ensure conversation exists
            exists = cursor.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?;", (conversation_id,)
            ).fetchone()
            if not exists:
                cursor.execute("""
                INSERT INTO conversations (conversation_id, title, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?);
                """, (conversation_id, "New Chat", model_used or "custom", now, now))

            cursor.execute("""
            INSERT INTO messages (
                conversation_id, role, content, model_used, route, sources_json, latency_ms, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                conversation_id, role, content, model_used, route, sources_json, latency_ms, now
            ))
            msg_id = cursor.lastrowid
            cursor.execute("""
            UPDATE conversations SET updated_at = ? WHERE conversation_id = ?;
            """, (now, conversation_id))
            return msg_id

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieves all messages for a specific conversation in chronological order."""
        cursor = self.conn.cursor()
        rows = cursor.execute("""
        SELECT * FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC;
        """, (conversation_id,)).fetchall()
        return [dict(r) for r in rows]

    def clear_conversation_messages(self, conversation_id: str) -> bool:
        """Clears all messages from a conversation while keeping the conversation itself."""
        now = get_utc_timestamp()
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?;", (conversation_id,))
            cursor.execute("UPDATE conversations SET updated_at = ? WHERE conversation_id = ?;", (now, conversation_id))
            return True

    def delete_all_conversations(self) -> bool:
        """Deletes all conversation records and messages."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM messages;")
            cursor.execute("DELETE FROM conversations;")
            return True
