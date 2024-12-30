import sqlite3
import os
from pathlib import Path
from typing import Optional, Tuple

DB_PATH = Path(__file__).parent.parent / "output" / "queue.db"


class QueueDB:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS tts_queue
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             text TEXT NOT NULL,
             voice TEXT DEFAULT 'af',
             stitch_long_output BOOLEAN DEFAULT 1,
             status TEXT DEFAULT 'pending',
             output_file TEXT,
             processing_time REAL,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        conn.commit()
        conn.close()

    def _ensure_table_if_needed(self, conn: sqlite3.Connection):
        """Create table if it doesn't exist, only called for write operations"""
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS tts_queue
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             text TEXT NOT NULL,
             voice TEXT DEFAULT 'af',
             stitch_long_output BOOLEAN DEFAULT 1,
             status TEXT DEFAULT 'pending',
             output_file TEXT,
             processing_time REAL,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        conn.commit()

    def add_request(self, text: str, voice: str, stitch_long_output: bool = True) -> int:
        """Add a new TTS request to the queue"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO tts_queue (text, voice, stitch_long_output) VALUES (?, ?, ?)", 
                (text, voice, stitch_long_output)
            )
            request_id = c.lastrowid
            conn.commit()
            return request_id
        except sqlite3.OperationalError:  # Table doesn't exist
            self._ensure_table_if_needed(conn)
            c = conn.cursor()
            c.execute(
                "INSERT INTO tts_queue (text, voice, stitch_long_output) VALUES (?, ?, ?)", 
                (text, voice, stitch_long_output)
            )
            request_id = c.lastrowid
            conn.commit()
            return request_id
        finally:
            conn.close()

    def get_next_pending(self) -> Optional[Tuple[int, str, str]]:
        """Get the next pending request"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                'SELECT id, text, voice, stitch_long_output FROM tts_queue WHERE status = "pending" ORDER BY created_at ASC LIMIT 1'
            )
            return c.fetchone()
        except sqlite3.OperationalError:  # Table doesn't exist
            return None
        finally:
            conn.close()

    def update_status(
        self,
        request_id: int,
        status: str,
        output_file: Optional[str] = None,
        processing_time: Optional[float] = None,
    ):
        """Update request status, output file, and processing time"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            if output_file and processing_time is not None:
                c.execute(
                    "UPDATE tts_queue SET status = ?, output_file = ?, processing_time = ? WHERE id = ?",
                    (status, output_file, processing_time, request_id),
                )
            elif output_file:
                c.execute(
                    "UPDATE tts_queue SET status = ?, output_file = ? WHERE id = ?",
                    (status, output_file, request_id),
                )
            else:
                c.execute(
                    "UPDATE tts_queue SET status = ? WHERE id = ?", (status, request_id)
                )
            conn.commit()
        except sqlite3.OperationalError:  # Table doesn't exist
            self._ensure_table_if_needed(conn)
            # Retry the update
            c = conn.cursor()
            if output_file and processing_time is not None:
                c.execute(
                    "UPDATE tts_queue SET status = ?, output_file = ?, processing_time = ? WHERE id = ?",
                    (status, output_file, processing_time, request_id),
                )
            elif output_file:
                c.execute(
                    "UPDATE tts_queue SET status = ?, output_file = ? WHERE id = ?",
                    (status, output_file, request_id),
                )
            else:
                c.execute(
                    "UPDATE tts_queue SET status = ? WHERE id = ?", (status, request_id)
                )
            conn.commit()
        finally:
            conn.close()

    def get_status(
        self, request_id: int
    ) -> Optional[Tuple[str, Optional[str], Optional[float]]]:
        """Get status, output file, and processing time for a request"""
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                "SELECT status, output_file, processing_time FROM tts_queue WHERE id = ?",
                (request_id,),
            )
            return c.fetchone()
        except sqlite3.OperationalError:  # Table doesn't exist
            return None
        finally:
            conn.close()
