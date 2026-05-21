# msc_fairness_project/src/utils/database.py

import os
import time
import sqlite3
from sqlite3 import Error
from pathlib import Path
from contextlib import contextmanager
from typing import Iterator, Optional

import pandas as pd

# --- Didactic Explanation ---
# This script is our single source of truth for the database.
# It defines the path, creates the database file if it doesn't exist,
# and sets up the 'videos' table with the correct schema.
# By keeping this separate, our main collector script doesn't need to worry
# about the database structure, making our code more modular.
# ---

# Define the path to the database file. We place it in the top-level 'data' folder.
# os.path.join is used to create a path that works on any operating system (Windows, macOS, Linux).

# -----------------------------------------------------------------------------
# 1) Database path (override with env var if needed) + ensure ./data exists
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = Path(os.environ.get("MSC_DB_FILE", str(DATA_DIR / "redtube_videos.db"))).resolve()

# -----------------------------------------------------------------------------
# 2) Connection helper with safe defaults (WAL, timeouts, foreign keys, row factory)
# -----------------------------------------------------------------------------

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """
    Apply SQLite PRAGMAs for safer concurrency and consistency.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open SQLite connection to configure.

    Notes
    -----
    - WAL mode improves writer/reader concurrency.
    - busy_timeout reduces 'database is locked' errors under contention.
    - foreign_keys enforces referential integrity across tables.
    """
    t0 = time.perf_counter()
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=5000;")  # ms
    cur.execute("PRAGMA foreign_keys=ON;")
    cur.close()
    dt = time.perf_counter() - t0
    print(f"[TIME] database._apply_pragmas: {dt:.3f}s")


def create_connection(check_same_thread: bool = True) -> Optional[sqlite3.Connection]:
    """
    Create (or open) a SQLite connection with safe defaults.

    Parameters
    ----------
    check_same_thread : bool, default True
        If False, allows sharing the connection across threads (use with care).

    Returns
    -------
    Optional[sqlite3.Connection]
        Open connection or None on failure.

    Notes
    -----
    - Respects ENV override MSC_DB_FILE to change database location.
    - Prints elapsed time.
    """
    t0 = time.perf_counter()
    try:
        conn = sqlite3.connect(
            str(DB_FILE),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=check_same_thread,
        )
        conn.row_factory = sqlite3.Row  # dict-like row access: row["title"]
        _apply_pragmas(conn)
        dt = time.perf_counter() - t0
        print(f"Successfully connected to SQLite database at {DB_FILE}")
        print(f"[TIME] database.create_connection: {dt:.2f}s")
        return conn
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None


@contextmanager
def get_conn(check_same_thread: bool = True) -> Iterator[sqlite3.Connection]:
    """
    Context-managed SQLite connection with auto-commit/rollback.

    Parameters
    ----------
    check_same_thread : bool, default True
        Passed through to `create_connection`.

    Yields
    ------
    sqlite3.Connection
        Open connection. Commits on normal exit; rolls back on exception.

    Notes
    -----
    Prints open/close timings.
    """
    t_open = time.perf_counter()
    conn = create_connection(check_same_thread=check_same_thread)
    if conn is None:
        raise RuntimeError("Failed to open database connection.")
    print(f"[TIME] database.get_conn(open)] {time.perf_counter() - t_open:.2f}s")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        t_close = time.perf_counter()
        conn.close()
        print(f"[TIME] database.get_conn(close)] {time.perf_counter() - t_close:.2f}s")


# -----------------------------------------------------------------------------
# 3) Schema (normalized for fairness slicing + resumable collection)
#    - videos: main facts
#    - video_tags: 1:N tags (enables clean group/audit slices)
#    - video_categories: 1:N categories
#    - audit_terms: curated identity terms (race/gender/orientation)
#    - collection_state: track API daily cap + resume pointers
# -----------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    url             TEXT,
    duration        INTEGER,             -- seconds (prefer INTEGER over TEXT)
    views           INTEGER,
    rating          REAL,
    ratings         INTEGER,
    publish_date    TEXT,                -- ISO-8601 string (YYYY-MM-DD)
    category_source TEXT,
    is_active       INTEGER,             -- 0/1
    retrieved_at    TEXT NOT NULL        -- ISO-8601 UTC timestamp
);

CREATE TABLE IF NOT EXISTS video_tags (
    video_id    INTEGER NOT NULL,
    tag         TEXT    NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    UNIQUE(video_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_video_tags_tag ON video_tags(tag);

CREATE TABLE IF NOT EXISTS video_categories (
    video_id    INTEGER NOT NULL,
    category    TEXT    NOT NULL,
    FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
    UNIQUE(video_id, category)
);

CREATE INDEX IF NOT EXISTS idx_video_categories_category  ON video_categories(category);
CREATE INDEX IF NOT EXISTS idx_video_categories_video_id   ON video_categories(video_id);

CREATE TABLE IF NOT EXISTS audit_terms (
    term    TEXT PRIMARY KEY,
    group_name TEXT NOT NULL            -- e.g., 'race', 'gender', 'orientation'
);

CREATE TABLE IF NOT EXISTS collection_state (
    day                 TEXT PRIMARY KEY,    -- YYYY-MM-DD (UTC)
    requests_used       INTEGER NOT NULL DEFAULT 0,
    last_page_fetched   INTEGER NOT NULL DEFAULT 0,
    reset_at            TEXT NOT NULL        -- ISO-8601 UTC timestamp of next reset
);

-- Helpful read paths
CREATE INDEX IF NOT EXISTS idx_videos_publish_date ON videos(publish_date);
CREATE INDEX IF NOT EXISTS idx_videos_is_active ON videos(is_active);
"""

def load_data_from_db(db_path: Path) -> pd.DataFrame:
    """
    Load current video data with aggregated tags from the SQLite database.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

    Returns
    -------
    pd.DataFrame
        Joined dataframe with columns from `videos` and a `tags` column.

    Notes
    -----
    Prints elapsed time and number of records loaded.
    """
    t0 = time.perf_counter()
    print(f"Connecting to database at: {db_path}...")
    try:
        with sqlite3.connect(str(db_path)) as con:
            query = """
            SELECT
                v.*,
                t.tags
            FROM
                videos v
            LEFT JOIN
                (SELECT video_id, GROUP_CONCAT(tag) as tags FROM video_tags GROUP BY video_id) t
            ON v.video_id = t.video_id;
            """
            df = pd.read_sql_query(query, con)
        print(f"Successfully loaded {len(df):,} video records.")
        dt = time.perf_counter() - t0
        print(f"[TIME] database.load_data_from_db: {dt:.2f}s")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Ensure required tables and indexes exist (idempotent).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to execute schema DDL.

    Notes
    -----
    Prints elapsed time.
    """
    t0 = time.perf_counter()
    try:
        conn.executescript(SCHEMA_SQL)
        print("Schema ensured: videos, video_tags, video_categories, audit_terms, collection_state.")
    except Error as e:
        print(f"Error creating tables: {e}")
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIME] database.create_tables: {dt:.2f}s")

# -----------------------------------------------------------------------------
# 4) Setup entrypoint (idempotent)
# -----------------------------------------------------------------------------
def setup_database() -> None:
    """
    Create the database file (if needed) and ensure schema is present.

    Notes
    -----
    Prints elapsed time for the whole setup.
    """
    t0 = time.perf_counter()
    with get_conn() as conn:
        create_tables(conn)
    print(f"[TIME] database.setup_database: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    # Self-check: initialize DB, then print a small sanity query.
    setup_database()
    try:
        with sqlite3.connect(str(DB_FILE)) as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='videos';")
            has_videos = cur.fetchone()[0]
            if has_videos:
                cur.execute("SELECT COUNT(*) FROM videos;")
                n = cur.fetchone()[0]
                print(f"[CHECK] videos table present with {n} rows.")
            else:
                print("[CHECK] videos table missing (unexpected).")
    except Exception as e:
        print(f"[CHECK] failed: {e}")
