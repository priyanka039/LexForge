# ─────────────────────────────────────────────
# database.py
# SQLite database for LexForge
#
# Stores:
#   cases    → your matters / client files
#   sessions → every piece of work done
#              (research, arguments, etc.)
#
# To switch to PostgreSQL later:
#   1. pip install psycopg2-binary
#   2. Change get_conn() to use psycopg2
#   3. Change AUTOINCREMENT → SERIAL
#   4. Everything else stays the same
# ─────────────────────────────────────────────

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "../data/lexforge.db"


def get_conn():
    """Get a database connection. Creates the DB file if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
    return conn


def init_db():
    """
    Create all tables if they don't exist.
    Safe to call on every startup.
    """
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cases (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            description TEXT    DEFAULT '',
            area_of_law TEXT    DEFAULT 'General',
            client_name TEXT    DEFAULT '',
            status      TEXT    DEFAULT 'active',
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id      INTEGER REFERENCES cases(id) ON DELETE SET NULL,
            session_type TEXT    NOT NULL,
            title        TEXT    NOT NULL,
            input_data   TEXT    DEFAULT '{}',
            output_data  TEXT    DEFAULT '{}',
            notes        TEXT    DEFAULT '',
            created_at   TEXT    DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_case    ON sessions(case_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_type    ON sessions(session_type);
        CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
    """)
    conn.commit()
    conn.close()


# ═════════════════════════════════════════════
# CASES
# ═════════════════════════════════════════════

def create_case(name: str, description: str = "", area_of_law: str = "General", client_name: str = "") -> dict:
    conn = get_conn()
    cur  = conn.execute(
        "INSERT INTO cases (name, description, area_of_law, client_name) VALUES (?,?,?,?)",
        (name, description, area_of_law, client_name)
    )
    conn.commit()
    row = conn.execute("SELECT * FROM cases WHERE id=?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def get_all_cases() -> list:
    conn  = get_conn()
    rows  = conn.execute("SELECT * FROM cases ORDER BY updated_at DESC").fetchall()
    cases = []
    for row in rows:
        c = dict(row)
        # Count sessions for each case
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE case_id=?", (c['id'],)
        ).fetchone()
        c['session_count'] = count['cnt'] if count else 0
        cases.append(c)
    conn.close()
    return cases


def get_case(case_id: int) -> dict | None:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_case(case_id: int, **kwargs) -> dict | None:
    allowed = {'name', 'description', 'area_of_law', 'client_name', 'status'}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_case(case_id)
    updates['updated_at'] = datetime.now().isoformat()
    sets   = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [case_id]
    conn   = get_conn()
    conn.execute(f"UPDATE cases SET {sets} WHERE id=?", values)
    conn.commit()
    row = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_case(case_id: int) -> bool:
    conn = get_conn()
    conn.execute("DELETE FROM cases WHERE id=?", (case_id,))
    conn.commit()
    conn.close()
    return True


# ═════════════════════════════════════════════
# SESSIONS
# ═════════════════════════════════════════════

def save_session(
    session_type: str,
    title:        str,
    input_data:   dict,
    output_data:  dict,
    case_id:      int | None = None,
    notes:        str = ""
) -> dict:
    conn = get_conn()
    cur  = conn.execute(
        """INSERT INTO sessions
           (case_id, session_type, title, input_data, output_data, notes)
           VALUES (?,?,?,?,?,?)""",
        (
            case_id,
            session_type,
            title,
            json.dumps(input_data),
            json.dumps(output_data),
            notes
        )
    )
    conn.commit()

    # Touch the case updated_at
    if case_id:
        conn.execute(
            "UPDATE cases SET updated_at=? WHERE id=?",
            (datetime.now().isoformat(), case_id)
        )
        conn.commit()

    row = conn.execute("SELECT * FROM sessions WHERE id=?", (cur.lastrowid,)).fetchone()
    conn.close()
    return _parse_session(dict(row))


def get_session(session_id: int) -> dict | None:
    conn = get_conn()
    row  = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return _parse_session(dict(row)) if row else None


def get_sessions(case_id: int | None = None, session_type: str | None = None, limit: int = 50) -> list:
    conn   = get_conn()
    query  = "SELECT * FROM sessions WHERE 1=1"
    params = []
    if case_id is not None:
        query  += " AND case_id=?"
        params.append(case_id)
    if session_type:
        query  += " AND session_type=?"
        params.append(session_type)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_parse_session(dict(r)) for r in rows]


def update_session_notes(session_id: int, notes: str) -> dict | None:
    conn = get_conn()
    conn.execute("UPDATE sessions SET notes=? WHERE id=?", (notes, session_id))
    conn.commit()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return _parse_session(dict(row)) if row else None


def update_session_case(session_id: int, case_id: int | None) -> dict | None:
    conn = get_conn()
    conn.execute("UPDATE sessions SET case_id=? WHERE id=?", (case_id, session_id))
    conn.commit()
    row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    conn.close()
    return _parse_session(dict(row)) if row else None


def delete_session(session_id: int) -> bool:
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()
    return True


def _parse_session(row: dict) -> dict:
    """Parse JSON fields in a session row."""
    try:
        row['input_data']  = json.loads(row.get('input_data',  '{}') or '{}')
    except Exception:
        row['input_data']  = {}
    try:
        row['output_data'] = json.loads(row.get('output_data', '{}') or '{}')
    except Exception:
        row['output_data'] = {}
    return row