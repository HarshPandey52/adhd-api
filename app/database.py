"""
app/database.py
================
SQLite-backed storage for patient visit history.

Prototype-stage storage: file-based SQLite (model/patients.db).
Designed so swapping to Postgres later only requires changing
the connection string — the query layer (db.py functions) stays the same.

⚠️ PROTOTYPE NOTICE: No authentication, no encryption-at-rest beyond
the filesystem. Do NOT use with real patient data until:
  1. Doctor login/auth is added
  2. Database is moved to a managed Postgres instance with backups
  3. Data handling is reviewed against applicable health-data law
     (HIPAA, DPDP Act, etc. depending on jurisdiction)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone
from contextlib import contextmanager

DB_DIR  = os.path.join(os.path.dirname(__file__), "..", "model")
DB_PATH = os.path.join(DB_DIR, "patients.db")


def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                visit_date TEXT NOT NULL,
                prediction_json TEXT,
                doctor_notes TEXT,
                prescription TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patient_id ON visits(patient_id)
        """)
        conn.commit()


@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def save_visit(patient_id: str, prediction: dict | None,
               doctor_notes: str | None, prescription: str | None) -> dict:
    """Inserts a new visit record for a patient. Returns the saved record."""
    now = datetime.now(timezone.utc).isoformat()
    prediction_json = json.dumps(prediction) if prediction else None

    with _get_conn() as conn:
        cursor = conn.execute(
            """INSERT INTO visits
               (patient_id, visit_date, prediction_json, doctor_notes, prescription, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (patient_id, now, prediction_json, doctor_notes, prescription, now)
        )
        conn.commit()
        visit_id = cursor.lastrowid

    return {
        "id": visit_id,
        "patient_id": patient_id,
        "visit_date": now,
        "prediction": prediction,
        "doctor_notes": doctor_notes,
        "prescription": prescription,
    }


def get_patient_history(patient_id: str) -> list[dict]:
    """Returns all visits for a patient, most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM visits WHERE patient_id = ?
               ORDER BY visit_date DESC""",
            (patient_id,)
        ).fetchall()

    visits = []
    for row in rows:
        visits.append({
            "id": row["id"],
            "patient_id": row["patient_id"],
            "visit_date": row["visit_date"],
            "prediction": json.loads(row["prediction_json"]) if row["prediction_json"] else None,
            "doctor_notes": row["doctor_notes"],
            "prescription": row["prescription"],
        })
    return visits


def update_visit_notes(visit_id: int, doctor_notes: str | None,
                        prescription: str | None) -> dict | None:
    """Updates notes/prescription on an existing visit (e.g. doctor adds notes after analysis)."""
    with _get_conn() as conn:
        conn.execute(
            """UPDATE visits SET doctor_notes = ?, prescription = ?
               WHERE id = ?""",
            (doctor_notes, prescription, visit_id)
        )
        conn.commit()

        row = conn.execute(
            "SELECT * FROM visits WHERE id = ?", (visit_id,)
        ).fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "visit_date": row["visit_date"],
        "prediction": json.loads(row["prediction_json"]) if row["prediction_json"] else None,
        "doctor_notes": row["doctor_notes"],
        "prescription": row["prescription"],
    }


def list_all_patients() -> list[str]:
    """Returns distinct patient IDs that have at least one visit."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT patient_id FROM visits ORDER BY patient_id"
        ).fetchall()
    return [row["patient_id"] for row in rows]
