"""Experiment registry : SQLite log de runs + artifacts trajectoires/metrics."""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "cache" / "experiments.db"


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            source TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            encoder TEXT,
            n_points INTEGER,
            latent_dim INTEGER,
            tags TEXT,
            payload_path TEXT
        );
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_created ON experiments(created_at DESC);")


def save_experiment(
    source: str,
    source_kind: str,
    payload: dict[str, Any],
    encoder: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Persiste un run. payload = full bundle JSON (coords + metrics)."""
    init_db()
    exp_id = uuid.uuid4().hex[:12]
    now = time.time()
    n_points = int(payload.get("n", len(payload.get("coords", [])) or 0))
    latent_dim = payload.get("latent_dim")
    payload_dir = DB_PATH.parent / "experiments"
    payload_dir.mkdir(exist_ok=True)
    fp = payload_dir / f"{exp_id}.json"
    fp.write_text(json.dumps(payload))

    with _conn() as c:
        c.execute(
            "INSERT INTO experiments (id, created_at, source, source_kind, encoder, n_points, latent_dim, tags, payload_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (exp_id, now, source, source_kind, encoder, n_points, latent_dim,
             json.dumps(tags or []), str(fp)),
        )
    return exp_id


def list_experiments(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT id, created_at, source, source_kind, encoder, n_points, latent_dim, tags "
            "FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "source": r["source"],
            "source_kind": r["source_kind"],
            "encoder": r["encoder"],
            "n_points": r["n_points"],
            "latent_dim": r["latent_dim"],
            "tags": json.loads(r["tags"] or "[]"),
        }
        for r in rows
    ]


def get_experiment(exp_id: str) -> dict | None:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT payload_path FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
    if not row:
        return None
    fp = Path(row["payload_path"])
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


def delete_experiment(exp_id: str) -> bool:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT payload_path FROM experiments WHERE id = ?", (exp_id,)
        ).fetchone()
        if not row:
            return False
        fp = Path(row["payload_path"])
        if fp.exists():
            fp.unlink()
        c.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
    return True
