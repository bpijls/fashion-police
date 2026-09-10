"""SQLite store for predictions + feedback.

Only anonymised data is written: the white-background / black-face overlay from
the compute tier, the ranked labels, and any correction the visitor taps in.
The raw uploaded frame is never persisted.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    record_id       TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    created_ts      REAL NOT NULL,
    top_label       TEXT NOT NULL,
    top_score       REAL NOT NULL,
    uncertain       INTEGER NOT NULL DEFAULT 0,
    ranked_json     TEXT NOT NULL,
    overlay_path    TEXT,
    user_correction TEXT,
    feedback_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_ts ON predictions(created_ts);
"""


def _now() -> tuple[str, float]:
    dt = datetime.now(timezone.utc)
    return dt.isoformat(), dt.timestamp()


class Store:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    async def _run(self, fn, *args):
        async with self._lock:
            return await asyncio.to_thread(fn, *args)

    # --- writes -----------------------------------------------------------

    async def save_prediction(
        self,
        record_id: str,
        *,
        top_label: str,
        top_score: float,
        uncertain: bool,
        ranked: list[dict],
        overlay_path: str | None,
    ) -> None:
        created_at, created_ts = _now()

        def _write():
            self._db.execute(
                "INSERT INTO predictions "
                "(record_id, created_at, created_ts, top_label, top_score, uncertain, ranked_json, overlay_path) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    record_id, created_at, created_ts, top_label, float(top_score),
                    1 if uncertain else 0, json.dumps(ranked), overlay_path,
                ),
            )
            self._db.commit()

        await self._run(_write)

    async def save_feedback(self, record_id: str, correct_label: str) -> bool:
        feedback_at, _ = _now()

        def _write() -> bool:
            cur = self._db.execute(
                "UPDATE predictions SET user_correction=?, feedback_at=? WHERE record_id=?",
                (correct_label, feedback_at, record_id),
            )
            self._db.commit()
            return cur.rowcount > 0

        return await self._run(_write)

    # --- reads ----------------------------------------------------------

    async def statistics(self) -> dict:
        def _read() -> dict:
            rows = self._db.execute(
                "SELECT top_label, user_correction FROM predictions"
            ).fetchall()
            total = len(rows)
            fb = sum(1 for r in rows if r["user_correction"])
            top: dict[str, int] = {}
            corr: dict[str, int] = {}
            for r in rows:
                top[r["top_label"]] = top.get(r["top_label"], 0) + 1
                if r["user_correction"]:
                    corr[r["user_correction"]] = corr.get(r["user_correction"], 0) + 1
            return {
                "total_predictions": total,
                "total_feedback": fb,
                "feedback_rate": (fb / total) if total else 0.0,
                "top_predictions": sorted(top.items(), key=lambda x: x[1], reverse=True),
                "user_corrections": sorted(corr.items(), key=lambda x: x[1], reverse=True),
            }

        return await self._run(_read)

    # --- retention ----------------------------------------------------

    async def prune_older_than(self, cutoff_ts: float) -> list[str]:
        """Delete rows older than cutoff; return the overlay paths that were removed."""

        def _prune() -> list[str]:
            rows = self._db.execute(
                "SELECT overlay_path FROM predictions WHERE created_ts < ? AND overlay_path IS NOT NULL",
                (cutoff_ts,),
            ).fetchall()
            self._db.execute("DELETE FROM predictions WHERE created_ts < ?", (cutoff_ts,))
            self._db.commit()
            return [r["overlay_path"] for r in rows]

        return await self._run(_prune)
