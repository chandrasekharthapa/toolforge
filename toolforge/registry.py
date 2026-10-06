"""SQLite-backed tool library, lesson memory and run log.

Tools are versioned: registering a tool under an existing name creates version n+1
and marks the previous version ``superseded``, so history (and provenance) is kept.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .models import Lesson, RunResult, Tool, ToolDraft, Verification

SCHEMA = """
CREATE TABLE IF NOT EXISTS tools (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    version       INTEGER NOT NULL,
    description   TEXT NOT NULL,
    parameters    TEXT NOT NULL,
    code          TEXT NOT NULL,
    tests         TEXT NOT NULL,
    verification  TEXT NOT NULL DEFAULT '{}',
    embedding     TEXT,
    status        TEXT NOT NULL DEFAULT 'active',
    uses          INTEGER NOT NULL DEFAULT 0,
    successes     INTEGER NOT NULL DEFAULT 0,
    failures      INTEGER NOT NULL DEFAULT 0,
    origin_task   TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    UNIQUE (name, version)
);
CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    need        TEXT NOT NULL,
    mistake     TEXT NOT NULL,
    fix         TEXT NOT NULL,
    embedding   TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    task           TEXT NOT NULL,
    answer         TEXT,
    created        TEXT,
    reused         TEXT,
    failed_needs   TEXT,
    llm_calls      INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    latency_s      REAL,
    trace          TEXT,
    created_at     TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: str = "toolforge.db") -> None:
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ tools
    def add_tool(self, draft: ToolDraft, embedding: np.ndarray | None, *, origin_task: str = "",
                 verification: Verification | None = None) -> Tool:
        with self._lock:
            row = self.conn.execute("SELECT MAX(version) FROM tools WHERE name = ?", (draft.name,)).fetchone()
            version = (row[0] or 0) + 1
            self.conn.execute("UPDATE tools SET status='superseded' WHERE name=? AND status='active'",
                              (draft.name,))
            cur = self.conn.execute(
                "INSERT INTO tools (name, version, description, parameters, code, tests, verification,"
                " embedding, origin_task, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    draft.name, version, draft.description, json.dumps(draft.parameters), draft.code,
                    json.dumps([t.model_dump() for t in draft.tests]),
                    (verification or Verification()).model_dump_json(),
                    json.dumps(embedding.tolist()) if embedding is not None else None,
                    origin_task, _now(),
                ),
            )
            self.conn.commit()
            return self.get_by_id(cur.lastrowid)

    def _to_tool(self, row: sqlite3.Row) -> Tool:
        return Tool(
            id=row["id"], name=row["name"], version=row["version"], description=row["description"],
            parameters=json.loads(row["parameters"]), code=row["code"], tests=json.loads(row["tests"]),
            verification=Verification.model_validate_json(row["verification"] or "{}"),
            status=row["status"], uses=row["uses"], successes=row["successes"],
            failures=row["failures"], origin_task=row["origin_task"], created_at=row["created_at"],
        )

    def get_by_id(self, tool_id: int) -> Tool:
        row = self.conn.execute("SELECT * FROM tools WHERE id=?", (tool_id,)).fetchone()
        if row is None:
            raise KeyError(tool_id)
        return self._to_tool(row)

    def get(self, name: str) -> Tool | None:
        """Latest active version of a tool."""
        row = self.conn.execute(
            "SELECT * FROM tools WHERE name=? AND status='active' ORDER BY version DESC LIMIT 1", (name,)
        ).fetchone()
        return self._to_tool(row) if row else None

    def versions(self, name: str) -> list[Tool]:
        rows = self.conn.execute("SELECT * FROM tools WHERE name=? ORDER BY version", (name,)).fetchall()
        return [self._to_tool(r) for r in rows]

    def list_tools(self, status: str | None = "active") -> list[Tool]:
        if status:
            rows = self.conn.execute("SELECT * FROM tools WHERE status=? ORDER BY name", (status,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM tools ORDER BY name, version").fetchall()
        return [self._to_tool(r) for r in rows]

    def tool_embeddings(self) -> list[tuple[Tool, np.ndarray | None]]:
        rows = self.conn.execute("SELECT * FROM tools WHERE status='active'").fetchall()
        return [(self._to_tool(r), np.array(json.loads(r["embedding"]), dtype=np.float32)
                 if r["embedding"] else None) for r in rows]

    def set_embedding(self, tool_id: int, embedding: np.ndarray) -> None:
        with self._lock:
            self.conn.execute("UPDATE tools SET embedding=? WHERE id=?",
                              (json.dumps(embedding.tolist()), tool_id))
            self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value="
                              "excluded.value", (key, value))
            self.conn.commit()

    def set_lesson_embedding(self, lesson_id: int, embedding: np.ndarray) -> None:
        with self._lock:
            self.conn.execute("UPDATE lessons SET embedding=? WHERE id=?", (json.dumps(embedding.tolist()), lesson_id))
            self.conn.commit()

    def record_use(self, tool_id: int, success: bool) -> None:
        col = "successes" if success else "failures"
        with self._lock:
            self.conn.execute(f"UPDATE tools SET uses=uses+1, {col}={col}+1 WHERE id=?", (tool_id,))
            self.conn.commit()

    def set_status(self, name: str, status: str) -> int:
        with self._lock:
            cur = self.conn.execute("UPDATE tools SET status=? WHERE name=? AND status='active'", (status, name))
            self.conn.commit()
            return cur.rowcount

    # ---------------------------------------------------------------- lessons
    def add_lesson(self, lesson: Lesson, embedding: np.ndarray | None) -> Lesson:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO lessons (need, mistake, fix, embedding, created_at) VALUES (?,?,?,?,?)",
                (lesson.need, lesson.mistake, lesson.fix,
                 json.dumps(embedding.tolist()) if embedding is not None else None, _now()),
            )
            self.conn.commit()
            return lesson.model_copy(update={"id": cur.lastrowid})

    def lesson_embeddings(self) -> list[tuple[Lesson, np.ndarray | None]]:
        rows = self.conn.execute("SELECT * FROM lessons").fetchall()
        return [(Lesson(id=r["id"], need=r["need"], mistake=r["mistake"], fix=r["fix"],
                        created_at=r["created_at"]),
                 np.array(json.loads(r["embedding"]), dtype=np.float32) if r["embedding"] else None)
                for r in rows]

    def list_lessons(self) -> list[Lesson]:
        return [lesson for lesson, _ in self.lesson_embeddings()]

    # ------------------------------------------------------------------- runs
    def log_run(self, result: RunResult) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO runs (task, answer, created, reused, failed_needs, llm_calls, input_tokens,"
                " output_tokens, latency_s, trace, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (result.task, result.answer, json.dumps(result.created), json.dumps(result.reused),
                 json.dumps(result.failed_needs), result.llm_calls, result.input_tokens,
                 result.output_tokens, result.latency_s, json.dumps(result.trace, default=str), _now()),
            )
            self.conn.commit()
            return cur.lastrowid

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("created", "reused", "failed_needs", "trace"):
                d[k] = json.loads(d[k] or "[]")
            out.append(d)
        return out

    def stats(self) -> dict[str, Any]:
        q = self.conn.execute
        runs = q("SELECT COUNT(*), COALESCE(SUM(input_tokens+output_tokens),0), COALESCE(AVG(latency_s),0)"
                 " FROM runs").fetchone()
        created = sum(len(json.loads(r[0] or "[]")) for r in q("SELECT created FROM runs"))
        reused = sum(len(json.loads(r[0] or "[]")) for r in q("SELECT reused FROM runs"))
        return {
            "active_tools": q("SELECT COUNT(*) FROM tools WHERE status='active'").fetchone()[0],
            "tool_versions": q("SELECT COUNT(*) FROM tools").fetchone()[0],
            "lessons": q("SELECT COUNT(*) FROM lessons").fetchone()[0],
            "runs": runs[0],
            "total_tokens": runs[1],
            "avg_latency_s": round(runs[2], 3),
            "tools_created": created,
            "tools_reused": reused,
            "reuse_rate": round(reused / (created + reused), 3) if created + reused else None,
            "tool_calls": q("SELECT COALESCE(SUM(uses),0) FROM tools").fetchone()[0],
        }

    def close(self) -> None:
        self.conn.close()


