from __future__ import annotations

"""
Worker_tandem v0.1
==================

Separate controller for a Codex <-> Cline tandem workflow.

Principles
----------
- This file does NOT modify or depend on Mini Worker internals.
- Its own runtime state lives in <project>/.worker_tandem/.
- Python + SQLite own state transitions and audit/event history.
- Codex is Planner/Reviewer only.
- Cline is the implementation worker.
- Neither Codex nor Cline writes tandem_state.db directly.
- Migration from Mini Worker is READ-ONLY and intended after STEP 2 is finished.

Typical workflow
----------------
1. Open project.
2. "Import from Mini Worker" after Mini Worker is closed and STEP 2 is complete.
3. Current step becomes READY_FOR_CODEX_PLAN.
4. Run Codex PLAN.
5. Human approves the plan and dispatches Cline.
6. Process Cline report.
7. Run Codex REVIEW.
8. APPROVE -> human accepts review -> next step.
   REVISE  -> dispatch repair to Cline -> review again.
"""

import json
import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional


APP_NAME = "Worker Tandem"
APP_VERSION = "0.1.2"
APP_DIR = ".worker_tandem"
DB_NAME = "tandem_state.db"

MINI_DIR = ".mini_build"
MINI_DB_NAME = "build_state.db"

DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_CODEX_TIMEOUT_SEC = 1800

#: PLAN/REVIEW sandbox. Codex must never be able to write project files here.
DEFAULT_CODEX_SANDBOX = "read-only"

#: Optional explicit Codex model passed as `-m <model>`.
#: None -> use the Codex configuration/default model.
#: Worker_tandem NEVER edits ~/.codex/config.toml automatically.
DEFAULT_CODEX_MODEL: Optional[str] = None

#: Reasoning-effort policy. Default PLAN/REVIEW is "low"; escalated only when
#: the situation genuinely demands it (see TandemController._codex_reasoning_effort).
CODEX_REASONING_LOW = "low"
CODEX_REASONING_MEDIUM = "medium"
CODEX_REASONING_HIGH = "high"
DEFAULT_CODEX_REASONING = CODEX_REASONING_LOW
VALID_CODEX_REASONING = {
    CODEX_REASONING_LOW,
    CODEX_REASONING_MEDIUM,
    CODEX_REASONING_HIGH,
}

#: Shared, mandatory prompt contract that keeps Codex cheap and to the point.
CODEX_EFFICIENCY_CONTRACT = """\
EFFICIENCY CONTRACT (mandatory):
- Explore the repository as little as possible; read only files required for THIS step.
- Do not print preamble, status commentary, or a plan of action.
- Do not repeat the context, the rules, or the step description.
- Output ONLY the requested JSON object and nothing else.
- summary: at most 5 sentences.
- issues and required_changes: at most 5 items each.
- Stop as soon as sufficient evidence has been obtained."""

#: Compact rule lists carried inside the minimized context JSON.
PLAN_CONTEXT_RULES = [
    "PLAN ONLY. Do not edit project files.",
    "Minimal repository exploration; read only what THIS step needs.",
    "Respect existing architecture invariants and backward compatibility.",
    "Return only the requested structured plan.",
    "If a required architecture decision is unresolved, set decision=BLOCKED.",
]

REVIEW_CONTEXT_RULES = [
    "REVIEW ONLY. Do not edit project files.",
    "Compare the approved plan, the Cline report, and git diff/checks.",
    "Do not approve based only on Cline's claim.",
    "Return only the requested structured review.",
    "If a required architecture decision is unresolved, set "
    "verdict=ARCHITECTURE_DECISION_REQUIRED.",
]

#: Canonical, human-editable plan document that lives with the project.
#: Worker_tandem READS it to sync step descriptions; the tandem DB keeps the
#: active imported/snapshotted execution state, NOT the editable plan. Mini
#: Worker is no longer the authoritative plan source.
CANONICAL_PLAN_NAME = "BUILD_PLAN.json"

#: Bounds for the minimized PLAN context. Keep it small: the full current step,
#: the locked controller rules, only the necessary previous VERIFIED steps, and
#: reference file NAMES - never repository or document dumps.
MAX_PREVIOUS_STEPS = 3
MAX_PREVIOUS_STEP_DESCRIPTION_CHARS = 400
MAX_PLAN_CONTEXT_CHARS = 20000

TERMINAL_STATES = {"VERIFIED", "SKIPPED", "ABORTED"}

VALID_STATES = {
    "PENDING",
    "READY_FOR_CODEX_PLAN",
    "CODEX_PLAN_RUNNING",
    "PLAN_READY",
    "WAITING_HUMAN_APPROVAL",
    "CLINE_DISPATCHED",
    "CLINE_REPORT_RECEIVED",
    "CODEX_REVIEW_RUNNING",
    "REVIEW_APPROVED",
    "REVISE",
    "BLOCKED",
    "FAILED",
    "VERIFIED",
    "SKIPPED",
    "ABORTED",
}

ALLOWED_TRANSITIONS = {
    "PENDING": {"READY_FOR_CODEX_PLAN", "SKIPPED", "ABORTED"},
    "READY_FOR_CODEX_PLAN": {"CODEX_PLAN_RUNNING", "BLOCKED", "ABORTED"},
    "CODEX_PLAN_RUNNING": {"PLAN_READY", "BLOCKED", "FAILED"},
    "PLAN_READY": {"WAITING_HUMAN_APPROVAL", "BLOCKED", "ABORTED"},
    "WAITING_HUMAN_APPROVAL": {"CLINE_DISPATCHED", "BLOCKED", "ABORTED"},
    "CLINE_DISPATCHED": {"CLINE_REPORT_RECEIVED", "BLOCKED", "FAILED"},
    "CLINE_REPORT_RECEIVED": {"CODEX_REVIEW_RUNNING", "BLOCKED", "FAILED"},
    "CODEX_REVIEW_RUNNING": {"REVIEW_APPROVED", "REVISE", "BLOCKED", "FAILED"},
    "REVIEW_APPROVED": {"VERIFIED", "BLOCKED", "ABORTED"},
    "REVISE": {"CLINE_DISPATCHED", "BLOCKED", "ABORTED"},
    "BLOCKED": {"READY_FOR_CODEX_PLAN", "ABORTED"},
    "FAILED": {"READY_FOR_CODEX_PLAN", "ABORTED"},
    "VERIFIED": set(),
    "SKIPPED": set(),
    "ABORTED": set(),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_text(path: Path, max_chars: int = 120_000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[TRUNCATED BY WORKER_TANDEM]..."
    return text


def git_capture(project_root: Path, args: list[str], max_chars: int = 120_000) -> str:
    git = shutil.which("git")
    if not git:
        return "git executable not found"
    try:
        cp = subprocess.run(
            [git, *args],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            shell=False,
        )
        text = (cp.stdout or "") + (("\nSTDERR:\n" + cp.stderr) if cp.stderr else "")
        return text[:max_chars]
    except Exception as exc:
        return f"git command failed: {exc}"


def is_sufficient_step_description(description: str, title: str = "") -> bool:
    """Whether a step description is a real task specification.

    A title-only step is NOT sufficient: the planner must never be sent the
    step title as if it were the task. Empty/whitespace descriptions and a
    description that normalized-equals the title are rejected.
    """
    text = (description or "").strip()
    if not text:
        return False
    if title:
        norm_desc = " ".join(text.casefold().split())
        norm_title = " ".join((title or "").strip().casefold().split())
        if norm_desc == norm_title:
            return False
    return True


@dataclass(slots=True)
class TandemStep:
    step_no: int
    phase: str
    title: str
    description: str
    state: str
    attempt: int
    max_attempts: int
    risk: str
    requires_human: bool


class TandemDB:
    """Authoritative Worker_tandem SQLite state."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.workdir = self.project_root / APP_DIR
        self.to_cline = self.workdir / "to_cline"
        self.from_cline = self.workdir / "from_cline"
        self.to_codex = self.workdir / "to_codex"
        self.from_codex = self.workdir / "from_codex"
        self.context_dir = self.workdir / "context"
        self.archive = self.workdir / "archive"
        self.backups = self.workdir / "backups"
        self.logs = self.workdir / "logs"

        for p in (
            self.workdir,
            self.to_cline,
            self.from_cline,
            self.to_codex,
            self.from_codex,
            self.context_dir,
            self.archive,
            self.backups,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)

        self.db_path = self.workdir / DB_NAME

        # SQLite connections are thread-affine by default. Worker_tandem runs
        # Codex in a background thread so the Tk GUI stays responsive.
        # Give each thread its own SQLite connection instead of sharing one
        # connection across Tk/main and worker threads.
        self._db_local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.RLock()

        # Initialize schema on the main/Tk thread connection.
        self._migrate()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        with self._connections_lock:
            self._connections.append(conn)
        return conn

    @property
    def db(self) -> sqlite3.Connection:
        conn = getattr(self._db_local, "connection", None)
        if conn is None:
            conn = self._new_connection()
            self._db_local.connection = conn
        return conn

    def close(self) -> None:
        # Close every per-thread connection created during this session.
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
        try:
            self._db_local.connection = None
        except Exception:
            pass

    def _migrate(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS steps (
                step_no INTEGER PRIMARY KEY,
                phase TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                risk TEXT NOT NULL DEFAULT 'LOW',
                requires_human INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                verified_at TEXT,
                last_update_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS plan_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                plan_hash TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS codex_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_no INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                plan_json TEXT NOT NULL,
                plan_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cline_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_no INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS codex_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_no INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                verdict TEXT NOT NULL,
                review_json TEXT NOT NULL,
                review_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_no INTEGER NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                actor TEXT NOT NULL DEFAULT 'human',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_no INTEGER,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            """
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )
        self.db.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def event(self, step_no: Optional[int], event_type: str, details: str = "") -> None:
        self.db.execute(
            "INSERT INTO events(step_no, event_type, details, created_at) VALUES(?,?,?,?)",
            (step_no, event_type, details, now_iso()),
        )
        self.db.commit()

    def backup(self, target: Optional[Path] = None) -> Path:
        if target is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            target = self.backups / f"worker_tandem_{stamp}.db"
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.db.commit()
        dst = sqlite3.connect(target)
        try:
            self.db.backup(dst)
        finally:
            dst.close()
        self.event(None, "DB_BACKUP", str(target))
        return target

    def integrity_check(self) -> tuple[bool, str]:
        value = self.db.execute("PRAGMA integrity_check").fetchone()[0]
        return value == "ok", str(value)

    def _validate_plan(self, plan: dict) -> None:
        if not isinstance(plan, dict):
            raise ValueError("BUILD_PLAN root must be an object.")
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("BUILD_PLAN must contain a non-empty 'steps' list.")

        seen: set[int] = set()
        for raw in steps:
            if not isinstance(raw, dict):
                raise ValueError("Every step must be an object.")
            if "step_no" not in raw or "title" not in raw:
                raise ValueError("Every step requires step_no and title.")
            n = int(raw["step_no"])
            if n <= 0 or n in seen:
                raise ValueError(f"Invalid/duplicate step_no: {n}")
            seen.add(n)
            risk = str(raw.get("risk", "LOW")).upper()
            if risk not in {"LOW", "MEDIUM", "HIGH"}:
                raise ValueError(f"Invalid risk at step {n}: {risk}")

    def import_plan(self, plan: dict, *, replace: bool = False) -> None:
        self._validate_plan(plan)
        plan_text = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2)
        plan_hash = sha256_text(plan_text)
        existing = self.get_meta("plan_hash", "")
        if existing and existing != plan_hash and not replace:
            raise RuntimeError("Worker_tandem already has a different plan.")

        if existing and existing != plan_hash:
            self.backup()

        with self.db:
            self.db.execute("DELETE FROM steps")
            ts = now_iso()
            for raw in sorted(plan["steps"], key=lambda x: int(x["step_no"])):
                self.db.execute(
                    """
                    INSERT INTO steps(
                        step_no, phase, title, description, state, attempt,
                        max_attempts, risk, requires_human,
                        created_at, last_update_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        int(raw["step_no"]),
                        str(raw.get("phase", "")),
                        str(raw["title"]),
                        str(raw.get("description", "")),
                        "PENDING",
                        0,
                        int(raw.get("max_attempts", 3)),
                        str(raw.get("risk", "LOW")).upper(),
                        1 if raw.get("requires_human", False) else 0,
                        ts,
                        ts,
                    ),
                )

            first = self.db.execute("SELECT MIN(step_no) FROM steps").fetchone()[0]
            if first is not None:
                self.db.execute(
                    "UPDATE steps SET state='READY_FOR_CODEX_PLAN', last_update_at=? WHERE step_no=?",
                    (ts, int(first)),
                )

            self.db.execute(
                "INSERT INTO plan_versions(version, plan_hash, plan_json, created_at) VALUES(?,?,?,?)",
                (
                    str(plan.get("plan_version", plan.get("version", ""))),
                    plan_hash,
                    plan_text,
                    ts,
                ),
            )
            self.db.execute(
                """
                INSERT INTO meta(key,value) VALUES('plan_hash',?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (plan_hash,),
            )
            self.db.execute(
                """
                INSERT INTO meta(key,value) VALUES('plan_version',?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(plan.get("plan_version", plan.get("version", ""))),),
            )

        self.event(None, "PLAN_IMPORTED", f"hash={plan_hash}")

    def migrate_from_mini(self) -> dict:
        """
        Read Mini Worker state WITHOUT modifying it.

        Safety model:
        - source DB is opened mode=ro
        - source steps are copied
        - verified/skipped/aborted states are preserved
        - the first non-terminal source step is reset to READY_FOR_CODEX_PLAN
        - all later non-terminal steps become PENDING
        """
        mini_db = self.project_root / MINI_DIR / MINI_DB_NAME
        if not mini_db.exists():
            raise FileNotFoundError(f"Mini Worker DB not found: {mini_db}")

        uri = f"file:{mini_db.as_posix()}?mode=ro"
        src = sqlite3.connect(uri, uri=True)
        src.row_factory = sqlite3.Row
        try:
            plan_row = src.execute(
                "SELECT version, plan_hash, plan_json FROM plan_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not plan_row:
                raise RuntimeError("Mini Worker contains no plan_versions row.")
            plan = json.loads(plan_row["plan_json"])
            self._validate_plan(plan)

            mini_steps = src.execute(
                """
                SELECT step_no, phase, title, description, state, attempt,
                       max_attempts, risk, requires_human,
                       created_at, started_at, finished_at, verified_at, last_update_at
                FROM steps
                ORDER BY step_no
                """
            ).fetchall()
        finally:
            src.close()

        if not mini_steps:
            raise RuntimeError("Mini Worker has no steps.")

        current_no: Optional[int] = None
        for row in mini_steps:
            if row["state"] not in TERMINAL_STATES:
                current_no = int(row["step_no"])
                break

        # If Mini finished all loaded steps, Worker_tandem keeps them terminal.
        with self.db:
            self.db.execute("DELETE FROM steps")
            self.db.execute("DELETE FROM plan_versions")
            self.db.execute("DELETE FROM codex_plans")
            self.db.execute("DELETE FROM cline_reports")
            self.db.execute("DELETE FROM codex_reviews")
            ts = now_iso()

            for row in mini_steps:
                n = int(row["step_no"])
                source_state = str(row["state"])
                if source_state in TERMINAL_STATES:
                    tandem_state = source_state
                elif current_no is not None and n == current_no:
                    tandem_state = "READY_FOR_CODEX_PLAN"
                else:
                    tandem_state = "PENDING"

                self.db.execute(
                    """
                    INSERT INTO steps(
                        step_no, phase, title, description, state, attempt,
                        max_attempts, risk, requires_human,
                        created_at, started_at, finished_at, verified_at, last_update_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        n,
                        str(row["phase"] or ""),
                        str(row["title"] or ""),
                        str(row["description"] or ""),
                        tandem_state,
                        int(row["attempt"] or 0),
                        int(row["max_attempts"] or 3),
                        str(row["risk"] or "LOW").upper(),
                        int(row["requires_human"] or 0),
                        str(row["created_at"] or ts),
                        row["started_at"],
                        row["finished_at"],
                        row["verified_at"],
                        ts,
                    ),
                )

            plan_text = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2)
            plan_hash = sha256_text(plan_text)
            self.db.execute(
                "INSERT INTO plan_versions(version, plan_hash, plan_json, created_at) VALUES(?,?,?,?)",
                (
                    str(plan.get("plan_version", plan.get("version", ""))),
                    plan_hash,
                    plan_text,
                    ts,
                ),
            )
            for key, value in (
                ("plan_hash", plan_hash),
                ("plan_version", str(plan.get("plan_version", plan.get("version", "")))),
                ("migrated_from_mini", str(mini_db)),
                ("migrated_at", ts),
            ):
                self.db.execute(
                    """
                    INSERT INTO meta(key,value) VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (key, value),
                )

        self.event(
            current_no,
            "MIGRATED_FROM_MINI",
            f"source={mini_db}; current={current_no if current_no is not None else 'all terminal'}",
        )
        return {
            "source": str(mini_db),
            "plan_version": str(plan.get("plan_version", plan.get("version", ""))),
            "current_step": current_no,
            "steps": len(mini_steps),
        }

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> TandemStep:
        return TandemStep(
            step_no=int(row["step_no"]),
            phase=str(row["phase"]),
            title=str(row["title"]),
            description=str(row["description"]),
            state=str(row["state"]),
            attempt=int(row["attempt"]),
            max_attempts=int(row["max_attempts"]),
            risk=str(row["risk"]),
            requires_human=bool(row["requires_human"]),
        )

    def list_steps(self) -> list[TandemStep]:
        rows = self.db.execute("SELECT * FROM steps ORDER BY step_no").fetchall()
        return [self._row_to_step(r) for r in rows]

    def current_step(self) -> Optional[TandemStep]:
        row = self.db.execute(
            """
            SELECT * FROM steps
            WHERE state NOT IN ('VERIFIED','SKIPPED','ABORTED')
            ORDER BY step_no
            LIMIT 1
            """
        ).fetchone()
        return self._row_to_step(row) if row else None

    def get_step(self, step_no: int) -> TandemStep:
        row = self.db.execute("SELECT * FROM steps WHERE step_no=?", (step_no,)).fetchone()
        if not row:
            raise KeyError(step_no)
        return self._row_to_step(row)

    def latest_plan(self) -> Optional[dict]:
        """The active snapshotted plan (controller_policy + steps), or None."""
        row = self.db.execute(
            "SELECT plan_json FROM plan_versions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return json.loads(row["plan_json"]) if row else None

    def plan_policy(self) -> dict:
        """The locked controller rules of the active plan (may be empty)."""
        plan = self.latest_plan() or {}
        policy = plan.get("controller_policy")
        return dict(policy) if isinstance(policy, dict) else {}

    def previous_verified_steps(self, step_no: int) -> list[TandemStep]:
        """VERIFIED steps strictly before ``step_no``, oldest first."""
        rows = self.db.execute(
            """
            SELECT * FROM steps
            WHERE state='VERIFIED' AND step_no < ?
            ORDER BY step_no
            """,
            (step_no,),
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def transition(self, step_no: int, to_state: str, detail: str = "") -> None:
        if to_state not in VALID_STATES:
            raise ValueError(f"Unknown state: {to_state}")
        step = self.get_step(step_no)
        if to_state not in ALLOWED_TRANSITIONS.get(step.state, set()):
            raise RuntimeError(f"Illegal transition: {step.state} -> {to_state}")

        ts = now_iso()
        with self.db:
            self.db.execute(
                "UPDATE steps SET state=?, last_update_at=? WHERE step_no=?",
                (to_state, ts, step_no),
            )
            self.db.execute(
                "INSERT INTO events(step_no,event_type,details,created_at) VALUES(?,?,?,?)",
                (step_no, f"STATE:{step.state}->{to_state}", detail, ts),
            )

    def increment_attempt(self, step_no: int) -> int:
        step = self.get_step(step_no)
        new_attempt = step.attempt + 1
        if new_attempt > step.max_attempts:
            raise RuntimeError(
                f"Step {step_no} exceeded max_attempts ({step.max_attempts})."
            )
        with self.db:
            self.db.execute(
                "UPDATE steps SET attempt=?, started_at=COALESCE(started_at,?), last_update_at=? WHERE step_no=?",
                (new_attempt, now_iso(), now_iso(), step_no),
            )
        return new_attempt

    def save_codex_plan(self, step_no: int, attempt: int, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        with self.db:
            self.db.execute(
                """
                INSERT INTO codex_plans(step_no,attempt,plan_json,plan_hash,created_at)
                VALUES(?,?,?,?,?)
                """,
                (step_no, attempt, text, sha256_text(text), now_iso()),
            )

    def latest_codex_plan(self, step_no: int) -> Optional[dict]:
        row = self.db.execute(
            "SELECT plan_json FROM codex_plans WHERE step_no=? ORDER BY id DESC LIMIT 1",
            (step_no,),
        ).fetchone()
        return json.loads(row["plan_json"]) if row else None

    def save_cline_report(self, step_no: int, attempt: int, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        with self.db:
            self.db.execute(
                """
                INSERT INTO cline_reports(step_no,attempt,report_json,report_hash,created_at)
                VALUES(?,?,?,?,?)
                """,
                (step_no, attempt, text, sha256_text(text), now_iso()),
            )

    def latest_cline_report(self, step_no: int) -> Optional[dict]:
        row = self.db.execute(
            "SELECT report_json FROM cline_reports WHERE step_no=? ORDER BY id DESC LIMIT 1",
            (step_no,),
        ).fetchone()
        return json.loads(row["report_json"]) if row else None

    def save_codex_review(self, step_no: int, attempt: int, payload: dict) -> None:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        verdict = str(payload.get("verdict", ""))
        with self.db:
            self.db.execute(
                """
                INSERT INTO codex_reviews(step_no,attempt,verdict,review_json,review_hash,created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (step_no, attempt, verdict, text, sha256_text(text), now_iso()),
            )

    def latest_codex_review(self, step_no: int) -> Optional[dict]:
        row = self.db.execute(
            "SELECT review_json FROM codex_reviews WHERE step_no=? ORDER BY id DESC LIMIT 1",
            (step_no,),
        ).fetchone()
        return json.loads(row["review_json"]) if row else None

    def approval(self, step_no: int, action: str, reason: str = "", actor: str = "human") -> None:
        with self.db:
            self.db.execute(
                """
                INSERT INTO approvals(step_no,action,reason,actor,created_at)
                VALUES(?,?,?,?,?)
                """,
                (step_no, action, reason, actor, now_iso()),
            )

    def mark_verified_and_advance(self, step_no: int) -> Optional[int]:
        step = self.get_step(step_no)
        if step.state != "REVIEW_APPROVED":
            raise RuntimeError("Only REVIEW_APPROVED may be accepted as VERIFIED.")

        ts = now_iso()
        with self.db:
            self.db.execute(
                """
                UPDATE steps
                SET state='VERIFIED', verified_at=?, finished_at=?, last_update_at=?
                WHERE step_no=?
                """,
                (ts, ts, ts, step_no),
            )
            row = self.db.execute(
                """
                SELECT step_no FROM steps
                WHERE step_no>? AND state='PENDING'
                ORDER BY step_no LIMIT 1
                """,
                (step_no,),
            ).fetchone()
            next_no = int(row["step_no"]) if row else None
            if next_no is not None:
                self.db.execute(
                    "UPDATE steps SET state='READY_FOR_CODEX_PLAN', last_update_at=? WHERE step_no=?",
                    (ts, next_no),
                )
            self.db.execute(
                "INSERT INTO events(step_no,event_type,details,created_at) VALUES(?,?,?,?)",
                (step_no, "STEP_VERIFIED", f"next={next_no}", ts),
            )
        return next_no

    def sync_step_description_from_plan(
        self,
        plan_path: Path,
        step_no: Optional[int] = None,
        *,
        actor: str = "human",
        reason: str = "",
    ) -> dict:
        """Controlled sync of ONE step description from the canonical plan.

        History-safe by design: it updates ONLY ``steps.description`` for the
        target (default: current) step and appends exactly one audit event.
        States, attempts, VERIFIED/PENDING flags, Codex plans and Cline reports
        are never touched, and the plan is never re-imported. The active plan
        snapshot (plan_versions + meta) is refreshed so the DB mirrors the
        canonical document.
        """
        plan_path = Path(plan_path)
        if not plan_path.exists():
            raise FileNotFoundError(f"Canonical BUILD_PLAN not found: {plan_path}")
        plan = read_json(plan_path)
        self._validate_plan(plan)

        target = self.current_step() if step_no is None else self.get_step(step_no)
        if target is None:
            raise RuntimeError("No active step to sync.")

        by_no = {int(raw["step_no"]): raw for raw in plan["steps"]}
        raw = by_no.get(target.step_no)
        if raw is None:
            raise RuntimeError(
                f"Canonical BUILD_PLAN has no step {target.step_no}."
            )
        new_description = str(raw.get("description", "")).strip()
        if not new_description:
            raise ValueError(
                f"Canonical BUILD_PLAN step {target.step_no} has an empty description."
            )

        old_description = target.description
        old_hash = sha256_text(old_description)
        new_hash = sha256_text(new_description)

        plan_text = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2)
        plan_hash = sha256_text(plan_text)
        plan_version = str(plan.get("plan_version", plan.get("version", "")))
        ts = now_iso()
        changed = old_description != new_description

        details = json.dumps(
            {
                "step_no": target.step_no,
                "old_description_hash": old_hash,
                "new_description_hash": new_hash,
                "changed": changed,
                "plan_version": plan_version,
                "plan_hash": plan_hash,
                "plan_path": str(plan_path),
                "actor": actor,
                "reason": reason,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        with self.db:
            self.db.execute(
                "UPDATE steps SET description=?, last_update_at=? WHERE step_no=?",
                (new_description, ts, target.step_no),
            )
            if self.get_meta("plan_hash", "") != plan_hash:
                self.db.execute(
                    """
                    INSERT INTO plan_versions(version, plan_hash, plan_json, created_at)
                    VALUES(?,?,?,?)
                    """,
                    (plan_version, plan_hash, plan_text, ts),
                )
            for key, value in (
                ("plan_version", plan_version),
                ("plan_hash", plan_hash),
            ):
                self.db.execute(
                    """
                    INSERT INTO meta(key,value) VALUES(?,?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (key, value),
                )
            self.db.execute(
                "INSERT INTO events(step_no,event_type,details,created_at) VALUES(?,?,?,?)",
                (target.step_no, "STEP_DESCRIPTION_SYNCED", details, ts),
            )

        return {
            "step_no": target.step_no,
            "changed": changed,
            "old_description_hash": old_hash,
            "new_description_hash": new_hash,
            "plan_version": plan_version,
            "plan_hash": plan_hash,
            "plan_path": str(plan_path),
            "actor": actor,
            "reason": reason,
        }


class ContextBuilder:
    #: Paths offered to Codex as references. Only the NAMES are sent; file
    #: contents are never dumped into the context (token efficiency).
    REFERENCE_FILES = (
        "ARCHITECTURE.md",
        "STRATEGY.md",
        "INTENT_MAP.md",
        "PERFORMANCE.md",
        "CLINE_PLAN_PROMPT.txt",
        "BUILD_PLAN.json",
        "Arhitect_goal/ARCHITECTURE.md",
        "Arhitect_goal/STRATEGY.md",
    )

    #: Bounds so git context stays small even on large/dirty trees.
    MAX_GIT_DIFF_CHARS = 40000
    MAX_GIT_STATUS_CHARS = 8000

    def __init__(self, db: TandemDB):
        self.db = db
        self.root = db.project_root

    def _step_block(self, step: TandemStep) -> dict:
        return {
            "step_no": step.step_no,
            "phase": step.phase,
            "title": step.title,
            "description": step.description,
            "risk": step.risk,
            "requires_human": step.requires_human,
            "attempt": step.attempt,
        }

    def _reference_names(self) -> list[str]:
        return [name for name in self.REFERENCE_FILES if (self.root / name).exists()]

    def _previous_step_block(self, step: TandemStep) -> dict:
        description = step.description or ""
        if len(description) > MAX_PREVIOUS_STEP_DESCRIPTION_CHARS:
            description = (
                description[:MAX_PREVIOUS_STEP_DESCRIPTION_CHARS]
                + "...[TRUNCATED BY WORKER_TANDEM]..."
            )
        return {
            "step_no": step.step_no,
            "phase": step.phase,
            "title": step.title,
            "description": description,
        }

    def project_context(self, step: TandemStep) -> dict:
        """Minimal but COMPLETE PLAN context for the current step.

        Carries the FULL current step (never title-only), the locked controller
        rules of the active plan, only the necessary previous VERIFIED steps,
        and reference file NAMES. Full document bodies, repository listings and
        unrelated history are deliberately NOT included; Codex reads only what
        this step needs.
        """
        if not is_sufficient_step_description(step.description, step.title):
            raise ValueError(
                f"Step {step.step_no} has no usable description (empty or "
                "title-only); refusing to send a non-task to Codex."
            )

        previous = self.db.previous_verified_steps(step.step_no)
        if len(previous) > MAX_PREVIOUS_STEPS:
            previous = previous[-MAX_PREVIOUS_STEPS:]

        return {
            "worker": {"name": APP_NAME, "version": APP_VERSION},
            "project_root": str(self.root),
            "step": self._step_block(step),
            "controller_policy": self.db.plan_policy(),
            "previous_steps": [self._previous_step_block(s) for s in previous],
            "rules": PLAN_CONTEXT_RULES,
            "reference_files": self._reference_names(),
        }

    def review_context(self, step: TandemStep) -> dict:
        """Minimal REVIEW context: step + approved plan + Cline report +
        rules + bounded git diff/status.
        """
        return {
            "worker": {"name": APP_NAME, "version": APP_VERSION},
            "project_root": str(self.root),
            "step": self._step_block(step),
            "rules": REVIEW_CONTEXT_RULES,
            "approved_plan": self.db.latest_codex_plan(step.step_no),
            "cline_report": self.db.latest_cline_report(step.step_no),
            "git_diff": git_capture(
                self.root,
                ["diff", "HEAD", "--no-ext-diff", "--unified=3"],
                max_chars=self.MAX_GIT_DIFF_CHARS,
            ),
            "git_status": git_capture(
                self.root,
                ["status", "--short", "--branch"],
                max_chars=self.MAX_GIT_STATUS_CHARS,
            ),
        }


# ---------------------------------------------------------------------------
# Codex 0.157.1 structured-output helpers
# ---------------------------------------------------------------------------

#: JSONL event types that mean the run is dead regardless of the exit code.
FATAL_CODEX_EVENTS = {"turn.failed", "thread.failed", "task.failed"}

#: stderr markers that indicate Codex's tool/MCP transport failed.
#: DIAGNOSTIC ONLY: a run that still produced a valid structured result is
#: NOT failed just because these appear in stderr.
CODEX_TRANSPORT_MARKERS = (
    "error decoding response body",
    "rmcp",
    "transport channel closed",
    "worker quit with fatal",
    "stream error",
    "connection reset",
)


def parse_jsonl(text: str) -> list[dict]:
    """Parse a Codex `--json` stdout stream into a list of event objects."""
    events: list[dict] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _summarise_event(event: dict) -> str:
    for key in ("error", "message", "detail", "reason"):
        if event.get(key):
            value = event[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            return f"{event.get('type', 'event')}: {value}"[:800]
    return json.dumps(event, ensure_ascii=False)[:800]


def fatal_event(events: list[dict]) -> Optional[str]:
    """Return a description of a definitively fatal event, else None."""
    for event in events:
        etype = str(event.get("type", "")).strip().lower()
        if etype in FATAL_CODEX_EVENTS:
            return _summarise_event(event)
        if etype in {"error", "fatal"}:
            error = event.get("error")
            if event.get("fatal") is True or (
                isinstance(error, dict) and error.get("fatal") is True
            ):
                return _summarise_event(event)
    return None


def error_notes(events: list[dict]) -> list[str]:
    """Non-fatal error notes; used only as diagnostics after a failure."""
    notes = []
    for event in events:
        if str(event.get("type", "")).strip().lower() in {"error", "fatal"}:
            notes.append(_summarise_event(event))
    return notes


def final_agent_message(events: list[dict]) -> Optional[str]:
    """Last agent message text found anywhere in the JSONL stream."""
    last = None
    for event in events:
        etype = str(event.get("type", "")).strip().lower()
        if etype == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and str(item.get("type", "")).lower() in (
                "agent_message",
                "assistant_message",
                "message",
            ):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    last = text
        elif etype in ("agent_message", "assistant_message"):
            text = event.get("text") or event.get("content")
            if isinstance(text, str) and text.strip():
                last = text
    return last


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model text."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if "\n" in stripped:
            first, rest = stripped.split("\n", 1)
            if first.strip() and not first.strip().startswith("{"):
                stripped = rest.strip()
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        return obj

    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[start:index + 1])
                    except (ValueError, TypeError):
                        obj = None
                    if isinstance(obj, dict):
                        return obj
                    break
        start = stripped.find("{", start + 1)
    return None


def validate_against_schema(value: Any, schema: Any, path: str = "$") -> list[str]:
    """Minimal, dependency-free JSON Schema check for Worker_tandem schemas.

    Returns a list of human-readable problems; an empty list means valid.
    Supports: type, enum, object(required/properties/additionalProperties),
    array(items), string, integer, number, boolean.
    """
    problems: list[str] = []
    if not isinstance(schema, dict):
        return problems

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        problems.append(f"{path}: {value!r} is not one of {enum}")

    stype = schema.get("type")
    if stype == "object":
        if not isinstance(value, dict):
            problems.append(f"{path}: expected object, got {type(value).__name__}")
            return problems
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                problems.append(f"{path}: missing required key '{key}'")
        for key, subschema in properties.items():
            if key in value:
                problems.extend(
                    validate_against_schema(value[key], subschema, f"{path}.{key}")
                )
        if schema.get("additionalProperties") is False:
            extra = [key for key in value if key not in properties]
            if extra:
                problems.append(f"{path}: unexpected keys {extra}")
    elif stype == "array":
        if not isinstance(value, list):
            problems.append(f"{path}: expected array, got {type(value).__name__}")
            return problems
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                problems.extend(
                    validate_against_schema(item, items, f"{path}[{index}]")
                )
    elif stype == "string":
        if not isinstance(value, str):
            problems.append(f"{path}: expected string, got {type(value).__name__}")
    elif stype == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected integer, got {type(value).__name__}")
    elif stype == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected number, got {type(value).__name__}")
    elif stype == "boolean":
        if not isinstance(value, bool):
            problems.append(f"{path}: expected boolean, got {type(value).__name__}")
    return problems


def looks_like_transport_failure(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in CODEX_TRANSPORT_MARKERS)


def tail_lines(text: str, count: int) -> str:
    lines = (text or "").splitlines()
    return "\n".join(lines[-count:]) if lines else ""


def publish_json_atomic(temp_path: Path, final_path: Path, payload: dict) -> None:
    """Write `payload` to `temp_path` (fsync), then os.replace onto `final_path`.

    Only a fully validated artifact ever appears under the final filename.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, final_path)


class CodexRunner:
    """
    Runs Codex CLI as a read/review agent.

    We intentionally do NOT pass --full-auto.
    The Worker_tandem contract is that Codex produces structured plan/review
    artifacts; Python remains the controller.

    Codex 0.157.1 invocation contract:
      codex exec --sandbox read-only --json --output-schema <FILE> -o <FILE>
      [--skip-git-repo-check] [-m <MODEL>] <PROMPT>

    `-o` always points at a temporary, non-final filename; only a validated
    artifact is published to the final filename with os.replace().
    """

    #: How many trailing stderr/stdout lines to keep in diagnostics.
    DIAG_TAIL_LINES = 40

    #: Minimal, side-effect-free prompt for the structured-output self-test.
    SELFTEST_PROMPT = (
        "This is a Worker_tandem integration self-test.\n"
        "Do not read, create, or modify any file.\n"
        "Reply with ONLY this JSON object and nothing else:\n"
        '{"ok": true, "message": "worker_tandem codex self-test"}'
    )

    def __init__(
        self,
        project_root: Path,
        executable: str = DEFAULT_CODEX_EXECUTABLE,
        timeout_sec: int = DEFAULT_CODEX_TIMEOUT_SEC,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.project_root = project_root
        self.executable = executable
        self.timeout_sec = timeout_sec
        # None -> use the Codex configuration/default model.
        # Worker_tandem never edits ~/.codex/config.toml automatically.
        self.model = model
        # None -> default reasoning effort (low) for PLAN/REVIEW.
        self.reasoning_effort = reasoning_effort or DEFAULT_CODEX_REASONING
        self.workdir = Path(project_root) / APP_DIR

    def resolved_executable(self) -> Optional[str]:
        if Path(self.executable).is_file():
            return str(Path(self.executable).resolve())
        return shutil.which(self.executable)

    def available(self) -> tuple[bool, str]:
        exe = self.resolved_executable()
        if not exe:
            return False, f"Codex executable not found: {self.executable}"
        try:
            cp = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
            )
            output = (cp.stdout or cp.stderr or "").strip()
            return cp.returncode == 0, output or exe
        except Exception as exc:
            return False, str(exc)

    def _is_git_repo(self) -> bool:
        git = shutil.which("git")
        if not git:
            return False
        try:
            cp = subprocess.run(
                [git, "-C", str(self.project_root), "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                shell=False,
            )
            return cp.returncode == 0 and cp.stdout.strip() == "true"
        except Exception:
            return False

    #: Positional prompt marker that makes `codex exec` read the prompt from
    #: stdin. The prompt MUST be delivered this way: on Windows the Codex CLI is
    #: usually an npm `.CMD` shim that re-parses its arguments through cmd.exe,
    #: and a multi-line argument is truncated at the first newline - only the
    #: first line would ever reach the model.
    STDIN_PROMPT = "-"

    def build_command(
        self,
        schema_path: Path,
        output_path: Path,
        reasoning_effort: Optional[str] = None,
    ) -> list[str]:
        """Build the `codex exec` command for a structured run.

        The prompt is deliberately NOT part of the command line; the caller
        streams it to stdin (see :data:`STDIN_PROMPT`). `-o` always points at a
        temporary, non-final filename. We never pass --full-auto: PLAN/REVIEW
        must run in a read-only sandbox. Reasoning effort is set explicitly via
        -c model_reasoning_effort="...".
        """
        exe = self.resolved_executable()
        if not exe:
            raise RuntimeError(
                f"Codex CLI not found. Expected executable: {self.executable}"
            )
        effort = reasoning_effort or self.reasoning_effort
        if effort not in VALID_CODEX_REASONING:
            effort = DEFAULT_CODEX_REASONING
        cmd = [
            exe,
            "exec",
            "--sandbox",
            DEFAULT_CODEX_SANDBOX,
            "--json",
            "-c",
            f'model_reasoning_effort="{effort}"',
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ]
        if not self._is_git_repo():
            cmd.append("--skip-git-repo-check")
        if self.model:
            cmd += ["-m", str(self.model)]
        cmd.append(self.STDIN_PROMPT)
        return cmd

    def _failure_report(
        self,
        returncode: int,
        events: list[dict],
        stdout: str,
        stderr: str,
        trace_path: Path,
        headline: str,
        extra: str = "",
    ) -> str:
        """Human-readable diagnostics for a failed structured run."""
        lines = [f"Codex structured run failed: {headline}"]
        if extra:
            lines.append(extra)
        lines.append(f"codex return code: {returncode}")
        fatal = fatal_event(events)
        if fatal:
            lines.append(f"fatal event: {fatal}")
        for note in error_notes(events)[:5]:
            lines.append(f"error event: {note}")
        if looks_like_transport_failure(stderr):
            lines.append(
                "detected Codex tool/MCP transport failure (for example "
                "'error decoding response body' / rmcp). This usually means the "
                "configured model or an MCP server in ~/.codex/config.toml is not "
                "usable from a headless 'codex exec' run. Worker_tandem does not "
                "edit Codex config; fix the model/MCP settings or set an explicit "
                "CodexRunner model override."
            )
        lines.append(f"trace file: {trace_path}")
        stderr_tail = tail_lines(stderr, self.DIAG_TAIL_LINES)
        if stderr_tail:
            lines.append("stderr tail:\n" + stderr_tail)
        stdout_tail = tail_lines(stdout, self.DIAG_TAIL_LINES)
        if stdout_tail:
            lines.append("stdout tail:\n" + stdout_tail)
        return "\n".join(lines)

    def self_test(self, timeout_sec: Optional[int] = None) -> tuple[bool, str]:
        """End-to-end structured-output self-test.

        Runs `codex exec` in a read-only sandbox against throw-away schema/output
        files inside a system temp directory. It does NOT touch the current STEP,
        the Worker_tandem FSM, SQLite, project source files, Mini Worker, or any
        project state.
        """
        exe = self.resolved_executable()
        if not exe:
            return False, f"Codex executable not found: {self.executable}"

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ok": {"type": "boolean"},
                "message": {"type": "string"},
            },
            "required": ["ok", "message"],
        }

        with tempfile.TemporaryDirectory(prefix="worker_tandem_codex_selftest_") as tmp:
            tmp_dir = Path(tmp)
            schema_path = tmp_dir / "selftest.schema.json"
            output_path = tmp_dir / "selftest.output.json"
            trace_path = tmp_dir / "selftest.trace.json"
            write_json_atomic(schema_path, schema)

            error = ""
            payload: Optional[dict] = None
            try:
                payload = self.run_structured(
                    self.SELFTEST_PROMPT,
                    schema_path,
                    output_path,
                    trace_path,
                    timeout_sec=timeout_sec or min(self.timeout_sec, 600),
                )
            except Exception as exc:  # diagnostics are the point of the self-test
                error = str(exc)

            trace_text = (
                safe_text(trace_path, 20000)
                if trace_path.exists()
                else "(no trace written)"
            )

        ok = bool(payload) and payload.get("ok") is True
        report = [
            f"status: {'OK' if ok else 'FAILED'}",
            f"executable: {exe}",
            f"model: {self.model or '(Codex config default)'}",
            f"sandbox: {DEFAULT_CODEX_SANDBOX}",
        ]
        if payload is not None:
            report.append("parsed payload: " + json.dumps(payload, ensure_ascii=False))
        if error:
            report.append("error: " + error)
        report.append("--- selftest trace ---")
        report.append(trace_text)
        return ok, "\n".join(report)

    def run_structured(
        self,
        prompt: str,
        schema_path: Path,
        output_path: Path,
        trace_path: Path,
        timeout_sec: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> dict:
        schema_path = Path(schema_path)
        output_path = Path(output_path)
        trace_path = Path(trace_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        if not schema_path.is_file():
            raise RuntimeError(f"Codex output schema not found: {schema_path}")

        # Codex must never write the final artifact directly.
        temp_output = output_path.with_name(output_path.name + ".codex_partial")
        try:
            if temp_output.exists():
                temp_output.unlink()
        except OSError:
            pass

        effort = reasoning_effort or self.reasoning_effort
        cmd = self.build_command(schema_path, temp_output, effort)
        timeout = timeout_sec or self.timeout_sec
        # The prompt travels on stdin, never as an argv element: on Windows the
        # Codex `.CMD` shim re-parses argv through cmd.exe and truncates a
        # multi-line prompt at its first newline.
        prompt_info = {
            "prompt_via": "stdin",
            "prompt_chars": len(prompt),
            "prompt": prompt,
        }

        try:
            cp = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            write_json_atomic(
                trace_path,
                {
                    "command": cmd,
                    **prompt_info,
                    "returncode": None,
                    "timeout_sec": timeout,
                    "reasoning_effort": effort,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "finished_at": now_iso(),
                    "error": "timeout",
                },
            )
            raise RuntimeError(
                f"Codex timed out after {timeout}s. See {trace_path}"
            ) from exc

        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        events = parse_jsonl(stdout)
        schema = read_json(schema_path)

        def record(resolution: str) -> None:
            write_json_atomic(
                trace_path,
                {
                    "command": cmd,
                    **prompt_info,
                    "returncode": cp.returncode,
                    "timeout_sec": timeout,
                    "reasoning_effort": effort,
                    "stdout": stdout,
                    "stderr": stderr,
                    "jsonl_events": events,
                    "resolution": resolution,
                    "finished_at": now_iso(),
                },
            )

        # A. process exit code ------------------------------------------------
        if cp.returncode != 0:
            record("failed:returncode")
            raise RuntimeError(
                self._failure_report(
                    cp.returncode,
                    events,
                    stdout,
                    stderr,
                    trace_path,
                    f"codex exec exited with code {cp.returncode}.",
                )
            )

        # B. definitively fatal JSONL event -----------------------------------
        fatal = fatal_event(events)
        if fatal:
            record("failed:fatal_event")
            raise RuntimeError(
                self._failure_report(
                    0,
                    events,
                    stdout,
                    stderr,
                    trace_path,
                    f"Codex reported a fatal event: {fatal}",
                )
            )

        # C. resolve a valid structured result ---------------------------------
        candidates: list[tuple[str, str]] = []
        if temp_output.is_file():
            candidates.append(
                (
                    "output_file",
                    temp_output.read_text(encoding="utf-8", errors="replace"),
                )
            )
        message = final_agent_message(events)
        if message:
            candidates.append(("jsonl_agent_message", message))
        if not events and stdout.strip():
            candidates.append(("stdout", stdout))

        payload = None
        resolution = "none"
        schema_problem = ""
        for source, text in candidates:
            obj = extract_json_object(text)
            if obj is None:
                continue
            # Step identity is controller-owned. Codex must never choose or
            # echo step_no; ignore/remove any model-provided value BEFORE
            # schema validation so it cannot trip additionalProperties=False.
            obj.pop("step_no", None)
            problems = validate_against_schema(obj, schema)
            if problems:
                if not schema_problem:
                    schema_problem = (
                        f"{source} JSON failed schema validation: "
                        + "; ".join(problems[:5])
                    )
                continue
            payload = obj
            resolution = f"validated:{source}"
            break

        record(resolution)

        # D. no valid result -> diagnose using the real cause -------------------
        if payload is None:
            raise RuntimeError(
                self._failure_report(
                    0,
                    events,
                    stdout,
                    stderr,
                    trace_path,
                    "Codex finished but produced no valid schema-conforming result.",
                    extra=schema_problem,
                )
            )

        # Publish atomically: only a validated artifact reaches the final name.
        publish_json_atomic(temp_output, output_path, payload)
        return payload


def codex_plan_schema() -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["PLAN_READY", "BLOCKED"]},
            "goal": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "changes": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "string"}},
            "architecture_risk": {"type": "string"},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "rollback": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": [
            "decision",
            "goal",
            "files",
            "changes",
            "checks",
            "architecture_risk",
            "open_questions",
            "rollback",
            "summary",
        ],
    }


def codex_review_schema() -> dict:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "APPROVE",
                    "REVISE",
                    "BLOCKED",
                    "ARCHITECTURE_DECISION_REQUIRED",
                ],
            },
            "issues": {"type": "array", "items": {"type": "string"}},
            "required_changes": {"type": "array", "items": {"type": "string"}},
            "checks": {"type": "array", "items": {"type": "string"}},
            "architecture_risk": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": [
            "verdict",
            "issues",
            "required_changes",
            "checks",
            "architecture_risk",
            "summary",
        ],
    }


class TandemController:
    def __init__(self, project_root: Path):
        self.db = TandemDB(project_root)
        self.context_builder = ContextBuilder(self.db)
        self.codex = CodexRunner(project_root, model=DEFAULT_CODEX_MODEL)

        write_json_atomic(self.db.context_dir / "codex_plan.schema.json", codex_plan_schema())
        write_json_atomic(
            self.db.context_dir / "codex_review.schema.json", codex_review_schema()
        )

    def close(self) -> None:
        self.db.close()

    def migrate_from_mini(self) -> dict:
        return self.db.migrate_from_mini()

    def canonical_plan_path(self) -> Path:
        """The human-editable plan document that lives with the project."""
        return self.db.project_root / CANONICAL_PLAN_NAME

    def sync_current_step_description_from_plan(
        self, actor: str = "human", reason: str = ""
    ) -> dict:
        """Sync the CURRENT step description from the canonical BUILD_PLAN.json.

        History-safe: no plan re-import, no attempt/state reset; the operation
        only refreshes the step description and records an audit event.
        """
        return self.db.sync_step_description_from_plan(
            self.canonical_plan_path(),
            None,
            actor=actor,
            reason=reason,
        )

    def _codex_reasoning_effort(self, step: TandemStep) -> str:
        """Reasoning-effort policy for PLAN and REVIEW.

        Default is "low". Escalate to "medium" for repeated/uncertain attempts
        (for example a step that has already been dispatched/revised) and to
        "high" for architecture-critical steps. This only changes how much
        Codex thinks; it never relaxes a correctness gate.
        """
        if str(step.risk).upper() == "HIGH" or step.requires_human:
            return CODEX_REASONING_HIGH
        if step.attempt > 1:
            return CODEX_REASONING_MEDIUM
        return CODEX_REASONING_LOW

    def run_codex_plan(self) -> dict:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "READY_FOR_CODEX_PLAN":
            raise RuntimeError(f"Current state must be READY_FOR_CODEX_PLAN, got {step.state}")

        # Guard BEFORE any state mutation: a title-only (or empty) description
        # is not a task and must never be sent to Codex.
        if not is_sufficient_step_description(step.description, step.title):
            raise RuntimeError(
                f"Step {step.step_no} description is empty or title-only; "
                f"sync it from {CANONICAL_PLAN_NAME} before running Codex PLAN."
            )

        attempt = self.db.increment_attempt(step.step_no)
        self.db.transition(step.step_no, "CODEX_PLAN_RUNNING", f"attempt={attempt}")
        step = self.db.get_step(step.step_no)
        effort = self._codex_reasoning_effort(step)

        context = self.context_builder.project_context(step)
        ctx_path = self.db.context_dir / f"step_{step.step_no:03d}_codex_plan_context.json"
        write_json_atomic(ctx_path, context)

        request_path = self.db.to_codex / f"step_{step.step_no:03d}_plan_request.json"
        request = {
            "type": "CODEX_PLAN_REQUEST",
            "step_no": step.step_no,
            "attempt": attempt,
            "reasoning_effort": effort,
            "context_file": str(ctx_path),
            "rules": PLAN_CONTEXT_RULES,
        }
        write_json_atomic(request_path, request)

        prompt = f"""
You are the Planner/Architect in Worker_tandem.

PLAN ONLY. DO NOT MODIFY PROJECT FILES.

Read the context JSON (minimized: this step, the rules, reference file names):
{ctx_path}

Current step:
STEP {step.step_no}: {step.title}

Description:
{step.description}

Your job:
- inspect the repository as LITTLE as possible; read only what THIS step needs;
- produce the safest concrete implementation plan for THIS STEP ONLY;
- preserve existing architecture and backward compatibility;
- identify files, checks, risks, open questions, and rollback;
- do not implement;
- if a required architecture decision is unresolved, set decision=BLOCKED.

{CODEX_EFFICIENCY_CONTRACT}

Return JSON conforming exactly to the provided output schema.
""".strip()

        output_path = self.db.from_codex / f"step_{step.step_no:03d}_plan.json"
        trace_path = self.db.logs / f"step_{step.step_no:03d}_codex_plan_trace.json"
        try:
            result = self.codex.run_structured(
                prompt,
                self.db.context_dir / "codex_plan.schema.json",
                output_path,
                trace_path,
                reasoning_effort=effort,
            )
        except Exception:
            self.db.transition(step.step_no, "FAILED", "Codex PLAN failed.")
            raise

        # Python/FSM owns step identity. The model's advisory payload must not
        # carry step_no; the controller injects the authoritative value before
        # the canonical artifact is published and stored.
        result["step_no"] = step.step_no
        write_json_atomic(output_path, result)
        self.db.save_codex_plan(step.step_no, attempt, result)

        decision = str(result.get("decision", "BLOCKED"))
        if decision == "PLAN_READY":
            self.db.transition(step.step_no, "PLAN_READY", "Codex plan ready.")
        else:
            self.db.transition(step.step_no, "BLOCKED", result.get("summary", "Codex blocked."))
        return result

    def approve_plan_and_dispatch_cline(self, reason: str = "") -> Path:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "PLAN_READY":
            raise RuntimeError(f"Expected PLAN_READY, got {step.state}")

        plan = self.db.latest_codex_plan(step.step_no)
        if not plan:
            raise RuntimeError("No Codex plan found.")

        self.db.transition(step.step_no, "WAITING_HUMAN_APPROVAL", "Human approval gate.")
        self.db.approval(step.step_no, "APPROVE_CODEX_PLAN", reason or "Approved in GUI")
        self.db.transition(step.step_no, "CLINE_DISPATCHED", "Cline task dispatched.")

        step = self.db.get_step(step.step_no)
        task = {
            "protocol": "worker-tandem/v0.1",
            "type": "CLINE_ACT",
            "step_no": step.step_no,
            "attempt": step.attempt,
            "phase": step.phase,
            "title": step.title,
            "description": step.description,
            "risk": step.risk,
            "requires_human": step.requires_human,
            "approved_codex_plan": plan,
            "instructions": [
                "Implement ONLY this step.",
                "Codex plan is approved guidance; do not broaden scope.",
                "Preserve architecture invariants.",
                "Run the checks/tests required by the approved plan.",
                "Do not modify Worker_tandem SQLite.",
                "Write the final report to from_cline using the exact schema below.",
                "STOP after the report.",
            ],
            "report_file": str(
                self.db.from_cline / f"step_{step.step_no:03d}_report.json"
            ),
            "report_schema": {
                "step_no": "integer",
                "status": "DONE|REVISE|BLOCKED|FAILED",
                "files_created": ["string"],
                "files_changed": ["string"],
                "files_deleted": ["string"],
                "tests": {
                    "passed": "integer",
                    "failed": "integer",
                    "command": "string",
                },
                "dependencies_added": ["string"],
                "architecture_questions": ["string"],
                "issues": ["string"],
                "summary": "string",
            },
        }
        task_path = self.db.to_cline / f"step_{step.step_no:03d}_task.json"
        write_json_atomic(task_path, task)
        self.db.event(step.step_no, "CLINE_TASK_WRITTEN", str(task_path))
        return task_path

    def process_cline_report(self) -> dict:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "CLINE_DISPATCHED":
            raise RuntimeError(f"Expected CLINE_DISPATCHED, got {step.state}")

        path = self.db.from_cline / f"step_{step.step_no:03d}_report.json"
        if not path.exists():
            raise FileNotFoundError(f"Cline report not found: {path}")

        report = read_json(path)
        if not isinstance(report, dict):
            raise RuntimeError("Cline report must be a JSON object.")
        if int(report.get("step_no", -1)) != step.step_no:
            raise RuntimeError("Cline report step_no mismatch.")
        status = str(report.get("status", "")).upper()
        if status not in {"DONE", "REVISE", "BLOCKED", "FAILED"}:
            raise RuntimeError(f"Invalid Cline report status: {status}")

        self.db.save_cline_report(step.step_no, step.attempt, report)

        if status == "DONE":
            self.db.transition(step.step_no, "CLINE_REPORT_RECEIVED", "Cline report accepted.")
        elif status == "REVISE":
            # Still send through Codex review; Codex decides the repair scope.
            self.db.transition(
                step.step_no,
                "CLINE_REPORT_RECEIVED",
                "Cline requested revision; Codex review required.",
            )
        elif status == "BLOCKED":
            self.db.transition(step.step_no, "BLOCKED", report.get("summary", "Cline blocked."))
        else:
            self.db.transition(step.step_no, "FAILED", report.get("summary", "Cline failed."))

        return report

    def run_codex_review(self) -> dict:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "CLINE_REPORT_RECEIVED":
            raise RuntimeError(f"Expected CLINE_REPORT_RECEIVED, got {step.state}")

        self.db.transition(step.step_no, "CODEX_REVIEW_RUNNING", "Codex review started.")
        step = self.db.get_step(step.step_no)
        effort = self._codex_reasoning_effort(step)

        context = self.context_builder.review_context(step)
        ctx_path = self.db.context_dir / f"step_{step.step_no:03d}_codex_review_context.json"
        write_json_atomic(ctx_path, context)

        request_path = self.db.to_codex / f"step_{step.step_no:03d}_review_request.json"
        write_json_atomic(
            request_path,
            {
                "type": "CODEX_REVIEW_REQUEST",
                "step_no": step.step_no,
                "attempt": step.attempt,
                "reasoning_effort": effort,
                "context_file": str(ctx_path),
                "rules": REVIEW_CONTEXT_RULES,
            },
        )

        prompt = f"""
You are the independent Reviewer/Supervisor in Worker_tandem.

REVIEW ONLY. DO NOT MODIFY PROJECT FILES.

Read the review context (minimized: step, approved plan, Cline report, git diff/status, rules):
{ctx_path}

Review STEP {step.step_no}: {step.title}

You must compare:
- the approved Codex plan;
- the actual Cline report;
- repository git diff/status;
- claimed tests/checks;
- architecture rules and project constraints.

Do not trust the implementation report by itself.
Do not change code.
Do not advance state.

Verdict rules:
APPROVE = implementation matches scope and no material issue remains.
REVISE = concrete repair can be performed by Cline within the same step.
BLOCKED = external dependency or non-architecture blocker prevents progress.
ARCHITECTURE_DECISION_REQUIRED = human architecture decision is required.

{CODEX_EFFICIENCY_CONTRACT}

Return only structured JSON.
""".strip()

        output_path = self.db.from_codex / f"step_{step.step_no:03d}_review.json"
        trace_path = self.db.logs / f"step_{step.step_no:03d}_codex_review_trace.json"

        try:
            review = self.codex.run_structured(
                prompt,
                self.db.context_dir / "codex_review.schema.json",
                output_path,
                trace_path,
                reasoning_effort=effort,
            )
        except Exception:
            self.db.transition(step.step_no, "FAILED", "Codex REVIEW failed.")
            raise

        # Python/FSM owns step identity (same rule as PLAN).
        review["step_no"] = step.step_no
        write_json_atomic(output_path, review)
        self.db.save_codex_review(step.step_no, step.attempt, review)
        verdict = str(review.get("verdict", "")).upper()

        if verdict == "APPROVE":
            self.db.transition(step.step_no, "REVIEW_APPROVED", review.get("summary", "Approved."))
        elif verdict == "REVISE":
            self.db.transition(step.step_no, "REVISE", review.get("summary", "Revision required."))
        elif verdict in {"BLOCKED", "ARCHITECTURE_DECISION_REQUIRED"}:
            self.db.transition(step.step_no, "BLOCKED", review.get("summary", verdict))
        else:
            self.db.transition(step.step_no, "FAILED", f"Unknown Codex verdict: {verdict}")
            raise RuntimeError(f"Unknown Codex verdict: {verdict}")

        return review

    def dispatch_repair(self) -> Path:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "REVISE":
            raise RuntimeError(f"Expected REVISE, got {step.state}")

        review = self.db.latest_codex_review(step.step_no)
        if not review:
            raise RuntimeError("No Codex review found.")
        if str(review.get("verdict", "")).upper() != "REVISE":
            raise RuntimeError("Latest Codex review is not REVISE.")

        attempt = self.db.increment_attempt(step.step_no)
        self.db.transition(step.step_no, "CLINE_DISPATCHED", f"repair attempt={attempt}")
        step = self.db.get_step(step.step_no)

        task = {
            "protocol": "worker-tandem/v0.1",
            "type": "CLINE_REPAIR",
            "step_no": step.step_no,
            "attempt": attempt,
            "title": step.title,
            "required_changes": review.get("required_changes", []),
            "issues": review.get("issues", []),
            "instructions": [
                "Repair ONLY the issues listed by Codex.",
                "Do not broaden scope.",
                "Run relevant tests/checks.",
                "Do not modify Worker_tandem SQLite.",
                "Overwrite the step report file with the new attempt report.",
                "STOP after the report.",
            ],
            "report_file": str(
                self.db.from_cline / f"step_{step.step_no:03d}_report.json"
            ),
            "report_schema": {
                "step_no": "integer",
                "status": "DONE|REVISE|BLOCKED|FAILED",
                "files_created": ["string"],
                "files_changed": ["string"],
                "files_deleted": ["string"],
                "tests": {
                    "passed": "integer",
                    "failed": "integer",
                    "command": "string",
                },
                "dependencies_added": ["string"],
                "architecture_questions": ["string"],
                "issues": ["string"],
                "summary": "string",
            },
        }
        task_path = self.db.to_cline / f"step_{step.step_no:03d}_repair_{attempt:02d}.json"
        write_json_atomic(task_path, task)
        self.db.event(step.step_no, "CLINE_REPAIR_WRITTEN", str(task_path))
        return task_path

    def accept_review(self, reason: str = "") -> Optional[int]:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "REVIEW_APPROVED":
            raise RuntimeError(f"Expected REVIEW_APPROVED, got {step.state}")
        self.db.approval(step.step_no, "ACCEPT_CODEX_REVIEW", reason or "Accepted in GUI")
        return self.db.mark_verified_and_advance(step.step_no)


class TandemGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1120x760")
        self.root.minsize(960, 680)

        self.controller: Optional[TandemController] = None
        self.busy = False

        self.project_var = tk.StringVar(value="No project open")
        self.plan_var = tk.StringVar(value="Plan: —")
        self.progress_var = tk.StringVar(value="Progress: —")
        self.current_var = tk.StringVar(value="Current: —")
        self.state_var = tk.StringVar(value="State: —")
        self.codex_var = tk.StringVar(value="Codex: not checked")
        self.footer_var = tk.StringVar(value="Worker_tandem is separate from Mini Worker.")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.project_var).pack(side="left", padx=18)

        ttk.Button(top, text="Open Project", command=self.open_project).pack(side="right", padx=4)
        ttk.Button(top, text="Backup DB", command=self.backup_db).pack(side="right", padx=4)
        ttk.Button(top, text="Check Codex", command=self.check_codex).pack(side="right", padx=4)

        info = ttk.LabelFrame(self.root, text="Status", padding=10)
        info.pack(fill="x", padx=10, pady=(0, 8))

        for var in (
            self.plan_var,
            self.progress_var,
            self.current_var,
            self.state_var,
            self.codex_var,
        ):
            ttk.Label(info, textvariable=var).pack(anchor="w", pady=1)

        actions = ttk.LabelFrame(self.root, text="Workflow", padding=10)
        actions.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_migrate = ttk.Button(
            actions, text="Import from Mini Worker", command=self.migrate_from_mini
        )
        self.btn_migrate.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_plan = ttk.Button(
            actions, text="1. Run Codex PLAN", command=self.run_codex_plan
        )
        self.btn_plan.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        self.btn_dispatch = ttk.Button(
            actions,
            text="2. Approve PLAN + Dispatch Cline",
            command=self.approve_plan_dispatch,
        )
        self.btn_dispatch.grid(row=0, column=2, padx=4, pady=4, sticky="ew")

        self.btn_report = ttk.Button(
            actions, text="3. Process Cline Report", command=self.process_cline_report
        )
        self.btn_report.grid(row=1, column=0, padx=4, pady=4, sticky="ew")

        self.btn_review = ttk.Button(
            actions, text="4. Run Codex REVIEW", command=self.run_codex_review
        )
        self.btn_review.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        self.btn_accept = ttk.Button(
            actions, text="5. Accept Review → Next", command=self.accept_review
        )
        self.btn_accept.grid(row=1, column=2, padx=4, pady=4, sticky="ew")

        self.btn_repair = ttk.Button(
            actions, text="REVISE → Dispatch Repair", command=self.dispatch_repair
        )
        self.btn_repair.grid(row=2, column=1, padx=4, pady=4, sticky="ew")

        self.btn_sync = ttk.Button(
            actions,
            text=f"Sync STEP description from {CANONICAL_PLAN_NAME}",
            command=self.sync_from_canonical_plan,
        )
        self.btn_sync.grid(row=2, column=0, padx=4, pady=4, sticky="ew")

        for i in range(3):
            actions.columnconfigure(i, weight=1)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        ttk.Label(left, text="Steps").pack(anchor="w")
        self.step_tree = ttk.Treeview(
            left,
            columns=("step", "state", "risk", "title"),
            show="headings",
            selectmode="browse",
        )
        self.step_tree.heading("step", text="#")
        self.step_tree.heading("state", text="State")
        self.step_tree.heading("risk", text="Risk")
        self.step_tree.heading("title", text="Title")
        self.step_tree.column("step", width=45, anchor="center")
        self.step_tree.column("state", width=150)
        self.step_tree.column("risk", width=60, anchor="center")
        self.step_tree.column("title", width=280)
        self.step_tree.pack(fill="both", expand=True)

        ttk.Label(right, text="Current plan / report / review").pack(anchor="w")
        self.details = tk.Text(right, wrap="word", font=("Consolas", 10))
        self.details.pack(fill="both", expand=True)
        self.details.configure(state="disabled")

        ttk.Label(
            self.root,
            textvariable=self.footer_var,
            relief="sunken",
            anchor="w",
            padding=5,
        ).pack(fill="x")

        self._set_buttons()

    def _set_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def open_project(self) -> None:
        folder = filedialog.askdirectory(title="Select Architecture Assistant project")
        if not folder:
            return
        if self.controller:
            self.controller.close()
        try:
            self.controller = TandemController(Path(folder))
            self.project_var.set(str(Path(folder).resolve()))
            self.footer_var.set("Project opened. Mini Worker remains untouched.")
            self.refresh()
        except Exception as exc:
            self.controller = None
            messagebox.showerror(APP_NAME, str(exc))

    def migrate_from_mini(self) -> None:
        if not self.controller:
            return
        if not messagebox.askyesno(
            APP_NAME,
            "Mini Worker should be CLOSED before migration.\n\n"
            "Worker_tandem will open .mini_build/build_state.db READ-ONLY "
            "and copy plan/progress into its own .worker_tandem database.\n\nContinue?",
        ):
            return
        try:
            result = self.controller.migrate_from_mini()
            messagebox.showinfo(
                APP_NAME,
                "Migration complete.\n\n"
                f"Plan: {result['plan_version']}\n"
                f"Steps: {result['steps']}\n"
                f"Current step: {result['current_step']}",
            )
            self.footer_var.set("Mini Worker state imported read-only.")
            self.refresh()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def sync_from_canonical_plan(self) -> None:
        if not self.controller:
            return
        path = self.controller.canonical_plan_path()
        if not messagebox.askyesno(
            APP_NAME,
            "Update the current step description from the canonical plan?\n\n"
            f"Source: {path}\n\n"
            "Attempts, events, VERIFIED/PENDING steps and Codex/Cline history "
            "are preserved. The plan is NOT re-imported.\n\nContinue?",
        ):
            return
        try:
            result = self.controller.sync_current_step_description_from_plan(
                reason="Manual sync from canonical BUILD_PLAN.json."
            )
            messagebox.showinfo(
                APP_NAME,
                "STEP description synced.\n\n"
                f"Step: {result['step_no']}\n"
                f"Changed: {result['changed']}\n"
                f"Plan: v{result['plan_version']} ({result['plan_hash'][:12]})\n"
                f"New description hash: {result['new_description_hash'][:12]}",
            )
            self.footer_var.set("STEP description synced from canonical plan.")
            self.refresh()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def check_codex(self) -> None:
        if not self.controller:
            return

        def job() -> dict:
            cli_ok, cli_info = self.controller.codex.available()
            test_ok, test_info = self.controller.codex.self_test()
            return {
                "cli_ok": cli_ok,
                "cli_info": cli_info,
                "selftest_ok": test_ok,
                "selftest_info": test_info,
            }

        def on_success(result: dict) -> None:
            cli = "OK" if result["cli_ok"] else "NOT READY"
            selftest = "OK" if result["selftest_ok"] else "FAILED"
            self.codex_var.set(
                f"Codex: CLI {cli} ({result['cli_info']}) | structured-output {selftest}"
            )
            self._set_details(
                "=== CODEX CLI ===\n"
                f"status: {cli}\n{result['cli_info']}\n\n"
                "=== CODEX STRUCTURED-OUTPUT SELF-TEST ===\n"
                f"{result['selftest_info']}"
            )
            self.footer_var.set(
                f"Codex check complete: CLI {cli}, structured-output self-test {selftest}."
            )

        self._run_background(
            "Checking Codex CLI and structured output...", job, on_success
        )

    def _run_background(self, label: str, fn, on_success) -> None:
        if self.busy:
            return
        self.busy = True
        self.footer_var.set(label)
        self._set_buttons()

        def worker():
            try:
                result = fn()
            except Exception as exc:
                # Exception variables are cleared when leaving an except block.
                # Pass the object as an argument to Tk's callback instead of
                # closing over `exc` in a lambda that runs later.
                self.root.after(0, self._background_error, exc)
                return
            self.root.after(0, self._background_success, result, on_success)

        threading.Thread(target=worker, daemon=True).start()

    def _background_error(self, exc: Exception) -> None:
        self.busy = False
        messagebox.showerror(APP_NAME, str(exc))
        self.footer_var.set("Operation failed.")
        self.refresh()

    def _background_success(self, result, on_success) -> None:
        self.busy = False
        try:
            on_success(result)
        finally:
            self.refresh()

    def run_codex_plan(self) -> None:
        if not self.controller:
            return
        self._run_background(
            "Codex is preparing PLAN...",
            self.controller.run_codex_plan,
            lambda r: self._set_details(json.dumps(r, ensure_ascii=False, indent=2)),
        )

    def approve_plan_dispatch(self) -> None:
        if not self.controller:
            return
        step = self.controller.db.current_step()
        if not step:
            return
        plan = self.controller.db.latest_codex_plan(step.step_no)
        if not plan:
            messagebox.showerror(APP_NAME, "No Codex plan found.")
            return

        preview = json.dumps(plan, ensure_ascii=False, indent=2)
        self._set_details(preview)
        if not messagebox.askyesno(
            APP_NAME,
            f"Approve Codex PLAN for STEP {step.step_no} and dispatch Cline?",
        ):
            return
        try:
            path = self.controller.approve_plan_and_dispatch_cline()
            messagebox.showinfo(
                APP_NAME,
                f"Cline task created:\n{path}\n\n"
                "Give this task file to Cline. Worker_tandem will wait for the report.",
            )
            self.footer_var.set("Cline task dispatched.")
            self.refresh()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def process_cline_report(self) -> None:
        if not self.controller:
            return
        try:
            report = self.controller.process_cline_report()
            self._set_details(json.dumps(report, ensure_ascii=False, indent=2))
            self.footer_var.set("Cline report processed.")
            self.refresh()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def run_codex_review(self) -> None:
        if not self.controller:
            return
        self._run_background(
            "Codex is reviewing Cline implementation...",
            self.controller.run_codex_review,
            lambda r: self._set_details(json.dumps(r, ensure_ascii=False, indent=2)),
        )

    def dispatch_repair(self) -> None:
        if not self.controller:
            return
        try:
            path = self.controller.dispatch_repair()
            messagebox.showinfo(
                APP_NAME,
                f"Repair task created:\n{path}\n\n"
                "Give it to Cline, then process the new report.",
            )
            self.footer_var.set("Cline repair task dispatched.")
            self.refresh()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def accept_review(self) -> None:
        if not self.controller:
            return
        step = self.controller.db.current_step()
        if not step:
            return
        review = self.controller.db.latest_codex_review(step.step_no)
        if review:
            self._set_details(json.dumps(review, ensure_ascii=False, indent=2))
        if not messagebox.askyesno(
            APP_NAME,
            f"Accept Codex review and mark STEP {step.step_no} VERIFIED?",
        ):
            return
        try:
            next_no = self.controller.accept_review()
            self.footer_var.set(
                f"STEP {step.step_no} VERIFIED. "
                + (f"Next: STEP {next_no}" if next_no else "Plan complete.")
            )
            self.refresh()
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def backup_db(self) -> None:
        if not self.controller:
            return
        default_name = f"worker_tandem_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.db"
        filename = filedialog.asksaveasfilename(
            title="Save Worker_tandem DB backup",
            initialdir=str(self.controller.db.backups),
            initialfile=default_name,
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            path = self.controller.db.backup(Path(filename))
            messagebox.showinfo(APP_NAME, f"Backup saved:\n{path}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def refresh(self) -> None:
        if not self.controller:
            self._set_buttons()
            return

        steps = self.controller.db.list_steps()
        verified = sum(1 for s in steps if s.state == "VERIFIED")
        self.plan_var.set(
            f"Plan: v{self.controller.db.get_meta('plan_version', '—')} | "
            f"{len(steps)} steps"
        )
        self.progress_var.set(f"Progress: {verified}/{len(steps)} VERIFIED")

        current = self.controller.db.current_step()
        if current:
            self.current_var.set(
                f"Current: STEP {current.step_no} — {current.title} | "
                f"attempt {current.attempt}/{current.max_attempts}"
            )
            self.state_var.set(
                f"State: {current.state} | risk {current.risk} | "
                f"{'HUMAN' if current.requires_human else 'policy'}"
            )

            detail_parts = []
            plan = self.controller.db.latest_codex_plan(current.step_no)
            report = self.controller.db.latest_cline_report(current.step_no)
            review = self.controller.db.latest_codex_review(current.step_no)
            if plan:
                detail_parts.append("=== CODEX PLAN ===\n" + json.dumps(plan, ensure_ascii=False, indent=2))
            if report:
                detail_parts.append("=== CLINE REPORT ===\n" + json.dumps(report, ensure_ascii=False, indent=2))
            if review:
                detail_parts.append("=== CODEX REVIEW ===\n" + json.dumps(review, ensure_ascii=False, indent=2))
            if detail_parts:
                self._set_details("\n\n".join(detail_parts))
        else:
            self.current_var.set("Current: plan complete")
            self.state_var.set("State: —")

        for item in self.step_tree.get_children():
            self.step_tree.delete(item)
        for s in steps:
            self.step_tree.insert(
                "",
                "end",
                values=(s.step_no, s.state, s.risk, s.title),
            )

        self._set_buttons()

    def _set_buttons(self) -> None:
        state = None
        if self.controller:
            step = self.controller.db.current_step()
            state = step.state if step else None

        disabled = self.busy or not self.controller

        def set_state(btn, enabled: bool) -> None:
            btn.configure(state=("normal" if enabled and not disabled else "disabled"))

        # Migration is intentionally manual; allow if project is open and not busy.
        self.btn_migrate.configure(
            state=("normal" if self.controller and not self.busy else "disabled")
        )
        set_state(self.btn_plan, state == "READY_FOR_CODEX_PLAN")
        set_state(self.btn_dispatch, state == "PLAN_READY")
        set_state(self.btn_report, state == "CLINE_DISPATCHED")
        set_state(self.btn_review, state == "CLINE_REPORT_RECEIVED")
        set_state(self.btn_accept, state == "REVIEW_APPROVED")
        set_state(self.btn_repair, state == "REVISE")

        # Canonical-plan sync is a manual maintenance action: allowed whenever a
        # project is open and no background job is running.
        self.btn_sync.configure(
            state=("normal" if self.controller and not self.busy else "disabled")
        )

    def on_close(self) -> None:
        try:
            if self.controller:
                self.controller.close()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    app = TandemGUI()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
