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
import re
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
from typing import Any, Mapping, Optional

from supervisor import (
    BUDGET_CHEAP,
    BUDGET_PREMIUM,
    BUDGET_STANDARD,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEEPSEEK_API_KEY_ENV,
    HEALTH_KEY_MISSING,
    ID_ANTHROPIC,
    ID_DEEPSEEK,
    ID_OPENAI,
    IDENTITY_STRIP_STEP_NO,
    OPENAI_API_KEY_ENV,
    PLAN_OPERATION,
    POLICY_MODE_AUTO,
    POLICY_MODE_MANUAL,
    PROVIDER_CODEX_LEGACY,
    REVIEW_OPERATION,
    VALIDATION_RUNNER,
    DeepSeekSupervisorAdapter,
    HttpTransport,
    LegacyCodexSupervisorAdapter,
    OpenAISupervisorAdapter,
    ProviderConfig,
    SupervisorPlanRequest,
    SupervisorPolicy,
    SupervisorReviewRequest,
    SupervisorRouter,
    SupervisorRunError,
    default_auto_policy,
    default_deepseek_config,
    default_openai_config,
    sanitize_diagnostic,
)
from supervisor.errors import CATEGORY_QUOTA_WITH_RESET, CATEGORY_RATE_LIMIT


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

#: Cline report readiness watcher (GUI, Tk-safe polling).
#: The watcher only ever inspects the exact expected report path for the
#: current STEP; it never scans the filesystem and adds no dependency.
CLINE_REPORT_POLL_MS = 2500

#: Hard cap so a stray/large file can never make the poll heavy.
CLINE_REPORT_MAX_BYTES = 1_048_576

#: Terminal statuses a Cline report may carry.
CLINE_REPORT_VALID_STATUSES = ("DONE", "REVISE", "BLOCKED", "FAILED")

#: Footer strings owned by the Cline-report watcher. The watcher may only
#: overwrite the footer with one of these (or when the footer is empty), so it
#: can never clobber a Codex PLAN/REVIEW or other foreground progress message.
WATCHER_FOOTER_WAITING = "Waiting for Cline report..."
WATCHER_FOOTER_READY = 'Cline report ready — click "Process Cline Report".'
WATCHER_FOOTER_INVALID = "Cline report invalid — waiting for replacement..."
WATCHER_FOOTER_MESSAGES = {
    WATCHER_FOOTER_WAITING,
    WATCHER_FOOTER_READY,
    WATCHER_FOOTER_INVALID,
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

#: A Codex REVIEW that could not complete for an *external/provider* reason
#: (usage limit, rate limit, timeout, transport, network, provider outage).
#: It is deliberately NOT ``FAILED``: the project implementation is untouched,
#: the accepted Cline report is preserved, and a human may retry the review
#: without consuming an attempt.
CODEX_REVIEW_RETRYABLE = "CODEX_REVIEW_RETRYABLE"

#: Audit event types written through the existing ``events`` table.
CODEX_REVIEW_RETRYABLE_EVENT = "CODEX_REVIEW_RETRYABLE"
CODEX_REVIEW_RETRY_EVENT = "CODEX_REVIEW_RETRY"
RECOVERY_REVIEW_RETRY_EVENT = "RECOVERY_REVIEW_RETRY"

VALID_STATES = {
    "PENDING",
    "READY_FOR_CODEX_PLAN",
    "CODEX_PLAN_RUNNING",
    "PLAN_READY",
    "WAITING_HUMAN_APPROVAL",
    "CLINE_DISPATCHED",
    "CLINE_REPORT_RECEIVED",
    "CODEX_REVIEW_RUNNING",
    CODEX_REVIEW_RETRYABLE,
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
    "CODEX_REVIEW_RUNNING": {
        "REVIEW_APPROVED",
        "REVISE",
        "BLOCKED",
        CODEX_REVIEW_RETRYABLE,
        "FAILED",
    },
    # Reached ONLY from CODEX_REVIEW_RUNNING and only for a classified
    # retryable external/provider failure. The single forward edge is an
    # explicit human Retry Codex REVIEW; BLOCKED/ABORTED are safe human exits.
    CODEX_REVIEW_RETRYABLE: {"CODEX_REVIEW_RUNNING", "BLOCKED", "ABORTED"},
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


# =========================================================================== #
# Git module (project utility; NEVER touches the FSM or the tandem state DB)
# =========================================================================== #

#: V1 supports exactly one remote name. Never pushed to silently.
GIT_REMOTE_NAME = "origin"

#: Read-only git commands (status/log/rev-parse...) use the short timeout.
GIT_READ_TIMEOUT_SEC = 60

#: Mutating git commands (add/commit/push) get the longer timeout.
GIT_WRITE_TIMEOUT_SEC = 120

#: V1 commit messages are single-line and bounded.
GIT_COMMIT_MESSAGE_MAX = 120

#: Action event types recorded through the existing ``events`` audit table.
GIT_EVENT_STATUS = "GIT_STATUS"
GIT_EVENT_COMMIT = "GIT_COMMIT"
GIT_EVENT_PUSH = "GIT_PUSH"


class GitError(RuntimeError):
    """A safe, classified Git failure.

    ``kind`` is one of the ERROR taxonomy values and is used by the GUI to
    choose wording; ``stderr`` carries the sanitized git stderr (if any).
    """

    def __init__(self, kind: str, message: str, *, stderr: str = ""):
        super().__init__(message)
        self.kind = kind
        self.stderr = stderr


@dataclass(frozen=True)
class GitFileEntry:
    """One ``git status --porcelain=v1 -z`` record.

    ``path`` is the repo-relative *current* (new) path, kept byte-exact after
    UTF-8 decoding (spaces, tabs, newlines and Unicode preserved). For
    renames/copies the porcelain stream emits the new path first and the
    original path in the following NUL field.
    """

    code: str
    path: str
    orig_path: Optional[str] = None
    staged: bool = False
    unstaged: bool = False
    untracked: bool = False
    change_type: str = "other"


def sanitize_remote_url(url: str) -> str:
    """Redact embedded credentials/tokens from a remote URL before display."""
    url = (url or "").strip()
    if not url:
        return ""
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if "@" in rest:
            rest = "***@" + rest.split("@", 1)[1]
        return f"{scheme}://{rest}"
    # scp-like ``user@host:path``.
    if "@" in url and ":" in url:
        user_host, _, tail = url.partition(":")
        if "@" in user_host:
            user_host = "***@" + user_host.split("@", 1)[1]
        return f"{user_host}:{tail}"
    return url


def validate_commit_message(message: str) -> tuple[bool, str]:
    """Validate a V1 commit message; returns ``(ok, cleaned_or_error)``."""
    text = (message or "").strip()
    if not text:
        return False, "Commit message is required."
    if "\n" in text or "\r" in text:
        return False, "Commit message must be a single line."
    if len(text) > GIT_COMMIT_MESSAGE_MAX:
        return (
            False,
            f"Commit message must be {GIT_COMMIT_MESSAGE_MAX} characters or fewer.",
        )
    return True, text


def is_runtime_or_sensitive(path: str) -> bool:
    """Whether a repo-relative path is a runtime/secret artifact.

    Such files are visibly marked in the GUI and are NEVER auto-selected.
    """
    normalized = (path or "").replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = normalized.split("/")
    name = parts[-1]
    lowered = [p.lower() for p in parts]
    if any(p == ".worker_tandem" for p in lowered):
        return True
    if any(p in {"logs", "secrets", "credentials"} for p in lowered):
        return True
    if name == ".env" or name.startswith(".env."):
        return True
    if name.endswith((".db", ".db-wal", ".db-shm", ".tmp")):
        return True
    return False


def _classify_git_change(code: str) -> str:
    x = code[0] if len(code) > 0 else " "
    y = code[1] if len(code) > 1 else " "
    if code == "??":
        return "untracked"
    if x == "R" or y == "R":
        return "renamed"
    if x == "C" or y == "C":
        return "copied"
    if x == "A" or y == "A":
        return "added"
    if x == "D" or y == "D":
        return "deleted"
    if x == "M" or y == "M":
        return "modified"
    if x == "T" or y == "T":
        return "typechange"
    return "other"


def parse_porcelain_v1_z(raw: "bytes | str") -> list[GitFileEntry]:
    """Parse ``git status --porcelain=v1 -z`` output safely.

    Records are NUL-delimited (never line-split), so filenames containing
    spaces, tabs, newlines or Unicode are preserved exactly. Rename/copy
    records carry the new path first, then the original path in the following
    NUL field (matching git's porcelain v1 ``-z`` output).
    """
    data = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else (raw or "")
    fields = data.split("\x00")
    entries: list[GitFileEntry] = []
    index = 0
    total = len(fields)
    while index < total:
        record = fields[index]
        index += 1
        if not record:
            continue
        code = record[:2]
        path = record[3:] if len(record) > 3 else ""
        x = code[0] if len(code) > 0 else " "
        y = code[1] if len(code) > 1 else " "
        orig_path: Optional[str] = None
        if x in ("R", "C") or y in ("R", "C"):
            if index < total:
                candidate = fields[index]
                index += 1
                if candidate:
                    orig_path = candidate
        entries.append(
            GitFileEntry(
                code=code,
                path=path,
                orig_path=orig_path,
                staged=x not in " ?",
                unstaged=y != " ",
                untracked=code == "??",
                change_type=_classify_git_change(code),
            )
        )
    return entries


def classify_git_error(returncode: int, stderr: str) -> str:
    """Map a git failure to a stable ERROR kind used by the GUI."""
    low = (stderr or "").lower()
    if "not a git repository" in low:
        return "NOT_A_REPO"
    if "dubious ownership" in low or "safe.directory" in low:
        return "DUBIOUS_OWNERSHIP"
    if (
        "authentication failed" in low
        or "could not read username" in low
        or "permission denied" in low
        or "invalid username or password" in low
    ):
        return "AUTH_FAILED"
    if (
        "could not resolve host" in low
        or "could not read from remote" in low
        or "network is unreachable" in low
    ):
        return "NETWORK"
    if "timed out" in low or "timeout" in low:
        return "TIMEOUT"
    if "hook" in low or "pre-commit" in low or "declined" in low:
        return "HOOK_FAILED"
    if (
        "non-fast-forward" in low
        or "fetch first" in low
        or "updates were rejected" in low
        or "! [rejected]" in low
        or "failed to push some refs" in low
    ):
        return "NON_FAST_FORWARD"
    if "nothing to commit" in low or "no changes added to commit" in low:
        return "NOTHING_TO_COMMIT"
    if "please tell me who you are" in low or "empty ident" in low:
        return "IDENTITY_MISSING"
    return "UNKNOWN"


class GitService:
    """Project-scoped Git adapter.

    Runs only the local ``git`` CLI with ``shell=False`` and an explicit argv
    list (no shell interpolation, ``--`` before user filenames). It performs no
    FSM/DB writes and no destructive operations: no force push, no
    reset/clean/rebase/merge/pull, and never ``git add .`` / ``git add -A``.
    """

    def __init__(self, project_root: Path, *, remote: str = GIT_REMOTE_NAME):
        self.project_root = Path(project_root)
        self.remote = remote

    # -- low level ------------------------------------------------------- #

    def available(self) -> bool:
        return shutil.which("git") is not None

    def _run_text(self, args: list[str], *, cwd: Path, timeout: int = GIT_READ_TIMEOUT_SEC):
        git = shutil.which("git")
        if not git:
            raise GitError("GIT_MISSING", "git executable not found")
        try:
            return subprocess.run(
                [git, *args],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError("TIMEOUT", f"git timed out after {timeout}s") from exc
        except OSError as exc:
            raise GitError("GIT_MISSING", f"git could not be executed: {exc}") from exc

    def _run_bytes(self, args: list[str], *, cwd: Path, timeout: int = GIT_READ_TIMEOUT_SEC):
        git = shutil.which("git")
        if not git:
            raise GitError("GIT_MISSING", "git executable not found")
        try:
            return subprocess.run(
                [git, *args],
                cwd=str(cwd),
                capture_output=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError("TIMEOUT", f"git timed out after {timeout}s") from exc
        except OSError as exc:
            raise GitError("GIT_MISSING", f"git could not be executed: {exc}") from exc

    # -- inspection ------------------------------------------------------ #

    def is_git_repo(self, *, cwd: Optional[Path] = None) -> bool:
        if not self.available():
            return False
        try:
            cp = self._run_text(
                ["rev-parse", "--is-inside-work-tree"],
                cwd=cwd or self.project_root,
            )
        except GitError:
            return False
        return cp.returncode == 0 and cp.stdout.strip() == "true"

    def get_repo_root(self, *, cwd: Optional[Path] = None) -> Optional[Path]:
        if not self.available():
            return None
        cp = self._run_text(["rev-parse", "--show-toplevel"], cwd=cwd or self.project_root)
        if cp.returncode != 0:
            return None
        root = cp.stdout.strip()
        return Path(root) if root else None

    def _root(self) -> Path:
        root = self.get_repo_root()
        if root is None:
            raise GitError("NOT_A_REPO", "Not a git repository.")
        return root

    def get_branch(self, repo_root: Path) -> tuple[Optional[str], bool]:
        """Return ``(branch, detached)``. Detached/unknown -> ``(None, True)``."""
        cp = self._run_text(["symbolic-ref", "--short", "-q", "HEAD"], cwd=repo_root)
        name = cp.stdout.strip()
        if cp.returncode == 0 and name:
            return name, False
        return None, True

    def get_remotes(self, repo_root: Path) -> list[str]:
        cp = self._run_text(["remote"], cwd=repo_root)
        return [line.strip() for line in cp.stdout.splitlines() if line.strip()]

    def get_remote_url(self, repo_root: Path, remote: str) -> str:
        cp = self._run_text(["remote", "get-url", remote], cwd=repo_root)
        if cp.returncode != 0:
            return ""
        return sanitize_remote_url(cp.stdout.strip())

    def get_status(self, repo_root: Path) -> list[GitFileEntry]:
        """Changed files via NUL-delimited porcelain v1 (never line-split)."""
        cp = self._run_bytes(["status", "--porcelain=v1", "-z"], cwd=repo_root)
        if cp.returncode != 0:
            stderr = (cp.stderr or b"").decode("utf-8", "replace")
            raise GitError(classify_git_error(cp.returncode, stderr), stderr.strip() or "git status failed", stderr=stderr)
        return parse_porcelain_v1_z(cp.stdout)

    def get_upstream(self, repo_root: Path, branch: str) -> Optional[str]:
        cp = self._run_text(
            ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=repo_root,
        )
        name = cp.stdout.strip()
        if cp.returncode == 0 and name and not name.startswith("@{"):
            return name
        return None

    def get_ahead_behind(self, repo_root: Path, upstream: str) -> tuple[int, int]:
        """Return ``(ahead, behind)`` relative to ``upstream``."""
        cp = self._run_text(
            ["rev-list", "--left-right", "--count", f"{upstream}...HEAD"],
            cwd=repo_root,
        )
        if cp.returncode != 0:
            raise GitError("NO_UPSTREAM", "Unable to compute ahead/behind.")
        parts = cp.stdout.split()
        if len(parts) != 2:
            raise GitError("NO_UPSTREAM", "Unexpected rev-list output.")
        return int(parts[1]), int(parts[0])  # (ahead, behind)

    def has_local_commits(self, repo_root: Path) -> bool:
        cp = self._run_text(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=repo_root)
        return cp.returncode == 0 and bool(cp.stdout.strip())

    def get_last_commit(self, repo_root: Path) -> tuple[str, str]:
        cp = self._run_text(["log", "-1", "--format=%H%x00%s"], cwd=repo_root)
        if cp.returncode != 0 or not cp.stdout.strip():
            return "", ""
        sha, _, subject = cp.stdout.strip().partition("\x00")
        return sha.strip(), subject.strip()

    # -- mutations (explicit, non-destructive) --------------------------- #

    def stage_files(self, files: list[str]) -> None:
        """Stage ONLY the given repo-relative files (``git add -- <files>``)."""
        cleaned = [str(f) for f in files if str(f)]
        if not cleaned:
            raise GitError("NO_FILES_SELECTED", "No files selected to stage.")
        for path in cleaned:
            if path in (".", "./", "-A", "--all", "*"):
                raise GitError("UNSAFE_PATHSPEC", f"Refusing unsafe pathspec: {path!r}")
        repo_root = self._root()
        cp = self._run_text(
            ["add", "--", *cleaned], cwd=repo_root, timeout=GIT_WRITE_TIMEOUT_SEC
        )
        if cp.returncode != 0:
            combined = f"{cp.stdout or ''}\n{cp.stderr or ''}"
            raise GitError(
                classify_git_error(cp.returncode, combined),
                (cp.stderr or "").strip() or (cp.stdout or "").strip() or "git add failed",
                stderr=combined.strip(),
            )

    def commit(self, message: str) -> str:
        ok, cleaned = validate_commit_message(message)
        if not ok:
            raise GitError("INVALID_MESSAGE", cleaned)
        repo_root = self._root()
        cp = self._run_text(
            ["commit", "-m", cleaned], cwd=repo_root, timeout=GIT_WRITE_TIMEOUT_SEC
        )
        if cp.returncode != 0:
            # ``git commit`` reports "nothing to commit" on stdout, so classify
            # on the combined output while still surfacing a usable message.
            combined = f"{cp.stdout or ''}\n{cp.stderr or ''}"
            raise GitError(
                classify_git_error(cp.returncode, combined),
                (cp.stderr or "").strip() or (cp.stdout or "").strip() or "git commit failed",
                stderr=combined.strip(),
            )
        return self.get_last_commit(repo_root)[0]

    def push(self, branch: str, *, set_upstream: bool = False) -> None:
        """Push the current branch. Never force; ``-u`` only when explicitly asked."""
        if not branch:
            raise GitError("NO_BRANCH", "Cannot push without a branch.")
        repo_root = self._root()
        args = ["push"]
        if set_upstream:
            args.append("-u")
        args.extend([self.remote, branch])
        cp = self._run_text(args, cwd=repo_root, timeout=GIT_WRITE_TIMEOUT_SEC)
        if cp.returncode != 0:
            combined = f"{cp.stdout or ''}\n{cp.stderr or ''}"
            raise GitError(
                classify_git_error(cp.returncode, combined),
                (cp.stderr or "").strip() or (cp.stdout or "").strip() or "git push failed",
                stderr=combined.strip(),
            )

    # -- aggregate ------------------------------------------------------- #

    def snapshot(self) -> dict:
        """One-shot repo snapshot for the GUI; never raises for expected states."""
        snap: dict = {
            "git_available": self.available(),
            "is_repo": False,
            "repo_root": "",
            "branch": "",
            "detached": False,
            "remote_name": "",
            "remote_url": "",
            "remote_present": False,
            "entries": [],
            "changed": 0,
            "staged": 0,
            "ahead": None,
            "behind": None,
            "has_upstream": False,
            "upstream": "",
            "can_push": False,
            "can_commit": False,
            "last_commit_hash": "",
            "last_commit_summary": "",
            "error": "",
        }
        if not snap["git_available"]:
            snap["error"] = "git executable not found"
            return snap
        try:
            root = self.get_repo_root()
        except GitError as exc:
            snap["error"] = str(exc)
            return snap
        if root is None:
            snap["error"] = "not a git repository"
            return snap

        snap["is_repo"] = True
        snap["repo_root"] = str(root)
        try:
            branch, detached = self.get_branch(root)
            snap["branch"] = branch or ""
            snap["detached"] = detached

            if self.remote in self.get_remotes(root):
                snap["remote_name"] = self.remote
                snap["remote_present"] = True
                snap["remote_url"] = self.get_remote_url(root, self.remote)

            entries = self.get_status(root)
            snap["entries"] = entries
            snap["staged"] = sum(1 for e in entries if e.staged)
            snap["changed"] = sum(1 for e in entries if e.unstaged or e.untracked)
            snap["last_commit_hash"], snap["last_commit_summary"] = self.get_last_commit(root)

            if branch and not detached:
                upstream = self.get_upstream(root, branch)
                snap["upstream"] = upstream or ""
                snap["has_upstream"] = bool(upstream)
                if upstream:
                    ahead, behind = self.get_ahead_behind(root, upstream)
                    snap["ahead"] = ahead
                    snap["behind"] = behind
                    snap["can_push"] = snap["remote_present"] and ahead > 0
                else:
                    # No upstream configured: LOCAL / NO UPSTREAM. Never pretend
                    # ahead == 0; allow Push Only while local commits exist.
                    snap["can_push"] = snap["remote_present"] and self.has_local_commits(root)
            snap["can_commit"] = bool(branch) and not detached
        except GitError as exc:
            snap["error"] = str(exc)
        return snap


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

    def cline_report_for_attempt(
        self, step_no: int, attempt: int
    ) -> Optional[tuple[dict, str]]:
        """The stored ``(report, report_hash)`` for one step+attempt, or None."""
        row = self.db.execute(
            "SELECT report_json, report_hash FROM cline_reports "
            "WHERE step_no=? AND attempt=? ORDER BY id DESC LIMIT 1",
            (step_no, attempt),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["report_json"]), str(row["report_hash"])

    def has_codex_review(self, step_no: int, attempt: int) -> bool:
        """Whether a Codex review already exists for this step+attempt."""
        row = self.db.execute(
            "SELECT 1 FROM codex_reviews WHERE step_no=? AND attempt=? LIMIT 1",
            (step_no, attempt),
        ).fetchone()
        return row is not None

    def has_event(self, step_no: int, event_type: str) -> bool:
        """Whether an audit event of this type already exists for the step."""
        row = self.db.execute(
            "SELECT 1 FROM events WHERE step_no=? AND event_type=? LIMIT 1",
            (step_no, event_type),
        ).fetchone()
        return row is not None

    def recover_failed_review_to_received(
        self,
        step_no: int,
        *,
        actor: str,
        reason: str,
        report_hash: str,
        attempt: int,
    ) -> None:
        """The one and only FSM bypass: ``FAILED -> CLINE_REPORT_RECEIVED``.

        Deliberately *not* a generic ``recovery_transition``: the target state is
        hard-coded here, so no other state pair can use this path. The attempt
        counter and the accepted Cline report are never touched, and the action
        is recorded as an explicit ``RECOVERY_REVIEW_RETRY`` audit event.
        """
        step = self.get_step(step_no)
        if step.state != "FAILED":
            raise RuntimeError(
                f"recover_failed_review_to_received requires FAILED, got {step.state}"
            )
        ts = now_iso()
        details = json.dumps(
            {
                "actor": actor,
                "reason": reason,
                "from_state": "FAILED",
                "to_state": "CLINE_REPORT_RECEIVED",
                "attempt": attempt,
                "report_hash": report_hash,
                "timestamp": ts,
            },
            ensure_ascii=False,
        )
        with self.db:
            self.db.execute(
                "UPDATE steps SET state='CLINE_REPORT_RECEIVED', last_update_at=? "
                "WHERE step_no=?",
                (ts, step_no),
            )
            self.db.execute(
                "INSERT INTO events(step_no,event_type,details,created_at) VALUES(?,?,?,?)",
                (
                    step_no,
                    "STATE:FAILED->CLINE_REPORT_RECEIVED",
                    "Recovered to retry Codex REVIEW (external provider failure).",
                    ts,
                ),
            )
            self.db.execute(
                "INSERT INTO events(step_no,event_type,details,created_at) VALUES(?,?,?,?)",
                (step_no, RECOVERY_REVIEW_RETRY_EVENT, details, ts),
            )

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


# ---------------------------------------------------------------------------
# Codex failure classification (retryable provider outage vs. genuine failure)
# ---------------------------------------------------------------------------

#: Categories that may be retried WITHOUT consuming an attempt, because the
#: provider - not the project - was temporarily unable to serve the request.
RETRYABLE_CODEX_CATEGORIES = frozenset(
    {
        "quota",
        "rate_limit",
        "timeout",
        "transport",
        "network",
        "provider_unavailable",
    }
)

#: JSONL event types whose message is the authoritative failure text.
CODEX_FAILURE_EVENT_TYPES = frozenset(
    {"error", "fatal", "turn.failed", "thread.failed", "task.failed"}
)

#: Explicit reset/retry-after semantics: the provider is temporarily exhausted
#: and the very same request may succeed later. This is what makes a
#: usage/quota message RETRYABLE - and it beats the generic upsell wording
#: ("upgrade", "purchase credits") that ships inside the same message.
_CODEX_RESET_MARKERS = (
    "try again at",
    "try again in",
    "try again later",
    "try again after",
    "reset at",
    "reset in",
    "resets at",
    "retry after",
    "cooldown",
    "available again",
)

#: Usage/quota wording.
_CODEX_QUOTA_MARKERS = (
    "usage limit",
    "hit your usage",
    "quota",
    "usage cap",
    "out of credits",
)

#: Rate-limit wording (retryable even without an explicit reset time).
_CODEX_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "resource_exhausted",
    "resource exhausted",
    "429",
)

#: Billing/account problems that will NOT resolve by retrying.
_CODEX_BILLING_MARKERS = (
    "billing",
    "payment required",
    "payment method",
    "insufficient",
    "no credits",
    "credit balance",
    "account disabled",
    "account suspended",
    "subscription",
    "purchase more credits",
    "upgrade to pro",
    "hard limit",
)

#: Authentication / invalid-configuration problems: never retryable.
_CODEX_AUTH_MARKERS = (
    "authentication",
    "authenticate",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "api key",
    "not authorized",
    "not logged in",
    "login required",
    "sign in",
    "401",
    "403",
)

#: Request/contract problems: the input or the output schema was rejected.
_CODEX_CONTRACT_MARKERS = (
    "invalid_request",
    "invalid request",
    "bad request",
    "schema",
    "unsupported",
    "contract",
    "400",
)

_CODEX_NETWORK_MARKERS = (
    "could not resolve host",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "connection refused",
    "connection reset",
    "econnreset",
    "econnrefused",
    "getaddrinfo",
    "no route to host",
)

_CODEX_TIMEOUT_MARKERS = (
    "timed out",
    "timeout",
    "deadline exceeded",
    "etimedout",
)

_CODEX_PROVIDER_UNAVAILABLE_MARKERS = (
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "temporarily unavailable",
    "overloaded",
    "502",
    "503",
    "504",
)


def sanitize_diagnostic(text: Any, max_chars: int = 800) -> str:
    """One-line, credential-redacted provider diagnostic safe to persist."""
    value = str(text or "").replace("\x00", " ").strip()
    value = " ".join(value.split())
    value = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|bearer)"
        r"(\s*[:=]\s*|\s+)\S+",
        r"\1\2***",
        value,
    )
    return value[:max_chars]


def _as_text(value: Any) -> str:
    """Best-effort text for a captured-process attribute (str or bytes)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value or "")


@dataclass(frozen=True)
class CodexFailure:
    """A classified Codex failure (never carries secrets)."""

    category: str
    retryable: bool
    source: str          # "jsonl" | "stderr" | "none"
    primary_error: str   # sanitized provider message


def _codex_failure_event_text(events: list[dict]) -> str:
    """The authoritative failure text carried by the JSONL error events."""
    parts: list[str] = []
    for event in events:
        if str(event.get("type", "")).strip().lower() not in CODEX_FAILURE_EVENT_TYPES:
            continue
        for key in ("error", "message", "detail", "reason"):
            value = event.get(key)
            if not value:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(str(value))
    return " | ".join(parts)


def _classify_codex_text(low: str) -> Optional[str]:
    """Category for one candidate text, or ``None`` when nothing matches."""
    if any(marker in low for marker in _CODEX_TIMEOUT_MARKERS):
        return "timeout"
    if any(marker in low for marker in _CODEX_AUTH_MARKERS):
        return "auth"
    if any(marker in low for marker in _CODEX_CONTRACT_MARKERS):
        return "contract"
    has_reset = any(marker in low for marker in _CODEX_RESET_MARKERS)
    has_quota = any(marker in low for marker in _CODEX_QUOTA_MARKERS)
    has_rate = any(marker in low for marker in _CODEX_RATE_LIMIT_MARKERS)
    has_billing = any(marker in low for marker in _CODEX_BILLING_MARKERS)
    if has_reset and (has_quota or has_rate or has_billing):
        # A usage/quota limit *with reset semantics* is temporary: retryable.
        # The upsell wording in the same message must not turn it into billing.
        return "quota"
    if has_rate:
        return "rate_limit"
    if has_billing:
        return "billing"
    if has_quota and "temporar" in low:
        return "quota"
    if has_quota:
        # No reset semantics: uncertain, so it fails closed as non-retryable.
        return "quota_unconfirmed"
    if any(marker in low for marker in _CODEX_PROVIDER_UNAVAILABLE_MARKERS):
        return "provider_unavailable"
    if any(marker in low for marker in _CODEX_NETWORK_MARKERS):
        return "network"
    if any(marker in low for marker in CODEX_TRANSPORT_MARKERS):
        return "transport"
    return None


def classify_codex_failure(
    *,
    events: Optional[list[dict]] = None,
    stderr: str = "",
    stdout: str = "",
    returncode: Optional[int] = None,
    timed_out: bool = False,
) -> CodexFailure:
    """Classify a Codex failure, preferring structured JSONL text over stderr.

    ``retryable`` means "the provider was temporarily unable to serve the
    request" (usage/quota limit with reset semantics, rate limit, timeout,
    transport, network, provider outage). Everything else - auth, billing,
    contract, schema, unknown - fails closed as NON-retryable.
    """
    if timed_out:
        return CodexFailure("timeout", True, "timeout", "Codex timed out.")

    event_text = _codex_failure_event_text(events or [])
    if event_text:
        category = _classify_codex_text(event_text.lower())
        if category is not None:
            return CodexFailure(
                category,
                category in RETRYABLE_CODEX_CATEGORIES,
                "jsonl",
                sanitize_diagnostic(event_text),
            )

    raw = "\n".join(part for part in (stderr, (stdout or "")[-4000:]) if part)
    if raw.strip():
        category = _classify_codex_text(raw.lower())
        if category is not None:
            return CodexFailure(
                category,
                category in RETRYABLE_CODEX_CATEGORIES,
                "stderr",
                sanitize_diagnostic(event_text or raw),
            )

    primary = sanitize_diagnostic(event_text or raw)
    return CodexFailure("unknown", False, "none", primary)


class CodexRunError(RuntimeError):
    """A failed Codex structured run with a classified, retryability-aware cause.

    ``retryable`` is True only for a classified external/provider outage; the
    controller then defers to ``CODEX_REVIEW_RETRYABLE`` instead of ``FAILED``.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        returncode: Optional[int] = None,
        provider: str = "",
        sanitized_error: str = "",
        source: str = "",
    ):
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.returncode = returncode
        self.provider = provider
        self.sanitized_error = sanitized_error
        self.source = source


#: Human sentence per deferred-review category (never a project failure).
REVIEW_DEFERRED_REASONS = {
    "quota": "Usage limit reached.",
    "rate_limit": "Rate limit reached.",
    "timeout": "The provider timed out.",
    "transport": "Provider transport failure.",
    "network": "Temporary network failure.",
    "provider_unavailable": "The provider is temporarily unavailable.",
}


def review_deferred_message(result: dict) -> str:
    """Human sentence for a deferred Codex REVIEW."""
    category = str((result or {}).get("category", "") or "").lower()
    return REVIEW_DEFERRED_REASONS.get(
        category, "The Codex provider is temporarily unavailable."
    )


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


def validate_cline_report(report: Any, step_no: int) -> list[str]:
    """Pure readiness/validation check for a Cline report object.

    Shared by the GUI readiness watcher and
    ``TandemController.process_cline_report`` so both always agree: the button
    can never be enabled for a report the FSM would then reject.

    Returns a list of human-readable problems; an empty list means READY. Pure
    and side-effect free: it only inspects the already-parsed JSON object (no
    file I/O, no FSM mutation).
    """
    if not isinstance(report, dict):
        return ["Cline report must be a JSON object."]

    try:
        report_step_no = int(report.get("step_no", -1))
    except (TypeError, ValueError):
        report_step_no = -1
    if report_step_no != step_no:
        # Step identity is authoritative; stop before schema noise.
        return ["Cline report step_no mismatch."]

    status = str(report.get("status", "")).upper()
    if status not in CLINE_REPORT_VALID_STATUSES:
        return [f"Invalid Cline report status: {status}"]

    return validate_against_schema(report, cline_report_schema())


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

    def provider_label(self) -> str:
        """Provider/model label for audit payloads (never a secret)."""
        return self.model or "(codex default)"

    def _failure_report(
        self,
        returncode: int,
        events: list[dict],
        stdout: str,
        stderr: str,
        trace_path: Path,
        headline: str,
        extra: str = "",
        *,
        category: str = "",
        retryable: bool = False,
        primary_error: str = "",
    ) -> str:
        """Human-readable diagnostics for a failed structured run.

        The classified provider cause leads: a usage/quota limit is presented as
        exactly that and is never reported as an MCP/model-configuration problem.
        """
        lines = [f"Codex structured run failed: {headline}"]
        if extra:
            lines.append(extra)
        lines.append(f"codex return code: {returncode}")
        lines.append(
            f"failure category: {category or 'unknown'} "
            f"(retryable={'true' if retryable else 'false'})"
        )
        if primary_error:
            lines.append(f"provider error: {primary_error}")
        fatal = fatal_event(events)
        if fatal and category not in {"quota", "rate_limit"}:
            lines.append(f"fatal event: {fatal}")
        for note in error_notes(events)[:5]:
            lines.append(f"error event: {note}")
        # The MCP/model-configuration hint only makes sense for a transport-shaped
        # outage. It must never appear as the primary cause of a quota, billing,
        # auth or contract rejection - there the JSONL provider text is
        # authoritative and the container hints are misleading.
        if category in {
            "transport",
            "network",
            "provider_unavailable",
            "timeout",
            "unknown",
        } and looks_like_transport_failure(stderr):
            lines.append(
                "detected Codex tool/MCP transport failure (for example "
                "'error decoding response body' / rmcp). The provider could not "
                "serve this request from a headless 'codex exec' run; this is a "
                "provider-side condition, not a project failure."
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
            failure = classify_codex_failure(
                stderr=_as_text(exc.stderr),
                stdout=_as_text(exc.stdout),
                timed_out=True,
            )
            raise CodexRunError(
                f"Codex timed out after {timeout}s "
                f"(failure category: {failure.category}). See {trace_path}",
                category=failure.category,
                retryable=failure.retryable,
                returncode=None,
                provider=self.provider_label(),
                sanitized_error=failure.primary_error,
                source=failure.source,
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
            failure = classify_codex_failure(
                events=events, stderr=stderr, stdout=stdout, returncode=cp.returncode
            )
            raise CodexRunError(
                self._failure_report(
                    cp.returncode,
                    events,
                    stdout,
                    stderr,
                    trace_path,
                    f"codex exec exited with code {cp.returncode}.",
                    category=failure.category,
                    retryable=failure.retryable,
                    primary_error=failure.primary_error,
                ),
                category=failure.category,
                retryable=failure.retryable,
                returncode=cp.returncode,
                provider=self.provider_label(),
                sanitized_error=failure.primary_error,
                source=failure.source,
            )

        # B. definitively fatal JSONL event -----------------------------------
        fatal = fatal_event(events)
        if fatal:
            record("failed:fatal_event")
            failure = classify_codex_failure(
                events=events, stderr=stderr, stdout=stdout, returncode=cp.returncode
            )
            raise CodexRunError(
                self._failure_report(
                    cp.returncode,
                    events,
                    stdout,
                    stderr,
                    trace_path,
                    f"Codex reported a fatal event: {fatal}",
                    category=failure.category,
                    retryable=failure.retryable,
                    primary_error=failure.primary_error,
                ),
                category=failure.category,
                retryable=failure.retryable,
                returncode=cp.returncode,
                provider=self.provider_label(),
                sanitized_error=failure.primary_error,
                source=failure.source,
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
            failure = classify_codex_failure(
                events=events, stderr=stderr, stdout=stdout, returncode=cp.returncode
            )
            raise CodexRunError(
                self._failure_report(
                    cp.returncode,
                    events,
                    stdout,
                    stderr,
                    trace_path,
                    "Codex finished but produced no valid schema-conforming result.",
                    extra=schema_problem,
                    category=failure.category,
                    retryable=failure.retryable,
                    primary_error=failure.primary_error,
                ),
                category=failure.category,
                retryable=failure.retryable,
                returncode=cp.returncode,
                provider=self.provider_label(),
                sanitized_error=failure.primary_error,
                source=failure.source,
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


def cline_report_schema() -> dict:
    """JSON Schema for the Cline ACT/REPAIR report artifact.

    Mirrors the ``report_schema`` advertised to Cline in the dispatched task.
    Extra keys are tolerated; only the required/typed keys are enforced. Python
    owns step identity, so ``step_no`` is additionally matched against the
    current step by :func:`validate_cline_report`.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "step_no": {"type": "integer"},
            "status": {"type": "string", "enum": list(CLINE_REPORT_VALID_STATUSES)},
            "files_created": {"type": "array", "items": {"type": "string"}},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "files_deleted": {"type": "array", "items": {"type": "string"}},
            "tests": {
                "type": "object",
                "properties": {
                    "passed": {"type": "integer"},
                    "failed": {"type": "integer"},
                    "command": {"type": "string"},
                },
                "required": ["passed", "failed", "command"],
            },
            "dependencies_added": {"type": "array", "items": {"type": "string"}},
            "architecture_questions": {"type": "array", "items": {"type": "string"}},
            "issues": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": [
            "step_no",
            "status",
            "files_created",
            "files_changed",
            "files_deleted",
            "tests",
            "dependencies_added",
            "architecture_questions",
            "issues",
            "summary",
        ],
    }


# --------------------------------------------------------------------------- #
# Supervisor provider registry + operator policy (M9)
# --------------------------------------------------------------------------- #
#
# The GUI selects among NON-SECRET provider ids only. API keys are read from the
# environment by the adapters themselves; only the env var NAME and
# FOUND/MISSING may ever be displayed, persisted or logged.

SUPERVISOR_POLICY_FILENAME = "supervisor_policy.json"

#: Persisted-format version (the file stays forward-readable).
SUPERVISOR_POLICY_VERSION = 1

#: Selector value -> operator-facing label. ``anthropic`` is displayed as
#: "Claude"; the policy keeps the router's canonical id so the AUTO
#: HIGH/REVIEW risk route resolves correctly.
SUPERVISOR_PROVIDER_LABELS = {
    "AUTO": "AUTO",
    ID_DEEPSEEK: "DeepSeek",
    ID_OPENAI: "OpenAI",
    ID_ANTHROPIC: "Claude",
    PROVIDER_CODEX_LEGACY: "Codex Legacy",
}

#: GUI selector order: AUTO first, exactly as specified for M9.
SUPERVISOR_PROVIDER_CHOICES = (
    "AUTO",
    ID_DEEPSEEK,
    ID_OPENAI,
    ID_ANTHROPIC,
    PROVIDER_CODEX_LEGACY,
)

#: Concrete providers a MANUAL policy may store (AUTO is a mode, not a provider).
SUPERVISOR_MANUAL_CHOICES = (
    ID_DEEPSEEK,
    ID_OPENAI,
    ID_ANTHROPIC,
    PROVIDER_CODEX_LEGACY,
)

SUPERVISOR_LABEL_TO_PROVIDER = {
    label: provider_id for provider_id, label in SUPERVISOR_PROVIDER_LABELS.items()
}

SUPERVISOR_MODE_CHOICES = (POLICY_MODE_AUTO, POLICY_MODE_MANUAL)
SUPERVISOR_BUDGET_CHOICES = (BUDGET_CHEAP, BUDGET_STANDARD, BUDGET_PREMIUM)

#: Providers whose model string may be set from the GUI. Codex Legacy keeps its
#: legacy CodexRunner configuration (M4 contract).
SUPERVISOR_MODEL_PROVIDERS = (ID_DEEPSEEK, ID_OPENAI, ID_ANTHROPIC)

#: Reserved for the future Anthropic adapter (M7); the env name follows the
#: shared ``env:NAME`` convention so a saved policy stays valid afterwards.
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Env var shown per provider. Only the NAME is ever resolved for display.
SUPERVISOR_KEY_ENV_NAMES = {
    ID_DEEPSEEK: DEEPSEEK_API_KEY_ENV,
    ID_OPENAI: OPENAI_API_KEY_ENV,
    ID_ANTHROPIC: ANTHROPIC_API_KEY_ENV,
}

#: Reverse lookup used by the GUI to render one status line per env var NAME.
SUPERVISOR_ENV_PROVIDER = {
    env_name: provider_id for provider_id, env_name in SUPERVISOR_KEY_ENV_NAMES.items()
}

#: Providers with a buildable adapter today. ``anthropic``/Claude is listed in
#: the GUI but has no adapter before M7 and is therefore never registered.
SUPERVISOR_BUILDABLE_PROVIDERS = (ID_DEEPSEEK, ID_OPENAI, PROVIDER_CODEX_LEGACY)


def supervisor_label(provider_id: str) -> str:
    """Operator-facing label for a selector value."""
    value = str(provider_id or "")
    return SUPERVISOR_PROVIDER_LABELS.get(value, value)


def supervisor_provider_from_label(label: str) -> str:
    """Inverse of :func:`supervisor_label` (unknown labels are returned as-is)."""
    value = str(label or "")
    return SUPERVISOR_LABEL_TO_PROVIDER.get(value, value)


def supervisor_model_defaults() -> dict[str, str]:
    """Default model string per API provider (display + persistence)."""
    return {
        ID_DEEPSEEK: DEFAULT_DEEPSEEK_MODEL,
        ID_OPENAI: DEFAULT_OPENAI_MODEL,
        ID_ANTHROPIC: "",
    }


def supervisor_api_key_status(
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """``FOUND``/``MISSING`` per env var NAME.

    Only the presence of a non-blank value is inspected; the value itself can
    never reach the return value, a widget, a file or a log line.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    status: dict[str, str] = {}
    for env_name in SUPERVISOR_KEY_ENV_NAMES.values():
        status[env_name] = (
            "FOUND" if str(environ.get(env_name) or "").strip() else "MISSING"
        )
    return status


def supervisor_provider_status_label(
    status: str, last_category: Optional[str] = None
) -> str:
    """Provider health -> the short label the GUI shows.

    ``READY`` / ``KEY MISSING`` / ``LIMIT`` (a READY provider whose most recent
    run failed on quota or rate limit) / ``ERROR`` / ``NOT CHECKED``.
    """
    value = str(status or "").strip().upper()
    if value == "READY":
        if str(last_category or "") in {CATEGORY_QUOTA_WITH_RESET, CATEGORY_RATE_LIMIT}:
            return "LIMIT"
        return "READY"
    if value == HEALTH_KEY_MISSING:
        return "KEY MISSING"
    if not value:
        return "NOT CHECKED"
    return f"ERROR ({value})"


def build_supervisor_provider(
    provider_id: str,
    *,
    transport: Any,
    env: Optional[Mapping[str, str]] = None,
    models: Optional[Mapping[str, str]] = None,
) -> Optional[Any]:
    """The real adapter for an API provider, or ``None`` when unsupported.

    Building an adapter performs NO I/O and never resolves a key value: the key
    is read from the environment at call time by the adapter itself.
    """
    chosen = dict(models or {})
    if provider_id == ID_DEEPSEEK:
        config = default_deepseek_config(
            model=chosen.get(ID_DEEPSEEK) or DEFAULT_DEEPSEEK_MODEL
        )
        return DeepSeekSupervisorAdapter(config, transport, env=env)
    if provider_id == ID_OPENAI:
        config = default_openai_config(
            model=chosen.get(ID_OPENAI) or DEFAULT_OPENAI_MODEL
        )
        return OpenAISupervisorAdapter(config, transport, env=env)
    # "Claude" (ID_ANTHROPIC) has no adapter before M7.
    return None


def default_supervisor_policy() -> SupervisorPolicy:
    """The unchanged M4 default: MANUAL + codex_legacy, no retry/fallback."""
    return SupervisorPolicy(
        mode=POLICY_MODE_MANUAL,
        plan_provider=PROVIDER_CODEX_LEGACY,
        review_provider=PROVIDER_CODEX_LEGACY,
        fallback_chain=(),
        max_provider_retries=0,
    )


class TandemController:
    def __init__(
        self,
        project_root: Path,
        *,
        supervisor_transport: Optional[Any] = None,
        supervisor_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.db = TandemDB(project_root)
        self.context_builder = ContextBuilder(self.db)
        self.codex = CodexRunner(project_root, model=DEFAULT_CODEX_MODEL)

        write_json_atomic(self.db.context_dir / "codex_plan.schema.json", codex_plan_schema())
        write_json_atomic(
            self.db.context_dir / "codex_review.schema.json", codex_review_schema()
        )

        # --- Provider-neutral supervisor (M4 + M9) --------------------------- #
        # self.codex is deliberately PRESERVED: "Check Codex", legacy
        # diagnostics and the existing tests keep using it directly. The
        # supervisor wraps that very instance - there is no second CodexRunner.
        #
        # M9: API providers are built on demand and cached; the shared HTTP
        # transport is created lazily and may be injected (tests stay offline).
        self._supervisor_transport = supervisor_transport
        self._supervisor_owned_transport = False
        self._supervisor_env = supervisor_env
        self._supervisor_providers: dict[str, Any] = {}
        self.supervisor_adapter = LegacyCodexSupervisorAdapter(
            self.codex,
            # Legacy Codex may echo the planner's step_no; mirror CodexRunner's
            # strip-before-validation behaviour so the persisted PLAN/REVIEW
            # payloads stay byte-identical (M4 parity gate). The controller
            # still injects the authoritative identity afterwards.
            identity_policy=IDENTITY_STRIP_STEP_NO,
            # CodexRunner.run_structured already validates structured output
            # against this same canonical schema, so the legacy path must not
            # double-validate: doing so would reject payloads the pre-M4
            # controller accepted (and change persisted artifacts).
            validation_mode=VALIDATION_RUNNER,
            config=ProviderConfig(
                provider_id=PROVIDER_CODEX_LEGACY,
                display_name="Codex Legacy",
                model=DEFAULT_CODEX_MODEL,
                enabled=True,
                supports_structured_output=True,
                cost_policy="free",
                priority=99,
            ),
        )
        self._supervisor_providers = {PROVIDER_CODEX_LEGACY: self.supervisor_adapter}

        # M9: the operator policy is loaded from
        # <project>/.worker_tandem/supervisor_policy.json. No file (the default)
        # reproduces the exact M4 behaviour: MANUAL + codex_legacy, one
        # registered provider.
        self.supervisor_policy = self.load_supervisor_policy()
        self.supervisor_router = self.build_supervisor_router(self.supervisor_policy)

    def close(self) -> None:
        if self._supervisor_owned_transport and self._supervisor_transport is not None:
            try:
                self._supervisor_transport.close()
            except Exception:
                pass
            self._supervisor_transport = None
            self._supervisor_owned_transport = False
        self.db.close()

    # ------------------------------------------------------------------ #
    # Supervisor policy + provider registry (M9)
    # ------------------------------------------------------------------ #
    def supervisor_policy_path(self) -> Path:
        """Where the non-secret operator policy is persisted (atomic writes)."""
        return self.db.workdir / SUPERVISOR_POLICY_FILENAME

    def read_supervisor_policy_file(self) -> dict:
        """The raw persisted policy; ``{}`` when absent or unreadable.

        A missing or corrupt file is never fatal: the M4 default applies
        instead, so an existing project always opens.
        """
        path = self.supervisor_policy_path()
        if not path.exists():
            return {}
        try:
            payload = read_json(path)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def load_supervisor_policy(self) -> SupervisorPolicy:
        """The effective policy: the saved file, else the M4 default."""
        payload = self.read_supervisor_policy_file()
        if not payload:
            return default_supervisor_policy()
        return SupervisorPolicy.from_dict(payload)

    def supervisor_models(self) -> dict[str, str]:
        """Per-provider model strings (defaults merged with the saved file)."""
        saved = self.read_supervisor_policy_file().get("models") or {}
        models = supervisor_model_defaults()
        for provider_id in SUPERVISOR_MODEL_PROVIDERS:
            value = str(saved.get(provider_id) or "").strip()
            if value:
                models[provider_id] = value
        return models

    def supervisor_enabled_providers(
        self, payload: Optional[dict] = None
    ) -> tuple[str, ...]:
        """Provider ids the operator allowed the router to use."""
        data = self.read_supervisor_policy_file() if payload is None else payload
        saved = data.get("enabled_providers")
        if not isinstance(saved, (list, tuple)) or not saved:
            return SUPERVISOR_MANUAL_CHOICES
        chosen = tuple(
            str(item) for item in saved if str(item) in SUPERVISOR_MANUAL_CHOICES
        )
        return chosen or SUPERVISOR_MANUAL_CHOICES

    @staticmethod
    def supervisor_provider_available(provider_id: str) -> bool:
        """Whether an adapter for this provider exists today."""
        return str(provider_id or "") in SUPERVISOR_BUILDABLE_PROVIDERS

    def supervisor_registered_ids(self, policy: SupervisorPolicy) -> tuple[str, ...]:
        """Provider ids the router should register for this policy.

        ``codex_legacy`` is always present (it is the controller's own runner);
        API providers are added only when the policy can actually route to them,
        so the M4 default keeps registering ``codex_legacy`` alone. Unbuildable
        ids (Claude before M7) stay in the policy and are skipped by the router.
        """
        enabled = self.supervisor_enabled_providers()
        wanted: list[str] = []
        if policy.is_auto:
            for risk in ("LOW", "MEDIUM", "HIGH"):
                for operation in (PLAN_OPERATION, REVIEW_OPERATION):
                    wanted.extend(policy.candidates_for(operation, risk))
        else:
            wanted.append(policy.provider_for(PLAN_OPERATION))
            wanted.append(policy.provider_for(REVIEW_OPERATION))

        ordered: list[str] = [PROVIDER_CODEX_LEGACY]
        for provider_id in wanted:
            if (
                provider_id
                and provider_id not in ordered
                and provider_id in enabled
                and self.supervisor_provider_available(provider_id)
            ):
                ordered.append(provider_id)
        return tuple(ordered)

    def supervisor_transport(self) -> Any:
        """The shared HTTP transport for API providers (lazily created)."""
        if self._supervisor_transport is None:
            self._supervisor_transport = HttpTransport()
            self._supervisor_owned_transport = True
        return self._supervisor_transport

    def supervisor_provider(self, provider_id: str) -> Optional[Any]:
        """The adapter for ``provider_id`` (built on demand, cached, no I/O)."""
        if provider_id == PROVIDER_CODEX_LEGACY:
            return self.supervisor_adapter
        if provider_id in self._supervisor_providers:
            return self._supervisor_providers[provider_id]
        adapter = build_supervisor_provider(
            provider_id,
            transport=self.supervisor_transport(),
            env=self._supervisor_env,
            models=self.supervisor_models(),
        )
        if adapter is not None:
            self._supervisor_providers[provider_id] = adapter
        return adapter

    def build_supervisor_router(
        self, policy: Optional[SupervisorPolicy] = None
    ) -> SupervisorRouter:
        """A router over exactly the providers this policy may route to."""
        effective = policy if policy is not None else self.supervisor_policy
        adapters: dict[str, Any] = {}
        for provider_id in self.supervisor_registered_ids(effective):
            adapter = self.supervisor_provider(provider_id)
            if adapter is not None:
                adapters[provider_id] = adapter
        if PROVIDER_CODEX_LEGACY not in adapters:
            adapters[PROVIDER_CODEX_LEGACY] = self.supervisor_adapter
        return SupervisorRouter(adapters, effective, emit=self._supervisor_emit)

    def save_supervisor_policy(
        self,
        policy: SupervisorPolicy,
        *,
        models: Optional[Mapping[str, str]] = None,
        enabled_providers: Optional[tuple[str, ...]] = None,
    ) -> dict:
        """Persist the non-secret policy atomically and apply it immediately.

        Only provider ids, mode/budget/fallback policy and model STRINGS are
        written: no API key, token or environment value can reach the file.
        """
        payload = policy.to_dict()
        payload["version"] = SUPERVISOR_POLICY_VERSION
        current_models = self.supervisor_models()
        payload["models"] = {
            provider_id: str(
                (models or {}).get(provider_id) or current_models.get(provider_id) or ""
            )
            for provider_id in SUPERVISOR_MODEL_PROVIDERS
        }
        payload["enabled_providers"] = list(
            enabled_providers or self.supervisor_enabled_providers()
        )
        write_json_atomic(self.supervisor_policy_path(), payload)

        # Model changes must rebuild the adapters (they are cached per provider).
        self._supervisor_providers = {PROVIDER_CODEX_LEGACY: self.supervisor_adapter}
        self.supervisor_policy = SupervisorPolicy.from_dict(payload)
        self.supervisor_router = self.build_supervisor_router(self.supervisor_policy)
        return payload

    def supervisor_key_status(self) -> dict[str, str]:
        """FOUND/MISSING per key env var (never a value)."""
        return supervisor_api_key_status(self._supervisor_env)

    def last_provider_category(self, provider_id: str) -> Optional[str]:
        """Most recent failure category recorded for a provider (in-memory).

        Used only to render the GUI's ``LIMIT`` state; a more recent success
        clears it. Nothing is persisted.
        """
        history = getattr(self.supervisor_router, "route_history", None) or []
        for entry in reversed(list(history)):
            if entry.get("provider") != provider_id:
                continue
            if entry.get("success"):
                return None
            for failure in entry.get("failures") or []:
                name, _, category = str(failure).partition("=")
                if (
                    name == provider_id
                    and category
                    and not category.startswith("skipped:")
                ):
                    return category
        return None

    def provider_status(self) -> dict:
        """Per-provider operator status: health + key presence (no completion)."""
        keys = self.supervisor_key_status()
        models = self.supervisor_models()
        registered = set(self.supervisor_router.registered_providers())
        status: dict[str, dict] = {}
        for provider_id in SUPERVISOR_MANUAL_CHOICES:
            env_name = SUPERVISOR_KEY_ENV_NAMES.get(provider_id, "")
            entry: dict[str, Any] = {
                "provider": provider_id,
                "label": supervisor_label(provider_id),
                "model": models.get(provider_id, ""),
                "key_env": env_name,
                "key": keys.get(env_name, "N/A") if env_name else "N/A",
                "registered": provider_id in registered,
                "health": None,
                "status": "NOT CHECKED",
                "detail": "",
                "last_category": None,
            }
            if not self.supervisor_provider_available(provider_id):
                entry["status"] = "UNAVAILABLE"
                entry["detail"] = "no adapter yet (Claude lands with M7)"
                status[provider_id] = entry
                continue
            adapter = self.supervisor_provider(provider_id)
            try:
                health_result = adapter.health_check()
                entry["health"] = getattr(health_result, "status", None)
                entry["last_category"] = self.last_provider_category(provider_id)
                entry["status"] = supervisor_provider_status_label(
                    entry["health"], entry["last_category"]
                )
                entry["detail"] = sanitize_diagnostic(
                    getattr(health_result, "detail", "") or ""
                )
                entry["model"] = getattr(health_result, "model", None) or entry["model"]
            except Exception as exc:  # a probe must never break the GUI
                entry["status"] = "ERROR"
                entry["detail"] = sanitize_diagnostic(exc)
            status[provider_id] = entry
        return status

    def current_routes(self) -> dict:
        """The effective PLAN/REVIEW route for the current step (read-only)."""
        step = self.db.current_step()
        risk = step.risk if step else "MEDIUM"
        policy = self.supervisor_policy
        routes: dict[str, dict] = {}
        for operation in (PLAN_OPERATION, REVIEW_OPERATION):
            chain = policy.candidates_for(operation, risk)
            routes[operation.lower()] = {
                "operation": operation,
                "mode": policy.mode,
                "risk": risk,
                "chain": list(chain),
                "primary": chain[0] if chain else None,
                "label": " -> ".join(supervisor_label(p) for p in chain) or "(none)",
            }
        return routes

    # -- supervisor plumbing (M4) ------------------------------------------- #

    def _supervisor_emit(self, event_type: str, payload: dict) -> None:
        """Audit hook for the supervisor router.

        Writes the provider-neutral event into the EXISTING ``events`` table
        (no schema change) and never logs secrets: the router sanitizes every
        value before it reaches this hook.
        """
        step_no = payload.get("step_no")
        self.db.event(
            step_no if isinstance(step_no, int) else None,
            event_type,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )

    def supervisor_label(self) -> str:
        """Short, read-only provider label for the GUI ('codex_legacy')."""
        summary = self.supervisor_router.summary()
        plan_provider = summary.get("plan_provider", "")
        review_provider = summary.get("review_provider", "")
        if plan_provider == review_provider:
            return plan_provider
        return f"{plan_provider} / {review_provider}"

    def supervisor_health(self) -> dict:
        """Health of every registered supervisor provider (cheap probes)."""
        return self.supervisor_router.health()

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
        """Backward-compatible alias (M4): PLAN now runs through the router."""
        return self.run_supervisor_plan()

    def run_supervisor_plan(self) -> dict:
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

        # M4: provider execution is routed through the provider-neutral
        # supervisor. The prompt, context, artifact paths and FSM semantics
        # above and below this block are unchanged.
        supervisor_request = SupervisorPlanRequest(
            step_no=step.step_no,
            attempt=attempt,
            risk=step.risk,
            requires_human=step.requires_human,
            prompt=prompt,
            context=context,
            schema=codex_plan_schema(),
        )
        try:
            supervisor_result = self.supervisor_router.plan(
                supervisor_request,
                risk=step.risk,
                requires_human=step.requires_human,
            )
        except Exception as exc:
            self.db.transition(step.step_no, "FAILED", "Codex PLAN failed.")
            raise self._provider_failure(exc)
        result = dict(supervisor_result.payload)

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

    def expected_cline_report_path(self, step_no: int) -> Path:
        """The exact, atomic final report artifact path for ``step_no``."""
        return self.db.from_cline / f"step_{step_no:03d}_report.json"

    def check_cline_report_readiness(self) -> dict:
        """Read-only readiness probe for the current step's Cline report.

        Never mutates the FSM and only ever inspects the exact expected report
        path (no filesystem scan). Returns a dict with:
            status:  "WAITING" | "READY" | "INVALID" | "N/A"
            detail:  short human-readable message
            path:    expected report path (str) or ""
            step_no: current step number, or None
            size:    file size in bytes when present, else None
            mtime:   file mtime (epoch seconds) when present, else None
            report:  parsed JSON object when status == "READY", else None

        A file whose size/mtime changes during the read is treated as WAITING
        ("report still being written") and is never validated. Only the atomic
        final filename is ever considered; temp/partial artifacts are ignored.
        """
        step = self.db.current_step()
        if not step or step.state != "CLINE_DISPATCHED":
            return {
                "status": "N/A",
                "detail": "no dispatch",
                "path": "",
                "step_no": None,
                "size": None,
                "mtime": None,
                "report": None,
            }

        path = self.expected_cline_report_path(step.step_no)
        result = {
            "status": "WAITING",
            "detail": "report not found",
            "path": str(path),
            "step_no": step.step_no,
            "size": None,
            "mtime": None,
            "report": None,
        }
        try:
            stat_before = path.stat()
        except OSError:
            return result

        result["size"] = stat_before.st_size
        result["mtime"] = stat_before.st_mtime
        if stat_before.st_size > CLINE_REPORT_MAX_BYTES:
            result["status"] = "INVALID"
            result["detail"] = "report exceeds size limit"
            return result

        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            result["status"] = "INVALID"
            result["detail"] = f"Cline report unreadable: {exc}"
            return result

        try:
            stat_after = path.stat()
        except OSError:
            return result
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        ):
            # The file changed while we read it: do not validate a moving
            # target, re-check on the next polling cycle instead.
            result["detail"] = "report still being written"
            result["size"] = stat_after.st_size
            result["mtime"] = stat_after.st_mtime
            return result

        result["size"] = stat_after.st_size
        result["mtime"] = stat_after.st_mtime

        try:
            report = json.loads(raw)
        except (ValueError, TypeError):
            result["status"] = "INVALID"
            result["detail"] = "JSON parse error"
            return result

        problems = validate_cline_report(report, step.step_no)
        if problems:
            result["status"] = "INVALID"
            result["detail"] = "; ".join(problems)
            return result

        result["status"] = "READY"
        result["detail"] = path.name
        result["report"] = report
        return result

    def process_cline_report(self) -> dict:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "CLINE_DISPATCHED":
            raise RuntimeError(f"Expected CLINE_DISPATCHED, got {step.state}")

        # The watcher and the FSM share one validator: the button can never be
        # enabled for a report the FSM would then reject.
        readiness = self.check_cline_report_readiness()
        status_kind = readiness["status"]
        if status_kind == "WAITING":
            raise FileNotFoundError(
                f"Cline report not found: {self.expected_cline_report_path(step.step_no)}"
            )
        if status_kind != "READY":
            raise RuntimeError(str(readiness.get("detail") or "Cline report invalid."))

        report = readiness["report"]
        status = str(report.get("status", "")).upper()

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
        """Backward-compatible alias (M4): REVIEW runs through the router."""
        return self.run_supervisor_review()

    def run_supervisor_review(self) -> dict:
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state not in {"CLINE_REPORT_RECEIVED", CODEX_REVIEW_RETRYABLE}:
            raise RuntimeError(
                "Expected CLINE_REPORT_RECEIVED or "
                f"{CODEX_REVIEW_RETRYABLE}, got {step.state}"
            )
        if self.db.cline_report_for_attempt(step.step_no, step.attempt) is None:
            raise RuntimeError(
                f"Step {step.step_no} attempt {step.attempt} has no accepted Cline report."
            )
        if self.db.has_codex_review(step.step_no, step.attempt):
            raise RuntimeError(
                f"Step {step.step_no} attempt {step.attempt} already has a Codex review."
            )

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

        # M4: provider execution is routed through the provider-neutral
        # supervisor. The accepted-report gate, attempt, artifact paths and
        # APPROVE/REVISE/BLOCKED transitions are unchanged.
        supervisor_request = SupervisorReviewRequest(
            step_no=step.step_no,
            attempt=step.attempt,
            risk=step.risk,
            requires_human=step.requires_human,
            prompt=prompt,
            context=context,
            schema=codex_review_schema(),
        )
        try:
            supervisor_result = self.supervisor_router.review(
                supervisor_request,
                risk=step.risk,
                requires_human=step.requires_human,
            )
        except SupervisorRunError as exc:
            if exc.retryable:
                # External/provider outage: the project implementation is NOT
                # implicated. Park the review for a human retry instead of
                # burning the step into FAILED at max_attempts.
                return self._defer_review(step, exc)
            self.db.transition(step.step_no, "FAILED", "Codex REVIEW failed.")
            raise self._provider_failure(exc)
        except Exception as exc:
            # Genuine/unclassified failure keeps the FAILED semantics.
            self.db.transition(step.step_no, "FAILED", "Codex REVIEW failed.")
            raise self._provider_failure(exc)

        # Python/FSM owns step identity (same rule as PLAN).
        review = dict(supervisor_result.payload)
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

    def _defer_review(self, step: TandemStep, exc: SupervisorRunError) -> dict:
        """Park REVIEW in ``CODEX_REVIEW_RETRYABLE`` for an external outage.

        The accepted Cline report, the attempt counter and any review are
        untouched; only the FSM state changes, plus a structured audit event
        that carries the real provider cause.

        M4 parity: the audit payload keeps the ORIGINATING provider's own
        classification (``origin_*``) so legacy Codex diagnostics stay exactly
        as before even though the canonical supervisor taxonomy renames some
        categories (for example Codex ``quota`` -> ``quota_with_reset``).
        """
        stored = self.db.cline_report_for_attempt(step.step_no, step.attempt)
        report_hash = stored[1] if stored else ""
        provider = self._provider_label(exc)
        category = self._provider_category(exc)
        returncode = self._provider_returncode(exc)
        detail = exc.sanitized_message or str(exc)
        self.db.transition(step.step_no, CODEX_REVIEW_RETRYABLE, category)
        self.db.event(
            step.step_no,
            CODEX_REVIEW_RETRYABLE_EVENT,
            json.dumps(
                {
                    "provider": provider,
                    "returncode": returncode,
                    "category": category,
                    "retryable": True,
                    "sanitized_error": detail,
                    "source": getattr(exc, "origin_source", "") or "",
                    "attempt": step.attempt,
                    "report_hash": report_hash,
                    "timestamp": now_iso(),
                },
                ensure_ascii=False,
            ),
        )
        return {
            "deferred": True,
            "step_no": step.step_no,
            "attempt": step.attempt,
            "state": CODEX_REVIEW_RETRYABLE,
            "category": category,
            "retryable": True,
            "provider": provider,
            "returncode": returncode,
            "detail": detail,
        }

    # -- provider-neutral <-> legacy bridges (M4) --------------------------- #

    @staticmethod
    def _provider_category(exc: SupervisorRunError) -> str:
        """The originating provider's own category when the bridge kept it."""
        return getattr(exc, "origin_category", None) or exc.category

    @staticmethod
    def _provider_label(exc: SupervisorRunError) -> str:
        """The originating provider's own label when the bridge kept it."""
        return getattr(exc, "origin_provider", None) or exc.provider

    @staticmethod
    def _provider_returncode(exc: SupervisorRunError) -> Optional[int]:
        """The originating provider's own numeric return code, if any."""
        code = getattr(exc, "origin_code", None)
        if code is None:
            return None
        try:
            return int(code)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _provider_failure(exc: Exception) -> Exception:
        """The originating provider exception, when the bridge preserved one.

        The supervisor error is always raised ``from`` the provider's own
        exception, so legacy callers that expect ``CodexRunError`` keep working
        while the neutral contract still surfaces ``SupervisorRunError``.
        """
        cause = getattr(exc, "__cause__", None)
        if isinstance(cause, Exception) and not isinstance(cause, SupervisorRunError):
            return cause
        return exc

    def review_recovery_qualified(self) -> bool:
        """Read-only: may the one-time FAILED -> review-retry recovery run?

        Mirrors every guard of
        :meth:`recover_failed_review_to_received` so the GUI only offers the
        recovery when it would actually succeed.
        """
        step = self.db.current_step()
        if not step or step.state != "FAILED":
            return False
        stored = self.db.cline_report_for_attempt(step.step_no, step.attempt)
        if stored is None:
            return False
        report, stored_hash = stored
        recomputed = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
        )
        if recomputed != stored_hash:
            return False
        if self.db.has_codex_review(step.step_no, step.attempt):
            return False
        if self.db.has_event(step.step_no, RECOVERY_REVIEW_RETRY_EVENT):
            return False
        return True

    def recover_failed_review_to_received(
        self, *, actor: str = "human", reason: str = ""
    ) -> dict:
        """ONE-TIME audited recovery: ``FAILED -> CLINE_REPORT_RECEIVED``.

        Restores only the minimum state needed to retry Codex REVIEW. It never
        changes the attempt counter, never re-runs PLAN or Cline, and never
        touches the accepted Cline report.
        """
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state != "FAILED":
            raise RuntimeError(f"Recovery requires FAILED, got {step.state}")
        stored = self.db.cline_report_for_attempt(step.step_no, step.attempt)
        if stored is None:
            raise RuntimeError(
                f"Step {step.step_no} attempt {step.attempt} has no accepted Cline report."
            )
        report, stored_hash = stored
        recomputed = sha256_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
        )
        if recomputed != stored_hash:
            raise RuntimeError(
                "The accepted Cline report does not match its stored hash; "
                "refusing recovery."
            )
        if self.db.has_codex_review(step.step_no, step.attempt):
            raise RuntimeError(
                f"Step {step.step_no} attempt {step.attempt} already has a Codex review."
            )
        if self.db.has_event(step.step_no, RECOVERY_REVIEW_RETRY_EVENT):
            raise RuntimeError(
                "Review recovery has already been used for this step; "
                "refusing a second run."
            )
        attempt_before = step.attempt
        self.db.recover_failed_review_to_received(
            step.step_no,
            actor=actor or "human",
            reason=reason
            or "Recovered to retry Codex REVIEW after an external provider failure.",
            report_hash=stored_hash,
            attempt=step.attempt,
        )
        after = self.db.get_step(step.step_no)
        if after.attempt != attempt_before:
            raise RuntimeError("Internal error: recovery changed the attempt count.")
        return {
            "step_no": after.step_no,
            "attempt": after.attempt,
            "state": after.state,
            "report_hash": stored_hash,
        }

    def retry_codex_review(self) -> dict:
        """Backward-compatible alias (M4)."""
        return self.retry_supervisor_review()

    def retry_supervisor_review(self) -> dict:
        """Human-triggered retry of supervisor REVIEW; never consumes an attempt."""
        step = self.db.current_step()
        if not step:
            raise RuntimeError("No active step.")
        if step.state not in {CODEX_REVIEW_RETRYABLE, "CLINE_REPORT_RECEIVED"}:
            raise RuntimeError(
                f"Retry Codex REVIEW requires {CODEX_REVIEW_RETRYABLE} "
                f"(or the recovered CLINE_REPORT_RECEIVED), got {step.state}"
            )
        stored = self.db.cline_report_for_attempt(step.step_no, step.attempt)
        if stored is None:
            raise RuntimeError(
                f"Step {step.step_no} attempt {step.attempt} has no accepted Cline report."
            )
        if self.db.has_codex_review(step.step_no, step.attempt):
            raise RuntimeError(
                f"Step {step.step_no} attempt {step.attempt} already has a Codex review."
            )
        self.db.event(
            step.step_no,
            CODEX_REVIEW_RETRY_EVENT,
            json.dumps(
                {
                    "attempt": step.attempt,
                    "report_hash": stored[1],
                    "timestamp": now_iso(),
                },
                ensure_ascii=False,
            ),
        )
        return self.run_codex_review()

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
        self.supervisor_var = tk.StringVar(value="Supervisor: —")
        self.footer_var = tk.StringVar(value="Worker_tandem is separate from Mini Worker.")

        # Cline report readiness watcher (Tk-safe polling; detection only).
        self.cline_report_var = tk.StringVar(value="Cline report: —")
        self._watcher_after_id: Optional[str] = None
        self._watcher_target: Optional[tuple[int, str]] = None
        self._cline_report_status = "N/A"
        self._cline_report_detail = ""
        self._cline_report_ready = False
        self._cline_report_cache: Optional[dict] = None
        self._cline_report_latest: dict = {
            "status": "N/A",
            "detail": "",
            "step_no": None,
            "size": None,
            "mtime": None,
        }

        # Git module (project-scoped utility; never touches the FSM).
        self.git_repo_var = tk.StringVar(value="Repository: —")
        self.git_branch_var = tk.StringVar(value="Branch: —")
        self.git_remote_var = tk.StringVar(value="Remote: —")
        self.git_changed_var = tk.StringVar(value="Changed: 0")
        self.git_staged_var = tk.StringVar(value="Staged: 0")
        self.git_ahead_var = tk.StringVar(value="Ahead: —")
        self.git_behind_var = tk.StringVar(value="Behind: —")
        self.git_state_var = tk.StringVar(value="Git: —")
        self.git_last_commit_var = tk.StringVar(value="Last commit: —")
        self.commit_message_var = tk.StringVar(value="")
        self._git_service: Optional[GitService] = None
        self._git_snapshot: dict = {}
        self._git_tree_paths: dict[str, str] = {}
        self._git_selected_paths: list[str] = []

        # Supervisor policy controls (M9). Only NON-SECRET provider-level
        # choices live here: API keys are read from the environment by the
        # adapters and are never displayed (only FOUND/MISSING is shown).
        self.supervisor_mode_var = tk.StringVar(value=POLICY_MODE_MANUAL)
        self.supervisor_plan_var = tk.StringVar(value=supervisor_label(PROVIDER_CODEX_LEGACY))
        self.supervisor_review_var = tk.StringVar(value=supervisor_label(PROVIDER_CODEX_LEGACY))
        self.supervisor_budget_var = tk.StringVar(value=BUDGET_STANDARD)
        self.supervisor_model_vars = {
            provider_id: tk.StringVar(value="")
            for provider_id in SUPERVISOR_MODEL_PROVIDERS
        }
        self.provider_status_vars = {
            provider_id: tk.StringVar(
                value=f"{supervisor_label(provider_id)}: not checked"
            )
            for provider_id in SUPERVISOR_MANUAL_CHOICES
        }
        self.provider_key_vars = {
            env_name: tk.StringVar(value=f"{env_name}: —")
            for env_name in SUPERVISOR_KEY_ENV_NAMES.values()
        }
        self.supervisor_plan_route_var = tk.StringVar(value="Current PLAN route: —")
        self.supervisor_review_route_var = tk.StringVar(value="Current REVIEW route: —")
        self._provider_status_cache: dict = {}
        self._supervisor_ui_key: Optional[tuple] = None

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
            self.cline_report_var,
            self.codex_var,
            self.supervisor_var,
        ):
            ttk.Label(info, textvariable=var).pack(anchor="w", pady=1)

        actions = ttk.LabelFrame(self.root, text="Workflow", padding=10)
        actions.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_migrate = ttk.Button(
            actions, text="Import from Mini Worker", command=self.migrate_from_mini
        )
        self.btn_migrate.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_plan = ttk.Button(
            actions, text="1. Run Supervisor PLAN", command=self.run_codex_plan
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
            actions, text="4. Run Supervisor REVIEW", command=self.run_codex_review
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

        # Human-triggered retry of a supervisor REVIEW that could not run for an
        # external/provider reason (usage limit, timeout, transport, network).
        # It never consumes an attempt and never re-runs Cline or PLAN.
        self.btn_retry_review = ttk.Button(
            actions, text="Retry Supervisor REVIEW", command=self.retry_codex_review
        )
        self.btn_retry_review.grid(row=2, column=2, padx=4, pady=4, sticky="ew")

        self.btn_sync = ttk.Button(
            actions,
            text=f"Sync STEP description from {CANONICAL_PLAN_NAME}",
            command=self.sync_from_canonical_plan,
        )
        self.btn_sync.grid(row=2, column=0, padx=4, pady=4, sticky="ew")

        for i in range(3):
            actions.columnconfigure(i, weight=1)

        self._build_supervisor_section()
        self._build_git_section()

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

    def _build_supervisor_section(self) -> None:
        """Operator controls for the multi-provider supervisor (M9).

        Only NON-SECRET, provider-level values can be chosen here; API keys stay
        in the environment (hence only FOUND/MISSING is ever rendered) and no
        key entry field exists in V1.
        """
        section = ttk.LabelFrame(self.root, text="Supervisor", padding=10)
        section.pack(fill="x", padx=10, pady=(0, 8))

        controls = ttk.Frame(section)
        controls.pack(fill="x")

        ttk.Label(controls, text="Mode:").grid(row=0, column=0, sticky="w")
        self.supervisor_mode_combo = ttk.Combobox(
            controls,
            textvariable=self.supervisor_mode_var,
            values=list(SUPERVISOR_MODE_CHOICES),
            state="readonly",
            width=10,
        )
        self.supervisor_mode_combo.grid(row=0, column=1, sticky="w", padx=(4, 16))

        ttk.Label(controls, text="PLAN provider:").grid(row=0, column=2, sticky="w")
        self.supervisor_plan_combo = ttk.Combobox(
            controls,
            textvariable=self.supervisor_plan_var,
            values=[supervisor_label(p) for p in SUPERVISOR_PROVIDER_CHOICES],
            state="readonly",
            width=16,
        )
        self.supervisor_plan_combo.grid(row=0, column=3, sticky="w", padx=(4, 16))

        ttk.Label(controls, text="REVIEW provider:").grid(row=0, column=4, sticky="w")
        self.supervisor_review_combo = ttk.Combobox(
            controls,
            textvariable=self.supervisor_review_var,
            values=[supervisor_label(p) for p in SUPERVISOR_PROVIDER_CHOICES],
            state="readonly",
            width=16,
        )
        self.supervisor_review_combo.grid(row=0, column=5, sticky="w", padx=(4, 16))

        ttk.Label(controls, text="Budget:").grid(row=0, column=6, sticky="w")
        self.supervisor_budget_combo = ttk.Combobox(
            controls,
            textvariable=self.supervisor_budget_var,
            values=list(SUPERVISOR_BUDGET_CHOICES),
            state="readonly",
            width=12,
        )
        self.supervisor_budget_combo.grid(row=0, column=7, sticky="w", padx=(4, 0))

        status = ttk.Frame(section)
        status.pack(fill="x", pady=(8, 0))
        for column, provider_id in enumerate(SUPERVISOR_MANUAL_CHOICES):
            ttk.Label(
                status,
                textvariable=self.provider_status_vars[provider_id],
                anchor="w",
            ).grid(row=0, column=column, sticky="w", padx=(0, 18))

        keys = ttk.Frame(section)
        keys.pack(fill="x", pady=(2, 0))
        for column, env_name in enumerate(SUPERVISOR_KEY_ENV_NAMES.values()):
            ttk.Label(keys, textvariable=self.provider_key_vars[env_name]).grid(
                row=0, column=column, sticky="w", padx=(0, 18)
            )

        routes = ttk.Frame(section)
        routes.pack(fill="x", pady=(2, 0))
        ttk.Label(routes, textvariable=self.supervisor_plan_route_var).pack(anchor="w")
        ttk.Label(routes, textvariable=self.supervisor_review_route_var).pack(
            anchor="w"
        )

        models = ttk.Frame(section)
        models.pack(fill="x", pady=(8, 0))
        for column, provider_id in enumerate(SUPERVISOR_MODEL_PROVIDERS):
            ttk.Label(models, text=f"{supervisor_label(provider_id)} model:").grid(
                row=0, column=column * 2, sticky="w"
            )
            ttk.Entry(
                models,
                textvariable=self.supervisor_model_vars[provider_id],
                width=22,
            ).grid(row=0, column=column * 2 + 1, sticky="w", padx=(4, 16))

        buttons = ttk.Frame(section)
        buttons.pack(fill="x", pady=(8, 0))
        self.btn_check_providers = ttk.Button(
            buttons, text="Check Providers", command=self.check_providers
        )
        self.btn_check_providers.grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.btn_save_policy = ttk.Button(
            buttons, text="Save Policy", command=self.save_policy
        )
        self.btn_save_policy.grid(row=0, column=1, padx=(0, 6), sticky="w")
        ttk.Label(
            buttons,
            text=(
                "Keys come from the environment (DEEPSEEK_API_KEY / "
                "OPENAI_API_KEY / ANTHROPIC_API_KEY) and are never stored, "
                "entered or displayed."
            ),
        ).grid(row=0, column=2, sticky="w", padx=(12, 0))

    def _build_git_section(self) -> None:
        git = ttk.LabelFrame(self.root, text="Git", padding=10)
        git.pack(fill="x", padx=10, pady=(0, 8))

        info = ttk.Frame(git)
        info.pack(fill="x")
        grid_rows = (
            (self.git_repo_var, 0, 0, 4),
            (self.git_branch_var, 1, 0, 2),
            (self.git_remote_var, 1, 2, 2),
            (self.git_changed_var, 2, 0, 1),
            (self.git_staged_var, 2, 1, 1),
            (self.git_ahead_var, 2, 2, 1),
            (self.git_behind_var, 2, 3, 1),
            (self.git_state_var, 3, 0, 1),
            (self.git_last_commit_var, 3, 1, 3),
        )
        for var, row, col, span in grid_rows:
            ttk.Label(info, textvariable=var, anchor="w").grid(
                row=row, column=col, columnspan=span, sticky="w", padx=(0, 14), pady=1
            )
        for col in range(4):
            info.columnconfigure(col, weight=1)

        ttk.Label(
            git,
            text="Changed files — select the files to stage (nothing is pre-selected):",
        ).pack(anchor="w", pady=(6, 2))
        self.git_tree = ttk.Treeview(
            git,
            columns=("status", "file", "note"),
            show="headings",
            selectmode="extended",
            height=5,
        )
        self.git_tree.heading("status", text="St")
        self.git_tree.heading("file", text="File")
        self.git_tree.heading("note", text="Note")
        self.git_tree.column("status", width=42, anchor="center")
        self.git_tree.column("file", width=640)
        self.git_tree.column("note", width=180)
        self.git_tree.pack(fill="x")
        self.git_tree.bind(
            "<<TreeviewSelect>>", lambda _event: self._on_git_selection_change()
        )

        msg_row = ttk.Frame(git)
        msg_row.pack(fill="x", pady=(8, 0))
        ttk.Label(msg_row, text="Commit message:").pack(side="left")
        self.commit_entry = ttk.Entry(msg_row, textvariable=self.commit_message_var)
        self.commit_entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.commit_message_var.trace_add(
            "write", lambda *_args: self._on_commit_message_change()
        )

        buttons = ttk.Frame(git)
        buttons.pack(fill="x", pady=(8, 0))
        self.btn_git_status = ttk.Button(buttons, text="Git Status", command=self.git_status)
        self.btn_git_status.grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        self.btn_git_commit = ttk.Button(
            buttons, text="Commit Local", command=self.git_commit_local
        )
        self.btn_git_commit.grid(row=0, column=1, padx=4, pady=2, sticky="ew")
        self.btn_git_commit_push = ttk.Button(
            buttons, text="Commit + Push", command=self.git_commit_and_push
        )
        self.btn_git_commit_push.grid(row=0, column=2, padx=4, pady=2, sticky="ew")
        self.btn_git_push = ttk.Button(buttons, text="Push Only", command=self.git_push_only)
        self.btn_git_push.grid(row=0, column=3, padx=4, pady=2, sticky="ew")
        for col in range(4):
            buttons.columnconfigure(col, weight=1)

    def _on_git_selection_change(self) -> None:
        self._git_selected_paths = [
            self._git_tree_paths[iid]
            for iid in self.git_tree.selection()
            if iid in self._git_tree_paths
        ]
        self._set_buttons()

    def _on_commit_message_change(self) -> None:
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
        self._stop_report_watcher()
        if self.controller:
            self.controller.close()
        self._supervisor_ui_key = None
        try:
            self.controller = TandemController(Path(folder))
            self.project_var.set(str(Path(folder).resolve()))
            self.footer_var.set("Project opened. Mini Worker remains untouched.")
            self.refresh()
            self._start_report_watcher()
            self._git_service = GitService(self.controller.db.project_root)
            self._git_snapshot = {}
            self.refresh_git()
        except Exception as exc:
            self.controller = None
            self._git_service = None
            self._git_snapshot = {}
            self._render_git_snapshot({})
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

    def check_providers(self) -> None:
        """Probe every provider in the background (never blocks Tk).

        The probe is cheap (a health endpoint / a CLI version probe) and never
        runs a completion. Results are cached in memory for display only.
        """
        if not self.controller:
            return

        def job() -> dict:
            return self.controller.provider_status()

        def on_success(result: dict) -> None:
            self._provider_status_cache = dict(result or {})
            self._render_provider_status()
            self.footer_var.set(
                "Provider check complete (no completion was run, no key was read)."
            )

        self._run_background("Checking supervisor providers...", job, on_success)

    def _render_provider_status(self) -> None:
        """Render the cached provider status (no probe, no secret)."""
        for provider_id, var in self.provider_status_vars.items():
            entry = self._provider_status_cache.get(provider_id)
            if not entry:
                var.set(f"{supervisor_label(provider_id)}: not checked")
                continue
            var.set(f"{supervisor_label(provider_id)}: {entry.get('status', '—')}")

        for env_name, var in self.provider_key_vars.items():
            provider_id = SUPERVISOR_ENV_PROVIDER.get(env_name)
            entry = self._provider_status_cache.get(provider_id) or {}
            var.set(f"{env_name}: {entry.get('key', '—')}")

        if not self._provider_status_cache:
            return
        lines = ["=== SUPERVISOR PROVIDERS ==="]
        for provider_id in SUPERVISOR_MANUAL_CHOICES:
            entry = self._provider_status_cache.get(provider_id) or {}
            detail = f" | {entry['detail']}" if entry.get("detail") else ""
            lines.append(
                f"{supervisor_label(provider_id)}: {entry.get('status', '—')}"
                f" | model {entry.get('model') or '(legacy Codex config)'}"
                f" | {entry.get('key_env') or 'no key'}: {entry.get('key', '—')}"
                f" | registered {bool(entry.get('registered'))}{detail}"
            )
        self._set_details("\n".join(lines))

    def _policy_from_ui(self) -> SupervisorPolicy:
        """Build the policy the combos describe (never a secret)."""
        base = (
            self.controller.supervisor_policy
            if self.controller
            else default_supervisor_policy()
        )
        data = base.to_dict()
        mode = str(self.supervisor_mode_var.get()).strip().upper()
        data["mode"] = mode if mode in SUPERVISOR_MODE_CHOICES else POLICY_MODE_MANUAL
        budget = str(self.supervisor_budget_var.get()).strip().upper()
        data["budget_mode"] = (
            budget if budget in SUPERVISOR_BUDGET_CHOICES else BUDGET_STANDARD
        )

        if data["mode"] == POLICY_MODE_AUTO:
            # AUTO is only useful with the M8 §2 route + fallback model, so a
            # project that never configured one is seeded from the default AUTO
            # policy instead of silently inheriting an empty chain.
            auto = default_auto_policy()
            if not data["fallback_chain"]:
                data["fallback_chain"] = list(auto.fallback_chain)
                data["max_provider_retries"] = auto.max_provider_retries
                data["allow_codex_legacy"] = True
                data["fallback_on_contract_failure"] = True
        else:
            plan_id = supervisor_provider_from_label(self.supervisor_plan_var.get())
            review_id = supervisor_provider_from_label(
                self.supervisor_review_var.get()
            )
            if plan_id not in SUPERVISOR_MANUAL_CHOICES:
                plan_id = base.provider_for(PLAN_OPERATION) or PROVIDER_CODEX_LEGACY
                if plan_id not in SUPERVISOR_MANUAL_CHOICES:
                    plan_id = PROVIDER_CODEX_LEGACY
            if review_id not in SUPERVISOR_MANUAL_CHOICES:
                review_id = base.provider_for(REVIEW_OPERATION) or plan_id
                if review_id not in SUPERVISOR_MANUAL_CHOICES:
                    review_id = plan_id
            data["plan_provider"] = plan_id
            data["review_provider"] = review_id
        return SupervisorPolicy.from_dict(data)

    def _sync_supervisor_controls(self, policy: SupervisorPolicy) -> None:
        """Reflect a saved policy in the combos.

        AUTO is expressed by the MODE selector: the provider combos are then
        read-only and show AUTO because the risk routes decide per operation. A
        MANUAL policy always names concrete, selectable providers.
        """
        mode = str(policy.mode).strip().upper()
        self.supervisor_mode_var.set(
            mode if mode in SUPERVISOR_MODE_CHOICES else POLICY_MODE_MANUAL
        )
        concrete = [supervisor_label(p) for p in SUPERVISOR_MANUAL_CHOICES]
        if policy.is_auto:
            self.supervisor_plan_combo.configure(values=[supervisor_label("AUTO")])
            self.supervisor_review_combo.configure(values=[supervisor_label("AUTO")])
            self.supervisor_plan_var.set(supervisor_label("AUTO"))
            self.supervisor_review_var.set(supervisor_label("AUTO"))
            self.supervisor_plan_combo.configure(state="disabled")
            self.supervisor_review_combo.configure(state="disabled")
        else:
            self.supervisor_plan_combo.configure(values=concrete, state="readonly")
            self.supervisor_review_combo.configure(values=concrete, state="readonly")
            self.supervisor_plan_var.set(
                supervisor_label(policy.provider_for(PLAN_OPERATION))
            )
            self.supervisor_review_var.set(
                supervisor_label(policy.provider_for(REVIEW_OPERATION))
            )
        budget = str(policy.budget_mode).strip().upper()
        self.supervisor_budget_var.set(
            budget if budget in SUPERVISOR_BUDGET_CHOICES else BUDGET_STANDARD
        )
        models = (
            self.controller.supervisor_models()
            if self.controller
            else supervisor_model_defaults()
        )
        for provider_id, var in self.supervisor_model_vars.items():
            var.set(models.get(provider_id, ""))

    def _refresh_supervisor_section(self) -> None:
        """Cheap, read-only refresh: routes from the policy, statuses from cache.

        Combo values are synchronised once per policy change so an operator's
        unsaved selection is never clobbered by a background refresh (and no
        health endpoint is probed here).
        """
        if not self.controller:
            self.supervisor_plan_route_var.set("Current PLAN route: —")
            self.supervisor_review_route_var.set("Current REVIEW route: —")
            self._provider_status_cache = {}
            self._supervisor_ui_key = None
            self._render_provider_status()
            return

        key = (
            self.controller.supervisor_policy.to_dict(),
            dict(self.controller.supervisor_models()),
            tuple(self.controller.supervisor_enabled_providers()),
        )
        if key != self._supervisor_ui_key:
            self._sync_supervisor_controls(self.controller.supervisor_policy)
            self._supervisor_ui_key = key

        routes = self.controller.current_routes()
        self.supervisor_plan_route_var.set(
            f"Current PLAN route: {routes['plan']['label']}"
        )
        self.supervisor_review_route_var.set(
            f"Current REVIEW route: {routes['review']['label']}"
        )
        self._render_provider_status()

    def save_policy(self) -> None:
        """Persist the operator policy (atomic write, no secrets) and apply it."""
        if not self.controller:
            messagebox.showerror(APP_NAME, "Open a project first.")
            return
        try:
            policy = self._policy_from_ui()
            models = {
                provider_id: var.get().strip()
                for provider_id, var in self.supervisor_model_vars.items()
            }
            payload = self.controller.save_supervisor_policy(policy, models=models)
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        self._supervisor_ui_key = None
        self.refresh()
        self.footer_var.set(
            "Supervisor policy saved to "
            f"{self.controller.supervisor_policy_path().name} ({payload['mode']})."
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
            self._start_report_watcher()
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

    def _on_review_success(self, result) -> None:
        """Render a completed review, or a deferred (retryable) review.

        A deferred review is NOT a project failure, so it must never look like
        one: the user gets an explicit reason and the retry is offered.
        """
        if isinstance(result, dict) and result.get("deferred"):
            reason = review_deferred_message(result)
            self.footer_var.set(f"Codex REVIEW deferred: {reason}")
            messagebox.showinfo(
                APP_NAME,
                "Codex REVIEW deferred:\n"
                f"{reason}\n\n"
                "The accepted Cline report is preserved and the attempt was NOT "
                "consumed. Retry is available from 'Retry Supervisor REVIEW' once "
                "the provider is available again.",
            )
            return
        self._set_details(json.dumps(result, ensure_ascii=False, indent=2))

    def run_codex_review(self) -> None:
        if not self.controller:
            return
        self._run_background(
            "Codex is reviewing Cline implementation...",
            self.controller.run_codex_review,
            self._on_review_success,
        )

    def _review_recovery_qualified(self) -> bool:
        """Read-only check for the strict one-time FAILED review recovery."""
        if not self.controller:
            return False
        try:
            return self.controller.review_recovery_qualified()
        except Exception:
            return False

    def retry_codex_review(self) -> None:
        if not self.controller:
            return
        step = self.controller.db.current_step()
        if not step:
            return
        if step.state == "FAILED":
            # Legacy case: a review was lost to an external provider failure.
            # Requires the explicit, audited, one-time recovery.
            if not self._review_recovery_qualified():
                messagebox.showerror(
                    APP_NAME,
                    "Codex REVIEW recovery is not available for this step.\n\n"
                    "It requires: state FAILED, an accepted Cline report for the "
                    "current attempt, no Codex review for the current attempt and "
                    "no earlier recovery.",
                )
                return
            if not messagebox.askyesno(
                APP_NAME,
                f"One-time audited recovery for STEP {step.step_no}?\n\n"
                "This restores FAILED -> CLINE_REPORT_RECEIVED so Codex REVIEW "
                "can be retried, then runs the review.\n\n"
                "It does NOT change the attempt count, does NOT re-run Cline and "
                "does NOT re-run PLAN. The accepted Cline report is preserved.",
            ):
                return
            try:
                recovered = self.controller.recover_failed_review_to_received(
                    actor="human",
                    reason=(
                        "Human-authorized recovery to retry Codex REVIEW after an "
                        "external provider failure."
                    ),
                )
            except Exception as exc:
                messagebox.showerror(APP_NAME, str(exc))
                return
            self.footer_var.set(
                f"STEP {recovered['step_no']} recovered to {recovered['state']} "
                f"(attempt {recovered['attempt']} unchanged)."
            )
            self.refresh()
        self._run_background(
            "Codex is re-running REVIEW...",
            self.controller.retry_codex_review,
            self._on_review_success,
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

    # ------------------------------------------------------------------ #
    # Git module (project utility; independent of FSM/Codex/watcher)
    # ------------------------------------------------------------------ #

    def _git_require(self) -> Optional[GitService]:
        if not self.controller:
            return None
        if self._git_service is None:
            self._git_service = GitService(self.controller.db.project_root)
        return self._git_service

    def _git_audit(self, event_type: str, **fields) -> None:
        """Best-effort audit via the existing events table (no schema change)."""
        if not self.controller:
            return
        payload = {"timestamp": now_iso(), "project": str(self.controller.db.project_root)}
        payload.update(fields)
        try:
            self.controller.db.event(
                None, event_type, json.dumps(payload, ensure_ascii=False)
            )
        except Exception:
            pass  # audit is best-effort; never block a Git action

    def refresh_git(self) -> None:
        """Refresh the Git panel asynchronously (never freezes Tk)."""
        if not self.controller:
            self._git_snapshot = {}
            self._render_git_snapshot({})
            return
        service = self._git_require()

        def job() -> dict:
            return service.snapshot()

        def on_success(snapshot: dict) -> None:
            self._git_snapshot = snapshot
            self._render_git_snapshot(snapshot)

        self._run_background("Refreshing Git status...", job, on_success)

    def _render_git_snapshot(self, snapshot: dict) -> None:
        snap = snapshot or {}
        is_repo = bool(snap.get("is_repo")) and self.controller is not None
        if not self.controller:
            self.git_state_var.set("Git: —")
            self.git_repo_var.set("Repository: —")
        elif not snap.get("git_available"):
            self.git_state_var.set("Git: GIT MISSING")
            self.git_repo_var.set("Repository: —")
        elif not is_repo:
            self.git_state_var.set("Git: NOT A GIT REPOSITORY")
            self.git_repo_var.set(f"Repository: {self.controller.db.project_root}")
        else:
            self.git_state_var.set("Git: READY")
            self.git_repo_var.set(f"Repository: {snap.get('repo_root') or '—'}")

        if is_repo:
            if snap.get("detached"):
                self.git_branch_var.set("Branch: (detached HEAD)")
            else:
                self.git_branch_var.set(f"Branch: {snap.get('branch') or '—'}")
            if snap.get("remote_present"):
                self.git_remote_var.set(
                    f"Remote: {snap.get('remote_name')} ({snap.get('remote_url')})"
                )
            else:
                self.git_remote_var.set("Remote: none")
            self.git_changed_var.set(f"Changed: {snap.get('changed', 0)}")
            self.git_staged_var.set(f"Staged: {snap.get('staged', 0)}")
            self._render_ahead_behind(snap)
            sha = snap.get("last_commit_hash") or ""
            subject = snap.get("last_commit_summary") or ""
            self.git_last_commit_var.set(
                f"Last commit: {sha[:8]} {subject}".rstrip() if sha else "Last commit: —"
            )
        else:
            self.git_branch_var.set("Branch: —")
            self.git_remote_var.set("Remote: —")
            self.git_changed_var.set("Changed: 0")
            self.git_staged_var.set("Staged: 0")
            self.git_ahead_var.set("Ahead: —")
            self.git_behind_var.set("Behind: —")
            self.git_last_commit_var.set("Last commit: —")

        self._render_git_files(snap.get("entries") or [])
        self._set_buttons()

    def _render_ahead_behind(self, snap: dict) -> None:
        if snap.get("has_upstream"):
            self.git_ahead_var.set(f"Ahead: {snap.get('ahead', 0)}")
            self.git_behind_var.set(f"Behind: {snap.get('behind', 0)}")
        elif snap.get("remote_present"):
            # Upstream is not configured: never pretend ahead == 0.
            self.git_ahead_var.set("Ahead: LOCAL / NO UPSTREAM")
            self.git_behind_var.set("Behind: —")
        else:
            self.git_ahead_var.set("Ahead: —")
            self.git_behind_var.set("Behind: —")

    def _render_git_files(self, entries) -> None:
        self.git_tree.delete(*self.git_tree.get_children())
        self._git_tree_paths = {}
        self._git_selected_paths = []
        for entry in entries:
            note = "RUNTIME / SENSITIVE" if is_runtime_or_sensitive(entry.path) else ""
            label = f"{entry.orig_path} -> {entry.path}" if entry.orig_path else entry.path
            iid = self.git_tree.insert("", "end", values=(entry.code, label, note))
            self._git_tree_paths[iid] = entry.path

    def git_status(self) -> None:
        if not self.controller:
            return
        self._git_audit(
            GIT_EVENT_STATUS,
            action="STATUS",
            branch=(self._git_snapshot or {}).get("branch", ""),
            success=True,
        )
        self.refresh_git()

    def _git_selected_files(self) -> list[str]:
        return list(self._git_selected_paths)

    def git_commit_local(self) -> None:
        self._git_commit(push=False)

    def git_commit_and_push(self) -> None:
        self._git_commit(push=True)

    def _git_commit(self, *, push: bool) -> None:
        service = self._git_require()
        if service is None:
            return
        snap = self._git_snapshot or {}
        if not snap.get("is_repo") or not snap.get("branch") or snap.get("detached"):
            messagebox.showerror(APP_NAME, "Commit requires a valid Git repo and branch.")
            return
        files = self._git_selected_files()
        if not files:
            messagebox.showerror(APP_NAME, "Select at least one changed file to stage.")
            return
        ok, cleaned = validate_commit_message(self.commit_message_var.get())
        if not ok:
            messagebox.showerror(APP_NAME, cleaned)
            return
        message = cleaned
        branch = snap.get("branch") or ""
        action = "Commit + Push" if push else "Commit Local"
        preview = "\n".join(f"  - {f}" for f in files[:30])
        if len(files) > 30:
            preview += f"\n  ... and {len(files) - 30} more"
        if not messagebox.askyesno(
            APP_NAME,
            f"{action}?\n\n"
            f"Repository: {snap.get('repo_root') or '—'}\n"
            f"Branch: {branch}\n"
            f"Message: {message}\n\n"
            f"Selected files ({len(files)}):\n{preview}",
        ):
            return

        def job() -> dict:
            try:
                service.stage_files(files)
                staged = {e.path for e in service.get_status() if e.staged}
                missing = [f for f in files if f not in staged]
                if missing:
                    return {
                        "ok": False,
                        "error_kind": "FILE_VANISHED",
                        "error": "Selected file(s) are no longer staged/changed:\n"
                        + "\n".join(f"  - {f}" for f in missing),
                    }
                commit_hash = service.commit(message)
            except GitError as exc:
                return {"ok": False, "error_kind": exc.kind, "error": str(exc)}
            result = {"ok": True, "commit_hash": commit_hash, "pushed": False, "push_error": ""}
            if push:
                try:
                    service.push(branch, set_upstream=False)
                    result["pushed"] = True
                except GitError as exc:
                    result["push_error"] = str(exc)
            return result

        def on_success(result: dict) -> None:
            if not result.get("ok"):
                self._git_audit(
                    GIT_EVENT_COMMIT,
                    action=action,
                    branch=branch,
                    files=files,
                    success=False,
                    error=str(result.get("error", ""))[:2000],
                )
                messagebox.showerror(APP_NAME, f"{action} failed.\n\n{result.get('error', '')}")
                self.refresh_git()
                return
            commit_hash = result["commit_hash"]
            self._git_audit(
                GIT_EVENT_COMMIT,
                action=action,
                branch=branch,
                files=files,
                commit_hash=commit_hash,
                success=True,
            )
            if push and result.get("pushed"):
                self._git_audit(
                    GIT_EVENT_PUSH,
                    action="PUSH",
                    branch=branch,
                    commit_hash=commit_hash,
                    success=True,
                )
                messagebox.showinfo(
                    APP_NAME,
                    f"Commit created: {commit_hash[:12]}\n"
                    f"Pushed {branch} to {service.remote}.",
                )
                self.refresh_git()
                return
            if push:
                # Push failed: the local commit is intentionally preserved.
                self._git_audit(
                    GIT_EVENT_PUSH,
                    action="PUSH",
                    branch=branch,
                    commit_hash=commit_hash,
                    success=False,
                    error=str(result.get("push_error", ""))[:2000],
                )
                messagebox.showwarning(
                    APP_NAME,
                    "Local commit created successfully.\n"
                    "GitHub push failed.\n\n"
                    f"Commit: {commit_hash}\n\n"
                    f"Reason:\n{result.get('push_error', '')}",
                )
                self.refresh_git()
                return
            messagebox.showinfo(APP_NAME, f"Commit created: {commit_hash[:12]}")
            self.refresh_git()

        self._run_background(f"{action}...", job, on_success)

    def git_push_only(self) -> None:
        service = self._git_require()
        if service is None:
            return
        snap = self._git_snapshot or {}
        if not snap.get("is_repo") or not snap.get("branch") or snap.get("detached"):
            messagebox.showerror(APP_NAME, "Push requires a valid Git repo and branch.")
            return
        if not snap.get("remote_present"):
            messagebox.showerror(APP_NAME, f"No '{service.remote}' remote found.")
            return
        branch = snap.get("branch") or ""
        has_upstream = bool(snap.get("has_upstream"))
        set_upstream = False
        if has_upstream:
            if int(snap.get("ahead") or 0) <= 0:
                messagebox.showinfo(APP_NAME, "Nothing to push (no local commits ahead).")
                return
            ahead_line = f"Ahead: {snap.get('ahead')}"
        else:
            # No upstream configured: push is possible, but creating the
            # upstream must be confirmed explicitly (never silently).
            if not snap.get("can_push"):
                messagebox.showinfo(APP_NAME, "Nothing to push.")
                return
            ahead_line = "Ahead: LOCAL / NO UPSTREAM"
            set_upstream = True

        prompt = (
            f"Repository: {snap.get('repo_root') or '—'}\n"
            f"Branch: {branch}\n"
            f"Remote: {snap.get('remote_name')} ({snap.get('remote_url')})\n"
            f"{ahead_line}\n"
        )
        if set_upstream:
            prompt = (
                "No upstream branch exists.\n"
                "Create upstream with:\n\n"
                f"git push -u {service.remote} {branch}\n\n" + prompt
            )
        if not messagebox.askyesno(APP_NAME, prompt + "\nProceed?"):
            return
        create_upstream = set_upstream

        def job() -> dict:
            try:
                service.push(branch, set_upstream=create_upstream)
                return {"ok": True}
            except GitError as exc:
                return {"ok": False, "error": str(exc), "error_kind": exc.kind}

        def on_success(result: dict) -> None:
            if not result.get("ok"):
                self._git_audit(
                    GIT_EVENT_PUSH,
                    action="PUSH",
                    branch=branch,
                    success=False,
                    error=str(result.get("error", ""))[:2000],
                )
                messagebox.showerror(
                    APP_NAME,
                    "Push failed (local commit preserved).\n\n" + str(result.get("error", "")),
                )
                self.refresh_git()
                return
            self._git_audit(GIT_EVENT_PUSH, action="PUSH", branch=branch, success=True)
            messagebox.showinfo(APP_NAME, f"Pushed {branch} to {service.remote}.")
            self.refresh_git()

        self._run_background(f"Pushing {branch} to {service.remote}...", job, on_success)

    def refresh(self) -> None:
        if not self.controller:
            self.supervisor_var.set("Supervisor: —")
            self._refresh_supervisor_section()
            self._refresh_cline_report_status()
            self._stop_report_watcher()
            self._set_buttons()
            return

        self.supervisor_var.set(f"Supervisor: {self.controller.supervisor_label()}")

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

        self._refresh_supervisor_section()
        self._refresh_cline_report_status()
        self._ensure_report_watcher()
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
        # Processing is only possible once the watcher has validated the exact
        # expected report for the current step (READY).
        set_state(
            self.btn_report,
            state == "CLINE_DISPATCHED" and self._cline_report_ready,
        )
        set_state(self.btn_review, state == "CLINE_REPORT_RECEIVED")
        set_state(self.btn_accept, state == "REVIEW_APPROVED")
        set_state(self.btn_repair, state == "REVISE")
        # Retry Codex REVIEW: enabled for a retryable external provider outage,
        # or for the strict one-time FAILED review-recovery case.
        set_state(
            self.btn_retry_review,
            state == CODEX_REVIEW_RETRYABLE
            or (state == "FAILED" and self._review_recovery_qualified()),
        )

        # Canonical-plan sync is a manual maintenance action: allowed whenever a
        # project is open and no background job is running.
        self.btn_sync.configure(
            state=("normal" if self.controller and not self.busy else "disabled")
        )

        # Supervisor policy controls (M9): both are project-scoped operator
        # actions, allowed whenever a project is open and nothing is running.
        self.btn_check_providers.configure(
            state=("normal" if self.controller and not self.busy else "disabled")
        )
        self.btn_save_policy.configure(
            state=("normal" if self.controller and not self.busy else "disabled")
        )

        # Git section: project utility. Commit requires a valid repo + branch,
        # an explicit file selection and a valid message; push additionally
        # requires the origin remote.
        snap = self._git_snapshot or {}
        git_repo = bool(snap.get("is_repo"))
        git_branch = bool(snap.get("branch")) and not snap.get("detached")
        git_origin = bool(snap.get("remote_present"))
        msg_ok, _ = validate_commit_message(self.commit_message_var.get())
        has_selection = bool(self._git_selected_paths)
        set_state(self.btn_git_status, True)
        set_state(
            self.btn_git_commit,
            git_repo and git_branch and has_selection and msg_ok,
        )
        set_state(
            self.btn_git_commit_push,
            git_repo and git_branch and git_origin and has_selection and msg_ok,
        )
        set_state(
            self.btn_git_push,
            git_repo and git_branch and git_origin and bool(snap.get("can_push")),
        )

    # ------------------------------------------------------------------ #
    # Cline report readiness watcher (V1: detection only, no auto-process)
    # ------------------------------------------------------------------ #

    def _start_report_watcher(self) -> None:
        self._cancel_report_watcher()
        if not self.controller:
            return
        self._refresh_cline_report_status()
        self._ensure_report_watcher()

    def _stop_report_watcher(self) -> None:
        self._cancel_report_watcher()
        self._watcher_target = None

    def _cancel_report_watcher(self) -> None:
        if self._watcher_after_id is not None:
            try:
                self.root.after_cancel(self._watcher_after_id)
            except Exception:
                pass
            self._watcher_after_id = None

    def _ensure_report_watcher(self) -> None:
        """Arm the poller iff the current step is CLINE_DISPATCHED, else stop."""
        if not self.controller:
            self._stop_report_watcher()
            return
        step = self.controller.db.current_step()
        if not step or step.state != "CLINE_DISPATCHED":
            self._stop_report_watcher()
            return
        if self._watcher_after_id is None:
            self._schedule_report_watcher()

    def _schedule_report_watcher(self) -> None:
        if self._watcher_after_id is not None:
            return
        self._watcher_after_id = self.root.after(
            CLINE_REPORT_POLL_MS, self._poll_cline_report
        )

    def _poll_cline_report(self) -> None:
        """One lightweight polling cycle; never blocks the Tk thread."""
        self._watcher_after_id = None
        try:
            self._refresh_cline_report_status()
        finally:
            # Re-arm only while the same step is still CLINE_DISPATCHED, so the
            # watcher retargets/stops automatically when the step changes.
            step = self.controller.db.current_step() if self.controller else None
            if step and step.state == "CLINE_DISPATCHED":
                self._schedule_report_watcher()

    @staticmethod
    def _report_signature(path: Path) -> Optional[tuple[int, int]]:
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _refresh_cline_report_status(self) -> None:
        """Recompute the report readiness and refresh the GUI status line."""
        if not self.controller:
            self._cline_report_cache = None
            self._apply_readiness(
                {"status": "N/A", "detail": "", "step_no": None, "size": None, "mtime": None}
            )
            return

        step = self.controller.db.current_step()
        if not step or step.state != "CLINE_DISPATCHED":
            self._cline_report_cache = None
            self._watcher_target = None
            self._apply_readiness(
                {"status": "N/A", "detail": "", "step_no": None, "size": None, "mtime": None}
            )
            return

        target = (step.step_no, step.state)
        if self._watcher_target != target:
            # Retargeted to a new step/state: drop any stale signature.
            self._watcher_target = target
            self._cline_report_cache = None

        path = self.controller.expected_cline_report_path(step.step_no)
        signature = self._report_signature(path)
        cache = self._cline_report_cache
        if (
            signature is not None
            and cache is not None
            and cache.get("step_no") == step.step_no
            and cache.get("signature") == signature
            and cache.get("status") in {"READY", "INVALID"}
        ):
            # The exact final file has not changed: reuse the cached result
            # instead of re-reading/re-parsing/re-validating it every cycle.
            self._apply_readiness(cache)
            return

        readiness = dict(self.controller.check_cline_report_readiness())
        readiness["step_no"] = step.step_no
        readiness["signature"] = signature
        if signature is None or readiness.get("status") == "WAITING":
            # Nothing is cached while the final report is absent or unstable.
            self._cline_report_cache = None
        else:
            self._cline_report_cache = readiness
        self._apply_readiness(readiness)

    def _apply_readiness(self, readiness: dict) -> None:
        status = str(readiness.get("status", "N/A"))
        detail = str(readiness.get("detail", ""))
        self._cline_report_latest = readiness
        changed = (status, detail) != (self._cline_report_status, self._cline_report_detail)
        self._cline_report_status = status
        self._cline_report_detail = detail
        self._cline_report_ready = status == "READY"
        self._update_cline_report_label()
        if changed:
            self._set_buttons()

    def _update_cline_report_label(self) -> None:
        status = self._cline_report_status
        detail = self._cline_report_detail
        if status == "READY":
            suffix = ""
            size = self._cline_report_latest.get("size")
            mtime = self._cline_report_latest.get("mtime")
            if isinstance(size, int):
                suffix = f" ({size} B"
                if isinstance(mtime, (int, float)) and mtime:
                    suffix += f", {datetime.fromtimestamp(mtime).strftime('%H:%M:%S')}"
                suffix += ")"
            self.cline_report_var.set(f"Cline report: READY — {detail}{suffix}")
            self._watcher_set_footer(WATCHER_FOOTER_READY)
        elif status == "INVALID":
            self.cline_report_var.set(f"Cline report: INVALID — {detail}")
            self._watcher_set_footer(WATCHER_FOOTER_INVALID)
        elif status == "WAITING":
            self.cline_report_var.set("Cline report: WAITING")
            self._watcher_set_footer(WATCHER_FOOTER_WAITING)
        else:
            self.cline_report_var.set("Cline report: —")

    def _watcher_set_footer(self, text: str) -> None:
        """Set the footer only while the watcher owns it (never clobbers ops)."""
        if self.busy:
            return
        current = self.footer_var.get()
        if current and current not in WATCHER_FOOTER_MESSAGES:
            return
        self.footer_var.set(text)

    def on_close(self) -> None:
        try:
            self._stop_report_watcher()
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
