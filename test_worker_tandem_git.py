"""Worker_tandem Git module tests.

Pins the V1 Git contract:
    - GitService is a pure, project-scoped adapter (no FSM coupling);
    - ``git status --porcelain=v1 -z`` is parsed as NUL-delimited records
      (spaces, tabs, newlines, Unicode and ``-`` names preserved);
    - only explicitly selected files are staged (``git add -- <files>``);
    - no force push, no ``add .`` / ``add -A``, no pull/rebase/merge/reset;
    - push without an upstream requires explicit ``-u`` confirmation.

The driver filename is not a valid module name, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

SRC = pathlib.Path(__file__).resolve().parent / "worker_tandem_v0.1.2.py"


def _load_worker_tandem():
    spec = importlib.util.spec_from_file_location("worker_tandem_git", SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["worker_tandem_git"] = module
    spec.loader.exec_module(module)
    return module


wt = _load_worker_tandem()

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git CLI not available")

PLAN_DEF = {
    "plan_version": "git-test-1",
    "steps": [
        {"step_no": 1, "phase": "FOUNDATION", "title": "ONE", "description": "d1", "risk": "LOW"},
    ],
}


def run_git(repo, *args):
    return subprocess.run(
        [GIT, *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )


def commit_all(repo, message="init"):
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", message)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    run_git(root, "init", "-b", "main")
    run_git(root, "config", "user.email", "test@example.com")
    run_git(root, "config", "user.name", "Test User")
    run_git(root, "config", "commit.gpgsign", "false")
    run_git(root, "config", "core.autocrlf", "false")
    return root


@pytest.fixture
def service(repo):
    return wt.GitService(repo)


# --------------------------------------------------------------------------- #
# Porcelain v1 -z parsing (no git needed)
# --------------------------------------------------------------------------- #


def test_parse_porcelain_z_preserves_spaces_unicode_and_dash():
    raw = (
        " M file with spaces.txt\x00"
        "?? café_ünïcode.txt\x00"
        "?? -leading-dash.txt\x00"
    )
    entries = wt.parse_porcelain_v1_z(raw)
    paths = [e.path for e in entries]
    assert "file with spaces.txt" in paths
    assert "café_ünïcode.txt" in paths
    assert "-leading-dash.txt" in paths
    assert entries[0].unstaged is True
    assert entries[1].untracked is True


def test_parse_porcelain_z_handles_tabs_and_newlines_in_names():
    raw = b"M  tab\tname.txt\x00?? line\nbreak.txt\x00"
    entries = wt.parse_porcelain_v1_z(raw)
    assert entries[0].path == "tab\tname.txt"
    assert entries[1].path == "line\nbreak.txt"


def test_parse_porcelain_z_rename_new_then_original():
    raw = b"R  new name.txt\x00old name.txt\x00"
    entries = wt.parse_porcelain_v1_z(raw)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.change_type == "renamed"
    assert entry.path == "new name.txt"
    assert entry.orig_path == "old name.txt"


def test_parse_porcelain_z_staged_vs_unstaged():
    raw = b"M  staged.txt\x00 M unstaged.txt\x00MM both.txt\x00"
    by_path = {e.path: e for e in wt.parse_porcelain_v1_z(raw)}
    assert by_path["staged.txt"].staged and not by_path["staged.txt"].unstaged
    assert by_path["unstaged.txt"].unstaged and not by_path["unstaged.txt"].staged
    assert by_path["both.txt"].staged and by_path["both.txt"].unstaged


def test_parse_porcelain_z_empty_and_trailing_nul():
    assert wt.parse_porcelain_v1_z("") == []
    assert wt.parse_porcelain_v1_z(b"\x00") == []


def test_validate_commit_message_rules():
    assert wt.validate_commit_message("  ok  ") == (True, "ok")
    assert wt.validate_commit_message("")[0] is False
    assert wt.validate_commit_message("   ")[0] is False
    assert wt.validate_commit_message("a\nb")[0] is False
    assert wt.validate_commit_message("x" * 121)[0] is False
    ok, text = wt.validate_commit_message("x" * 120)
    assert ok and text == "x" * 120


def test_runtime_or_sensitive_detection():
    assert wt.is_runtime_or_sensitive(".worker_tandem/tandem_state.db")
    assert wt.is_runtime_or_sensitive("app.db")
    assert wt.is_runtime_or_sensitive("app.db-wal")
    assert wt.is_runtime_or_sensitive("logs/run.log")
    assert wt.is_runtime_or_sensitive(".env")
    assert wt.is_runtime_or_sensitive("a/.env.local")
    assert wt.is_runtime_or_sensitive("sub/secrets/token.txt")
    assert not wt.is_runtime_or_sensitive("src/app.py")
    assert not wt.is_runtime_or_sensitive("worker_tandem_v0.1.2.py")


def test_sanitize_remote_url_redacts_credentials():
    assert (
        wt.sanitize_remote_url("https://user:tok@github.com/a/b.git")
        == "https://***@github.com/a/b.git"
    )
    assert (
        wt.sanitize_remote_url("https://github.com/a/b.git")
        == "https://github.com/a/b.git"
    )
    assert wt.sanitize_remote_url("git@github.com:a/b.git") == "***@github.com:a/b.git"
    assert wt.sanitize_remote_url("") == ""


def test_classify_git_error_kinds():
    assert wt.classify_git_error(1, "fatal: not a git repository") == "NOT_A_REPO"
    assert (
        wt.classify_git_error(1, "fatal: detected dubious ownership in repository")
        == "DUBIOUS_OWNERSHIP"
    )
    assert (
        wt.classify_git_error(1, "fatal: Authentication failed for 'https://x'")
        == "AUTH_FAILED"
    )
    assert wt.classify_git_error(1, "Could not resolve host: github.com") == "NETWORK"
    assert (
        wt.classify_git_error(1, "! [rejected] main -> main (non-fast-forward)")
        == "NON_FAST_FORWARD"
    )
    assert (
        wt.classify_git_error(1, "husky - pre-commit hook exited with code 1")
        == "HOOK_FAILED"
    )
    assert (
        wt.classify_git_error(1, "nothing to commit, working tree clean")
        == "NOTHING_TO_COMMIT"
    )


# --------------------------------------------------------------------------- #
# Repository detection / branch / remote / status
# --------------------------------------------------------------------------- #


@needs_git
def test_repo_detected(service, repo):
    assert service.available() is True
    assert service.is_git_repo() is True
    assert service.get_repo_root() == repo.resolve()


@needs_git
def test_non_repo_folder(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    svc = wt.GitService(plain)
    assert svc.is_git_repo() is False
    assert svc.get_repo_root() is None
    snap = svc.snapshot()
    assert snap["is_repo"] is False
    assert snap["error"] == "not a git repository"


@needs_git
def test_branch_detected(service, repo):
    (repo / "a.txt").write_text("a", encoding="utf-8")
    commit_all(repo)
    snap = service.snapshot()
    assert snap["is_repo"] is True
    assert snap["branch"] == "main"
    assert snap["detached"] is False
    assert len(snap["last_commit_hash"]) == 40
    assert snap["last_commit_summary"] == "init"


@needs_git
def test_detached_head(service, repo):
    (repo / "a.txt").write_text("a", encoding="utf-8")
    commit_all(repo)
    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    assert head
    run_git(repo, "checkout", "--detach", head)
    snap = service.snapshot()
    assert snap["detached"] is True
    assert snap["branch"] == ""
    assert snap["can_commit"] is False


@needs_git
def test_origin_detected_and_sanitized(service, repo):
    commit_all(repo)
    run_git(repo, "remote", "add", "origin", "https://user:token@github.com/a/b.git")
    snap = service.snapshot()
    assert snap["remote_present"] is True
    assert snap["remote_name"] == "origin"
    assert snap["remote_url"] == "https://***@github.com/a/b.git"


@needs_git
def test_no_origin(service, repo):
    commit_all(repo)
    snap = service.snapshot()
    assert snap["remote_present"] is False
    assert snap["remote_name"] == ""
    assert snap["can_push"] is False


@needs_git
def test_changed_files_parsing(service, repo):
    (repo / "a.txt").write_text("a", encoding="utf-8")
    commit_all(repo)
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    (repo / "new.txt").write_text("n", encoding="utf-8")
    by_path = {e.path: e for e in service.get_status(repo)}
    assert by_path["a.txt"].change_type == "modified"
    assert by_path["new.txt"].change_type == "untracked"
    assert by_path["new.txt"].untracked is True


@needs_git
def test_staged_vs_unstaged(service, repo):
    (repo / "s.txt").write_text("1", encoding="utf-8")
    (repo / "u.txt").write_text("1", encoding="utf-8")
    commit_all(repo)
    (repo / "s.txt").write_text("2", encoding="utf-8")
    (repo / "u.txt").write_text("2", encoding="utf-8")
    service.stage_files(["s.txt"])
    by_path = {e.path: e for e in service.get_status(repo)}
    assert by_path["s.txt"].staged is True
    assert by_path["u.txt"].staged is False
    assert by_path["u.txt"].unstaged is True


@needs_git
def test_renamed_file(service, repo):
    (repo / "old name.txt").write_text("x", encoding="utf-8")
    commit_all(repo)
    (repo / "old name.txt").rename(repo / "new name.txt")
    run_git(repo, "add", "--", "old name.txt", "new name.txt")
    entry = {e.path: e for e in service.get_status(repo)}["new name.txt"]
    assert entry.change_type == "renamed"
    assert entry.orig_path == "old name.txt"


@needs_git
def test_deleted_file(service, repo):
    (repo / "gone.txt").write_text("x", encoding="utf-8")
    commit_all(repo)
    (repo / "gone.txt").unlink()
    entry = {e.path: e for e in service.get_status(repo)}["gone.txt"]
    assert entry.change_type == "deleted"


@needs_git
def test_unicode_and_space_filenames(service, repo):
    names = ["café ünïcode.txt", "emoji 🚀 file.txt", "plain space.txt"]
    for name in names:
        (repo / name).write_text("x", encoding="utf-8")
    paths = {e.path for e in service.get_status(repo)}
    for name in names:
        assert name in paths


@needs_git
def test_filename_starting_with_dash(service, repo):
    (repo / "-dash file.txt").write_text("x", encoding="utf-8")
    assert "-dash file.txt" in {e.path for e in service.get_status(repo)}
    service.stage_files(["-dash file.txt"])
    staged = run_git(repo, "diff", "--cached", "--name-only").stdout
    assert "-dash file.txt" in staged


@needs_git
def test_filename_with_tab(service, repo):
    name = "tab\tname.txt"
    try:
        (repo / name).write_text("x", encoding="utf-8")
    except OSError:
        pytest.skip("platform does not allow a tab in filenames")
    assert name in {e.path for e in service.get_status(repo)}


@needs_git
def test_repo_under_path_with_spaces(tmp_path):
    root = tmp_path / "OneDrive - Test Folder" / "My Project"
    root.mkdir(parents=True)
    run_git(root, "init", "-b", "main")
    run_git(root, "config", "user.email", "t@e.com")
    run_git(root, "config", "user.name", "T")
    (root / "a.txt").write_text("1", encoding="utf-8")
    svc = wt.GitService(root)
    assert svc.is_git_repo() is True
    svc.stage_files(["a.txt"])
    svc.commit("feat: spaced path")
    assert svc.snapshot()["is_repo"] is True
    assert svc.snapshot()["last_commit_summary"] == "feat: spaced path"


# --------------------------------------------------------------------------- #
# Safe staging: selected files only, "--", never add . / add -A
# --------------------------------------------------------------------------- #


@needs_git
def test_stage_only_selected_files(service, repo):
    for name in ("a.txt", "b.txt", "c.txt"):
        (repo / name).write_text("1", encoding="utf-8")
    commit_all(repo)
    for name in ("a.txt", "b.txt", "c.txt"):
        (repo / name).write_text("2", encoding="utf-8")
    service.stage_files(["a.txt", "c.txt"])
    staged = {line for line in run_git(repo, "diff", "--cached", "--name-only").stdout.split() if line}
    assert staged == {"a.txt", "c.txt"}


@needs_git
def test_stage_uses_double_dash_and_never_add_all(service, repo, monkeypatch):
    (repo / "a.txt").write_text("1", encoding="utf-8")
    commit_all(repo)
    (repo / "a.txt").write_text("2", encoding="utf-8")
    seen = []
    real_run = wt.subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            seen.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(wt.subprocess, "run", spy)
    service.stage_files(["a.txt"])

    add_cmds = [c for c in seen if len(c) > 1 and c[1] == "add"]
    assert add_cmds, "expected a git add invocation"
    for cmd in add_cmds:
        assert "." not in cmd[2:]
        assert "-A" not in cmd
        assert "--all" not in cmd
        assert "--" in cmd
        assert cmd[cmd.index("--") + 1 :] == ["a.txt"]


@needs_git
def test_stage_rejects_blanket_pathspecs(service):
    for bad in (".", "./", "-A", "--all", "*"):
        with pytest.raises(wt.GitError) as exc:
            service.stage_files([bad])
        assert exc.value.kind == "UNSAFE_PATHSPEC"


@needs_git
def test_stage_empty_selection_rejected(service):
    with pytest.raises(wt.GitError) as exc:
        service.stage_files([])
    assert exc.value.kind == "NO_FILES_SELECTED"


@needs_git
def test_stage_vanished_file_raises(service, repo):
    (repo / "a.txt").write_text("1", encoding="utf-8")
    commit_all(repo)
    with pytest.raises(wt.GitError):
        service.stage_files(["does_not_exist.txt"])


# --------------------------------------------------------------------------- #
# Commit
# --------------------------------------------------------------------------- #


@needs_git
def test_local_commit_success(service, repo):
    (repo / "a.txt").write_text("1", encoding="utf-8")
    commit_all(repo)
    (repo / "a.txt").write_text("2", encoding="utf-8")
    service.stage_files(["a.txt"])
    commit_hash = service.commit("feat: change a")
    assert len(commit_hash) == 40
    assert service.get_status(repo) == []
    assert service.get_last_commit(repo)[1] == "feat: change a"


@needs_git
def test_empty_message_rejected_by_commit(service, repo):
    with pytest.raises(wt.GitError) as exc:
        service.commit("   ")
    assert exc.value.kind == "INVALID_MESSAGE"


@needs_git
def test_multiline_message_rejected_by_commit(service, repo):
    with pytest.raises(wt.GitError) as exc:
        service.commit("line1\nline2")
    assert exc.value.kind == "INVALID_MESSAGE"


@needs_git
def test_nothing_to_commit_is_classified(service, repo):
    (repo / "a.txt").write_text("1", encoding="utf-8")
    commit_all(repo)
    with pytest.raises(wt.GitError) as exc:
        service.commit("feat: nothing")
    assert exc.value.kind == "NOTHING_TO_COMMIT"


@needs_git
def test_hook_failure_surfaces_stderr(service, repo):
    (repo / "a.txt").write_text("1", encoding="utf-8")
    commit_all(repo)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho blocked by hook >&2\nexit 1\n", encoding="utf-8")
    try:
        hook.chmod(0o755)
    except OSError:
        pass
    (repo / "a.txt").write_text("2", encoding="utf-8")
    service.stage_files(["a.txt"])
    with pytest.raises(wt.GitError) as exc:
        service.commit("feat: blocked by hook")
    assert exc.value.kind == "HOOK_FAILED"
    assert exc.value.stderr


# --------------------------------------------------------------------------- #
# Push (never force; upstream created only after explicit -u approval)
# --------------------------------------------------------------------------- #


@needs_git
def test_commit_and_push_success(service, repo, tmp_path):
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    (repo / "a.txt").write_text("1", encoding="utf-8")
    service.stage_files(["a.txt"])
    commit_hash = service.commit("feat: first")
    service.push("main", set_upstream=True)
    assert len(commit_hash) == 40
    assert "main" in run_git(bare, "branch", "--list").stdout
    snap = service.snapshot()
    assert snap["has_upstream"] is True
    assert snap["ahead"] == 0
    assert snap["behind"] == 0


@needs_git
def test_push_only_after_upstream_configured(service, repo, tmp_path):
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    (repo / "a.txt").write_text("1", encoding="utf-8")
    service.stage_files(["a.txt"])
    service.commit("feat: first")
    service.push("main", set_upstream=True)
    assert service.snapshot()["can_push"] is False

    (repo / "a.txt").write_text("2", encoding="utf-8")
    service.stage_files(["a.txt"])
    service.commit("feat: second")
    snap = service.snapshot()
    assert snap["has_upstream"] is True
    assert snap["ahead"] == 1 and snap["behind"] == 0
    assert snap["can_push"] is True
    service.push("main")
    assert service.snapshot()["ahead"] == 0


@needs_git
def test_no_upstream_is_local_no_upstream(service, repo, tmp_path):
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    (repo / "a.txt").write_text("1", encoding="utf-8")
    service.stage_files(["a.txt"])
    service.commit("feat: first")
    snap = service.snapshot()
    assert snap["remote_present"] is True
    assert snap["has_upstream"] is False
    assert snap["ahead"] is None      # never pretend ahead == 0
    assert snap["behind"] is None
    assert snap["can_push"] is True   # local commits can still be pushed (with -u)


@needs_git
def test_push_argv_never_forces_and_respects_upstream_flag(service, repo, tmp_path, monkeypatch):
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    (repo / "a.txt").write_text("1", encoding="utf-8")
    service.stage_files(["a.txt"])
    service.commit("feat: first")

    seen = []
    real_run = wt.subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            seen.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(wt.subprocess, "run", spy)
    service.push("main", set_upstream=True)

    push_cmds = [c for c in seen if len(c) > 1 and c[1] == "push"]
    assert push_cmds
    for cmd in push_cmds:
        assert not any(a in {"--force", "-f", "--force-with-lease"} for a in cmd)
        assert not any(a in {"pull", "rebase", "merge", "reset", "clean"} for a in cmd)
    assert "-u" in push_cmds[0]
    assert push_cmds[0][-2:] == ["origin", "main"]


@needs_git
def test_push_failure_keeps_local_commit(service, repo, tmp_path):
    run_git(repo, "remote", "add", "origin", str(tmp_path / "missing.git"))
    (repo / "a.txt").write_text("1", encoding="utf-8")
    service.stage_files(["a.txt"])
    commit_hash = service.commit("feat: first")
    with pytest.raises(wt.GitError):
        service.push("main")
    # The local commit must be preserved (never rolled back by Worker Tandem).
    assert service.get_last_commit(repo)[0] == commit_hash


@needs_git
def test_non_fast_forward_rejected_without_pull(service, repo, tmp_path, monkeypatch):
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    (repo / "a.txt").write_text("base", encoding="utf-8")
    service.stage_files(["a.txt"])
    service.commit("feat: base")
    service.push("main", set_upstream=True)

    other = tmp_path / "other"
    run_git(tmp_path, "clone", str(bare), str(other))
    run_git(other, "config", "user.email", "o@e.com")
    run_git(other, "config", "user.name", "O")
    (other / "a.txt").write_text("other", encoding="utf-8")
    run_git(other, "add", "-A")
    run_git(other, "commit", "-m", "other")
    assert run_git(other, "push", "origin", "main").returncode == 0

    (repo / "a.txt").write_text("mine", encoding="utf-8")
    service.stage_files(["a.txt"])
    service.commit("feat: mine")

    seen = []
    real_run = wt.subprocess.run

    def spy(cmd, *args, **kwargs):
        if isinstance(cmd, list):
            seen.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(wt.subprocess, "run", spy)
    with pytest.raises(wt.GitError) as exc:
        service.push("main")
    assert exc.value.kind == "NON_FAST_FORWARD"
    for cmd in seen:
        assert not any(a in {"pull", "rebase", "merge", "reset", "clean"} for a in cmd)
    assert service.get_last_commit(repo)[1] == "feat: mine"


# --------------------------------------------------------------------------- #
# Environment robustness + no FSM coupling
# --------------------------------------------------------------------------- #


def test_git_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(wt.shutil, "which", lambda name: None)
    svc = wt.GitService(tmp_path)
    assert svc.available() is False
    assert svc.is_git_repo() is False
    assert svc.get_repo_root() is None
    snap = svc.snapshot()
    assert snap["git_available"] is False
    assert snap["error"] == "git executable not found"


@needs_git
def test_timeout_is_classified(service, repo, monkeypatch):
    def boom(cmd, *args, **kwargs):
        raise wt.subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(wt.subprocess, "run", boom)
    with pytest.raises(wt.GitError) as exc:
        service.get_status(repo)
    assert exc.value.kind == "TIMEOUT"


@needs_git
def test_git_action_does_not_mutate_fsm(repo):
    controller = wt.TandemController(repo)
    controller.db.import_plan(PLAN_DEF)
    try:
        before = controller.db.current_step().state
        svc = wt.GitService(controller.db.project_root)
        (repo / "a.txt").write_text("1", encoding="utf-8")
        svc.stage_files(["a.txt"])
        svc.commit("feat: x")
        svc.snapshot()
        assert controller.db.current_step().state == before
        assert controller.db.get_step(1).attempt == 0
        assert controller.db.latest_codex_plan(1) is None
        assert controller.db.latest_cline_report(1) is None
        assert controller.db.latest_codex_review(1) is None
    finally:
        controller.db.close()


# --------------------------------------------------------------------------- #
# GUI: button policy, runtime marking, LOCAL / NO UPSTREAM, responsiveness
# --------------------------------------------------------------------------- #


@pytest.fixture
def gui(repo):
    try:
        app = wt.TandemGUI()
    except Exception as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk unavailable: {exc}")
    app.root.withdraw()
    controller = wt.TandemController(repo)
    controller.db.import_plan(PLAN_DEF)
    app.controller = controller
    app._git_service = wt.GitService(repo)
    yield app, controller, repo
    try:
        app.on_close()
    except Exception:
        pass
    controller.db.close()


@needs_git
def test_gui_nothing_selected_by_default(gui):
    app, controller, repo = gui
    (repo / "a.txt").write_text("1", encoding="utf-8")
    snap = app._git_service.snapshot()
    app._git_snapshot = snap
    app._render_git_snapshot(snap)
    assert app.git_tree.selection() == ()
    assert app._git_selected_paths == []
    assert str(app.btn_git_commit["state"]) == "disabled"
    assert str(app.btn_git_commit_push["state"]) == "disabled"


@needs_git
def test_gui_button_states_track_selection_and_message(gui):
    app, controller, repo = gui
    (repo / "a.txt").write_text("1", encoding="utf-8")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "init")
    (repo / "a.txt").write_text("2", encoding="utf-8")
    snap = app._git_service.snapshot()
    app._git_snapshot = snap
    app._render_git_snapshot(snap)

    assert str(app.btn_git_status["state"]) == "normal"
    assert str(app.btn_git_commit["state"]) == "disabled"
    assert str(app.btn_git_commit_push["state"]) == "disabled"
    assert str(app.btn_git_push["state"]) == "disabled"  # no origin

    children = app.git_tree.get_children()
    assert children
    app.git_tree.selection_set(children[0])
    app._on_git_selection_change()
    assert str(app.btn_git_commit["state"]) == "disabled"  # no message yet

    app.commit_message_var.set("feat: gui")
    assert str(app.btn_git_commit["state"]) == "normal"
    assert str(app.btn_git_commit_push["state"]) == "disabled"  # still no origin

    app.commit_message_var.set("")
    assert str(app.btn_git_commit["state"]) == "disabled"


@needs_git
def test_gui_runtime_files_marked_not_selected(gui):
    app, controller, repo = gui
    (repo / "app.db").write_text("x", encoding="utf-8")
    snap = app._git_service.snapshot()
    app._git_snapshot = snap
    app._render_git_snapshot(snap)
    rows = [app.git_tree.item(i, "values") for i in app.git_tree.get_children()]
    notes = {row[1]: row[2] for row in rows}
    assert notes.get("app.db") == "RUNTIME / SENSITIVE"
    assert any(note == "RUNTIME / SENSITIVE" for note in notes.values())
    assert app.git_tree.selection() == ()


@needs_git
def test_gui_shows_local_no_upstream(gui, tmp_path):
    app, controller, repo = gui
    bare = tmp_path / "remote.git"
    run_git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    run_git(repo, "remote", "add", "origin", str(bare))
    (repo / "a.txt").write_text("1", encoding="utf-8")
    run_git(repo, "add", "--", "a.txt")
    run_git(repo, "commit", "-m", "init")
    snap = app._git_service.snapshot()
    app._git_snapshot = snap
    app._render_git_snapshot(snap)
    assert "LOCAL / NO UPSTREAM" in app.git_ahead_var.get()
    assert app.git_behind_var.get() == "Behind: —"
    assert "origin" in app.git_remote_var.get()
    assert str(app.btn_git_push["state"]) == "normal"


@needs_git
def test_gui_non_repo_state(tmp_path):
    try:
        app = wt.TandemGUI()
    except Exception as exc:  # pragma: no cover - headless CI
        pytest.skip(f"Tk unavailable: {exc}")
    app.root.withdraw()
    plain = tmp_path / "plain"
    plain.mkdir()
    controller = wt.TandemController(plain)
    controller.db.import_plan(PLAN_DEF)
    app.controller = controller
    app._git_service = wt.GitService(plain)
    try:
        snap = app._git_service.snapshot()
        app._git_snapshot = snap
        app._render_git_snapshot(snap)
        assert app.git_state_var.get() == "Git: NOT A GIT REPOSITORY"
        assert str(app.btn_git_commit["state"]) == "disabled"
        assert str(app.btn_git_commit_push["state"]) == "disabled"
        assert str(app.btn_git_push["state"]) == "disabled"
    finally:
        app.on_close()
        controller.db.close()


@needs_git
def test_gui_refresh_is_non_blocking(gui):
    app, controller, repo = gui
    (repo / "a.txt").write_text("1", encoding="utf-8")
    start = time.perf_counter()
    app.git_status()
    assert time.perf_counter() - start < 1.0
    assert app.busy is True  # the snapshot runs off the Tk thread

    def stop_when_done():
        if not app.busy:
            app.root.quit()
        else:
            app.root.after(25, stop_when_done)

    app.root.after(25, stop_when_done)
    app.root.after(20000, app.root.quit)  # safety timeout, never hang the suite
    app.root.mainloop()

    assert app.busy is False
    assert "READY" in app.git_state_var.get()
