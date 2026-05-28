from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import subprocess
import tempfile
import time
from urllib import request as urllib_request
from zipfile import ZipFile
from pathlib import Path


TOOLBOX_MODULE_DIRS = ("TimelapsedHRpQCT", "MotionScoreHRpQCT", "HRpQCTSegmentation", "ScancoIO")
GITHUB_REPO = "wallematthias/SlicerTimelapsedHRpQCT"
GITHUB_BRANCH = "main"
GITHUB_API_COMMIT_URL = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
GITHUB_ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{GITHUB_BRANCH}.zip"
HTTP_HEADERS = {"User-Agent": "SlicerTimelapsedHRpQCT-Updater/1.0"}


@dataclass(frozen=True)
class ModuleUpdateContext:
    toolbox_root: Path
    git_root: Path | None
    strategy: str


@dataclass(frozen=True)
class ToolboxUpdateCheck:
    context: ModuleUpdateContext
    installed_revision: str | None
    latest_revision: str
    update_available: bool
    message: str


@dataclass(frozen=True)
class ToolboxUpdateResult:
    context: ModuleUpdateContext
    updated: bool
    restart_required: bool
    message: str


def _has_toolbox_modules(path: Path) -> bool:
    return all((path / name).is_dir() for name in TOOLBOX_MODULE_DIRS)


def find_toolbox_root(start_path: str | Path) -> Path:
    """Return the folder that contains all toolbox module directories."""
    start = Path(start_path).resolve()
    candidates = [start if start.is_dir() else start.parent, *((start if start.is_dir() else start.parent).parents)]
    for candidate in candidates:
        if _has_toolbox_modules(candidate):
            return candidate
    raise RuntimeError(f"Could not locate SlicerTimelapsedHRpQCT toolbox root from {start}")


def find_git_root(start_path: str | Path) -> Path | None:
    """Return the nearest git checkout root, including worktree .git files."""
    start = Path(start_path).resolve()
    base = start if start.is_dir() else start.parent
    for candidate in [base, *base.parents]:
        if _has_toolbox_modules(candidate) and (candidate / ".git").exists():
            return candidate
    return None


def detect_update_context(start_path: str | Path) -> ModuleUpdateContext:
    toolbox_root = find_toolbox_root(start_path)
    git_root = find_git_root(toolbox_root)
    return ModuleUpdateContext(
        toolbox_root=toolbox_root,
        git_root=git_root,
        strategy="git" if git_root is not None else "zip",
    )


def _run_git(args: list[str], cwd: Path, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _read_json_url(url: str, *, timeout: int = 8) -> dict:
    req = urllib_request.Request(url, headers=HTTP_HEADERS)
    with urllib_request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def latest_remote_revision(*, timeout: int = 8) -> str:
    payload = _read_json_url(GITHUB_API_COMMIT_URL, timeout=timeout)
    sha = str(payload.get("sha") or "").strip()
    if not sha:
        raise RuntimeError("GitHub update check did not return a commit SHA.")
    return sha


def installed_revision(context: ModuleUpdateContext) -> str | None:
    if context.git_root is not None:
        result = _run_git(["rev-parse", "HEAD"], context.git_root)
        if result.returncode == 0:
            return result.stdout.strip() or None

    manifest = context.toolbox_root / ".slicer_toolbox_update.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            sha = str(payload.get("revision") or "").strip()
            return sha or None
        except Exception:
            return None
    return None


def check_for_updates(start_path: str | Path, *, timeout: int = 8) -> ToolboxUpdateCheck:
    context = detect_update_context(start_path)
    latest = latest_remote_revision(timeout=timeout)
    installed = installed_revision(context)
    available = installed is None or not latest.startswith(installed) and not installed.startswith(latest)
    if available:
        where = "git checkout" if context.strategy == "git" else "manual download"
        message = f"Update available for {where}: {short_revision(installed)} -> {short_revision(latest)}."
    else:
        message = f"Toolbox is up to date ({short_revision(latest)})."
    return ToolboxUpdateCheck(
        context=context,
        installed_revision=installed,
        latest_revision=latest,
        update_available=available,
        message=message,
    )


def short_revision(revision: str | None) -> str:
    if not revision:
        return "unknown"
    return str(revision)[:12]


def git_worktree_is_clean(git_root: Path) -> bool:
    result = _run_git(["status", "--porcelain"], git_root)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git status failed").strip())
    return not result.stdout.strip()


def update_git_checkout(context: ModuleUpdateContext) -> ToolboxUpdateResult:
    if context.git_root is None:
        raise RuntimeError("Not a git checkout.")
    if not git_worktree_is_clean(context.git_root):
        return ToolboxUpdateResult(
            context=context,
            updated=False,
            restart_required=False,
            message=(
                "This toolbox is a git checkout with local changes. "
                f"Update manually in {context.git_root} after saving or stashing your work."
            ),
        )
    result = _run_git(["pull", "--ff-only"], context.git_root, timeout=120)
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if result.returncode != 0:
        return ToolboxUpdateResult(
            context=context,
            updated=False,
            restart_required=False,
            message=f"Git fast-forward update failed:\n{output}",
        )
    return ToolboxUpdateResult(
        context=context,
        updated="Already up to date" not in output,
        restart_required=True,
        message=f"Git checkout updated with fast-forward pull.\n{output}",
    )


def _download_file(url: str, destination: Path, *, timeout: int = 30) -> None:
    req = urllib_request.Request(url, headers=HTTP_HEADERS)
    with urllib_request.urlopen(req, timeout=timeout) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _find_extracted_toolbox_root(extract_dir: Path) -> Path:
    for candidate in [extract_dir, *extract_dir.iterdir()]:
        if candidate.is_dir() and _has_toolbox_modules(candidate):
            return candidate
    raise RuntimeError("Downloaded archive did not contain the expected toolbox modules.")


def update_manual_download(context: ModuleUpdateContext, *, revision: str | None = None) -> ToolboxUpdateResult:
    if context.git_root is not None:
        raise RuntimeError("Refusing to replace a git checkout with a zip download.")
    parent = context.toolbox_root.parent
    backup = parent / f"{context.toolbox_root.name}-backup-{time.strftime('%Y%m%d-%H%M%S')}"
    with tempfile.TemporaryDirectory(prefix="slicer_toolbox_update_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / "toolbox.zip"
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir()
        _download_file(GITHUB_ZIP_URL, zip_path)
        with ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        extracted_root = _find_extracted_toolbox_root(extract_dir)
        shutil.move(str(context.toolbox_root), str(backup))
        try:
            shutil.move(str(extracted_root), str(context.toolbox_root))
            manifest = {
                "repo": GITHUB_REPO,
                "branch": GITHUB_BRANCH,
                "revision": revision or latest_remote_revision(),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "backup_path": str(backup),
            }
            (context.toolbox_root / ".slicer_toolbox_update.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            if context.toolbox_root.exists():
                shutil.rmtree(context.toolbox_root, ignore_errors=True)
            shutil.move(str(backup), str(context.toolbox_root))
            raise
    return ToolboxUpdateResult(
        context=context,
        updated=True,
        restart_required=True,
        message=(
            "Manual toolbox download updated from GitHub main. "
            f"Previous folder was backed up at {backup}."
        ),
    )


def update_toolbox(start_path: str | Path, *, latest_revision: str | None = None) -> ToolboxUpdateResult:
    context = detect_update_context(start_path)
    if context.strategy == "git":
        return update_git_checkout(context)
    return update_manual_download(context, revision=latest_revision)
