"""Private remote execution helpers for the central Batch Processor.

The public toolbox ships this adapter, but it stays dormant unless a user
points Slicer at a private config file. The remote environment only needs the
core CLI packages; it does not need Slicer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
from typing import Iterable, Mapping


CONFIG_ENV_VAR = "SLICER_BONE_BATCH_REMOTE_CONFIG"
BACKEND_ENV_VAR = "SLICER_BONE_BATCH_BACKEND"


@dataclass(frozen=True)
class RemoteBatchConfig:
    """Configuration for SSH/SLURM-backed batch execution."""

    name: str
    host: str
    remote_root: str
    python: str
    work_dir: str
    local_root: str | None = None
    ssh: str = "ssh"
    rsync: str = "rsync"
    scheduler: str = "slurm"
    setup_command: str = ""
    environment: Mapping[str, str] = field(default_factory=dict)
    sbatch_options: tuple[str, ...] = ()

    def remote_dataset_root(self, dataset_root: str | Path) -> str:
        """Return the remote dataset root matching a local or remote input root."""
        root = str(Path(dataset_root).expanduser()) if dataset_root else ""
        if self.local_root:
            local_root = str(Path(self.local_root).expanduser())
            if root == local_root or root.startswith(local_root + os.sep):
                suffix = root[len(local_root):].lstrip(os.sep)
                return _posix_join(self.remote_root, suffix)
        return self.remote_root if root and not root.startswith("/") else (root or self.remote_root)

    def local_dataset_root(self) -> Path | None:
        """Return the optional local mirror root used for output load-back."""
        return Path(self.local_root).expanduser() if self.local_root else None

    def remote_args(self, args: Iterable[str], *, dataset_root: str | Path | None = None) -> list[str]:
        """Map local dataset-root arguments inside a CLI argv vector to remote paths."""
        local_roots = []
        if dataset_root:
            local_roots.append(str(Path(dataset_root).expanduser()))
        if self.local_root:
            configured = str(Path(self.local_root).expanduser())
            local_roots.append(configured)
            try:
                local_roots.append(str(Path(configured).resolve()))
            except Exception:
                pass
        local_roots = _unique_existing_prefixes(local_roots)
        remote_root = self.remote_root
        mapped: list[str] = []
        for arg in args:
            value = str(arg)
            matched_root = _matching_local_prefix(value, local_roots)
            if matched_root:
                suffix = value[len(matched_root):].lstrip(os.sep)
                mapped.append(_posix_join(remote_root, suffix))
            else:
                mapped.append(value)
        return mapped


class SshSlurmBatchBackend:
    """Build SSH commands that submit toolbox CLIs to SLURM."""

    def __init__(self, config: RemoteBatchConfig) -> None:
        if config.scheduler.lower() not in {"slurm", "ssh-slurm"}:
            raise ValueError(f"Unsupported remote scheduler: {config.scheduler}")
        self.config = config

    def discover_argv(self, *, families: Iterable[str] = ()) -> list[str]:
        """Return an SSH argv that emits remote normalized discovery JSON."""
        family_args = " ".join(
            f"--family {shlex.quote(str(family))}"
            for family in families
            if str(family).strip()
        )
        command = (
            f"{shlex.quote(self.config.python)} -m bone_imaging_derivatives.remote_discovery "
            f"{shlex.quote(self.config.remote_root)} {family_args}"
        ).strip()
        return [self.config.ssh, self.config.host, self._login_shell(command)]

    def submit_argv(self, args: Iterable[str], *, job_name: str) -> list[str]:
        """Return an SSH argv that submits one CLI command and prints the job id."""
        remote_args = [str(arg) for arg in args]
        command = " ".join([shlex.quote(self.config.python), *[shlex.quote(arg) for arg in remote_args]])
        job_token = _safe_job_token(job_name)
        remote_dir = _posix_join(self.config.work_dir, "jobs", job_token)
        script_path = _posix_join(remote_dir, "run.sh")
        log_path = _posix_join(remote_dir, "slurm.log")
        env_exports = "\n".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in sorted(self.config.environment.items())
            if _valid_shell_name(key)
        )
        setup = self.config.setup_command.strip()
        script_body = "\n".join(
            line
            for line in (
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                env_exports,
                setup,
                f"cd {shlex.quote(self.config.remote_root)}",
                command,
            )
            if line
        )
        sbatch_options = " ".join(shlex.quote(str(option)) for option in self.config.sbatch_options if str(option).strip())
        submit = (
            f"mkdir -p {shlex.quote(remote_dir)} && "
            f"cat > {shlex.quote(script_path)} <<'BONE_BATCH_SCRIPT'\n{script_body}\nBONE_BATCH_SCRIPT\n"
            f"chmod +x {shlex.quote(script_path)} && "
            f"sbatch --parsable --job-name={shlex.quote(job_token)} --output={shlex.quote(log_path)} --error={shlex.quote(log_path)} "
            f"{sbatch_options} {shlex.quote(script_path)}"
        )
        return [self.config.ssh, self.config.host, self._login_shell(submit)]

    def status_argv(self, job_id: str) -> list[str]:
        """Return an SSH argv that prints a compact SLURM state for ``job_id``."""
        safe_job = shlex.quote(str(job_id).strip())
        command = (
            f"state=$(squeue -h -j {safe_job} -o %T 2>/dev/null | head -n 1 || true); "
            'if [ -n "$state" ]; then echo "$state"; '
            f"else sacct -n -P -j {safe_job} --format=State,ExitCode 2>/dev/null | head -n 1; fi"
        )
        return [self.config.ssh, self.config.host, self._login_shell(command)]

    def cancel_argv(self, job_id: str) -> list[str]:
        """Return an SSH argv that cancels ``job_id``."""
        return [self.config.ssh, self.config.host, f"scancel {shlex.quote(str(job_id).strip())}"]

    def log_argv(self, job_name: str, *, lines: int = 80) -> list[str]:
        """Return an SSH argv that tails the saved SLURM log for ``job_name``."""
        job_token = _safe_job_token(job_name)
        log_path = _posix_join(self.config.work_dir, "jobs", job_token, "slurm.log")
        command = f"tail -n {int(lines)} {shlex.quote(log_path)} 2>/dev/null || true"
        return [self.config.ssh, self.config.host, self._login_shell(command)]

    @staticmethod
    def parse_job_id(output: str) -> str | None:
        """Extract a SLURM job id from ``sbatch`` output."""
        text = str(output or "").strip()
        if not text:
            return None
        first = text.splitlines()[-1].strip()
        match = re.search(r"\b(\d+)(?:[.;]\S*)?\b", first)
        return match.group(1) if match else None

    def sync_output_argv(self, family: str) -> list[str]:
        """Return an rsync argv for one derivative family back to the local mirror."""
        local_root = self.config.local_dataset_root()
        if local_root is None:
            raise ValueError("Remote batch config needs local_root before outputs can be loaded in Slicer.")
        source = f"{self.config.host}:{_posix_join(self.config.remote_root, 'derivatives', family)}/"
        target = local_root / "derivatives" / family
        return [self.config.rsync, "-az", source, str(target) + os.sep]

    @staticmethod
    def _login_shell(command: str) -> str:
        return f"bash -lc {shlex.quote(command)}"


def load_remote_batch_config(path: str | Path | None = None) -> RemoteBatchConfig:
    """Load a private remote backend config from JSON or simple YAML."""
    raw_path = str(path or os.environ.get(CONFIG_ENV_VAR, "")).strip()
    if not raw_path:
        raise ValueError(f"Set {CONFIG_ENV_VAR} to a remote batch config file.")
    config_path = Path(raw_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Remote batch config does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Remote batch config is not a file: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(text) if config_path.suffix.lower() == ".json" else _parse_simple_yaml(text)
    return _config_from_mapping(payload)


def _config_from_mapping(payload: Mapping[str, object]) -> RemoteBatchConfig:
    required = ("name", "host", "remote_root", "python", "work_dir")
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Remote batch config is missing: {', '.join(missing)}")
    environment = payload.get("environment") or {}
    if not isinstance(environment, Mapping):
        raise ValueError("Remote batch config 'environment' must be a mapping.")
    sbatch_options = payload.get("sbatch_options") or ()
    if isinstance(sbatch_options, str):
        sbatch_options = tuple(shlex.split(sbatch_options))
    return RemoteBatchConfig(
        name=str(payload["name"]).strip(),
        host=str(payload["host"]).strip(),
        remote_root=str(payload["remote_root"]).rstrip("/"),
        python=str(payload["python"]).strip(),
        work_dir=str(payload["work_dir"]).rstrip("/"),
        local_root=str(payload["local_root"]).rstrip(os.sep) if payload.get("local_root") else None,
        ssh=str(payload.get("ssh") or "ssh"),
        rsync=str(payload.get("rsync") or "rsync"),
        scheduler=str(payload.get("scheduler") or "slurm"),
        setup_command=str(payload.get("setup_command") or ""),
        environment={str(key): str(value) for key, value in environment.items()},
        sbatch_options=tuple(str(option) for option in sbatch_options),
    )


def _parse_simple_yaml(text: str) -> dict[str, object]:
    payload: dict[str, object] = {}
    current_mapping: str | None = None
    current_list: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith((" ", "\t")):
            if raw_line.strip().startswith("-") and (current_list or current_mapping):
                list_key = current_list or current_mapping
                if list_key and not isinstance(payload.get(list_key), list):
                    payload[list_key] = []
                current_list = list_key
                current_mapping = None
                values = payload.setdefault(str(list_key), [])
                if isinstance(values, list):
                    values.append(str(_coerce_scalar(raw_line.strip()[1:].strip())))
                continue
            if current_mapping and ":" in raw_line:
                key, value = raw_line.strip().split(":", 1)
                mapping = payload.setdefault(current_mapping, {})
                if isinstance(mapping, dict):
                    mapping[key.strip()] = _coerce_scalar(value.strip())
                continue
        current_mapping = None
        current_list = None
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            payload[key] = {}
            current_mapping = key
        elif value == "[]":
            payload[key] = []
            current_list = key
        else:
            payload[key] = _coerce_scalar(value)
    return payload


def _coerce_scalar(value: str) -> object:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _unique_existing_prefixes(paths: Iterable[str]) -> list[str]:
    """Return de-duplicated local prefixes, longest first for stable path mapping."""
    seen: set[str] = set()
    unique: list[str] = []
    for raw_path in paths:
        path = str(raw_path or "").rstrip(os.sep)
        if not path or path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return sorted(unique, key=len, reverse=True)


def _matching_local_prefix(value: str, local_roots: Iterable[str]) -> str:
    for local_root in local_roots:
        if value == local_root or value.startswith(local_root + os.sep):
            return local_root
    return ""


def _posix_join(*parts: str) -> str:
    cleaned = [str(part).strip("/") for part in parts if str(part).strip("/")]
    if not cleaned:
        return "/"
    return "/" + "/".join(cleaned) if str(parts[0]).startswith("/") else "/".join(cleaned)


def _safe_job_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "-_" else "-" for char in str(value or "bone-batch"))
    return token.strip("-")[:80] or "bone-batch"


def _valid_shell_name(value: str) -> bool:
    return bool(value) and (value[0].isalpha() or value[0] == "_") and all(char.isalnum() or char == "_" for char in value)
