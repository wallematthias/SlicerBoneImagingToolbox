from __future__ import annotations

import os
import shlex
import subprocess
import sys
import shutil
import tempfile
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path


def clean_pip_environment(base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    for key in (
        "CC",
        "CXX",
        "CPP",
        "PYTHONHOME",
        "PYTHONPATH",
        "ITK_AUTOLOAD_PATH",
        "SITK_AUTOLOAD_PATH",
        "SimpleITK_AUTOLOAD_PATH",
    ):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


@contextmanager
def slicer_pip_constraints():
    """Keep Slicer's installed SimpleITK build during dependency resolution.

    An ordinary PyPI wheel is not interchangeable with Slicer's custom build,
    even when their version numbers match. Do not combine this constraint with
    dependency-wide --force-reinstall. Repair an already replaced build by
    restoring SimpleITK from the matching official Slicer distribution.
    """
    try:
        version = metadata.version("SimpleITK")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Slicer's bundled SimpleITK is missing. Restore it from the matching "
            "official Slicer distribution before installing toolbox packages."
        ) from exc
    with tempfile.TemporaryDirectory(prefix="bone-toolbox-pip-") as directory:
        path = Path(directory) / "constraints.txt"
        path.write_text(f"SimpleITK=={version}\n", encoding="utf-8")
        yield path


def slicer_pip_install(command):
    install_args = shlex.split(str(command))
    if "--force-reinstall" in install_args and "--no-deps" not in install_args:
        raise ValueError(
            "Dependency-wide --force-reinstall can replace Slicer's SimpleITK. "
            "Use an ordinary upgrade, or --no-deps for a targeted reinstall."
        )
    with slicer_pip_constraints() as constraints:
        args = [sys.executable, "-m", "pip", "install", "--constraint", str(constraints), *install_args]
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=clean_pip_environment(),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or f"pip install failed with exit code {completed.returncode}")
    return completed.stdout


def slicer_python_executable(application_path=None):
    """Return PythonSlicer rather than the Slicer GUI executable for child jobs."""
    if application_path:
        application = Path(application_path).expanduser().resolve()
        candidate = application.parent.parent / "bin" / "PythonSlicer"
        if candidate.exists():
            return str(candidate)
    sibling = Path(sys.executable).resolve().parent / "PythonSlicer"
    if sibling.exists():
        return str(sibling)
    return shutil.which("PythonSlicer") or sys.executable
