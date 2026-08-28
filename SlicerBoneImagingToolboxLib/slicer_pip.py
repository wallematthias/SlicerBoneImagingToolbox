from __future__ import annotations

import os
import shlex
import subprocess
import sys


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


def slicer_pip_install(command):
    args = [sys.executable, "-m", "pip", "install", *shlex.split(str(command))]
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
