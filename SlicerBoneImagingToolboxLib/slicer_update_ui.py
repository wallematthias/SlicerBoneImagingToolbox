from __future__ import annotations

from pathlib import Path

import qt
import slicer

from .registry import TOOLBOX_DISPLAY_NAME
from .updater import check_for_updates, short_revision, update_toolbox


def run_toolbox_update_dialog(module_file: str | Path, *, log=None) -> None:
    """Check for toolbox updates and optionally install them."""

    def _log(message: str) -> None:
        if log is not None:
            try:
                log(str(message))
            except Exception:
                pass

    try:
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        check = check_for_updates(module_file)
    except Exception as exc:
        slicer.util.errorDisplay(f"Update check failed:\n{exc}")
        return
    finally:
        try:
            slicer.app.restoreOverrideCursor()
        except Exception:
            pass

    _log(f"[update] {check.message}\n")
    if not check.update_available:
        slicer.util.infoDisplay(check.message, windowTitle="Toolbox Update")
        return

    if check.context.strategy == "git":
        prompt = (
            f"{check.message}\n\n"
            f"Installed folder:\n{check.context.toolbox_root}\n\n"
            "This is a git checkout. I can run:\n"
            "git pull --ff-only\n\n"
            "This will only update if the checkout is clean and fast-forwardable. Continue?"
        )
    else:
        prompt = (
            f"{check.message}\n\n"
            f"Installed folder:\n{check.context.toolbox_root}\n\n"
            "This manual download will be replaced with the latest GitHub main zip. "
            "The current folder will be backed up first. Continue?"
        )
    if not slicer.util.confirmYesNoDisplay(prompt, windowTitle=f"Update {TOOLBOX_DISPLAY_NAME}"):
        return

    try:
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        result = update_toolbox(module_file, latest_revision=check.latest_revision)
    except Exception as exc:
        slicer.util.errorDisplay(f"Update failed:\n{exc}")
        return
    finally:
        try:
            slicer.app.restoreOverrideCursor()
        except Exception:
            pass

    _log(f"[update] {result.message}\n")
    if result.restart_required:
        slicer.util.infoDisplay(
            f"{result.message}\n\nRestart Slicer to use the updated toolbox.",
            windowTitle=f"Updated to {short_revision(check.latest_revision)}",
        )
    else:
        slicer.util.infoDisplay(result.message, windowTitle="Toolbox Update")

