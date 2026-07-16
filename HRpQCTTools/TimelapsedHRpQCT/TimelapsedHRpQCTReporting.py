from __future__ import annotations

from TimelapsedHRpQCTLib.Reporting import *  # noqa: F401,F403

try:
    from slicer.ScriptedLoadableModule import ScriptedLoadableModule
except Exception:  # pragma: no cover - only used outside Slicer test imports
    ScriptedLoadableModule = object


class TimelapsedHRpQCTReporting(ScriptedLoadableModule):
    """Hidden compatibility module for legacy top-level reporting imports."""

    def __init__(self, parent=None):
        try:
            super().__init__(parent)
        except TypeError:
            pass
        if parent is not None:
            parent.hidden = True
            parent.title = "Timelapsed HR-pQCT Reporting Compatibility"
