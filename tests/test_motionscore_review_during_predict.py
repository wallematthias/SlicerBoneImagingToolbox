from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "MotionScoreHRpQCT" / "MotionScoreHRpQCT.py"


def _module_source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_motionscore_prediction_resume_controls_are_wired() -> None:
    source = _module_source()
    assert 'self.runButton = qt.QPushButton("Predict / Resume")' in source
    assert 'self.forcePredictCheck = qt.QCheckBox("Reprocess existing predictions")' in source
    assert 'args.append("--force")' in source
    assert 'MIN_CORE_VERSION = "2.5.8"' in source


def test_motionscore_review_stays_available_while_predict_runs() -> None:
    source = _module_source()
    assert 'predict_running = bool(not enabled and self._active_task_name == "predict")' in source
    assert "self.scanCombo.enabled = review_enabled" in source
    assert "self.applyButton.enabled = review_enabled" in source
    assert "self._start_live_review_timer()" in source
    assert "self.refreshReview(quiet=True)" in source
    assert "from motionscore.review.store import apply_manual_review" in source
