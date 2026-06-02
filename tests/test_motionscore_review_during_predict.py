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


def test_motionscore_preloads_next_scan_without_background_mrml() -> None:
    source = _module_source()
    assert 'self.preloadNextScanCheck = qt.QCheckBox("Preload next scan")' in source
    assert "ThreadPoolExecutor(max_workers=1" in source
    assert "def _read_aim_for_preload" in source
    assert "def _load_preloaded_aim" in source
    assert "self._try_get_preloaded_scan(scan_id)" in source

    preload_reader_start = source.index("def _read_aim_for_preload")
    preload_reader_end = source.index("def _clear_preload_cache")
    preload_reader = source[preload_reader_start:preload_reader_end]
    assert "slicer.mrmlScene" not in preload_reader
    assert "slicer.util.updateVolumeFromArray" not in preload_reader


def test_motionscore_manual_grade_uses_in_process_fast_path() -> None:
    source = _module_source()
    on_apply_start = source.index("def onApplyManual")
    on_apply_end = source.index("def _review_artifact_path")
    on_apply = source[on_apply_start:on_apply_end]
    assert '"review-apply"' not in on_apply
    assert "self._apply_manual_review_in_process(" in on_apply

    assert "def _on_manual_applied_fast" in source
    assert "self._refresh_and_load_next(previous_scan_id=scan_id, refresh=False)" in source
