from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "HRpQCTTools" / "MotionScoreHRpQCT" / "MotionScoreHRpQCT.py"


def _module_source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_motionscore_prediction_resume_controls_are_wired() -> None:
    source = _module_source()
    assert 'self.runButton = qt.QPushButton("Predict / Resume")' in source
    assert 'self.forcePredictCheck = qt.QCheckBox("Reprocess existing predictions")' in source
    assert 'args.append("--force")' in source
    assert 'MIN_CORE_VERSION = "2.5.11"' in source


def test_motionscore_review_stays_available_while_predict_runs() -> None:
    source = _module_source()
    assert 'predict_running = bool(not enabled and self._active_task_name == "predict")' in source
    assert "self.scanCombo.enabled = review_enabled" in source
    assert "self.applyButton.enabled = review_enabled" in source
    assert "self._start_live_review_timer()" in source
    assert "self.refreshReview(quiet=True)" in source
    assert "from motionscore.review.store import apply_manual_review" in source


def test_motionscore_grading_shortcuts_use_application_event_filter() -> None:
    source = _module_source()

    assert "_QT_OBJECT_BASE = getattr(qt, \"QObject\", object)" in source
    assert "class _GradingShortcutEventFilter(_QT_OBJECT_BASE):" in source
    assert "owner._handle_grading_shortcut_event(obj, event)" in source
    assert "self._grading_shortcut_filter = _GradingShortcutEventFilter(self)" in source
    assert "app.installEventFilter(self._grading_shortcut_filter)" in source
    assert "self._install_grading_shortcuts()" not in source
    assert "qt.QShortcut" not in source
    assert "def _handle_grading_shortcut_event" in source

    handler_start = source.index("def _handle_grading_shortcut_event")
    handler_end = source.index("def _auto_load_enabled", handler_start)
    handler = source[handler_start:handler_end]
    assert "qt.QEvent.KeyPress" in handler
    assert "event.isAutoRepeat()" in handler
    assert "qt.Qt.ControlModifier" in handler
    assert "qt.Qt.MetaModifier" in handler
    assert "qt.Qt.Key_1" in handler
    assert "qt.Qt.Key_5" in handler
    assert "self.onQuickSelectGrade(grade)" in handler
    assert "return True" in handler


def test_motionscore_preloads_next_scan_without_background_mrml() -> None:
    source = _module_source()
    assert "self.cacheAheadSlider = qt.QSlider(qt.Qt.Horizontal)" in source
    assert "self.cacheAheadSlider.minimum = 0" in source
    assert "self.cacheAheadSlider.maximum = 20" in source
    assert 'self.cacheAheadSpin = qt.QSpinBox()' in source
    assert "self.cacheAheadSpin.minimum = 0" in source
    assert "self.cacheAheadSpin.maximum = 20" in source
    assert "cacheAheadRow.addWidget(self.cacheAheadSlider)" in source
    assert "cacheAheadRow.addWidget(self.cacheAheadSpin)" in source
    assert "MotionScore/CacheAheadCount" in source
    assert "DEFAULT_CACHE_AHEAD_COUNT = 5" in source
    assert "ThreadPoolExecutor(max_workers=1" in source
    assert "def _read_aim_for_preload" in source
    assert "def _load_preloaded_aim" in source
    assert "self._try_get_preloaded_scan(scan_id)" in source

    preload_reader_start = source.index("def _read_aim_for_preload")
    preload_reader_end = source.index("def _clear_preload_cache")
    preload_reader = source[preload_reader_start:preload_reader_end]
    assert "slicer.mrmlScene" not in preload_reader
    assert "slicer.util.updateVolumeFromArray" not in preload_reader


def test_motionscore_defers_preload_after_volume_display() -> None:
    source = _module_source()

    assert "PRELOAD_AFTER_LOAD_DELAY_MS = 250" in source
    assert "def _schedule_preload_next_scan_after_load" in source
    assert "qt.QTimer.singleShot(" in source
    assert "PRELOAD_AFTER_LOAD_DELAY_MS," in source
    assert "self._schedule_preload_next_scan_after_load(scan_id)" in source

    load_selected_start = source.index("def onLoadSelectedScan")
    load_selected_end = source.index("def _remove_loaded_scan_volume")
    load_selected = source[load_selected_start:load_selected_end]
    assert "self._schedule_preload_next_scan(scan_id)" not in load_selected


def test_motionscore_uses_single_worker_disk_cache_window() -> None:
    source = _module_source()

    assert "CACHE_WARM_IDLE_DELAY_MS" in source
    assert "CACHE_WARM_CONTINUE_DELAY_MS" in source
    assert "MAX_CACHE_AHEAD_COUNT = 20" in source
    assert "DEFAULT_CACHE_AHEAD_COUNT = 5" in source
    assert "import tempfile" in source
    assert "import numpy as np" in source
    assert "def _disk_cache_dir" in source
    assert "def _disk_cache_entry_paths" in source
    assert "def _raw_file_signature" in source
    assert "def _load_payload_from_disk_cache" in source
    assert "def _write_payload_to_disk_cache" in source
    assert "def _prune_disk_cache_window" in source
    assert "def _disk_cache_payload_exists" in source
    assert "def _schedule_cache_warm_after_idle" in source
    assert "def _start_cache_warm_if_idle" in source
    assert "def _next_uncached_scan_entry" in source
    assert "def _warm_disk_cache_entry_external" in source
    assert "def _poll_cache_warm_worker" in source
    assert "def _ensure_async_cache_state" in source
    assert "def _load_cached_payload_for_preload" in source
    assert "np.save(" in source
    assert "np.load(" in source
    assert "allow_pickle=False" in source

    schedule_start = source.index("def _schedule_preload_next_scan")
    schedule_end = source.index("def _schedule_preload_next_scan_after_load", schedule_start)
    schedule_preload = source[schedule_start:schedule_end]
    assert "if self._cache_ahead_count() <= 0:" in schedule_preload
    assert "self._preload_executor.submit(" in schedule_preload
    assert "self._load_cached_payload_for_preload" in schedule_preload
    assert "self._read_or_cache_aim_for_preload" not in schedule_preload
    assert "self._read_preload_window" not in schedule_preload
    assert "hot=1" in schedule_preload
    assert "cache miss; skipping remote background read" not in source
    assert "self._load_payload_from_disk_cache(scan_id, raw_path)" in source
    assert "self._prune_disk_cache_window(scan_entries)" in source

    warmer_start = source.index("def _start_cache_warm_if_idle")
    warmer_end = source.index("def _poll_cache_warm_worker", warmer_start)
    cache_warmer = source[warmer_start:warmer_end]
    assert "self._cache_warm_executor.submit(" in cache_warmer
    assert "self._warm_disk_cache_entry_external" in cache_warmer
    assert "self._active_load_future is not None" in cache_warmer
    assert "self._preload_future is not None" in cache_warmer

    external_start = source.index("def _warm_disk_cache_entry_external")
    external_end = source.index("def _poll_cache_warm_worker", external_start)
    external = source[external_start:external_end]
    assert "subprocess.run(" in external
    assert "/usr/bin/nice" in external
    assert "from motionscore.io.aim import read_aim" in external

    poll_start = source.index("def _poll_cache_warm_worker")
    poll_end = source.index("def _selected_load_timer_instance", poll_start)
    poll_warmer = source[poll_start:poll_end]
    assert "CACHE_WARM_CONTINUE_DELAY_MS" in poll_warmer
    assert "self._schedule_cache_warm_after_idle(self._loaded_scan_id" in poll_warmer


def test_motionscore_cache_ahead_count_controls_preload_window() -> None:
    source = _module_source()

    assert "def _cache_ahead_count" in source
    count_start = source.index("def _cache_ahead_count")
    count_end = source.index("def _preload_enabled", count_start)
    count_func = source[count_start:count_end]
    assert "self.cacheAheadSpin.value" in count_func
    assert "MAX_CACHE_AHEAD_COUNT" in count_func
    assert "max(0, min(MAX_CACHE_AHEAD_COUNT" in count_func

    preload_enabled_start = source.index("def _preload_enabled")
    preload_enabled_end = source.index("def _set_selected_manual_grade", preload_enabled_start)
    preload_enabled = source[preload_enabled_start:preload_enabled_end]
    assert "return self._cache_ahead_count() > 0" in preload_enabled

    window_start = source.index("def _preload_window_scan_ids")
    window_end = source.index("def _clear_preload_cache", window_start)
    window = source[window_start:window_end]
    assert "window_size = self._cache_ahead_count()" in window
    assert "if window_size <= 0:" in window
    assert "start + window_size" in window


def test_motionscore_async_load_displays_only_current_request() -> None:
    source = _module_source()

    assert "self._load_request_token = 0" in source

    selected_start = source.index("def onLoadSelectedScan")
    selected_end = source.index("def _remove_loaded_scan_volume", selected_start)
    selected = source[selected_start:selected_end]
    assert "request_token = self._next_load_request_token()" in selected
    assert "queue_selected_load(scan_id, raw_path, request_token)" in selected

    assert "def _next_load_request_token" in source

    queue_start = source.index("def _queue_selected_scan_load")
    queue_end = source.index("def _poll_selected_scan_load_worker", queue_start)
    queue = source[queue_start:queue_end]
    assert "request_token" in queue
    assert "self._active_load_request_token = request_token" in queue

    poll_start = source.index("def _poll_selected_scan_load_worker")
    poll_end = source.index("def _poll_preload_worker", poll_start)
    poll = source[poll_start:poll_end]
    assert "request_token = self._active_load_request_token" in poll
    assert "request_token != self._load_request_token" in poll
    assert "payload.get(\"scan_id\") != scan_id" in poll
    assert "discarded stale async scan load" in poll


def test_motionscore_dataset_path_change_does_not_discover_remote_dataset() -> None:
    source = _module_source()

    path_changed_start = source.index("def onDatasetPathChanged")
    path_changed_end = source.index("def onScanSelectionChanged", path_changed_start)
    path_changed = source[path_changed_start:path_changed_end]

    assert "_discover_scan_ids_for_dataset" not in path_changed
    assert "self.scanCombo.clear()" in path_changed
    assert "Click Load Dataset" in path_changed
    assert 'getattr(self, "_clear_selected_scan_load", None)' in path_changed


def test_motionscore_load_dataset_does_not_walk_raw_dataset_tree() -> None:
    source = _module_source()

    refresh_start = source.index("def refreshReview")
    refresh_end = source.index("def _start_live_review_timer", refresh_start)
    refresh_review = source[refresh_start:refresh_end]
    missing_index_start = refresh_review.index("if not index_path.exists():")
    missing_index_end = refresh_review.index("return", missing_index_start)
    missing_index_branch = refresh_review[missing_index_start:missing_index_end]

    summary_start = source.index("def _update_dataset_summary")
    summary_end = source.index("def onTrainingModeToggled", summary_start)
    dataset_summary = source[summary_start:summary_end]

    assert "_discover_scan_ids_for_dataset" not in missing_index_branch
    assert "_discover_scan_ids_for_dataset" not in dataset_summary
    assert 'self._set_run_scope_items([])' in missing_index_branch
    assert "processed scan(s) indexed" in dataset_summary
    assert "no MotionScore index found yet" in dataset_summary


def test_motionscore_wrapper_uses_derivatives_family_root() -> None:
    source = _module_source()
    helper = source[source.index("def _derivative_family_root") : source.index("def _models_dir")]

    assert 'return self._derivative_family_root(dataset, "MotionScore")' in source
    assert 'return self._derivative_family_root(self._review_output_root, "MotionScore")' in source
    assert 'self._derivative_family_root(plan.output_root, "MotionScore")' in source
    assert 'if root.name == "derivatives":' in helper
    assert '/ "derivatives" / family' in helper


def test_motionscore_selected_scan_fallback_load_is_async() -> None:
    source = _module_source()

    load_start = source.index("def onLoadSelectedScan")
    load_end = source.index("def _remove_loaded_scan_volume", load_start)
    load_selected = source[load_start:load_end]

    assert "queue_selected_load(scan_id, raw_path, request_token)" in load_selected
    assert "self._load_payload_from_disk_cache(scan_id, raw_path)" not in load_selected
    assert "slicer.util.loadVolume(raw_path)" not in load_selected
    assert "self._load_aim_with_core(raw_path)" not in load_selected
    assert 'getattr(self, "_queue_selected_scan_load", None)' in load_selected
    assert "restart Slicer" in load_selected
    assert "self._ensure_async_cache_state()" in source

    assert "self._active_load_future = None" in source
    assert "def _queue_selected_scan_load" in source
    assert "def _poll_selected_scan_load_worker" in source
    assert "def _selected_load_timer_instance" in source

    queue_start = source.index("def _queue_selected_scan_load")
    queue_end = source.index("def _poll_selected_scan_load_worker", queue_start)
    queue_load = source[queue_start:queue_end]
    assert "self._preload_executor.submit(" in queue_load
    assert "self._read_or_cache_aim_for_preload" in queue_load

    poll_start = source.index("def _poll_selected_scan_load_worker")
    poll_end = source.index("def _load_preloaded_aim", poll_start)
    poll_load = source[poll_start:poll_end]
    assert "future.result()" in poll_load
    assert "self._load_preloaded_aim(payload)" in poll_load
    assert "self._schedule_preload_next_scan_after_load(scan_id)" in poll_load


def test_motionscore_manual_grade_uses_in_process_fast_path() -> None:
    source = _module_source()
    on_apply_start = source.index("def onApplyManual")
    on_apply_end = source.index("def _review_artifact_path")
    on_apply = source[on_apply_start:on_apply_end]
    assert '"review-apply"' not in on_apply
    assert "self._apply_manual_review_in_process(" in on_apply

    assert "def _on_manual_applied_fast" in source
    assert "self._refresh_and_load_next(previous_scan_id=scan_id, refresh=False)" in source


def test_motionscore_manual_grade_rebuilds_queue_on_next_scan() -> None:
    source = _module_source()

    assert "from SlicerBoneImagingToolboxLib.motionscore_review import next_review_scan_id" in source

    advance_start = source.index("def _refresh_and_load_next")
    advance_end = source.index("def _set_run_scope_items", advance_start)
    advance = source[advance_start:advance_end]

    assert "target_scan_id = next_review_scan_id(previous_scan_id, self._scan_ids_for_scope())" in advance
    assert "self._rebuild_scan_combo(preferred_scan_id=target_scan_id)" in advance
    assert 'self._rebuild_scan_combo(preferred_scan_id="")' not in advance
