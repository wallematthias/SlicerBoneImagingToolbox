"""Subprocess entry point for MotionScore Slicer scene runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MotionScore on a Slicer scene volume export.")
    parser.add_argument("volume_npz")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scan-id", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--confidence-threshold", type=int, default=75)
    parser.add_argument("--manual-only", action="store_true")
    parser.add_argument("--model-root")
    parser.add_argument("--model-id", default="base-v1")
    parser.add_argument("--slice-step", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stackheight", type=int, default=168)
    parser.add_argument("--slice-batch-size", type=int, default=64)
    return parser


def _load_scene_npz(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        volume = np.asarray(data["volume_xyz"])
        metadata = {
            "spacing": data["spacing"].tolist() if "spacing" in data else [],
            "origin": data["origin"].tolist() if "origin" in data else [],
        }
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D scene volume, got {volume.shape}")
    return volume, metadata


def _scene_session(args: argparse.Namespace, volume_path: Path):
    from motionscore.dataset.models import RawSession

    return RawSession(
        subject_id=str(args.subject_id),
        site=str(args.site),
        session_id=str(args.session_id),
        raw_image_path=volume_path,
    )


def _write_scene_prediction(args: argparse.Namespace, volume: np.ndarray, volume_path: Path) -> Path:
    from motionscore.cli import (
        PREDICTION_FIELDS,
        _index_row_from_prediction,
        _requested_model_storage_id,
        _session_output_paths,
        _upsert_index_rows,
    )
    from motionscore.dataset.layout import get_derivatives_root
    from motionscore.inference.model import ModelEnsemble
    from motionscore.inference.scoring import predict_scan
    from motionscore.review.preview import write_slice_profile_png
    from motionscore.review.store import initialize_or_update_review
    from motionscore.utils import to_relpath, utc_now_iso, write_tsv

    derivatives_root = get_derivatives_root(args.output_root, args.output_root).resolve()
    derivatives_root.mkdir(parents=True, exist_ok=True)
    session = _scene_session(args, volume_path)
    storage_model_id = "manual-only"
    model_version = "manual-only"
    resolved_model_id = "manual-only"
    prediction = None
    if args.manual_only:
        print("[scene] manual-only mode: skipping CNN inference")
    else:
        ensemble = ModelEnsemble(
            model_dir=None,
            model_root=Path(args.model_root).resolve() if args.model_root else None,
            model_id=args.model_id,
            device=args.device,
        )
        prediction = predict_scan(
            volume,
            ensemble=ensemble,
            stackheight=int(args.stackheight),
            slice_batch_size=int(args.slice_batch_size),
            slice_step=int(args.slice_step),
            retain_preprocessed=False,
        )
        storage_model_id = ensemble.resolved_model_id()
        resolved_model_id = storage_model_id
        model_version = ensemble.model_identity()
        print(f"[scene] using torch device={ensemble.model_device()}")

    paths = _session_output_paths(
        derivatives_root=derivatives_root,
        session=session,
        model_id=storage_model_id,
        scan_id=args.scan_id,
    )
    if prediction is None:
        prediction_row = {
            "scan_id": args.scan_id,
            "subject_id": args.subject_id,
            "raw_image_path": str(volume_path.resolve()),
            "preview_png_path": "",
            "slice_profile_png_path": "",
            "automatic_grade": "",
            "automatic_confidence": "",
            "manual_mode": "1",
            "model_id": resolved_model_id,
            "mean_confidence": "",
            "stack_ranges": "",
            "stack_grades": "",
            "stack_confidences": "",
            "slice_grades": "",
            "slice_confidences": "",
            "model_version": model_version,
            "predicted_at": utc_now_iso(),
        }
    else:
        prediction_row = {
            "scan_id": args.scan_id,
            "subject_id": args.subject_id,
            "raw_image_path": str(volume_path.resolve()),
            "preview_png_path": "",
            "slice_profile_png_path": "",
            "automatic_grade": str(prediction.automatic_grade),
            "automatic_confidence": str(prediction.automatic_confidence),
            "manual_mode": "0",
            "model_id": resolved_model_id,
            "mean_confidence": str(prediction.mean_confidence),
            "stack_ranges": json.dumps(prediction.stack_ranges),
            "stack_grades": json.dumps(prediction.stack_grades),
            "stack_confidences": json.dumps(prediction.stack_confidences),
            "slice_grades": json.dumps(prediction.slice_grades),
            "slice_confidences": json.dumps(prediction.slice_confidences),
            "model_version": model_version,
            "predicted_at": utc_now_iso(),
        }
        try:
            profile_path = write_slice_profile_png(
                prediction=prediction,
                output_path=paths["slice_profile_png"],
            )
            prediction_row["slice_profile_png_path"] = to_relpath(profile_path, derivatives_root)
        except Exception as exc:
            print(f"[scene] skipping slice profile PNG: {exc}", file=sys.stderr)

    write_tsv(paths["predictions_tsv"], [prediction_row], PREDICTION_FIELDS)
    initialize_or_update_review(
        review_tsv_path=paths["review_tsv"],
        review_json_path=paths["review_json"],
        review_audit_path=paths["review_audit"],
        prediction_rows=[prediction_row],
        confidence_threshold=int(args.confidence_threshold),
        training_mode=False,
    )
    _upsert_index_rows(
        derivatives_root,
        [
            _index_row_from_prediction(
                session=session,
                scan_id=args.scan_id,
                prediction_row=prediction_row,
                derivatives_root=derivatives_root,
                predictions_tsv=paths["predictions_tsv"],
                review_tsv=paths["review_tsv"],
                review_json=paths["review_json"],
                review_audit=paths["review_audit"],
                model_id=resolved_model_id,
                model_version=model_version,
            )
        ],
    )
    print(f"[scene] {args.scan_id} wrote MotionScore outputs under: {derivatives_root}")
    return derivatives_root


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    volume_path = Path(args.volume_npz).resolve()
    volume, metadata = _load_scene_npz(volume_path)
    print(
        f"[scene] loaded {volume_path.name}: shape={tuple(volume.shape)} "
        f"spacing={metadata.get('spacing', [])}"
    )
    _write_scene_prediction(args, volume, volume_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
