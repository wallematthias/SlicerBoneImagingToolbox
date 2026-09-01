from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import SimpleITK as sitk


def _is_aim_path(path: Path) -> bool:
    return ".aim" in path.name.lower()


def _read_image(path: Path):
    if _is_aim_path(path):
        from ScancoIOLib import aim_io

        return aim_io.read_aim(path, scaling="density")
    return sitk.ReadImage(str(path)), {}


def _read_segmentation_source(path: Path, method: str):
    if method == "laplace_hamming" and _is_aim_path(path):
        from ScancoIOLib import aim_io

        image, metadata = aim_io.read_aim(path, scaling="native")
        return sitk.Cast(image, sitk.sitkInt16), metadata
    return None, {}


def _write_mask_aim(mask_image, source_path: Path, output_path: Path, source_metadata: dict, role: str) -> str:
    from ScancoIOLib import aim_io

    metadata = dict(source_metadata or {})
    metadata["source_file"] = str(source_path)
    metadata["mask_role"] = str(role)
    aim_io.write_aim(
        sitk.Cast(mask_image > 0, sitk.sitkUInt8),
        output_path,
        metadata=metadata,
        unit="native",
        mask=True,
    )
    return str(output_path)


def _sidecar_path(mask_path: Path) -> Path:
    if mask_path.name.lower().endswith(".nii.gz"):
        return mask_path.with_name(mask_path.name[:-7] + ".json")
    return mask_path.with_suffix(".json")


def _write_sidecar(mask_path: Path, *, role: str, config: dict, metadata: dict, source_metadata: dict) -> str:
    sidecar_path = _sidecar_path(mask_path)
    sidecar = {
        "schema": "bone-contour-mask-provenance-v1",
        "role": role,
        "mask_path": str(mask_path),
        "source_image": str(config["image_path"]),
        "site": str(config["site"]),
        "segmentation_method": str(config["segmentation_method"]),
        "periosteal_contour_method": str(config["periosteal_contour_method"]),
        "endosteal_contour_method": str(config["endosteal_contour_method"]),
        "output_format": str(metadata.get("output_format") or config.get("output_format") or "auto"),
        "parameters": dict(config.get("params") or {}),
        "algorithm_metadata": dict(metadata or {}),
        "source_metadata": dict(source_metadata or {}),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(sidecar_path)


def _parameters(site: str, segmentation_method: str, periosteal_method: str, endosteal_method: str, params: dict):
    from bone_contouring import (
        ContourParameters,
        InnerContourParameters,
        OuterContourParameters,
        SegmentationParameters,
    )

    params = dict(params or {})
    segmentation = dict(params.get("segmentation", {}))
    outer = dict(params.get("outer", {}))
    inner = dict(params.get("inner", {}))
    geodesic = dict(params.get("geodesic", {}))
    package_segmentation_method = {
        "seg_gauss": "gauss",
        "laplace_hamming": "laplace_hamming",
        "adaptive": "adaptive",
        "none": "gauss",
    }.get(segmentation_method, segmentation_method)
    outer_method = "geodesic" if periosteal_method == "geodesic_fracture" else periosteal_method

    return ContourParameters(
        modality=str(params.get("modality", "xct2")),
        site=str(site),
        outer=OuterContourParameters(
            contour_method=outer_method,
            periosteal_threshold=float(outer.get("periosteal_threshold", 300.0)),
            periosteal_kernel_size=int(outer.get("periosteal_kernel_size", outer.get("periosteal_kernelsize", 5))),
            periosteal_open_radius=int(outer.get("periosteal_open_radius", outer.get("periosteal_openradius", 2))),
            gaussian_sigma=float(outer.get("gaussian_sigma", 1.5)),
            use_adaptive_threshold=bool(outer.get("use_adaptive_threshold", False)),
            fill_holes=bool(outer.get("fill_holes", True)),
            geodesic_bone_threshold=float(geodesic.get("bone_threshold", outer.get("geodesic_bone_threshold", 250.0))),
            geodesic_fill_holes=bool(geodesic.get("fill_holes", outer.get("geodesic_fill_holes", True))),
        ),
        inner=InnerContourParameters(
            contour_method=endosteal_method,
            site=str(site),
            endosteal_threshold=float(inner.get("endosteal_threshold", 500.0)),
            endosteal_kernel_size=int(inner.get("endosteal_kernel_size", inner.get("endosteal_kernelsize", 3))),
            gaussian_sigma=float(inner.get("gaussian_sigma", 1.5)),
            use_adaptive_threshold=bool(inner.get("use_adaptive_threshold", False)),
            peel=int(inner.get("peel", 3)),
            trabecular_close_radius=inner.get("trabecular_close_radius"),
        ),
        segmentation=SegmentationParameters(
            enabled=segmentation_method != "none",
            method=package_segmentation_method,
            gaussian_sigma=float(segmentation.get("gaussian_sigma", 0.8)),
            trab_threshold=float(segmentation.get("trab_threshold", 320.0)),
            cort_threshold=float(segmentation.get("cort_threshold", 450.0)),
            adaptive_low_threshold=float(segmentation.get("adaptive_low_threshold", 100.0)),
            adaptive_high_threshold=float(segmentation.get("adaptive_high_threshold", 300.0)),
            adaptive_block_size=int(segmentation.get("adaptive_block_size", 13)),
            min_size_voxels=int(segmentation.get("min_size_voxels", 64)),
            keep_largest_component=bool(segmentation.get("keep_largest_component", True)),
            laplace_hamming_low_pass_cutoff=float(segmentation.get("laplace_hamming_low_pass_cutoff", 0.3)),
            laplace_hamming_threshold=float(segmentation.get("laplace_hamming_threshold", 15564.0)),
            laplace_hamming_epsilon=float(segmentation.get("laplace_hamming_epsilon", 0.45)),
            laplace_hamming_min_size_voxels=int(segmentation.get("laplace_hamming_min_size_voxels", 70)),
            laplace_hamming_backend=str(segmentation.get("laplace_hamming_backend", "cpu")),
            use_segmentation_aligned_contour_support=bool(
                segmentation.get("use_segmentation_aligned_contour_support", False)
            ),
        ),
    )


def run(config: dict) -> dict:
    from bone_contouring import generate_masks_from_image

    image_path = Path(config["image_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(config["output_prefix"])
    output_format = str(config.get("output_format") or "auto").lower()
    segmentation_method = str(config["segmentation_method"])
    periosteal_method = str(config["periosteal_contour_method"])
    endosteal_method = str(config["endosteal_contour_method"])
    is_aim = _is_aim_path(image_path)
    if output_format not in {"auto", "aim", "nifti"}:
        raise ValueError("Output format must be one of: auto, aim, nifti.")
    write_aim = output_format == "aim" or (output_format == "auto" and is_aim)
    if write_aim and not is_aim:
        raise ValueError("AIM output requires an AIM input so source scanner metadata can be preserved.")

    print(f"[worker] reading image: {image_path.name}", flush=True)
    image, image_metadata = _read_image(image_path)
    print("[worker] preparing segmentation source", flush=True)
    segmentation_image, native_metadata = _read_segmentation_source(image_path, segmentation_method)
    contour_params = _parameters(
        str(config["site"]),
        segmentation_method,
        periosteal_method,
        endosteal_method,
        dict(config.get("params") or {}),
    )
    print("[worker] generating masks", flush=True)
    generated = generate_masks_from_image(image, contour_params, segmentation_image=segmentation_image)
    if segmentation_method == "none":
        generated.seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        generated.seg.CopyInformation(image)
    if periosteal_method == "none":
        generated.full = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        generated.full.CopyInformation(image)
    if endosteal_method != "standard":
        generated.trab = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        generated.trab.CopyInformation(image)
        generated.cort = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        generated.cort.CopyInformation(image)

    roles = ["full", "trab", "cort", "seg"]
    if periosteal_method == "none":
        roles.remove("full")
    if endosteal_method != "standard":
        roles = [role for role in roles if role in {"full", "seg"}]

    written = {}
    sidecars = {}
    source_metadata = native_metadata or image_metadata
    print(f"[worker] writing {len(roles)} {('AIM' if write_aim else 'NIfTI')} mask(s)", flush=True)
    for role in roles:
        mask_image = getattr(generated, role)
        if write_aim:
            output_path = output_dir / f"{output_prefix}_mask-{role}.AIM"
            written[role] = _write_mask_aim(
                mask_image,
                image_path,
                output_path,
                source_metadata,
                role,
            )
        else:
            output_path = output_dir / f"{output_prefix}_mask-{role}.nii.gz"
            sitk.WriteImage(sitk.Cast(mask_image > 0, sitk.sitkUInt8), str(output_path))
            written[role] = str(output_path)
        sidecars[role] = _write_sidecar(
            Path(written[role]),
            role=role,
            config=config,
            metadata={
                **generated.metadata,
                "emitted_roles": roles,
                "output_format": "aim" if write_aim else "nifti",
            },
            source_metadata=source_metadata,
        )
    return {
        "written": written,
        "sidecars": sidecars,
        "metadata": {
            **generated.metadata,
            "emitted_roles": roles,
            "output_format": "aim" if write_aim else "nifti",
            "aim_outputs": written if write_aim else {},
            "provenance_sidecars": sidecars,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run(payload)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
