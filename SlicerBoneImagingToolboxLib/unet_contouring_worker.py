"""Headless scene adapter; model and tissue algorithms belong to bone-contouring."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    from bone_contouring.unet.inference import Segmenter
    from bone_contouring import generate_bone_segmentation, resolve_preset, SegmentationParameters

    with np.load(args.input, allow_pickle=False) as snapshot:
        masks = Segmenter(args.device).segment(snapshot["density"], progress=lambda text: print(text, flush=True))
        settings = json.loads(str(snapshot["segmentation"]))
        if settings is not None:
            print("Generating independent bone-tissue SEG...", flush=True)
            image = sitk.GetImageFromArray(snapshot["tissue_image"])
            image.SetSpacing(tuple(float(value) for value in snapshot["spacing"]))
            images = {}
            for role in ("full", "trab", "cort"):
                mask = sitk.GetImageFromArray(masks[role].astype(np.uint8))
                mask.CopyInformation(image)
                images[role + "_mask"] = mask
            parameters = replace(resolve_preset(), segmentation=SegmentationParameters(**settings))
            masks["seg"] = sitk.GetArrayFromImage(generate_bone_segmentation(image, parameters, **images))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **masks)
    print(f"Completed: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
