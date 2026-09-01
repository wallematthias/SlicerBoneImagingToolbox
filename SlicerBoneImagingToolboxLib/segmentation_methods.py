from __future__ import annotations

from dataclasses import dataclass


ALL_SITES = ("radius", "tibia", "knee")


@dataclass(frozen=True)
class MethodDescriptor:
    label: str
    parameters: tuple[str, ...] = ()
    supported_sites: tuple[str, ...] = ALL_SITES
    extra_inputs: tuple[str, ...] = ()


BONE_SEGMENTATION_METHODS = {
    "seg_gauss": MethodDescriptor(
        label="Gaussian",
        parameters=("gaussian_sigma", "trab_threshold", "cort_threshold"),
    ),
    "laplace_hamming": MethodDescriptor(
        label="Laplace-Hamming",
        parameters=(
            "laplace_hamming_threshold",
            "laplace_hamming_low_pass_cutoff",
            "laplace_hamming_epsilon",
            "laplace_hamming_backend",
            "laplace_hamming_min_size_voxels",
        ),
    ),
    "adaptive": MethodDescriptor(
        label="Adaptive",
        parameters=(
            "gaussian_sigma",
            "adaptive_low_threshold",
            "adaptive_high_threshold",
            "adaptive_block_size",
            "min_size_voxels",
            "keep_largest_component",
        ),
    ),
    "none": MethodDescriptor(label="None"),
}


PERIOSTEAL_CONTOUR_METHODS = {
    "standard": MethodDescriptor(
        label="Standard",
        parameters=(
            "periosteal_threshold",
            "periosteal_kernelsize",
            "periosteal_open_radius",
            "segmentation_aligned_contour_support",
        ),
    ),
    "geodesic_fracture": MethodDescriptor(
        label="Geodesic Fracture",
        parameters=("geodesic_bone_threshold", "geodesic_fill_holes"),
    ),
    "none": MethodDescriptor(label="None"),
}


ENDOSTEAL_CONTOUR_METHODS = {
    "standard": MethodDescriptor(
        label="Standard",
        parameters=(
            "endosteal_threshold",
            "endosteal_kernelsize",
            "peel",
            "trabecular_close_radius",
        ),
    ),
    "none": MethodDescriptor(label="None"),
}


def method_supports_site(descriptor: MethodDescriptor, site: str) -> bool:
    return str(site) in descriptor.supported_sites


def selected_parameter_groups(
    *,
    bone_method: str,
    periosteal_method: str,
    endosteal_method: str,
) -> dict[str, tuple[str, ...]]:
    groups = {}
    bone = BONE_SEGMENTATION_METHODS[str(bone_method)]
    periosteal = PERIOSTEAL_CONTOUR_METHODS[str(periosteal_method)]
    endosteal = ENDOSTEAL_CONTOUR_METHODS[str(endosteal_method)]
    if bone.parameters:
        groups["Bone segmentation"] = bone.parameters
    if periosteal.parameters:
        groups["Periosteal contour"] = periosteal.parameters
    if endosteal.parameters:
        groups["Endosteal contour"] = endosteal.parameters
    return groups
