"""Site detection must not guess unknown anatomy."""
import pytest


@pytest.mark.parametrize("metadata,texts,expected", [
    ({"processing_log": {"Site": "20"}}, ["voi-tibiaright"], "radius"),
    ({"processing_log_dict": {"Site": 38}}, [], "tibia"),
    ({"processing_log": "Site                         21\n"}, [], "radius"),
    ({"processing_log": {"Site": 0}}, ["INSR_148_DR_C1.AIM"], "radius"),
    ({}, ["sub-01_ses-01_voi-tibialeft_xct.AIM"], "tibia"),
    ({}, ["patella"], "knee"),
    ({}, ["D0000308"], None),
])
def test_site_detection_prefers_header_and_does_not_guess(metadata, texts, expected):
    from SlicerBoneImagingToolboxLib.contouring_presets import detect_site
    assert detect_site(metadata, texts) == expected
