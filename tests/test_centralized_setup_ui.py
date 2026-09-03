from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PUBLIC_MODULES = (
    ROOT / "IOTools" / "ScancoIO" / "ScancoIO.py",
    ROOT / "HRpQCTTools" / "MotionScoreHRpQCT" / "MotionScoreHRpQCT.py",
    ROOT / "HRpQCTTools" / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py",
    ROOT / "HRpQCTTools" / "MechanoregulationHRpQCT" / "MechanoregulationHRpQCT.py",
    ROOT / "HRpQCTTools" / "BoneMicroarchitecture" / "BoneMicroarchitecture.py",
    ROOT / "HRpQCTTools" / "PlateRodMorphometryHRpQCT" / "PlateRodMorphometryHRpQCT.py",
    ROOT / "HRpQCTTools" / "ParOSolFEA" / "ParOSolFEA.py",
    ROOT / "CTTools" / "SpineSegmentationCT" / "SpineSegmentationCT.py",
)


def test_public_modules_do_not_expose_local_install_or_update_buttons() -> None:
    forbidden = (
        "Check toolbox updates",
        "Check Toolbox Updates",
        "Install / update",
        "Install / Update",
        "Install from PyPI",
        "Install Conda MPS Runtime",
        "Install / Download Models",
    )

    for path in PUBLIC_MODULES:
        source = path.read_text(encoding="utf-8")
        assert not any(text in source for text in forbidden), path


def test_mechanoregulation_and_spine_have_public_icons() -> None:
    mechreg = ROOT / "HRpQCTTools" / "MechanoregulationHRpQCT" / "Resources" / "Icons" / "MechanoregulationHRpQCT.png"
    spine = ROOT / "CTTools" / "SpineSegmentationCT" / "Resources" / "Icons" / "SpineSegmentationCT.png"

    assert mechreg.is_file()
    assert spine.is_file()
