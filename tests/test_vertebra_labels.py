import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SlicerBoneImagingToolboxLib.vertebra_labels import format_verse_label, verse_label_name


def test_verse_label_name_maps_cervical_thoracic_and_lumbar_levels() -> None:
    assert verse_label_name(1) == "C1"
    assert verse_label_name(7) == "C7"
    assert verse_label_name(8) == "T1"
    assert verse_label_name(19) == "T12"
    assert verse_label_name(20) == "L1"
    assert verse_label_name(25) == "L6"


def test_format_verse_label_preserves_unknown_labels() -> None:
    assert format_verse_label(26) == "Label 26"
    assert format_verse_label("20") == "L1"
    assert format_verse_label("custom") == "Label custom"
