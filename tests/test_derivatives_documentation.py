from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_derivatives_documentation_describes_shared_contract():
    text = (ROOT / "docs" / "derivatives.md").read_text(encoding="utf-8")

    assert "Registration" in text
    assert "CommonRegion" in text
    assert "scan_region_native_common" in text
    assert "Scene mode" in text
    assert "Batch mode" in text
    assert "dependency generation" in text
    assert "FEA" in text
    assert "Mechanoregulation" in text
    assert "VoidSpace" in text


def test_readme_links_derivative_contract():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/derivatives.md" in readme
    assert "derivative workflow contract" in readme
