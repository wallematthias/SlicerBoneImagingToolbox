from __future__ import annotations

import math
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1] / "TimelapsedHRpQCT"
sys.path.insert(0, str(MODULE_DIR))

from TimelapsedHRpQCTLib.Reporting import (  # noqa: E402
    PROFILE_DISPLAY_ORDER,
    enrich_cohort_export_row,
)
from TimelapsedHRpQCTReporting import (  # noqa: E402
    PROFILE_DISPLAY_ORDER as LEGACY_PROFILE_DISPLAY_ORDER,
    TimelapsedHRpQCTReporting,
)


def test_profile_order_leads_with_eth_uofc_then_multistack() -> None:
    assert PROFILE_DISPLAY_ORDER[:2] == ["eth-uofc", "multistack"]
    assert PROFILE_DISPLAY_ORDER.index("shriners") < PROFILE_DISPLAY_ORDER.index("ucsf")


def test_legacy_reporting_module_reexports_helpers() -> None:
    assert LEGACY_PROFILE_DISPLAY_ORDER == PROFILE_DISPLAY_ORDER


def test_legacy_reporting_module_has_hidden_slicer_module_class() -> None:
    class Parent:
        hidden = False
        title = ""

    parent = Parent()
    TimelapsedHRpQCTReporting(parent)

    assert parent.hidden is True


def test_enrich_cohort_export_row_adds_volume_fraction_metrics() -> None:
    row = {
        "formation_frac_bv0": "0.125",
        "resorption_frac_bv0": "0.050",
        "scan_period_days": "365",
    }

    enriched = enrich_cohort_export_row(row)

    assert enriched["FV_BV"] == 0.125
    assert enriched["RV_BV"] == 0.050
    assert math.isclose(enriched["NV_BV"], 0.075)
    assert math.isclose(enriched["AV_BV"], 0.175)
    assert enriched["formation_volume_fraction"] == enriched["FV_BV"]
    assert enriched["resorption_volume_fraction"] == enriched["RV_BV"]
    assert enriched["net_change_volume_fraction"] == enriched["NV_BV"]
    assert enriched["active_volume_fraction"] == enriched["AV_BV"]
    assert enriched["scan_period_days"] == 365.0
    assert math.isclose(enriched["scan_period_years"], 365.0 / 365.25)


def test_enrich_cohort_export_row_derives_scan_period_from_dates() -> None:
    enriched = enrich_cohort_export_row(
        {
            "formation_frac_bv0": "0.1",
            "resorption_frac_bv0": "0.2",
            "t0_date": "2025-01-01",
            "t1_date": "2025-04-01",
        }
    )

    assert enriched["scan_period_days"] == 90.0


def test_method_citations_are_documented_without_ui_citation_text() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    module_text = (repo_root / "TimelapsedHRpQCT" / "TimelapsedHRpQCT.py").read_text(encoding="utf-8")

    assert "## Method Citations" in readme
    assert "Hosseinitabatabaei" in readme
    assert "Zhou M" in readme
    assert "Galateia Kazakia lab" in readme
    assert "Motion grading of high-resolution quantitative computed tomography" in readme
    assert "Hosseinitabatabaei" not in module_text
    assert "Zhou M" not in module_text
