from __future__ import annotations

from datetime import date, datetime
from math import isfinite


PROFILE_DISPLAY_ORDER = [
    "eth-uofc",
    "multistack",
    "standard",
    "single-stack",
    "low-memory",
    "xct1-standard",
    "shriners",
    "ucsf",
]


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if isfinite(out) else float("nan")


def _first_numeric(row, names):
    for name in names:
        if name in row and str(row.get(name, "")).strip():
            value = _as_float(row.get(name))
            if isfinite(value):
                return value
    return float("nan")


def _parse_date(value):
    text = str(value or "").strip()
    if not text:
        return None
    for parser in (date.fromisoformat, datetime.fromisoformat):
        try:
            parsed = parser(text)
            return parsed.date() if isinstance(parsed, datetime) else parsed
        except ValueError:
            pass
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def scan_period_days_from_row(row):
    days = _first_numeric(
        row,
        (
            "scan_period_days",
            "scan_interval_days",
            "interval_days",
            "days_between_scans",
            "time_between_scans_days",
            "followup_days",
        ),
    )
    if isfinite(days):
        return days

    years = _first_numeric(
        row,
        (
            "scan_period_years",
            "scan_interval_years",
            "interval_years",
            "years_between_scans",
            "time_between_scans_years",
            "followup_years",
        ),
    )
    if isfinite(years):
        return years * 365.25

    t0_date = None
    t1_date = None
    for t0_name, t1_name in (
        ("t0_date", "t1_date"),
        ("scan_date_t0", "scan_date_t1"),
        ("scan0_date", "scan1_date"),
        ("baseline_scan_date", "followup_scan_date"),
    ):
        t0_date = _parse_date(row.get(t0_name))
        t1_date = _parse_date(row.get(t1_name))
        if t0_date and t1_date:
            break
    if t0_date and t1_date:
        return float((t1_date - t0_date).days)
    return float("nan")


def enrich_cohort_export_row(row):
    enriched = dict(row)
    fv_bv = _as_float(row.get("formation_frac_bv0", row.get("FV_BV")))
    rv_bv = _as_float(row.get("resorption_frac_bv0", row.get("RV_BV")))
    nv_bv = fv_bv - rv_bv if isfinite(fv_bv) and isfinite(rv_bv) else float("nan")
    av_bv = fv_bv + rv_bv if isfinite(fv_bv) and isfinite(rv_bv) else float("nan")

    enriched["FV_BV"] = fv_bv
    enriched["RV_BV"] = rv_bv
    enriched["NV_BV"] = nv_bv
    enriched["AV_BV"] = av_bv
    enriched["formation_volume_fraction"] = fv_bv
    enriched["resorption_volume_fraction"] = rv_bv
    enriched["net_change_volume_fraction"] = nv_bv
    enriched["active_volume_fraction"] = av_bv

    days = scan_period_days_from_row(row)
    enriched["scan_period_days"] = days if isfinite(days) else ""
    enriched["scan_period_years"] = days / 365.25 if isfinite(days) else ""
    return enriched
