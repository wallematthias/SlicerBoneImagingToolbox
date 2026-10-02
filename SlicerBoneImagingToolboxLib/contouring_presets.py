"""Conservative site detection for the Contouring UI (no Slicer imports)."""
import re


def _site_category(value):
    try:
        from bone_imaging_derivatives import normalize_site
        normalized = str(normalize_site(str(value or "")) or "")
    except ImportError:
        normalized = ""  # Setup may not have installed runtime packages yet.
    for site in ("radius", "tibia", "knee"):
        if normalized.startswith(site):
            return site
    text = str(value or "").lower()
    for site in ("radius", "tibia", "knee"):
        if site in text:
            return site
    tokens = re.split(r"[^a-z0-9]+", text)
    for token in tokens:
        category = {"dr": "radius", "rl": "radius", "rr": "radius",
                    "dt": "tibia", "tl": "tibia", "tr": "tibia", "patella": "knee"}.get(token)
        if category:
            return category
    return None


def detect_site(metadata, texts=()):
    """Prefer the Scanco Site code; filenames are only a recovery path."""
    candidates = []
    for log_key in ("processing_log", "processing_log_dict", "processing_log_raw"):
        log = metadata.get(log_key)
        if isinstance(log, dict):
            candidates.extend(value for key, value in log.items() if str(key).strip().lower() in {"site", "scan site"})
        elif isinstance(log, str):
            match = re.search(r"^\s*Site\s*[:=]?\s+(\d+)\b", log, re.MULTILINE | re.IGNORECASE)
            if match:
                candidates.append(match.group(1))
    candidates.extend(metadata.get(key) for key in ("site", "scan_site", "Site"))
    for value in (*candidates, *texts):
        site = _site_category(value)
        if site:
            return site
    return None
