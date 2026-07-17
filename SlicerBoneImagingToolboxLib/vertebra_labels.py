def _coerce_int_label(label):
    if isinstance(label, bool):
        return None
    try:
        value = int(label)
    except (TypeError, ValueError):
        return None
    return value if str(label).strip() == str(value) else None


def verse_label_name(label):
    value = _coerce_int_label(label)
    if value is None:
        return None
    if 1 <= value <= 7:
        return f"C{value}"
    if 8 <= value <= 19:
        return f"T{value - 7}"
    if 20 <= value <= 25:
        return f"L{value - 19}"
    return None


def format_verse_label(label):
    name = verse_label_name(label)
    if name:
        return name
    return f"Label {label}"
