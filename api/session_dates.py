"""
api/session_dates.py
────────────────────
Infer the date a session took place from its recording URL, so batch uploads
of past sessions keep their real chronology (trends depend on it).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_NAMED_MONTH_RE = re.compile(r"/(20\d{2})/([A-Za-z]{3})[a-z]*/(\d{1,2})/")
_NUMERIC_RE = re.compile(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})")


def infer_session_date(url: Optional[str]) -> Optional[str]:
    """Pull a date out of a URL, e.g. '/2026/Sep/21/' or '2026-09-21'. None if absent."""
    if not url:
        return None
    m = _NAMED_MONTH_RE.search(url)
    if m and m.group(2).lower() in _MONTHS:
        y, mon, d = int(m.group(1)), _MONTHS[m.group(2).lower()], int(m.group(3))
    else:
        m = _NUMERIC_RE.search(url)
        if not m:
            return None
        y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mon, d).isoformat()
    except ValueError:
        return None
