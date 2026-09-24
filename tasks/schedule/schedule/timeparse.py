"""Parse clock times like "9:05", "09:05", "9am", "12pm", "12am", "7:30pm" into minutes after
midnight. 12am is midnight (0) and 12pm is noon (720). Invalid input raises ValueError.
"""
import re

PAT = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$")


def to_minutes(s):
    m = PAT.match(s.strip().lower())
    if not m:
        raise ValueError(s)
    h, mm, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm":
        h += 12
    if h > 23 or mm > 59:
        raise ValueError(s)
    return h * 60 + mm
