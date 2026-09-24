"""Meetings and conflict detection.

A meeting occupies [start, end) in minutes. Two meetings conflict when they overlap for at least
one minute; back-to-back meetings (one ends when the next starts) do not conflict. conflicts()
returns pairs of titles (earlier-starting first) in order of the first meeting's start time.
"""
from dataclasses import dataclass

from .timeparse import to_minutes


@dataclass
class Meeting:
    title: str
    start: int
    end: int

    @classmethod
    def parse(cls, title, start, end):
        return cls(title, to_minutes(start), to_minutes(end))


def overlaps(a, b):
    return a.start <= b.end and b.start <= a.end


def conflicts(meetings):
    ms = sorted(meetings, key=lambda m: m.start)
    out = []
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            if overlaps(a, b):
                out.append((b.title, a.title))
    return out
