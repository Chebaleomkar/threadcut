"""Render a grid as text: one line per row, '#' for live cells and '.' for dead ones."""
from .grid import Grid


def render(g: Grid):
    return "\n".join("".join("#" if (c, r) in g.alive else "." for c in range(g.cols)) for r in range(g.rows))
