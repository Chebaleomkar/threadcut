"""One generation of Conway's rules.

A live cell with 2 or 3 live neighbours survives; a dead cell with exactly 3 becomes alive; all
other cells are dead in the next generation.
"""
from .grid import Grid


def step(g: Grid):
    nxt = set()
    for r in range(g.rows):
        for c in range(g.cols):
            n = g.neighbours(r, c)
            if (r, c) in g.alive and n in (2, 3):
                nxt.add((r, c))
            elif n == 3:
                nxt.add((r, c))
    return Grid(g.rows, g.cols, nxt)
