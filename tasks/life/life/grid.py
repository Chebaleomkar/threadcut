"""A toroidal grid for Conway's Game of Life.

Cells are addressed as (row, col). The grid wraps around on both axes, so the neighbour of the
last column is the first column. neighbours(r, c) counts live cells among the 8 surrounding cells
(never the cell itself).
"""


class Grid:
    def __init__(self, rows, cols, alive=()):
        self.rows, self.cols = rows, cols
        self.alive = set(alive)

    def neighbours(self, r, c):
        n = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in self.alive:
                    n += 1
        return n
