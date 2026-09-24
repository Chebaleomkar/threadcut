from life.grid import Grid
from life.rules import step
from life.render import render


def test_neighbours_exclude_self():
    g = Grid(5, 5, {(2, 2), (2, 3)})
    assert g.neighbours(2, 2) == 1


def test_neighbours_wrap():
    g = Grid(4, 4, {(0, 0)})
    assert g.neighbours(3, 3) == 1


def test_blinker():
    g = Grid(5, 5, {(2, 1), (2, 2), (2, 3)})
    assert step(g).alive == {(1, 2), (2, 2), (3, 2)}


def test_render_rows_and_cols():
    g = Grid(2, 3, {(0, 2), (1, 0)})
    assert render(g) == "..#\n#.."
