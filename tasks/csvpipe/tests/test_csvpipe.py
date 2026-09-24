from csvpipe.parse import parse
from csvpipe.clean import clean
from csvpipe.aggregate import revenue_by_region

DATA = """region,item,qty,price
west,Rivet,2,1.5
north east, "Bolt, M4", 10, 0.25

south,Nut,4,0.10
north east,Washer,0,1.00
south,Screw,x,0.50
west,Pin,,0.05
west,Clip,3,
"""


def test_parse_quotes_and_blanks():
    rows = parse(DATA)
    assert len(rows) == 7
    assert rows[1]["item"] == "Bolt, M4"
    assert rows[1]["region"] == "north east"


def test_clean():
    rows = clean(parse(DATA))
    assert [r["item"] for r in rows] == ["Rivet", "Bolt, M4", "Nut"]
    assert rows[1]["region"] == "North East"


def test_revenue():
    rows = clean(parse(DATA))
    assert revenue_by_region(rows) == {"North East": 2.5, "South": 0.4, "West": 3.0}
    assert list(revenue_by_region(rows)) == ["North East", "South", "West"]
