import pytest
from inventory.models import Item
from inventory.store import Store
from inventory.report import format_money, report


def make():
    s = Store()
    s.add(Item("W1", "Widget", 250, 3))
    s.add(Item("A9", "Zapper", 1000, 1))
    s.add(Item("G2", "Gadget", 199, 10))
    return s


def test_total_cents():
    assert Item("x", "x", 250, 3).total_cents() == 750


def test_remove_to_zero_deletes():
    s = make()
    s.remove("A9", 1)
    assert "A9" not in s.items


def test_remove_too_many():
    s = make()
    with pytest.raises(ValueError):
        s.remove("W1", 4)
    assert s.items["W1"].qty == 3


def test_low_stock_inclusive():
    assert make().low_stock(3) == ["A9", "W1"]


def test_format_money():
    assert format_money(1250) == "$12.50"
    assert format_money(5) == "$0.05"


def test_report():
    assert report(make()).splitlines() == [
        "Gadget (G2): 10 x $1.99 = $19.90",
        "Widget (W1): 3 x $2.50 = $7.50",
        "Zapper (A9): 1 x $10.00 = $10.00",
        "TOTAL: $37.40",
    ]
