"""Plain-text stock report.

format_money(1250) -> "$12.50". Report lines are sorted by item name and look like:
"Widget (W1): 3 x $2.50 = $7.50". The last line is "TOTAL: $<sum>".
"""
from .store import Store


def format_money(cents):
    return f"${cents / 100}"


def report(store: Store):
    lines = []
    for item in sorted(store.items.values(), key=lambda i: i.sku):
        lines.append(f"{item.name} ({item.sku}): {item.qty} x {format_money(item.price_cents)}"
                     f" = {format_money(item.total_cents())}")
    lines.append(f"TOTAL: {format_money(store.value_cents())}")
    return "\n".join(lines)
