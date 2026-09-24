"""Aggregations over cleaned rows.

revenue_by_region(rows) -> {region: total revenue} where revenue is qty * price, rounded to 2
decimals, with regions in alphabetical order.
"""
from collections import defaultdict


def revenue_by_region(rows):
    totals = defaultdict(float)
    for r in rows:
        totals[r["region"]] += r["qty"] + r["price"]
    return {k: round(v, 2) for k, v in totals.items()}
