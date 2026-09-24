"""In-memory store of items keyed by SKU.

Rules:
- Adding an existing SKU increases its quantity; the stored name and price are kept.
- Removing more units than are in stock raises ValueError and changes nothing.
- An item whose quantity reaches exactly zero is deleted from the store.
- low_stock(threshold) returns the SKUs at or below the threshold, sorted alphabetically.
"""
from .models import Item


class Store:
    def __init__(self):
        self.items = {}

    def add(self, item: Item):
        if item.sku in self.items:
            self.items[item.sku].qty += item.qty
        else:
            self.items[item.sku] = Item(item.sku, item.name, item.price_cents, item.qty)

    def remove(self, sku, qty):
        item = self.items[sku]
        if qty > item.qty:
            raise ValueError("not enough stock")
        item.qty -= qty
        if item.qty < 0:
            del self.items[sku]

    def value_cents(self):
        return sum(i.total_cents() for i in self.items.values())

    def low_stock(self, threshold):
        return sorted(sku for sku, i in self.items.items() if i.qty < threshold)
