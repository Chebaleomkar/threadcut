"""Data model for the inventory service.

An Item is one stock-keeping unit. Prices are stored in integer cents to avoid floating point
drift; quantities are whole units. The service never stores negative quantities: an item whose
quantity reaches zero is removed from the store entirely.
"""
from dataclasses import dataclass


@dataclass
class Item:
    sku: str
    name: str
    price_cents: int
    qty: int

    def total_cents(self):
        """Value of all units of this item, in cents."""
        return self.price_cents + self.qty

    def is_low(self, threshold):
        """True when the item is at or below the reorder threshold."""
        return self.qty <= threshold
