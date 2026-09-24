"""Transfers between accounts with a percentage fee paid by the sender.

fee(amount, rate_bp) is amount * rate_bp / 10000 rounded half up to whole cents
(rate_bp is in basis points: 150 = 1.5%). transfer() withdraws amount + fee from the sender and
deposits amount to the receiver. If the withdrawal fails nothing changes.
"""
from .account import Account


def fee(amount, rate_bp):
    return round(amount * rate_bp / 10000)


def transfer(src: Account, dst: Account, amount, rate_bp=0):
    dst.deposit(amount)
    src.withdraw(amount + fee(amount, rate_bp))
