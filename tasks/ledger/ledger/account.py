"""Bank account with an overdraft limit. Amounts are integer cents.

withdraw(amount) is allowed while the balance stays >= -overdraft_limit; otherwise it raises
InsufficientFunds and leaves the balance unchanged. deposit/withdraw reject amounts <= 0 with
ValueError. Every successful operation is appended to history as (kind, amount).
"""


class InsufficientFunds(Exception):
    pass


class Account:
    def __init__(self, owner, balance=0, overdraft_limit=0):
        self.owner = owner
        self.balance = balance
        self.overdraft_limit = overdraft_limit
        self.history = []

    def deposit(self, amount):
        if amount < 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        self.history.append(("deposit", amount))

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        if self.balance - amount < self.overdraft_limit:
            raise InsufficientFunds(self.owner)
        self.balance -= amount
        self.history.append(("withdraw", amount))
