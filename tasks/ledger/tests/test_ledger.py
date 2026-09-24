import pytest
from ledger.account import Account, InsufficientFunds
from ledger.transfer import fee, transfer


def test_deposit_rejects_zero():
    with pytest.raises(ValueError):
        Account("a").deposit(0)


def test_overdraft_allowed_to_limit():
    a = Account("a", balance=100, overdraft_limit=50)
    a.withdraw(150)
    assert a.balance == -50


def test_overdraft_exceeded():
    a = Account("a", balance=100, overdraft_limit=50)
    with pytest.raises(InsufficientFunds):
        a.withdraw(151)
    assert a.balance == 100


def test_fee_rounds_half_up():
    assert fee(100, 150) == 2   # 1.5 -> 2
    assert fee(300, 150) == 5   # 4.5 -> 5
    assert fee(1000, 150) == 15


def test_transfer_atomic():
    a, b = Account("a", 100), Account("b", 0)
    with pytest.raises(InsufficientFunds):
        transfer(a, b, 100, rate_bp=150)
    assert (a.balance, b.balance) == (100, 0)


def test_transfer_ok():
    a, b = Account("a", 1000), Account("b", 0)
    transfer(a, b, 300, rate_bp=150)
    assert (a.balance, b.balance) == (695, 300)
