from textstats.tokenize import tokenize
from textstats.counts import counts, top_n
from textstats.summary import summary


def test_tokenize_apostrophes_and_case():
    assert tokenize("Don't stop, DON'T!") == ["don't", "stop", "don't"]


def test_tokenize_digits():
    assert tokenize("R2-D2 v2.0") == ["r2", "d2", "v2", "0"]


def test_counts():
    assert counts("a A b")["a"] == 2


def test_top_n():
    text = "the cat and the dog and the cat bird"
    assert top_n(text, 2) == [("cat", 2), ("bird", 1)]


def test_summary():
    assert summary("aa bb aa") == "words=3 unique=2 avg_len=2.00"


def test_summary_empty():
    assert summary("") == "words=0 unique=0 avg_len=0.00"
