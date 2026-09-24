"""One-line summary of a text.

summary(text) -> "words=<n> unique=<u> avg_len=<a>" where avg_len is the mean word length rounded
to 2 decimals. An empty text gives "words=0 unique=0 avg_len=0.00".
"""
from .tokenize import tokenize


def summary(text):
    words = tokenize(text)
    unique = len(words)
    avg = sum(len(w) for w in words) / len(words)
    return f"words={len(words)} unique={unique} avg_len={avg:.2f}"
