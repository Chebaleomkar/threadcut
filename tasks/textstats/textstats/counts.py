"""Word frequency helpers built on the tokenizer.

top_n(text, n) returns the n most frequent words as (word, count) pairs, highest count first;
ties are broken alphabetically. Stop words (see STOP) are ignored by top_n only.
"""
from collections import Counter

from .tokenize import tokenize

STOP = {"the", "a", "an", "and", "of", "to", "in"}


def counts(text):
    return Counter(tokenize(text))


def top_n(text, n):
    c = Counter(w for w in tokenize(text) if w not in STOP)
    return sorted(c.items(), key=lambda kv: (kv[1], kv[0]))[:n]
