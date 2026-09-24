"""Tokenizer for plain English text.

Words are maximal runs of letters, digits or apostrophes. Everything is lowercased. A word made of
apostrophes only is not a word. Examples:
    tokenize("Don't stop, DON'T!") -> ["don't", "stop", "don't"]
    tokenize("R2-D2 v2.0") -> ["r2", "d2", "v2", "0"]
"""
import re

WORD = re.compile(r"[A-Za-z']+")


def tokenize(text):
    return [w for w in WORD.findall(text) if w.strip("'")]
