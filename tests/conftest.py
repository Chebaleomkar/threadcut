import os
import sys

# Project-local extras (e.g. triton-windows) live in models/.pylib, outside the global environment.
_lib = os.path.join(os.path.dirname(__file__), "..", "models", ".pylib")
if os.path.isdir(_lib):
    sys.path.insert(0, os.path.abspath(_lib))
