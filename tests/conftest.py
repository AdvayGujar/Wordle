"""Pytest configuration.

Puts `src/` on sys.path so tests can `from wordle.x import y` without needing
PYTHONPATH=src on the command line. Just run `pytest` from the repo root.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
