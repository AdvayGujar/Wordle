"""Tests for the advisor-mode colour parser."""

import pytest

from attempt1.play import parsePattern


@pytest.mark.parametrize("raw,expected", [
    ("gyxgy", (2, 1, 0, 2, 1)),
    ("GYXGY", (2, 1, 0, 2, 1)),
    ("2 1 0 0 2", (2, 1, 0, 0, 2)),
    ("21200", (2, 1, 2, 0, 0)),
    ("g y x g y", (2, 1, 0, 2, 1)),
    ("green yellow grey grey green", (2, 1, 0, 0, 2)),
    ("🟩 🟨 ⬜ ⬜ 🟩", (2, 1, 0, 0, 2)),
])
def testParsesFriendlyColourForms(raw, expected):
    assert parsePattern(raw) == expected


@pytest.mark.parametrize("raw", ["", "gyxg", "gyxgyx", "gyxqy", "green yellow"])
def testRejectsMalformedColourInput(raw):
    with pytest.raises(ValueError):
        parsePattern(raw)
