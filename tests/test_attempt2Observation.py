"""Tests for the Attempt-2 observation encoder.

The observation has to be a fixed width (one network handles every turn) and must agree
exactly with the core encoder it delegates to.
"""

import numpy as np
import pytest

from attempt2 import config
from attempt2.observation import OBSERVATION_WIDTH, observationFor, observationWidth
from wordle.features import encodeLetterState, encodeRounds
from wordle.feedback import feedback


def _history(secret, guesses):
    return [(guess, feedback(guess, secret)) for guess in guesses]


def testDefaultWidthMatchesTheDefaultEncoder():
    assert OBSERVATION_WIDTH == observationWidth(config.OBSERVATION_MODE)
    assert OBSERVATION_WIDTH == len(
        encodeRounds([], config.MAX_ROUNDS, "compact")
    )


def testRoundsModeIsSixtyDimsAndLetterStateIsTwoHundredAndEight():
    assert observationWidth("rounds") == 60
    assert observationWidth("letterState") == 208


def testWidthIsConstantWhateverTheHistoryLength():
    secret = "crane"
    history = _history(secret, ["slate", "pious", "crown"])

    widths = {
        len(observationFor(history[:count]))
        for count in range(len(history) + 1)
    }
    assert widths == {observationWidth("rounds")}


def testTheVectorIsExactlyTheCoreEncoding():
    secret = "crane"
    history = _history(secret, ["slate", "pious"])

    expected = encodeRounds(history, config.MAX_ROUNDS, "compact")
    actual = observationFor(history, "rounds")

    assert actual.tolist() == [float(value) for value in expected]


def testLetterStateModeIsExactlyTheCoreEncoding():
    secret = "crane"
    history = _history(secret, ["slate"])

    expected = encodeLetterState(history)
    actual = observationFor(history, "letterState")

    assert actual.tolist() == [float(value) for value in expected]


def testAnEmptyHistoryIsAllPadding():
    vector = observationFor([])

    assert vector.tolist() == [-1.0] * observationWidth("rounds")


def testTheVectorIsFloat32():
    assert observationFor([]).dtype == np.float32


def testUnknownModeIsRejected():
    with pytest.raises(ValueError, match="unknown observation mode"):
        observationFor([], "telepathy")

    with pytest.raises(ValueError, match="unknown observation mode"):
        observationWidth("telepathy")
