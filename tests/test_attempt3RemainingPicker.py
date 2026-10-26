"""Tests for B2, the expected-survivors ablation.

What is pinned:

- at the empty history it opens with `raise`, which has the smallest E[|C'|] as well as the
  highest entropy, so the opener cannot separate the two criteria (§3.2);
- it picks the guess with the smallest expected remaining candidate set;
- a singleton is guessed, not probed;
- it never repeats a played word and is deterministic;
- an empty feasible set fails loudly.

The interesting comparison with B1 is not the opener but the later turns, where the two
criteria diverge; the Gate B table measures that, and this file only pins the mechanism.
"""

import numpy as np
import pytest

from attempt3.patterns import guessStatistics
from attempt3.pickers import PickingState, stateFor
from attempt3.remainingPicker import bestGuess, pickGuess
from wordle.vocabulary import answersVocab


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


def testTheOpenerIsRaise():
    assert pickGuess([]) == _index("raise")


def testItPicksTheSmallestExpectedRemainingSet():
    candidates = _indices(["crate", "trace", "abbey", "sassy", "mucky"])
    guesses = _indices(["raise", "slate", "abbey", "crane", "mucky"])

    chosen = bestGuess(candidates, guesses)
    remaining = guessStatistics(guesses, candidates).expectedRemaining

    assert remaining[list(guesses).index(chosen)] == pytest.approx(remaining.min())


def testItNeverRepeatsAPlayedWord():
    history = [("raise", (1, 0, 0, 0, 0))]
    chosen = pickGuess(history)
    state = stateFor(history)

    assert chosen != _index("raise")
    assert chosen in set(state.guesses.tolist())


def testASingletonIsGuessedRatherThanProbed(monkeypatch):
    answer = _index("abbey")
    state = PickingState(
        candidates=np.array([answer]),
        guesses=np.array([answer, _index("crane")]),
    )
    monkeypatch.setattr("attempt3.remainingPicker.stateFor", lambda history: state)

    assert pickGuess([("raise", (0, 0, 0, 0, 0))]) == answer


def testItIsDeterministic():
    history = [("raise", (1, 0, 0, 0, 0))]

    assert {pickGuess(history) for _ in range(3)} == {pickGuess(history)}


def testAnEmptyFeasibleSetFailsLoudly():
    with pytest.raises(ValueError):
        bestGuess(np.array([], dtype=np.int64), _indices(["raise"]))
