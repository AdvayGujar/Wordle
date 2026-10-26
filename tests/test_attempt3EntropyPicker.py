"""Tests for B1, the pure-information ablation.

What is pinned:

- at the empty history it opens with `raise`, the highest-entropy answer (§2.2);
- it picks the highest-entropy guess of the pool, and the pool includes probes;
- it will play a word that is *not* a candidate when that word is more informative - the
  defining behaviour of the ablation, measured after `raise` with one yellow;
- a singleton is guessed, not probed;
- it never repeats a played word and is deterministic;
- an empty feasible set fails loudly.
"""

import numpy as np
import pytest

from attempt3.entropyPicker import bestGuess, pickGuess
from attempt3.patterns import guessStatistics
from attempt3.pickers import PickingState, stateFor
from wordle.vocabulary import answersVocab


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


def testTheOpenerIsRaise():
    assert pickGuess([]) == _index("raise")


def testItPicksTheHighestEntropyGuessInThePool():
    candidates = _indices(["crate", "trace", "abbey", "sassy", "mucky"])
    guesses = _indices(["raise", "slate", "abbey", "crane", "mucky"])

    chosen = bestGuess(candidates, guesses)
    entropies = guessStatistics(guesses, candidates).entropies

    assert entropies[list(guesses).index(chosen)] == pytest.approx(entropies.max())


def testItPlaysAProbeWhenTheProbeCarriesMoreInformation():
    # Measured: after `raise` with one yellow, the highest-entropy word is `count`, which
    # the evidence has not ruled out - a probe, not a candidate. Forbidding probes would
    # change this row of the table, which is why the pool allows them.
    history = [("raise", (1, 0, 0, 0, 0))]
    chosen = pickGuess(history)
    state = stateFor(history)

    assert answersVocab().wordAt(chosen) == "count"
    assert chosen not in set(state.candidates.tolist())


def testItNeverRepeatsAPlayedWord():
    history = [("raise", (1, 0, 0, 0, 0))]

    chosen = pickGuess(history)

    assert chosen != _index("raise")


def testASingletonIsGuessedRatherThanProbed(monkeypatch):
    answer = _index("abbey")
    state = PickingState(
        candidates=np.array([answer]),
        guesses=np.array([answer, _index("crane")]),
    )
    monkeypatch.setattr("attempt3.entropyPicker.stateFor", lambda history: state)

    assert pickGuess([("raise", (0, 0, 0, 0, 0))]) == answer


def testItIsDeterministic():
    history = [("raise", (1, 0, 0, 0, 0))]

    assert {pickGuess(history) for _ in range(3)} == {pickGuess(history)}


def testAnEmptyFeasibleSetFailsLoudly():
    with pytest.raises(ValueError):
        bestGuess(np.array([], dtype=np.int64), _indices(["raise"]))
