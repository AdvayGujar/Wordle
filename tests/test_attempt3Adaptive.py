"""Tests for C1, the expected-score policy.

What is pinned:

- it opens with `raise`, with and without probes;
- after `raise` with one yellow it plays `count`, a word the evidence has ruled out - the
  criterion values the information enough to probe;
- the no-probe variant answers the same question with `court`, a candidate, so the switch
  does what it claims;
- a candidate beats a probe when the information is equal, which is the win-chance term;
- a singleton is guessed; a played word is never repeated; the result is deterministic;
- the `attempt2.eval` adapter returns a valid answer index and sees no secret.
"""

import numpy as np
import pytest

from attempt3.adaptivePicker import bestGuess, pickGuess, policy
from attempt3.pickers import PickingState, stateFor
from wordle.vocabulary import answersVocab, loadAnswers


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


def testTheOpenerIsRaiseWithAndWithoutProbes():
    assert pickGuess([]) == _index("raise")
    assert pickGuess([], allowProbes=False) == _index("raise")


def testItPlaysAProbeWhenTheInformationIsWorthIt():
    history = [("raise", (1, 0, 0, 0, 0))]
    chosen = pickGuess(history)
    state = stateFor(history)

    assert answersVocab().wordAt(chosen) == "count"
    assert chosen not in set(state.candidates.tolist())


def testTheNoProbeVariantStaysInsideTheFeasibleSet():
    history = [("raise", (1, 0, 0, 0, 0))]
    chosen = pickGuess(history, allowProbes=False)
    state = stateFor(history)

    assert answersVocab().wordAt(chosen) == "court"
    assert chosen in set(state.candidates.tolist())


def testACandidateBeatsAnEquallyInformativeProbe(monkeypatch):
    # Two candidates: every guess can carry at most 1 bit, and any split is 1-1. The
    # candidate's win chance therefore decides, which is exactly the §2.7 term.
    first = _index("abbey")
    second = _index("crate")
    state = PickingState(
        candidates=np.array([first, second]),
        guesses=np.array([first, second, _index("fuzzy")]),
    )
    monkeypatch.setattr("attempt3.adaptivePicker.stateFor", lambda history: state)

    assert pickGuess([("jazzy", (0, 0, 0, 0, 0))]) in {first, second}


def testASingletonIsGuessed(monkeypatch):
    answer = _index("abbey")
    state = PickingState(
        candidates=np.array([answer]),
        guesses=np.array([answer, _index("crane")]),
    )
    monkeypatch.setattr("attempt3.adaptivePicker.stateFor", lambda history: state)

    assert pickGuess([("mummy", (0, 0, 0, 0, 0))]) == answer


def testItNeverRepeatsAPlayedWord():
    history = [("raise", (1, 0, 0, 0, 0))]

    assert pickGuess(history) != _index("raise")


def testItIsDeterministic():
    history = [("raise", (0, 1, 0, 0, 0))]

    assert {pickGuess(history) for _ in range(3)} == {pickGuess(history)}


def testTheAdapterReturnsAValidAnswerIndex():
    from attempt2.eval import PolicyState

    for allowProbes in (True, False):
        act = policy(allowProbes=allowProbes)

        for history in ([], [("raise", (1, 0, 0, 0, 0))]):
            action = act(PolicyState(observation=None, mask=None, history=tuple(history)))

            assert isinstance(action, int)
            assert 0 <= action < len(loadAnswers())


def testTheNoProbeAdapterOnlyReturnsCandidates():
    from attempt2.eval import PolicyState

    history = [("raise", (1, 0, 0, 0, 0))]
    action = policy(allowProbes=False)(PolicyState(None, None, tuple(history)))

    assert action in set(stateFor(history).candidates.tolist())


def testAnEmptyFeasibleSetFailsLoudly():
    with pytest.raises(ValueError):
        bestGuess(np.array([], dtype=np.int64), _indices(["raise"]))
