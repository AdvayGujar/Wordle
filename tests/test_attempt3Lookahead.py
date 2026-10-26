"""Tests for C2, the two-ply lookahead.

What is pinned:

- the two-ply value agrees with an independent brute force built on `feedback()`, on a small
  feasible set, for every possible first word;
- the all-green branch costs nothing further, so a two-candidate node values at 1.5, the
  same as guessing the answer now;
- the policy opens with `least`, which is *not* the one-ply choice (`raise`) - the lookahead
  really does change the answer;
- a singleton is guessed; a played word is never repeated; the result is deterministic;
- the adapter returns a valid index, and malformed inputs fail loudly.
"""

import math

import numpy as np
import pytest

from attempt3.adaptivePicker import pickGuess as greedyPick
from attempt3.lookahead import bestGuess, pickGuess, policy, twoPlyScore
from attempt3.pickers import PickingState
from wordle.feedback import feedback
from wordle.feedbackTable import patternCode
from wordle.vocabulary import answersVocab, loadAnswers

GREEN = patternCode((2, 2, 2, 2, 2))


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


def _slowScore(guess, candidates):
    """§2.7's score from `feedback()` alone - no Attempt-3 code involved."""
    counts = {}
    for candidate in candidates:
        code = patternCode(feedback(guess, candidate))
        counts[code] = counts.get(code, 0) + 1

    size = len(candidates)
    entropy = -sum((n / size) * math.log2(n / size) for n in counts.values())
    current = math.log2(size)
    chance = 1.0 / size if guess in candidates else 0.0
    left = current - entropy
    estimate = 2.0 ** (-left) + 2.0 * (1.0 - 2.0 ** (-left)) + 1.5 * left / 11.5

    return chance + (1.0 - chance) * (1.0 + estimate)


def _bruteTwoPly(first, candidates, seconds):
    """The two-ply value by nested loops: branches, then the best second word in each."""
    groups = {}
    for candidate in candidates:
        code = patternCode(feedback(first, candidate))
        groups.setdefault(code, []).append(candidate)

    size = len(candidates)
    total = 0.0

    for code, branch in groups.items():
        if code == GREEN:
            continue
        total += len(branch) / size * min(_slowScore(word, branch) for word in seconds)

    return 1.0 + total


def testTheTwoPlyScoreMatchesBruteForceOnATinySet():
    words = ["abbey", "crate", "trace", "mummy", "fuzzy"]
    candidates = _indices(words)
    seconds = _indices(words)

    for first in candidates:
        word = answersVocab().wordAt(first)
        expected = _bruteTwoPly(word, words, words)

        assert twoPlyScore(first, candidates, seconds) == pytest.approx(expected, abs=1e-9)


def testTheGreenBranchCostsNothingFurther():
    # Two candidates, guess one: half the time the game ends (cost 1), half the time one
    # more guess names the other. Total 1.5, not 2.5.
    candidates = _indices(["abbey", "crate"])

    assert twoPlyScore(_index("abbey"), candidates, candidates) == pytest.approx(1.5)


def testTheOpenerIsNotTheGreedyChoice():
    assert answersVocab().wordAt(greedyPick([])) == "raise"
    assert answersVocab().wordAt(pickGuess([])) == "least"


def testItNeverRepeatsAPlayedWord():
    history = [("least", (0, 0, 0, 0, 0))]

    assert pickGuess(history) != _index("least")


def testASingletonIsGuessed(monkeypatch):
    answer = _index("abbey")
    state = PickingState(
        candidates=np.array([answer]),
        guesses=np.array([answer, _index("crane")]),
    )
    monkeypatch.setattr("attempt3.lookahead.stateFor", lambda history: state)

    assert pickGuess([("mummy", (0, 0, 0, 0, 0))]) == answer


def testItIsDeterministic():
    history = [("least", (0, 0, 0, 0, 0))]

    assert {pickGuess(history) for _ in range(2)} == {pickGuess(history)}


def testTheAdapterReturnsAValidAnswerIndex():
    from attempt2.eval import PolicyState

    action = policy()(PolicyState(None, None, ()))

    assert isinstance(action, int)
    assert 0 <= action < len(loadAnswers())


def testAnEmptyFeasibleSetFailsLoudly():
    with pytest.raises(ValueError):
        bestGuess(np.array([], dtype=np.int64), _indices(["raise"]))


def testAnEmptySecondShortlistFailsLoudly():
    candidates = _indices(["abbey", "crate"])

    with pytest.raises(ValueError):
        twoPlyScore(_index("abbey"), candidates, np.array([], dtype=np.int64))


def testAZeroShortlistFailsLoudly():
    candidates = _indices(["abbey", "crate"])

    with pytest.raises(ValueError):
        bestGuess(candidates, candidates, firstCount=0)


def testAnOutOfRangeFirstWordFailsLoudly():
    candidates = _indices(["abbey", "crate"])

    with pytest.raises(ValueError):
        twoPlyScore(len(loadAnswers()) + 5, candidates, candidates)
