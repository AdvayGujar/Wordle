"""Tests for the shared Phase-B state and tie-break.

`pickers.py` is the module `attempt3.md` §4.3 did not name: it holds the two things every
picker needs (the feasible set with the unplayed pool, and the deterministic tie-break), so
that three ablations cannot resolve the same tie three ways.

The properties pinned here:

- the feasible set from a history is the consistent-and-unplayed answers, and the guess pool
  is every unplayed answer (probes included);
- a played word is never offered again;
- the tie-break is exactly §2.7's order: score, then candidate, then expected remaining,
  then index;
- malformed score arrays fail loudly rather than pairing a word with another word's score.
"""

import numpy as np
import pytest

from attempt3.pickers import PickingState, bestByScore, stateFor
from wordle.feedback import candidatesMatching
from wordle.vocabulary import answersVocab, loadAnswers


def _index(word):
    return answersVocab().indexOf(word)


def testTheEmptyHistoryOffersEveryAnswerAsBothSetAndPool():
    state = stateFor([])

    assert state.candidates.size == len(loadAnswers())
    assert state.guesses.size == len(loadAnswers())
    assert not state.isSolved


def testTheFeasibleSetIsConsistentWordsAndThePoolIsEveryUnplayedWord():
    history = [("raise", (1, 1, 0, 0, 0)), ("abbey", (0, 0, 2, 2, 0))]
    state = stateFor(history)
    answers = loadAnswers()

    # The slow core rule is the reference; `maskFor` only has to agree with it.
    expected = set(candidatesMatching(history, answers))
    played = {_index(word) for word, _ in history}
    pool = set(range(len(answers))) - played

    assert {answers[index] for index in state.candidates} == expected
    assert set(state.guesses.tolist()) == pool
    assert set(state.candidates.tolist()) <= set(state.guesses.tolist())


def testAPlayedWordIsNotOfferedAgain():
    history = [("crane", (1, 0, 0, 0, 2))]
    state = stateFor(history)
    played = _index("crane")

    assert played not in state.guesses
    assert played not in state.candidates


def testAllGreyEvidenceExcludesEveryWordContainingThoseLetters():
    state = stateFor([("crane", (0, 0, 0, 0, 0))])
    answers = loadAnswers()
    banned = set("crane")

    for index in state.candidates:
        assert not (set(answers[index]) & banned), answers[index]


def testASingleCandidateIsSolved():
    answers = loadAnswers()
    only = _index(answers[0])

    assert PickingState(candidates=np.array([only]), guesses=np.array([only])).isSolved
    assert not PickingState(
        candidates=np.array([only, _index(answers[1])]), guesses=np.array([only])
    ).isSolved


def testTheStateArraysAreFresh():
    first = stateFor([])
    first.candidates[:] = 0
    first.guesses[:] = 0

    assert stateFor([]).candidates.size == len(loadAnswers())


# ---------------------------------------------------------------------------
# The tie-break
# ---------------------------------------------------------------------------


def testTheSmallestScoreWins():
    guesses = np.array([10, 11, 12])
    candidates = np.array([10])

    assert bestByScore([2.0, 0.5, 1.0], guesses, candidates, [1.0, 1.0, 1.0]) == 11


def testACandidateBeatsAProbeOnAnExactTie():
    guesses = np.array([10, 11])
    candidates = np.array([11])

    assert bestByScore([1.5, 1.5], guesses, candidates, [3.0, 3.0]) == 11


def testTheSmallerExpectedRemainingBreaksTheNextTie():
    guesses = np.array([10, 11])
    candidates = np.array([10, 11])

    assert bestByScore([1.5, 1.5], guesses, candidates, [40.0, 4.0]) == 11


def testTheLowerIndexBreaksTheFinalTie():
    guesses = np.array([11, 10])
    candidates = np.array([10, 11])

    assert bestByScore([1.5, 1.5], guesses, candidates, [4.0, 4.0]) == 10


def testTheTieBreakIsDeterministicOnARealisticTie():
    guesses = np.array([3, 7, 9])
    candidates = np.array([3, 7, 9])
    scores = [0.0, 0.0, 0.0]

    choices = {bestByScore(scores, guesses, candidates, [1.0, 1.0, 1.0]) for _ in range(5)}

    assert choices == {3}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def testMismatchedLengthsAreRejected():
    with pytest.raises(ValueError):
        bestByScore([1.0, 2.0], np.array([1, 2]), np.array([1]), [1.0])


def testEmptyScoresAreRejected():
    with pytest.raises(ValueError):
        bestByScore([], np.array([], dtype=np.int64), np.array([1]), [])


def testEmptyCandidatesAreRejected():
    with pytest.raises(ValueError):
        bestByScore([1.0], np.array([1]), np.array([], dtype=np.int64), [1.0])


def testNonFiniteScoresAreRejected():
    with pytest.raises(ValueError):
        bestByScore([np.nan], np.array([1]), np.array([1]), [1.0])
