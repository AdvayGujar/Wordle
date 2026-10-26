"""Tests for B3, the fixed two-word ablation.

What is pinned:

- the pair's information is the entropy of its joint 243 x 243 pattern distribution, so it
  is symmetric - the criterion cannot choose an order;
- the *conditional* information is not symmetric, which is why the order is set by the
  individual entropies;
- the pair cannot carry more than its two words carry separately;
- the pair is two different playable words, oriented so the more informative one is first;
- the policy plays the pair on the first two turns and hands over to B2 on the third;
- a singleton is guessed, and a pair needs at least two words to exist.
"""

import numpy as np
import pytest

from attempt3 import remainingPicker
from attempt3.pairPicker import (
    bestPair,
    conditionalEntropy,
    jointEntropy,
    pickGuess,
    rootPair,
)
from attempt3.pickers import PickingState, stateFor
from wordle.vocabulary import answersVocab, loadAnswers


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


ALL = np.arange(len(loadAnswers()), dtype=np.int64)


def testTheJointEntropyIsSymmetric():
    first = _index("raise")
    second = _index("salon")

    assert jointEntropy(first, second, ALL) == pytest.approx(
        jointEntropy(second, first, ALL)
    )


def testTheConditionalInformationIsNotSymmetric():
    first = _index("raise")
    second = _index("salon")

    firstGivenSecond = conditionalEntropy(first, second, ALL)
    secondGivenFirst = conditionalEntropy(second, first, ALL)

    assert firstGivenSecond == pytest.approx(2.9738, abs=1e-3)
    assert secondGivenFirst == pytest.approx(2.4283, abs=1e-3)
    assert firstGivenSecond != pytest.approx(secondGivenFirst)


def testAPairCannotCarryMoreThanItsTwoWordsSeparately():
    words = _indices(["raise", "slate", "crate", "trace", "crane", "slate"][:5])
    pair = bestPair(words, words, shortlist=5)

    assert pair.jointEntropy <= pair.firstEntropy + pair.secondEntropy + 1e-9


def testTheBestPairIsTwoDistinctPlayableWordsOrTheHigherEntropyFirst():
    words = _indices(["raise", "salon", "trice", "slate", "mamma"])
    pair = bestPair(words, words, shortlist=5)

    assert pair.first != pair.second
    assert {pair.first, pair.second} <= set(words.tolist())
    assert pair.firstEntropy >= pair.secondEntropy
    assert pair.conditionalInformation == pytest.approx(
        pair.jointEntropy - pair.firstEntropy
    )


def testAPairNeedsTwoWords():
    words = np.array([_index("raise")])

    with pytest.raises(ValueError):
        bestPair(words, words, shortlist=1)


def testTheFixedPairIsPlayedOnTheFirstTwoTurns():
    pair = rootPair()
    firstWord = answersVocab().wordAt(pair.first)
    history = [(firstWord, (0, 0, 0, 0, 0))]

    assert stateFor(history).candidates.size > 1
    assert pickGuess([]) == pair.first
    assert pickGuess(history) == pair.second


def testTheThirdTurnHandsOverToB2():
    pair = rootPair()
    history = [
        (answersVocab().wordAt(pair.first), (0, 0, 0, 0, 0)),
        (answersVocab().wordAt(pair.second), (0, 1, 0, 0, 0)),
    ]

    assert pickGuess(history) == remainingPicker.pickGuess(history)


def testThePairDoesNotRepeatAPlayedWord():
    pair = rootPair()
    history = [(answersVocab().wordAt(pair.first), (0, 0, 0, 0, 0))]

    assert pickGuess(history) not in {pair.first}


def testASingletonIsGuessedRatherThanFollowingThePair(monkeypatch):
    answer = _index("abbey")
    state = PickingState(
        candidates=np.array([answer]),
        guesses=np.array([answer, _index("crane")]),
    )
    monkeypatch.setattr("attempt3.pairPicker.stateFor", lambda history: state)

    assert pickGuess([]) == answer


def testTheSameIndexTwiceIsRejected():
    with pytest.raises(ValueError):
        jointEntropy(_index("raise"), _index("raise"), ALL)


def testAnEmptyCandidateSetIsRejected():
    with pytest.raises(ValueError):
        jointEntropy(_index("raise"), _index("salon"), np.array([], dtype=np.int64))


def testTheRootPairIsFixedAcrossCalls():
    assert rootPair() == rootPair()
