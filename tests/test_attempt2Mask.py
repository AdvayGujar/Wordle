"""Tests for the Attempt-2 action mask.

The mask must equal the slow reference - the feedback-consistent answers minus the words
already played - because the mask is what keeps the agent from wasting its training signal
on illegal guesses.
"""

import numpy as np
import pytest

from attempt2.actionMask import legalActions, maskFor
from wordle.feedback import candidatesMatching, feedback
from wordle.vocabulary import answersVocab, loadAnswers


def _history(secret, guesses):
    return [(guess, feedback(guess, secret)) for guess in guesses]


def _reference(history):
    """The slow way: the core filter, minus the words already played."""
    played = {guess for guess, _ in history}
    consistent = candidatesMatching(history, loadAnswers())
    return {word for word in consistent if word not in played}


@pytest.mark.parametrize(
    "secret, guesses",
    [
        ("crane", []),
        ("crane", ["slate"]),
        ("crane", ["slate", "pious"]),
        ("sassy", ["raise", "crown", "dusty"]),
        ("abbey", ["adieu", "bobby"]),
    ],
)
def testMaskAgreesWithTheSlowReference(secret, guesses):
    history = _history(secret, guesses)
    index = answersVocab().index

    expected = {index[word] for word in _reference(history)}
    actual = set(np.flatnonzero(maskFor(history)))

    assert actual == expected


def testAnEmptyHistoryMakesEveryAnswerLegal():
    mask = maskFor([])

    assert mask.shape == (len(loadAnswers()),)
    assert mask.all()


def testAGuessedWordIsNeverLegalAgain():
    secret = "crane"
    history = _history(secret, ["slate"])
    index = answersVocab().index

    assert not maskFor(history)[index["slate"]]


def testTheSecretStaysLegalUntilItIsGuessed():
    # The secret is consistent with every round by definition, so it must never be masked
    # out - otherwise the agent could not finish the game.
    secret = "crane"
    history = _history(secret, ["slate", "pious"])
    index = answersVocab().index

    assert maskFor(history)[index[secret]]


def testTheMaskShrinksAsEvidenceAccumulates():
    secret = "crane"
    counts = []

    for guesses in ([], ["slate"], ["slate", "pious"], ["slate", "pious", "crate"]):
        counts.append(int(maskFor(_history(secret, guesses)).sum()))

    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def testTheMaskIsDeterministic():
    history = _history("crane", ["slate", "pious"])

    assert maskFor(history).tolist() == maskFor(history).tolist()


def testLegalActionsMatchTheMask():
    history = _history("crane", ["slate"])
    mask = maskFor(history)

    assert legalActions(history) == [int(i) for i in np.flatnonzero(mask)]
