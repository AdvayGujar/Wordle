"""Tests for the game rules in feedback.py.

The plan calls this "the most important and most-tested file", with repeated
letters being the classic bug. These lock that behaviour in.
"""

import pytest

from wordle.feedback import candidatesMatching, feedback
from wordle.vocabulary import loadAnswers, loadWords


# ---------------------------------------------------------------------------
# feedback(): the documented cases from plan.md section 6.1
# ---------------------------------------------------------------------------

def testFeedbackKnownCases():
    # The original repeated-letter bug case.
    assert feedback("aaaba", "ababa") == (2, 0, 2, 2, 2)
    # Double-r case.
    assert feedback("error", "shear") == (1, 0, 0, 0, 2)
    # Overlapping letters where an early green must not be overwritten.
    assert feedback("crane", "crate") == (2, 2, 2, 0, 2)


def testFeedbackIdenticalWordIsAllGreen():
    assert feedback("apple", "apple") == (2, 2, 2, 2, 2)


def testFeedbackNoSharedLettersIsAllGray():
    assert feedback("crane", "spilt") == (0, 0, 0, 0, 0)


def testFeedbackReturnsFiveColorCodes():
    pattern = feedback("crane", "apple")
    assert len(pattern) == 5
    assert all(code in (0, 1, 2) for code in pattern)


# ---------------------------------------------------------------------------
# Repeated letters -- the caveat the plan calls out explicitly
# ---------------------------------------------------------------------------

def testYellowIsAwardedOnlyAsOftenAsTheLetterRemains():
    # Secret 'crane' has one 'a' and one 'r'. Guess 'array' has two of each, so
    # the first occurrence earns a colour and the surplus one must be gray.
    assert feedback("array", "crane") == (1, 2, 0, 0, 0)


def testGreenIsNotDowngradedToYellow():
    # Both 'a's in the guess are in position in the secret.
    assert feedback("aaabb", "aabbb") == (2, 2, 0, 2, 2)


def testSurplusGuessLettersAreGrayNotYellow():
    # Secret 'books' has two 'o's; the guess has five. Only the two positions
    # that match exactly are green -- there is nothing left to make yellow.
    assert feedback("ooooo", "books") == (0, 2, 2, 0, 0)


def testRepeatedGreensAllStayGreen():
    # Three 'e's line up exactly; the two that don't must not borrow a colour
    # from the ones that do.
    assert feedback("eeeee", "eerie") == (2, 2, 0, 0, 2)


def testGreensAreCountedBeforeYellows():
    # Secret 'ababa': the 'a's at positions 0, 2 and 4 are green, consuming all
    # three of the secret's 'a's, so the remaining guess 'a's stay gray.
    assert feedback("aaaaa", "ababa") == (2, 0, 2, 0, 2)


def testGrayWhenGuessLetterCountExceedsSecretCount():
    # Secret 'crane' has a single 'a'; guess 'sassy' has one 'a' -> one yellow.
    assert feedback("sassy", "crane") == (0, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# candidatesMatching()
# ---------------------------------------------------------------------------

def testCandidatesMatchingNarrowsToTheSecret():
    constraints = [("crane", feedback("crane", "apple"))]
    candidates = candidatesMatching(constraints, loadAnswers())
    assert "apple" in candidates
    assert len(candidates) < len(loadAnswers())


def testCandidatesMatchingAllGreenIsExactlyTheSecret():
    secret = "apple"
    constraints = [(secret, feedback(secret, secret))]
    assert candidatesMatching(constraints, loadAnswers()) == [secret]


def testCandidatesMatchingWithNoConstraintsReturnsEverything():
    answers = loadAnswers()
    assert candidatesMatching([], answers) == list(answers)


def testCandidatesMatchingAcceptsAnyIterable():
    """The vocab argument doesn't have to be the full 2,315-word list."""
    assert candidatesMatching([], ["apple", "crane"]) == ["apple", "crane"]


def testCandidatesMatchingEveryResultIsConsistent():
    """Every returned candidate must genuinely reproduce all patterns."""
    constraints = [
        ("crane", feedback("crane", "apple")),
        ("slate", feedback("slate", "apple")),
    ]
    candidates = candidatesMatching(constraints, loadWords())
    assert candidates, "expected at least one candidate"
    for word in candidates:
        for guess, pattern in constraints:
            assert feedback(guess, word) == pattern
