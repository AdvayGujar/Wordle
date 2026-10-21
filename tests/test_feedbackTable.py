"""Tests for the precomputed feedback table used by Attempt 2.

The table is a NumPy rewrite of `feedback()`, so the important thing to pin is that it
agrees with the pure-Python rules -- including the repeated-letter cases -- and that the
fast candidate filter returns exactly what `candidatesMatching()` returns.
"""

import pytest

from wordle.feedback import candidatesMatching, feedback
from wordle.feedbackTable import (
    PATTERN_COUNT,
    candidatesMatchingFast,
    decodePattern,
    feedbackCode,
    feedbackTable,
    patternCode,
)
from wordle.vocabulary import answersVocab, loadAnswers, loadWords


def _code(pattern):
    return patternCode(pattern)


# ---------------------------------------------------------------------------
# Pattern codes
# ---------------------------------------------------------------------------

def testPatternCodeRoundTripsEveryPattern():
    for code in range(PATTERN_COUNT):
        assert patternCode(decodePattern(code)) == code


def testPatternCodeIsBaseThreeWithPositionZeroMostSignificant():
    assert patternCode((0, 0, 0, 0, 0)) == 0
    assert patternCode((2, 2, 2, 2, 2)) == PATTERN_COUNT - 1
    assert patternCode((1, 0, 0, 0, 0)) == 81
    assert patternCode((0, 0, 0, 0, 1)) == 1


@pytest.mark.parametrize("bad", [(0, 1), (0, 1, 2, 3, 0), (0, 1, 2, 0, 3), []])
def testPatternCodeRejectsInvalidPatterns(bad):
    with pytest.raises(ValueError):
        patternCode(bad)


@pytest.mark.parametrize("bad", [-1, PATTERN_COUNT, PATTERN_COUNT + 5])
def testDecodePatternRejectsOutOfRangeCodes(bad):
    with pytest.raises(ValueError):
        decodePattern(bad)


# ---------------------------------------------------------------------------
# The table agrees with feedback()
# ---------------------------------------------------------------------------

def testTableShapeAndDtype():
    table = feedbackTable()
    answers = loadAnswers()
    assert table.shape == (len(answers), len(answers))
    assert table.dtype.name == "uint8"


def testTableMatchesFeedbackForEveryGuessOfRepeatedLetterSecrets():
    # 'sassy', 'array' and 'abbey' exercise repeated letters in both directions.
    answers = loadAnswers()
    index = answersVocab().index
    table = feedbackTable()

    for secret in ("sassy", "array", "abbey", "crane"):
        column = table[:, index[secret]]
        actual = [_code(feedback(guess, secret)) for guess in answers]
        assert column.tolist() == actual


def testFeedbackCodeMatchesFeedback():
    assert feedbackCode("crane", "crate") == _code(feedback("crane", "crate"))
    assert feedbackCode("aaaba", "ababa") == _code(feedback("aaaba", "ababa"))
    assert feedbackCode("ooooo", "books") == _code(feedback("ooooo", "books"))


def testFeedbackCodeFallsBackForNonAnswerWords():
    guess = next(word for word in loadWords() if word not in answersVocab().index)
    assert feedbackCode(guess, "crane") == _code(feedback(guess, "crane"))


# ---------------------------------------------------------------------------
# The fast candidate filter returns the same set as the core one
# ---------------------------------------------------------------------------

def _evidence(secret, guesses):
    return [(guess, feedback(guess, secret)) for guess in guesses]


@pytest.mark.parametrize(
    "secret, guesses",
    [
        ("crane", ["slate", "pious"]),
        ("sassy", ["raise", "crown", "dusty"]),
        ("abbey", ["adieu", "bobby"]),
        ("eerie", ["slate", "their"]),
    ],
)
def testFastFilterMatchesCoreFilter(secret, guesses):
    constraints = _evidence(secret, guesses)
    expected = candidatesMatching(constraints, loadAnswers())
    actual = candidatesMatchingFast(constraints)
    assert actual == expected


def testFastFilterKeepsOrderAndRestrictsToGivenVocabulary():
    constraints = [("crane", feedback("crane", "crate"))]
    answers = loadAnswers()
    vocab = answers[:200]
    assert candidatesMatchingFast(constraints, vocab) == candidatesMatching(constraints, vocab)


def testFastFilterSupportsNonAnswerGuesses():
    guess = next(word for word in loadWords() if word not in answersVocab().index)
    constraints = [(guess, feedback(guess, "crane"))]
    assert candidatesMatchingFast(constraints) == candidatesMatching(constraints, loadAnswers())


def testFastFilterSupportsNonAnswerVocabularyWords():
    answers = answersVocab().index
    vocab = list(loadAnswers()[:50])
    vocab += [word for word in loadWords() if word not in answers][:50]
    constraints = [("crane", feedback("crane", "crate"))]
    assert candidatesMatchingFast(constraints, vocab) == candidatesMatching(constraints, vocab)


def testFastFilterWithNoConstraintsReturnsWholeVocabulary():
    assert candidatesMatchingFast([]) == list(loadAnswers())


def testFastFilterCanEliminateEverything():
    # 'crane' all-green leaves only 'crane', and 'crane' cannot be all-grey to 'slate'.
    constraints = [("crane", (2, 2, 2, 2, 2)), ("slate", (0, 0, 0, 0, 0))]
    assert candidatesMatchingFast(constraints) == candidatesMatching(constraints, loadAnswers())
