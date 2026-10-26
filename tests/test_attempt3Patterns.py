"""Tests for the Attempt-3 pattern statistics.

The histogram is the foundation every picker stands on, so the important properties are:

- it must agree with the slow `feedback()` rule, including repeated letters;
- every row must sum to the size of the candidate set;
- the derived quantities must match hand-computed small distributions;
- the Jensen gap must be the non-negative quantity derived in the module, not the signed
  `log2(E) - H` difference an earlier note used;
- malformed input must fail loudly instead of producing a plausible-looking ranking.

None of these tests needs a full game. The feedback table is built once and cached.
"""

import numpy as np
import pytest

from attempt3.patterns import (
    PATTERN_COUNT,
    entropyFromCounts,
    expectedRemainingFromCounts,
    guessStatistics,
    jensenGapFromCounts,
    patternCounts,
)
from wordle.feedback import feedback
from wordle.feedbackTable import patternCode
from wordle.vocabulary import answersVocab, loadAnswers


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


def _counts(*rows):
    """A well-formed counts array from per-row lists of bin values."""
    result = np.zeros((len(rows), PATTERN_COUNT), dtype=np.int64)
    for row, values in enumerate(rows):
        result[row, : len(values)] = values
    return result


def _slowCounts(guesses, candidates):
    """The reference histogram: one `feedback()` call per pair."""
    counts = np.zeros((len(guesses), PATTERN_COUNT), dtype=np.int64)
    for row, guess in enumerate(guesses):
        for candidate in candidates:
            counts[row, patternCode(feedback(guess, candidate))] += 1
    return counts


# ---------------------------------------------------------------------------
# The histogram
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "guesses, candidates",
    [
        (["raise"], ["crane"]),
        (["raise", "slate"], ["crate", "trace", "abbey", "sassy"]),
        (["crane", "eerie", "abbey", "sassy"],
         ["crate", "mucky", "eerie", "bobby", "trace", "abbey"]),
    ],
)
def testPatternCountsMatchTheSlowFeedbackRule(guesses, candidates):
    expected = _slowCounts(guesses, candidates)
    actual = patternCounts(_indices(guesses), _indices(candidates))

    assert np.array_equal(actual, expected)


def testAKnownRepeatedLetterPatternIsCounted():
    guess = _index("crane")
    candidate = _index("crate")
    code = patternCode(feedback("crane", "crate"))

    counts = patternCounts([guess], [candidate])

    assert counts.shape == (1, PATTERN_COUNT)
    assert counts[0, code] == 1
    assert counts[0].sum() == 1


def testEveryHistogramRowSumsToTheCandidateCount():
    rng = np.random.default_rng(7)
    answerCount = len(loadAnswers())
    guesses = rng.choice(answerCount, size=50, replace=False)
    candidates = rng.choice(answerCount, size=100, replace=False)

    counts = patternCounts(guesses, candidates)

    assert counts.shape == (50, PATTERN_COUNT)
    assert np.all(counts.sum(axis=1) == 100)
    assert np.issubdtype(counts.dtype, np.integer)


def testAnEmptyCandidateSetProducesZeroRows():
    counts = patternCounts([0, 1, 2], [])

    assert counts.shape == (3, PATTERN_COUNT)
    assert not counts.any()


def testAnEmptyGuessSetProducesNoRows():
    counts = patternCounts([], [0, 1])

    assert counts.shape == (0, PATTERN_COUNT)


def testPatternCountsAcceptListsTuplesArraysAndGenerators():
    guesses = _indices(["raise", "slate"])
    candidates = _indices(["crate", "trace", "abbey"])
    expected = patternCounts(guesses, candidates)

    variants = [
        patternCounts(list(guesses), list(candidates)),
        patternCounts(tuple(guesses), tuple(candidates)),
        patternCounts(iter(guesses), iter(candidates)),
    ]

    for variant in variants:
        assert np.array_equal(variant, expected)


def testPatternCountsAreIndependentOfCandidateOrder():
    guesses = _indices(["raise", "slate", "eerie"])
    candidates = _indices(["crate", "trace", "abbey", "sassy", "mucky"])

    straight = patternCounts(guesses, candidates)
    reversedCandidates = patternCounts(guesses, candidates[::-1])

    assert np.array_equal(straight, reversedCandidates)


def testPatternCountsAreIndependentOfGuessOrder():
    guesses = _indices(["raise", "slate", "eerie"])
    candidates = _indices(["crate", "trace", "abbey", "sassy"])

    straight = patternCounts(guesses, candidates)
    reversedGuesses = patternCounts(guesses[::-1], candidates)

    assert np.array_equal(straight[::-1], reversedGuesses)


def testPatternCountsDoNotMutateTheirInputs():
    guesses = _indices(["raise", "slate"])
    candidates = _indices(["crate", "trace"])
    guessesBefore = guesses.copy()
    candidatesBefore = candidates.copy()

    patternCounts(guesses, candidates)

    assert np.array_equal(guesses, guessesBefore)
    assert np.array_equal(candidates, candidatesBefore)


@pytest.mark.parametrize("bad", [-1, len(loadAnswers()), 10_000])
def testPatternCountsRejectOutOfRangeIndices(bad):
    with pytest.raises(ValueError, match="out of range"):
        patternCounts([bad], [0])

    with pytest.raises(ValueError, match="out of range"):
        patternCounts([0], [bad])


@pytest.mark.parametrize("bad", [[0.0], [True], ["0"]])
def testPatternCountsRejectNonIntegerIndices(bad):
    with pytest.raises(ValueError, match="integers"):
        patternCounts(bad, [0])


@pytest.mark.parametrize("bad", [0, np.array([[0, 1]]), np.array([[[0]]])])
def testPatternCountsRejectNonOneDimensionalInput(bad):
    with pytest.raises(ValueError, match="one-dimensional"):
        patternCounts(bad, [0])


@pytest.mark.parametrize("duplicates", [[0, 0], [5, 6, 5]])
def testPatternCountsRejectDuplicateIndices(duplicates):
    with pytest.raises(ValueError, match="unique"):
        patternCounts(duplicates, [0])

    with pytest.raises(ValueError, match="unique"):
        patternCounts([0], duplicates)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("branches", [1, 2, 3, 4, 243])
def testEntropyOfAUniformSplitIsLog2OfTheBranchCount(branches):
    counts = _counts([1] * branches)

    entropy = entropyFromCounts(counts)

    assert entropy[0] == pytest.approx(np.log2(branches))


def testEntropyOfASkewedSplitMatchesTheFormula():
    entropy = entropyFromCounts(_counts([3, 1]))

    expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert entropy[0] == pytest.approx(expected)


def testEntropyIgnoresEmptyBins():
    withEmptyBins = _counts([0, 3, 0, 1, 0])

    assert entropyFromCounts(withEmptyBins)[0] == pytest.approx(
        entropyFromCounts(_counts([3, 1]))[0]
    )


def testEntropyOfSeveralRowsIsComputedRowByRow():
    entropy = entropyFromCounts(_counts([1], [1, 1], [1, 1, 1, 1], [2, 2]))

    assert entropy.tolist() == pytest.approx([0.0, 1.0, 2.0, 1.0])


def testEntropyDoesNotMutateTheCounts():
    counts = _counts([3, 1])
    before = counts.copy()

    entropyFromCounts(counts)

    assert np.array_equal(counts, before)


# ---------------------------------------------------------------------------
# Expected remaining candidates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1], 1.0),
        ([1, 1], 1.0),
        ([2, 2], 2.0),
        ([3, 1], 2.5),
        ([2, 1], 5.0 / 3.0),
        ([1, 1, 1, 1], 1.0),
    ],
)
def testExpectedRemainingMatchesHandCounts(values, expected):
    assert expectedRemainingFromCounts(_counts(values))[0] == pytest.approx(expected)


def testExpectedRemainingMatchesItsDefinitionOnARealHistogram():
    guesses = _indices(["raise", "slate", "eerie"])
    candidates = _indices(["crate", "trace", "abbey", "sassy", "mucky"])

    counts = patternCounts(guesses, candidates)
    expected = (counts.astype(float) ** 2).sum(axis=1) / counts.sum(axis=1)

    assert expectedRemainingFromCounts(counts) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The Jensen gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values", [[1], [1, 1], [2, 2], [1, 1, 1, 1]])
def testJensenGapIsZeroForUniformBranches(values):
    gap = jensenGapFromCounts(_counts(values))

    assert gap[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([3, 1], 0.1332062193464951),
        ([2, 1], 0.0702989274995393),
    ],
)
def testJensenGapMatchesTheDerivedValue(values, expected):
    assert jensenGapFromCounts(_counts(values))[0] == pytest.approx(expected)


def testJensenGapEqualsEntropyMinusItsLowerBound():
    counts = _counts([7, 3, 2, 1])
    entropy = entropyFromCounts(counts)[0]
    expected = expectedRemainingFromCounts(counts)[0]
    total = counts.sum(axis=1)[0]

    assert jensenGapFromCounts(counts)[0] == pytest.approx(
        entropy - np.log2(total / expected)
    )


def testJensenGapIsNotTheSignedLogMeanDifference():
    # Two equal branches: H = 1, E[remaining] = 1, so log2(E) - H = -1. The true Jensen
    # gap is zero, because the branch sizes are perfectly uniform.
    counts = _counts([1, 1])

    assert jensenGapFromCounts(counts)[0] == pytest.approx(0.0, abs=1e-12)


def testJensenGapIsNonNegativeOnRealHistograms():
    guesses = _indices(["raise", "slate", "crane", "eerie", "abbey"])
    candidates = np.arange(200, dtype=np.int64)

    stats = guessStatistics(guesses, candidates)

    assert np.all(stats.jensenGap >= -1e-12)


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def testGuessStatisticsAgreesWithTheIndividualFunctions():
    guesses = _indices(["raise", "slate", "crane"])
    candidates = _indices(["crate", "trace", "abbey", "sassy", "mucky", "eerie"])

    stats = guessStatistics(guesses, candidates)

    assert stats.candidateCount == len(candidates)
    assert np.array_equal(stats.guesses, guesses)
    assert np.array_equal(stats.counts, patternCounts(guesses, candidates))
    assert stats.entropies == pytest.approx(entropyFromCounts(stats.counts))
    assert stats.expectedRemaining == pytest.approx(
        expectedRemainingFromCounts(stats.counts)
    )
    assert stats.jensenGap == pytest.approx(jensenGapFromCounts(stats.counts))


def testGuessStatisticsReportsBranchDiagnostics():
    guesses = _indices(["crane", "raise"])
    candidates = _indices(["crate", "trace", "abbey", "sassy", "mucky"])
    slow = _slowCounts(["crane", "raise"], ["crate", "trace", "abbey", "sassy", "mucky"])

    stats = guessStatistics(guesses, candidates)

    assert stats.nonEmptyBranches.tolist() == (slow > 0).sum(axis=1).tolist()
    assert stats.largestBranch.tolist() == slow.max(axis=1).tolist()


def testGuessStatisticsOnTheFullAnswerSetReproducesTheKnownRaiseFigures():
    # These are the root figures quoted in attempt3.md §2.2. They are pinned here because
    # a silent change to the feedback table would change every later ranking.
    stats = guessStatistics([_index("raise")], np.arange(len(loadAnswers())))

    assert stats.nonEmptyBranches[0] == 132
    assert stats.largestBranch[0] == 168
    assert stats.entropies[0] == pytest.approx(5.8779, abs=1e-4)
    assert stats.expectedRemaining[0] == pytest.approx(61.00, abs=0.01)


def testGuessStatisticsOnSlateShowsTheCorrectedJensenGap():
    stats = guessStatistics([_index("slate")], np.arange(len(loadAnswers())))

    assert stats.entropies[0] == pytest.approx(5.8558, abs=1e-4)
    assert stats.expectedRemaining[0] == pytest.approx(71.57, abs=0.01)
    # The signed log2(E) - H is about 0.31. The true Jensen gap is about 0.84.
    assert stats.jensenGap[0] == pytest.approx(0.84, abs=0.01)


def testGuessStatisticsOfNoGuessesReturnsEmptyArrays():
    stats = guessStatistics([], [0, 1, 2])

    assert stats.counts.shape == (0, PATTERN_COUNT)
    assert stats.entropies.shape == (0,)
    assert stats.expectedRemaining.shape == (0,)
    assert stats.jensenGap.shape == (0,)
    assert stats.nonEmptyBranches.shape == (0,)
    assert stats.largestBranch.shape == (0,)


def testGuessStatisticsRejectsAnEmptyCandidateSet():
    with pytest.raises(ValueError, match="at least one"):
        guessStatistics([0, 1], [])


# ---------------------------------------------------------------------------
# Malformed counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "badCounts",
    [
        np.zeros(PATTERN_COUNT, dtype=np.int64),
        np.zeros((2, PATTERN_COUNT - 1), dtype=np.int64),
        np.ones((1, PATTERN_COUNT), dtype=float),
        np.ones((1, PATTERN_COUNT), dtype=bool),
        -np.ones((1, PATTERN_COUNT), dtype=np.int64),
        np.zeros((1, PATTERN_COUNT), dtype=np.int64),
    ],
)
def testStatisticsRejectMalformedCounts(badCounts):
    for function in (
        entropyFromCounts,
        expectedRemainingFromCounts,
        jensenGapFromCounts,
    ):
        with pytest.raises(ValueError):
            function(badCounts)


def testStatisticsRejectAValidRowNextToAnEmptyRow():
    counts = np.zeros((2, PATTERN_COUNT), dtype=np.int64)
    counts[0, 0] = 1

    with pytest.raises(ValueError, match="at least one candidate"):
        entropyFromCounts(counts)


def testStatisticsOfNoRowsReturnEmptyArrays():
    counts = np.zeros((0, PATTERN_COUNT), dtype=np.int64)

    assert entropyFromCounts(counts).shape == (0,)
    assert expectedRemainingFromCounts(counts).shape == (0,)
    assert jensenGapFromCounts(counts).shape == (0,)
