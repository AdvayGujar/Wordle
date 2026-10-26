"""Tests for the Attempt-3 decision criterion and the rejected expected-guesses model.

The score decides every guess the strategy makes, so the properties pinned here are the
ones that would otherwise fail silently:

- the f estimate has the calibration it claims (f(0) = 1, f(11.5) ~ 3.5, increasing);
- the two worked examples of `attempt3.md` §2.7 reproduce exactly;
- a full score agrees with a slow `feedback()`-per-pair reference, including repeated
  letters, so a wrong histogram or a wrong win chance cannot pass unnoticed;
- a candidate beats an equally informative probe, and a singleton set prefers the answer;
- the rejected c(m) closed form matches its own definition and brute force;
- the all-green branch costs no further guesses, which is the correction §2.5 calls out;
- malformed input fails loudly instead of producing a plausible ranking.

Nothing here plays a game; the pickers and the experiment are tested in their own files.
"""

import math

import numpy as np
import pytest

from attempt3.objective import (
    GREEN_CODE,
    cappedGuesses,
    cappedGuessesBySummation,
    entropyToExpectedScore,
    expectedGuesses,
    expectedGuessesFromCounts,
    remainingGuessEstimate,
    scoreGuesses,
)
from attempt3.patterns import guessStatistics, patternCounts
from wordle.feedback import feedback
from wordle.feedbackTable import PATTERN_COUNT, patternCode
from wordle.vocabulary import answersVocab, loadAnswers


def _index(word):
    return answersVocab().indexOf(word)


def _indices(words):
    return np.array([_index(word) for word in words], dtype=np.int64)


def _slowScore(guess, candidates):
    """The §2.7 score computed from `feedback()` one pair at a time."""
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


def _counts(*rows):
    """A well-formed histogram from per-row lists of bin values."""
    result = np.zeros((len(rows), PATTERN_COUNT), dtype=np.int64)
    for row, values in enumerate(rows):
        result[row, : len(values)] = values
    return result


# ---------------------------------------------------------------------------
# The f estimate
# ---------------------------------------------------------------------------


def testTheEstimateIsOneWhenNothingIsLeft():
    assert remainingGuessEstimate(0.0) == pytest.approx(1.0)


def testTheEstimateIsCalibratedToAFullBoard():
    # §2.7: log2(2315) ~ 11.5 bits should look like about 3.5 further guesses.
    assert remainingGuessEstimate(11.5) == pytest.approx(3.50, abs=0.01)


def testTheEstimateIncreasesWithTheUncertaintyLeft():
    bits = np.linspace(0.0, 11.5, 50)

    assert np.all(np.diff(remainingGuessEstimate(bits)) > 0)


def testTheEstimateRejectsNegativeOrNonFiniteInput():
    with pytest.raises(ValueError):
        remainingGuessEstimate(-0.1)

    with pytest.raises(ValueError):
        remainingGuessEstimate(np.inf)


# ---------------------------------------------------------------------------
# The two worked examples of §2.7
# ---------------------------------------------------------------------------


def testTwoCandidatesGuessingOneScoresOneAndAHalf():
    # p = 0.5, H = H0 = 1, so score = 0.5 + 0.5 * (1 + f(0)) = 1.5.
    assert entropyToExpectedScore(1.0, 1.0, 0.5) == pytest.approx(1.5)


def testTwoCandidatesAProbeThatSplitsThemScoresTwo():
    # p = 0, H = H0 = 1, so score = 1 + f(0) = 2.0. The candidate wins, as it should.
    assert entropyToExpectedScore(1.0, 1.0, 0.0) == pytest.approx(2.0)


def testOneHundredCandidatesTheWorkedExampleNumbers():
    current = math.log2(100)

    probe = entropyToExpectedScore(6.0, current, 0.0)
    candidate = entropyToExpectedScore(1.0, current, 0.01)

    assert probe == pytest.approx(2.44, abs=0.01)
    assert candidate == pytest.approx(3.69, abs=0.01)
    assert probe < candidate


# ---------------------------------------------------------------------------
# The scoring over a real feasible set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "guesses, candidates",
    [
        (["raise"], ["crane"]),
        (["raise", "slate"], ["crate", "trace", "abbey", "sassy"]),
        (["crane", "eerie", "abbey"],
         ["crate", "mucky", "eerie", "bobby", "trace", "abbey"]),
    ],
)
def testScoresAgreeWithTheSlowFeedbackReference(guesses, candidates):
    scored = scoreGuesses(_indices(guesses), _indices(candidates))
    expected = [_slowScore(guess, candidates) for guess in guesses]

    assert np.allclose(scored.scores, expected)


def testTheBestScoringGuessMatchesTheSlowReference():
    guesses = ["crane", "eerie", "abbey", "slate", "mucky"]
    candidates = ["crate", "mucky", "eerie", "bobby", "trace", "abbey"]

    scored = scoreGuesses(_indices(guesses), _indices(candidates))
    expected = [_slowScore(guess, candidates) for guess in guesses]

    assert list(scored.guesses) [int(np.argmin(scored.scores))] == _index(
        guesses[int(np.argmin(expected))]
    )


def testACandidateBeatsAnEquallyInformativeProbe():
    # Same information, different win chance: that is the whole point of the p term.
    current = math.log2(10)

    candidate = entropyToExpectedScore(2.0, current, 0.1)
    probe = entropyToExpectedScore(2.0, current, 0.0)

    assert candidate < probe


def testASingletonFeasibleSetPrefersTheAnswer():
    scored = scoreGuesses(_indices(["raise", "slate"]), _indices(["raise"]))

    best = int(np.argmin(scored.scores))

    assert bool(scored.isCandidate[best])
    assert scored.guesses[best] == _index("raise")
    assert scored.scores[best] == pytest.approx(1.0)


def testWinChanceIsOneOverTheCandidateCountForCandidatesAndZeroOtherwise():
    candidates = ["crate", "trace", "abbey", "sassy"]
    scored = scoreGuesses(_indices(["raise", "abbey"]), _indices(candidates))

    assert list(scored.isCandidate) == [False, True]
    assert scored.winChance[0] == 0.0
    assert scored.winChance[1] == pytest.approx(0.25)


def testInformationLeftIsTheCurrentEntropyMinusTheGuessEntropy():
    scored = scoreGuesses(_indices(["raise", "slate"]), _indices(["crate", "trace"]))

    assert np.allclose(
        scored.informationLeft, math.log2(2) - scored.entropies
    )


def testAnEmptyGuessSetProducesNoScores():
    scored = scoreGuesses([], _indices(["raise"]))

    assert scored.scores.shape == (0,)
    assert scored.guesses.shape == (0,)


def testScoringDoesNotMutateItsInputs():
    guesses = _indices(["raise", "slate"])
    candidates = _indices(["crate", "trace"])
    guessesBefore = guesses.copy()
    candidatesBefore = candidates.copy()

    scoreGuesses(guesses, candidates)

    assert np.array_equal(guesses, guessesBefore)
    assert np.array_equal(candidates, candidatesBefore)


def testScoringIsDeterministic():
    first = scoreGuesses(_indices(["raise", "slate"]), _indices(["crate", "trace"]))
    second = scoreGuesses(_indices(["raise", "slate"]), _indices(["crate", "trace"]))

    assert np.array_equal(first.scores, second.scores)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def testAnEntropyAboveTheFeasibleSetEntropyIsRejected():
    with pytest.raises(ValueError):
        entropyToExpectedScore(2.0, 1.0, 0.5)


def testAWinChanceOutsideZeroToOneIsRejected():
    with pytest.raises(ValueError):
        entropyToExpectedScore(0.5, 1.0, 1.5)

    with pytest.raises(ValueError):
        entropyToExpectedScore(0.5, 1.0, -0.1)


def testNegativeOrNonFiniteEntropiesAreRejected():
    with pytest.raises(ValueError):
        entropyToExpectedScore(-0.1, 1.0, 0.0)

    with pytest.raises(ValueError):
        entropyToExpectedScore(np.nan, 1.0, 0.0)


def testAnEmptyCandidateSetIsRejected():
    with pytest.raises(ValueError):
        scoreGuesses(_indices(["raise"]), [])


def testAWinChanceThatCannotBroadcastIsRejected():
    with pytest.raises(ValueError):
        entropyToExpectedScore(np.array([0.5, 0.6]), 1.0, np.array([0.0, 0.0, 0.0]))


def testMalformedHistogramsAreRejected():
    with pytest.raises(ValueError):
        expectedGuessesFromCounts(np.zeros((1, PATTERN_COUNT - 1), dtype=np.int64))

    with pytest.raises(ValueError):
        expectedGuessesFromCounts(np.zeros((1, PATTERN_COUNT), dtype=np.float64))

    with pytest.raises(ValueError):
        expectedGuessesFromCounts(np.zeros((1, PATTERN_COUNT), dtype=np.int64))

    with pytest.raises(ValueError):
        expectedGuessesFromCounts(
            np.full((1, PATTERN_COUNT), -1, dtype=np.int64)
        )


def testASizeBelowOneIsRejected():
    with pytest.raises(ValueError):
        cappedGuesses(0)

    with pytest.raises(ValueError):
        cappedGuessesBySummation(0)


# ---------------------------------------------------------------------------
# The rejected model: c(m)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size, expected",
    [(1, 1.0), (2, 1.5), (4, 2.5), (5, 3.0), (10, 4.0), (100, 4.9)],
)
def testTheCappedMeanMatchesItsClosedForm(size, expected):
    assert float(cappedGuesses(size)) == pytest.approx(expected)
    assert float(cappedGuessesBySummation(size)) == pytest.approx(expected)


def testTheCappedMeanMatchesBruteForce():
    for size in range(1, 41):
        brute = sum(min(turn, 5) for turn in range(1, size + 1)) / size

        assert float(cappedGuesses(size)) == pytest.approx(brute)
        assert float(cappedGuessesBySummation(size)) == pytest.approx(brute)


def testTheCappedMeanIsIncreasing():
    sizes = np.arange(1, 60)

    assert np.all(np.diff(cappedGuesses(sizes)) > 0)


# ---------------------------------------------------------------------------
# The rejected model: the functional
# ---------------------------------------------------------------------------


def testTheAllGreenBranchCostsNoFurtherGuess():
    # C = {raise, slate}, guess raise: half the time the game ends now, half the time one
    # further guess names slate. So 1 + 0.5*0 + 0.5*1 = 1.5, not 2.
    value = expectedGuesses(_indices(["raise"]), _indices(["raise", "slate"]))

    assert value[0] == pytest.approx(1.5)


def testAProbeGetsNoDiscount():
    value = expectedGuesses(_indices(["crane"]), _indices(["raise", "slate"]))

    # Two singleton branches, no all-green one: 1 + 0.5*1 + 0.5*1 = 2.
    assert value[0] == pytest.approx(2.0)


def testTheFunctionalEqualsTheNaiveSumMinusTheGreenCorrection():
    guesses = _indices(["raise", "abbey", "crane"])
    candidates = _indices(["crate", "mucky", "eerie", "bobby", "trace", "abbey"])

    counts = patternCounts(guesses, candidates)
    sizes = counts.astype(np.float64)
    totals = sizes.sum(axis=1)
    costs = np.zeros_like(sizes)
    positive = sizes > 0
    costs[positive] = cappedGuesses(sizes[positive])

    naive = 1.0 + (sizes / totals[:, None] * costs).sum(axis=1)
    greens = sizes[:, GREEN_CODE]

    assert np.allclose(expectedGuessesFromCounts(counts), naive - greens / totals)


def testTheRejectedModelRanksTheOpenersAsMeasured():
    """§2.5.1, re-measured at Gate A over all 2,315 answers.

    The functional's top three are `trace`, `slate` and `raise` (ranks 1, 2 and 18), so it
    does disagree with entropy - `raise` is entropy's best and only 18th here. But it is
    not a *skew-loving* ranking: `fuzzy`, `vivid` and `mummy` land at ranks 2,314, 2,303
    and 2,306, and the Spearman correlation with entropy over all openers is 0.95. The
    model is kept because it is the documented negative result, not as a ranking.
    """
    guesses = _indices(["raise", "slate", "trace"])
    candidates = np.arange(len(loadAnswers()))

    functional = expectedGuesses(guesses, candidates)
    entropy = guessStatistics(guesses, candidates).entropies

    assert functional[2] < functional[1] < functional[0]  # trace < slate < raise
    assert entropy[0] > entropy[1] > entropy[2]  # raise > slate > trace
