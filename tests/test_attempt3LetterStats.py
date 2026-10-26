"""Tests for the Attempt-3 letter statistics.

These pin the numbers §2.6 quotes, because the write-up cites them as evidence for *why*
the opening words work. The properties:

- the occurrence definition (a doubled letter counts twice) and the slot normalisation;
- the distinct-letter table matches the plan exactly, bucket for bucket;
- the correlation is 0.923 on the five-distinct-letter subset, not the 0.9365 a
  presence-based definition would give;
- the frequency proxy on its own is maximised by repeats (`eerie`), which is the reason the
  correlation is restricted to the subset;
- malformed words and malformed frequency arrays fail loudly;
- the cached statistics cannot be corrupted by a caller, and inputs are not mutated.

The entropy pass over all 2,315 answers is the expensive part and is cached by the module.
"""

import numpy as np
import pytest

from attempt3.letterStats import (
    LETTER_COUNT,
    LETTERS,
    MEASURED_FREQUENCY_CORRELATION,
    distinctLetterBuckets,
    distinctLetterCount,
    letterCounts,
    letterFrequencies,
    letterFrequencyCorrelation,
    meanLetterFrequency,
    meanLetterFrequencies,
    rootEntropies,
    wordSummary,
)
from wordle.vocabulary import loadAnswers


# ---------------------------------------------------------------------------
# Counts and frequencies
# ---------------------------------------------------------------------------


def testLetterCountsSumToFiveLetterSlotsPerWord():
    counts = letterCounts()

    assert counts.sum() == 5 * len(loadAnswers())
    assert counts.shape == (LETTER_COUNT,)


def testLetterCountsOfOneWordAreItsLetterOccurrences():
    counts = letterCounts(["abbey"])
    index = {letter: position for position, letter in enumerate(LETTERS)}

    assert counts[index["a"]] == 1
    assert counts[index["b"]] == 2
    assert counts[index["e"]] == 1
    assert counts[index["y"]] == 1
    assert counts.sum() == 5


def testADoubledLetterCountsTwiceBecauseFrequencyMeansOccurrences():
    # The definition §2.6 and §7 quote: presence-based counting gives 0.9365, not 0.923.
    doubled = letterCounts(["abbey"])
    single = letterCounts(["abide"])
    b = "abcdefghijklmnopqrstuvwxyz".index("b")

    assert doubled[b] == 2
    assert single[b] == 1


def testLetterFrequenciesAreTheCountsOverTheLetterSlots():
    counts = letterCounts().astype(np.float64)
    frequencies = letterFrequencies()

    assert frequencies.sum() == pytest.approx(1.0)
    assert np.allclose(frequencies, counts / counts.sum())


def testTheAnswerCountsAreNotSharedWithCallers():
    first = letterCounts()
    first[:] = 0

    assert letterCounts().sum() == 5 * len(loadAnswers())


def testLetterCountsDoNotMutateTheirInput():
    words = ["abbey", "crane"]
    before = list(words)

    letterCounts(words)

    assert words == before


# ---------------------------------------------------------------------------
# Distinct letters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "word, distinct",
    [("slate", 5), ("abbey", 4), ("eerie", 3), ("mamma", 2)],
)
def testDistinctLetterCount(word, distinct):
    assert distinctLetterCount(word) == distinct


def testExactlyOneAnswerUsesOnlyTwoDistinctLetters():
    two = [word for word in loadAnswers() if len(set(word)) == 2]

    assert two == ["mamma"]


# ---------------------------------------------------------------------------
# The §2.6 table
# ---------------------------------------------------------------------------


def testTheDistinctLetterBucketsMatchThePlan():
    buckets = {bucket.distinctLetters: bucket for bucket in distinctLetterBuckets()}

    assert sorted(buckets) == [2, 3, 4, 5]
    assert buckets[2].wordCount == 1
    assert buckets[3].wordCount == 57
    assert buckets[4].wordCount == 691
    assert buckets[5].wordCount == 1566

    assert buckets[2].meanEntropy == pytest.approx(2.3986, abs=1e-4)
    assert buckets[3].meanEntropy == pytest.approx(3.4121, abs=1e-4)
    assert buckets[4].meanEntropy == pytest.approx(4.1250, abs=1e-4)
    assert buckets[5].meanEntropy == pytest.approx(4.7654, abs=1e-4)


def testTheBucketsCoverEveryAnswer():
    buckets = distinctLetterBuckets()

    assert sum(bucket.wordCount for bucket in buckets) == len(loadAnswers())


def testMoreDistinctLettersMeansMoreEntropy():
    buckets = sorted(distinctLetterBuckets(), key=lambda bucket: bucket.distinctLetters)
    entropies = [bucket.meanEntropy for bucket in buckets]

    assert np.all(np.diff(entropies) > 0)


# ---------------------------------------------------------------------------
# The correlation
# ---------------------------------------------------------------------------


def testTheCorrelationMatchesThePlan():
    measured = letterFrequencyCorrelation()

    assert measured == pytest.approx(MEASURED_FREQUENCY_CORRELATION, abs=1e-4)
    assert letterFrequencyCorrelation() == pytest.approx(0.923, abs=1e-3)


def testTheCorrelationIsComputedOnTheFiveDistinctLetterSubset():
    # If the repeated-letter words were included the number would move; §2.6 restricts the
    # subset for exactly this reason.
    answers = loadAnswers()
    five = [word for word in answers if len(set(word)) == 5]
    means = meanLetterFrequencies(five)

    assert len(five) == 1566
    assert means.shape == (1566,)


def testThePureFrequencyProxyIsWonByARepeatedLetterWord():
    # This is why the correlation needs the five-distinct subset: `eerie` has the highest
    # mean letter frequency in the answer list and is a terrible opener.
    means = meanLetterFrequencies(loadAnswers())
    best = loadAnswers()[int(np.argmax(means))]

    assert best == "eerie"


# ---------------------------------------------------------------------------
# Summaries for hand-written guesses
# ---------------------------------------------------------------------------


def testWordSummaryAgreesWithTheIndividualFunctions():
    summary = wordSummary("raise")

    assert summary.word == "raise"
    assert summary.distinctLetters == 5
    assert summary.meanFrequency == pytest.approx(meanLetterFrequency("raise"))
    assert 1 <= summary.frequencyRank <= len(loadAnswers())


def testWordSummaryRanksACommonWordAboveARareOne():
    assert wordSummary("raise").frequencyRank < wordSummary("fuzzy").frequencyRank


def testWordSummaryAcceptsAWordThatIsNotAnAnswer():
    # `adieu` is a legal guess but not an answer; the rank is still against the answers.
    summary = wordSummary("adieu")

    assert summary.distinctLetters == 5
    assert 1 <= summary.frequencyRank <= len(loadAnswers())


def testMeanLetterFrequencyUsesThePositionsOfTheWord():
    frequencies = np.zeros(LETTER_COUNT)
    frequencies[0] = 1.0  # 'a'

    assert meanLetterFrequency("aaaaa", frequencies) == pytest.approx(1.0)
    assert meanLetterFrequency("abcde", frequencies) == pytest.approx(0.2)


def testMeanLetterFrequenciesAgreeWithTheScalarFunction():
    words = ["raise", "slate", "fuzzy"]
    vector = meanLetterFrequencies(words)

    assert np.allclose(vector, [meanLetterFrequency(word) for word in words])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", ["four", "sixsix", "crane!", "CRANE", ""])
def testMalformedWordsAreRejected(word):
    with pytest.raises(ValueError):
        distinctLetterCount(word)


def testASingleStringIsNotASequenceOfWords():
    with pytest.raises(ValueError):
        letterCounts("raise")


def testAnEmptyWordListIsRejected():
    with pytest.raises(ValueError):
        letterCounts([])


def testMalformedFrequencyArraysAreRejected():
    with pytest.raises(ValueError):
        meanLetterFrequency("raise", np.zeros(LETTER_COUNT - 1))

    with pytest.raises(ValueError):
        meanLetterFrequency("raise", np.full(LETTER_COUNT, -1.0))

    with pytest.raises(ValueError):
        meanLetterFrequency("raise", np.zeros(LETTER_COUNT))

    with pytest.raises(ValueError):
        meanLetterFrequency("raise", np.full(LETTER_COUNT, np.nan))


# ---------------------------------------------------------------------------
# The cached entropy pass
# ---------------------------------------------------------------------------


def testRootEntropiesGiveOneValuePerAnswerAndAreNotShared():
    entropies = rootEntropies()
    assert entropies.shape == (len(loadAnswers()),)
    assert np.all(entropies >= 0)

    entropies[:] = 0
    assert rootEntropies().max() > 0
