"""Tests for the exhaustive opener search.

The search is a measurement instrument, so what matters is that it measures the right
thing: the right opener, the right number of games, and a ranking that is sorted by the
number of guesses it claims to be sorted by.
"""

import numpy as np
import pytest

from attempt2.openerSearch import evaluateOpener, ranked, refine, search
from wordle.vocabulary import loadAnswers


def testEvaluatingOneOpenerReturnsAUsableMeasurement():
    index = loadAnswers().index("slate")

    evaluated, average, winRate, standardError = evaluateOpener((index, 20, 500000))

    assert evaluated == index
    assert 1.0 <= average <= 6.0
    assert 0.0 <= winRate <= 1.0
    assert standardError > 0.0


def testTheSameOpenerAlwaysMeasuresTheSame():
    index = loadAnswers().index("slate")

    first = evaluateOpener((index, 20, 500000))
    second = evaluateOpener((index, 20, 500000))

    assert first == second


def testDifferentSeedBasesMeasureDifferentSamples():
    index = loadAnswers().index("slate")

    first = evaluateOpener((index, 40, 500000))
    second = evaluateOpener((index, 40, 900000))

    assert first[1] != second[1]


def testTheSearchCoversEveryAnswer():
    answers = list(loadAnswers())

    results = search(answers[:6], games=3, workers=1)

    assert len(results) == 6
    assert {row[0] for row in results} == set(answers[:6])


def testTheRankingIsSortedByTheNumberOfGuesses():
    answers = list(loadAnswers())

    results = search(answers[:8], games=5, workers=1)
    ordered = ranked(results)

    averages = [row[1] for row in ordered]
    assert averages == sorted(averages)


def testTheSearchDoesNotDependOnTheNumberOfWorkers():
    answers = list(loadAnswers())

    single = search(answers[:4], games=5, workers=1)
    parallel = search(answers[:4], games=5, workers=2)

    assert single == parallel


def testTheRefinementMeasuresTheCandidatesItIsGiven():
    answers = list(loadAnswers())
    candidates = ["slate", "crane"]

    results = refine(candidates, games=5, workers=1)

    assert {row[0] for row in results} == set(candidates)


def testAGoodOpenerBeatsABadOne():
    # The whole point of the module: the measurement has to be able to tell them apart.
    answers = list(loadAnswers())

    results = dict(
        (word, average)
        for word, average, _, _ in refine(["slate", "jazzy"], games=120, workers=1)
    )

    assert results["slate"] < results["jazzy"]
