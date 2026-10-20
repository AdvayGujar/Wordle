"""Tests for the Attempt-1 candidate-relative feature builder."""

import pytest

from attempt1.features import FEATURE_WIDTH, MAX_ROUNDS, ROUND_FEATURES, candidateFeatures
from wordle.feedback import feedback


def evidenceForOneRound(secret = "stare", guess = "crane"):
    return [(guess, feedback(guess, secret))]


def testWidthIsConstantAtEveryRoundCount():
    for rounds in range(0, MAX_ROUNDS + 1):
        evidence = [("crane", (0, 1, 2, 0, 2)) for _ in range(rounds)]
        assert len(candidateFeatures(evidence, "stare")) == FEATURE_WIDTH


def testUnplayedRoundsArePadding():
    vector = candidateFeatures(evidenceForOneRound(), "stare", MAX_ROUNDS)
    assert vector[ROUND_FEATURES:] == [-1] * (ROUND_FEATURES * (MAX_ROUNDS - 1))
    assert len(vector) == FEATURE_WIDTH


def testFeaturesDependOnTheCandidate():
    evidence = evidenceForOneRound()
    assert candidateFeatures(evidence, "crane") != candidateFeatures(evidence, "stare")


def testFeaturesArePure():
    evidence = evidenceForOneRound()
    before = list(evidence)
    assert candidateFeatures(evidence, "stare") == candidateFeatures(evidence, "stare")
    assert evidence == before


def testTooManyRoundsRaises():
    evidence = [("crane", (0, 0, 0, 0, 0))] * (MAX_ROUNDS + 1)
    with pytest.raises(ValueError):
        candidateFeatures(evidence, "stare")


@pytest.mark.parametrize("bad", ["cran", "cranes", "cr4ne", "CRANE", ""])
def testMalformedCandidateRaises(bad):
    with pytest.raises(ValueError):
        candidateFeatures(evidenceForOneRound(), bad)
