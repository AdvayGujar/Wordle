"""Tests for the candidate-row builder used by Attempt 1."""

from random import Random

from attempt1.common import (
    candidateMatrix,
    candidateRows,
    statesFor,
    testSecrets as heldOutTestSecrets,
    trainSecrets,
)
from attempt1.features import FEATURE_WIDTH
from wordle.feedback import candidatesMatching
from wordle.vocabulary import loadAnswers


def testCandidateRowsHaveTheRightShapeAndBothLabels():
    states = statesFor(["stare"], 1)
    features, labels, groups, candidates = candidateRows(states, Random(0), 5, 5)

    assert features.shape[1] == FEATURE_WIDTH
    assert len(features) == len(labels) == len(groups) == len(candidates)
    assert set(labels) == {0, 1}
    assert all(group == 0 for group in groups)


def testPositiveRowsAreTrulyConsistentAndNegativesAreNot():
    states = statesFor(["stare"], 1)
    evidence = states[0][1]
    consistent = set(candidatesMatching(evidence, loadAnswers()))
    _, labels, _, candidates = candidateRows(states, Random(0), 5, 5)

    for candidate, label in zip(candidates, labels):
        assert (candidate in consistent) == bool(label)


def testCandidateMatrixCoversEveryAnswer():
    states = statesFor(["stare"], 2)
    matrix = candidateMatrix(states[0][1])
    assert matrix.shape == (len(loadAnswers()), FEATURE_WIDTH)


def testTrainAndTestSecretsAreDisjoint():
    assert set(trainSecrets()).isdisjoint(set(heldOutTestSecrets()))
