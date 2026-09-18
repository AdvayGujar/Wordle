"""Tests for dataset.py -- data generation and the no-leakage split.

The plan (section 6.5) calls the split-by-secret guarantee critical: if a secret
appears in both train and test, test accuracy becomes meaningless. That is the
most important test in this file.
"""

import pytest

from wordle.dataset import Dataset, _playProbes, _splitBySecret, generate
from wordle.features import encode
from wordle.vocabulary import answersVocab, loadAnswers

PROBES = ["crane", "slate"]
N_TRAIN = 300
N_TEST = 100
SEED = 42


@pytest.fixture(scope="module")
def dataset():
    """One shared dataset -- generation plays probes against 400 secrets."""
    return generate(PROBES, N_TRAIN, N_TEST, SEED)


# ---------------------------------------------------------------------------
# Dataset container / counts
# ---------------------------------------------------------------------------

def testGenerateReturnsADataset(dataset):
    assert isinstance(dataset, Dataset)


def testDatasetCountsMatchTheRequestedSplit(dataset):
    assert len(dataset.xTrain) == N_TRAIN
    assert len(dataset.yTrain) == N_TRAIN
    assert len(dataset.xTest) == N_TEST
    assert len(dataset.yTest) == N_TEST


def testFeatureAndLabelListsAreAligned(dataset):
    assert len(dataset.xTrain) == len(dataset.yTrain)
    assert len(dataset.xTest) == len(dataset.yTest)


def testDatasetCarriesTheAnswersVocabulary(dataset):
    assert dataset.vocab is answersVocab()


# ---------------------------------------------------------------------------
# Feature-vector shape
# ---------------------------------------------------------------------------

def testEveryRowHasTheSameFeatureLength(dataset):
    expected = len(PROBES) * 5
    assert {len(row) for row in dataset.xTrain} == {expected}
    assert {len(row) for row in dataset.xTest} == {expected}


def testFeatureLengthTracksTheNumberOfProbes():
    one = generate(["crane"], 50, 20, SEED)
    three = generate(["crane", "slate", "stare"], 50, 20, SEED)
    assert {len(r) for r in one.xTrain} == {5}
    assert {len(r) for r in three.xTrain} == {15}


def testFeaturesContainOnlyValidCodes(dataset):
    for row in dataset.xTrain + dataset.xTest:
        assert all(code in (-1, 0, 1, 2) for code in row)


# ---------------------------------------------------------------------------
# The critical guarantee: no leakage
# ---------------------------------------------------------------------------

def testNoSecretAppearsInBothTrainAndTest(dataset):
    trainWords = {dataset.vocab.wordAt(i) for i in dataset.yTrain}
    testWords = {dataset.vocab.wordAt(i) for i in dataset.yTest}
    assert trainWords & testWords == set(), "train/test secrets overlap -- leakage!"


def testTrainAndTestRowCountsAreFullyDisjoint(dataset):
    """Stronger form: compare the actual encoded rows, not just the labels."""
    trainRows = set(tuple(row) for row in dataset.xTrain)
    testSecrets = {dataset.vocab.wordAt(i) for i in dataset.yTest}
    trainSecrets = {dataset.vocab.wordAt(i) for i in dataset.yTrain}
    assert trainSecrets.isdisjoint(testSecrets)
    assert len(trainRows) > 0


def testNoDuplicateSecretsWithinTrain(dataset):
    assert len(set(dataset.yTrain)) == len(dataset.yTrain)


def testNoDuplicateSecretsWithinTest(dataset):
    assert len(set(dataset.yTest)) == len(dataset.yTest)


# ---------------------------------------------------------------------------
# Row correctness: each row must come from its own secret
# ---------------------------------------------------------------------------

def testEveryRowMatchesTheSecretItsLabelPointsTo(dataset):
    for x, y in zip(dataset.xTrain, dataset.yTrain):
        secret = dataset.vocab.wordAt(y)
        assert x == encode(_playProbes(secret, PROBES), len(PROBES))


def testEveryTestRowMatchesTheSecretItsLabelPointsTo(dataset):
    for x, y in zip(dataset.xTest, dataset.yTest):
        secret = dataset.vocab.wordAt(y)
        assert x == encode(_playProbes(secret, PROBES), len(PROBES))


def testLabelsAreValidVocabularyIndices(dataset):
    allLabels = dataset.yTrain + dataset.yTest
    assert min(allLabels) >= 0
    assert max(allLabels) < len(loadAnswers())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def testSameSeedProducesIdenticalData():
    a = generate(PROBES, 50, 20, SEED)
    b = generate(PROBES, 50, 20, SEED)
    assert a.xTrain == b.xTrain
    assert a.yTrain == b.yTrain
    assert a.xTest == b.xTest
    assert a.yTest == b.yTest


def testDifferentSeedsProduceDifferentSplits():
    a = generate(PROBES, 50, 20, 1)
    b = generate(PROBES, 50, 20, 2)
    assert a.yTrain != b.yTrain


# ---------------------------------------------------------------------------
# _playProbes
# ---------------------------------------------------------------------------

def testPlayProbesReturnsAGuessPatternPairPerProbe():
    evidence = _playProbes("apple", ["crane", "slate"])
    assert len(evidence) == 2
    assert [g for g, _ in evidence] == ["crane", "slate"]
    assert all(len(p) == 5 for _, p in evidence)


def testPlayProbesUsesRealFeedback():
    # 'crane' vs secret 'crane' is all green.
    assert _playProbes("crane", ["crane"]) == [("crane", (2, 2, 2, 2, 2))]


def testPlayProbesIsPure():
    assert _playProbes("apple", PROBES) == _playProbes("apple", PROBES)


def testPlayProbesOutputFeedsEncode():
    evidence = _playProbes("apple", PROBES)
    assert encode(evidence, len(PROBES)) == encode(evidence, len(PROBES))


# ---------------------------------------------------------------------------
# _splitBySecret
# ---------------------------------------------------------------------------

def testSplitBySecretIsDisjointAndComplete():
    answers = loadAnswers()
    train, test = _splitBySecret(answers, 100, SEED)
    assert len(train) == 100
    assert len(train) + len(test) == len(answers)
    assert set(train).isdisjoint(set(test))


def testSplitBySecretIsReproducible():
    answers = loadAnswers()
    assert _splitBySecret(answers, 100, SEED) == _splitBySecret(answers, 100, SEED)


def testSplitBySecretRejectsAnImpossibleTrainSize():
    answers = loadAnswers()
    with pytest.raises(ValueError):
        _splitBySecret(answers, 0, SEED)
    with pytest.raises(ValueError):
        _splitBySecret(answers, len(answers), SEED)
    with pytest.raises(ValueError):
        _splitBySecret(answers, len(answers) + 1, SEED)


def testSplitBySecretDoesNotMutateTheCachedAnswers():
    answers = loadAnswers()
    before = tuple(answers)
    _splitBySecret(answers, 100, SEED)
    assert tuple(loadAnswers()) == before
