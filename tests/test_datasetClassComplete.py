"""Tests for the class-complete generator used by Attempt 1.

The key distinction from generateVariable(): a classifier cannot predict a class it has
never seen, so training covers every answer word and the held-out set is fresh EVIDENCE,
not unseen secrets.
"""

import pytest

from wordle.dataset import (
    Dataset,
    evidenceFor,
    generateClassComplete,
    selectTestSecrets,
    splitSecrets,
)
from wordle.features import roundBlockSize
from wordle.vocabulary import loadAnswers, answersVocab

MAX_ROUNDS = 5
N_TEST = 60
SEED = 42
MODE = "compact"
WIDTH = MAX_ROUNDS * roundBlockSize(MODE)


@pytest.fixture(scope="module")
def dataset():
    return generateClassComplete(3, N_TEST, SEED, MAX_ROUNDS, MODE)


def testReturnsADatasetWithTheRequestedCounts(dataset):
    assert isinstance(dataset, Dataset)
    assert len(dataset.xTrain) == len(loadAnswers())
    assert len(dataset.yTrain) == len(loadAnswers())
    assert len(dataset.xTest) == N_TEST
    assert len(dataset.yTest) == N_TEST


def testRecordsTheEvidenceSplitAndWidth(dataset):
    assert dataset.split == "evidence"
    assert dataset.encoder == "rounds"
    assert dataset.maxRounds == MAX_ROUNDS
    assert dataset.letterMode == MODE
    assert dataset.width == WIDTH


def testEveryClassAppearsInTrainingOnce(dataset):
    """This is the property the secret split could not provide."""
    labels = set(dataset.yTrain)
    assert labels == set(range(len(loadAnswers())))
    assert len(dataset.yTrain) == len(labels)


def testEveryTestLabelIsALearnedClass(dataset):
    assert set(dataset.yTest) <= set(dataset.yTrain)


def testEveryRowHasTheSameWidth(dataset):
    assert {len(row) for row in dataset.xTrain + dataset.xTest} == {WIDTH}


def testTestSecretsAreTheShuffledTail():
    answers = loadAnswers()
    selected = selectTestSecrets(N_TEST, SEED)
    assert len(selected) == N_TEST
    assert len(set(selected)) == N_TEST
    assert set(selected) <= set(answers)


def testSelectTestSecretsIsReproducibleAndRejectsBadCounts():
    assert selectTestSecrets(N_TEST, SEED) == selectTestSecrets(N_TEST, SEED)
    with pytest.raises(ValueError):
        selectTestSecrets(-1, SEED)
    with pytest.raises(ValueError):
        selectTestSecrets(len(loadAnswers()) + 1, SEED)


def testNoTestRowIsACopyOfItsOwnTrainingRow():
    """Different evidence for the same secret -- never the identical feature row."""
    vocab = answersVocab()

    for rounds in (1, 2, 3, 4, 5):
        ds = generateClassComplete(rounds, N_TEST, SEED, MAX_ROUNDS, MODE)
        trainRow = {label: tuple(row) for row, label in zip(ds.xTrain, ds.yTrain)}

        for row, label in zip(ds.xTest, ds.yTest):
            assert tuple(row) != trainRow[label], (
                "test row for %r is its training row (rounds=%d)"
                % (vocab.wordAt(label), rounds)
            )


def testTestRowsArePrefixesAcrossRoundCounts():
    """The 1-round and 5-round test rows describe the same games, cut off earlier."""
    short = generateClassComplete(1, N_TEST, SEED, MAX_ROUNDS, MODE)
    long = generateClassComplete(5, N_TEST, SEED, MAX_ROUNDS, MODE)
    block = roundBlockSize(MODE)

    longRow = {label: row for row, label in zip(long.xTest, long.yTest)}

    for row, label in zip(short.xTest, short.yTest):
        assert row[:block] == longRow[label][:block]


def testSameSeedProducesIdenticalData():
    a = generateClassComplete(3, N_TEST, SEED, MAX_ROUNDS, MODE)
    b = generateClassComplete(3, N_TEST, SEED, MAX_ROUNDS, MODE)
    assert a.xTrain == b.xTrain and a.yTrain == b.yTrain
    assert a.xTest == b.xTest and a.yTest == b.yTest


def testDifferentSeedsProduceDifferentData():
    a = generateClassComplete(3, N_TEST, 1, MAX_ROUNDS, MODE)
    b = generateClassComplete(3, N_TEST, 2, MAX_ROUNDS, MODE)
    assert a.xTrain != b.xTrain


def testACopyOfTheTrainingEvidenceIsRejected():
    """A picker that ignores the RNG makes test == train; that must fail loudly."""
    with pytest.raises(ValueError):
        generateClassComplete(1, 5, SEED, MAX_ROUNDS, MODE,
                              lambda roundIndex, rng, evidence: "crane")


def testRejectsAnImpossibleRoundCount():
    with pytest.raises(ValueError):
        generateClassComplete(0, N_TEST, SEED, MAX_ROUNDS, MODE)
    with pytest.raises(ValueError):
        generateClassComplete(6, N_TEST, SEED, MAX_ROUNDS, MODE)


def testSplitSecretsIsPublicAndDisjoint():
    train, test = splitSecrets(100, SEED)
    assert len(train) == 100
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == len(loadAnswers())


def testEvidenceForIsPrefixAndDeterministic():
    secret = loadAnswers()[0]
    one = evidenceFor(secret, 1, SEED, MAX_ROUNDS)
    five = evidenceFor(secret, 5, SEED, MAX_ROUNDS)
    assert one == five[:1]
    assert evidenceFor(secret, 5, SEED, MAX_ROUNDS) == five
    with pytest.raises(ValueError):
        evidenceFor(secret, 0, SEED, MAX_ROUNDS)
