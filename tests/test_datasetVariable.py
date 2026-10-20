"""Tests for generateVariable() -- varied-guess evidence with a stable width."""

import pytest

from wordle.dataset import Dataset, _secretRng, generateVariable
from wordle.features import ALPHABET, PADDING, roundBlockSize
from wordle.vocabulary import answersVocab, loadAnswers

MAX_ROUNDS = 5
N_TRAIN = 200
N_TEST = 60
SEED = 42
MODE = "compact"
WIDTH = MAX_ROUNDS * roundBlockSize(MODE)


@pytest.fixture(scope="module")
def dataset():
    return generateVariable(3, N_TRAIN, N_TEST, SEED, MAX_ROUNDS, MODE)


def guessLettersOf(row, roundIndex, mode = MODE):
    """Decode the guessed word out of one round block."""
    block = roundBlockSize(mode)
    chunk = row[roundIndex * block:(roundIndex + 1) * block]

    if mode == "compact":
        return "".join(ALPHABET[int(i)] for i in chunk[5:10])

    letters = []
    for position in range(5):
        start = 5 + position * len(ALPHABET)
        segment = chunk[start:start + len(ALPHABET)]
        letters.append(ALPHABET[segment.index(1)])
    return "".join(letters)


# ---------------------------------------------------------------------------
# Shape and self-description
# ---------------------------------------------------------------------------

def testReturnsADatasetWithTheRequestedCounts(dataset):
    assert isinstance(dataset, Dataset)
    assert len(dataset.xTrain) == N_TRAIN
    assert len(dataset.xTest) == N_TEST
    assert len(dataset.yTrain) == N_TRAIN
    assert len(dataset.yTest) == N_TEST


def testRecordsItsOwnWidthSoTheSolverCanMatchIt(dataset):
    assert dataset.maxRounds == MAX_ROUNDS
    assert dataset.letterMode == MODE
    assert dataset.encoder == "rounds"
    assert dataset.width == WIDTH


def testEveryRowHasTheSameWidthAtAnyRoundCount():
    for rounds in (1, 2, 3, 4, 5):
        ds = generateVariable(rounds, 40, 15, SEED, MAX_ROUNDS, MODE)
        assert {len(row) for row in ds.xTrain} == {WIDTH}
        assert {len(row) for row in ds.xTest} == {WIDTH}


def testLabelsAreValidVocabularyIndices(dataset):
    labels = dataset.yTrain + dataset.yTest
    assert min(labels) >= 0
    assert max(labels) < len(loadAnswers())


def testCarriesTheAnswersVocabulary(dataset):
    assert dataset.vocab is answersVocab()


# ---------------------------------------------------------------------------
# The critical guarantee: no leakage
# ---------------------------------------------------------------------------

def testNoSecretAppearsInBothTrainAndTest(dataset):
    trainWords = {dataset.vocab.wordAt(i) for i in dataset.yTrain}
    testWords = {dataset.vocab.wordAt(i) for i in dataset.yTest}
    assert trainWords & testWords == set(), "train/test secrets overlap -- leakage!"


def testSplitBySecretDoesNotMutateTheCachedAnswers(dataset):
    assert len(loadAnswers()) == 2315


# ---------------------------------------------------------------------------
# Variable length
# ---------------------------------------------------------------------------

def testMoreRoundsMeansLessPadding():
    one = generateVariable(1, 20, 8, SEED, MAX_ROUNDS, MODE)
    five = generateVariable(5, 20, 8, SEED, MAX_ROUNDS, MODE)

    def paddingCount(row):
        return sum(1 for value in row if value == PADDING)

    assert paddingCount(one.xTrain[0]) > paddingCount(five.xTrain[0])
    assert paddingCount(five.xTrain[0]) == 0


def testRoundsAreAPrefixOfLongerRuns():
    """Same seed + same secret => the n-round row starts with the n-round evidence.

    This is what makes the 1..5 round datasets describe the same games, cut off earlier.
    """
    one = generateVariable(1, 30, 10, SEED, MAX_ROUNDS, MODE)
    five = generateVariable(5, 30, 10, SEED, MAX_ROUNDS, MODE)

    fiveByLabel = {label: row for label, row in zip(five.yTrain, five.xTrain)}
    block = roundBlockSize(MODE)

    for label, row in zip(one.yTrain, one.xTrain):
        assert row[:block] == fiveByLabel[label][:block]


def testGuessesWithinAGameAreDistinct():
    ds = generateVariable(5, 30, 10, SEED, MAX_ROUNDS, MODE)
    for row in ds.xTrain:
        guesses = [guessLettersOf(row, r) for r in range(5)]
        assert len(set(guesses)) == 5


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def testSameSeedProducesIdenticalData():
    a = generateVariable(3, 30, 10, SEED, MAX_ROUNDS, MODE)
    b = generateVariable(3, 30, 10, SEED, MAX_ROUNDS, MODE)
    assert a.xTrain == b.xTrain and a.yTrain == b.yTrain
    assert a.xTest == b.xTest and a.yTest == b.yTest


def testDifferentSeedsProduceDifferentData():
    a = generateVariable(3, 30, 10, 1, MAX_ROUNDS, MODE)
    b = generateVariable(3, 30, 10, 2, MAX_ROUNDS, MODE)
    assert a.yTrain != b.yTrain


def testSecretRngIsReproducible():
    assert _secretRng(SEED, 7).random() == _secretRng(SEED, 7).random()
    assert _secretRng(SEED, 7).random() != _secretRng(SEED, 8).random()


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def testRejectsAnImpossibleRoundCount():
    with pytest.raises(ValueError):
        generateVariable(0, 30, 10, SEED, MAX_ROUNDS, MODE)
    with pytest.raises(ValueError):
        generateVariable(6, 30, 10, SEED, MAX_ROUNDS, MODE)


def testHonoursACustomGuessPicker():
    calls = []
    words = ["crane", "slate"]

    def picker(roundIndex, rng, evidence):
        calls.append((roundIndex, len(evidence)))
        return words[roundIndex]

    ds = generateVariable(2, 20, 8, SEED, MAX_ROUNDS, MODE, picker)
    # The picker is called once per round for every secret: 20 train + 8 test, 2 rounds each.
    assert len(calls) == 28 * 2
    assert calls[:2] == [(0, 0), (1, 1)]
    assert guessLettersOf(ds.xTrain[0], 0) == "crane"
    assert guessLettersOf(ds.xTrain[0], 1) == "slate"


def testCustomPickerSeesTheGrowingEvidence():
    """A picker can only choose consistently if it can see what was already played."""
    seen = []

    def picker(roundIndex, rng, evidence):
        seen.append(list(evidence))
        return ["crane", "slate"][roundIndex]

    generateVariable(2, 1, 1, SEED, MAX_ROUNDS, MODE, picker)

    # Two secrets, two rounds each: call 0 has no evidence, call 1 sees round 1.
    assert seen[0] == []
    assert seen[2] == []
    assert len(seen[1]) == 1 and seen[1][0][0] == "crane"
    assert len(seen[3]) == 1 and seen[3][0][0] == "crane"


def testAStuckPickerRaisesInsteadOfLoopingForever():
    """Regression: a picker that always returns one word used to hang generateVariable."""
    with pytest.raises(ValueError):
        generateVariable(2, 20, 8, SEED, MAX_ROUNDS, MODE,
                         lambda roundIndex, rng, evidence: "crane")


def testOnehotModeProducesTheWiderRows():
    ds = generateVariable(2, 20, 8, SEED, MAX_ROUNDS, "onehot")
    assert {len(row) for row in ds.xTrain} == {MAX_ROUNDS * roundBlockSize("onehot")}
    assert guessLettersOf(ds.xTrain[0], 0, "onehot") == guessLettersOf(
        generateVariable(2, 20, 8, SEED, MAX_ROUNDS, MODE).xTrain[0], 0
    )
