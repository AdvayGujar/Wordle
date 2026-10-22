"""Tests for the Attempt-2 train / held-out split.

The property that matters: the environment may only ever play a secret that is not in the
held-out half. That is asserted here rather than left to a comment.
"""

import pytest

from attempt2 import config
from attempt2.splits import assertNoLeakage, heldOutSecrets, trainSecrets
from wordle.vocabulary import loadAnswers


def testSplitSizesMatchTheConfiguredProtocol():
    assert len(trainSecrets()) == config.N_TRAIN_SECRETS
    assert len(heldOutSecrets()) == config.N_TEST_SECRETS


def testTheTwoHalvesAreDisjoint():
    assert set(trainSecrets()).isdisjoint(heldOutSecrets())


def testBothHalvesComeFromTheAnswerList():
    answers = set(loadAnswers())
    assert set(trainSecrets()) <= answers
    assert set(heldOutSecrets()) <= answers


def testTheSplitIsDeterministic():
    # Same seed, same split - otherwise "rerun the experiment" would not rerun it.
    assert trainSecrets() == trainSecrets()
    assert heldOutSecrets() == heldOutSecrets()


def testTheSplitIsTheCoreSplit():
    # Attempt 2 must not invent a second split protocol (plan.md §8.6).
    from wordle.dataset import splitSecrets

    expectedTrain, expectedTest = splitSecrets(config.N_TRAIN_SECRETS, config.SEED)
    assert trainSecrets() == list(expectedTrain)
    assert heldOutSecrets() == list(expectedTest)


def testAssertNoLeakagePassesOnTheRealSplit():
    assert assertNoLeakage() is True


def testAssertNoLeakageCatchesAnOverlap():
    train = trainSecrets()
    test = list(heldOutSecrets())
    test[0] = train[0]

    with pytest.raises(ValueError, match="leak"):
        assertNoLeakage(train, test)


def testAssertNoLeakageCatchesAWrongSizedHalf():
    with pytest.raises(ValueError, match="training secrets"):
        assertNoLeakage(trainSecrets()[:-1], heldOutSecrets())

    with pytest.raises(ValueError, match="held-out secrets"):
        assertNoLeakage(trainSecrets(), heldOutSecrets()[:-1])
