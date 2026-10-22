"""Tests for the replay buffer.

The buffer is where the off-by-one errors live: a next state that is really the current
state, a mask that does not match the observation, a terminal flag that means "ran out of
time" instead of "the game is over". Each of those would make DQN learn something subtly
wrong, so each is pinned here.
"""

import numpy as np
import pytest

from attempt2.replayBuffer import ReplayBuffer
from wordle.vocabulary import loadAnswers

ACTION_COUNT = len(loadAnswers())
OBSERVATION_WIDTH = 60


def _transition(action=0, reward=1.0, done=False, marker=0.0):
    observation = np.full(OBSERVATION_WIDTH, marker, dtype=np.float32)
    nextObservation = np.full(OBSERVATION_WIDTH, marker + 1.0, dtype=np.float32)
    mask = np.ones(ACTION_COUNT, dtype=bool)
    nextMask = np.zeros(ACTION_COUNT, dtype=bool)
    nextMask[0] = True

    return observation, mask, action, reward, nextObservation, nextMask, done


def testSamplingReturnsTheStoredTransition():
    buffer = ReplayBuffer(capacity=10, seed=1)
    observation, mask, action, reward, nextObservation, nextMask, done = _transition(
        action=5, reward=0.5, marker=3.0
    )
    buffer.add(observation, mask, action, reward, nextObservation, nextMask, done)

    batch = buffer.sample(1)

    assert batch["actions"][0] == 5
    assert batch["rewards"][0] == pytest.approx(0.5)
    assert batch["observations"][0][0] == pytest.approx(3.0)
    assert batch["nextObservations"][0][0] == pytest.approx(4.0)
    assert batch["nextMasks"][0].sum() == 1
    assert batch["dones"][0] == 0.0


def testSamplingMoreThanTheBufferHoldsIsAnError():
    buffer = ReplayBuffer(capacity=10, seed=1)
    buffer.add(*_transition())

    with pytest.raises(ValueError, match="cannot sample"):
        buffer.sample(5)


def testTheBufferDropsTheOldestTransitionsWhenItIsFull():
    buffer = ReplayBuffer(capacity=3, seed=1)

    for action in range(5):
        buffer.add(*_transition(action=action))

    assert len(buffer) == 3

    batch = buffer.sample(3)
    assert set(batch["actions"].tolist()) == {2, 3, 4}


def testTheSampledArraysAllHaveTheSameLength():
    buffer = ReplayBuffer(capacity=20, seed=1)

    for action in range(10):
        buffer.add(*_transition(action=action))

    batch = buffer.sample(4)

    assert batch["observations"].shape == (4, OBSERVATION_WIDTH)
    assert batch["masks"].shape == (4, ACTION_COUNT)
    assert batch["actions"].shape == (4,)
    assert batch["rewards"].shape == (4,)
    assert batch["nextObservations"].shape == (4, OBSERVATION_WIDTH)
    assert batch["nextMasks"].shape == (4, ACTION_COUNT)
    assert batch["dones"].shape == (4,)


def testTheTerminalFlagIsStoredAsAFloat():
    buffer = ReplayBuffer(capacity=10, seed=1)
    buffer.add(*_transition(done=True))

    batch = buffer.sample(1)

    assert batch["dones"][0] == 1.0
