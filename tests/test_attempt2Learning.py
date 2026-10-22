"""Tests for the Attempt-2 learning algorithms.

These are not accuracy tests - learning is measured on the held-out secrets and reported
in `attempt2.md`. They are tests that the machinery is correct, including one that
regression-tests a bug that was actually hit: the policy loss was built from a detached
log-probability, so the policy head never received a gradient and only the value head
ever learned. The agent looked like it was training, and the curve never moved.
"""

import numpy as np
import pytest
import torch

from attempt2.dqn import DqnTrainer
from attempt2.policies import ACTION_COUNT, INPUT_WIDTH, stackObservations
from attempt2.ppo import PpoTrainer
from attempt2.reinforce import ReinforceTrainer
from attempt2.rollout import Choice, collectEpisode
from attempt2.wordleEnv import trainingEnv


def _episode(env, chooser, seed=1):
    return collectEpisode(env, chooser, seed=seed)


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------

def testThePolicyHeadReceivesAGradient():
    # The regression test for the detached-log-probability bug. If the policy loss is
    # built from a detached tensor, this gradient is zero and the policy never learns.
    trainer = ReinforceTrainer(seed=1)
    env = trainingEnv()

    episode = _episode(env, trainer.chooser())
    trainer.update(episode)

    gradient = trainer.net.policyHead.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def testTheValueHeadAlsoReceivesAGradient():
    trainer = ReinforceTrainer(seed=1)
    env = trainingEnv()

    episode = _episode(env, trainer.chooser())
    trainer.update(episode)

    gradient = trainer.net.valueHead.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def testReinforceProducesALearningCurve():
    trainer = ReinforceTrainer(seed=1)
    env = trainingEnv()
    curve = []

    trainer.train(5, env=env, curve=curve)

    assert len(curve) == 5
    for row in curve:
        assert 1 <= row["episode"] <= 5
        assert isinstance(row["reward"], float)
        assert row["won"] in (True, False)


def testTheEntropyScheduleMovesFromStartToEnd():
    trainer = ReinforceTrainer(entropyStart=0.2, entropyEnd=0.01, seed=1)

    assert trainer.entropyFor(0, 100) == pytest.approx(0.2)
    assert trainer.entropyFor(99, 100) == pytest.approx(0.01)
    assert trainer.entropyFor(500, 100) == pytest.approx(0.01)


def testTheChooserOnlyOffersLegalActions():
    trainer = ReinforceTrainer(seed=1)
    env = trainingEnv()

    episode = _episode(env, trainer.chooser())

    for transition in episode.transitions:
        assert transition.mask[transition.action]


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

def testTheBellmanTargetOnlyUsesTheLegalActionsOfTheNextState():
    trainer = DqnTrainer(seed=1)

    observations = np.zeros((1, INPUT_WIDTH), dtype=np.float32)
    masks = np.ones((1, ACTION_COUNT), dtype=bool)
    nextObservations = np.zeros((1, INPUT_WIDTH), dtype=np.float32)

    # Only two actions are legal in the next state, and one of them has the higher Q.
    nextMasks = np.zeros((1, ACTION_COUNT), dtype=bool)
    nextMasks[0, 3] = True
    nextMasks[0, 7] = True

    batch = {
        "observations": observations,
        "masks": masks,
        "actions": np.array([3]),
        "rewards": np.array([0.5], dtype=np.float32),
        "nextObservations": nextObservations,
        "nextMasks": nextMasks,
        "dones": np.array([0.0], dtype=np.float32),
    }

    with torch.no_grad():
        nextValues = trainer.targetNet(stackObservations(nextObservations))
    expectedBest = max(float(nextValues[0, 3]), float(nextValues[0, 7]))

    targets = trainer.bellmanTargets(batch)

    assert float(targets[0]) == pytest.approx(0.5 + trainer.discount * expectedBest)


def testATerminalTransitionHasNoBootstrappedValue():
    trainer = DqnTrainer(seed=1)

    batch = {
        "observations": np.zeros((1, INPUT_WIDTH), dtype=np.float32),
        "masks": np.ones((1, ACTION_COUNT), dtype=bool),
        "actions": np.array([0]),
        "rewards": np.array([1.0], dtype=np.float32),
        "nextObservations": np.zeros((1, INPUT_WIDTH), dtype=np.float32),
        "nextMasks": np.ones((1, ACTION_COUNT), dtype=bool),
        "dones": np.array([1.0], dtype=np.float32),
    }

    targets = trainer.bellmanTargets(batch)

    assert float(targets[0]) == pytest.approx(1.0)


def testDqnOnlyExploresLegalActions():
    trainer = DqnTrainer(seed=1)
    env = trainingEnv()

    trainer.epsilon = 1.0
    episode = _episode(env, trainer.chooser())

    for transition in episode.transitions:
        assert transition.mask[transition.action]


def testTheEpsilonScheduleAnneals():
    trainer = DqnTrainer(epsilonStart=1.0, epsilonEnd=0.05, seed=1)

    assert trainer.epsilonFor(0, 100) == pytest.approx(1.0)
    assert trainer.epsilonFor(99, 100) == pytest.approx(0.05)


def testDqnProducesALearningCurveAndFillsTheBuffer():
    trainer = DqnTrainer(seed=1)
    env = trainingEnv()
    curve = []

    trainer.train(5, env=env, curve=curve)

    assert len(curve) == 5
    assert len(trainer.buffer) > 0


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def testTheAdvantageOfTheLastStepOfAnEpisodeHasNoBootstrap():
    trainer = PpoTrainer(seed=1)

    advantages = trainer._advantages([1.0], [0.5])

    # delta = reward + discount * 0 (the game is over) - value
    assert advantages[0] == pytest.approx(1.0 - 0.5)


def testTheAdvantagesWalkBackwardsThroughTheEpisode():
    trainer = PpoTrainer(discount=0.9, gaeLambda=0.5, seed=1)

    advantages = trainer._advantages([0.0, 0.0], [0.0, 0.0])

    assert advantages[0] == pytest.approx(0.0)
    assert advantages[1] == pytest.approx(0.0)


def testPpoBuildsABatchWithAdvantagesAndReturns():
    trainer = PpoTrainer(seed=1)
    env = trainingEnv()

    episode = _episode(env, trainer.chooser())
    batch = trainer.buildBatch([episode])

    assert len(batch["transitions"]) == len(episode.transitions)
    assert len(batch["advantages"]) == len(episode.transitions)
    assert len(batch["returns"]) == len(episode.transitions)
    assert len(batch["logProbabilities"]) == len(episode.transitions)


def testPpoUpdatesThePolicyHead():
    trainer = PpoTrainer(seed=1)
    env = trainingEnv()

    episode = _episode(env, trainer.chooser())
    trainer.update(trainer.buildBatch([episode]))

    gradient = trainer.net.policyHead.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0


def testPpoProducesALearningCurve():
    trainer = PpoTrainer(seed=1, episodesPerUpdate=2)
    env = trainingEnv()
    curve = []

    trainer.train(4, env=env, curve=curve)

    assert len(curve) == 4


# ---------------------------------------------------------------------------
# The shared rollout loop
# ---------------------------------------------------------------------------

def testTheEpisodeLoopRecordsTheOutcome():
    env = trainingEnv()

    def chooser(state, _):
        return Choice(0)

    episode = _episode(env, chooser)

    assert episode.totalReward == pytest.approx(
        episode.guessesUsed * -0.05 + (1.0 if episode.won else 0.0)
    )
    assert 1 <= episode.guessesUsed <= 6
    assert len(episode.transitions) == episode.guessesUsed


def testTheEpisodeIsMarkedDoneAtTheEndWhicheverWayItEnded():
    env = trainingEnv()

    def chooser(state, _):
        return Choice(0)

    episode = _episode(env, chooser)

    assert all(transition.done for transition in episode.transitions[-1:])
    assert not any(transition.done for transition in episode.transitions[:-1])
