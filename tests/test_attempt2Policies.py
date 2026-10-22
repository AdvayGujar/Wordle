"""Tests for the Attempt-2 networks and masked action selection.

The mask is the load-bearing idea in this attempt, so these tests are mostly about one
thing: an illegal action must never be chosen, and must never carry any probability.
"""

import numpy as np
import pytest
import torch

from attempt2.policies import (
    ACTION_COUNT,
    INPUT_WIDTH,
    MaskedPolicyNet,
    QNetwork,
    greedyPolicyFromScorer,
    maskedEntropy,
    maskedLogProbs,
    maskedProbabilities,
    normalizeObservation,
    selectAction,
    stackMasks,
    stackObservations,
)
from attempt2.eval import PolicyState


def _scores():
    """Bounded scores, so a softmax over them cannot overflow."""
    return torch.linspace(0.0, 1.0, ACTION_COUNT)


def _orderedScores():
    """Scores that rise with the index, for the greedy-choice tests."""
    return torch.arange(ACTION_COUNT, dtype=torch.float32)


def _mask(legal):
    mask = torch.zeros(ACTION_COUNT, dtype=torch.bool)
    mask[legal] = True
    return mask


def testAnIllegalActionHasNoProbability():
    mask = _mask([3, 17, 900])
    probabilities = maskedProbabilities(_scores(), mask)

    assert probabilities[3] > 0
    assert probabilities[17] > 0
    assert probabilities[900] > 0
    assert probabilities[0] == 0.0
    assert probabilities[1] == 0.0
    assert float(probabilities.sum()) == pytest.approx(1.0)


def testTheIllegalActionsGetNoGradient():
    # The masked logits are a constant, so no gradient can flow to them - which is what
    # stops the policy from ever learning to prefer an illegal action. Two legal actions
    # are needed: with only one, the probability is 1 and the log-probability is 0, so
    # its gradient is zero for a completely different reason.
    logits = _scores().clone().requires_grad_(True)
    mask = _mask([5, 6])

    maskedLogProbs(logits, mask)[5].backward()

    assert logits.grad[5].item() != 0.0
    assert float(logits.grad[0].abs().sum()) == 0.0


def testSamplingNeverPicksAnIllegalAction():
    generator = torch.Generator()
    generator.manual_seed(0)
    mask = _mask([7, 8, 9])

    for _ in range(200):
        action = selectAction(_scores(), mask, generator=generator)
        assert action in (7, 8, 9)


def testGreedyTakesTheBestLegalAction():
    mask = _mask([2, 100, 1000])

    assert selectAction(_orderedScores(), mask, greedy=True) == 1000


def testGreedyIgnoresAnIllegalActionWithAHigherScore():
    # The scores rise with the index, so the highest-scoring action is illegal here.
    mask = _mask([1, 2])

    assert selectAction(_orderedScores(), mask, greedy=True) == 2


def testTheEntropyOfAOneHotDistributionIsZero():
    mask = _mask([4])
    logits = torch.full((ACTION_COUNT,), -50.0)
    logits[4] = 50.0

    assert float(maskedEntropy(logits, mask)) == pytest.approx(0.0, abs=1e-4)


def testTheEntropyOfASpreadDistributionIsLarge():
    mask = _mask(list(range(ACTION_COUNT)))

    entropy = float(maskedEntropy(torch.zeros(ACTION_COUNT), mask))

    assert entropy == pytest.approx(np.log(ACTION_COUNT), rel=1e-3)


def testTheNormalisationMapsTheObservationIntoZeroAndOne():
    vector = np.linspace(-1, 25, INPUT_WIDTH).astype(np.float32)

    normalised = normalizeObservation(vector)

    assert float(normalised.min()) == pytest.approx(0.0)
    assert float(normalised.max()) == pytest.approx(1.0)


def testThePaddingMarkerStaysDistinctFromARealColour():
    # The observation mixes colour codes (0..2), letter indices (0..25) and a -1 padding
    # marker. After normalising, "no evidence here" must not look like "grey".
    padded = np.full(INPUT_WIDTH, -1.0, dtype=np.float32)
    grey = np.full(INPUT_WIDTH, 0.0, dtype=np.float32)

    assert normalizeObservation(padded)[0] < normalizeObservation(grey)[0]


def testThePolicyNetProducesOneLogitPerAnswerAndOneValue():
    net = MaskedPolicyNet()
    observation = stackObservations([np.zeros(INPUT_WIDTH, dtype=np.float32)])

    logits, value = net(observation)

    assert logits.shape == (1, ACTION_COUNT)
    assert value.shape == (1,)


def testTheQNetworkProducesOneValuePerAnswer():
    net = QNetwork()
    observation = stackObservations([np.zeros(INPUT_WIDTH, dtype=np.float32)])

    assert net(observation).shape == (1, ACTION_COUNT)


def testTheScorerPolicyTakesTheBestLegalAction():
    def scorer(state):
        return torch.arange(ACTION_COUNT, dtype=torch.float32)

    mask = np.zeros(ACTION_COUNT, dtype=bool)
    mask[42] = True
    state = PolicyState(np.zeros(INPUT_WIDTH, dtype=np.float32), mask, ())

    assert greedyPolicyFromScorer(scorer)(state) == 42


def testStackingProducesABatch():
    observations = stackObservations([np.zeros(INPUT_WIDTH, dtype=np.float32)] * 4)
    masks = stackMasks([np.ones(ACTION_COUNT, dtype=bool)] * 4)

    assert observations.shape == (4, INPUT_WIDTH)
    assert masks.shape == (4, ACTION_COUNT)
    assert masks.dtype == torch.bool
