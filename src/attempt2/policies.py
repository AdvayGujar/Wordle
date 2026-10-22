"""The networks, and the masked action selection every algorithm shares.

WHY THE MASK LIVES HERE AND NOWHERE ELSE: an illegal action is not "unlikely", it is
impossible. The mask is applied to the logits *before* the softmax, so an illegal action has
probability exactly zero and the gradient never pushes towards it. Putting that in one
place means all three algorithms get it, and there is no way for one of them to forget.

WHY THE NETWORKS ARE SO SMALL: the input is 60 numbers and the output is 2,315 logits. A
two-layer MLP is already ~675,000 parameters, and the task is not one that needs more - the
difficulty is the exploration, not the capacity.

WHY THE INPUT IS NORMALISED: the raw observation mixes colour codes (0..2), letter indices
(0..25) and a -1 padding marker. Dividing by the range of the mode maps them all into
[0, 1] and - the padding marker lands on 0 while a real colour lands slightly above it, so
"no evidence here" stays distinguishable from "grey". The mapping is fixed and derived from
`observationBounds`, so there are no running statistics to drift and a run is reproducible.
"""

import numpy as np
import torch
from torch import nn

from attempt2 import config
from attempt2.observation import observationBounds, observationWidth
from wordle.vocabulary import loadAnswers

INPUT_WIDTH = observationWidth(config.OBSERVATION_MODE)
ACTION_COUNT = len(loadAnswers())
HIDDEN_SIZES = (256, 256)

# A number large enough to lose to any real logit, small enough not to produce inf when
# it is subtracted from a large negative one.
NEGATIVE_INFINITY = -1e9


def normalizeObservation(observation, mode=config.OBSERVATION_MODE):
    """Map an observation into [0, 1] using the mode's own bounds.

    WHY THE BOUNDS AND NOT A RUNNING MEAN: a running normaliser would make the input
    depend on the order the data arrived, and it would make "the same observation" a
    different vector at different points in training. A fixed map keeps the observation a
    pure function of the history.
    """
    low, high = observationBounds(mode)
    return (observation - low) / (high - low)


class MaskedPolicyNet(nn.Module):
    """Observation -> (policy logits over the answers, state value).

    WHY A SHARED TRUNK: the policy and the value describe the same state, so the features
    that make a state worth guessing in are the features that make it valuable. One trunk
    with two heads is also cheaper than two networks.
    """

    def __init__(self, inputWidth=INPUT_WIDTH, actionCount=ACTION_COUNT,
                 hiddenSizes=HIDDEN_SIZES):
        super().__init__()

        self.inputWidth = inputWidth
        self.actionCount = actionCount

        layers = []
        width = inputWidth
        for size in hiddenSizes:
            layers.append(nn.Linear(width, size))
            layers.append(nn.ReLU())
            width = size
        self.trunk = nn.Sequential(*layers)

        self.policyHead = nn.Linear(width, actionCount)
        self.valueHead = nn.Linear(width, 1)

    def forward(self, observation):
        features = self.trunk(observation)
        return self.policyHead(features), self.valueHead(features).squeeze(-1)


class QNetwork(nn.Module):
    """Observation -> a Q value for every answer. Used by DQN."""

    def __init__(self, inputWidth=INPUT_WIDTH, actionCount=ACTION_COUNT,
                 hiddenSizes=HIDDEN_SIZES):
        super().__init__()

        self.inputWidth = inputWidth
        self.actionCount = actionCount

        layers = []
        width = inputWidth
        for size in hiddenSizes:
            layers.append(nn.Linear(width, size))
            layers.append(nn.ReLU())
            width = size
        layers.append(nn.Linear(width, actionCount))
        self.layers = nn.Sequential(*layers)

    def forward(self, observation):
        return self.layers(observation)


def maskedLogits(logits, mask):
    """Push the logits of every illegal action far below any legal one.

    WHY A LARGE NEGATIVE NUMBER AND NOT `-inf`: `-inf` propagates NaNs through a softmax
    when a whole row is masked, and `log_softmax` of `-inf` is `-inf` in a way that
    poisons the gradient. A finite floor keeps the arithmetic clean while still making
    the illegal actions' probability indistinguishable from zero.
    """
    return logits.masked_fill(~mask, NEGATIVE_INFINITY)


def maskedLogProbs(logits, mask):
    """Log-probabilities over the legal actions only."""
    return torch.log_softmax(maskedLogits(logits, mask), dim=-1)


def maskedProbabilities(logits, mask):
    """Probabilities over the legal actions only."""
    return torch.softmax(maskedLogits(logits, mask), dim=-1)


def selectAction(logits, mask, generator=None, greedy=False):
    """Pick one legal action from a masked distribution.

    WHY THE SAMPLING HAPPENS HERE: every algorithm has to choose an action the same way,
    and the choice has to respect the mask. A single implementation means the exploration
    of one algorithm cannot accidentally differ from another's.

    `generator` makes the choice reproducible; `greedy` takes the best legal action, which
    is what evaluation and DQN's exploitation step both want.
    """
    if greedy:
        masked = maskedLogits(logits, mask)
        legal = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if legal.numel() == 0:
            raise ValueError("no legal actions; the mask cannot be empty")
        values = masked[legal]
        return int(legal[int(values.argmax())].item())

    probabilities = maskedProbabilities(logits, mask)
    action = torch.multinomial(probabilities, 1, generator=generator)
    return int(action.item())


def maskedEntropy(logits, mask):
    """The entropy of the masked action distribution.

    WHY IT IS NEEDED: a policy that puts all its mass on one word stops exploring, and on
    this task the first guess decides most of the game. The entropy bonus is what keeps
    the agent sampling openers while it is still learning which one is good.
    """
    logProbabilities = maskedLogProbs(logits, mask)
    probabilities = torch.exp(logProbabilities)
    return -(probabilities * logProbabilities).sum(dim=-1)


def observationTensor(observation):
    """A normalised observation as a float32 tensor."""
    return torch.as_tensor(
        normalizeObservation(np.asarray(observation, dtype=np.float32)), dtype=torch.float32
    )


def maskTensor(mask):
    """A mask as a bool tensor."""
    return torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool)


def stackObservations(observations):
    """A batch of observations as a normalised float32 tensor."""
    batch = np.asarray(observations, dtype=np.float32)
    return torch.as_tensor(normalizeObservation(batch), dtype=torch.float32)


def stackMasks(masks):
    """A batch of masks as a bool tensor."""
    return torch.as_tensor(np.asarray(masks, dtype=bool), dtype=torch.bool)


def greedyPolicyFromScorer(scorer):
    """A policy that takes the best legal action according to a scoring function.

    WHY A SCORER AND NOT A NETWORK: the actor-critic algorithms score an action by its
    policy logit and DQN scores it by its Q value, but the *evaluation* of both is the
    same question - which legal action has the highest score. One wrapper means the
    evaluation code cannot accidentally treat a Q value as a logit.
    """
    def policy(state):
        with torch.no_grad():
            values = scorer(state)

        scores = np.where(np.asarray(state.mask, dtype=bool),
                          values.detach().cpu().numpy(), -np.inf)
        return int(np.argmax(scores))

    return policy


def samplingPolicyFromScorer(scorer, generator=None):
    """A policy that samples from the masked softmax of the scores.

    WHY A SCORER: the actor-critic algorithms sample from the softmax of their policy
    logits; DQN samples epsilon-greedily from its Q values, which is a different
    distribution. Both can be expressed over the same scorer interface.
    """
    def policy(state):
        with torch.no_grad():
            values = scorer(state)

        logits = torch.as_tensor(values.detach().cpu().numpy(), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(state.mask, dtype=bool), dtype=torch.bool)

        return selectAction(logits, mask, generator=generator)

    return policy


def greedyPolicy(net):
    """A network as an evaluation policy: always take the best legal action.

    WHY GREEDY FOR EVALUATION: the evaluation has to measure the policy the agent has
    learned, not the luck of what it sampled. The stochastic policy is what explores;
    the greedy one is what is judged.
    """
    def policy(state):
        observation = stackObservations([state.observation])
        mask = stackMasks([state.mask])

        with torch.no_grad():
            logits, _ = net(observation)

        return selectAction(logits[0], mask[0], greedy=True)

    return policy


def samplingPolicy(net, generator=None):
    """A network as a policy that samples from the masked distribution."""
    def policy(state):
        observation = stackObservations([state.observation])
        mask = stackMasks([state.mask])

        with torch.no_grad():
            logits, _ = net(observation)

        return selectAction(logits[0], mask[0], generator=generator)

    return policy
