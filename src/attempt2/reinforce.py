"""REINFORCE with a learned value baseline (the A2C-style policy gradient).

WHY THIS ALGORITHM GOES FIRST: it is the simplest policy gradient there is - no replay
buffer, no target network, no importance ratios. If the learning curve is flat for this
algorithm, the problem is the setup (the reward, the observation, the masking), not the
optimiser, and adding DQN and PPO on top would only hide the bug. So this one has to work
before the others are attempted (plan.md, Gate C).

THE UPDATE: for each episode, compute the discounted return of every step, subtract the
value estimate to get an advantage, and push the policy towards the actions that were
taken in proportion to how good they turned out to be:

```
loss = -mean(logProbability(action) * advantage) - entropyCoefficient * entropy
```

WHY THE VALUE BASELINE: the raw return is a bad learning signal on its own, because most
episodes end in a loss and the returns are all similar and negative. Subtracting the
predicted value centres the signal on "was this turn better than I expected?", which is
what the policy should actually learn from.

WHY THE ENTROPY BONUS: the first guess decides most of the game, so a policy that puts
all its mass on one word stops exploring and cannot recover if that word was wrong. The
entropy bonus keeps the agent sampling openers while it is still learning which one is
good. It is switched off for the final evaluation.
"""

import torch
from torch import nn

from attempt2 import config
from attempt2.policies import (
    ACTION_COUNT,
    INPUT_WIDTH,
    MaskedPolicyNet,
    maskedEntropy,
    maskedLogProbs,
    selectAction,
    stackMasks,
    stackObservations,
)
from attempt2.rollout import Choice, discountedReturns, episodeStream
from attempt2.wordleEnv import trainingEnv


class ReinforceTrainer:
    """REINFORCE with a value baseline, on the masked action space."""

    def __init__(self, learningRate=1e-3, discount=0.99, entropyStart=0.01,
                 entropyEnd=0.01, hiddenSizes=(256, 256), seed=config.SEED, device="cpu"):
        self.device = torch.device(device)
        self.learningRate = learningRate
        self.discount = discount

        # WHY AN ENTROPY SCHEDULE RATHER THAN A FIXED VALUE: the opener has to be
        # discovered by sampling, and the sampling noise is much larger than the
        # difference between openers (measured: 0.46 guesses across a sample of openers,
        # against ~1.0 guesses of noise in the number of guesses). Ranking them needs
        # roughly a hundred samples each, so the policy has to keep exploring for a long
        # time before it is allowed to commit. The schedule starts wide and tightens.
        self.entropyStart = entropyStart
        self.entropyEnd = entropyEnd
        self.entropyCoefficient = entropyStart

        torch.manual_seed(seed)
        self.net = MaskedPolicyNet(INPUT_WIDTH, ACTION_COUNT, hiddenSizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learningRate)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.episodeCount = 0

    def entropyFor(self, episodeIndex, totalEpisodes):
        """The entropy coefficient at a point in training.

        WHY IT IS A SCHEDULE AND NOT A CONSTANT: a constant strong bonus keeps the policy
        spread over all 2,315 answers forever and it never commits; a constant zero (or
        weak) bonus lets it collapse onto whichever answer looked lucky in the first few
        hundred episodes, before any of them had been sampled often enough to be ranked.
        """
        if totalEpisodes <= 1:
            return self.entropyEnd

        fraction = min(1.0, max(0.0, episodeIndex / float(totalEpisodes - 1)))
        return self.entropyStart + (self.entropyEnd - self.entropyStart) * fraction

    # ------------------------------------------------------------------
    # Choosing
    # ------------------------------------------------------------------

    def chooser(self):
        """The callback the episode loop calls. Returns the action and the numbers the
        update needs."""
        def choose(state, _):
            observation = stackObservations(state.observation).unsqueeze(0).to(self.device)
            mask = stackMasks(state.mask).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits, value = self.net(observation)
                logProbabilities = maskedLogProbs(logits, mask)
                action = selectAction(logits[0], mask[0], generator=self.generator)

            return Choice(action, {
                "logProbability": float(logProbabilities[0, action].item()),
                "value": float(value.item()),
            })

        return choose

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, episode):
        """One policy-gradient update from a finished episode."""
        if not episode.transitions:
            return {}

        observations = stackObservations(
            [transition.observation for transition in episode.transitions]
        ).to(self.device)
        masks = stackMasks(
            [transition.mask for transition in episode.transitions]
        ).to(self.device)
        actions = torch.as_tensor(
            [transition.action for transition in episode.transitions],
            dtype=torch.long, device=self.device,
        )
        returns = discountedReturns(
            [transition.reward for transition in episode.transitions], self.discount
        )
        returnsTensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        logits, values = self.net(observations)
        logProbabilities = maskedLogProbs(logits, masks)

        # The advantage: how much better this turn turned out than the agent expected.
        # It is DETACHED on purpose - the advantage is the weight of the update, not
        # something the policy may change by changing the value function.
        advantages = returnsTensor - values.detach()

        # WHY NORMALISE THE ADVANTAGE: the scale of the advantage changes over training
        # (early on the returns are noisy and small, later they are consistent). Scaling
        # them to unit variance keeps the step size meaningful throughout.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # WHY THE FRESH LOG-PROBABILITIES AND NOT THE ONES FROM THE CHOOSER: the score
        # function estimator is `grad log pi(a|s) * advantage`, so the log-probability in
        # the loss has to be the one the CURRENT network produces - otherwise the
        # gradient never reaches the policy head and only the value head ever learns.
        # The action itself still comes from the chooser.
        chosenLogProbabilities = logProbabilities.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policyLoss = -(chosenLogProbabilities * advantages).mean()
        valueLoss = ((values - returnsTensor) ** 2).mean()
        entropy = maskedEntropy(logits, masks).mean()

        loss = policyLoss + valueLoss - self.entropyCoefficient * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policyLoss": float(policyLoss.item()),
            "valueLoss": float(valueLoss.item()),
            "entropy": float(entropy.item()),
        }

    def scores(self, state):
        """The policy logit of every answer, for evaluation."""
        observation = stackObservations([state.observation]).to(self.device)

        with torch.no_grad():
            logits, _ = self.net(observation)

        return logits[0]

    def train(self, episodes, env=None, seedBase=config.SEED, curve=None,
              totalEpisodes=None, evaluation=None, evaluationEvery=None):
        """Run `episodes` episodes of training.

        `curve` is a list the per-episode numbers are appended to; it is the learning
        curve that the report plots. `evaluation` is an optional callable that scores the
        current policy - the validation slice, never the held-out set.
        """
        env = env if env is not None else trainingEnv()
        chooser = self.chooser()
        totalEpisodes = totalEpisodes if totalEpisodes is not None else episodes

        for index, episode in episodeStream(env, chooser, episodes, seedBase):
            self.entropyCoefficient = self.entropyFor(self.episodeCount, totalEpisodes)
            metrics = self.update(episode)
            self.episodeCount += 1

            if curve is not None:
                curve.append({
                    "episode": self.episodeCount,
                    "reward": episode.totalReward,
                    "won": episode.won,
                    "guessesUsed": episode.guessesUsed,
                    "firstGuess": episode.firstGuess,
                    "entropy": metrics.get("entropy", 0.0),
                    "entropyCoefficient": self.entropyCoefficient,
                    "loss": metrics.get("loss", 0.0),
                })

            if evaluation is not None and evaluationEvery:
                if self.episodeCount % evaluationEvery == 0:
                    evaluation(self, self.episodeCount)

        return curve if curve is not None else []
