"""PPO with a masked action space.

WHY PPO IS THE ALGORITHM MOST LIKELY TO WORK HERE: the problem is noisy. The difference
between a good opening word and a bad one is a fraction of a guess, and the noise in a
single episode is about a whole guess. A method that learns from one episode at a time
(REINFORCE) sees mostly noise; PPO collects a whole batch of episodes, estimates the
advantage with a learned value function and generalised advantage estimation, and then
takes several careful steps on it. That is exactly the machinery this problem needs.

THE CLIPPED SURROGATE: the probability ratio between the new and the old policy is
clipped to a range around 1, so a single update cannot move the policy too far. It is the
safeguard that keeps a noisy advantage estimate from destroying a good policy.

THE MASK, ONCE MORE: the log-probabilities are always computed over the legal actions of
that state, both when the action was chosen and when the ratio is recomputed during the
update. The two have to use the same mask, or the ratio is meaningless.
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
from attempt2.rollout import Choice, episodeStream
from attempt2.wordleEnv import trainingEnv


class PpoTrainer:
    """Clipped-surrogate PPO with a value head, on the masked action space."""

    def __init__(self, learningRate=3e-4, discount=0.99, gaeLambda=0.95,
                 clipCoefficient=0.2, epochs=4, batches=4, entropyCoefficient=0.01,
                 episodesPerUpdate=32, valueCoefficient=0.5, hiddenSizes=(256, 256),
                 seed=config.SEED, device="cpu"):
        self.device = torch.device(device)
        self.learningRate = learningRate
        self.discount = discount
        self.gaeLambda = gaeLambda
        self.clipCoefficient = clipCoefficient
        self.epochs = epochs
        self.batches = batches
        self.entropyCoefficient = entropyCoefficient
        self.episodesPerUpdate = episodesPerUpdate
        self.valueCoefficient = valueCoefficient

        torch.manual_seed(seed)
        self.net = MaskedPolicyNet(INPUT_WIDTH, ACTION_COUNT, hiddenSizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learningRate)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        self.episodeCount = 0
        self.updateCount = 0

    # ------------------------------------------------------------------
    # Choosing
    # ------------------------------------------------------------------

    def chooser(self):
        """The callback the episode loop calls."""
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

    def update(self, batch):
        """One PPO update over a batch of episodes."""
        observations = stackObservations(
            [transition.observation for transition in batch["transitions"]]
        ).to(self.device)
        masks = stackMasks(
            [transition.mask for transition in batch["transitions"]]
        ).to(self.device)
        actions = torch.as_tensor(
            [transition.action for transition in batch["transitions"]],
            dtype=torch.long, device=self.device,
        )
        oldLogProbabilities = torch.as_tensor(
            batch["logProbabilities"], dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            batch["advantages"], dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            batch["returns"], dtype=torch.float32, device=self.device
        )

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        count = len(actions)
        batchSize = max(1, count // self.batches)

        for _ in range(self.epochs):
            order = torch.randperm(count, device=self.device)

            for start in range(0, count, batchSize):
                index = order[start:start + batchSize]
                if index.numel() == 0:
                    continue

                logits, values = self.net(observations[index])
                logProbabilities = maskedLogProbs(logits, masks[index])
                chosen = logProbabilities.gather(1, actions[index].unsqueeze(-1)).squeeze(-1)

                ratio = torch.exp(chosen - oldLogProbabilities[index])

                unclipped = ratio * advantages[index]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clipCoefficient, 1.0 + self.clipCoefficient
                ) * advantages[index]
                policyLoss = -torch.min(unclipped, clipped).mean()

                valueLoss = ((values - returns[index]) ** 2).mean()
                entropy = maskedEntropy(logits, masks[index]).mean()

                loss = (policyLoss
                        + self.valueCoefficient * valueLoss
                        - self.entropyCoefficient * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()

        self.updateCount += 1

        return {"loss": float(loss.item()), "policyLoss": float(policyLoss.item())}

    def scores(self, state):
        """The policy logit of every answer, for evaluation."""
        observation = stackObservations([state.observation]).to(self.device)

        with torch.no_grad():
            logits, _ = self.net(observation)

        return logits[0]

    def buildBatch(self, episodes):
        """Turn a batch of episodes into the arrays the update needs.

        WHY THE ADVANTAGE IS COMPUTED PER EPISODE: generalised advantage estimation
        walks backwards through an episode, and the walk must not cross the end of one -
        the value of the state after the last guess of a finished game is zero, because
        the game is over.
        """
        transitions = []
        logProbabilities = []
        advantages = []
        returns = []

        for episode in episodes:
            rewards = [transition.reward for transition in episode.transitions]
            values = [transition.extras["value"] for transition in episode.transitions]

            episodeAdvantages = self._advantages(rewards, values)
            episodeReturns = [
                advantage + value for advantage, value in zip(episodeAdvantages, values)
            ]

            for transition, advantage, episodeReturn in zip(
                episode.transitions, episodeAdvantages, episodeReturns
            ):
                transitions.append(transition)
                logProbabilities.append(transition.extras["logProbability"])
                advantages.append(advantage)
                returns.append(episodeReturn)

        return {
            "transitions": transitions,
            "logProbabilities": logProbabilities,
            "advantages": advantages,
            "returns": returns,
        }

    def _advantages(self, rewards, values):
        """Generalised advantage estimation over one episode."""
        advantages = [0.0] * len(rewards)
        running = 0.0

        for index in range(len(rewards) - 1, -1, -1):
            nextValue = values[index + 1] if index + 1 < len(values) else 0.0
            delta = rewards[index] + self.discount * nextValue - values[index]
            running = delta + self.discount * self.gaeLambda * running
            advantages[index] = running

        return advantages

    def train(self, episodes, env=None, seedBase=config.SEED, curve=None,
              totalEpisodes=None, evaluation=None, evaluationEvery=None):
        """Run `episodes` episodes of training.

        WHY THE EPISODES ARE COLLECTED IN A BATCH BEFORE ANY UPDATE: PPO's advantage
        estimate is only stable when it is averaged over many episodes, and the clipped
        objective assumes the policy has not moved much since the data was collected.
        """
        env = env if env is not None else trainingEnv()
        chooser = self.chooser()

        collected = []

        for index, episode in episodeStream(env, chooser, episodes, seedBase):
            self.episodeCount += 1
            collected.append(episode)

            if curve is not None:
                curve.append({
                    "episode": self.episodeCount,
                    "reward": episode.totalReward,
                    "won": episode.won,
                    "guessesUsed": episode.guessesUsed,
                    "firstGuess": episode.firstGuess,
                    "entropy": 0.0,
                    "entropyCoefficient": self.entropyCoefficient,
                    "loss": 0.0,
                })

            if len(collected) >= self.episodesPerUpdate:
                self.update(self.buildBatch(collected))
                collected = []

        if collected:
            self.update(self.buildBatch(collected))

        if evaluation is not None and evaluationEvery:
            if self.episodeCount % evaluationEvery == 0:
                evaluation(self, self.episodeCount)

        return curve if curve is not None else []
