"""DQN with a masked action space.

WHY DQN IS THE NATURAL PAIRING WITH THIS ENVIRONMENT: the action space is discrete and
small enough to score every answer, and the mask turns "which answer?" into a question
with a definite right answer - the highest Q value among the *legal* ones. A value-based
method can use that directly.

THE TWO MASKS THAT MATTER:

1. **Exploration** picks a random LEGAL action. Picking a random answer and hoping it is
   legal would waste most of the steps - and the illegal guess would still be played,
   because the environment plays what it is told.
2. **The Bellman target** maximises over the legal actions of the NEXT state, not over
   all 2,315 answers. A target that included the eliminated answers would be asking the
   network to predict the value of guesses the agent is not allowed to make.

WHY A TARGET NETWORK: the target moves slowly, so the value the network is chasing does
not move with every step. Without it, the target chases the network and the training
diverges.
"""

import numpy as np
import torch
from torch import nn

from attempt2 import config
from attempt2.policies import (
    ACTION_COUNT,
    INPUT_WIDTH,
    QNetwork,
    maskedLogits,
    stackMasks,
    stackObservations,
)
from attempt2.replayBuffer import ReplayBuffer
from attempt2.rollout import Choice, episodeStream
from attempt2.wordleEnv import trainingEnv


class DqnTrainer:
    """Masked DQN with a replay buffer and a target network."""

    def __init__(self, learningRate=1e-3, discount=0.99, bufferSize=100000, batchSize=64,
                 epsilonStart=1.0, epsilonEnd=0.05, targetUpdateEvery=500,
                 trainEvery=4, hiddenSizes=(256, 256), seed=config.SEED, device="cpu"):
        self.device = torch.device(device)
        self.learningRate = learningRate
        self.discount = discount
        self.batchSize = batchSize
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.targetUpdateEvery = targetUpdateEvery
        self.trainEvery = trainEvery

        torch.manual_seed(seed)
        self.net = QNetwork(INPUT_WIDTH, ACTION_COUNT, hiddenSizes).to(self.device)
        self.targetNet = QNetwork(INPUT_WIDTH, ACTION_COUNT, hiddenSizes).to(self.device)
        self.targetNet.load_state_dict(self.net.state_dict())
        self.targetNet.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learningRate)
        self.buffer = ReplayBuffer(bufferSize, seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        # WHY A DEDICATED NUMPY GENERATOR: the exploration draws have to come from one
        # seeded stream, so a run is reproducible and the draws do not depend on how many
        # times torch happened to call its own generator.
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilonStart

        self.stepCount = 0
        self.episodeCount = 0
        self.updateCount = 0

    # ------------------------------------------------------------------
    # Choosing
    # ------------------------------------------------------------------

    def epsilonFor(self, step, totalSteps):
        """The exploration rate at a point in training.

        WHY IT ANNEALS FROM 1.0: the agent has to try many answers before any of them has
        been sampled often enough to be ranked (measured: openers differ by ~0.46 guesses,
        against ~1.0 guesses of noise in the number of guesses). Exploration that stops
        early locks the agent onto whichever answer looked lucky first.
        """
        if totalSteps <= 1:
            return self.epsilonEnd

        fraction = min(1.0, step / float(totalSteps - 1))
        return self.epsilonStart + (self.epsilonEnd - self.epsilonStart) * fraction

    def chooser(self):
        """The callback the episode loop calls.

        WHY IT READS `self.epsilon` AND NOT A PARAMETER: the exploration rate is annealed
        over training, and the chooser is created once. Reading the current value keeps
        the schedule and the behaviour in step.
        """
        def choose(state, _):
            observation = stackObservations([state.observation]).to(self.device)

            with torch.no_grad():
                values = self.net(observation)

            legal = np.flatnonzero(np.asarray(state.mask, dtype=bool))
            if len(legal) == 0:
                raise ValueError("no legal actions; the mask cannot be empty")

            if self.rng.random() < self.epsilon:
                action = int(legal[int(self.rng.integers(len(legal)))])
            else:
                scores = values[0].cpu().numpy()
                action = int(legal[int(np.argmax(scores[legal]))])

            return Choice(action)

        return choose

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def bellmanTargets(self, batch):
        """The regression target for a batch of transitions.

        WHY THE MAXIMUM IS OVER THE LEGAL ACTIONS ONLY: the next state's mask says which
        answers the agent is allowed to guess there. Taking the maximum over all 2,315
        answers would ask the network to predict the value of guesses that are already
        eliminated - a value no policy could ever realise.
        """
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        nextObservations = stackObservations(batch["nextObservations"]).to(self.device)
        nextMasks = stackMasks(batch["nextMasks"]).to(self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            nextValues = self.targetNet(nextObservations)
            best = maskedLogits(nextValues, nextMasks).max(dim=-1).values
            return rewards + self.discount * best * (1.0 - dones)

    def update(self):
        """One gradient step on a batch of transitions."""
        batch = self.buffer.sample(self.batchSize)

        observations = stackObservations(batch["observations"]).to(self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)

        values = self.net(observations).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        targets = self.bellmanTargets(batch)

        loss = ((values - targets) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.updateCount += 1
        if self.updateCount % self.targetUpdateEvery == 0:
            self.targetNet.load_state_dict(self.net.state_dict())

        return {"loss": float(loss.item())}

    def scores(self, state):
        """The Q value of every answer, for evaluation."""
        observation = stackObservations([state.observation]).to(self.device)

        with torch.no_grad():
            return self.net(observation)[0]

    def train(self, episodes, env=None, seedBase=config.SEED, curve=None,
              totalEpisodes=None, evaluation=None, evaluationEvery=None):
        """Run `episodes` episodes of training."""
        env = env if env is not None else trainingEnv()
        totalEpisodes = totalEpisodes if totalEpisodes is not None else episodes

        # The exploration schedule is measured in STEPS, not episodes, so the number of
        # guesses per episode does not change how fast exploration fades.
        stepsPerEpisode = 5
        totalSteps = totalEpisodes * stepsPerEpisode

        for index, episode in episodeStream(env, self.chooser(), episodes, seedBase):
            self.episodeCount += 1
            self.epsilon = self.epsilonFor(self.stepCount, totalSteps)

            for transition in episode.transitions:
                self.buffer.add(
                    transition.observation, transition.mask, transition.action,
                    transition.reward, transition.nextObservation, transition.nextMask,
                    transition.done,
                )
                self.stepCount += 1

                if self.stepCount % self.trainEvery == 0 and len(self.buffer) >= self.batchSize:
                    self.update()

            if curve is not None:
                curve.append({
                    "episode": self.episodeCount,
                    "reward": episode.totalReward,
                    "won": episode.won,
                    "guessesUsed": episode.guessesUsed,
                    "firstGuess": episode.firstGuess,
                    "entropy": 0.0,
                    "entropyCoefficient": self.epsilon,
                    "loss": 0.0,
                })

            if evaluation is not None and evaluationEvery:
                if self.episodeCount % evaluationEvery == 0:
                    evaluation(self, self.episodeCount)

        return curve if curve is not None else []
