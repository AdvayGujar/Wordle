"""A uniform replay buffer for the value-based algorithm.

WHY A SEPARATE MODULE: the buffer is the piece of DQN that is easy to get subtly wrong -
off-by-one on the next state, forgetting the next mask, mixing terminals with
time-limit endings - and it is the piece that has nothing to do with learning. Keeping it
on its own means it can be tested against hand-built sequences instead of being observed
through a training run.

WHY THE NEXT MASK IS STORED: the Bellman target is the maximum Q value over the actions
that are legal in the *next* state, not over all 2,315 answers. If the buffer did not
carry the next mask, DQN would be optimising towards the value of a guess the agent is not
allowed to make.
"""

from collections import deque

import numpy as np


class ReplayBuffer:
    """A fixed-size buffer of transitions, sampled uniformly."""

    def __init__(self, capacity=100000, seed=None):
        self.capacity = capacity
        self._transitions = deque(maxlen=capacity)
        self._random = np.random.default_rng(seed)

    def add(self, observation, mask, action, reward, nextObservation, nextMask, done):
        """Store one transition.

        `done` means the episode ended for a *terminal* reason - the guess was the secret.
        A time-limit ending (six guesses used) is NOT terminal: the game was cut short,
        so the value of the next state is not zero.
        """
        self._transitions.append(
            (
                np.asarray(observation, dtype=np.float32),
                np.asarray(mask, dtype=bool),
                int(action),
                float(reward),
                np.asarray(nextObservation, dtype=np.float32),
                np.asarray(nextMask, dtype=bool),
                bool(done),
            )
        )

    def sample(self, batchSize):
        """A batch of transitions, as parallel arrays."""
        if len(self._transitions) < batchSize:
            raise ValueError(
                "cannot sample %d transitions from a buffer holding %d"
                % (batchSize, len(self._transitions))
            )

        batch = self._random.choice(len(self._transitions), size=batchSize, replace=False)
        transitions = [self._transitions[position] for position in batch]

        observations, masks, actions, rewards = [], [], [], []
        nextObservations, nextMasks, dones = [], [], []

        for observation, mask, action, reward, nextObservation, nextMask, done in transitions:
            observations.append(observation)
            masks.append(mask)
            actions.append(action)
            rewards.append(reward)
            nextObservations.append(nextObservation)
            nextMasks.append(nextMask)
            dones.append(done)

        return {
            "observations": np.stack(observations),
            "masks": np.stack(masks),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "nextObservations": np.stack(nextObservations),
            "nextMasks": np.stack(nextMasks),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def __len__(self):
        return len(self._transitions)
