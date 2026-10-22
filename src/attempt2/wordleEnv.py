"""The Wordle environment: a Gymnasium `Env` the agent plays against.

WHAT THIS IS: one episode is one Wordle game. The agent sees the evidence so far, picks an
answer word, and the environment returns the colours plus a reward. Nothing about the
secret is visible until the agent guesses it.

THE GYMNASIUM CONTRACT (gymnasium 1.1):
- `reset(seed=...)` starts a new episode and returns `(observation, info)`.
- `step(action)` returns `(observation, reward, terminated, truncated, info)`.
- `terminated` means the episode ended by the rules (the guess was the secret);
  `truncated` means it ended by the clock (six guesses used). They are separate because a
  learning algorithm treats "reached the goal" and "ran out of time" differently.

WHY THE SECRET IS CHOSEN FROM A FIXED POOL: the agent learns by playing, so if the
environment could sample any of the 2,315 answers during training, every answer would be
contaminated and a "test win rate" would mean nothing. `reset()` in "train" mode can only
draw from `attempt2.splits.trainSecrets()`; "eval" mode can only draw from the held-out
secrets. There is deliberately no attribute holding "all answers" that a later reset could
reach (plan.md §8.6).

WHY FEEDBACK COMES FROM THE TABLE: `wordle.feedbackTable.feedbackTable()` precomputes every
guess/secret pair, so one round of feedback is an array read instead of a pure-Python
dictionary pass. An RL run makes millions of these calls, and the difference is the
difference between minutes and hours.

WHY THE ACTION IS AN ANSWER INDEX: the action space is the 2,315 answers (plan.md §8.2), so
the action IS the guess - there is no mapping layer to get wrong. The consequence, recorded
in `attempt2.md`, is that the agent can never play a legal non-answer as a probe.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from attempt2 import config
from attempt2.actionMask import maskFor
from attempt2.observation import observationBounds, observationFor, observationWidth
from attempt2.splits import assertNoLeakage, heldOutSecrets, trainSecrets, trainingSecrets
from wordle.feedbackTable import decodePattern, feedbackTable
from wordle.vocabulary import answersVocab, loadAnswers

# The two ways an episode can end. "train" samples training secrets, "eval" samples the
# held-out ones. The names are load-bearing: mixing them up is the leak §8.6 warns about.
TRAIN_MODE = "train"
EVAL_MODE = "eval"
MODES = (TRAIN_MODE, EVAL_MODE)


class WordleEnv(gym.Env):
    """One Wordle game per episode, with legal-action masking."""

    metadata = {"render_modes": []}

    def __init__(self, mode=TRAIN_MODE, observationMode=config.OBSERVATION_MODE,
                 stepPenalty=config.STEP_PENALTY, winReward=config.WIN_REWARD,
                 lossReward=config.LOSS_REWARD, maxGuesses=config.MAX_GUESSES,
                 secrets=None):
        super().__init__()

        if mode not in MODES:
            raise ValueError("mode must be one of %r, got %r" % (MODES, mode))

        self.mode = mode
        self.observationMode = observationMode
        self.stepPenalty = stepPenalty
        self.winReward = winReward
        self.lossReward = lossReward
        self.maxGuesses = maxGuesses

        self._trainSecrets = trainSecrets()
        self._testSecrets = heldOutSecrets()

        # WHY THE ASSERTION HERE AND NOT ONLY IN A TEST: the environment is the one place
        # that can leak, so it checks its own inputs every time it is constructed.
        assertNoLeakage(self._trainSecrets, self._testSecrets)

        self._secretPool = self._trainSecrets if mode == TRAIN_MODE else self._testSecrets

        # WHY A SET: `reset(options={"secret": ...})` has to check the forced secret
        # against the pool, and a set lookup is O(1) instead of a scan of 1,915 words on
        # every episode.
        self._poolSet = set(self._secretPool)

        # WHY AN OPTIONAL SUBSET: training uses the training half minus the validation
        # slice, so the environment has to be able to sample from a smaller pool. The
        # subset is checked against this mode's pool, so it can never widen the pool and
        # the leak guard still holds.
        if secrets is not None:
            requested = list(secrets)
            outside = [word for word in requested if word not in self._poolSet]
            if outside:
                raise ValueError(
                    "%d secret(s) are not in the %s pool, e.g. %r"
                    % (len(outside), mode, outside[0])
                )
            self._secretPool = requested
            self._poolSet = set(requested)

        answers = loadAnswers()
        low, high = observationBounds(observationMode)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(observationWidth(observationMode),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(answers))

        self._history = []
        self._secret = None
        self._secretIndex = None
        self._mask = None
        self._finished = False

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Start a new episode.

        WHY `super().reset(seed=seed)`: that is how Gymnasium seeds `self.np_random`, and
        seeding it is what makes a run reproducible. The secret is drawn from that seeded
        generator, so `reset(seed=42)` always starts the same game.

        WHY `options={"secret": ...}` EXISTS: an evaluation has to play a *specific*
        secret, one after another, on the same set every time. Passing the secret
        through the standard Gymnasium `options` channel is the idiomatic way to do that,
        and it is checked against this mode's pool - so an eval-mode env refuses a
        training secret and a train-mode env refuses a held-out one. The leak guard
        therefore holds for forced secrets too, not just for sampled ones.
        """
        super().reset(seed=seed)

        secret = None
        if options:
            secret = options.get("secret")

        if secret is None:
            position = int(self.np_random.integers(len(self._secretPool)))
            secret = self._secretPool[position]
        elif secret not in self._poolSet:
            raise ValueError(
                "secret %r is not in the %s secret pool; this environment cannot play it"
                % (secret, self.mode)
            )

        self._secret = secret
        self._secretIndex = answersVocab().indexOf(self._secret)

        self._history = []
        self._mask = None
        self._finished = False

        observation = observationFor(self._history, self.observationMode)
        info = {
            "round": 0,
            "candidates": self.action_space.n,
            "guess": None,
            "solved": False,
        }

        return observation, info

    def step(self, action):
        """Play one guess and return the outcome."""
        if self._finished:
            raise RuntimeError(
                "the episode is already over; call reset() before stepping again"
            )

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                "action must be an answer index in [0, %d), got %r"
                % (self.action_space.n, action)
            )

        guess = answersVocab().wordAt(action)
        pattern = decodePattern(int(feedbackTable()[action, self._secretIndex]))
        self._history.append((guess, pattern))
        self._mask = None

        solved = guess == self._secret
        guessesUsed = len(self._history)
        truncated = guessesUsed >= self.maxGuesses and not solved
        self._finished = solved or truncated

        observation = observationFor(self._history, self.observationMode)
        info = {
            "round": guessesUsed,
            "candidates": int(self.action_mask().sum()),
            "guess": guess,
            "solved": solved,
        }

        reward = self._rewardFor(solved, truncated)

        return observation, reward, solved, truncated, info

    def _rewardFor(self, solved, truncated):
        """The reward for one finished round.

        WHY THIS SHAPE: every guess costs the step penalty, so a win in 3 returns 0.85 and
        a win in 6 returns 0.70 - finishing early is worth something but never worth more
        than winning. A loss returns only the penalties.
        """
        reward = self.stepPenalty
        if solved:
            reward += self.winReward
        elif truncated:
            reward += self.lossReward
        return reward

    # ------------------------------------------------------------------
    # Masking and diagnostics
    # ------------------------------------------------------------------

    def action_mask(self):
        """Boolean array over the action space: True where the action is legal.

        WHY IT IS CACHED: the policy asks for the mask every turn, and recomputing it
        would double the cost of a step. The cache is cleared on every new round, so it
        can never be stale.
        """
        if self._mask is None:
            self._mask = maskFor(self._history)
        return self._mask

    @property
    def history(self):
        """The `(guess, pattern)` pairs played so far."""
        return list(self._history)

    @property
    def secret(self):
        """The secret of the current episode.

        FOR DIAGNOSTICS AND TESTS ONLY. A policy that reads this is not solving Wordle, it
        is reading the answer key; the training code never touches it.
        """
        return self._secret

    @property
    def guessesUsed(self):
        return len(self._history)


def trainingEnv(**options):
    """An environment that samples only the secrets the agent may train on.

    WHY A HELPER: the training secrets are the training half *minus* the validation
    slice, and every trainer has to use that exact pool. Spelling it out in each trainer
    would be four chances to spell it differently.
    """
    return WordleEnv(mode=TRAIN_MODE, secrets=trainingSecrets(), **options)
