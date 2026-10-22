"""The one place where every Attempt-2 number is measured.

WHY A SINGLE HARNESS: the comparison in the final report is only meaningful if every row
was produced by the same code, on the same secrets, with the same rules. A baseline
measured by its own script and an agent measured by the training loop would differ in
some invisible way - a different max-guesses, a different secret set, a different
definition of "average guesses" - and the table would then be comparing two different
experiments. So there is exactly one `evaluate()` and every policy goes through it.

WHAT A POLICY IS: a callable that receives a `PolicyState` and returns an action index
(the index of the answer to guess). The state carries the three things a policy is allowed
to see - the observation vector, the legal-action mask, and the history of rounds - and
nothing else. In particular it never carries the secret.

WHY THE HARNESS PLAYS ILLEGAL ACTIONS INSTEAD OF FIXING THEM: the mask is a constraint on
the POLICY, not on the game. The environment plays whatever it is told, exactly like the
real game does - if an agent asks to play a word the feedback has already eliminated, it
loses a guess, and that is the honest cost of not using the mask. Every such request is
counted in `illegalActions`, so a policy that ignores the mask is visible in the report
rather than silently rescued.

WHY THE ENVIRONMENT IS USED RATHER THAN A REIMPLEMENTED GAME LOOP: the harness must score
exactly the environment the agent will be trained on - same rules, same mask, same reward.
A second copy of the game loop would drift from the first one, and the drift would show up
as a difference between training and evaluation.
"""

from dataclasses import dataclass
from random import Random

import numpy as np

from attempt2 import config
from attempt2.splits import validationSecrets
from attempt2.wordleEnv import EVAL_MODE, TRAIN_MODE, WordleEnv


@dataclass(frozen=True)
class PolicyState:
    """Everything a policy is allowed to see.

    WHY A DATACLASS AND NOT THREE ARGUMENTS: the policy interface has to stay the same
    when Phase C adds a learned policy, and a named bundle makes it obvious that the
    secret is not part of it.
    """

    observation: np.ndarray
    mask: np.ndarray
    history: tuple


@dataclass(frozen=True)
class EvaluationResult:
    """The score of one policy over one set of secrets."""

    name: str
    games: int
    wins: int
    losses: int
    winRate: float
    avgGuessesOnWins: float
    avgGuessesAll: float
    guessDistribution: dict
    firstGuessCounts: dict
    illegalActions: int
    perGame: tuple

    def summary(self):
        """A one-line summary, for logs."""
        return (
            "%s: %.1f%% win rate, %.2f guesses on wins, %d illegal actions"
            % (self.name, self.winRate * 100, self.avgGuessesOnWins, self.illegalActions)
        )


def evaluate(policy, secrets, name="policy", mode=EVAL_MODE,
             maxGuesses=config.MAX_GUESSES, seed=config.SEED):
    """Play every secret with this policy and score the result.

    `mode` decides which pool the secrets must come from: an eval-mode environment refuses
    a training secret, and a train-mode environment refuses a held-out one. That check is
    what makes "we evaluated on the held-out set" a property of the code rather than a
    claim in the report.
    """
    env = WordleEnv(mode=mode, maxGuesses=maxGuesses)

    wins = 0
    distribution = {turn: 0 for turn in range(1, maxGuesses + 1)}
    firstGuessCounts = {}
    illegalActions = 0
    perGame = []

    for index, secret in enumerate(secrets):
        result = _playOneGame(env, policy, secret, seed + index)
        wins += result["won"]
        illegalActions += result["illegalActions"]

        if result["won"]:
            distribution[result["guessesUsed"]] += 1

        firstGuess = result["firstGuess"]
        firstGuessCounts[firstGuess] = firstGuessCounts.get(firstGuess, 0) + 1
        perGame.append((secret, result["won"], result["guessesUsed"]))

    games = len(perGame)
    guessesOnWins = [guesses for _, won, guesses in perGame if won]
    guessesAll = [guesses if won else maxGuesses for _, won, guesses in perGame]

    return EvaluationResult(
        name=name,
        games=games,
        wins=wins,
        losses=games - wins,
        winRate=wins / games if games else 0.0,
        avgGuessesOnWins=float(np.mean(guessesOnWins)) if guessesOnWins else 0.0,
        avgGuessesAll=float(np.mean(guessesAll)) if guessesAll else 0.0,
        guessDistribution=distribution,
        firstGuessCounts=firstGuessCounts,
        illegalActions=illegalActions,
        perGame=tuple(perGame),
    )


def _playOneGame(env, policy, secret, seed):
    """Play one game and return what happened."""
    observation, _ = env.reset(seed=seed, options={"secret": secret})

    illegalActions = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_mask()
        state = PolicyState(observation, mask, tuple(env.history))

        action = int(policy(state))
        if not 0 <= action < env.action_space.n:
            raise ValueError(
                "policy %r returned action %r, which is not an answer index"
                % (getattr(policy, "__name__", policy), action)
            )

        if not mask[action]:
            illegalActions += 1

        observation, _, terminated, truncated, _ = env.step(action)

    return {
        "won": terminated,
        "guessesUsed": env.guessesUsed,
        "firstGuess": env.history[0][0],
        "illegalActions": illegalActions,
    }


def evaluateValidation(policy, name="validation", **options):
    """Score a policy on the validation slice of the training secrets.

    WHY A SEPARATE FUNCTION: the validation secrets are training secrets, so the
    environment must be in train mode to play them - an eval-mode env would (correctly)
    refuse. Keeping this in one place means no training script can accidentally point the
    held-out evaluation at the validation set, or the validation evaluation at the
    held-out set.
    """
    return evaluate(
        policy, validationSecrets(), name=name, mode=TRAIN_MODE, **options
    )


def formatResultsTable(results):
    """Format a list of EvaluationResults as a markdown table.

    WHY THE DISTRIBUTION IS A COLUMN AND NOT A SEPARATE TABLE: the distribution is what
    separates two policies with the same win rate, and putting it next to the win rate
    stops a reader from comparing the wrong thing.
    """
    header = "| Policy | Win rate | Avg guesses (wins) | Avg guesses (all) | Wins by turn | Illegal |"
    divider = "|---|---|---|---|---|---|"

    lines = [header, divider]

    for result in results:
        distribution = ", ".join(
            "%d:%d" % (turn, count)
            for turn, count in sorted(result.guessDistribution.items())
            if count
        )
        lines.append(
            "| %s | %.1f%% | %.2f | %.2f | %s | %d |"
            % (
                result.name,
                result.winRate * 100,
                result.avgGuessesOnWins,
                result.avgGuessesAll,
                distribution or "-",
                result.illegalActions,
            )
        )

    return "\n".join(lines)


def firstGuessTable(counts, limit=10):
    """Format a first-guess histogram as a markdown table of the most used openers."""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]

    lines = ["| Opening guess | Times used |", "|---|---|"]
    for word, count in ranked:
        lines.append("| %s | %d |" % (word, count))

    return "\n".join(lines)
