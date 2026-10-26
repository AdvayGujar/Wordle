"""Play Attempt 3: the computed strategy plays, and you can take a turn.

MODES
  (default)  interactive - you type guess 1, the strategy plays the rest.
  --advise   the strategy proposes each word and you give back the colours. It opens, you
             report what the board showed, and it proposes the next word from the evidence
             it now has.
  --auto     evaluate the chosen strategy over held-out games, against the baselines.

WHICH STRATEGY
  --strategy picks which computed policy plays:

      c1          expected score, probes allowed (the attempt's policy, 3.46)
      c1-noprobe  expected score with the pool restricted to the feasible set (3.53)
      c2          two-ply lookahead (3.48)
      b1          greedy entropy (3.46)
      b2          greedy expected survivors (3.48)
      b3          the fixed pair `trice` then `salon`, then B2 (3.56)

  The numbers are the held-out result of `attempt3.md` §3.10; `--auto` re-measures on
  whichever secrets you ask for, so it is a spot check, not the reported result.

WHY THE HUMAN MAY PLAY ANY LEGAL WORD: the strategy's guess pool is the 2,315 answers, but a
person playing Wordle may type any legal guess. `WordleEnv.playWord()` plays those with the
same rules, so the evidence the strategy reasons about is the evidence a real game produces.

WHY THE ADVISOR IS HONEST: every strategy here is a pure function of the history -
`pickGuess` takes the (word, colours) rounds and nothing else - so the advisor mode cannot
see the secret. That is also why it can propose a word before any game exists.

Run it:

    PYTHONPATH=src .venv/bin/python -m attempt3.play
    PYTHONPATH=src .venv/bin/python -m attempt3.play --advise
    PYTHONPATH=src .venv/bin/python -m attempt3.play --strategy b1 --auto --games 100
"""

import argparse
import random

from attempt2.eval import evaluate, formatResultsTable
from attempt2.play import promptGuess, promptPattern, render, renderTurn
from attempt2.splits import heldOutSecrets
from attempt2.wordleEnv import EVAL_MODE, WordleEnv
from attempt3 import (
    adaptivePicker,
    config,
    entropyPicker,
    lookahead,
    pairPicker,
    remainingPicker,
)
from attempt3.experiment import referencePolicies
from wordle.feedback import candidatesMatching
from wordle.vocabulary import answersVocab, loadAnswers

MAX_GUESSES = config.MAX_GUESSES


def _noProbes(history):
    """C1 with the guess pool restricted to the feasible set."""
    return adaptivePicker.pickGuess(history, allowProbes=False)


# name -> (label for the screen, a function from history to answer index)
STRATEGIES = {
    "c1": ("C1 expected score", adaptivePicker.pickGuess),
    "c1-noprobe": ("C1 expected score, no probes", _noProbes),
    "c2": ("C2 two-ply", lookahead.pickGuess),
    "b1": ("B1 entropy", entropyPicker.pickGuess),
    "b2": ("B2 remaining", remainingPicker.pickGuess),
    "b3": ("B3 pair, then B2", pairPicker.pickGuess),
}


def strategyFor(name):
    """The (label, pickGuess) pair for a strategy name, with a loud error for a typo."""
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(
            "unknown strategy %r; choose one of %s" % (name, ", ".join(sorted(STRATEGIES)))
        ) from None


def nextWord(pickGuess, history):
    """The strategy's next word as a string, computed from the history alone."""
    return answersVocab().wordAt(pickGuess(tuple(history)))


def asPolicy(pickGuess):
    """Adapt `pickGuess(history)` to the `PolicyState` callable the harness wants."""

    def policy(state):
        return pickGuess(state.history)

    policy.__name__ = "attempt3Strategy"
    return policy


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def playInteractive(pickGuess, label):
    """You type guess 1; the strategy plays the rest."""
    env = WordleEnv(mode=EVAL_MODE, maxGuesses=MAX_GUESSES)
    env.reset(seed=random.Random().randrange(1 << 30))

    print("=" * 78)
    print("INTERACTIVE GAME - you type guess 1, %s plays the rest" % label)
    print("=" * 78)
    print("  The secret is one of the %d held-out answers." % len(heldOutSecrets()))
    print()

    guess = promptGuess([], "Your first guess")
    _, _, solved, truncated, _ = env.playWord(guess)
    print()
    renderTurn(env.history, 1)

    while not (solved or truncated):
        _, _, solved, truncated, _ = env.step(pickGuess(tuple(env.history)))
        renderTurn(env.history, env.guessesUsed)
        if not (solved or truncated):
            print("  candidates left: %d" % int(env.action_mask().sum()))

    if solved:
        print("\n  WON in %d guesses." % env.guessesUsed)
    else:
        print("\n  LOST. The secret was %s." % env.secret.upper())


def advise(pickGuess, label):
    """The strategy proposes each word; you give back the colours.

    The strategy never sees the secret: each proposal comes from the history of guesses and
    colours alone, so the mode is an honest demonstration of the policy.
    """
    answers = loadAnswers()
    history = []

    print("=" * 78)
    print("ADVISOR - %s proposes each word, you give back the colours" % label)
    print("=" * 78)
    print("  Enter the five colours of the word it proposes, e.g. GYXXG or 2 1 0 0 2.")
    print("  Type 'quit' at any prompt to stop.")
    print()

    while True:
        suggestion = nextWord(pickGuess, history)
        print("\n  The strategy's word: %s" % suggestion.upper())

        pattern = promptPattern(suggestion)
        trial = history + [(suggestion, pattern)]

        if not candidatesMatching(trial, answers):
            print("  No answer is consistent with those colours. Check them and retry.")
            continue

        history.append((suggestion, pattern))
        render(history)

        if all(color == 2 for color in pattern):
            print("\n  SOLVED - all five letters are green.")
            return

        if len(history) >= MAX_GUESSES:
            print("\n  That was guess %d; there is no next turn." % MAX_GUESSES)
            return

        consistent = candidatesMatching(history, answers)
        print("\n  Consistent answers left: %d" % len(consistent))
        if len(consistent) == 1:
            print("  Only one answer remains.")


def playAuto(name, games):
    """Evaluate the chosen strategy and the references on held-out secrets."""
    label, pickGuess = strategyFor(name)
    secrets = heldOutSecrets()[:games]

    results = [evaluate(asPolicy(pickGuess), secrets, name=label)]
    for referenceName, referencePolicy in referencePolicies(includeAgents=False):
        results.append(evaluate(referencePolicy, secrets, name=referenceName))

    print(
        "spot check on %d held-out secrets "
        "(the reported numbers come from attempt3.experiment)" % len(secrets)
    )
    print()
    print(formatResultsTable(results))
    print()
    print("The Illegal column counts off-candidate probes, not errors.")


def main():
    parser = argparse.ArgumentParser(description="Play Attempt 3")
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default="c1",
        help="which computed strategy plays (default: c1)",
    )
    parser.add_argument(
        "--advise",
        action="store_true",
        help="the strategy proposes each word; you give back the colours",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="evaluate the strategy over held-out games instead of playing one",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=config.N_TEST_SECRETS,
        help="how many held-out secrets to evaluate in --auto mode",
    )
    arguments = parser.parse_args()

    if arguments.auto:
        playAuto(arguments.strategy, arguments.games)
        return

    label, pickGuess = strategyFor(arguments.strategy)

    if arguments.advise:
        advise(pickGuess, label)
    else:
        playInteractive(pickGuess, label)


if __name__ == "__main__":
    main()
