"""Play Attempt 2: you type the first guess and the learned agent plays the rest.

MODES
  (default)  interactive - you type guess 1, the agent plays the rest.
  --advise   you type every guess AND its colours; the agent suggests the next word.
             This is the human-in-the-loop mode: the agent never sees the secret, only
             the history you enter.
  --auto     evaluate the agent over many games, against the same baselines.

WHICH AGENT
  --algo picks the trained agent: reinforce (the best), dqn or ppo. The agent is loaded
  from `artifacts/`, so it has to have been trained first:

      PYTHONPATH=src .venv/bin/python -m attempt2.train --algo reinforce --episodes 250000 \\
          --learning-rate 0.01 --entropy-start 0.2 --entropy-end 0.01

WHY THE HUMAN MAY PLAY ANY LEGAL WORD: the agent's action space is the 2,315 answers,
because that is what it chooses from, but a person playing Wordle may type any legal guess.
`WordleEnv.playWord()` plays those guesses with the same rules, so the evidence the agent
reasons about is the evidence a real game would produce.

Run it:

    PYTHONPATH=src .venv/bin/python -m attempt2.play
    PYTHONPATH=src .venv/bin/python -m attempt2.play --advise
    PYTHONPATH=src .venv/bin/python -m attempt2.play --auto --games 400
"""

import argparse
import random

from attempt1.play import parsePattern
from attempt2 import config
from attempt2.actionMask import maskFor
from attempt2.baselines import attempt1SolverPolicy, fixedOpenerRandomPolicy, maskedRandomPolicy
from attempt2.eval import PolicyState, evaluate, formatResultsTable
from attempt2.observation import observationFor
from attempt2.policies import greedyPolicyFromScorer
from attempt2.splits import heldOutSecrets
from attempt2.wordleEnv import EVAL_MODE, WordleEnv
from wordle.feedback import candidatesMatching
from wordle.game import checkGuess
from wordle.vocabulary import answersVocab, loadAnswers

MAX_GUESSES = config.MAX_GUESSES
SQUARES = {0: "⬜", 1: "🟨", 2: "🟩"}
COLOR_NAMES = ("grey", "yellow", "green")

ALGORITHMS = ("reinforce", "dqn", "ppo")


def loadAgent(algorithm):
    """Load a trained agent and return it as a policy."""
    import torch

    from attempt2.dqn import DqnTrainer
    from attempt2.ppo import PpoTrainer
    from attempt2.reinforce import ReinforceTrainer

    trainerClass = {
        "reinforce": ReinforceTrainer, "dqn": DqnTrainer, "ppo": PpoTrainer,
    }[algorithm]
    trainer = trainerClass(seed=config.SEED)

    from attempt2.experiment import ARTIFACT_DIR

    path = ARTIFACT_DIR / ("%s-model.pt" % algorithm)
    if not path.exists():
        raise SystemExit(
            "no trained model at %s\n"
            "train one first:\n"
            "  PYTHONPATH=src .venv/bin/python -m attempt2.train --algo %s --episodes 250000"
            % (path, algorithm)
        )

    trainer.net.load_state_dict(torch.load(path, map_location="cpu"))
    trainer.net.eval()

    return greedyPolicyFromScorer(trainer.scores)


# ---------------------------------------------------------------------------
# The agent's suggestion, from a history alone
# ---------------------------------------------------------------------------

def suggestionFor(policy, history):
    """The agent's next guess for a history.

    WHY IT NEEDS NO ENVIRONMENT: the observation and the mask are both functions of the
    history alone, so the agent's advice can be computed without a game - which is what
    makes the advisor mode honest. The agent never sees the secret.
    """
    observation = observationFor(history)
    mask = maskFor(history)
    state = PolicyState(observation, mask, tuple(history))

    action = int(policy(state))
    return answersVocab().wordAt(action)


# ---------------------------------------------------------------------------
# Prompts and rendering
# ---------------------------------------------------------------------------

def promptGuess(history, label="Your guess"):
    """Ask for a guess, refusing anything that is not a fresh legal word."""
    played = {guess for guess, _ in history}

    while True:
        raw = input("%s (5 letters, or 'quit'): " % label).strip().lower()
        if raw in ("quit", "exit", "q"):
            raise SystemExit("bye")
        try:
            guess = checkGuess(raw)
        except ValueError as error:
            print("  %s" % error)
            continue
        if guess in played:
            print("  %s was already played; a useful solver never repeats a guess." % guess)
            continue
        return guess


def promptPattern(guess):
    """Ask for the five colours of a guess."""
    while True:
        raw = input(
            "Colours for %s (G/Y/X per position, or e.g. 2 1 0 0 2): " % guess.upper()
        )
        if raw.strip().lower() in ("quit", "exit", "q"):
            raise SystemExit("bye")
        try:
            return parsePattern(raw)
        except ValueError as error:
            print("  %s" % error)


def render(history, secret=None):
    """Print the whole board."""
    print()
    for turn, (guess, pattern) in enumerate(history, start=1):
        squares = "".join(SQUARES[color] for color in pattern)
        colors = ", ".join(COLOR_NAMES[color] for color in pattern)
        print("  %d. %s  %s   [%s]" % (turn, guess.upper(), squares, colors))

    if secret:
        print("\n  secret: %s" % secret.upper())


def renderTurn(history, turn):
    """Print one turn of the board.

    WHY ONE TURN AND NOT THE WHOLE BOARD: in the interactive game each turn arrives on
    its own, and reprinting the earlier turns after every guess makes the new one easy to
    miss.
    """
    guess, pattern = history[turn - 1]
    squares = "".join(SQUARES[color] for color in pattern)
    colors = ", ".join(COLOR_NAMES[color] for color in pattern)
    print("  %d. %s  %s   [%s]" % (turn, guess.upper(), squares, colors))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def playInteractive(policy, algorithm):
    """You type guess 1; the agent plays the rest."""
    env = WordleEnv(mode=EVAL_MODE, maxGuesses=MAX_GUESSES)
    env.reset(seed=random.Random().randrange(1 << 30))

    print("=" * 78)
    print("INTERACTIVE GAME - you type guess 1, the %s agent plays the rest" % algorithm)
    print("=" * 78)
    print("  The secret is one of the %d held-out answers." % len(heldOutSecrets()))
    print()

    guess = promptGuess([], "Your first guess")
    _, _, solved, truncated, _ = env.playWord(guess)
    print()
    renderTurn(env.history, 1)

    while not (solved or truncated):
        state = PolicyState(
            observationFor(env.history), env.action_mask(), tuple(env.history)
        )
        action = int(policy(state))
        _, _, solved, truncated, _ = env.step(action)
        renderTurn(env.history, env.guessesUsed)
        if not (solved or truncated):
            print("  candidates left: %d" % int(env.action_mask().sum()))

    if solved:
        print("\n  WON in %d guesses." % env.guessesUsed)
    else:
        print("\n  LOST. The secret was %s." % env.secret.upper())


def advise(policy, algorithm):
    """You enter every guess and its colours; the agent suggests the next word."""
    answers = loadAnswers()
    history = []

    print("=" * 78)
    print("ADVISOR - you enter each guess and its colours; the %s agent suggests the next"
          % algorithm)
    print("=" * 78)
    print("  Enter colours position by position, e.g. GYXXG or 2 1 0 0 2.")
    print("  Type 'quit' at any prompt to stop.")
    print()

    while True:
        guess = promptGuess(history, "Your guess")
        pattern = promptPattern(guess)
        trial = history + [(guess, pattern)]

        if not candidatesMatching(trial, answers):
            print("  No answer is consistent with that guess + colours. Check them and retry.")
            continue

        history.append((guess, pattern))
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
        print("  Next best word: %s" % suggestionFor(policy, history).upper())
        print()


def playAuto(algorithm, games):
    """Evaluate the agent against the baselines on the held-out secrets."""
    secrets = heldOutSecrets()[:games]

    policy = loadAgent(algorithm)
    baselines = [
        ("masked random", maskedRandomPolicy(config.SEED)),
        ("masked random, fixed opener 'crane'",
         fixedOpenerRandomPolicy("crane", config.SEED + 1)),
    ]

    try:
        baselines.append(("Attempt-1 solver (model + rule mask)", attempt1SolverPolicy()))
    except FileNotFoundError:
        pass

    results = [evaluate(policy, secrets, name="%s (learned)" % algorithm)]
    results += [evaluate(baseline, secrets, name=name) for name, baseline in baselines]

    print(formatResultsTable(results))


def main():
    parser = argparse.ArgumentParser(description="Play Attempt 2")
    parser.add_argument("--algo", choices=ALGORITHMS, default="reinforce",
                        help="which trained agent to use (default: reinforce)")
    parser.add_argument("--advise", action="store_true",
                        help="you enter every guess and its colours; the agent advises")
    parser.add_argument("--auto", action="store_true",
                        help="evaluate the agent over many games instead of playing one")
    parser.add_argument("--games", type=int, default=config.N_TEST_SECRETS,
                        help="how many held-out secrets to evaluate in --auto mode")
    arguments = parser.parse_args()

    if arguments.auto:
        playAuto(arguments.algo, arguments.games)
        return

    policy = loadAgent(arguments.algo)

    if arguments.advise:
        advise(policy, arguments.algo)
    else:
        playInteractive(policy, arguments.algo)


if __name__ == "__main__":
    main()
