"""Attempt 1 solver: drive the candidate-consistency model through real games.

MODES
  interactive - you type guess 1, the model plays the rest.
  auto        - evaluate the model and baselines over many games.
  advise      - you type every guess AND its colours; the model suggests the next best word.
                This is the human-in-the-loop mode: the model never sees the secret, only the
                evidence you enter.

TWO PICKER POLICIES
  masked     - the rule-based candidate filter keeps only feedback-consistent words, then the
               model chooses among them. The safe solver.
  model-only - the model scores all 2,315 answers with no rule mask. The honest test of what
               the classifier learned.

WHY BOTH: the posterior over consistent candidates is uniform, so the model cannot rank the
true secret above the others. `masked` therefore measures the rules; `model-only` measures
the learned consistency checker. Report both instead of hiding the distinction.

Run interactively:
    PYTHONPATH=src .venv/bin/python src/attempt1/play.py

Ask for a suggestion after each guess:
    PYTHONPATH=src .venv/bin/python src/attempt1/play.py --advise

Run an evaluation:
    PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --games 400
    PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --model-only
"""

import argparse
import random

import numpy as np

from attempt1.common import (
    PLAY_RESULTS_FILE,
    SEED,
    loadArtifact,
    positiveScores,
    saveJson,
    testSecrets,
)
from attempt1.features import FEATURE_WIDTH, candidateFeatures
from wordle.feedback import candidatesMatching, feedback
from wordle.game import GameState, checkGuess, playGameDetailed
from wordle.vocabulary import loadAnswers, loadWords

MAX_GUESSES = 6
SQUARES = {0: "⬜", 1: "🟨", 2: "🟩"}
COLOR_NAMES = ("grey", "yellow", "green")

# What the advisor accepts for one position. Letters are the compact form; words and digits
# are the friendly forms. Grey is both "grey" and the American "gray".
COLOR_CODES = {
    "g": 2, "green": 2, "2": 2, "🟩": 2,
    "y": 1, "yellow": 1, "1": 1, "🟨": 1,
    "x": 0, "grey": 0, "gray": 0, "0": 0, "⬜": 0,
}


def parsePattern(raw):
    """Parse five colours entered by the advisor user.

    Accepts compact strings ("gyxgy", "21200"), separate tokens ("g y x g y", "2 1 0 0 2"),
    and words ("green yellow grey grey green"). Raises ValueError with the accepted forms.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("enter five colours, e.g. GYXXG or 2 1 0 0 2")

    cleaned = raw.strip().lower().replace(",", " ")
    tokens = cleaned.split()

    if len(tokens) == 1 and tokens[0] not in COLOR_CODES:
        tokens = list(tokens[0])

    if len(tokens) != 5:
        raise ValueError("need exactly 5 colours, got %d (e.g. GYXXG or 2 1 0 0 2)" % len(tokens))

    pattern = []
    for token in tokens:
        if token not in COLOR_CODES:
            raise ValueError("unknown colour %r; use G/Y/X, 2/1/0, or green/yellow/grey" % token)
        pattern.append(COLOR_CODES[token])

    return tuple(pattern)


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------

def modelScores(model, evidence, candidates):
    rows = np.asarray(
        [candidateFeatures(evidence, candidate) for candidate in candidates],
        dtype = float,
    )
    return positiveScores(model, rows)


def pickWithModel(model, history, mask = True, firstGuess = None):
    """Return the model's preferred legal next guess.

    WHY the mask: `candidatesMatching` is the exact rule. Without it the model is being
    asked to rank words it may have already eliminated. With it, the model only breaks ties
    inside a set whose members are (under a uniform secret) equally likely.
    """
    if not history:
        return firstGuess

    answers = loadAnswers()
    played = {guess for guess, _ in history}

    if mask:
        pool = candidatesMatching(history, answers)
    else:
        pool = list(answers)

    pool = [word for word in pool if word not in played]

    if not pool:
        raise RuntimeError("no legal candidates left; the history and vocabulary disagree")

    scores = modelScores(model, history, pool)
    return pool[int(np.argmax(scores))]


def modelFromArtifact(artifact):
    if artifact["featureWidth"] != FEATURE_WIDTH:
        raise ValueError(
            "model was trained with %d features, but this build produces %d"
            % (artifact["featureWidth"], FEATURE_WIDTH)
        )
    return artifact["model"]


def makeModelPicker(artifact, firstGuess, mask, interactiveFirst = False):
    model = modelFromArtifact(artifact)

    def picker(history):
        if not history:
            if interactiveFirst:
                return promptFirstGuess()
            return firstGuess
        return pickWithModel(model, history, mask = mask)

    return picker


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def randomLegalPicker(rng):
    words = loadWords()

    def picker(history):
        played = {guess for guess, _ in history}
        pool = [word for word in words if word not in played]
        return pool[rng.randrange(len(pool))]

    return picker


def randomConsistentPicker(rng):
    answers = loadAnswers()

    def picker(history):
        played = {guess for guess, _ in history}
        pool = [word for word in candidatesMatching(history, answers) if word not in played]
        return pool[rng.randrange(len(pool))] if pool else answers[rng.randrange(len(answers))]

    return picker


def fixedWordPicker(word):
    return lambda history: word


def fixedOpenerConsistentPicker(firstGuess, rng):
    consistent = randomConsistentPicker(rng)

    def picker(history):
        if not history:
            return firstGuess
        return consistent(history)

    return picker


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def summarise(results, maxGuesses = MAX_GUESSES):
    wins = [result for result in results if result.won]
    guessesOnWins = [result.guessesUsed for result in wins]
    guessesAll = [result.guessesUsed if result.won else maxGuesses for result in results]
    distribution = {turn: 0 for turn in range(1, maxGuesses + 1)}

    for result in results:
        if result.won:
            distribution[result.guessesUsed] += 1

    return {
        "games": len(results),
        "wins": len(wins),
        "winRate": len(wins) / len(results),
        "averageGuessesOnWins": float(np.mean(guessesOnWins)) if wins else 0.0,
        "averageGuessesAll": float(np.mean(guessesAll)),
        "distribution": distribution,
    }


def evaluate(factory, secrets, maxGuesses = MAX_GUESSES):
    results = []

    for index, secret in enumerate(secrets):
        rng = random.Random(SEED + index)
        results.append(playGameDetailed(secret, factory(rng), maxGuesses))

    return summarise(results, maxGuesses)


def printEvaluation(name, summary):
    distribution = summary["distribution"]
    bars = " ".join("%d:%d" % (turn, distribution[turn]) for turn in distribution)
    print(
        "  %-28s win %.1f%%  avg(wins) %.2f  avg(all) %.2f  [%s]"
        % (
            name,
            100 * summary["winRate"],
            summary["averageGuessesOnWins"],
            summary["averageGuessesAll"],
            bars,
        )
    )


def evaluateAll(artifact, firstGuess, games, modelOnly):
    secrets = testSecrets()[:games]
    print("=" * 78)
    print("AUTO EVALUATION - %d held-out games" % len(secrets))
    print("=" * 78)
    print("  first guess: %s" % firstGuess)
    print()

    maskedPicker = makeModelPicker(artifact, firstGuess, mask = True)
    modelOnlyPicker = makeModelPicker(artifact, firstGuess, mask = False)
    summaries = {}

    summaries["modelMasked"] = evaluate(lambda rng: maskedPicker, secrets)
    printEvaluation("model (masked)", summaries["modelMasked"])
    summaries["modelOnly"] = evaluate(lambda rng: modelOnlyPicker, secrets)
    printEvaluation("model (model-only)", summaries["modelOnly"])
    summaries["randomLegal"] = evaluate(lambda rng: randomLegalPicker(rng), secrets)
    printEvaluation("random-legal", summaries["randomLegal"])
    summaries["randomConsistent"] = evaluate(lambda rng: randomConsistentPicker(rng), secrets)
    printEvaluation("random-consistent", summaries["randomConsistent"])
    summaries["fixedWord"] = evaluate(lambda rng: fixedWordPicker(firstGuess), secrets)
    printEvaluation("fixed-word", summaries["fixedWord"])
    summaries["fixedOpenerRandom"] = evaluate(
        lambda rng: fixedOpenerConsistentPicker(firstGuess, rng), secrets
    )
    printEvaluation("fixed-opener + random", summaries["fixedOpenerRandom"])
    print()

    payload = {
        "games": len(secrets),
        "firstGuess": firstGuess,
        "primaryMode": "model-only" if modelOnly else "masked",
        "secrets": secrets,
        "summaries": summaries,
    }
    saveJson(PLAY_RESULTS_FILE, payload)
    print("  results saved: %s" % PLAY_RESULTS_FILE)
    print()
    return secrets


# ---------------------------------------------------------------------------
# Interactive
# ---------------------------------------------------------------------------

def promptGuess(history, label = "Your guess"):
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


def promptFirstGuess():
    return promptGuess([], "Your first guess")


def promptPattern(guess):
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


def advise(artifact, modelOnly):
    """Human types every guess and its colours; the model suggests the next word.

    This is the only mode where the model never sees the secret at all: it gets exactly the
    history the human enters. It is therefore the most honest interactive demonstration.
    """
    model = modelFromArtifact(artifact)
    answers = loadAnswers()
    history = []

    print("=" * 78)
    print("ADVISOR - you enter each guess and its colours; the model suggests the next word")
    print("=" * 78)
    print("  Enter colours position by position, e.g. GYXXG or 2 1 0 0 2.")
    print("  Type 'quit' at any prompt to stop.")
    print("  Model mode: %s" % ("model-only (no rule mask)" if modelOnly else "masked"))
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
            print("\n  That was guess 6; there is no next turn.")
            return

        consistent = candidatesMatching(history, answers)
        suggestion = pickWithModel(model, history, mask = not modelOnly)
        print("\n  Consistent answers left: %d" % len(consistent))
        if len(consistent) == 1:
            print("  Only one answer remains.")
        print("  Next best word: %s" % suggestion.upper())
        print()


def render(history, secret = None):
    print()
    for turn, (guess, pattern) in enumerate(history, start = 1):
        squares = "".join(SQUARES[color] for color in pattern)
        colors = ", ".join(COLOR_NAMES[color] for color in pattern)
        print("  %d. %s  %s   [%s]" % (turn, guess.upper(), squares, colors))

    if secret:
        print("\n  secret: %s" % secret.upper())


def playInteractive(artifact, firstGuess, modelOnly):
    secrets = testSecrets()
    secret = secrets[random.Random().randrange(len(secrets))]
    picker = makeModelPicker(artifact, firstGuess, mask = not modelOnly, interactiveFirst = True)

    print("=" * 78)
    print("INTERACTIVE GAME - you type guess 1, the model plays the rest")
    print("=" * 78)
    print("  model mode: %s" % ("model-only" if modelOnly else "masked"))

    state = GameState()

    for _ in range(MAX_GUESSES):
        guess = checkGuess(picker(state.guesses))
        pattern = feedback(guess, secret)
        state.addGuess(guess, pattern)
        render(state.guesses)
        if state.isSolved(secret):
            print("\n  WON in %d guesses." % len(state.guesses))
            return
        if len(state.guesses) < MAX_GUESSES:
            print("  candidates left: %d"
                  % len(candidatesMatching(state.guesses, loadAnswers())))

    print("\n  LOST. The secret was %s." % secret.upper())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parseArgs():
    parser = argparse.ArgumentParser(description = "Attempt 1 Wordle solver")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--auto", action = "store_true",
                      help = "evaluate instead of playing interactively")
    mode.add_argument("--advise", action = "store_true",
                      help = "enter every guess + its colours; the model suggests the next word")
    parser.add_argument("--first", default = "crane", help = "fixed first guess (default: crane)")
    parser.add_argument("--games", type = int, default = 400, help = "number of evaluation games")
    parser.add_argument("--model-only", action = "store_true",
                        help = "do not apply the feedback-consistency rule mask")
    return parser.parse_args()


def main():
    args = parseArgs()

    try:
        artifact = loadArtifact()
    except FileNotFoundError as error:
        print(error)
        return 1

    if args.advise:
        advise(artifact, args.model_only)
        return 0

    firstGuess = checkGuess(args.first.strip().lower())

    if args.auto:
        evaluateAll(artifact, firstGuess, args.games, args.model_only)
    else:
        playInteractive(artifact, firstGuess, args.model_only)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
