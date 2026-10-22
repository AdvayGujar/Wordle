"""Find the best opening word by exhaustive search over all 2,315 answers.

WHY THIS EXISTS: the reinforcement-learning agent in Phase C was trying to learn, from
reward alone, which word to open with - and it found a good one (`slate`, 3.795 guesses
with random play afterwards) but had no way to know whether it was the best. This module
answers that question by brute force: every answer is tried as the opener, with enough
games behind each one to rank them.

IT IS NOT AN AGENT AND NOT A BASELINE TO BEAT BY PLAYING. It is a *reference point*: it
says what the ceiling of the opening-word decision is, and how far the learned agent got.
The cost is the point too - it takes about a million simulated games, which is exactly the
sampling budget the analysis in `attempt2.md` says the agent needed.

WHY THE SEARCH RUNS ON THE TRAINING SECRETS: the best opener is a choice made from
training data, like any other hyper-parameter. The held-out 400 stay untouched until the
final evaluation.

TWO STAGES: the first pass gives every opener 400 games, which is enough to separate the
good ones from the bad ones but not to separate the good ones from each other (400 games
put a standard error of about 0.05 guesses on the average, and the best openers are within
0.05 of each other). So the top candidates are then re-measured with many more games.

Run it:

    PYTHONPATH=src .venv/bin/python -m attempt2.openerSearch
    PYTHONPATH=src .venv/bin/python -m attempt2.openerSearch --games 400 --refine-games 2000
"""

import argparse
import csv
import multiprocessing
from pathlib import Path

import numpy as np

from attempt2 import config
from attempt2.eval import PolicyState
from attempt2.wordleEnv import trainingEnv
from wordle.vocabulary import loadAnswers

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
SEARCH_FILE = ARTIFACT_DIR / "opener-search.csv"
SEARCH_ALL_FILE = ARTIFACT_DIR / "opener-search-all.csv"

# How many games each opener gets in the first pass, and in the refinement.
DEFAULT_GAMES = 400
DEFAULT_REFINE_GAMES = 2000
DEFAULT_REFINE_COUNT = 100

# The seed base for the search. Each opener derives its own seeds from its index, so the
# result does not depend on how many workers ran it.
SEED_BASE = 500000


def evaluateOpener(task):
    """Play `games` games opening with one word, then masked random.

    Returns `(index, averageGuesses, winRate, standardError)`.

    WHY MASKED RANDOM AFTER THE OPENER: the point is to measure the *opener*. Every
    later turn is a uniform choice among the legal answers, so the only thing that varies
    between the openers is the first guess.
    """
    index, games, seedBase = task

    word = loadAnswers()[index]
    env = trainingEnv()
    rng = np.random.default_rng(seedBase + index)

    guesses = []
    wins = 0

    for game in range(games):
        env.reset(seed=seedBase + index * 100000 + game)

        used = 0
        terminated = truncated = False

        while not (terminated or truncated):
            if used == 0:
                action = index
            else:
                legal = np.flatnonzero(env.action_mask())
                action = int(rng.choice(legal))

            _, _, terminated, truncated, _ = env.step(action)
            used += 1

        guesses.append(used)
        wins += int(terminated)

    average = float(np.mean(guesses))
    standardError = float(np.std(guesses) / np.sqrt(games))

    return index, average, wins / games, standardError


def search(words, games, seedBase=SEED_BASE, workers=None):
    """Evaluate every word as an opener.

    The results do not depend on the number of workers: each opener's games use seeds
    derived from its own index and its own random stream.
    """
    tasks = [(index, games, seedBase) for index in range(len(words))]

    if workers and workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(evaluateOpener, tasks, chunksize=16)
    else:
        results = [evaluateOpener(task) for task in tasks]

    return _asRows(results, words)


def _asRows(results, words):
    """The worker output as `(word, average, winRate, standardError)` rows."""
    return [
        (words[index], average, winRate, standardError)
        for index, average, winRate, standardError in results
    ]


def ranked(results):
    """The results, best (fewest guesses) first."""
    return sorted(results, key=lambda row: (row[1], row[0]))


def refine(candidates, games, seedBase=SEED_BASE, workers=None):
    """Re-measure the top candidates with more games.

    WHY A SECOND PASS: 400 games put a standard error of about 0.05 guesses on each
    average, and the best openers sit within that of each other. Ranking them properly
    needs more games than ranking them against the bulk.
    """
    answers = loadAnswers()
    indices = [answers.index(word) for word in candidates]

    tasks = [(index, games, seedBase + 1) for index in indices]

    if workers and workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(evaluateOpener, tasks, chunksize=4)
    else:
        results = [evaluateOpener(task) for task in tasks]

    return _asRows(results, answers)


def writeResults(rows, path=SEARCH_FILE):
    """Write a ranking as a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "opener", "avgGuesses", "winRate", "standardError"])
        for rank, row in enumerate(rows, start=1):
            word, average, winRate, standardError = row
            writer.writerow([rank, word, "%.4f" % average, "%.4f" % winRate,
                             "%.4f" % standardError])

    return path


def evaluateOpenersWithPolicy(policy, words, games, seedBase=SEED_BASE):
    """Play games that open with a given word and then follow a policy.

    WHY THIS IS USEFUL: it separates the two decisions. The opener search says which
    word carries the most information; this says what an agent scores when it is handed
    that word and has to play the rest of the game with its own learned policy. The
    difference between the two numbers is the cost of the agent's later play.
    """
    answers = loadAnswers()
    env = trainingEnv()

    rows = []

    for word in words:
        index = answers.index(word)
        rng = np.random.default_rng(seedBase + index)

        guesses = 0
        wins = 0

        for game in range(games):
            env.reset(seed=seedBase + index * 100000 + game)

            used = 0
            terminated = truncated = False

            while not (terminated or truncated):
                if used == 0:
                    action = index
                else:
                    mask = env.action_mask()
                    state = PolicyState(observation, mask, tuple(env.history))
                    action = int(policy(state))

                observation, _, terminated, truncated, _ = env.step(action)
                used += 1

            guesses += used
            wins += int(terminated)

        rows.append((word, guesses / games, wins / games))

    return rows


def printTable(rows, count=20):
    """Print the top of the ranking, with the resolution of each measurement.

    WHY THE STANDARD ERROR IS SHOWN: the difference between the best openers is
    smaller than the error on a single measurement, so a table without it invites the
    reader to treat noise as a ranking.
    """
    print("| Rank | Opener | Avg guesses | +/- | Win rate |")
    print("|---|---|---|---|---|")

    for rank, row in enumerate(rows[:count], start=1):
        word, average, winRate, standardError = row
        print("| %d | `%s` | %.3f | %.3f | %.1f%% |"
              % (rank, word, average, standardError, winRate * 100))


def main():
    parser = argparse.ArgumentParser(description="Search all 2,315 openers")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="games per opener in the first pass")
    parser.add_argument("--refine-games", type=int, default=DEFAULT_REFINE_GAMES,
                        help="games per opener in the refinement pass")
    parser.add_argument("--refine-count", type=int, default=DEFAULT_REFINE_COUNT,
                        help="how many candidates to re-measure")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    words = list(loadAnswers())
    print("searching %d openers, %d games each (%d workers)"
          % (len(words), arguments.games, arguments.workers))

    results = search(words, arguments.games, workers=arguments.workers)
    first = ranked(results)

    print()
    print("=== first pass, all %d openers, %d games each ===" % (len(words), arguments.games))
    printTable(first, 20)

    print()
    print("=== refining the top %d with %d games each ===" % (
        arguments.refine_count, arguments.refine_games))

    candidates = [word for word, _, _, _ in first[:arguments.refine_count]]
    refined = ranked(refine(candidates, arguments.refine_games, workers=arguments.workers))

    print()
    printTable(refined, 20)

    path = writeResults(refined)
    allPath = writeResults(first, SEARCH_ALL_FILE)
    print()
    print("worst opener: %s at %.3f guesses" % (first[-1][0], first[-1][1]))
    print("written: %s" % path)
    print("written: %s" % allPath)


if __name__ == "__main__":
    main()
