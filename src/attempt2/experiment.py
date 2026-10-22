"""Run the whole Attempt 2 experiment and write the final tables and figures.

WHAT IT DOES: evaluates every policy — the baselines and the three trained agents — on the
same 400 held-out secrets, through the same harness (`attempt2.eval.evaluate`), and writes

- `results.json` — the table, the settings and the environment facts
- `comparison.png` — the comparison bar chart
- `learning-curves.png` — the learning curve of each algorithm
- `first-guess.png` — what each agent opens with

WHAT IT DOES NOT DO: it never trains. The agents are loaded from `artifacts/`, so the
experiment can be re-run and re-plotted without repeating 27 minutes of training per
algorithm, and the held-out 400 are read exactly once per policy.

Run it:

    PYTHONPATH=src .venv/bin/python -m attempt2.experiment
"""

import csv
import json
from pathlib import Path

from attempt2 import config
from attempt2.baselines import attempt1SolverPolicy, fixedOpenerRandomPolicy, maskedRandomPolicy
from attempt2.eval import evaluate, formatResultsTable
from attempt2.policies import greedyPolicyFromScorer
from attempt2.splits import heldOutSecrets
from wordle.vocabulary import answersVocab

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_FILE = ARTIFACT_DIR / "experiment-results.json"
COMPARISON_PLOT = ARTIFACT_DIR / "comparison.png"
CURVES_PLOT = ARTIFACT_DIR / "learning-curves.png"
FIRST_GUESS_PLOT = ARTIFACT_DIR / "first-guess.png"

ALGORITHMS = ("reinforce", "dqn", "ppo")

# The information-theoretic reference: the best opener found by the exhaustive search,
# played with masked random. It is the number the learned agents have to beat, and it is
# not a baseline a policy can play on its own - it is a two-part reference.
BEST_OPENER = "slate"


def loadAgent(algorithm, artifactDirectory=ARTIFACT_DIR):
    """Load a trained agent and return it as an evaluation policy."""
    import torch

    from attempt2.dqn import DqnTrainer
    from attempt2.ppo import PpoTrainer
    from attempt2.reinforce import ReinforceTrainer

    trainerClass = {"reinforce": ReinforceTrainer, "dqn": DqnTrainer, "ppo": PpoTrainer}[algorithm]
    trainer = trainerClass(seed=config.SEED)

    path = artifactDirectory / ("%s-model.pt" % algorithm)
    if not path.exists():
        raise FileNotFoundError(
            "no trained model at %s; run attempt2.train --algo %s first" % (path, algorithm)
        )

    trainer.net.load_state_dict(torch.load(path, map_location="cpu"))
    trainer.net.eval()

    return greedyPolicyFromScorer(trainer.scores)


def openerThenOpener(firstWord, rest):
    """A policy that opens with one word and then follows another policy."""
    opener = answersVocab().indexOf(firstWord)

    def policy(state):
        if not state.history:
            return opener
        return rest(state)

    return policy


def baselinePolicies():
    """The baseline policies, with the Attempt-1 row last (it needs a trained model)."""
    return [
        ("masked random", maskedRandomPolicy(config.SEED)),
        ("masked random, fixed opener 'crane'",
         fixedOpenerRandomPolicy("crane", config.SEED + 1)),
        ("best opener 'slate' + masked random",
         openerThenOpener(BEST_OPENER, maskedRandomPolicy(config.SEED + 2))),
    ]


def agentPolicies(artifactDirectory=ARTIFACT_DIR):
    """The trained agents, as evaluation policies."""
    policies = []

    for algorithm in ALGORITHMS:
        try:
            policies.append(("%s (learned)" % algorithm, loadAgent(algorithm, artifactDirectory)))
        except FileNotFoundError:
            print("skipping %s: no trained model" % algorithm)

    try:
        policies.append(("Attempt-1 solver (model + rule mask)", attempt1SolverPolicy()))
    except FileNotFoundError:
        print("skipping the Attempt-1 row: no trained model")

    return policies


def evaluateAll(secrets):
    """Score every policy on the same secrets."""
    results = []

    for name, policy in baselinePolicies():
        results.append(evaluate(policy, secrets, name=name))

    for name, policy in agentPolicies():
        results.append(evaluate(policy, secrets, name=name))

    return results


def readCurve(algorithm, artifactDirectory=ARTIFACT_DIR):
    """The learning curve of one algorithm, as a list of rows."""
    path = artifactDirectory / ("%s-curve.csv" % algorithm)

    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plotComparison(results, path=COMPARISON_PLOT):
    """Bar chart of the average guesses, one bar per policy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [result.name for result in results]
    values = [result.avgGuessesOnWins for result in results]
    winRates = [result.winRate * 100 for result in results]

    figure, axes = plt.subplots(figsize=(11, 5))

    bars = axes.barh(names, values, color="#4C78A8")
    axes.invert_yaxis()
    axes.set_xlabel("average guesses on wins (lower is better)")
    axes.set_title("Attempt 2 - every policy on the same 400 held-out secrets")
    axes.grid(axis="x", alpha=0.3)

    for bar, value, winRate in zip(bars, values, winRates):
        axes.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
                  "%.2f  (%.1f%% win)" % (value, winRate), va="center", fontsize=9)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

    return path


def readValidation(algorithm, artifactDirectory=ARTIFACT_DIR):
    """The validation points of one algorithm, as (episode, averageGuesses) pairs.

    WHY THEY ARE PLOTTED NEXT TO THE TRAINING CURVE: the training curve is the policy
    scoring itself on the secrets it trained on, and a policy can improve that number
    without improving anything. Plotting the validation points on the same axes makes
    the generalisation gap visible instead of hiding it behind a falling line.
    """
    path = artifactDirectory / ("%s-results.json" % algorithm)

    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))

    return [
        (point["episode"], point["avgGuessesOnWins"])
        for point in payload.get("validation", [])
    ]


def plotLearningCurves(path=CURVES_PLOT):
    """The learning curve of each algorithm, with its validation points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 5))

    for algorithm in ALGORITHMS:
        rows = readCurve(algorithm)
        if not rows:
            continue

        episodes = [int(row["episode"]) for row in rows]
        guesses = [
            float(row["guessesUsed"]) if row["won"] == "True" else 6.0
            for row in rows
        ]

        window = max(1, len(guesses) // 100)
        smoothed = [
            sum(guesses[start:start + window]) / len(guesses[start:start + window])
            for start in range(0, len(guesses), window)
        ]
        smoothedEpisodes = episodes[::window][:len(smoothed)]

        axes.plot(smoothedEpisodes, smoothed, label="%s (training)" % algorithm, linewidth=1.2)

        validation = readValidation(algorithm)
        if validation:
            axes.plot(
                [episode for episode, _ in validation],
                [value for _, value in validation],
                "o", markersize=5, label="%s (validation)" % algorithm,
            )

    axes.set_xlabel("episode")
    axes.set_ylabel("average guesses (lower is better)")
    axes.set_title("Attempt 2 - learning curves, with the validation points")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

    return path


def plotFirstGuesses(results, path=FIRST_GUESS_PLOT):
    """What each agent opens with, over the held-out games."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = [result for result in results if result.firstGuessCounts]

    figure, axes = plt.subplots(figsize=(10, 5))

    for result in agents:
        counts = result.firstGuessCounts
        top = sorted(counts.items(), key=lambda item: -item[1])[:8]
        axes.barh(
            [word for word, _ in top],
            [count for _, count in top],
            alpha=0.5,
            label=result.name,
        )

    axes.set_xlabel("games opened with this word")
    axes.set_title("Attempt 2 - the opening word each policy used")
    axes.legend()
    axes.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

    return path


def writeResults(results, path=RESULTS_FILE, extra=None):
    """Write the table and the environment facts as JSON."""
    payload = {
        "secrets": len(heldOutSecrets()),
        "seed": config.SEED,
        "stepPenalty": config.STEP_PENALTY,
        "winReward": config.WIN_REWARD,
        "observationMode": config.OBSERVATION_MODE,
        "policies": [
            {
                "name": result.name,
                "winRate": result.winRate,
                "avgGuessesOnWins": result.avgGuessesOnWins,
                "avgGuessesAll": result.avgGuessesAll,
                "guessDistribution": {str(k): v for k, v in result.guessDistribution.items()},
                "firstGuessCounts": result.firstGuessCounts,
                "illegalActions": result.illegalActions,
            }
            for result in results
        ],
    }

    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return path


def main():
    secrets = heldOutSecrets()
    print("evaluating every policy on the %d held-out secrets" % len(secrets))
    print()

    results = evaluateAll(secrets)

    print(formatResultsTable(results))
    print()

    print("opening words used:")
    for result in results:
        counts = result.firstGuessCounts
        top = sorted(counts.items(), key=lambda item: -item[1])[:3]
        print("  %-42s %s" % (result.name, ", ".join("%s (%d)" % pair for pair in top) or "-"))
    print()

    paths = [
        plotComparison(results),
        plotLearningCurves(),
        plotFirstGuesses(results),
        writeResults(results),
    ]

    for path in paths:
        print("written: %s" % path)


if __name__ == "__main__":
    main()
