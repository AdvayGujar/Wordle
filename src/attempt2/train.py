"""Train one agent and write the artifacts the report needs.

Run it:

    PYTHONPATH=src .venv/bin/python -m attempt2.train --algo reinforce --episodes 200000
    PYTHONPATH=src .venv/bin/python -m attempt2.train --algo dqn --episodes 50000
    PYTHONPATH=src .venv/bin/python -m attempt2.train --algo ppo --episodes 100000

WHAT IT WRITES (into `src/attempt2/artifacts/`):
- `<algo>-model.pt`  - the trained network's weights
- `<algo>-curve.csv` - the learning curve, one row per logged episode
- `<algo>-results.json` - the settings, the curve summary and the validation numbers

WHY THE VALIDATION NUMBERS ARE IN THERE AND THE HELD-OUT ONES ARE NOT: the held-out 400
may be read once, for the final table (plan.md §8.6). Everything a training run uses to
decide anything - the learning curve, the settings, the point at which to stop - comes
from the validation slice of the training half.
"""

import argparse
import csv
import inspect
import json
import time
from pathlib import Path

from attempt2 import config
from attempt2.eval import evaluateValidation
from attempt2.policies import greedyPolicyFromScorer

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# The algorithms this CLI can run. Each one is a factory: settings in, trainer out.
ALGORITHMS = ("reinforce", "dqn", "ppo")


def buildTrainer(algorithm, seed, options):
    """Build the trainer for an algorithm name.

    WHY THE OPTIONS ARE FILTERED: the three algorithms do not share a hyper-parameter
    set - DQN has an epsilon schedule and no entropy bonus, for instance. Passing an
    option an algorithm does not understand would be a crash halfway through a long run,
    so the options are matched against the trainer's signature and the ones that do not
    apply are dropped with a warning.
    """
    trainerClass = _trainerClasses()[algorithm]

    supported = inspect.signature(trainerClass.__init__).parameters
    accepted = {}
    dropped = []

    for name, value in options.items():
        if name in supported:
            accepted[name] = value
        else:
            dropped.append(name)

    if dropped:
        print("warning: %s does not use %s; ignoring" % (algorithm, ", ".join(dropped)))

    return trainerClass(seed=seed, **accepted)


def _trainerClasses():
    """The trainer class for each algorithm name."""
    from attempt2.dqn import DqnTrainer
    from attempt2.ppo import PpoTrainer
    from attempt2.reinforce import ReinforceTrainer

    return {
        "reinforce": ReinforceTrainer,
        "dqn": DqnTrainer,
        "ppo": PpoTrainer,
    }


def train(algorithm, episodes, seed=config.SEED, options=None, evaluationEvery=None,
          artifactDirectory=None, verbose=True):
    """Train one agent, log the curve, and write the artifacts.

    Returns a dict with the settings, the curve and the validation results.
    """
    options = options if options is not None else {}
    artifactDirectory = artifactDirectory if artifactDirectory is not None else ARTIFACT_DIR
    artifactDirectory.mkdir(parents=True, exist_ok=True)

    trainer = buildTrainer(algorithm, seed, options)
    curve = []

    validation = []

    def evaluate(trainer, episode):
        result = evaluateValidation(greedyPolicyFromScorer(trainer.scores))
        validation.append({
            "episode": episode,
            "winRate": result.winRate,
            "avgGuessesOnWins": result.avgGuessesOnWins,
        })
        if verbose:
            print("  episode %7d | validation: %.1f%% win, %.2f guesses"
                  % (episode, result.winRate * 100, result.avgGuessesOnWins), flush=True)

    start = time.time()
    trainer.train(
        episodes,
        curve=curve,
        totalEpisodes=episodes,
        evaluation=evaluate,
        evaluationEvery=evaluationEvery,
    )
    elapsed = time.time() - start

    finalValidation = evaluateValidation(greedyPolicyFromScorer(trainer.scores))

    payload = {
        "algorithm": algorithm,
        "episodes": episodes,
        "seed": seed,
        "options": {key: str(value) for key, value in options.items()},
        "elapsedSeconds": elapsed,
        "validation": validation,
        "finalValidation": {
            "winRate": finalValidation.winRate,
            "avgGuessesOnWins": finalValidation.avgGuessesOnWins,
            "avgGuessesAll": finalValidation.avgGuessesAll,
            "firstGuessCounts": finalValidation.firstGuessCounts,
        },
    }

    modelPath = artifactDirectory / ("%s-model.pt" % algorithm)
    curvePath = artifactDirectory / ("%s-curve.csv" % algorithm)
    resultsPath = artifactDirectory / ("%s-results.json" % algorithm)

    torchSave(trainer, modelPath)
    writeCurve(curve, curvePath)
    resultsPath.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload["modelPath"] = str(modelPath)
    payload["curvePath"] = str(curvePath)
    payload["resultsPath"] = str(resultsPath)

    return payload


def torchSave(trainer, path):
    """Save the network's weights."""
    import torch

    torch.save(trainer.net.state_dict(), path)
    return path


def writeCurve(curve, path):
    """Write the learning curve as a CSV."""
    if not curve:
        path.write_text("", encoding="utf-8")
        return path

    fields = ["episode", "reward", "won", "guessesUsed", "firstGuess", "entropy",
              "entropyCoefficient", "loss"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(curve)

    return path


def main():
    parser = argparse.ArgumentParser(description="Train an Attempt-2 agent")
    parser.add_argument("--algo", choices=ALGORITHMS, required=True)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-start", type=float, default=None)
    parser.add_argument("--entropy-end", type=float, default=None)
    parser.add_argument("--evaluation-every", type=int, default=10000)
    parser.add_argument("--artifacts", default=None,
                        help="where to write the model, the curve and the results")
    arguments = parser.parse_args()

    options = {}
    if arguments.learning_rate is not None:
        options["learningRate"] = arguments.learning_rate
    if arguments.entropy_start is not None:
        options["entropyStart"] = arguments.entropy_start
    if arguments.entropy_end is not None:
        options["entropyEnd"] = arguments.entropy_end

    print("training %s for %d episodes (seed %d)" % (
        arguments.algo, arguments.episodes, arguments.seed))
    print("options: %s" % (options or "defaults"))

    payload = train(
        arguments.algo,
        arguments.episodes,
        seed=arguments.seed,
        options=options,
        evaluationEvery=arguments.evaluation_every,
        artifactDirectory=Path(arguments.artifacts) if arguments.artifacts else None,
    )

    print()
    print("finished in %.1f s" % payload["elapsedSeconds"])
    print("validation: %.1f%% win, %.2f guesses" % (
        payload["finalValidation"]["winRate"] * 100,
        payload["finalValidation"]["avgGuessesOnWins"],
    ))
    print("model:  %s" % payload["modelPath"])
    print("curve:  %s" % payload["curvePath"])
    print("results: %s" % payload["resultsPath"])


if __name__ == "__main__":
    main()
