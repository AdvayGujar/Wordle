"""Phase D: every policy on the same 400 held-out secrets, and the figures.

WHAT IT DOES: evaluates Attempt 3's six policies and every Attempt 1/2 reference through the
same harness (`attempt2.eval.evaluate`), on the same 400 held-out secrets, and writes

- `artifacts/experiment-results.json` - the table, the settings and the paired comparisons;
- `comparison.png` - average guesses per policy;
- `distribution.png` - wins by turn, per policy.

THE HELD-OUT SET IS READ ONCE. This is the only place in Attempt 3 that calls
`heldOutSecrets()`, and the policies it scores were fixed before the run: Gate C chose
between them on the validation slice (§3.9). Nothing here may be tuned afterwards, or the
number stops being a held-out result.

WHY THE REFERENCES ARE RE-EVALUATED RATHER THAN QUOTED FROM Attempt 2's REPORT: a comparison
is only as good as its common setup. Re-running every row through one harness, on one secret
set, removes the chance that a stale number from another run is being compared with a fresh
one. The learned agents are loaded from `attempt2/artifacts/`, so nothing is retrained.

WHY THE FIGURES LIVE NEXT TO THE CODE AND NOT IN `artifacts/`: the README embeds them, so
they have to be tracked files. Attempts 1 and 2 do the same with their figures.
"""

import json
from pathlib import Path

from attempt2 import config as attempt2Config
from attempt2.baselines import (
    attempt1SolverPolicy,
    fixedOpenerRandomPolicy,
    maskedRandomPolicy,
)
from attempt2.eval import EVAL_MODE, evaluate, formatResultsTable
from attempt2.experiment import loadAgent, openerThenOpener
from attempt2.splits import heldOutSecrets
from attempt3 import adaptivePicker, entropyPicker, lookahead, pairPicker, remainingPicker
from attempt3.config import (
    GUESS_SET,
    LOOKAHEAD_FIRST_SHORTLIST,
    LOOKAHEAD_SECOND_SHORTLIST,
    SCORE_ANCHOR_BITS,
    SCORE_SLOPE,
)
from attempt3.gateC import pairedSummary

HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts"

RESULTS_FILE = ARTIFACT_DIR / "experiment-results.json"
COMPARISON_PLOT = HERE / "comparison.png"
DISTRIBUTION_PLOT = HERE / "distribution.png"

# Attempt 2's best opener, re-run here so the row is produced by this run.
BEST_OPENER = "slate"

ATTEMPT3_COLOUR = "#4C78A8"
REFERENCE_COLOUR = "#B0B7C3"


def _pickerPolicy(module):
    """Adapt one of the Phase-B pickers (`pickGuess(history)`) to the harness interface."""

    def policy(state):
        return module.pickGuess(state.history)

    policy.__name__ = module.__name__
    return policy


def attempt3Policies():
    """Attempt 3's six policies, in the order the write-up discusses them."""
    return [
        ("C1 expected score", adaptivePicker.policy()),
        ("C1 no probes", adaptivePicker.policy(allowProbes=False)),
        ("C2 two-ply", lookahead.policy()),
        ("B1 entropy", _pickerPolicy(entropyPicker)),
        ("B2 remaining", _pickerPolicy(remainingPicker)),
        ("B3 pair, then B2", _pickerPolicy(pairPicker)),
    ]


def referencePolicies(includeAgents=True):
    """The Attempt 1/2 rows.

    `includeAgents=False` skips the four rows that need artifacts, which is what the tests
    and a dry run use: loading them imports Torch and reads 8 MB of weights.
    """
    policies = [
        ("masked random (A2)", maskedRandomPolicy(attempt2Config.SEED)),
        ("crane + random (A2)",
         fixedOpenerRandomPolicy("crane", attempt2Config.SEED + 1)),
        ("slate + random (A2)",
         openerThenOpener(BEST_OPENER, maskedRandomPolicy(attempt2Config.SEED + 2))),
    ]

    if not includeAgents:
        return policies

    for algorithm in ("reinforce", "ppo", "dqn"):
        try:
            policies.append(
                ("%s (A2)" % algorithm, loadAgent(algorithm))
            )
        except FileNotFoundError:
            print("skipping %s: no trained model" % algorithm)

    try:
        policies.append(("Attempt-1 solver", attempt1SolverPolicy()))
    except FileNotFoundError:
        print("skipping the Attempt-1 row: no trained model")

    return policies


def allPolicies(includeAgents=True):
    """Every row of the final table."""
    return attempt3Policies() + referencePolicies(includeAgents=includeAgents)


def evaluateAll(secrets, includeAgents=True, mode=EVAL_MODE):
    """Score every policy on the given secrets, through the one harness."""
    results = []

    for name, policy in allPolicies(includeAgents=includeAgents):
        print("  playing %s ..." % name)
        results.append(evaluate(policy, secrets, name=name, mode=mode))

    return results


def _sortedByGuesses(results):
    return sorted(results, key=lambda result: result.avgGuessesOnWins)


def _isAttempt3(name):
    return name.startswith(("C1", "C2", "B1", "B2", "B3"))


def plotComparison(results, path=COMPARISON_PLOT):
    """Bar chart of average guesses, Attempt 3 highlighted against the references."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = _sortedByGuesses(results)
    names = [result.name for result in ordered]
    values = [result.avgGuessesOnWins for result in ordered]
    winRates = [result.winRate * 100 for result in ordered]
    colours = [
        ATTEMPT3_COLOUR if _isAttempt3(result.name) else REFERENCE_COLOUR
        for result in ordered
    ]

    figure, axes = plt.subplots(figsize=(11, 5.5))

    bars = axes.barh(names, values, color=colours)
    axes.invert_yaxis()
    axes.set_xlabel("average guesses on wins (lower is better)")
    axes.set_title("Attempt 3 - every policy on the same 400 held-out secrets")
    axes.grid(axis="x", alpha=0.3)

    for bar, value, winRate in zip(bars, values, winRates):
        axes.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            "%.2f  (%.1f%% win)" % (value, winRate),
            va="center",
            fontsize=9,
        )

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

    return path


def plotDistribution(results, path=DISTRIBUTION_PLOT):
    """Stacked bars of wins by turn, so the shape of each policy is visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = _sortedByGuesses(results)
    names = [result.name for result in ordered]
    turns = sorted({turn for result in ordered for turn in result.guessDistribution})

    figure, axes = plt.subplots(figsize=(11, 5.5))

    left = [0] * len(ordered)
    for turn in turns:
        counts = [result.guessDistribution.get(turn, 0) for result in ordered]
        axes.barh(names, counts, left=left, label="turn %d" % turn)
        left = [base + count for base, count in zip(left, counts)]

    axes.invert_yaxis()
    axes.set_xlabel("games won")
    axes.set_title("Attempt 3 - when each policy wins, on 400 held-out secrets")
    axes.legend(fontsize=8, ncol=len(turns))
    axes.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)

    return path


def writeResults(results, path=RESULTS_FILE, secrets=None):
    """Write the table, the settings and the paired comparisons as JSON."""
    byName = {result.name: result for result in results}
    reference = byName.get("slate + random (A2)")
    attempt3Results = [r for r in results if _isAttempt3(r.name)]

    payload = {
        "secrets": len(secrets) if secrets is not None else None,
        "settings": {
            "guessSet": GUESS_SET,
            "scoreAnchorBits": SCORE_ANCHOR_BITS,
            "scoreSlope": SCORE_SLOPE,
            "lookaheadFirstShortlist": LOOKAHEAD_FIRST_SHORTLIST,
            "lookaheadSecondShortlist": LOOKAHEAD_SECOND_SHORTLIST,
        },
        "policies": [
            {
                "name": result.name,
                "winRate": result.winRate,
                "avgGuessesOnWins": result.avgGuessesOnWins,
                "avgGuessesAll": result.avgGuessesAll,
                "guessDistribution": {
                    str(k): v for k, v in result.guessDistribution.items()
                },
                "firstGuessCounts": result.firstGuessCounts,
                "offCandidateProbes": result.illegalActions,
            }
            for result in results
        ],
        "pairedAgainstC1": (
            [
                pairedSummary(
                    byName["C1 expected score"], result
                )
                for result in results
                if result.name != "C1 expected score"
                and (result.name == reference.name or result in attempt3Results)
            ]
            if reference is not None and "C1 expected score" in byName
            else []
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return path


def main(secrets=None, includeAgents=True, mode=EVAL_MODE):
    """Run the experiment. Defaults to the 400 held-out secrets, read once."""
    if secrets is None:
        secrets = heldOutSecrets()

    print("evaluating every policy on the %d secrets" % len(secrets))
    print()

    results = evaluateAll(secrets, includeAgents=includeAgents, mode=mode)

    print()
    print(formatResultsTable(_sortedByGuesses(results)))
    print()
    print("The Illegal column counts off-candidate probes, not errors (§4.1.5).")

    byName = {result.name: result for result in results}
    c1 = byName.get("C1 expected score")

    if c1 is not None:
        print()
        print("paired against C1, per secret (difference = other - C1; negative = other):")
        for left, right in (
            ("C1 expected score", "B1 entropy"),
            ("C1 expected score", "slate + random (A2)"),
            ("C1 expected score", "C1 no probes"),
            ("C1 expected score", "C2 two-ply"),
        ):
            if left not in byName or right not in byName:
                continue
            summary = pairedSummary(byName[left], byName[right])
            print(
                "  %-22s %+.3f +/- %.3f  (better %d, worse %d, same %d)"
                % (
                    right,
                    summary["meanDifference"],
                    1.96 * summary["standardError"],
                    summary["better"],
                    summary["worse"],
                    summary["same"],
                )
            )

    print()
    for path in (
        plotComparison(results),
        plotDistribution(results),
        writeResults(results, secrets=secrets),
    ):
        print("written: %s" % path)


if __name__ == "__main__":
    main()
