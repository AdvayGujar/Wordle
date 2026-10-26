"""Gate C: does the expected-score policy beat pure entropy on the validation slice?

WHY THIS MODULE EXISTS: `attempt3.md` §4.8 makes Gate C a measurement - C1 has to beat both
B1 (pure entropy, 3.46 from Gate B) and its own no-probe variant, on the same 200 validation
secrets. As with Gate B, the plan names no harness, so this is it.

WHY B1 IS RE-RUN RATHER THAN QUOTED: the three rows have to be produced by one process, one
harness and one set of secrets, or a difference in the setup can masquerade as a difference
in the policy. B1's number should reproduce §3.8 exactly; if it does not, something changed
and the comparison is void.

WHY C2 IS HERE: it is a Phase-C policy and its number belongs next to C1's, even though the
gate itself only constrains C1.

THE "OFF-CANDIDATE" COLUMN counts probes: legal guesses that the evidence has ruled out.
C1 with probes and C2 play them on purpose; the no-probe row should be 0.
"""

import json
from pathlib import Path

import numpy as np

from attempt2.eval import evaluateValidation, formatResultsTable
from attempt3 import adaptivePicker, entropyPicker, lookahead

RESULT_FILE = Path(__file__).resolve().parent / "artifacts" / "gateC-results.json"


def _policies():
    """The four rows, built lazily so the caches are cold for each run."""

    def entropy(state):
        return entropyPicker.pickGuess(state.history)

    entropy.__name__ = "entropyPicker"

    return (
        ("B1 entropy (reference)", entropy),
        ("C1 expected score", adaptivePicker.policy()),
        ("C1 expected score, no probes", adaptivePicker.policy(allowProbes=False)),
        ("C2 two-ply", lookahead.policy()),
    )


def pairedSummary(reference, other):
    """Per-secret paired comparison of two results, in guesses.

    WHY PAIRED: both policies play the same 200 secrets, so the useful question is not
    "how far apart are the averages" but "on how many secrets did one need fewer guesses".
    The standard error of the paired difference is far smaller than the standard error of
    either average, which is the only way a 0.02-guess gap can be judged at all.

    A loss is costed at the full guess budget, exactly as `avgGuessesAll` does, so the
    comparison does not reward losing early.
    """

    from attempt2 import config

    bySecret = {secret: (won, guesses) for secret, won, guesses in reference.perGame}

    differences = []
    better = worse = same = 0

    for secret, won, guesses in other.perGame:
        referenceWon, referenceGuesses = bySecret[secret]
        cost = guesses if won else config.MAX_GUESSES
        referenceCost = referenceGuesses if referenceWon else config.MAX_GUESSES
        difference = cost - referenceCost

        differences.append(difference)
        if difference < 0:
            better += 1
        elif difference > 0:
            worse += 1
        else:
            same += 1

    values = np.asarray(differences, dtype=np.float64)

    # One secret has no spread to estimate; reporting 0.0 is honest, and nan would poison a
    # caller's arithmetic for a case that only appears in tests.
    standardError = (
        float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    )

    return {
        "reference": reference.name,
        "other": other.name,
        "meanDifference": float(values.mean()),
        "standardError": standardError,
        "better": better,
        "worse": worse,
        "same": same,
    }


def run():
    """Evaluate every Phase-C policy on the validation secrets."""
    return [
        evaluateValidation(policy, name=name) for name, policy in _policies()
    ]


def main():
    results = run()
    print(formatResultsTable(results))
    print()
    print("The Illegal column counts off-candidate probes, not errors (§4.1.5).")
    print()

    byName = {result.name: result for result in results}

    print("paired, per secret (difference = right - left; negative means the right policy")
    print("needed fewer guesses):")
    for left, right in (
        ("C1 expected score", "B1 entropy (reference)"),
        ("C1 expected score", "C1 expected score, no probes"),
        ("C1 expected score", "C2 two-ply"),
    ):
        summary = pairedSummary(byName[left], byName[right])
        print(
            "  %-34s vs %-30s %+.3f +/- %.3f  (better %d, worse %d, same %d)"
            % (
                left,
                right,
                summary["meanDifference"],
                1.96 * summary["standardError"],
                summary["better"],
                summary["worse"],
                summary["same"],
            )
        )

    reference = byName["B1 entropy (reference)"]
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": result.name,
            "games": result.games,
            "wins": result.wins,
            "winRate": result.winRate,
            "avgGuessesOnWins": result.avgGuessesOnWins,
            "avgGuessesAll": result.avgGuessesAll,
            "guessDistribution": result.guessDistribution,
            "firstGuessCounts": result.firstGuessCounts,
            "offCandidateProbes": result.illegalActions,
            "pairedAgainstB1": (
                None
                if result.name == reference.name
                else pairedSummary(reference, result)
            ),
        }
        for result in results
    ]
    RESULT_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print("written: %s" % RESULT_FILE)


if __name__ == "__main__":
    main()
