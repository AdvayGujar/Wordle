"""Smoke test for the environment and the evaluation harness.

WHAT IT CHECKS: that the environment can actually be played, that the harness can score a
policy, and how fast the whole loop runs. It is the cheapest possible end-to-end test of
everything built so far, and it is the Phase-B gate - if this does not run, nothing above
it can be trusted.

Run it:

    PYTHONPATH=src .venv/bin/python -m attempt2.smoke
    PYTHONPATH=src .venv/bin/python -m attempt2.smoke --games 200
"""

import argparse
import time

from attempt2 import config
from attempt2.baselines import maskedRandomPolicy
from attempt2.eval import evaluate, formatResultsTable
from attempt2.splits import heldOutSecrets


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the Attempt-2 environment")
    parser.add_argument("--games", type=int, default=50,
                        help="how many held-out secrets to play (default: 50)")
    arguments = parser.parse_args()

    secrets = heldOutSecrets()[:arguments.games]
    policy = maskedRandomPolicy(config.SEED)

    # WHY THE WARM-UP: the first call builds the precomputed feedback table, which costs
    # about a second and has nothing to do with how fast the game plays. Timing without
    # it would report the table build as if it were the environment's speed.
    evaluate(policy, secrets[:1], name="warm-up")

    start = time.time()
    result = evaluate(policy, secrets, name="masked random")
    elapsed = time.time() - start

    steps = sum(guesses for _, won, guesses in result.perGame)

    print(formatResultsTable([result]))
    print()
    print("games:            %d" % result.games)
    print("steps:            %d" % steps)
    print("elapsed:          %.2f s" % elapsed)
    print("steps/sec:        %.0f" % (steps / elapsed if elapsed else 0))
    print("distinct openers: %d" % len(result.firstGuessCounts))
    print()

    # The gate: a random policy that can never win would mean the environment, the mask
    # or the harness is broken, and every number measured later would be meaningless.
    if result.wins == 0:
        raise SystemExit(
            "FAILED: the masked random policy won none of %d games; the environment, the "
            "mask or the harness is broken" % result.games
        )

    print("OK: the environment can be played and scored.")


if __name__ == "__main__":
    main()
