"""Gate B: measure the three Phase-B ablations on the validation slice.

WHY THIS MODULE EXISTS: `attempt3.md` §4.8 says the B pickers are measured on the
validation slice and their numbers recorded *before* C1 is tuned. That measurement needs a
harness, and the plan did not name one - `experiment.py` is Phase D. This module is that
harness and nothing more: it wires each picker to `attempt2.eval.evaluateValidation`, prints
the table, and writes `gateB-results.json`.

WHY THE VALIDATION SLICE AND NOT THE HELD-OUT 400: the held-out secrets may be read once,
for the final table (§4.1.6). Gate B exists to decide whether the criteria behave, so it
uses the 200 validation secrets that are part of the training half. `evaluateValidation`
enforces that by putting the environment in train mode, so this harness cannot accidentally
read the held-out set.

WHY IT IS SEPARATE FROM `experiment.py`: Phase D's experiment will reproduce these rows
alongside C1, C2 and the Attempt-1/2 policies, on all three secret sets. Keeping the Gate B
harness small means the numbers that decide Phase C are not entangled with the final report.

WHAT THE "OFF-CANDIDATE" COLUMN MEANS: the harness counts every action whose mask is False,
which includes a probe - a legal Wordle guess that is inconsistent with the evidence. That
is a deliberate move for these pickers, not an error, so the table calls it off-candidate
probes (§4.1.5) rather than illegal actions.
"""

import json
from pathlib import Path

from attempt2.eval import evaluateValidation, formatResultsTable
from attempt3 import entropyPicker, pairPicker, remainingPicker
from wordle.vocabulary import answersVocab

RESULT_FILE = Path(__file__).resolve().parent / "artifacts" / "gateB-results.json"

POLICIES = (
    ("B1 entropy", entropyPicker),
    ("B2 remaining", remainingPicker),
    ("B3 pair, then B2", pairPicker),
)


def _adapter(module):
    """Turn a picker's `pickGuess(history)` into the callable the harness wants.

    The picker is given only the history - not the observation, not the mask, and never the
    secret - so this adapter cannot leak. The mask the harness passes is ignored on purpose:
    the pickers compute their own feasible set from the same history, through the same rule.
    """

    def policy(state):
        return module.pickGuess(state.history)

    return policy


def run():
    """Evaluate every Phase-B picker on the validation secrets."""
    return [
        evaluateValidation(_adapter(module), name=name)
        for name, module in POLICIES
    ]


def main():
    words = answersVocab()
    pair = pairPicker.rootPair()

    print(
        "B3 fixed pair: %s then %s  (joint %.3f bits; %s carries %.3f, second adds %.3f)"
        % (
            words.wordAt(pair.first),
            words.wordAt(pair.second),
            pair.jointEntropy,
            words.wordAt(pair.first),
            pair.firstEntropy,
            pair.conditionalInformation,
        )
    )
    print()

    results = run()
    print(formatResultsTable(results))
    print()
    print("The Illegal column counts off-candidate probes: legal guesses the evidence has")
    print("already ruled out, which these pickers play on purpose (§4.1.5). Not an error")
    print("count, and not a repeated word.")

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
        }
        for result in results
    ]
    RESULT_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print("written: %s" % RESULT_FILE)


if __name__ == "__main__":
    main()
