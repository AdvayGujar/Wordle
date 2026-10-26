"""B1: the pure-information ablation - maximise the pattern entropy every turn.

WHAT THIS IS: the classical advice, implemented literally. At every turn, score all playable
answers by the entropy of the colour pattern they would produce over the feasible set, and
play the one with the highest entropy. This is §2.2's criterion with nothing added.

WHAT IT IS FOR: it is the control for C1. `attempt3.md` §3.7 measured a prototype of the
expected-score criterion against "pure entropy + probes" at 3.4625 against 3.655, so this
module exists to reproduce that ablation with committed code. If the win-chance term of
§2.7 is worth nothing, this picker will match C1 and the term should be deleted.

WHAT IT IGNORES, BY CONSTRUCTION:

- **Winning now.** A candidate that wins immediately and a probe that splits the set equally
  can have the same entropy, so entropy alone prefers neither.
- **The six-guess cap.** Entropy is just as valuable on the sixth turn as on the first,
  which is false: a guess that leaves two candidates is worthless when no guess remains.

WHY THE SINGLETON RULE IS EXPLICIT: with one candidate left every guess has entropy 0, so
the entropy ranking is a tie and the tie-break decides. The tie-break already prefers a
candidate, which would pick the answer - but only by accident of the ordering. The rule is
written down so a later change to the tie-break cannot quietly break the endgame.

WHY PROBES ARE IN THE POOL: a word outside the feasible set can be more informative than any
candidate in it, and forbidding it would silently turn this into a different ablation.
"""

from attempt3.patterns import guessStatistics
from attempt3.pickers import bestByScore, stateFor


def bestGuess(candidates, guesses):
    """The highest-entropy guess, as an answer index.

    `candidates` is the feasible set and `guesses` is the playable pool; both are index
    arrays, as everywhere in Attempt 3. Ties follow `pickers.bestByScore`: a candidate
    first, then the smaller expected remaining set, then the lower index.
    """
    statistics = guessStatistics(guesses, candidates)

    return bestByScore(
        -statistics.entropies,
        guesses,
        candidates,
        statistics.expectedRemaining,
    )


def pickGuess(history):
    """Play the best guess for this history, by the entropy criterion."""
    state = stateFor(history)

    if state.isSolved:
        return int(state.candidates[0])

    return bestGuess(state.candidates, state.guesses)
