"""B2: the expected-survivors ablation - minimise E[|C'|] every turn.

WHAT THIS IS: instead of maximising the information a guess carries, minimise the expected
number of candidates it leaves behind,

    E[|C'| | g, C] = (1/|C|) sum_p n_p^2.

The two criteria are linked by an identity (§2.2): maximising the pattern entropy is the
same as minimising the expected *logarithm* of the survivors. They are not the same ranking,
because the logarithm is concave, and the gap between them is the Jensen gap computed in
`patterns.py`. This picker is the other half of that comparison.

WHY IT IS WORTH A ROW: `raise` has the highest entropy and the smallest E[|C'|] (§3.2), so
for the opening word the two criteria agree. They diverge on later turns, and the size of
that divergence is exactly what the Gate B table measures.

WHAT IT STILL IGNORES: the win chance. Like B1 it cannot tell "guess the answer" from "probe
that splits the set", and it values the sixth turn the same as the first. The singleton rule
below is written out for the same reason as in B1: with one candidate every guess leaves
one, so the ranking is all ties and the tie-break does the work.

WHY PROBES ARE IN THE POOL: same reason as B1. `E[|C'|]` is defined for a guess outside C,
and excluding those words would make this the no-probe ablation instead.
"""

from attempt3.patterns import guessStatistics
from attempt3.pickers import bestByScore, stateFor


def bestGuess(candidates, guesses):
    """The guess with the smallest expected remaining candidate set, as an answer index.

    Ties follow `pickers.bestByScore`: a candidate first, then the smaller expected
    remaining set, then the lower index.
    """
    statistics = guessStatistics(guesses, candidates)

    return bestByScore(
        statistics.expectedRemaining,
        guesses,
        candidates,
        statistics.expectedRemaining,
    )


def pickGuess(history):
    """Play the best guess for this history, by the expected-survivors criterion."""
    state = stateFor(history)

    if state.isSolved:
        return int(state.candidates[0])

    return bestGuess(state.candidates, state.guesses)
