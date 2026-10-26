"""C1: the expected-score policy - the strategy Attempt 3 is judged on.

WHAT THIS IS: §2.7's criterion, applied greedily every turn. Score every playable answer by

    score(g) = p(g) + (1 - p(g)) (1 + f(H0 - H(g)))

and play the smallest. `objective.py` owns the numbers and `pickers.py` owns the tie-break;
this module only decides which words are eligible and adds the adapter the evaluation
harness calls.

WHY THIS IS NOT JUST ENTROPY: the `p(g)` term is the chance the guess ends the game now. It
is 1/|C| for a candidate and 0 for a probe, so the criterion will accept a slightly less
informative candidate over a slightly more informative probe - and the reverse when the
information gap is large. B1 has no such term, which is the whole difference Gate C
measures.

WHY THE PROBES SWITCH EXISTS: §3.7 and §4.8 compare this policy against the *no-probe*
variant, which restricts the guess pool to the feasible set. With probes off, `p` is the
same 1/|C| for every guess, the win-chance term is a constant, and the criterion collapses
to "maximise the entropy" - so the ablation also tests whether probes are worth their risk.

WHY IT REMEMBERS DECISIONS: a policy is a pure function of the history, and an evaluation
plays several hundred games that share long prefixes - every game that opens with `raise`
and sees the same first pattern asks the same question again. The cache is keyed by the
history and the probes flag, and stores one integer, so it saves whole turns without keeping
any state that could leak between games.

WHY THE ADAPTER IGNORES THE MASK: `attempt2.eval` offers the observation, the mask and the
history. This policy needs only the history, and computing the feasible set from it is
cheaper than reconciling it with a mask built for a different guess pool. Ignoring the mask
is also what lets the policy play a probe at all - the mask marks probes as False, and
`evaluate` counts them as off-candidate moves rather than refusing them.
"""

from functools import cache

from attempt3.objective import scoreGuesses
from attempt3.pickers import bestByScore, stateFor


def bestGuess(candidates, guesses):
    """The smallest-score guess over one feasible set, as an answer index.

    `candidates` is the feasible set, `guesses` the words that may be played. Ties follow
    `pickers.bestByScore`, which is §2.7's tie-break: a candidate first, then the smaller
    expected remaining set, then the lower index.
    """
    scored = scoreGuesses(guesses, candidates)

    return bestByScore(
        scored.scores,
        guesses,
        candidates,
        scored.expectedRemaining,
    )


def pickGuess(history, allowProbes=True):
    """Play the best expected-score guess for this history.

    With `allowProbes=False` the pool is the feasible set itself, which is the §4.8
    no-probe ablation. Nothing else changes.
    """
    return _pick(tuple(history), bool(allowProbes))


def policy(allowProbes=True):
    """A policy callable for `attempt2.eval`, closing over the probes flag.

    The returned callable takes a `PolicyState` and reads only its history, so the adapter
    cannot see the secret even by accident.
    """

    def act(state):
        return pickGuess(state.history, allowProbes=allowProbes)

    act.__name__ = "adaptivePicker" if allowProbes else "adaptivePickerNoProbes"
    return act


@cache
def _pick(history, allowProbes):
    state = stateFor(history)

    if state.isSolved:
        return int(state.candidates[0])

    guesses = state.guesses if allowProbes else state.candidates

    return bestGuess(state.candidates, guesses)


def main():
    """One call, for a smoke check: what does the criterion choose at the empty history?"""
    from wordle.vocabulary import answersVocab, loadAnswers

    words = answersVocab()

    for allowProbes in (True, False):
        choice = pickGuess([], allowProbes=allowProbes)
        label = "with probes" if allowProbes else "no probes  "
        print("%s -> %s" % (label, words.wordAt(choice)))

    answerCount = len(loadAnswers())
    scored = scoreGuesses(range(answerCount), range(answerCount))
    ranked = sorted(range(scored.scores.size), key=lambda index: scored.scores[index])[:5]

    print("\ntop five at the empty history:")
    for index in ranked:
        print(
            "  %s  score %.4f  H %.4f bits  E[|C'|] %.1f"
            % (
                words.wordAt(index),
                scored.scores[index],
                scored.entropies[index],
                scored.expectedRemaining[index],
            )
        )


if __name__ == "__main__":
    main()
