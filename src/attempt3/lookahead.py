"""C2: one ply of lookahead - value a first word by the best second word in each branch.

WHAT THIS IS: §2.8's local refinement. The greedy criteria in B1, B2 and C1 all judge a
guess by what it does *now*. This module judges it by what happens next: play the first
word, look at each colour branch it can produce, and score the branch by the best second
word available *in that branch*. The first word's value is the branch-weighted average of
those costs.

    twoPly(g) = 1 + sum over non-green branches p of (n_p / m) * min_w score(w | C_p)

The leading 1 is the guess itself. The all-green branch contributes nothing further, because
a green branch means the game is over - that is the same correction `objective.py` applies
to the rejected model, and getting it wrong here would make every first word look one guess
worse than it is.

WHY THIS IS NOT A FULL TREE SOLVE: it looks one ply ahead and then assumes the branch is
finished by a single greedy choice. A real solve would recurse to depth six over every
subset, which is a large dynamic program (§2.8 says why this attempt does not do it). This
is a local improvement on C1 and is measured as such.

WHY THE TWO SHORTLISTS: the exact version needs, for one first word, the best second word
for every branch - and that search is itself over all 2,315 answers. Exhaustively that is
every first word x every branch x every second word, which is minutes per turn. Both screens
use the one-ply expected score at the node: the first words are the best few by that score,
and the second words are the same best few. A word that is terrible on its own turn is not
rescued by the branch it lands in, so the screen costs little in practice and is stated here
rather than hidden.

WHY IT RE-PLANS EVERY TURN: after the first word and its colours, the state is the branch,
and the lookahead runs again there. That is why the policy is self-consistent: the second
word played is exactly the second word the first turn's value assumed.
"""

from functools import cache

import numpy as np

from attempt3 import config
from attempt3.objective import GREEN_CODE, scoreGuesses
from attempt3.pickers import bestByScore, stateFor
from wordle.feedbackTable import PATTERN_COUNT, feedbackTable
from wordle.vocabulary import loadAnswers


def twoPlyScore(first, candidates, seconds):
    """The expected total guesses of playing `first`, then the best of `seconds` per branch.

    `candidates` is the feasible set and `seconds` the words allowed as the follow-up. The
    result is in the same units as `objective.scoreGuesses`: total guesses from this node,
    including the guess of `first` itself.
    """
    candidateArray = np.asarray(candidates, dtype=np.int64)
    secondArray = np.asarray(seconds, dtype=np.int64)

    if candidateArray.size == 0:
        raise ValueError("candidates must contain at least one answer index")

    if secondArray.size == 0:
        raise ValueError("seconds must contain at least one answer index")

    firstIndex = int(first)
    if not 0 <= firstIndex < len(loadAnswers()):
        raise ValueError("first must be an answer index, got %r" % (first,))

    codes = feedbackTable()[firstIndex, candidateArray]
    counts = np.bincount(codes, minlength=PATTERN_COUNT)

    total = 0.0
    for code in np.flatnonzero(counts):
        if code == GREEN_CODE:
            # The guess was the answer: the game ended, so this branch needs no follow-up.
            continue

        branch = candidateArray[codes == code]
        branchScores = scoreGuesses(secondArray, branch).scores
        total += float(counts[code]) * float(branchScores.min())

    return 1.0 + total / candidateArray.size


def bestGuess(candidates, guesses, firstCount=None, secondCount=None):
    """The first word with the smallest two-ply score, as an answer index.

    The screens are `firstCount` first words and `secondCount` follow-ups, both taken from
    the one-ply expected-score ranking. Ties use §2.7's order, via `pickers.bestByScore`.
    """
    firstCount = config.LOOKAHEAD_FIRST_SHORTLIST if firstCount is None else firstCount
    secondCount = config.LOOKAHEAD_SECOND_SHORTLIST if secondCount is None else secondCount

    if firstCount < 1 or secondCount < 1:
        raise ValueError("both shortlists must contain at least one word")

    scored = scoreGuesses(guesses, candidates)
    order = np.argsort(scored.scores, kind="stable")
    guessesArray = np.asarray(guesses)

    firsts = guessesArray[order][:firstCount]
    seconds = guessesArray[order][:secondCount]

    if firsts.size == 0:
        raise ValueError("guesses must contain at least one answer index")

    values = np.array([twoPlyScore(first, candidates, seconds) for first in firsts])

    return bestByScore(
        values,
        firsts,
        candidates,
        scored.expectedRemaining[order][:firstCount],
    )


def pickGuess(history):
    """Play the best two-ply guess for this history."""
    return _pick(tuple(history))


def policy():
    """A policy callable for `attempt2.eval`; it reads only the history."""

    def act(state):
        return pickGuess(state.history)

    act.__name__ = "lookahead"
    return act


@cache
def _pick(history):
    state = stateFor(history)

    if state.isSolved:
        return int(state.candidates[0])

    return bestGuess(state.candidates, state.guesses)


def main():
    """One call, for a smoke check: the two-ply choice against the one-ply choice."""
    from time import perf_counter

    from wordle.vocabulary import answersVocab

    from attempt3.adaptivePicker import pickGuess as greedyPick

    words = answersVocab()

    greedy = greedyPick([])

    started = perf_counter()
    twoPly = pickGuess([])
    elapsed = perf_counter() - started

    print("one-ply  C1 -> %s" % words.wordAt(greedy))
    print("two-ply  C2 -> %s  (%.2fs for the search)" % (words.wordAt(twoPly), elapsed))


if __name__ == "__main__":
    main()
