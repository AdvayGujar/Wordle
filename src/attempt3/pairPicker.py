"""B3: the two-word ablation - choose a fixed pair jointly, then play it.

WHAT THIS IS: §3.4 asked whether two guesses can identify the word. The budget says yes
(2 x 7.92 = 15.85 bits against 11.18), and the measurement said no (37% unique). This picker
is the cheapest way to turn that question into a policy: choose two words **together** so
that the pair's joint pattern distribution carries as much information as possible, play
them on the first two turns, then continue with B2.

WHY JOINT ENTROPY IS THE INFORMATION OF THE PAIR: the colours are deterministic given the
secret, so I((G1,G2);S) = H(G1,G2) - H(G1,G2 | S) = H(G1,G2). The pair's information *is*
the entropy of its joint 243 x 243 pattern distribution over the feasible set. Its symmetry
is why the criterion cannot choose an order; the order is set by the individual entropies so
that the more informative word goes first.

WHY A SHORTLIST: evaluating every ordered pair is 2,315 x 2,314 joint histograms - minutes
of work for an ablation whose answer is one fixed pair of words. The first word is
restricted to the `SHORTLIST` highest-entropy guesses, and the second is drawn from the same
list. That is a screen, not a proof of global optimality; the measured pair is reported next
to its joint entropy so a reader can judge it.

WHY B2 AFTER THE PAIR: the ablation has to differ from B2 in exactly one thing - the opening
pair - or the row confounds two changes. Falling back to the same greedy rule keeps the
comparison clean.

WHY THE PAIR IS PLAYED BY TURN, NOT BY MATCHING THE HISTORY: a fixed pair is a fixed plan.
If the first word wins, the game is over and the second is never played; otherwise the
second is played on turn two whatever the branch was. Branch-aware choice is `lookahead`
(C2), a different module and a different row.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from attempt3 import remainingPicker
from attempt3.patterns import guessStatistics
from attempt3.pickers import stateFor
from wordle.feedbackTable import PATTERN_COUNT, feedbackTable
from wordle.vocabulary import loadAnswers

JOINT_COUNT = PATTERN_COUNT * PATTERN_COUNT

# WHY 200: wide enough to contain every plausible high-information word, small enough that
# the joint screen is 19,900 histograms instead of 5.4 million.
SHORTLIST = 200


@dataclass(frozen=True)
class PairChoice:
    """A fixed opening pair, with the numbers that justify it.

    `jointEntropy` is the pair's information in bits. `conditionalInformation` is what the
    second word adds after the first, H(G1,G2) - H(G1); it is the quantity that decides
    whether the second word is doing real work or repeating the first.
    """

    first: int
    second: int
    jointEntropy: float
    firstEntropy: float
    secondEntropy: float
    conditionalInformation: float


def jointEntropy(first, second, candidates):
    """The entropy in bits of the pair's joint pattern distribution over `candidates`."""
    counts = _jointCounts(first, second, candidates)
    probabilities = counts / counts.sum()

    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = -probabilities[positive] * np.log2(probabilities[positive])

    return float(terms.sum())


def conditionalEntropy(second, first, candidates):
    """H(G_second | G_first): what the second word adds once the first has been played."""
    return jointEntropy(first, second, candidates) - _singleEntropy(first, candidates)


def bestPair(candidates, guesses, shortlist=SHORTLIST):
    """Search the shortlist for the pair with the highest joint entropy.

    The search is a screen over the highest-entropy guesses, not an exhaustive one. Ties go
    to the earlier pair in (entropy, index) order, so the result is deterministic; the pair
    is then oriented with the higher-entropy word first.
    """
    statistics = guessStatistics(guesses, candidates)
    order = np.argsort(-statistics.entropies, kind="stable")
    shortlisted = np.asarray(guesses)[order][:shortlist]

    if shortlisted.size < 2:
        raise ValueError("the guess pool must contain at least two words to form a pair")

    best = None
    for position, first in enumerate(shortlisted):
        for second in shortlisted[position + 1:]:
            value = jointEntropy(first, second, candidates)
            if best is None or value > best[0]:
                best = (value, int(first), int(second))

    value, first, second = best
    firstEntropy = _singleEntropy(first, candidates)
    secondEntropy = _singleEntropy(second, candidates)

    if secondEntropy > firstEntropy:
        first, second = second, first
        firstEntropy, secondEntropy = secondEntropy, firstEntropy

    return PairChoice(
        first=first,
        second=second,
        jointEntropy=value,
        firstEntropy=firstEntropy,
        secondEntropy=secondEntropy,
        conditionalInformation=value - firstEntropy,
    )


@cache
def rootPair():
    """The fixed opening pair, chosen at the empty history and cached.

    Every game in an evaluation starts from the same state, so the search runs once per
    process. A fixed pair is the definition of the ablation; recomputing it per game would
    make it something else.
    """
    answers = np.arange(len(loadAnswers()), dtype=np.int64)
    return bestPair(answers, answers)


def pickGuess(history):
    """Play the pair on the first two turns, then hand over to B2."""
    state = stateFor(history)

    if state.isSolved:
        return int(state.candidates[0])

    if len(history) < 2:
        pair = rootPair()
        return pair.first if len(history) == 0 else pair.second

    return remainingPicker.pickGuess(history)


def _jointCounts(first, second, candidates):
    """The 243 x 243 joint histogram of the two guesses' patterns over `candidates`."""
    candidateIndices = np.asarray(candidates, dtype=np.int64)

    if candidateIndices.size == 0:
        raise ValueError("candidates must contain at least one answer index")

    if int(first) == int(second):
        raise ValueError("a pair must be two different words, got the same index twice")

    indices = np.asarray([first, second], dtype=np.int64)
    codes = feedbackTable()[np.ix_(indices, candidateIndices)]
    joint = codes[0].astype(np.int64) * PATTERN_COUNT + codes[1]

    return np.bincount(joint, minlength=JOINT_COUNT)


def _singleEntropy(index, candidates):
    """One guess's own pattern entropy over `candidates`, for the pair's bookkeeping."""
    statistics = guessStatistics([index], candidates)
    return float(statistics.entropies[0])
