"""The decision criterion: expected score, and the rejected expected-guesses model.

WHY THIS MODULE EXISTS: `patterns.py` measures what a guess is worth in bits. Bits are not
the game. A guess can be slightly less informative and still be better, because it might
win immediately, and because the game is capped at six guesses. This module turns the
histogram into the single number the pickers minimise, and it keeps the criterion that was
measured and *rejected* next to the one that is used, so the write-up can point at tested
code instead of a claim.

THE CRITERION (derived in `attempt3.md` §2.7). For a feasible set C with m = |C|, a guess g
with pattern entropy H(g), and H0 = log2(m) bits of current uncertainty:

    p(g)    = 1/m if g is in C, else 0        (chance it wins right now)
    score(g) = p(g) + (1 - p(g)) * (1 + f(H0 - H(g)))
    f(x)     = 2^-x + 2(1 - 2^-x) + 1.5x/11.5

and the guess with the **smallest** score is played. The two terms have different jobs:

- `p(g)` says "this guess can end the game now". Without it the criterion cannot tell a
  candidate from an equally informative probe, which is exactly the endgame failure the
  rejected model below demonstrates.
- `f(H0 - H(g))` estimates the guesses still needed for the uncertainty the guess leaves
  behind. f(0) = 1 (the answer is forced after one more guess), and f is increasing, so
  leaving less uncertainty is always better, all else equal.

WHY f IS SHAPED THAT WAY: f(0) = 1 exactly - with no uncertainty left, one guess names the
answer. The first two terms are 2 - 2^-x, which rises from 1 towards 2 as the set widens:
that is the cost of the last one or two guesses. The linear term carries the growth from
there, calibrated so that a full board's worth of uncertainty (11.5 bits, log2 2315) is
about 3.5 further guesses. `SCORE_ANCHOR_BITS` and `SCORE_SLOPE` are fixed in `config.py`
before any evaluation and are not tuned on the held-out secrets.

WHY PROBES MATTER: if guesses are restricted to C, then p = 1/m for every guess, p is a
constant, and "minimise the score" becomes "maximise H(g)" - plain entropy. Allowing guesses
outside C is what gives the criterion something to trade off, and it is worth 0.08 guesses
in the measurement of §3.7.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: it does not pick. It returns a score per guess.
Ranking, the deterministic tie-break and the "singleton means play the answer" rule belong
to the pickers, so the criterion can be tested on its own and reused by every ablation.

THE REJECTED MODEL: `attempt3.md` §2.5 derives E[guesses] = 1 + sum_p (n_p/m) c(n_p) under
a uniform-random continuation, with c(m) = E[min(T, 5)]. It is concave, which is the
cleanest explanation of why entropy and expected guesses rank openers differently. Used as
a greedy rule it loses every game, because once the feasible set is a singleton every guess
scores the same and it keeps probing instead of winning (§2.5.1). It stays here, tested, as
the documented negative result.
"""

from dataclasses import dataclass

import numpy as np

from attempt3.config import MAX_GUESSES, SCORE_ANCHOR_BITS, SCORE_SLOPE
from attempt3.patterns import guessStatistics
from wordle.feedbackTable import PATTERN_COUNT, patternCode

# The all-green pattern is the one branch that needs no further guess: the guess WAS the
# answer. `patternCode` is used rather than the literal 242 so this stays correct if the
# colour encoding ever changes.
GREEN_CODE = patternCode((2, 2, 2, 2, 2))

# c(m) caps the *further* guesses, not the total: the first guess is already spent.
_FURTHER_GUESS_CAP = MAX_GUESSES - 1

# Entropies are floats, and H(g) <= log2(m) only up to rounding. A tiny overshoot is
# arithmetic noise, not a caller error; anything larger is a wrong H0 or a wrong histogram.
_INFORMATION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GuessScores:
    """Every score ingredient for a set of guesses over one feasible set.

    WHY THE BUNDLE: the score alone is not enough to explain or debug a choice. Storing the
    entropy, the information left behind, the win chance and the expected remaining count
    alongside it means the write-up and the tie-break can read the same numbers the picker
    used, instead of recomputing them and possibly disagreeing.
    """

    guesses: np.ndarray
    candidateCount: int
    scores: np.ndarray
    entropies: np.ndarray
    informationLeft: np.ndarray
    winChance: np.ndarray
    expectedRemaining: np.ndarray
    isCandidate: np.ndarray


def remainingGuessEstimate(informationLeft):
    """f(x): the further-guess estimate for `x` bits of uncertainty left.

    f(x) = 2^-x + 2(1 - 2^-x) + SCORE_SLOPE * x / SCORE_ANCHOR_BITS.

    f(0) = 1 and f is strictly increasing. It is an estimate of how many more guesses a
    still-uncertain feasible set implies, in the same units as the score, not a probability
    and not a bound.
    """
    bits = _asNonNegative(informationLeft, "informationLeft")
    twoToMinusX = np.exp2(-bits)
    return twoToMinusX + 2.0 * (1.0 - twoToMinusX) + SCORE_SLOPE * bits / SCORE_ANCHOR_BITS


def entropyToExpectedScore(entropies, currentEntropy, winChance):
    """The expected-score criterion, from entropies and win chances.

    `entropies` is H(g) per guess, `currentEntropy` is H0 = log2(m), and `winChance` is
    p(g) - 1/m for a candidate, 0 for a probe. All three are validated: an entropy above
    H0 would mean more information than the feasible set contains, and a win chance outside
    [0, 1] is not a probability.
    """
    entropyValues = _asNonNegative(entropies, "entropies")
    current = _asScalarNonNegative(currentEntropy, "currentEntropy")
    chances = _asProbabilities(winChance, "winChance", entropyValues.shape)

    informationLeft = current - entropyValues

    if np.any(informationLeft < -_INFORMATION_TOLERANCE):
        overshoot = float(np.max(-informationLeft))
        raise ValueError(
            "a guess's entropy exceeds the feasible set's entropy by %.6g bits; "
            "entropies must be <= log2(candidateCount)" % overshoot
        )

    informationLeft = np.maximum(informationLeft, 0.0)
    costIfNotNow = 1.0 + remainingGuessEstimate(informationLeft)

    return chances + (1.0 - chances) * costIfNotNow


def scoreGuesses(guesses, candidates):
    """Score every guess against one feasible set with the §2.7 criterion.

    `guesses` and `candidates` are sequences of answer indices, as in `patterns.py`.
    A guess is a *candidate* exactly when the all-green branch is non-empty, which is read
    off the histogram: a non-candidate can never produce all greens, so the check needs no
    second index lookup and cannot disagree with the counts.

    The result is not sorted. The smallest score wins; ties are the pickers' problem.
    """
    statistics = guessStatistics(guesses, candidates)
    candidateCount = statistics.candidateCount
    currentEntropy = float(np.log2(candidateCount))

    # A green count of exactly one means the guess is one of the candidates: against
    # itself it is all greens, and no other candidate can be all greens for that guess.
    isCandidate = statistics.counts[:, GREEN_CODE] > 0
    winChance = isCandidate.astype(np.float64) / candidateCount

    informationLeft = np.maximum(currentEntropy - statistics.entropies, 0.0)

    return GuessScores(
        guesses=statistics.guesses,
        candidateCount=candidateCount,
        scores=entropyToExpectedScore(
            statistics.entropies, currentEntropy, winChance
        ),
        entropies=statistics.entropies,
        informationLeft=informationLeft,
        winChance=winChance,
        expectedRemaining=statistics.expectedRemaining,
        isCandidate=isCandidate,
    )


# ---------------------------------------------------------------------------
# The rejected model: expected guesses under a uniform continuation (§2.5)
# ---------------------------------------------------------------------------


def cappedGuesses(size):
    """c(m) = E[min(T, k)] for T uniform on 1..m, capped at k = MAX_GUESSES - 1.

    Closed form:

        c(m) = (m + 1) / 2                          for m < k
        c(m) = k - k(k - 1) / (2m)                  for m >= k

    WHY IT IS CONCAVE AND INCREASING FOR m >= k: the cap makes an extra candidate in an
    already-huge branch nearly free - those games are lost either way. That is the whole
    reason this objective prefers one huge branch to a flat spread, and therefore ranks
    openers differently from entropy.
    """
    sizes = _asSizes(size)

    small = (sizes + 1.0) / 2.0
    large = _FURTHER_GUESS_CAP - (
        _FURTHER_GUESS_CAP * (_FURTHER_GUESS_CAP - 1)
    ) / (2.0 * sizes)

    return np.where(sizes < _FURTHER_GUESS_CAP, small, large)


def cappedGuessesBySummation(size):
    """c(m) evaluated from its definition, as the exact cross-check of the closed form.

    For a non-negative integer T, E[min(T, k)] = sum over t = 1..k of P(T >= t), and
    P(T >= t) = max(0, m - t + 1) / m for uniform T on 1..m. This is a different route to
    the same number, so it catches an algebra slip in `cappedGuesses`.
    """
    sizes = _asSizes(size)
    totals = np.zeros_like(sizes, dtype=np.float64)

    for step in range(1, _FURTHER_GUESS_CAP + 1):
        totals = totals + np.maximum(sizes - step + 1.0, 0.0) / sizes

    return totals


def expectedGuessesFromCounts(counts):
    """1 + sum_p (n_p/m) c(n_p), with the all-green branch costing zero further guesses.

    This is the §2.5 functional. The all-green branch is special: if the guess is a
    candidate, that branch is exactly the guess and the game has already ended, so it needs
    no further guess. The naive formula charges c(1) = 1 for it and therefore overcounts by
    1/m - small at the root, but wrong in general, and it is the difference between a model
    that can win and one that cannot.
    """
    validated, totals = _checkedCounts(counts)
    sizes = validated.astype(np.float64)

    costs = np.zeros_like(sizes)
    positive = sizes > 0
    costs[positive] = cappedGuesses(sizes[positive])

    expected = 1.0 + (sizes / totals[:, None] * costs).sum(axis=1)

    # counts[:, GREEN_CODE] is 1 for a candidate and 0 for a probe, so this subtracts
    # c(1) * 1/m exactly when the guess can win now.
    greens = sizes[:, GREEN_CODE]
    return expected - greens / totals


def expectedGuesses(guesses, candidates):
    """The §2.5 functional for indexed guesses and candidates, as a convenience wrapper."""
    return expectedGuessesFromCounts(guessStatistics(guesses, candidates).counts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _asNonNegative(values, label):
    """A float array of finite, non-negative numbers, one-dimensional or scalar."""
    array = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError("%s must be finite, got %r" % (label, values))

    if np.any(array < 0):
        raise ValueError("%s must be non-negative, got %r" % (label, values))

    return array


def _asScalarNonNegative(value, label):
    """A single finite, non-negative float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number, got %r" % (label, value)) from None

    if not np.isfinite(number) or number < 0:
        raise ValueError("%s must be finite and non-negative, got %r" % (label, value))

    return number


def _asProbabilities(values, label, shape):
    """A float array of probabilities in [0, 1], broadcastable to `shape`."""
    array = np.asarray(values, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError("%s must be finite, got %r" % (label, values))

    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("%s must lie in [0, 1], got %r" % (label, values))

    try:
        return np.broadcast_to(array, shape).astype(np.float64, copy=False)
    except ValueError:
        raise ValueError(
            "%s is not broadcastable to the entropies' shape %r" % (label, shape)
        ) from None


def _asSizes(size):
    """A float array of positive candidate counts."""
    array = np.asarray(size, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError("size must be finite, got %r" % (size,))

    if np.any(array < 1):
        raise ValueError("size must be at least 1, got %r" % (size,))

    return array


def _checkedCounts(counts):
    """Check a pattern histogram and return it with one total per row.

    The same contract as the `patterns` counts helpers. It is re-checked here rather than
    imported because a silently malformed histogram would rank words wrongly, and the check
    is a few comparisons against a gather that costs milliseconds.
    """
    array = np.asarray(counts)

    if array.ndim != 2 or array.shape[1] != PATTERN_COUNT:
        raise ValueError(
            "counts must have shape (guesses, %d), got %r"
            % (PATTERN_COUNT, array.shape)
        )

    if not np.issubdtype(array.dtype, np.integer) or array.dtype == bool:
        raise ValueError("counts must be integers, got dtype %r" % array.dtype)

    if array.size and array.min() < 0:
        raise ValueError("counts must be non-negative")

    totals = array.sum(axis=1, dtype=np.int64)

    if np.any(totals <= 0):
        raise ValueError("every counts row must describe at least one candidate")

    return array.astype(np.int64, copy=False), totals
