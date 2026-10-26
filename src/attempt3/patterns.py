"""The pattern distribution a guess produces, and the information it carries.

WHY THIS MODULE EXISTS: the strategy needs, for every candidate guess, the distribution
of the 243 colour patterns it would produce over the still-possible answers. Computing
that pair by pair with `wordle.feedback.feedback()` is far too slow: one turn scores up
to 2,315 guesses against up to 2,315 candidates. The precomputed feedback table turns the
whole turn into one array gather and one `bincount`, which is why the information-theoretic
screen costs milliseconds instead of a simulation.

WHAT A HISTOGRAM MEANS: `counts[i, code]` is the number of candidates that would answer
guess `i` with colour pattern `code`. The code is `wordle.feedbackTable.patternCode`, so
it is the same integer the table and the action mask use. Each row sums to the number of
candidates.

THE QUANTITIES:

    n_p                = counts[guess, p]
    H(guess)           = -sum_p (n_p / m) log2(n_p / m)          (information carried)
    E[remaining]       = sum_p n_p^2 / m                        (expected size of C')
    Jensen gap         = H - log2(m / E[remaining])             (>= 0)

The Jensen gap is derived in `jensenGapFromCounts`. It is **not** the signed difference
`log2(E[remaining]) - H` that an earlier note quoted; that difference can be negative and
is not a Jensen gap.

WHY THE FUNCTIONS TAKE INDICES, NOT WORDS: every caller already holds answer indices, and
the vocabulary is built once. An index check is cheaper than a string lookup, and it makes
the vectorised gather a single `np.ix_`. Duplicate indices are rejected: a duplicated
candidate would be counted twice and silently change the distribution.

WHY EMPTY CANDIDATES ARE ALLOWED IN THE HISTOGRAM BUT NOT IN THE STATISTICS: a histogram
over an empty set is well defined (all zeros), and a caller may want to check a branch that
turned out to be impossible. Entropy and expected remaining size divide by the candidate
count, so they are undefined there and raise a loud `ValueError` instead of returning a
`nan` that would poison a ranking.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from wordle.feedbackTable import PATTERN_COUNT, feedbackTable
from wordle.vocabulary import loadAnswers


@dataclass(frozen=True)
class GuessStatistics:
    """Every A1 quantity for a set of guesses over one feasible set.

    WHY A BUNDLE: the histogram is the expensive part, and entropy, expected remaining
    size, the Jensen gap and the branch diagnostics are all derived from it. Returning
    them together means a caller cannot compute the histogram twice and get inconsistent
    numbers, and a picker can sort by any of the fields without redoing the gather.
    """

    guesses: np.ndarray
    candidateCount: int
    counts: np.ndarray
    entropies: np.ndarray
    expectedRemaining: np.ndarray
    jensenGap: np.ndarray
    nonEmptyBranches: np.ndarray
    largestBranch: np.ndarray


def patternCounts(guesses, candidates):
    """A `(len(guesses), 243)` histogram of feedback patterns over `candidates`.

    `guesses` and `candidates` are one-dimensional sequences of indices into
    `wordle.vocabulary.loadAnswers()`. Each row counts how many candidates produce each
    of the 243 colour codes against that guess; every row therefore sums to
    `len(candidates)`.

    Duplicates are rejected because they would double-count a candidate. The inputs are
    never mutated, and the order of the candidates does not affect the result.
    """
    guessIndices = _asIndices(guesses, "guess")
    candidateIndices = _asIndices(candidates, "candidate")
    return _patternCounts(guessIndices, candidateIndices)


def entropyFromCounts(counts):
    """Shannon entropy in bits for each row of a pattern histogram.

    `counts` must be a `(guesses, 243)` array of non-negative integers, each row with a
    positive total. The result is the expected information the guess carries: a high
    value means the colours are hard to predict, which means the guess separates the
    candidates into many similarly sized groups.
    """
    validated, totals = _validatedCounts(counts)
    return _entropyFromCounts(validated, totals)


def expectedRemainingFromCounts(counts):
    """The expected number of candidates left after each guess.

    This is `sum_p n_p^2 / m`: the expectation of the branch *size*. It is not the size
    of the expected logarithm, and the two differ by the Jensen gap below.
    """
    validated, totals = _validatedCounts(counts)
    return _expectedRemainingFromCounts(validated, totals)


def jensenGapFromCounts(counts):
    """The non-negative Jensen gap between entropy and the log expected remaining count.

    DERIVATION. With `m` candidates and branch sizes `n_p`,

        H = log2(m) - E[log2(n_p)].

    `log2` is concave, so `E[log2(n_p)] <= log2(E[n_p])`, and therefore

        H >= log2(m / E[n_p]).

    The gap returned here is

        H - log2(m / E[n_p]) = log2(E[n_p]) - E[log2(n_p)] >= 0.

    It is zero when every non-empty branch has the same size, and grows as the branch
    sizes become more unequal.

    WHY NOT `log2(E[n_p]) - H`: that difference is signed. For two equal branches it is
    `-1`, so it cannot be a Jensen gap. The earlier note that quoted ~0.31 bits for
    `slate` was reporting that signed difference; the correct gap for `slate` is ~0.84.
    """
    validated, totals = _validatedCounts(counts)
    return _jensenGapFromCounts(validated, totals)


def guessStatistics(guesses, candidates):
    """Run one histogram and return every A1 statistic for each guess.

    The candidate set must be non-empty; entropy and expected remaining size are
    undefined over an empty set. An empty guess set is allowed and returns empty arrays.
    """
    guessIndices = _asIndices(guesses, "guess")
    candidateIndices = _asIndices(candidates, "candidate")

    if candidateIndices.size == 0:
        raise ValueError("candidates must contain at least one answer index")

    counts = _patternCounts(guessIndices, candidateIndices)
    totals = counts.sum(axis=1, dtype=np.int64)

    if np.any(totals != candidateIndices.size):
        raise RuntimeError(
            "internal error: a histogram row does not sum to the candidate count"
        )

    return GuessStatistics(
        guesses=guessIndices,
        candidateCount=int(candidateIndices.size),
        counts=counts,
        entropies=_entropyFromCounts(counts, totals),
        expectedRemaining=_expectedRemainingFromCounts(counts, totals),
        jensenGap=_jensenGapFromCounts(counts, totals),
        nonEmptyBranches=(counts > 0).sum(axis=1),
        largestBranch=counts.max(axis=1),
    )


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def _patternCounts(guessIndices, candidateIndices):
    """The vectorised histogram. Both inputs are already validated index arrays."""
    guessCount = guessIndices.size
    counts = np.zeros((guessCount, PATTERN_COUNT), dtype=np.int32)

    if guessCount == 0 or candidateIndices.size == 0:
        return counts

    # WHY THE OFFSET TRICK: `bincount` is the fast histogram primitive, but it is 1-D.
    # Adding `row * 243` to every code puts each guess's codes in its own range, so one
    # bincount over the flattened array produces the whole 2-D histogram at once.
    codes = feedbackTable()[np.ix_(guessIndices, candidateIndices)].astype(np.int32)
    rowOffsets = (np.arange(guessCount, dtype=np.int32) * PATTERN_COUNT)[:, None]
    flat = (codes + rowOffsets).ravel()

    counts[:] = np.bincount(
        flat, minlength=guessCount * PATTERN_COUNT
    ).reshape(guessCount, PATTERN_COUNT)
    return counts


# ---------------------------------------------------------------------------
# The statistics, over already-validated counts
# ---------------------------------------------------------------------------


def _entropyFromCounts(counts, totals):
    probabilities = counts.astype(np.float64) / totals[:, None]
    terms = np.zeros_like(probabilities)

    positive = probabilities > 0
    terms[positive] = -probabilities[positive] * np.log2(probabilities[positive])

    return terms.sum(axis=1)


def _expectedRemainingFromCounts(counts, totals):
    weights = counts.astype(np.float64)
    return (weights * weights).sum(axis=1) / totals


def _jensenGapFromCounts(counts, totals):
    entropies = _entropyFromCounts(counts, totals)
    expected = _expectedRemainingFromCounts(counts, totals)
    return entropies - np.log2(totals / expected)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@cache
def _answerCount():
    """How many answers the feedback table has columns for."""
    return len(loadAnswers())


def _asIndices(values, label):
    """Turn a one-dimensional sequence into a checked int64 index array.

    WHY THE CHECKS: a wrong index either raises deep inside NumPy with a message that
    does not mention which argument was wrong, or - worse - silently indexes the wrong
    answer. A duplicate candidate is the dangerous case: it would be counted twice and
    the distribution would no longer be a distribution.
    """
    if isinstance(values, np.ndarray):
        array = values
    else:
        try:
            array = np.asarray(list(values))
        except TypeError:
            raise ValueError(
                "%s indices must be a one-dimensional sequence, got %r" % (label, values)
            ) from None

    if array.ndim != 1:
        raise ValueError(
            "%s indices must be one-dimensional, got shape %r" % (label, array.shape)
        )

    if array.size == 0:
        return array.astype(np.int64, copy=False)

    if not np.issubdtype(array.dtype, np.integer) or array.dtype == bool:
        raise ValueError(
            "%s indices must be integers, got dtype %r" % (label, array.dtype)
        )

    array = array.astype(np.int64, copy=False)
    smallest = int(array.min())
    largest = int(array.max())
    answerCount = _answerCount()

    if smallest < 0 or largest >= answerCount:
        raise ValueError(
            "%s index out of range [0, %d): min %d, max %d"
            % (label, answerCount, smallest, largest)
        )

    if np.unique(array).size != array.size:
        raise ValueError("%s indices must be unique" % label)

    return array


def _validatedCounts(counts):
    """Check a counts array and return it with one total per row."""
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
