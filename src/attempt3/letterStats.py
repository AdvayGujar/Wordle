"""The letter-frequency statistics behind the opening-word ranking (§2.6).

WHY THIS MODULE EXISTS: `patterns.py` computes what a guess is worth, but it says nothing
about *why* the best guesses are what they are. The measured answer is two effects:

- **Distinct letters.** A guess with a repeated letter cannot separate the candidates into
  as many outcomes, because the colours of the repeated letter are constrained by how many
  times it occurs in the secret. Measured mean pattern entropy over all 2,315 answers:

      distinct letters     answers     mean entropy
            2                 1          2.3986 bits
            3                57          3.4121 bits
            4               691          4.1250 bits
            5             1,566          4.7654 bits

- **Common letters.** Among the 1,566 five-distinct-letter answers, the correlation between
  a word's mean letter frequency and its pattern entropy is 0.923. Words made of common
  letters split the field well; words made of rare letters do not.

Together those explain the ranking without any search. They are a *diagnostic*, not a
policy: nothing in the pickers calls this module, and it is not a shortcut to a good guess
(this file's own numbers show the pure frequency proxy prefers repeats - see below).

WHY "FREQUENCY" MEANS OCCURRENCES, NOT WORD-PRESENCE: this decision is forced by the number
the plan quotes. Counting each letter once per answer that contains it gives a correlation
of **0.9365**; counting every occurrence (so the two `b`s of `abbey` both count) gives
**0.9231**, which is the 0.923 of §2.6 and §7. The occurrence definition is also the one
that matches the thing being explained: a repeated letter really does occupy two positions
and really is constrained twice by the colour rule.

WHY THE CORRELATION IS RESTRICTED TO FIVE-DISTINCT-LETTER WORDS: the frequency proxy on its
own is maximised by repeated common letters. The highest mean letter frequency in the whole
answer list belongs to `eerie`, and the top five are `eerie`, `eater`, `erase`, `rarer`,
`elate` - none of them good openers, because the repeats waste positions. Restricting to the
1,566 words with five distinct letters removes that distortion and isolates the effect the
section is about.
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from attempt3.patterns import guessStatistics
from wordle.vocabulary import loadAnswers

LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTER_COUNT = len(LETTERS)
WORD_LENGTH = 5

# The §2.6 correlation, quoted in tests so a definition change cannot pass silently.
MEASURED_FREQUENCY_CORRELATION = 0.9231


@dataclass(frozen=True)
class DistinctLetterBucket:
    """One row of the distinct-letter table: how many answers, and how much they split."""

    distinctLetters: int
    wordCount: int
    meanEntropy: float


@dataclass(frozen=True)
class WordSummary:
    """The letter-level description of one word, for sanity-checking a hand-written guess.

    `frequencyRank` is 1 for the answer with the highest mean letter frequency, with ties
    sharing the better rank. It is a shortcut for "does this word use common letters at
    all", not a guess ranking: `eerie` outranks `raise` on it.
    """

    word: str
    distinctLetters: int
    meanFrequency: float
    frequencyRank: int


def letterCounts(words=None):
    """How many times each letter occurs across `words` (default: all answers).

    Returns a fresh 26-element integer array in `LETTERS` order. A letter that occurs twice
    in a word is counted twice.
    """
    if words is None:
        return _answerCounts().copy()

    return _countLetters(_asWords(words, "words"))


def letterFrequencies(words=None):
    """The chance that a random letter position in a random word is each letter.

    This is `letterCounts(words) / (5 * len(words))`, a distribution over the 26 letters
    that sums to 1. Scale does not matter to the correlation of §2.6; the interpretation
    does, which is why the normalisation is total letter slots rather than word count.
    """
    counts = letterCounts(words).astype(np.float64)
    return counts / counts.sum()


def distinctLetterCount(word):
    """How many different letters a word uses (`abbey` -> 4, `eerie` -> 3)."""
    return len(set(_checkedWord(word)))


def meanLetterFrequency(word, frequencies=None):
    """The mean letter frequency of one word, over its five positions.

    A repeated letter contributes at each position, so `abbey` averages two `b`s. For a
    five-distinct-letter word this is the mean over its distinct letters, which is the
    quantity of §2.6.
    """
    checked = _checkedWord(word)

    if frequencies is None:
        values = letterFrequencies()
    else:
        values = _asFrequencies(frequencies)

    return float(np.mean([values[LETTERS.index(letter)] for letter in checked]))


def meanLetterFrequencies(words, frequencies=None):
    """`meanLetterFrequency` for each word in a sequence, as one float array."""
    checked = _asWords(words, "words")

    if frequencies is None:
        values = letterFrequencies()
    else:
        values = _asFrequencies(frequencies)

    return np.array(
        [np.mean([values[LETTERS.index(letter)] for letter in word]) for word in checked],
        dtype=np.float64,
    )


def distinctLetterBuckets():
    """The §2.6 table: answers grouped by how many distinct letters they use.

    The entropy of each word is its pattern entropy at the empty history, i.e. against all
    2,315 answers, so the bucket means are comparable across buckets. Buckets are returned
    in ascending order of distinct-letter count.
    """
    answers = loadAnswers()
    counts = np.array([len(set(word)) for word in answers])
    entropies = rootEntropies()

    buckets = []
    for distinct in sorted(set(counts.tolist())):
        mask = counts == distinct
        buckets.append(
            DistinctLetterBucket(
                distinctLetters=int(distinct),
                wordCount=int(mask.sum()),
                meanEntropy=float(entropies[mask].mean()),
            )
        )

    return tuple(buckets)


def letterFrequencyCorrelation():
    """The Pearson correlation between mean letter frequency and pattern entropy.

    Computed over the five-distinct-letter answers only, which is the definition §2.6 and
    §7 quote (0.923). Including repeated-letter words drags it to a different, less
    meaningful number, because the frequency proxy rewards the repeats themselves.
    """
    answers = loadAnswers()
    fiveDistinct = [index for index, word in enumerate(answers) if len(set(word)) == 5]

    entropies = rootEntropies()[fiveDistinct]
    frequencies = meanLetterFrequencies([answers[index] for index in fiveDistinct])

    return float(np.corrcoef(frequencies, entropies)[0, 1])


def wordSummary(word):
    """Distinct-letter count, mean frequency and frequency rank for one word.

    This is the sanity check §4.2 asks for: a hand-written guess can be described without
    running a game. A word that is not an answer is allowed; its rank is still computed
    against the answer list.
    """
    checked = _checkedWord(word)
    value = meanLetterFrequency(checked)
    means = _answerMeanFrequencies()

    return WordSummary(
        word=checked,
        distinctLetters=len(set(checked)),
        meanFrequency=value,
        frequencyRank=1 + int((means > value).sum()),
    )


def rootEntropies():
    """Every answer's pattern entropy against the full answer set.

    A1 computes this in one vectorised pass; caching matters because the bucket table and
    the correlation both need it and each costs a 2,315 x 2,315 gather to rebuild. The
    public function returns a copy, because a cached array handed straight out is a cached
    array a caller can overwrite.
    """
    return _rootEntropies().copy()


@cache
def _rootEntropies():
    indices = np.arange(len(loadAnswers()), dtype=np.int64)
    return guessStatistics(indices, indices).entropies


# ---------------------------------------------------------------------------
# Cached answer-list statistics
# ---------------------------------------------------------------------------


@cache
def _answerCounts():
    return _countLetters(loadAnswers())


@cache
def _answerMeanFrequencies():
    return meanLetterFrequencies(loadAnswers())


def _countLetters(words):
    """The occurrence count of each letter over already-validated words."""
    counts = np.zeros(LETTER_COUNT, dtype=np.int64)

    for word in words:
        for letter in word:
            counts[LETTERS.index(letter)] += 1

    return counts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _checkedWord(word):
    """A 5-letter lowercase word, or a loud error naming what was wrong."""
    if not isinstance(word, str) or len(word) != WORD_LENGTH:
        raise ValueError(
            "a word must be a %d-letter string, got %r" % (WORD_LENGTH, word)
        )

    if any(letter not in LETTERS for letter in word):
        raise ValueError("a word must contain only a-z, got %r" % (word,))

    return word


def _asWords(words, label):
    """A non-empty sequence of checked words, as a list."""
    if isinstance(words, str):
        raise ValueError(
            "%s must be a sequence of words, not a single string (%r)" % (label, words)
        )

    try:
        items = list(words)
    except TypeError:
        raise ValueError(
            "%s must be a sequence of words, got %r" % (label, words)
        ) from None

    if not items:
        raise ValueError("%s must contain at least one word" % label)

    for word in items:
        _checkedWord(word)

    return items


def _asFrequencies(values):
    """A 26-element, finite, non-negative, non-empty frequency distribution."""
    array = np.asarray(values, dtype=np.float64)

    if array.shape != (LETTER_COUNT,):
        raise ValueError(
            "frequencies must have shape (%d,), got %r" % (LETTER_COUNT, array.shape)
        )

    if not np.all(np.isfinite(array)):
        raise ValueError("frequencies must be finite, got %r" % (values,))

    if np.any(array < 0):
        raise ValueError("frequencies must be non-negative, got %r" % (values,))

    if array.sum() <= 0:
        raise ValueError("frequencies must have a positive total")

    return array
