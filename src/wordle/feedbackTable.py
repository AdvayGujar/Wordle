"""Precomputed feedback codes for the answer vocabulary.

WHY THIS MODULE EXISTS: `wordle.feedback.feedback()` is pure Python and does a dictionary
pass for every call. Attempt 2's reinforcement-learning loop calls it millions of times, so
the plan precomputes every guess/secret pair once (2,315 x 2,315 = 5.36M entries, about
5.4 MB as uint8) and looks the answer up in an array instead.

The table is built lazily on first use and cached for the process. Building it takes about
a second because each secret is handled with NumPy across all guesses; the algorithm is
the same green-first, yellow-second pass as `feedback()`, and the tests cross-check it
against `feedback()` rather than trusting the rewrite.

This module does not change `feedback()` or `candidatesMatching()`. It sits beside them,
so every existing caller keeps the documented pure-Python behaviour.
"""

from functools import cache

import numpy as np

from wordle.feedback import feedback
from wordle.vocabulary import answersVocab, loadAnswers

PATTERN_LENGTH = 5
PATTERN_STATES = 3
PATTERN_COUNT = PATTERN_STATES ** PATTERN_LENGTH
_ALPHABET_SIZE = 26
# Position 0 is the most significant base-3 digit, so a code reads left to right.
_PLACE_WEIGHTS = tuple(PATTERN_STATES ** power for power in range(PATTERN_LENGTH - 1, -1, -1))


def patternCode(pattern):
    """Turn a 5-colour pattern into one integer in [0, 243).

    WHY: the lookup table stores one byte per pair instead of five, and comparing a whole
    row of guesses against one code is a single vectorised equality test.
    """
    if len(pattern) != PATTERN_LENGTH:
        raise ValueError("pattern must have %d entries, got %r" % (PATTERN_LENGTH, pattern))

    code = 0
    for state in pattern:
        if state not in (0, 1, 2):
            raise ValueError("pattern entries must be 0, 1 or 2, got %r" % (pattern,))
        code = code * PATTERN_STATES + state
    return code


def decodePattern(code):
    """Inverse of `patternCode`, returned as a tuple so it compares with `feedback()`."""
    if not isinstance(code, (int, np.integer)) or not 0 <= code < PATTERN_COUNT:
        raise ValueError("code must be an integer in [0, %d), got %r" % (PATTERN_COUNT, code))

    states = [0] * PATTERN_LENGTH
    for position in range(PATTERN_LENGTH - 1, -1, -1):
        code, states[position] = divmod(int(code), PATTERN_STATES)
    return tuple(states)


@cache
def feedbackTable():
    """(2315, 2315) uint8 table: row = guess index, column = secret index.

    WHY CACHED: the table costs about a second and ~5.4 MB. Rebuilding it per game would
    erase the speedup it exists to provide.
    """
    answers = loadAnswers()
    letters = _letterIndices(answers)
    bySecret = np.empty((len(answers), len(answers)), dtype=np.uint8)

    for secretIndex in range(len(answers)):
        bySecret[secretIndex] = _codesForSecret(letters, letters[secretIndex])

    return np.ascontiguousarray(bySecret.T)


def feedbackCode(guess, secret):
    """Feedback code for one pair, using the table when both words are answers."""
    index = answersVocab().index
    guessIndex = index.get(guess)
    secretIndex = index.get(secret)

    if guessIndex is None or secretIndex is None:
        return patternCode(feedback(guess, secret))
    return int(feedbackTable()[guessIndex, secretIndex])


def candidatesMatchingFast(constraints, vocab=None):
    """Vectorised equivalent of `wordle.feedback.candidatesMatching()`.

    WHY: the solver and the RL environment recompute the consistent set every turn, and
    that is exactly what the precomputed table makes cheap. Guesses that are legal but not
    answers still work through the per-word fallback, so the contract is unchanged.
    """
    answers = loadAnswers()
    index = answersVocab().index
    words = answers if vocab is None else tuple(vocab)
    constraints = list(constraints)

    mask = np.ones(len(answers), dtype=bool)
    slowConstraints = []
    for guess, pattern in constraints:
        guessIndex = index.get(guess)
        if guessIndex is None:
            slowConstraints.append((guess, pattern))
        else:
            mask &= feedbackTable()[guessIndex] == patternCode(pattern)

    if vocab is None and not slowConstraints:
        return [answers[position] for position in np.flatnonzero(mask)]

    return [
        word
        for word in words
        if _isConsistent(word, index, mask, slowConstraints, constraints)
    ]


def _isConsistent(word, index, mask, slowConstraints, constraints):
    """Does one vocabulary word survive the fast mask and the slow guesses?

    WHY check all constraints for non-answer words: the mask only has a column for answers,
    so a legal-but-not-answer word has to be checked against everything by hand.
    """
    wordIndex = index.get(word)
    if wordIndex is None:
        return all(feedback(guess, word) == pattern for guess, pattern in constraints)
    if not mask[wordIndex]:
        return False
    return all(feedback(guess, word) == pattern for guess, pattern in slowConstraints)


def _letterIndices(words):
    """Turn words into a (word, position) array of 0-25 letter indices."""
    return np.array([[ord(letter) - ord("a") for letter in word] for word in words], dtype=np.int8)


def _codesForSecret(letters, secretLetters):
    """Feedback code of every guess against one secret, as a uint8 array."""
    counts = np.bincount(secretLetters, minlength=_ALPHABET_SIZE)
    green = letters == secretLetters
    result = np.zeros(letters.shape, dtype=np.uint8)
    result[green] = 2

    remaining = _remainingAfterGreens(letters, green, counts)
    for position in range(PATTERN_LENGTH):
        letter = letters[:, position]
        yellow = (~green[:, position]) & (remaining[:, position] > 0)
        result[yellow, position] = 1
        for later in range(position + 1, PATTERN_LENGTH):
            remaining[yellow & (letters[:, later] == letter), later] -= 1

    return result.dot(np.array(_PLACE_WEIGHTS, dtype=np.uint16)).astype(np.uint8)


def _remainingAfterGreens(letters, green, counts):
    """Secret counts left for each guess position after the green pass.

    WHY per position: a guess can repeat a letter, so every position needs the count of
    *its* letter minus the greens that letter already consumed anywhere in the guess.
    """
    guessCount = letters.shape[0]
    remaining = np.empty(letters.shape, dtype=np.int16)

    for position in range(PATTERN_LENGTH):
        letter = letters[:, position]
        greensForLetter = np.zeros(guessCount, dtype=np.int16)
        for other in range(PATTERN_LENGTH):
            greensForLetter += green[:, other] & (letters[:, other] == letter)
        remaining[:, position] = counts[letter] - greensForLetter

    return remaining
