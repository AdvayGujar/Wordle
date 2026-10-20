"""Candidate-relative features for Attempt 1's consistency classifier.

THE TASK THIS SUPPORTS
A classifier over 2,315 answer classes cannot rank the secret above other candidates: with a
uniform secret and deterministic feedback, every feedback-consistent candidate is equally
likely. The learnable question is binary and candidate-relative:

    given the evidence and a candidate word, is the candidate consistent with every round?

This module builds that feature vector WITHOUT calling feedback(guess, candidate). That call
is the label; putting it in the features would leak the answer and the model would learn
nothing. Everything here is a comparison a model could in principle learn from:
positional equality, letter counts, and the counts of green/yellow/grey marks in the guess.

WHY the width is fixed: a model must be consulted after 1, 2, 3, 4, or 5 rounds. Padded
rounds are -1 blocks, so the width is always 5 * 50 = 250, and one model serves every turn.
"""

from collections import Counter

from wordle.features import ALPHABET, LETTER_INDEX, PADDING

# Per round:
#   5 positions x [guess letter, candidate letter, colour, equality]       = 20
#   5 positions x [guess letter, candidate count, guess count,
#                  green count, yellow count, grey count]                  = 30
ROUND_FEATURES = 50
MAX_ROUNDS = 5
FEATURE_WIDTH = MAX_ROUNDS * ROUND_FEATURES   # 250


def _checkCandidate(candidate):
    """Validate the candidate at the boundary; a bad word would silently poison a row."""
    if not isinstance(candidate, str) or len(candidate) != 5:
        raise ValueError("candidate must be a 5-letter string, got %r" % (candidate,))
    for letter in candidate:
        if letter not in ALPHABET:
            raise ValueError("candidate %r has a non-lowercase letter %r" % (candidate, letter))


def candidateFeatures(evidence, candidate, maxRounds = MAX_ROUNDS):
    """Turn (evidence, candidate) into one fixed-width feature vector.

    The features never compare the candidate against the recorded pattern directly; they
    expose the raw ingredients the rule uses (equality, counts, colour counts) and leave
    the classifier to learn the rule.
    """
    if len(evidence) > maxRounds:
        raise ValueError(
            "got %d rounds of evidence but maxRounds=%d; slice the history first"
            % (len(evidence), maxRounds)
        )

    _checkCandidate(candidate)
    candidateCounts = Counter(candidate)
    vector = []

    for roundIndex in range(maxRounds):
        if roundIndex >= len(evidence):
            vector.extend([PADDING] * ROUND_FEATURES)
            continue

        guess, pattern = evidence[roundIndex]

        if len(guess) != 5 or len(pattern) != 5:
            raise ValueError("malformed evidence round: %r" % ((guess, pattern),))

        guessCounts = Counter(guess)
        greenCounts = Counter(guess[p] for p in range(5) if pattern[p] == 2)
        yellowCounts = Counter(guess[p] for p in range(5) if pattern[p] == 1)
        greyCounts = Counter(guess[p] for p in range(5) if pattern[p] == 0)

        for position in range(5):
            vector.extend([
                LETTER_INDEX[guess[position]],
                LETTER_INDEX[candidate[position]],
                pattern[position],
                int(candidate[position] == guess[position]),
            ])

        for position in range(5):
            letter = guess[position]
            vector.extend([
                LETTER_INDEX[letter],
                candidateCounts[letter],
                guessCounts[letter],
                greenCounts[letter],
                yellowCounts[letter],
                greyCounts[letter],
            ])

    if len(vector) != FEATURE_WIDTH:
        raise AssertionError(
            "feature width drifted: got %d, expected %d" % (len(vector), FEATURE_WIDTH)
        )

    return vector


def featureDescription():
    """One-line layout, useful when explaining a row in a notebook or the write-up."""
    return (
        "250 features = 5 rounds x (5 positions x [guess, candidate, colour, equal] + "
        "5 positions x [letter, candidateCount, guessCount, green, yellow, grey])"
    )
