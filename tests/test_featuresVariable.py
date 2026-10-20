"""Tests for the letter-aware, variable-length encoders in features.py.

These are the encoders Attempt 1 uses. encode() itself is pinned by test_features.py and
must not change.
"""

import pytest

from wordle.features import (
    ALPHABET,
    COMPACT_BLOCK,
    ELIMINATED,
    GREEN_OFFSET,
    LETTER_STATES,
    NEVER,
    ONEHOT_BLOCK,
    PADDING,
    PRESENT,
    encode,
    encodeLetterState,
    encodeRounds,
    roundBlockSize,
)
from wordle.feedback import feedback

MAX_ROUNDS = 5
BLOCK = 8


def blockOf(vector, letterIndex):
    """The 8-slot state block for one letter."""
    start = letterIndex * BLOCK
    return vector[start:start + BLOCK]


# ---------------------------------------------------------------------------
# Width: the property that lets ONE model serve every turn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expected", [("compact", 50), ("onehot", 675)])
@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 4, 5])
def testWidthIsConstantAtEveryRoundCount(mode, expected, rounds):
    evidence = [("crane", (2, 0, 1, 0, 0)) for _ in range(rounds)]
    assert len(encodeRounds(evidence, MAX_ROUNDS, mode)) == expected


def testRoundBlockSizeMatchesTheEncodedBlock():
    assert roundBlockSize("compact") == COMPACT_BLOCK
    assert roundBlockSize("onehot") == ONEHOT_BLOCK
    assert COMPACT_BLOCK == 5 + 5
    assert ONEHOT_BLOCK == 5 + 5 * len(ALPHABET)


def testRoundBlockSizeRejectsAnUnknownMode():
    with pytest.raises(ValueError):
        roundBlockSize("nonsense")


# ---------------------------------------------------------------------------
# Layout: colours then letters, padding only in the tail
# ---------------------------------------------------------------------------

def testACompactRoundRoundTripsToColoursThenLetterIndices():
    vector = encodeRounds([("crane", (2, 0, 1, 0, 0))], MAX_ROUNDS)
    assert vector[:10] == [2, 0, 1, 0, 0, 2, 17, 0, 13, 4]


def testUnplayedRoundsArePaddingOnly():
    vector = encodeRounds([("crane", (0, 0, 0, 0, 0))], MAX_ROUNDS)
    assert vector[10:] == [PADDING] * 40


def testAnEmptyHistoryIsAllPadding():
    assert encodeRounds([], MAX_ROUNDS) == [PADDING] * 50
    assert encodeRounds([], MAX_ROUNDS, "onehot") == [PADDING] * 675


def testPaddingIsDistinguishableFromGrey():
    grey = encodeRounds([("crane", (0, 0, 0, 0, 0))], MAX_ROUNDS)
    assert grey[0] == 0
    assert grey[10] == PADDING


def testOnehotRoundsHaveExactlyOneHotLetterPerPosition():
    vector = encodeRounds([("crane", (0, 0, 0, 0, 0))], MAX_ROUNDS, "onehot")
    for position in range(5):
        start = 5 + position * len(ALPHABET)
        slot = vector[start:start + len(ALPHABET)]
        assert sum(slot) == 1


# ---------------------------------------------------------------------------
# THE point of the redesign: letters are now carried
# ---------------------------------------------------------------------------

def testTheSamePatternFromDifferentGuessesEncodesDifferently():
    """The inverse of testEncodeIgnoresTheGuessWordItself in test_features.py."""
    pattern = (1, 2, 0, 0, 0)
    assert encodeRounds([("crane", pattern)], MAX_ROUNDS) != encodeRounds(
        [("slate", pattern)], MAX_ROUNDS
    )


def testTheOldColourOnlyEncoderStillIgnoresTheGuess():
    """encode() must be untouched: the existing suite depends on it."""
    pattern = (1, 2, 0, 0, 0)
    assert encode([("crane", pattern)], 1) == encode([("slate", pattern)], 1)


# ---------------------------------------------------------------------------
# Purity and mutation
# ---------------------------------------------------------------------------

def testEncodeRoundsIsPure():
    evidence = [("crane", (2, 0, 1, 0, 0))]
    assert encodeRounds(evidence, MAX_ROUNDS) == encodeRounds(evidence, MAX_ROUNDS)


def testEncodeRoundsDoesNotMutateItsInput():
    evidence = [("crane", (2, 0, 1, 0, 0))]
    before = list(evidence)
    encodeRounds(evidence, MAX_ROUNDS)
    assert evidence == before


def testEncodeLetterStateDoesNotMutateItsInput():
    evidence = [("crane", (2, 0, 1, 0, 0))]
    before = list(evidence)
    encodeLetterState(evidence)
    assert evidence == before


# ---------------------------------------------------------------------------
# Loud failure on bad input
# ---------------------------------------------------------------------------

def testTooManyRoundsRaisesInsteadOfGrowingTheVector():
    evidence = [("crane", (2, 0, 1, 0, 0))] * 6
    with pytest.raises(ValueError):
        encodeRounds(evidence, MAX_ROUNDS)


@pytest.mark.parametrize("bad", [("cran", (0, 0, 0, 0, 0)),
                                 ("cranes", (0, 0, 0, 0, 0)),
                                 ("cr4ne", (0, 0, 0, 0, 0)),
                                 ("crane", (0, 0, 0, 0)),
                                 ("crane", (0, 0, 3, 0, 0))])
def testMalformedEvidenceRaises(bad):
    with pytest.raises(ValueError):
        encodeRounds([bad], MAX_ROUNDS)
    with pytest.raises(ValueError):
        encodeLetterState([bad])


# ---------------------------------------------------------------------------
# encodeLetterState
# ---------------------------------------------------------------------------

def testLetterStateHasTheRightLengthAndOneStatePerLetter():
    vector = encodeLetterState([])
    assert len(vector) == 26 * LETTER_STATES == 208
    for letterIndex in range(len(ALPHABET)):
        assert sum(blockOf(vector, letterIndex)) == 1


def testEveryLetterStartsAsNeverGuessed():
    vector = encodeLetterState([])
    for letterIndex in range(len(ALPHABET)):
        assert blockOf(vector, letterIndex)[NEVER] == 1


def testFreeLettersAreEliminatedAndUsedLettersAreNot():
    vector = encodeLetterState([("crane", (0, 0, 0, 0, 0))])
    crane = "crane"
    for letterIndex, letter in enumerate(ALPHABET):
        block = blockOf(vector, letterIndex)
        if letter in crane:
            assert block[ELIMINATED] == 1
        else:
            assert block[NEVER] == 1


def testYellowBecomesPresentAndGreenRecordsItsPosition():
    vector = encodeLetterState([("crane", (2, 0, 1, 0, 0))])
    assert blockOf(vector, ALPHABET.index("c"))[GREEN_OFFSET + 0] == 1
    assert blockOf(vector, ALPHABET.index("a"))[PRESENT] == 1


def testGreenOutranksGreyFromTheSameRound():
    """feedback("error", "shear") is (1,0,0,0,2): r is grey at 1 and 2 but green at 4.

    Treating that grey as elimination would delete a letter that IS in the secret -- and
    the symptom would look like a bad model rather than a bad feature.
    """
    assert feedback("error", "shear") == (1, 0, 0, 0, 2)

    vector = encodeLetterState([("error", feedback("error", "shear"))])

    assert blockOf(vector, ALPHABET.index("r"))[GREEN_OFFSET + 4] == 1
    assert blockOf(vector, ALPHABET.index("r"))[ELIMINATED] == 0
    assert blockOf(vector, ALPHABET.index("e"))[PRESENT] == 1
    assert blockOf(vector, ALPHABET.index("o"))[ELIMINATED] == 1


def testGreenOutranksWeakerEvidenceFromALaterRound():
    evidence = [("crane", (2, 0, 0, 0, 0)), ("crate", (2, 0, 0, 0, 0))]
    vector = encodeLetterState(evidence)
    assert blockOf(vector, ALPHABET.index("c"))[GREEN_OFFSET + 0] == 1


def testLetterStateGrowsWithEvidenceButKeepsItsLength():
    short = encodeLetterState([("crane", (2, 0, 1, 0, 0))])
    long = encodeLetterState([("crane", (2, 0, 1, 0, 0)), ("slate", (0, 1, 0, 2, 0))])
    assert len(short) == len(long) == 208
    assert short != long
