"""Evidence -> model input.

WHY THIS MODULE CHANGED
The original encoder kept only the colour codes and threw the guessed words away. That was
valid ONLY because every game used the same fixed probes. Attempt 1 lets the model feed its
own guesses back as evidence, and then colours alone are ambiguous: a pattern of (1,0,0,0,0)
against "crane" and against "stout" describe entirely different facts, yet produced identical
numbers. Hence the two letter-aware encoders below.
"""

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
LETTER_INDEX = {c: i for i, c in enumerate(ALPHABET)}

# WHY -1: every slot needs a "no evidence here yet" value that can never be confused with
# real evidence. It matches the -1 encode() already used for missing colours, and sits
# outside every real range (colours 0..2, letter indices 0..25, one-hot slots 0/1).
# Using 0 instead would read as "grey" for colours and "letter absent" for one-hot slots.
PADDING = -1

# WHY the split: "compact" carries 5 letter INDICES per round (tiny, good for trees);
# "onehot" carries 5 x 26 one-hot letters (wide, but linear/kNN/SVM models can use the
# letter identity directly instead of treating a letter as an ordinal number).
COMPACT_BLOCK = 10       # 5 colours + 5 letter indices (0..25)
ONEHOT_BLOCK = 135       # 5 colours + 5 x 26 one-hot letters

LETTER_MODES = ("compact", "onehot")

# WHY 8: each letter gets a one-hot over mutually exclusive states, so the model reads
# "letter e is green at position 0" directly rather than inferring it from raw codes.
LETTER_STATES = 8        # never / eliminated / present-somewhere / green@0 .. green@4
NEVER, ELIMINATED, PRESENT = 0, 1, 2
GREEN_OFFSET = 3


def encode(gameState, probes):
    """Colour-only encoder: concatenate the 5 colour codes of each round, pad with -1.

    WHY STILL HERE: it is the original Attempt-1-in-name-only encoder, and
    test_features.py::testEncodeIgnoresTheGuessWordItself deliberately pins its
    colour-only behaviour. Keeping it untouched means the 77 pre-existing tests stay
    meaningful as a regression guard while the new encoders are built beside it.
    Do NOT use it for a solver that picks its own guesses - see the module docstring.
    """
    vectorColor = []

    for _, pattern in gameState:
        vectorColor.extend(pattern)

    while len(vectorColor) < probes * 5:
        vectorColor.append(-1)

    return vectorColor


def _checkRound(guess, pattern):
    """Reject malformed evidence loudly.

    WHY: a guess of the wrong length or a colour outside 0/1/2 would otherwise be encoded
    into a plausible-looking feature vector, and the model would be blamed for the
    resulting rubbish accuracy. Failing here points at the real culprit.
    """
    if len(guess) != 5:
        raise ValueError("guess must be 5 letters, got %r" % (guess,))
    if len(pattern) != 5:
        raise ValueError("pattern must have 5 colours, got %r" % (pattern,))
    for c in guess:
        if c not in LETTER_INDEX:
            raise ValueError("guess %r has a non-lowercase-letter character %r" % (guess, c))
    for color in pattern:
        if color not in (0, 1, 2):
            raise ValueError("pattern %r has a colour outside 0/1/2" % (pattern,))


def roundBlockSize(letterMode = "compact"):
    """Width of one round block.

    WHY: callers (dataset.py, the solver) must compute and ASSERT the total width
    (maxRounds * roundBlockSize) instead of hardcoding 50 or 675. Hardcoded widths are how
    a train/solve mismatch sneaks in silently - the single most common way this pipeline
    breaks.
    """
    if letterMode == "compact":
        return COMPACT_BLOCK
    if letterMode == "onehot":
        return ONEHOT_BLOCK
    raise ValueError("letterMode must be one of %r, got %r" % (LETTER_MODES, letterMode))


def encodeRounds(guesses, maxRounds = 5, letterMode = "compact"):
    """Encode (guess, pattern) evidence as padded per-round blocks.

    WHY: this is the encoder that makes "one model for every turn" possible. The evidence
    grows from 1 to 5 rounds as the game goes on; padding to maxRounds means the vector
    width never changes, so a single model can be asked at turn 2 and at turn 6 alike.

    WHY the colour/letter split: colours say WHAT the game told us, letters say WHAT WE
    ASKED. Without the letters the same colour pattern from two different guesses is
    indistinguishable, which is exactly the ambiguity that breaks a self-driving solver.

    Raises if given more rounds than maxRounds rather than silently growing the vector.
    """
    block = roundBlockSize(letterMode)
    rounds = len(guesses)

    if rounds > maxRounds:
        raise ValueError(
            "got %d rounds of evidence but maxRounds=%d; slice the history first, "
            "e.g. guesses[:maxRounds]" % (rounds, maxRounds)
        )

    vector = []

    for guess, pattern in guesses:
        _checkRound(guess, pattern)
        vector.extend(pattern)

        if letterMode == "compact":
            vector.extend(LETTER_INDEX[c] for c in guess)
        else:
            for c in guess:
                onehot = [0] * len(ALPHABET)
                onehot[LETTER_INDEX[c]] = 1
                vector.extend(onehot)

    while len(vector) < maxRounds * block:
        vector.append(PADDING)

    return vector


def encodeLetterState(guesses):
    """Aggregate every round into one fixed-size letter-state vector (26 x 8 = 208).

    WHY: this is the rival to encodeRounds, and plan.md 7.3 asks for the comparison to be
    settled by measurement rather than argument. Instead of storing rounds side by side, it
    folds all evidence into per-letter facts ("eliminated", "present somewhere", "green at
    position p"). The length is constant with no padding at all, and the information still
    grows with each round that is played.

    WHY green outranks yellow outranks grey: a confirmed fact must never be overwritten by
    weaker evidence from a later round. The repeated-letter case proves it matters -
    feedback("error", "shear") returns greys for r AND a green at position 4, and a naive
    "grey means eliminated" rule would delete a letter that is genuinely in the secret.

    WHY "always grey" is a sound test for "not in the secret": feedback() awards yellow
    greedily until the secret's supply of that letter runs out, so any letter that appears
    in a guess AND is in the secret must earn at least one yellow or green.
    """
    never = [True] * len(ALPHABET)
    present = [False] * len(ALPHABET)
    greenAt = [PADDING] * len(ALPHABET)

    for guess, pattern in guesses:
        _checkRound(guess, pattern)

        for position, (letter, color) in enumerate(zip(guess, pattern)):
            i = LETTER_INDEX[letter]
            never[i] = False

            if color == 2:
                if greenAt[i] == PADDING:
                    greenAt[i] = position
            elif color == 1:
                present[i] = True

    vector = []

    for i in range(len(ALPHABET)):
        if greenAt[i] != PADDING:
            state = GREEN_OFFSET + greenAt[i]
        elif present[i]:
            state = PRESENT
        elif not never[i]:
            state = ELIMINATED
        else:
            state = NEVER

        onehot = [0] * LETTER_STATES
        onehot[state] = 1
        vector.extend(onehot)

    return vector
