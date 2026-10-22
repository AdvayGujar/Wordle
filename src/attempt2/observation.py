"""Evidence history -> one fixed-length vector the policy can read.

WHY A FIXED WIDTH: a single network has to handle turn 1 and turn 5 with the same weights.
The evidence grows as the game goes on, so it has to be padded to a constant length; the
core encoders already do exactly that, and this module only chooses between them and
converts to the dtype the network wants.

THE TWO ENCODERS (both already in `wordle.features`, neither rewritten here):

- **"rounds"** - `encodeRounds(history, maxRounds=6, "compact")`, 60 dims: five colour
  codes and five letter indices per round, padded with -1. Keeps the turn structure, so the
  policy can tell "nothing yet" from "three rounds of evidence".
- **"letterState"** - `encodeLetterState(history)`, 208 dims: per-letter facts ("eliminated",
  "present somewhere", "green at position p") folded into one-hot states. No padding at all,
  because the length is constant by construction.

WHY "rounds" IS THE DEFAULT: it is six times smaller, and the number of rounds played is
genuinely informative about how the game is going. The other mode stays available so the
experiment can settle the choice by measurement rather than by argument (plan.md §8.10 A3).

WHY THE VALUES ARE NOT NORMALISED HERE: raw encodings mix colour codes (0..2), letter
indices (0..25) and the -1 padding marker, and a network does better when those scales are
brought into range. Normalisation is a property of the NETWORK, not of the evidence, so it
belongs in `attempt2.policies` - this module stays a faithful transcription of the core
encoder, which also makes it easy to test against the core directly.
"""

import numpy as np

from attempt2 import config
from wordle.features import encodeLetterState, encodeRounds, roundBlockSize

# The two core encoders, keyed by the mode names used in `config.OBSERVATION_MODES`.
_ENCODERS = {
    "rounds": lambda history: encodeRounds(history, config.MAX_ROUNDS, "compact"),
    "letterState": encodeLetterState,
}


def observationWidth(mode=config.OBSERVATION_MODE):
    """Width of the observation vector for a mode.

    WHY COMPUTED AND NOT HARDCODED: a width that disagrees with the encoder is the
    classic train/solve mismatch, and it fails far away from its cause. This derives the
    number from the same core functions the encoder uses.
    """
    if mode not in _ENCODERS:
        raise ValueError(
            "unknown observation mode %r; expected one of %r"
            % (mode, config.OBSERVATION_MODES)
        )

    if mode == "rounds":
        return config.MAX_ROUNDS * roundBlockSize("compact")

    return len(encodeLetterState([]))


# The width of the default mode. The env's observation space is built from this.
OBSERVATION_WIDTH = observationWidth(config.OBSERVATION_MODE)


def observationBounds(mode=config.OBSERVATION_MODE):
    """The `(low, high)` values the observation vector can take, per mode.

    WHY THIS EXISTS: the env has to declare a `Box` for the observation, and an
    unbounded `(-inf, inf)` box is a lie - it says the vector could hold any float, when
    in fact it holds colour codes (0..2), letter indices (0..25) and the -1 padding
    marker. Real bounds also let the environment checker verify observations properly.
    """
    if mode not in _ENCODERS:
        raise ValueError(
            "unknown observation mode %r; expected one of %r"
            % (mode, config.OBSERVATION_MODES)
        )

    if mode == "rounds":
        return -1.0, 25.0

    # The letter-state encoder emits one-hot states: 0 or 1 only.
    return 0.0, 1.0


def observationFor(history, mode=config.OBSERVATION_MODE):
    """Encode the evidence history as a 1-D float32 vector.

    WHY float32: this is the input to a torch network, and float64 would double the
    arithmetic for no benefit at these magnitudes.

    The vector is the core encoding, unmodified - see the module docstring for why the
    normalisation lives elsewhere.
    """
    if mode not in _ENCODERS:
        raise ValueError(
            "unknown observation mode %r; expected one of %r"
            % (mode, config.OBSERVATION_MODES)
        )

    return np.asarray(_ENCODERS[mode](history), dtype=np.float32)
