"""Every constant Attempt 3 uses, in one place.

WHY A MODULE FOR CONSTANTS: the numbers here decide what the result MEANS. The score
anchor and slope are fixed *before* any evaluation, so a picker cannot quietly tune them
against the held-out secrets and call the tuned policy a result. The guess set decides
whether the action space matches Attempts 1-2 (answers only) or is the full 14,855-word
list, which changes every number in the comparison. If those lived in three modules, the
report could describe a run that never happened.

THE DECISIONS THIS FILE ENCODES (settled in `attempt3.md` §4.1, before Phase B):

- **Split** - the seed and the sizes are re-exported from `attempt2.config` rather than
  copied. There is one split protocol in the project, and copying `SEED = 42` here would
  let the two attempts drift apart while still looking like the same 400 secrets.
- **Guess set** - the 2,315 answers only (`GUESS_SET = "answers"`). The action space then
  matches Attempts 1 and 2, so `attempt2.eval.evaluate` measures this attempt unchanged.
  The cost is that legal non-answers (`soare`, `adieu`, ...) cannot be played; the
  full-list prototype in `attempt3.md` §3.7 is the recorded reference for what that costs.
- **Probes** - a guess does not have to be consistent with the feedback (`ALLOW_PROBES`).
  This is what makes the expected-score criterion different from plain entropy (§2.7): if
  the guess set is restricted to the feasible set, every guess wins with the same
  probability and the win-chance term becomes a constant.
- **Score** - `SCORE_ANCHOR_BITS` and `SCORE_SLOPE` are the two constants of the
  `f(x) = 2^-x + 2(1 - 2^-x) + 1.5x/11.5` estimate in §2.7. They calibrate "bits of
  uncertainty left" to "further guesses", and they are *not* tuned on the held-out 400.
"""

from attempt2.config import (
    MAX_GUESSES,
    N_TEST_SECRETS,
    N_TRAIN_SECRETS,
    N_VALIDATION_SECRETS,
    SEED,
)

__all__ = [
    "ALLOW_PROBES",
    "GUESS_SET",
    "GUESS_SET_MODES",
    "MAX_GUESSES",
    "N_TEST_SECRETS",
    "N_TRAIN_SECRETS",
    "N_VALIDATION_SECRETS",
    "SCORE_ANCHOR_BITS",
    "SCORE_SLOPE",
    "SEED",
]

# ---------------------------------------------------------------------------
# The expected-score criterion (§2.7)
# ---------------------------------------------------------------------------

# WHY 11.5: it is log2(2315) rounded down - a full board's worth of uncertainty. The
# linear term of f is scaled so that f(11.5) is about 3.5 further guesses, which is what a
# uniformly random strategy actually needs on a whole answer list.
SCORE_ANCHOR_BITS = 11.5

# WHY 1.5: the weight of that linear term. The first two terms of f model the one-guess
# and two-guess contributions exactly; 1.5/11.5 is the slope that makes the tail match the
# observed cost of the hard branches. Both numbers are fixed before evaluation.
SCORE_SLOPE = 1.5

# ---------------------------------------------------------------------------
# The action space
# ---------------------------------------------------------------------------

# "answers" -> the 2,315 secret words, so indices match `wordle.vocabulary.loadAnswers()`
#              and the feedback table's columns. This is the build (§4.1.1).
# "legal"   -> all 14,855 legal guesses. Needs a 14,855 x 2,315 table and its own game
#              loop; kept as a documented option, not built.
GUESS_SET_MODES = ("answers", "legal")
GUESS_SET = "answers"

# WHY TRUE: a guess outside the feasible set is the only way the win-chance term can trade
# information against an immediate win (§2.7). With probes disabled the criterion collapses
# to pure entropy and loses 0.08 guesses, measured in §3.7.
ALLOW_PROBES = True
