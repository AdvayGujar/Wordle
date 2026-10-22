"""Every constant Attempt 2 uses, in one place.

WHY A MODULE FOR CONSTANTS: the numbers in this attempt decide what the results MEAN. A
different split size silently makes the held-out win rate incomparable with Attempt 1; a
different step penalty silently changes what "optimal" means. If they live in five modules,
the report can quietly describe a run that did not happen. Every one of them is therefore
defined here once, and the modules import them.

THE DECISIONS THIS FILE ENCODES (settled before Phase A, see plan.md §8.11):

- **Split** - `SEED = 42`, `N_TRAIN_SECRETS = 1915`, so the 400 held-out secrets are
  `shuffled[1915:]`. The size matches Attempt 1's 400-game evaluation, and the Attempt-1
  solver is re-run on exactly these same 400 secrets so the comparison is like-for-like.
- **Observation** - the "rounds" encoder (60 dims) is the default; "letterState" (208 dims)
  stays available so the experiment can compare the two.
- **Action space** - the 2,315 answers only. Naming the answer is always legal, and the
  smaller space is what makes this learnable. It also means the agent can never play a
  legal non-answer as a probe, which is a limitation of the result and must be stated.
- **Reward** - +1 for a win, 0 for a loss, and a -0.05 step penalty, so a win in 3 guesses
  (0.85) beats a win in 6 (0.70) and beats any loss.
"""

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

# WHY 42: the same seed Attempt 1 uses, so the two attempts share one shuffle of the answer
# list and therefore one held-out set.
SEED = 42

# WHY 1915: 2,315 answers minus the 400 held out for evaluation. The split is by whole
# secret (the core's `splitSecrets`), so no secret is ever seen in training and evaluated.
N_TRAIN_SECRETS = 1915
N_TEST_SECRETS = 400

# WHY A VALIDATION SLICE: the held-out 400 may be read once, for the final table (§8.6).
# Everything else - the learning curve, the choice of hyper-parameters, the decision to
# stop training - has to be decided on data the agent has not been tuned against. So 200
# of the 1,915 training secrets are held back from *training* and used for validation.
# The agent trains on 1,715 and is judged on 200 during training and on 400 at the end.
N_VALIDATION_SECRETS = 200

# ---------------------------------------------------------------------------
# Game rules
# ---------------------------------------------------------------------------

# WHY 6: standard Wordle. Six guesses, and the observation must be able to hold a full
# game, so MAX_ROUNDS is the same number.
MAX_GUESSES = 6
MAX_ROUNDS = MAX_GUESSES

# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

# "rounds"  -> encodeRounds(history, MAX_ROUNDS, "compact")  -> 6 x 10  =  60 dims
# "letterState" -> encodeLetterState(history)                 -> 26 x 8  = 208 dims
OBSERVATION_MODES = ("rounds", "letterState")
OBSERVATION_MODE = "rounds"

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

WIN_REWARD = 1.0
LOSS_REWARD = 0.0

# WHY -0.05: small enough that winning is always worth far more than not winning, big
# enough to break the tie between a win in 3 and a win in 6 (0.85 vs 0.70). A large
# penalty teaches the agent to guess wildly to end episodes sooner (plan.md §8.8).
STEP_PENALTY = -0.05
