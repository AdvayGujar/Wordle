# Attempt 2 — reinforcement learning (the agent learns a strategy on its own)

**Status:** 🟡 **Phase A complete** (environment, split, observation, action mask). Phases B–D
(baselines, learning algorithms, experiment + write-up) are planned but not built. This file
records what Attempt 2 is trying to do, what has been built so far, and every design choice
behind it. It will be extended with results as the phases land.

**Files (Phase A)**

| Piece | Purpose |
|---|---|
| `config.py` | Every constant Attempt 2 uses: seed, split sizes, reward weights, guess limit, observation mode |
| `splits.py` | The train / held-out secret lists, plus a no-leakage assertion |
| `observation.py` | Evidence history → one fixed-length vector the policy can read |
| `actionMask.py` | Which of the 2,315 answers are still legal to guess, given the evidence |
| `wordleEnv.py` | The Gymnasium environment: one episode = one Wordle game |

**Files (planned)**

| Piece | Purpose |
|---|---|
| `baselines.py` | Masked random, random legal, fixed word, and the Attempt-1 solver |
| `eval.py` | The one measurement harness every policy is scored with |
| `policies.py` | The network(s), and masked action selection |
| `reinforce.py`, `dqn.py`, `ppo.py` | The three algorithms |
| `replayBuffer.py`, `train.py`, `smoke.py`, `experiment.py` | Buffer, CLI, smoke test, experiment runner |
| `attempt2.md` | This file |

---

## Architecture at a glance

Attempt 2 has one loop — *the agent plays a game and gets a reward* — and three places where
the data flows differently: **training** (the agent plays against training secrets and the
policy improves), **evaluation** (the agent plays against the 400 held-out secrets and is
never allowed to improve), and **the episode itself** (what happens between a guess and the
next observation).

**Episode anatomy — one turn of one game (built)**

```
              observation (60 floats)          action_mask (2315 bools)
                  from observation.py             from actionMask.py
                        │                                │
                        └──────────────┬─────────────────┘
                                       v
                            ┌─────────────────────┐
                            │  policy network     │   Phase C (not built yet)
                            │  obs -> 2315 logits │   masks the logits to -inf
                            └──────────┬──────────┘   on every illegal action
                                       v
                                 masked logits
                                       │
                                       v
                            choose one legal action
                                       │
                                       v
                       action = answer index = the guess word
                                       │
                                       v
                     feedbackTable()[action, secretIndex]   ──> 5 colours
                     (array lookup, no Python feedback())
                                       │
                                       v
                            history += (guess, pattern)
                                       │
                    ┌──────────────────┴──────────────────┐
                    v                                     v
          observation for the next turn        solved?  or  6 guesses used?
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   v
                     terminated / truncated  +  reward
```

**Training path (Phase A built; the learning loop is Phase C)**

```
            splitSecrets(1915, seed=42)                    [core, reused]
                          │
            ┌─────────────┴──────────────┐
            v                            v
   trainSecrets (1915)        heldOutSecrets (400, disjoint)
            │                            │
            │                            │
   WordleEnv(mode="train")     WordleEnv(mode="eval")
   reset(seed) samples ONLY    reset(seed) samples ONLY
   from trainSecrets           from heldOutSecrets
            │                            │
            v                            v
   ┌────────────────────┐      ┌────────────────────┐
   │ agent picks a guess│      │ agent picks a guess│
   │ env returns obs,   │      │ env returns obs,   │
   │ reward, done       │      │ reward, done       │
   └─────────┬──────────┘      └─────────┬──────────┘
             │                            │
             v                            v
   policy.update(...)              NO UPDATES - frozen policy
   (REINFORCE / DQN / PPO)          (measured, then reported)
             │                            │
             v                            v
      learning curve              win rate / avg guesses /
   (reward, win rate)             guess distribution / first guess
```

**Evaluation path — what every reported number must come from (Phase B, planned)**

```
                       policy or baseline (one pick function)
                                      │
                                      v
                            eval.evaluate(policy, secrets)
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            v                         v                         v
       win rate              avg guesses on wins        first-guess counts
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      v
                              the comparison table
```

**How to read it.**

- **Data → core:** Attempt 2 adds **no new core code**. It reuses `splitSecrets`,
  `loadAnswers`, `answersVocab`, `feedbackTable`, `encodeRounds` and `encodeLetterState`
  exactly as they are, and puts everything new under `src/attempt2/`.
- **The split is the load-bearing piece.** An RL agent learns by *playing*, so if the
  environment can sample an evaluated secret during training, every held-out number is
  contaminated. `WordleEnv` holds the split itself and samples from exactly one half.
- **The mask is the reason the agent can learn at all.** Without it the policy has to spend
  its capacity rediscovering "don't repeat a guess" and "don't guess an eliminated word".
  With it, the legal set often collapses to a handful of words by turn 4.
- **One measurement harness.** Baselines and trained agents are scored by the same function
  on the same 400 secrets. If a number is reported anywhere else, it must come from there.
- **Two halves, one agent.** Training mode and evaluation mode are the same class with a
  different secret pool — not a flag that might be set wrongly halfway through a run.

**Data shapes at each boundary**

| Boundary | Shape / type | Meaning |
|---|---|---|
| Secret lists | `tuple[str]` | 1,915 training secrets; 400 held-out secrets |
| Observation | `np.ndarray`, `(60,)`, float32 | 6 rounds × (5 colours + 5 letter indices), padded with `-1` |
| Action mask | `np.ndarray`, `(2315,)`, bool | True where the answer is still legal to guess |
| Action | `int` in `[0, 2315)` | The index of the guessed answer |
| History | `list[tuple[str, tuple[int, ...]]]` | `(guess, pattern)` pairs in play order |
| Reward | `float` | `+1` win, `0` loss, `-0.05` per guess played |
| Step result | `(obs, reward, terminated, truncated, info)` | Gymnasium 1.1 contract |
| `info` | `dict` | `round`, `candidates` (legal actions left), `guess`, `solved` |

---

## 1. What Attempt 2 is actually trying to do

Attempt 1 learned the *rules*: given the evidence and a candidate word, is that word
consistent? It scored AUC 1.000, and it still could not beat the plain rule filter, because
with a uniformly chosen secret **every consistent candidate is equally likely**. That is a
theorem about the game, not a modelling failure.

The question Attempt 1 could not answer is a *decision* question:

> Having narrowed the field with the rules, **which** word should be guessed next — and
> **which** guess would tell me the most?

Attempt 2 reframes that as reinforcement learning: an agent plays games, receives a reward, and
improves its policy. There are no labels for "the right guess" — only the reward that the
game itself produces.

### 1.1 It is reward-driven learning, not "unsupervised learning"

The original plan called this "unsupervised learning with no human input". That name hides
the mechanism: clustering and PCA have no environment and no reward, whereas this agent has
both. No *labels* are needed, but a *reward* is, and the reward here comes from `feedback()`
plus win/loss. Calling it reinforcement learning is the honest name, and it is the name used
from here on.

### 1.2 Libraries are allowed here — and deliberately not used

Attempt 3 exists to show the Module-5 calculus doing real work. Attempt 2 makes the opposite
trade: the point is the *learning setup* (environment, masking, reward, evaluation without
leakage), not the derivation of a gradient. `torch` and `gymnasium` are used; the three
algorithms are implemented directly rather than pulled from a reinforcement-learning library.
Two reasons:

1. **No new dependency.** `stable-baselines3` is not installed and adding it would change
   `requirements.txt` for a project whose whole point is being readable.
2. **The masking stays visible.** Action masking is the detail that decides whether this
   works, and implementing it ourselves means the mask is applied in one obvious place
   (the logits) rather than delegated to a library feature.

---

## 2. The split (`splits.py`) — the most important file here

### 2.1 The leak this prevents

Attempt 1's rule was "split by secret": no secret appears in both training and test. RL makes
that harder to guarantee, because the agent learns *by playing games*. If the environment can
sample any of the 2,315 answers during training, then every answer is contaminated, and a
"test win rate" is a number about the training set wearing a lab coat. The leak is not in the
agent guessing a test word — guessing any answer is legal — it is in the environment
*sampling* one.

### 2.2 The functions

```python
trainSecrets() -> list[str]        # the secrets the agent may play during training
heldOutSecrets() -> list[str]      # held out; the environment samples these ONLY in eval mode
assertNoLeakage(train=None, test=None) -> True   # raises ValueError if the halves overlap
```

`assertNoLeakage` also checks the *sizes*: 1,915 training and 400 held out. A split that
silently changed size would invalidate every comparison in the report, so the size is part of
the contract, not a detail.

### 2.3 The decisions behind it

- **Reuse `splitSecrets`, do not invent a second protocol.** The core already splits by whole
  secret, and Attempt 1 depends on that same function. Two definitions of "held out" would
  drift apart.
- **`nTrain = 1915`, so 400 are held out.** The plan originally said `nTrain = 2115` (200 held
  out) *and* that the held-out 200 must be the ones used for the Attempt-1 comparison — but
  Attempt 1 actually holds out `shuffled[600:1000]`, 400 words. Those are different sets, so
  the comparison would not have been like-for-like. The resolution: Attempt 2 holds out
  `shuffled[1915:]` (400 words, the same evaluation size as Attempt 1) and **re-runs the
  Attempt-1 solver on exactly those 400** when the comparison table is built. The RL agent
  still gets 1,915 training secrets, which is the variety RL needs — 600 would be thin.
- **The split lives on the environment, not in a global.** `WordleEnv` holds the two lists and
  `reset()` picks one of them by mode. There is deliberately no "all answers" attribute that a
  later `reset()` could reach.
- **The assertion is in the constructor.** `WordleEnv.__init__` calls `assertNoLeakage` every
  time it is built, so the environment cannot be constructed in a leaky configuration even if a
  test forgets to check.
- **The name is `heldOutSecrets`, not `testSecrets`.** Pytest collects any imported name
  that starts with `test` as a test function and warns about it. A detail, but it cost a
  confusing warning during Phase A.

---

## 3. The observation (`observation.py`)

### 3.1 The functions

```python
observationFor(history, mode) -> np.ndarray   # (60,) float32 for the default mode
observationWidth(mode) -> int                  # 60 for "rounds", 208 for "letterState"
observationBounds(mode) -> (low, high)         # the real value range, for the Box space
OBSERVATION_WIDTH                              # 60, for the default mode
```

### 3.2 The design choices

- **Fixed width, always.** One network has to handle turn 1 and turn 5 with the same weights.
  The evidence grows as the game goes on, so it has to be padded to a constant length. The
  core encoders already do this, so this module *chooses* between them and converts to
  float32; it does not re-implement them.
- **Two encoders, one default.** `rounds` (`encodeRounds`, 60 dims: five colour codes and five
  letter indices per round, padded with `-1`) keeps the turn structure, so the policy can tell
  "nothing played yet" from "three rounds of evidence". `letterState` (`encodeLetterState`,
  208 dims) folds the evidence into per-letter facts ("eliminated", "present somewhere",
  "green at position *p*"). `rounds` is the default — six times smaller, and the number of
  rounds played is genuinely informative. `letterState` stays available so the experiment can
  settle the choice by measurement instead of by argument.
- **The width is derived, never hardcoded.** `observationWidth` calls the same core functions
  the encoder uses, so a width that disagreed with the encoder cannot happen. A hardcoded 60
  is exactly the kind of number that goes stale and then fails far away from its cause.
- **The values are not normalised here.** The raw vector mixes colour codes (0–2), letter
  indices (0–25) and the `-1` padding marker, and a network does better when those scales are
  brought into range — but normalisation is a property of the *network*, not of the evidence.
  Keeping this module a faithful transcription of the core encoder also makes it trivially
  testable against the core.
- **Real bounds instead of `-inf, inf`.** The `Box` space declares the values the vector can
  actually take. An unbounded box is a small lie, and it also stops Gymnasium's own environment
  checker from verifying observations properly.

---

## 4. The action mask (`actionMask.py`)

### 4.1 The functions

```python
maskFor(history) -> np.ndarray(bool, (2315,))   # True where the answer is legal to guess
legalActions(history) -> list[int]              # the same thing as indices
```

### 4.2 What "legal" means

A word is legal when it is **consistent with every round of feedback so far** *and* **has not
already been played**. The secret is always consistent by definition, so the secret is never
masked out — otherwise the agent could not finish the game.

### 4.3 The design choices

- **A hard constraint, not a penalty.** An illegal action is not "unlikely", it is impossible.
  The mask is applied to the logits before the softmax (`-inf` on every illegal action), so an
  illegal action has probability exactly zero and the gradient never pushes towards it.
- **A function of the history alone.** The mask reads no secret and uses no randomness, so
  the same history always produces the same mask. That is what keeps the environment
  deterministic and removes the "non-stationarity with masking" pitfall entirely.
- **Built from the precomputed table, directly.** `feedbackTable()` holds every
  (guess, secret) feedback pair as one byte. Comparing one table row against the pattern code
  gives the consistency of all 2,315 answers in a single vectorised operation. Going through
  the core's `candidatesMatchingFast` instead meant two Python loops over up to 2,315 words on
  every turn; the direct version is roughly **10× faster** end-to-end (587 → 5,600 steps/sec
  including the rest of the step). The *rule* is unchanged — same table, same `patternCode` —
  and `tests/test_attempt2Mask.py` pins the result against the slow `candidatesMatching()`
  reference. No core file was changed.
- **A fallback for hand-written evidence.** A guess that is not an answer has no row in the
  table, so that path falls back to the core filter. The agent only ever plays answers, so
  this exists for callers that pass evidence they typed themselves.

---

## 5. The environment (`wordleEnv.py`)

### 5.1 The functions

```python
class WordleEnv(gym.Env):
    __init__(mode="train"|"eval", observationMode=..., stepPenalty=..., winReward=...,
             lossReward=..., maxGuesses=6)
    reset(seed=None, options=None) -> (observation, info)
    step(action) -> (observation, reward, terminated, truncated, info)
    action_mask() -> np.ndarray(bool, (2315,))
    history -> list[tuple[str, tuple[int, ...]]]     # rounds played so far
    secret -> str                                    # DIAGNOSTICS ONLY
    guessesUsed -> int
```

### 5.2 The design choices

- **The action *is* the guess.** The action space is the 2,315 answers, so an action is an
  answer index and the guess is `answersVocab().wordAt(action)`. There is no mapping layer
  to get wrong. The cost of this choice is real and is written down in §8: the agent can
  never play a legal non-answer as a probe.
- **`terminated` and `truncated` are kept apart.** `terminated` means the guess was the
  secret; `truncated` means the six guesses ran out. A learning algorithm treats "reached
  the goal" and "ran out of time" differently, so collapsing them would throw away
  information the algorithms need.
- **The reward is `+1` for a win, `0` for a loss, `-0.05` per guess played.** A win in 3
  guesses returns 0.85 and a win in 6 returns 0.70, so finishing early is worth something —
  but never more than winning. The step penalty is what gives the agent a reason to converge
  quickly rather than to survive; without it the return of every win would be identical and
  the agent would have no signal about *how* to win. A large penalty would teach it to guess
  wildly to end episodes sooner, so `-0.1` is the only alternative value to be tried.
- **Feedback comes from the table, not from `feedback()`.** One round of feedback is an
  array read instead of a pure-Python dictionary pass. An RL run makes millions of these
  calls; this is the difference between minutes and hours.
- **`reset(seed=…)` is reproducible.** `super().reset(seed=seed)` seeds Gymnasium's
  `self.np_random`, and the secret is drawn from that generator, so the same seed always
  starts the same game. The env, the policy and the exploration all have to be seeded, or
  results do not reproduce.
- **Stepping after the episode ends raises.** `step()` refuses to run once the game is over
  instead of quietly returning a meaningless reward. This caught a real bug during the first
  smoke run: after a win the action mask is empty, and a loop that kept stepping got an empty
  legal set. Failing loudly points at the loop that forgot to stop.
- **The mask is cached and invalidated per round.** The policy asks for it every turn;
  recomputing it would double the cost of a step. The cache is cleared on every new round,
  so it can never be stale.

---

## 6. What Phase A measured

Two numbers came out of building the environment, and both matter for Phase B.

### 6.1 A masked *random* policy already wins ~100% of games

2,000 games with a uniform-random choice among the legal (consistent, unguessed) answers:
**100% wins**. A 300-game spot check: 98%.

The mask does the hard work. The secret is always still consistent, so the legal set shrinks
fast, and a uniform pick among a handful of words hits it. This is the same lesson Attempt 1
ended on — the *rules* solve the game and the remaining headroom is *strategy* — now showing
up in the RL setting.

What it changes:

- **Win rate can no longer judge the agent.** The ceiling is ~100% and the random baseline
  already reaches it, so "the agent wins 99%" says nothing about whether it learned
  anything.
- **The metrics that still discriminate are average guesses used, the guess-count
  distribution, and the first-guess distribution** — the last of which is the headline
  result Attempt 2 promised (the plan's "did it discover a strong opener by itself?").
- **Phase B must include "masked random" as the real bar.** A legal-random baseline is ~0%
  and would make every policy look brilliant.

### 6.2 The environment is fast enough

With the feedback table warm, the env runs **~15,000–20,000 steps/sec** (mask + observation
+ step; ~5,600/sec measured through a harness that also pays for `flatnonzero` and the random
choice). The one-off cost is building the feedback table (~1 s, cached for the process).

The plan warned that "millions of steps need vectorised/parallel envs". At this speed that is
not true for the planned scale — a 1M-step run is minutes of environment time — so no
vectorised environment machinery is needed.

---

## 7. How to run and test

```bash
# the whole suite (238 tests: 196 pre-existing + 42 new)
.venv/bin/python -m pytest tests/ -q

# the Phase-A tests on their own
.venv/bin/python -m pytest tests/test_attempt2Splits.py tests/test_attempt2Observation.py \
                          tests/test_attempt2Mask.py tests/test_attempt2Env.py -q

# drive the environment by hand
PYTHONPATH=src .venv/bin/python -c "
from attempt2.wordleEnv import WordleEnv
env = WordleEnv()
obs, info = env.reset(seed=42)
print(obs.shape, info['candidates'])
print(env.action_mask().sum(), 'legal actions')
"
```

The environment also passes Gymnasium's own `check_env`, which is the cheapest way to catch a
contract violation before an algorithm is built on top of it.

**Test files**

| File | What it pins |
|---|---|
| `tests/test_attempt2Splits.py` | Disjointness, determinism, the sizes, and that the split is the core's split |
| `tests/test_attempt2Observation.py` | Fixed width, exact agreement with the core encoders, the padding |
| `tests/test_attempt2Mask.py` | The mask equals the slow `candidatesMatching()` reference, minus the played words |
| `tests/test_attempt2Env.py` | The Gymnasium contract, the reward values, and that training never samples a held-out secret |

---

## 8. Phases B–D — what comes next

| Phase | Modules | The question it answers |
|---|---|---|
| **B** | `baselines.py`, `eval.py`, `smoke.py` | What does the agent have to beat, and how is it measured? |
| **C** | `policies.py`, `replayBuffer.py`, `reinforce.py`, `dqn.py`, `ppo.py`, `train.py` | Can it learn? REINFORCE first — if the simplest algorithm cannot show a non-flat curve, adding two more only hides the bug |
| **D** | `experiment.py`, this file | What did it learn on its own, and where does it fall short? |

Phase B is the next gate, and it has one non-obvious requirement: **the Attempt-1 solver must
be re-run on the Attempt-2 held-out set** (`shuffled[1915:]`, 400 secrets), because Attempt
1's own 400-game evaluation used a different slice (`shuffled[600:1000]`). Without that, the
"Attempt 1 vs Attempt 2" row in the table would be comparing two different test sets.

---

## 9. Limitations and open questions

- **The agent cannot play a non-answer.** The action space is the 2,315 answers, so the agent
  can never use a legal probe word that is not a possible secret. This is a real restriction
  on the strategy it can learn, and it must be stated in the final results rather than
  discovered by a reader.
- **Win rate is not a usable headline.** See §6.1. The final table has to lead with average
  guesses and the first-guess distribution.
- **A uniform posterior still applies.** As in Attempt 1, every consistent candidate is
  equally likely a priori. What the agent can learn is which consistent candidate is *worth
  guessing* given what it already knows — a strategy, not a prediction.
- **The step penalty changes what "optimal" means.** −0.05 is a modelling choice, not a fact
  about Wordle. A larger penalty would reward different behaviour, and the report has to say
  which penalty produced which number.
- **The split is fixed at seed 42.** One split means one held-out set. Differences of a
  percentage point on 400 games are within noise; the report should treat them as such.
- **Phase C is the open risk.** The environment is fast and correct, but whether a policy
  gradient over a 2,315-action space with a hard mask converges on this task is exactly what
  Phases C and D have to find out — the honest outcome may be "it learns a good opener and
  little else".
