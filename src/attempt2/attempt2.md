# Attempt 2 — reinforcement learning (the agent learns a strategy on its own)

**Status:** ✅ **complete** (Phases A–D: environment, split, observation, action mask,
evaluation harness, baselines, the three algorithms, the exhaustive opener search, the
experiment and the play modes). This file records what Attempt 2 is trying to do, what was
built, every design choice behind it, and what was measured.

**Files (built)**

| Piece | Purpose |
|---|---|
| `config.py` | Every constant Attempt 2 uses: seed, split sizes, reward weights, guess limit, observation mode |
| `splits.py` | The train / held-out / validation secret lists, plus a no-leakage assertion |
| `observation.py` | Evidence history → one fixed-length vector the policy can read |
| `actionMask.py` | Which of the 2,315 answers are still legal to guess, given the evidence |
| `wordleEnv.py` | The Gymnasium environment: one episode = one Wordle game |
| `eval.py` | The single measurement harness every policy is scored with |
| `baselines.py` | The policies the learned agent has to beat |
| `smoke.py` | The end-to-end smoke test: can the environment be played and scored? |
| `policies.py` | The networks, and the masked action selection all three algorithms share |
| `rollout.py` | The episode loop the three algorithms share |
| `replayBuffer.py` | The uniform replay buffer DQN learns from |
| `reinforce.py` | REINFORCE with a value baseline |
| `dqn.py` | Masked DQN with a target network |
| `ppo.py` | Clipped-surrogate PPO with a value head |
| `train.py` | The training CLI: model, learning curve and results |
| `openerSearch.py` | The exhaustive search for the best opening word |
| `experiment.py` | Evaluates every policy on the held-out secrets and writes the figures |
| `play.py` | The play modes: interactive, `--advise` and `--auto` |
| `attempt2.md` | This file |

**Files (planned)**

None — every module of Attempt 2 is built. The next attempt is Attempt 3 (information
theory), planned in `src/attempt3/attempt3.md`.

---

## Architecture at a glance

Attempt 2 has one loop — *the agent plays a game and gets a reward* — and three places where
the data flows differently: **the episode** (what happens between a guess and the next
observation), **training** (the agent plays against training secrets and the policy
improves), and **evaluation** (the agent plays against the 400 held-out secrets and is never
allowed to improve).

**Episode anatomy — one turn of one game (built)**

```
              observation (60 floats)          action_mask (2315 bools)
                  from observation.py             from actionMask.py
                        │                                │
                        └──────────────┬─────────────────┘
                                       v
                            ┌─────────────────────┐
                            │  policy network     │   reinforce.py / dqn.py / ppo.py
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

**Evaluation path — the one place every number comes from (Phase B, built)**

```
                     policy  ──────────────────────────────┐
                        │                                  │
                        v                                  │
              PolicyState(observation, mask, history)     │
                        │                                  │
                        v                                  │
              action = policy(state)                      │
                        │                                  │
                        v                                  │
            ┌───────────┴───────────┐                      │
            │ is the action legal?  │                      │
            └───────────┬───────────┘                      │
                yes │         │ no                         │
                    │         └──> counted as an illegal   │
                    │              action, and PLAYED       │
                    v                  anyway               │
            env.step(action) <──────────────────────────────┘
                    │
                    v
      ┌─────────────────────────────────────────────┐
      │ win rate · avg guesses on wins · avg guesses │
      │ over all games · wins-by-turn distribution · │
      │ first-guess counts · illegal actions        │
      └─────────────────────────────────────────────┘
```

**How to read it.**

- **Data → core:** Attempt 2 adds **no new core code**. It reuses `splitSecrets`,
  `loadAnswers`, `answersVocab`, `feedbackTable`, `encodeRounds` and `encodeLetterState`
  exactly as they are, and puts everything new under `src/attempt2/`.
- **The split is the load-bearing piece.** An RL agent learns by *playing*, so if the
  environment can sample an evaluated secret during training, every held-out number is
  contaminated. `WordleEnv` holds the split itself and samples from exactly one half — and
  refuses a secret from the other half even when one is forced.
- **The mask is the reason the agent can learn at all.** Without it the policy has to spend
  its capacity rediscovering "don't repeat a guess" and "don't guess an eliminated word".
  With it, the legal set often collapses to a handful of words by turn 4.
- **One measurement harness.** Baselines and trained agents are scored by the same function,
  on the same 400 secrets, through the same environment. If a number is reported anywhere
  else, it must come from there.

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
| `PolicyState` | `(observation, mask, history)` | Everything a policy may see — never the secret |
| `EvaluationResult` | dataclass | The score of one policy over one set of secrets |

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
  Attempt-1 solver on exactly those 400**. The RL agent still gets 1,915 training secrets,
  which is the variety RL needs — 600 would be thin.
- **The split lives on the environment, not in a global.** `WordleEnv` holds the two lists and
  `reset()` picks one of them by mode. There is deliberately no "all answers" attribute that a
  later `reset()` could reach.
- **The assertion is in the constructor.** `WordleEnv.__init__` calls `assertNoLeakage` every
  time it is built, so the environment cannot be constructed in a leaky configuration even if a
  test forgets to check.
- **Forced secrets are checked too.** `reset(options={"secret": ...})` (added in Phase B so the
  harness can play a specific secret) validates the secret against the mode's pool: an
  eval-mode env refuses a training secret and a train-mode env refuses a held-out one. The
  leak guard holds for forced secrets, not just for sampled ones.
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

## 6. The evaluation harness (`eval.py`)

### 6.1 The functions

```python
PolicyState(observation, mask, history)     # everything a policy may see
evaluate(policy, secrets, name, mode, maxGuesses, seed) -> EvaluationResult
formatResultsTable(results) -> str          # the markdown table for the report
firstGuessTable(counts, limit) -> str       # the opening-guess histogram
```

`EvaluationResult` carries `winRate`, `avgGuessesOnWins`, `avgGuessesAll`, the wins-by-turn
distribution, the first-guess counts, the number of illegal actions, and the per-game record.

### 6.2 The design choices

- **One harness for everything.** A baseline measured by its own script and an agent
  measured by its training loop would differ in some invisible way — a different guess
  limit, a different secret set, a different definition of "average guesses" — and the table
  would then compare two different experiments. So there is exactly one `evaluate()`.
- **A policy is `policy(state) -> action index`.** `PolicyState` bundles the observation, the
  mask and the history into one named object, so the interface is obvious and it is visible
  that the secret is *not* part of it. The Attempt-1 pickers are adapted to it rather than
  reimplemented.
- **The environment is used, not a reimplemented game loop.** The harness must score exactly
  the environment the agent will be trained on — same rules, same mask, same reward. A second
  copy of the loop would drift, and the drift would show up as a difference between training
  and evaluation.
- **Illegal actions are played and counted, not rescued.** The mask is a constraint on the
  *policy*, not on the game. If a policy asks to play a word the feedback already eliminated,
  it loses a guess — that is the honest cost of not using the mask — and the harness reports
  the count so the behaviour is visible instead of hidden.
- **The mode decides which secrets may be scored.** `evaluate(..., mode="eval")` uses an
  eval-mode environment, which refuses a training secret; `mode="train"` refuses a held-out
  one. "We evaluated on the held-out set" is therefore a property of the code, not a claim
  in the report.

---

## 7. The baselines (`baselines.py`) and what they say

### 7.1 The policies

| Baseline | What it is | Why it is here |
|---|---|---|
| `maskedRandomPolicy` | Uniform choice among the legal actions | **The bar.** It uses the same information the agent has and still wins almost every game |
| `fixedOpenerRandomPolicy` | `crane` first, then masked random | Isolates the value of the *opening* move from the rest of the game |
| `unmaskedRandomPolicy` | Uniform choice among all 2,315 answers, ignoring the mask | The ablation: what is the mask worth? |
| `fixedWordPolicy` | Always the same word | The control: a strategy that does nothing |
| `attempt1SolverPolicy` | Attempt 1's trained model + rule mask, adapted | The row the agent has to be compared against |

### 7.2 Measured on the 400 held-out secrets

```
| Policy                                | Win rate | Avg guesses (wins) | Avg guesses (all) | Illegal |
|---------------------------------------|----------|---------------------|--------------------|---------|
| masked random                         | 98.5%    | 4.07                | 4.10               | 0       |
| masked random, fixed opener 'crane'   | 99.0%    | 3.80                | 3.83               | 0       |
| unmasked random (no mask)             | 0.2%     | -                   | 5.99               | 1950    |
| fixed word ('crane' every turn)       | 0.2%     | -                   | 5.99               | 1995    |
| Attempt-1 solver (model + rule mask)  | 97.8%    | 3.90                | 3.95               | 0       |
```

### 7.3 Reading this honestly

- **The bar is 3.80 average guesses, not 3.90.** The strongest baseline is *masked random
  with a fixed opener* — it does not even look at the evidence, it just plays `crane` and
  then picks at random among the survivors. It beats the Attempt-1 solver (3.90) because one
  well-chosen opener is worth more than a better tie-break on later turns. A learned agent
  that merely matches Attempt 1's 3.90 has learned nothing at all.
- **Win rate is saturated and cannot judge the agent.** The two best baselines are 99.0% and
  98.5% — 0.5 points apart on 400 games, which is noise. Average guesses is the metric that
  separates them (3.80 vs 4.07), and it is the metric the report has to lead with.
- **The mask is worth everything.** Unmasked random wins one game in 400 and makes ~4.9
  illegal guesses per game: it spends its turns playing words the feedback has already
  eliminated. This is the quantitative version of the argument for masking in §4.
- **The Attempt-1 row reproduces.** Its 97.8% here matches the 97.8% Attempt 1 reported on
  its own 400 secrets (`shuffled[600:1000]`). Different secret set, same result — which is
  evidence that the harness and the split are measuring what they claim to.
- **A curiosity worth stating:** `crane` is one of the 400 held-out secrets, so the
  fixed-opener baseline wins in a single guess on that one game (the `1:1` in its wins-by-turn
  row). It is one game in 400, and the opener was fixed before the split, but it is the kind
  of detail that should be visible rather than discovered later.

---

## 8. What Phase A measured (environment facts)

Two numbers came out of building the environment, and both matter for Phase B.

### 8.1 A masked *random* policy already wins ~100% of games

2,000 games with a uniform-random choice among the legal (consistent, unguessed) answers:
**100% wins**. A 300-game spot check: 98%. On the full held-out set (§7.2): 98.5%.

The mask does the hard work. The secret is always still consistent, so the legal set shrinks
fast, and a uniform pick among a handful of words hits it. This is the same lesson Attempt 1
ended on — the *rules* solve the game and the remaining headroom is *strategy* — now showing
up in the RL setting.

### 8.2 The environment is fast enough

With the feedback table warm, the environment runs **~23,000 steps/sec** end-to-end
(measured through the harness, which also pays for the policy call and the bookkeeping). The
one-off cost is building the feedback table (~1 s, cached for the process).

The plan warned that "millions of steps need vectorised/parallel envs". At this speed that is
not true for the planned scale — a 1M-step run is under a minute of environment time — so no
vectorised environment machinery is needed.

---

## 9. The learning algorithms (Phase C)

### 9.1 What is shared and what is not

The three algorithms differ in their **loss**, not in how a game is played. The episode
loop lives once, in `rollout.py`, so a difference between their learning curves is a
difference in the algorithm and not a difference in the rules of the game.

| Module | What it does |
|---|---|
| `policies.py` | `MaskedPolicyNet` (observation → 2,315 logits + a value), `QNetwork`, the masked log-probabilities, and `selectAction` |
| `rollout.py` | `collectEpisode`, `episodeStream`, `discountedReturns` — the shared loop |
| `replayBuffer.py` | A uniform buffer; it stores the **next mask** as well as the next observation |
| `reinforce.py` | REINFORCE with a learned value baseline |
| `dqn.py` | DQN with a target network and ε-greedy exploration over the legal actions |
| `ppo.py` | PPO with generalised advantage estimation and a clipped surrogate |
| `train.py` | The CLI: `--algo`, `--episodes`, `--seed`, `--learning-rate`, and the artifacts it writes |

### 9.2 The mask, in all three algorithms

The mask is applied to the **logits before the softmax**, so an illegal action has
probability exactly zero and receives exactly zero gradient. It appears in three places,
and all three have to be right:

1. **Choosing** — `selectAction` samples or takes the argmax over the legal actions only.
2. **DQN's exploration** — ε-greedy picks a random *legal* action. Picking a random answer
   and hoping it is legal would waste most of the steps.
3. **DQN's Bellman target** — the maximum is taken over the legal actions of the *next*
   state, not over all 2,315 answers. A target that included the eliminated answers would
   ask the network to predict the value of guesses that cannot be made.

### 9.3 Why the exploration schedules are annealed

This was the single most consequential design decision in Phase C, and it was made from a
measurement rather than an intuition. Measured directly (400 games per opener, masked
random afterwards):

| Opener | Average guesses |
|---|---|
| `slate` | 3.795 |
| `crane` | 3.860 |
| `raise` | 3.865 |
| `roast` | 3.873 |
| `actor` | 3.940 |
| `fight` | 4.095 |
| `aback` | 4.255 |

**The openers differ by 0.46 guesses. The noise in a single game is about 1.0 guesses.**
Ranking two openers that differ by 0.10 guesses needs roughly 400 samples *each*. Sweeping
all 2,315 openers once, with enough samples to rank them, is about 230,000 episodes.

So both schedules are annealed:

- **REINFORCE's entropy bonus** goes from 0.2 down to 0.01 across training. A fixed strong
  bonus keeps the policy spread over all 2,315 answers and it never commits; a fixed weak
  (or zero) bonus lets it collapse onto whichever answer looked lucky in the first few
  hundred episodes, before any of them had been sampled often enough to be ranked.
- **DQN's ε** goes from 1.0 down to 0.05, measured in *steps* rather than episodes so the
  number of guesses per episode does not change how fast exploration fades.

### 9.4 Results — validation (during training) and held-out (read once at the end)

| Algorithm | Episodes | Time | Validation | **Held-out (400)** | Opener it chose |
|---|---|---|---|---|---|
| REINFORCE | 250,000 | 27 min | 98.0% win, **3.81** guesses | 97.8% win, **3.85** guesses | `actor` |
| PPO | 150,000 | 18 min | 100% win, 3.98 guesses | 98.8% win, 4.01 guesses | `slate` |
| DQN | 150,000 | 22 min | 96.5% win, 4.27 guesses | 95.8% win, 4.34 guesses | `fight` |

Against the Phase B baselines, on the same 400 held-out secrets:

| Policy | Win rate | Avg guesses (wins) |
|---|---|---|
| masked random, fixed opener `crane` | 99.0% | **3.80** |
| **REINFORCE (250k episodes)** | 97.8% | **3.85** |
| Attempt-1 solver (model + rule mask) | 97.8% | 3.90 |
| PPO (150k episodes) | 98.8% | 4.01 |
| masked random | 98.5% | 4.07 |
| DQN (150k episodes) | 95.8% | 4.34 |

### 9.5 Reading this honestly

- **The agent learns. It beats the random baseline and it beats Attempt 1.** REINFORCE
  reaches 3.85 guesses against masked random's 4.07 and the Attempt-1 solver's 3.90. On
  the opener alone that is a real difference: the agent chose `actor` (3.940 with random
  play afterwards) and reached 3.85, so **its later turns are worth about 0.09 guesses**
  over picking at random among the legal answers.
- **It does not beat one hand-picked word.** `crane` + random scores 3.80, and the agent
  never found an opener that beats it. Given §9.3 — the signal is an order of magnitude
  smaller than the noise — that is the expected outcome, not a surprise, and it is the
  honest ceiling of this attempt.
- **PPO found the best opener in the sample — and §10 shows it found the best opener of
  all 2,315.** It chose `slate`, and an exhaustive search over every possible opener
  confirms `slate` is the top of the ranking. It still scored 4.01, because its *later*
  greedy choices are worse than a random pick among the legal answers. A confidently wrong
  policy loses to a random legal guess, and evaluating greedily is not a free win for a
  partially-trained agent.
- **But PPO genuinely found good openers — the others did not.** Measuring the 8 words
  each agent actually played the most (400 games each, masked random afterwards):

  | Agent | Mean quality of its 8 most-played openers |
  |---|---|
  | **PPO** | **3.887** |
  | REINFORCE | 4.027 |
  | 8 openers picked at random | 4.043 |
  | DQN | 4.166 |

  PPO's words are genuinely better than random ones, so its concentration was learning,
  not luck. REINFORCE's are indistinguishable from random, so its final pick of `actor`
  was mildly lucky. DQN's are *worse* than random, which is consistent with its
  Q-values never concentrating at all — after 150,000 episodes it was still playing all
  2,315 openers almost evenly.
- **The agent learned the information content of an opener, not a word.** PPO's two most
  played openers are `slate` and `stale` — the same five letters in a different order —
  and it was swapping between them (36% `slate` against 60% `stale`) when training
  ended. Two anagrams carry almost the same information, which is what you would expect
  if the policy had learned "open with a well-spread common letter set" rather than a
  particular spelling.
- **DQN is the only algorithm below the random baseline.** 2,315 actions is too many for a
  value-based method to generalise over from a 60-dimensional state.
- **The validation numbers track the held-out numbers closely** (3.81 vs 3.85 for
  REINFORCE), which is the point of having a validation slice at all.

### 9.6 A bug worth recording

The REINFORCE policy loss was first built from a **detached** log-probability — the one
recorded when the action was chosen. The gradient therefore never reached the policy head,
and only the value head ever learned. The agent looked exactly like it was training: the
loss moved, a curve was plotted, the validation number was printed every 25,000 episodes.
Nothing changed, because the policy was not being updated at all.

It was found by measuring the thing the loss is supposed to measure: the learning rate made
no difference whatsoever to the policy's output, across four orders of magnitude, which is
not something a working learner does.

`tests/test_attempt2Learning.py` now asserts that the policy head receives a non-zero
gradient, because **"the loss is finite" is not the same as "the policy is learning"**.

### 9.7 The training setup

```
PYTHONPATH=src .venv/bin/python -m attempt2.train --algo reinforce --episodes 250000 \
    --learning-rate 0.01 --entropy-start 0.2 --entropy-end 0.01 --evaluation-every 25000
```

The CLI writes `artifacts/<algo>-model.pt`, `artifacts/<algo>-curve.csv` and
`artifacts/<algo>-results.json`. The artifacts are generated files and are not tracked.

The options are matched against the trainer's signature, so passing an option an algorithm
does not use (`--entropy-start` to DQN, say) is reported and ignored rather than crashing
halfway through a long run.

**Training never touches the held-out 400.** The training environment samples only the
1,715 training secrets; the validation slice (200) is used for the learning curve and for
every decision made during training; the held-out 400 were read once, at the end.

---

## 10. The best opener, by exhaustive search

### 10.1 The question

Phase C produced three agents and each of them picked an opening word. But "which word did
it pick" is not the interesting question — the interesting question is **how good could
any opener be?** An agent that finds `slate` has done something remarkable, or nothing at
all, and the only way to tell is to check all 2,315 candidates.

This is not a learning problem. It is a computation problem, and it is affordable:
`openerSearch.py` plays every answer as the opening guess, with masked-random play
afterwards, and ranks them.

| Stage | What | Cost |
|---|---|---|
| First pass | all 2,315 openers × 400 games | 926,000 games (~5 min on one core, ~1 min on eight) |
| Refinement | the top 60 × 4,000 games | 240,000 games |

### 10.2 The result

| Rank | Opener | Avg guesses | ± | Win rate |
|---|---|---|---|---|
| 1 | **`slate`** | **3.779** | 0.015 | 98.7% |
| 2 | `least` | 3.796 | 0.015 | 98.9% |
| 3 | `crate` | 3.822 | 0.015 | 98.8% |
| 4 | `slant` | 3.829 | 0.015 | 99.0% |
| 5 | `react` | 3.832 | 0.015 | 98.9% |
| 6 | `dealt` | 3.836 | 0.015 | 98.6% |

Across all 2,315 openers (first pass, 400 games each):

| | Avg guesses |
|---|---|
| Best | 3.750 |
| 1st percentile | 3.828 |
| 10th percentile | 3.915 |
| **Median** | **4.050** |
| 90th percentile | 4.235 |
| Worst (`jazzy`) | 4.605 |

**So the opening-word decision is worth about 0.30 guesses** — the difference between the
median opener and the best one. That is the entire ceiling of this decision, and it
bounds what any opener-learning agent could possibly achieve.

### 10.3 PPO found the best opener of all 2,315

This is the headline of Attempt 2. PPO, given no knowledge of the word list beyond the
action space and learning only from the reward signal, **converged onto `slate` — the top
of the exhaustive ranking.** It was not given the answer and it was not told which words
were good.

The other two agents did not: REINFORCE settled on `actor` (3.935 with 4,000 games) and
DQN on `fight` (4.157) — both mediocre. So the result belongs to PPO specifically.

### 10.4 What the best openers have in common

Look at the top of the ranking and the pattern is obvious. Every one of the best 20 is
built from the same small pool of common letters — 18 of the 20 contain `a`, 18 contain
`e`, and the rest draw on `t`, `l`, `s`, `c`, `r`, `n`, `p`:

```
slate  least  slant  react  dealt  leant  slice  leapt  scale  trace
cleat  alter  paste  plane  stare  crest  steal  blast  learn  crate
```

`slate`, `least`, `slant`, `stale`, `dealt`, `leant` are six permutations of the same five
letters, and they score within 0.07 of each other. **The information in an opening word is
in which common letters it tests, not in how they are arranged** — which is exactly what
PPO appears to have learned, and why it kept swapping between `slate` and `stale` (the
same five letters) when training ended.

### 10.5 The group of near-equals, and what it means for the RL result

**144 of the 2,315 openers are within 0.05 guesses of the best one**, and 382 are within
0.10. The good openers are a large, tightly packed group, not a single peak.

So the opening-word decision has a clear *target* (be in the top group, worth ~0.25
guesses over the median) but almost no *resolution* inside that group. Any of those 144
words is an equally good answer, and telling them apart needs ~4,000 games each.

This is the quantitative version of §9.3, and it explains the whole Phase C result: the
agent was chasing a difference of 0.02–0.05 guesses against a per-game noise of about 1.0
guess. PPO got lucky in the best possible way — it landed on the actual best word — but
its advantage was not that it had *identified* the best opener. It had found a member of
the good group.

### 10.6 Separating the opener from the strategy

The two decisions can be swapped independently, which says exactly what each agent got
right. All numbers are on the 400 held-out secrets:

| Opener | Later play | Avg guesses |
|---|---|---|
| `slate` | REINFORCE's learned policy | **3.682** |
| `crane` | masked random | 3.752 |
| `slate` | masked random | 3.768 |
| `actor` (REINFORCE's own) | REINFORCE's learned policy | 3.847 |
| `slate` (PPO's own) | PPO's learned policy | 4.013 |
| random | masked random | 4.056 |

- **REINFORCE learned a good strategy and a mediocre opener.** Its later play beats random
  (3.682 against 3.768 with the same opener), but its own opener is worth 0.22 guesses to
  it. Handed `slate`, the same policy is the best result in the project.
- **PPO learned the best opener and a poor strategy.** With the ceiling opener, its later
  play still costs 0.245 guesses against random legal guesses.
- **The combination is the best of the project** — 3.682 against the Attempt-1 solver's
  3.90 and the best hand-picked baseline's 3.752.

One caveat, stated rather than buried: the 3.682 is the best of several combinations
compared *on the held-out set*, so it is mildly optimistic. On the validation slice — 200
training secrets that the held-out evaluation never touched — the same combination scores
3.753 against 3.754 for `slate` plus masked random, i.e. **the gap over random is not
distinguishable from noise**. The robust claims are that `slate` is the best opener, that
PPO found it, and that REINFORCE's later play is at least as good as random. The claim
that the combination is significantly better than the best baseline is *not* supported.

---

## 11. The play modes

`play.py` gives Attempt 2 the same three modes Attempt 1 has.

| Mode | Command | What it does |
|---|---|---|
| Interactive | `python -m attempt2.play` | You type guess 1; the agent plays the rest |
| Advisor | `python -m attempt2.play --advise` | The agent proposes each word; you give back the colours |
| Automatic | `python -m attempt2.play --auto --games 400` | The agent is evaluated against the baselines on the held-out secrets |

`--algo` picks the agent: `reinforce` (the default and the best), `dqn` or `ppo`.

**Three details worth knowing:**

- **The advisor is agent-driven.** It opens with its own word, you report the five colours
  the board showed, and it proposes the next word from the evidence it now has:

  ```
    The agent's word: ACTOR
  Colours for ACTOR (G/Y/X per position, or e.g. 2 1 0 0 2): GYXXG
    1. ACTOR  🟩🟨⬜⬜🟩   [green, yellow, grey, grey, green]

    Consistent answers left: 3
  ```

  Colours that no secret could have produced are refused and the same word is proposed
  again, so a mistyped colour cannot poison the rest of the session.
- **The human may play any legal word, not only an answer** (interactive mode). The agent's
  action space is the 2,315 answers, because that is what it chooses from, but a person
  playing Wordle can type any legal guess. `WordleEnv.playWord()` plays those with the same
  rules, so the evidence the agent reasons about is the evidence a real game produces — and
  the agent's mask is recomputed from that history.
- **The advisor mode never sees the secret.** The observation and the mask are both
  functions of the history alone, so the advice is computed without a game at all. The
  same history always produces the same suggestion, which is the test that pins it.

---

## 12. How to run and test

```bash
# the whole suite (316 tests: 196 pre-existing + 120 new)
.venv/bin/python -m pytest tests/ -q

# the Attempt-2 tests on their own
.venv/bin/python -m pytest tests/ -q -k attempt2

# smoke test: can the environment be played and scored?
PYTHONPATH=src .venv/bin/python -m attempt2.smoke --games 50

# the baseline table, on the 400 held-out secrets
PYTHONPATH=src .venv/bin/python -m attempt2.baselines

# train an agent (writes the model, the curve and the results)
PYTHONPATH=src .venv/bin/python -m attempt2.train --algo reinforce --episodes 250000 \
    --learning-rate 0.01 --entropy-start 0.2 --entropy-end 0.01

# search all 2,315 openers for the best one (about 1.2 million games)
PYTHONPATH=src .venv/bin/python -m attempt2.openerSearch --games 400 --refine-games 4000

# the whole experiment: every policy on the held-out secrets, plus the figures
PYTHONPATH=src .venv/bin/python -m attempt2.experiment

# play: you type the first guess and the agent plays the rest
PYTHONPATH=src .venv/bin/python -m attempt2.play

# advise: the agent proposes each word, you give back the colours
PYTHONPATH=src .venv/bin/python -m attempt2.play --advise

# evaluate the agent against the baselines
PYTHONPATH=src .venv/bin/python -m attempt2.play --auto --games 400

# a different agent: --algo reinforce | dqn | ppo
PYTHONPATH=src .venv/bin/python -m attempt2.play --algo ppo

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
| `tests/test_attempt2Env.py` | The Gymnasium contract, the reward values, that training never samples a held-out secret, and that a forced secret is checked against the pool |
| `tests/test_attempt2Eval.py` | The harness counts wins, guesses and illegal actions correctly, and refuses the wrong secrets |
| `tests/test_attempt2Baselines.py` | Each baseline is what its name says |
| `tests/test_attempt2Policies.py` | An illegal action has no probability and no gradient; the greedy and sampling choices are legal |
| `tests/test_attempt2ReplayBuffer.py` | The buffer keeps the next mask, drops the oldest transitions, and reports its own sizes |
| `tests/test_attempt2Learning.py` | The policy head receives a gradient (the regression test for §9.6), DQN's target respects the next mask, PPO's advantages do not cross an episode |
| `tests/test_attempt2OpenerSearch.py` | The search is deterministic, independent of the worker count, and can tell a good opener from a bad one |
| `tests/test_attempt2Play.py` | The environment accepts the guesses the real game accepts and refuses the rest, and the advisor's suggestion depends only on the history |

---

## 13. The final results (Phase D)

`experiment.py` evaluates every policy — the baselines and the three trained agents — on
the same 400 held-out secrets, through the same harness, and writes the table, the three
figures and `results.json`. It never trains: the agents are loaded from `artifacts/`, so
the experiment can be re-run and re-plotted without repeating 27 minutes of training per
algorithm.

### 13.1 Every policy on the same 400 held-out secrets

| Policy | Win rate | Avg guesses (wins) | Avg guesses (all) | Opens with |
|---|---|---|---|---|
| masked random | 98.5% | 4.07 | 4.10 | random |
| masked random, fixed opener `crane` | 99.0% | 3.80 | 3.83 | `crane` |
| best opener `slate` + masked random | 99.0% | **3.77** | 3.79 | `slate` |
| **REINFORCE (learned)** | 97.8% | **3.85** | 3.90 | `actor` |
| Attempt-1 solver (model + rule mask) | 97.8% | 3.90 | 3.95 | `crane` |
| PPO (learned) | 98.8% | 4.01 | 4.04 | `slate` |
| DQN (learned) | 95.8% | 4.34 | 4.41 | `fight` |

**The honest summary:** one learned agent (REINFORCE) beats both the random baseline
(4.07) and the Attempt-1 solver (3.90), and none of them beats one hand-picked opening
word (3.77).

### 13.2 The finding that matters most: the generalisation gap

The learning curves say PPO improved a lot — its training average fell from 4.05 to 3.59.
Its held-out score says otherwise. The same policy, scored on three different sets of
secrets:

| Policy | training 400 | validation 200 | held-out 400 |
|---|---|---|---|
| masked random | 4.098 | 4.055 | 4.090 |
| **REINFORCE** | 3.995 | 3.850 | **3.895** |
| **PPO** | **3.590** | 3.985 | **4.037** |
| DQN | 4.128 | 4.330 | 4.412 |

Masked random is identical on all three sets (4.06–4.10), so the sets are equally hard.
The differences are the policies'. **PPO's 0.45-guess advantage on the training secrets
is almost entirely memorisation**, and on the held-out secrets it is no better than random.

**Why a policy can overfit when it never sees the secret.** The observation is the
feedback history — it does *not* contain the set of still-consistent candidates. But the
best next guess depends on that set, not on the feedback summary. Two different candidate
sets can produce the same feedback and want different guesses. The network cannot see the
difference, so the best it can do is learn the average over the candidate sets it
encountered — and PPO learned that average *confidently*, having 250,000 episodes to
memorise it. On the held-out secrets the candidate sets are different, the memorised
answer is uncorrelated with the right one, and a confident wrong guess costs more than a
random one would.

REINFORCE, with a weaker signal and no strong convergence, memorised less: a 0.10-guess
gap, and it transfers.

**This is the precise gap that an information-theoretic attempt closes.** A picker that
*computes* the candidate set from the feedback and picks the best guess for it has no such
gap: the candidate set is recoverable exactly, and the choice does not have to be
memorised. See `src/attempt3/attempt3.md`.

### 13.3 What each agent learned

- **REINFORCE** learned a good *strategy* and a mediocre opener. Its later play beats
  random legal guessing (3.68 against 3.77 with the same opener), but its opener `actor`
  is worth 0.22 guesses to it.
- **PPO** learned the best opener of all 2,315 (§10) and a *poor* strategy — its later
  greedy choices are worse than a random legal guess, which is what its overfitted
  confident policy predicts.
- **DQN** learned nothing useful. 2,315 actions is too many for a value-based method to
  generalise over from a 60-dimensional state, and it is the only policy below the random
  baseline on every set.

### 13.4 The figures

- `comparison.png` — every policy's average guesses on the held-out secrets (this is the
  one the README embeds).
- The three figures sit next to the code in `src/attempt2/` rather than in `artifacts/`,
  because the README has to link to a tracked file. Attempt 1 does the same with
  `accuracy.png`.
- `learning-curves.png` — each algorithm's training curve **with its validation points on
  the same axes**, because a falling training curve on its own is exactly the thing that
  hid PPO's overfitting.
- `first-guess.png` — the opening word each policy used over the held-out games.

---

## 14. Limitations and open questions

- **The agent cannot play a non-answer.** The action space is the 2,315 answers, so the agent
  can never use a legal probe word that is not a possible secret. This is a real restriction
  on the strategy it can learn, and it must be stated in the final results rather than
  discovered by a reader.
- **Win rate is not a usable headline.** See §8.1. The final table has to lead with average
  guesses and the first-guess distribution.
- **A uniform posterior still applies.** As in Attempt 1, every consistent candidate is
  equally likely a priori. What the agent can learn is which consistent candidate is *worth
  guessing* given what it already knows — a strategy, not a prediction.
- **The step penalty changes what "optimal" means.** −0.05 is a modelling choice, not a fact
  about Wordle. A larger penalty would reward different behaviour, and the report has to say
  which penalty produced which number.
- **The split is fixed at seed 42.** One split means one held-out set. Differences of a
  percentage point on 400 games are within noise; the report should treat them as such.
- **The strategy headroom is mostly not reachable from the reward alone.** Phase C
  measured why: openers differ by 0.46 guesses and a single game is noisy by about 1.0, so
  ranking them needs ~100 samples each and ~230,000 episodes to sweep the action space once.
  The agent beat the random baseline and Attempt 1, and it did not beat one hand-picked
  word. That is the honest ceiling of this attempt, and it is a statement about the signal,
  not about the implementation.
- **Greedy evaluation is not free.** PPO found the best opener in the sample and still
  scored worse than that opener played with random legal guesses afterwards, because its
  later turns were confidently wrong. A policy that is still learning should be judged
  against the rules, not against its own confidence.
- **The observation does not determine the candidate set.** This is the deepest limitation
  and it is measured (§13.2): the policy sees the feedback history, but the best next guess
  depends on which words are still consistent — information the observation does not
  contain. A learned policy can only learn the training distribution's average, and a
  confident policy over that average transfers badly (PPO: 3.59 on the training secrets,
  4.04 on the held-out ones — no better than random). The information-theoretic approach in
  Attempt 3 does not have this problem, because it computes the candidate set exactly.
- **One learned agent beats the baselines, and none beats one word.** REINFORCE's 3.85
  beats masked random (4.07) and the Attempt-1 solver (3.90); the hand-picked opener is
  3.77. That is the whole of Attempt 2's result.
