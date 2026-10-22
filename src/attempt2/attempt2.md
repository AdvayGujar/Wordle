# Attempt 2 — reinforcement learning (the agent learns a strategy on its own)

**Status:** 🟡 **Phases A and B complete** (environment, split, observation, action mask,
evaluation harness, baselines). Phases C–D (the learning algorithms, and the experiment +
results) are planned but not built. This file records what Attempt 2 is trying to do, what
has been built so far, and every design choice behind it. It will be extended with the
agent's results as the phases land.

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

**Files (planned)**

| Piece | Purpose |
|---|---|
| `experiment.py` | Runs everything and produces the comparison tables and plots |
| `attempt2.md` | This file |

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
- **PPO found the best opener in the sample and still lost to it.** It chose `slate`
  (3.795 with random play) and scored 4.01. Its *later* greedy choices are worse than a
  random pick among the legal answers. A confidently wrong policy loses to a random legal
  guess, and evaluating greedily is not a free win for a partially-trained agent.
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

## 10. How to run and test

```bash
# the whole suite (295 tests: 196 pre-existing + 99 new)
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

---

## 11. Phase D — what comes next

| Phase | Modules | The question it answers |
|---|---|---|
| **D** | `experiment.py`, this file | What did it learn on its own, and where does it fall short? |

Phase C cleared part of the bar set in §7.2 — REINFORCE beat masked random (4.07) and the
Attempt-1 solver (3.90) — and missed the rest: the hand-picked `crane` opener is still
3.80, and no agent found an opener better than it. Phase D turns all of this into the final
tables and plots, and states the limits of the result.

---

## 12. Limitations and open questions

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
- **Phase D is the open risk.** The environment is fast and correct and the algorithms
  train, but the final tables have to state plainly that the learned agent ties a
  hand-picked opener rather than beating it.
