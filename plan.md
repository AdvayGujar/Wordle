# Wordle Solver — A Self-Guided Learning Plan

This is a plan for **you to build yourself**. It explains the goal, the math you'll
use, the steps to follow, and how to verify your work — but deliberately does **not**
write the code for you. Implement each step on your own and learn by doing.

> **Scope note:** For now we are **not** doing the information-gain / entropy approach.
> The ML path (predicting the answer word) is the focus. The entropy idea is noted as
> deferred at the end and can be revisited later.

---

## 1. Project overview

**Goal:** Build an AI that solves Wordle, in two attempts that grow in sophistication:

- **Attempt 1 (predefined models):** Learn how a supervised ML model works end-to-end.
  Train several scikit-learn models on the task *"predict the secret answer word from
  the game's feedback,"* and compare their predictions.
- **Attempt 2 (custom math):** Re-implement the **same task from scratch** in NumPy,
  using vector calculus from *Mathematics for Machine Learning* (Module 5: gradients,
  Jacobians, directional derivatives, chain rule). No scikit-learn / PyTorch.

The point of Attempt 2 is to *see the math from Module 5 doing real work*, rather than
calling a library that hides it.

### Wordle rules (recap)
- A secret 5-letter word is chosen.
- You guess a 5-letter word; the game colors each position:
  - **Green** — correct letter in the correct position.
  - **Yellow** — correct letter, wrong position.
  - **Gray** — letter not in the word (with a repeated-letter caveat).
- You get up to **6 guesses**.

Your solver must pick guesses that narrow down the possibilities until the secret is found.

### Your two stated aims (make sure the plan serves both)
1. "Use predefined models … get a basic understanding of how a model works, train various
   models, compare their predictions." → **Attempt 1.**
2. "Understand the math behind these models … make my own ML model using the math rather
   than predefined libraries." → **Attempt 2**, mapping directly onto Module 5.

---

## 2. The core idea: make "predicting the answer" a supervised problem

A raw Wordle round is *sequential* (guess → feedback → guess). To train a supervised
model we convert it into fixed **feature vectors** and **labels**:

- **Label `y`:** the secret answer word, as an index into your answer vocabulary.
- **Features `x`:** an encoding of the *evidence you have so far* — the colors you saw
  from your previous guesses.

The model learns the conditional distribution `P(answer | evidence)`. At solve time you
feed it the current evidence and it predicts the most likely answer. This is a genuine
multi-class classification problem (≈2,300 classes), which is why it's a good learning
vehicle.

**Why this is a good learning task**
- It's *real*: the model literally outputs an answer word.
- It's *scalable in difficulty*: with 1 probe the evidence is weak (low accuracy), with 3
  probes it's strong (high accuracy). You can plot accuracy vs. number of probes to *see*
  the model learning to use information.
- It needs a full ML pipeline: data generation → feature engineering → train/test split →
  training → evaluation → prediction comparison.

### Important mental model: `answers` vs `words`
- **`answers`** (2,315) = the words the game can *pick as the secret*. This is the model's
  **class space / prediction space.** The model is trained over `answers` and only ever
  predicts an answer word.
- **`words`** (~14,855) = every *legal guess* you can type. This is only needed for
  *searching over possible probes* (the deferred entropy approach). **The ML model does
  not need `words` to pick a guess** — it predicts from `answers`, and guessing an answer
  is always a legal move.

---

## 3. Progress tracker (current status)

| Area | Status |
|---|---|
| Environment (`.venv` + `requirements.txt`) | ✅ Done |
| Word-list data (`src/data/words.txt`, `src/data/answers.txt`) | ✅ Done |
| `src/wordle/feedback.py` | ✅ Done — repeated-letter bug fixed |
| `src/wordle/vocabulary.py` | ✅ Done — refactored (`Vocabulary` dataclass, camelCase) |
| `src/wordle/game.py` | ⬜ To do |
| `src/wordle/features.py` | ⬜ To do |
| `src/wordle/dataset.py` | ⬜ To do |
| Tests | ⬜ To do |
| Attempt 1 (predefined models) | ⬜ To do |
| Attempt 2 (from-scratch model) | ⬜ To do |

---

## 4. Environment setup — ✅ DONE

- Virtual environment created: `.venv`.
- Dependencies installed and pinned in `requirements.txt`:
  `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `pytest`.
- Your `.venv` is Python 3.9. **Note:** if you ever need a newer NumPy/scikit-learn that
  requires a later Python, recreate the venv with `python3.13` (verified installed).

---

## 5. Data — ✅ DONE

The real NYT-style word lists are downloaded and committed as:
- `src/data/words.txt` — full 5-letter vocabulary (answers + allowed guesses, **14,855**).
- `src/data/answers.txt` — the 2,315 possible *secret* words.

**Note on location:** your data lives at `src/data/` (not a top-level `data/`). That matches
how `vocabulary.py` resolves its path (`src/wordle` → parent → `data`). Keep them in sync.

**Optional follow-up:** you currently don't have a `download_wordlist.py` checked in. The
original sources (if you want to regenerate later):
- Full vocab: `https://raw.githubusercontent.com/tabatkins/wordle-list/main/words`
- Answers (stored as a Python list literal — parse with `ast.literal_eval`):
  `https://raw.githubusercontent.com/LaurentLessard/wordlesolver/main/solutions_nyt.txt`

---

## 6. Shared core — file layout & function-by-function breakdown

The shared core is a small package at `src/wordle/`. Each file has **one responsibility**.
Below is the complete contract for every function in every file, plus the current status.

```
src/wordle/
├── __init__.py    # package marker; optional re-exports
├── feedback.py    # game rules: colors, feedback(), candidate filtering   [✅ done]
├── vocabulary.py  # load + clean words.txt / answers.txt, word<->index    [✅ done]
├── game.py        # GameState + play_game() loop (solver driver)          [⬜ to do]
├── features.py    # evidence -> fixed-length feature vector                [⬜ to do]
├── dataset.py     # simulate games -> (X, y, vocab) + no-leakage split     [⬜ to do]
└── entropy.py     # DEFERRED — information-gain / probe picker             [⏸ deferred]
```

### 6.1 `feedback.py` — pure game rules (no I/O, no randomness) — ✅ done

The most important and most-tested file. Everything else depends on it being correct.

| Function | Purpose | Status |
|---|---|---|
| `feedback(guess, secret) -> tuple[int, ...]` | The core rule. Compares a guess against the true secret and returns 5 color codes (`0=gray`, `1=yellow`, `2=green`). Logic: **green pass first** (correct letter+position, "use up" that letter), **then yellow/gray pass** (only award yellow as many times as the letter genuinely remains in the secret). | ✅ correct |
| `candidates_matching(constraints: list[(guess, pattern)], vocab) -> list[str]` | Given recorded `(guess, pattern)` pairs and a vocabulary, return every word that could still be the secret. A candidate `w` is possible iff `feedback(guess, w) == pattern` for *all* constraints — reuse `feedback`. | ✅ correct |

**Fixed ✅** — the yellow pass now skips already-green positions (`if result[x] == 2: continue`),
so greens are never overwritten. Verified against these cases (tests should still be written
into `tests/`, see §9):
- `feedback("aaaba", "ababa") == (2, 0, 2, 2, 2)` — the original bug case
- `feedback("error", "shear") == (1, 0, 0, 0, 2)` — the double-`r` case
- `feedback("crane", "crate") == (2, 2, 2, 0, 2)`

### 6.2 `vocabulary.py` — loading words + word↔index mapping — ✅ done

| Function | Purpose | Status |
|---|---|---|
| `loadWords() -> tuple[str, ...]` | Reads `src/data/words.txt`, returns the full ~14,855-word vocabulary (all legal guesses). Cached (`@cache`). | ✅ done |
| `loadAnswers() -> tuple[str, ...]` | Reads `src/data/answers.txt`, returns the 2,315 possible *secrets*. Cached. | ✅ done |
| `class Vocabulary` (dataclass, `frozen=True`) | Holds `words` (tuple) and `index` (`dict`), with `indexOf(word)` and `wordAt(i)` for word↔index lookup. | ✅ done |
| `answersVocab() -> Vocabulary` | Returns the cached `Vocabulary` built over `loadAnswers()` — the **one** authoritative label vocab. Reuse it for both train and test. | ✅ done |

**Resolved ✅** — duplication removed (single `answersVocab()` over `answers`), and the
`Vocabulary` dataclass centralizes word↔index lookup.

**Remaining minor item (optional):** the cached `Vocabulary.index` dict is still mutable and
shared across calls. Access it only through `indexOf()` / `wordAt()` (never mutate `vocab.index`
directly) so a caller can't corrupt the shared cache.

**Naming convention:** this project deliberately uses **camelCase** (`loadWords`, `answersVocab`,
`indexOf`, `wordAt`). Keep it consistent across the package.

**The one rule to remember:** build the `word -> index` mapping **once**, over `loadAnswers()`,
and reuse it identically for train and test, so labels are stable.

### 6.3 `game.py` — the solve-time driver — ⬜ to do

| Function | Purpose |
|---|---|
| `class GameState` | A small mutable container holding history so far, e.g. `guesses: list[(guess, pattern)]`. Handed to the guess-picker so it knows the evidence. |
| `play_game(secret, guess_picker, max_guesses=6) -> int` | Plays a full round: loop — ask the picker for a guess given the state → compute feedback → append → stop if correct. Returns **guesses used** on a win, or `0` on a loss. The single reusable driver. |
| *(optional)* `GameState.add_guess(...)` / `is_solved(secret)` | Small helpers to keep `play_game` clean. |

**Key design choice:** `guess_picker` is *any* callable `(GameState) -> guess`. That seam is
where different solver strategies (ML model, random baseline) plug in later. Build the driver
to accept a picker, not hardcode a strategy. This file is lower priority than features/dataset
if you want to reach model training first.

### 6.4 `features.py` — evidence → model input — ⬜ to do

| Function | Purpose |
|---|---|
| `encode(probes: list[(guess, pattern)]) -> list[int]` | Convert the evidence into a **fixed-length** numeric vector. Simplest version: concatenate the 5 color codes per probe (1 probe → 5 features, 2 → 10, 3 → 15). |
| *(optional)* `letter_presence_features(probes) -> list[int]` | Adds a 26-long vector (a–z) marking whether each letter appeared in any probe. Richer signal; try later and compare accuracy. |
| *(optional)* `make_encoder(n_probes) -> Callable` | A factory returning a configured encoder with a known length, so feature length is consistent. |

**Keep it pure** — same input always gives the same output. No I/O, no randomness.

### 6.5 `dataset.py` — the one file with randomness — ⬜ to do

| Function | Purpose |
|---|---|
| `generate(probes: list[str], n_train, n_test, seed=None) -> Dataset` | The data factory. Splits secrets **by word** (no leakage), plays the probes against each secret, encodes features, returns train/test examples. Seeds the RNG for reproducibility. |
| *(helper)* `_play_probes(secret, probes) -> list[(guess, pattern)]` | Runs the probes against one secret, returns their evidence. |
| *(helper)* `_split_by_secret(answers, n_train, seed)` | Picks `n_train` distinct secrets for training; the rest are test. Guarantees `train ∩ test == ∅`. |
| `class Dataset` | Container holding `X_train, y_train, X_test, y_test, vocab` (words + word→index map) so callers get one clean object. |

**Critical guarantee:** the no-leakage split-by-secret logic lives *only* here, and a test
asserts it. Train examples come only from train-secrets; test examples only from test-secrets.

### 6.6 `entropy.py` — DEFERRED (not doing information gain for now)

Out of scope for the current milestone. If revisited later, it would hold:
`feedback_distribution(guess, candidates)`, `entropy_of(guess, candidates)`,
`best_opening(candidates, vocab)`, `best_probe(state, candidates, vocab)`. It would import
only from `feedback.py` and `vocabulary.py`. **Do not build this yet.**

### 6.7 `__init__.py` — package surface (optional)

Re-exports the public API so attempt scripts can `from wordle import feedback, play_game,
generate, load_answers` instead of importing each submodule. Pure convenience.

### Dependency graph & suggested build order

```
feedback.py   (leaf — nothing imports it except others)
vocabulary.py (leaf)
features.py   (leaf)
game.py       → uses feedback.py, vocabulary.py
dataset.py    → uses feedback.py, vocabulary.py, features.py
entropy.py    → uses feedback.py, vocabulary.py   [deferred]
__init__.py   → re-exports the above
```

**Build order:** ✅ `feedback.py` fixed → ✅ `vocabulary.py` tidied → `features.py` →
`game.py` → `dataset.py` (+ no-leakage test). Then Attempt 1.

---

## 7. Attempt 1 — predefined models (scikit-learn)

### 7.1 Task
Train classifiers: `features (evidence) → predicted answer word (index)` over `answers`.

### 7.2 Models to try (good spread to compare)
1. **Multinomial Logistic Regression** (linear, interpretable baseline).
2. **k-Nearest Neighbors** (distance-based; no training).
3. **Random Forest** (tree ensemble).
4. **Linear SVM** (margin-based) — optionally **RBF SVM**.
5. *Optional:* Gradient Boosting, or scikit-learn's `MLPClassifier`.

### 7.3 What to compare (this is the "train and compare predictions" goal)
- **Top-1 accuracy**, reported **per probe count** (1, 2, 3 probes) — accuracy should climb
  as evidence grows.
- A **comparison table** and a **bar/line plot**.
- Look at *where* models disagree (predicted answer differs) — a small disagreement table.
- A **confusion summary** for the best model.

### 7.4 Turn the classifier into a solver
The classifier alone outputs a word; to *play Wordle* you use it inside `play_game`:
1. Start with the opening probes (their feedback is your initial evidence).
2. Predict the most likely answer; **guess it** (an answer word — always legal).
3. Feed the new feedback back in as evidence; re-predict.
4. Repeat until correct or 6 guesses.

Measure **solve rate** (fraction of games won) and **average guesses**, compared against a
trivial baseline (e.g., always guess the same fixed word).

---

## 8. Attempt 2 — from-scratch model using vector calculus

Now rebuild the **same** attempt-1 task using only NumPy and your own math. This is where
Module 5 comes alive.

### 8.1 Model: multinomial logistic regression
- **Forward:** `logits = X · W + b`, then `P = softmax(logits)`, where
  `softmax(z)_k = exp(z_k) / Σ_j exp(z_j)`.
- **Loss:** cross-entropy, `L = −(1/N) Σ_n log(P_{n,y_n})`.

### 8.2 Derive the gradient by hand (this is the vector-calculus core)
Work through, on paper, why the gradient of `L` w.r.t. the weight matrix is

```
∇_W L = Xᵀ (P − Y) / N
```

where `Y` is the one-hot label matrix. To get there you will use:
- **Chain rule** (composite function: softmax ∘ linear map).
- **Jacobian of softmax** — `∂P_i / ∂z_j` has a tidy closed form
  (`P_i(δ_ij − P_j)`), which collapses to the elegant `P − Y` form for cross-entropy.
- **Gradient as a matrix** (your weight matrix `W` is a `features × classes` matrix, so
  the gradient is a matrix of partial derivatives).

Implement the gradient **explicitly** (no autodiff). Numerically check it with
**finite differences** — perturb `W` and confirm `(L(W+ε) − L(W)) / ε ≈ ∇_W L`.

### 8.3 Optimize with gradient descent
- Update `W ← W − lr · ∇_W L` over epochs/mini-batches.
- Plot **loss vs. iteration**; expect it to fall smoothly.

### 8.4 Module-5 convergence check (directional derivative)
Before trusting gradient descent, verify the math that *justifies* it:
- The **directional derivative** of `L` at `W` in direction `d` is `∇_W L · d`.
- The first-order (Taylor) prediction of the loss change after a small step is
  `ΔL ≈ lr · (∇_W L · d)` with `d = −∇_W L`.
- Check empirically that (a) the directional derivative along `−∇_W L` is **negative**
  (going downhill) and (b) the actual observed `ΔL` tracks the first-order prediction for
  small step sizes. This is Module 5's point that **the gradient is the direction of
  steepest ascent**, so `−gradient` is the best descent direction.

### 8.5 Optional stretch: a 1-hidden-layer network (richer Jacobian practice)
Extend to an MLP: `h = activation(X·W1+b1)`, `logits = h·W2+b2`. Backprop becomes a fuller
**chain rule / Jacobian** exercise. Compare to the plain softmax model.

### 8.6 Compare to Attempt 1
Run your from-scratch model on the **same test split and probe counts** and report its
accuracy beside the scikit-learn models. Small differences are expected; explain them.

---

## 9. Testing and acceptance checklist

- `pytest` passes for: `feedback()` (including repeated letters), feature-vector shapes,
  data-pipeline counts, no-leakage split assertion.
- Both `train_compare` scripts run end-to-end and produce a comparison table + plots.
- Attempt-2 accuracy reported side-by-side with Attempt 1.
- Final answer: a short write-up comparing all models and *what you learned from each*.

---

## 10. Edge cases and failure modes to anticipate

- **Repeated letters in feedback** — get this exactly right; it's the most common bug.
  Now fixed in `feedback.py` (green-overwrite guard). Lock it in with tests.
- **Leakage** — split by whole secrets, not by individual rows.
- **≈2315 classes** — fine on CPU with a few thousand samples, but if training is slow,
  reduce the sample size or the candidate set; don't change the test set logic.
- **Ambiguous evidence** — with 1 probe many answers share a pattern, so accuracy is
  naturally capped. Report per-probe accuracy instead of chasing impossible perfection.
- **Determinism** — seed RNGs so results are reproducible.
- **Vocab consistency** — build the `word -> index` map once over `loadAnswers()` and reuse it
  for both train and test, so labels stay stable.

---

## 11. Suggested implementation order (for you)

1. Environment + `requirements.txt` — ✅ done.
2. Word-list data — ✅ done (`src/data/`).
3. Fix `feedback.py` (repeated-letter bug) + write tests — ✅ bug fixed; tests still to add.
4. Tidy `vocabulary.py` (dedupe, answers-based vocab, avoid shared mutable dict) — ✅ done.
5. `features.py`.
6. `game.py` (optional to defer until the solver loop).
7. `dataset.py` + no-leakage split test.
8. Attempt 1: train & compare scikit-learn models; solver loop; report.
9. Attempt 2: from-scratch softmax + gradient (finite-difference check) +
   directional-derivative convergence check; optional MLP; comparison report.
10. Update `README.md` with the roadmap and your findings.

---

## 12. Learning checkpoints — you'll know it's working when…

- [ ] `feedback()` passes a repeated-letter test you wrote yourself (fix the current bug).
- [ ] Accuracy on the test set **increases** as you go from 1 → 2 → 3 probes.
- [ ] Your hand-written `∇_W L = Xᵀ(P − Y)/N` matches a finite-difference check.
- [ ] The loss curve descends and the directional derivative along `−∇` is negative.
- [ ] Your from-scratch model's accuracy is in the same ballpark as scikit-learn's.
- [ ] You can explain, in one paragraph each: what a gradient is, what a Jacobian is,
      why `−∇` is the steepest-descent direction, and how the chain rule produces backprop.

Build each piece yourself, run it, and only move on when the checkpoint at that step holds.
That is the whole point of this project — the learning, not the finished code.

---

## 13. Deferred (not doing now)

- **Information-gain / entropy baseline** — the rule-based "pick the guess with max expected
  information" approach. Not in scope for the current milestone. If revisited later, its
  natural home is `src/wordle/entropy.py`, and it would use the full `words` list to search
  over possible probes and provide a rival solver to compare against the ML approach.
