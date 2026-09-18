# Wordle Solver — A Self-Guided Learning Plan

This is a plan for **you to build yourself**. It explains the goal, the math you'll
use, the steps to follow, and how to verify your work — but deliberately does **not**
write the code for you. Implement each step on your own and learn by doing.

> **Scope note:** For now we are **not** doing the information-gain / entropy approach.
> The ML path (predicting the answer word) is the focus. The entropy idea is noted as
> deferred at the end and can be revisited later.

---

## 0. 📍 WHERE TO RESUME — read this first

**Status as of the last session: the entire shared core is finished and tested, and the ML
stack (§4) is installed and pinned. Nothing is blocked.**
Attempt 1 has **not** started — it is the next step (§7).

### ✅ Done
| Piece | Notes |
|---|---|
| `src/data/` | `words.txt` (14,855) + `answers.txt` (2,315) |
| `src/wordle/__init__.py` | empty package marker |
| `src/wordle/feedback.py` | `feedback()`, `candidatesMatching()` — repeated-letter bug fixed |
| `src/wordle/vocabulary.py` | `loadWords`, `loadAnswers`, `Vocabulary`, `answersVocab` |
| `src/wordle/features.py` | `encode(guesses, probes)` — color-only, `-1` padding |
| `src/wordle/game.py` | `GameState` + `playGame(secret, gamePicker, maxGuesses=6)` |
| `src/wordle/dataset.py` | `generate()` + `Dataset` + no-leakage split |
| `tests/` | **77 tests, all passing** |

### ✅ No blockers
§4 is complete — the ML stack is installed, pinned, and verified. Attempt 1 can start now.

### ▶️ Next steps, in order
1. **Attempt 1 (§7)** — full roadmap in **§7.4**, file layout in §7.2, pitfalls §7.7,
   deliverables checklist §7.8. Suggested new files: `src/attempt1/trainCompare.py`,
   `src/attempt1/solve.py`.
   - Headline: generate one dataset per probe count (1/2/3, same seed) → baseline → train the
     models → comparison table + accuracy plot → disagreement table → solver via the picker
     (§7.6) → solve rate vs. the fixed-word baseline.
2. **Attempt 2 (§8)** — full math (preserved) + roadmap in **§8.7**, concept map in §8.11,
   pitfalls §8.8. Suggested new files: `src/attempt2/softmaxModel.py`, `src/attempt2/checks.py`.
   - Headline: implement softmax + cross-entropy + the hand-derived gradient
     `∇_W L = Xᵀ(P − Y)/N` → **verify with finite differences (§8.4) before training** →
     directional-derivative check (§8.6) → gradient descent + loss curve → compare to Attempt 1
     (§8.10).

### ⚠️ Known decisions to respect (details in §6)
- Imports are **package-absolute** (`from wordle.feedback import feedback`); run with
  `PYTHONPATH=src`, or just `.venv/bin/python -m pytest tests/` for tests.
- **camelCase** naming throughout (`encode`, `playGame`, `nTrain`, …).
- `encode` is **color-only** — valid only because probes are **fixed** and identical at train
  and solve time.
- The picker receives the **raw guess list**, not a `GameState`.

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
| Environment (`.venv` + `requirements.txt`) | ✅ Done — venv (Python 3.9.6) with the full ML stack installed & pinned (§4) |
| Word-list data (`src/data/words.txt`, `src/data/answers.txt`) | ✅ Done |
| `src/wordle/feedback.py` | ✅ Done — repeated-letter bug fixed |
| `src/wordle/vocabulary.py` | ✅ Done — refactored (`Vocabulary` dataclass, camelCase) |
| `src/wordle/game.py` | ✅ Done — reviewed & verified; both bugs fixed (§6.3) |
| `src/wordle/__init__.py` | ✅ Done — empty package marker (§6.7) |
| `src/wordle/features.py` | ✅ Done — reviewed & tested (§6.4) |
| `src/wordle/dataset.py` | ✅ Done — reviewed & tested; 2 bugs fixed (§6.5) |
| Tests | ✅ Done — 77 tests in `tests/`, all passing (§9) |
| Attempt 1 (predefined models) | ⬜ To do — **next up** (§7) |
| Attempt 2 (from-scratch model) | ⬜ To do |

---

## 4. Environment setup — ✅ DONE

- Virtual environment exists: `.venv` (Python 3.9.6).
- ✅ **The ML stack is installed and pinned** in `requirements.txt` (27 pinned packages):
  `numpy 2.0.2`, `scipy 1.13.1`, `scikit-learn 1.6.1`, `matplotlib 3.9.4`, `pandas 2.3.3`,
  plus their transitive deps (`joblib`, `threadpoolctl`, `pillow`, `contourpy`, `fonttools`, …).
- ✅ `pytest 8.4.2` is installed and pinned too, so `tests/` runs.
- ✅ `pip` is now **26.0.1** (upgraded from 21.2.4), so the old-pip wheel concern is gone.
- **Verified end-to-end:** all five libraries import; `generate(('crane','slate'), nTrain=300,
  nTest=100, seed=42)` produces the expected `(300, 10)` / `(100, 10)` feature matrices
  (`nProbes × 5 = 10`); `LogisticRegression` fits and scores on them; and the full test
  suite reports **77 passed** on this stack.
- **Python-version note (resolved):** the venv is Python 3.9.6, and the versions pinned above
  are the newest releases that still support 3.9. That is a coherent, working set, so
  recreating the venv with `python3.13` (verified installed) is **optional**, not required.

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
├── __init__.py    # package marker (empty)                                 [✅ done]
├── feedback.py    # game rules: colors, feedback(), candidate filtering   [✅ done]
├── vocabulary.py  # load + clean words.txt / answers.txt, word<->index    [✅ done]
├── game.py        # GameState + playGame() loop (solver driver)              [✅ done]
├── features.py    # evidence -> fixed-length feature vector               [✅ done]
├── dataset.py     # simulate games -> (X, y, vocab) + no-leakage split     [✅ done]
└── entropy.py     # DEFERRED — information-gain / probe picker             [⏸ deferred]
```

### 6.1 `feedback.py` — pure game rules (no I/O, no randomness) — ✅ done

The most important and most-tested file. Everything else depends on it being correct.

| Function | Purpose | Status |
|---|---|---|
| `feedback(guess, secret) -> tuple[int, ...]` | The core rule. Compares a guess against the true secret and returns 5 color codes (`0=gray`, `1=yellow`, `2=green`). Logic: **green pass first** (correct letter+position, "use up" that letter), **then yellow/gray pass** (only award yellow as many times as the letter genuinely remains in the secret). | ✅ correct |
| `candidatesMatching(constraints, vocab) -> list[str]` | Given recorded `(guess, pattern)` pairs and a vocabulary, return every word that could still be the secret. A candidate `w` is possible iff `feedback(guess, w) == pattern` for *all* constraints — reuse `feedback`. | ✅ correct |

**Fixed ✅** — the yellow pass now skips already-green positions (`if result[x] == 2: continue`),
so greens are never overwritten. Locked in by `tests/test_feedback.py` (see §9):
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

### 6.3 `game.py` — the solve-time driver — ✅ done (implementation reviewed)

**Your actual implementation:** `GameState` holds a private `_guesses` list exposed via a
`guesses` property; `addGuess(guess, pattern)` appends; `isSolved(secret)` returns whether the
last guess equals the secret; `playGame(secret, gamePicker, maxGuesses=6)` runs the loop and
returns guesses used on a win or `0` on a loss.

| Function | Purpose | Status |
|---|---|---|
| `class GameState` | Mutable container holding history: `_guesses: list[(guess, pattern)]`, exposed via the `guesses` property. Handed to the picker so it knows the evidence. | ✅ done |
| `GameState.addGuess(guess, pattern)` | Appends a `(guess, pattern)` pair to the history. | ✅ done |
| `GameState.isSolved(secret) -> bool` | `True` if the most recent guess equals `secret`. Safe on an empty history (returns `False`). | ✅ done |
| `playGame(secret, gamePicker, maxGuesses=6) -> int` | Loop: ask picker for a guess → compute `feedback()` → append → stop if solved. Returns **guesses used** on a win, `0` on a loss. | ✅ done |

**Design decision you made (differs from the original plan):** your picker signature is
`gamePicker(gameState.guesses)` — it receives the **raw guess list**, not the whole
`GameState`. That's a perfectly valid, simpler seam. Keep it consistent with any future
features/dataset code that also needs the evidence.

**Bugs — both FIXED ✅ (verified by running):**
1. ✅ **`__innit__` typo** — now `__init__`; `_guesses` initializes and `addGuess()` works.
2. ✅ **`isSolved` empty-history `IndexError`** — the length check now runs *before* the
   unpack (`if len(self._guesses) > 0 and self._guesses[-1][0] == secret`), so an empty
   history returns `False` instead of crashing. Verified:
   - `playGame` win → returns guesses used (e.g. `2`)
   - `playGame` loss → returns `0`
   - `GameState().isSolved("apple")` → `False` (no crash)

**Key design choice:** `gamePicker` is *any* callable `(guesses) -> guess`. That seam is where
different solver strategies (ML model, random baseline) plug in later. The driver accepts a
picker and never hardcodes a strategy. Covered by `tests/test_game.py`.

### 6.4 `features.py` — evidence → model input — ✅ done (reviewed & tested)

**Your actual implementation:** a single pure `encode(gameState, probes)` where the first
argument is your list of `(guess, pattern)` pairs (e.g. `GameState.guesses`) and `probes` is
the fixed number of evidence rounds the model always expects. It concatenates each probe's 5
color codes, then pads with `-1` up to `probes * 5`.

| Function | Purpose | Status |
|---|---|---|
| `encode(gameState, probes) -> list[int]` | Concatenates the 5 color codes per probe and **pads with `-1`** until the output is `probes * 5` long. Color-only (letters dropped) — assumes **fixed opening probes**. | ✅ done |

**Test results (verified by running):**
- 1 probe, `probes=1` → `[2,0,1,0,0]` ✅
- 2 probes, `probes=3` → 10 real colors + 5 `-1`s = 15 ✅
- 0 probes, `probes=2` → ten `-1`s ✅
- 3 probes, `probes=3` → 15 real colors, no pad ✅

**Notes / optional follow-ups:**
1. ⚠️ **No upper cap** — if more evidence arrives than `probes`, the output *exceeds* the
   fixed length (it pads up but never truncates). Fine while evidence ≤ `probes`; revisit if
   the solver re-predicts over many rounds.
2. **Color-only baseline** — because it drops *which letters* were colored, it only works if
   the probes are fixed across games. The optional letter-aware encodings (`letterPresenceFeatures`,
   position-wise `(letter, color)`) fix this — try later and compare accuracy.
3. The optional `makeEncoder(n_probes)` factory isn't built yet; not required to proceed.

**Keep it pure** — same input always gives the same output. No I/O, no randomness.

### 6.5 `dataset.py` — the one file with randomness — ✅ done (reviewed & tested)

**Your actual implementation:** `generate(probes, nTrain, nTest, seed=None)` splits the secrets
by word, replays the probes against each secret, encodes the evidence, and returns a `Dataset`
of `(xTrain, yTrain, xTest, yTest, vocab)`.

| Function | Purpose | Status |
|---|---|---|
| `class Dataset` | Dataclass holding `xTrain, yTrain, xTest, yTest, vocab`. | ✅ done |
| `_playProbes(secret, probes)` | Runs each probe against one secret; returns `[(probe, pattern), ...]`. Pure. | ✅ done |
| `_splitBySecret(answers, nTrain, seed)` | Shuffles a **copy** with a seeded local `Random`, then slices `[:nTrain]` / `[nTrain:]`. Disjointness is automatic. Raises `ValueError` if `nTrain` isn't in `(0, len(answers))`. | ✅ done |
| `generate(probes, nTrain, nTest, seed=None) -> Dataset` | Splits, trims test to `nTest`, builds both `(x, y)` lists, returns the `Dataset`. | ✅ done |

**Two bugs found and FIXED during review:**
1. ✅ The test loop iterated `trainSecrets` instead of `testSecrets` — this was the exact
   leakage failure §10 warns about (test rows would have come from train secrets).
2. ✅ `generate` was missing its `return Dataset(...)` (returned `None`).

**Verified by running (`probes=["crane","slate"]`, nTrain=1000, nTest=200, seed=42):**
- `xTrain/yTrain` = 1000 each; `xTest/yTest` = 200 each ✅
- every row has **10 features** (2 probes × 5), train and test ✅
- **leakage overlap = 0** ✅
- same seed → identical data; different seed → different data ✅
- every row re-derives correctly from its secret (checked against an independent split) ✅
- labels all within `0..2314` ✅
- guard raises `ValueError: nTrain must be in (0, 2315)` ✅

**Optional follow-ups (not blockers):**
1. `Dataset` doesn't store `probes` — adding it would make a saved dataset self-describing
   (the solver must replay the *same* probes).
2. `repr(Dataset)` prints the entire 2,315-word vocab (very noisy in notebooks/REPL). Marking
   the field `vocab: object = field(repr=False)` would fix that.
3. ✅ Tests now exist — see `tests/test_dataset.py` (§9), including the no-leakage assertion.

**Critical guarantee:** the no-leakage split-by-secret logic lives *only* here. Train examples
come only from train-secrets; test examples only from test-secrets.

### 6.6 `entropy.py` — DEFERRED (not doing information gain for now)

Out of scope for the current milestone. If revisited later, it would hold:
`feedback_distribution(guess, candidates)`, `entropy_of(guess, candidates)`,
`best_opening(candidates, vocab)`, `best_probe(state, candidates, vocab)`. It would import
only from `feedback.py` and `vocabulary.py`. **Do not build this yet.**

### 6.7 `__init__.py` — package marker — ✅ created (empty)

Created as an **empty file** to mark `src/wordle/` as an importable package, so tests and
training scripts can do `from wordle.game import playGame` instead of relying on the current
working directory.

**Import convention (decided):** submodules use **package-absolute** imports, e.g.
`from wordle.feedback import feedback` (not bare `from feedback import ...`, not relative
`from .feedback import ...`).

**How to run code (important):** put `src` on the path, e.g.
```bash
PYTHONPATH=src python3 -c "from wordle.game import playGame"
```
or just run `.venv/bin/python -m pytest tests/ -q` from the repo root — `tests/conftest.py`
puts `src` on the path, so pytest needs no `PYTHONPATH`. **Consequence:** running a lone script
*from inside* `src/wordle/` (`cd src/wordle && python3 game.py`) no longer resolves — package
usage is the supported path now. A `pyproject.toml` + `pip install -e .` later would remove the
need for `PYTHONPATH` entirely.

*(Optional, not done yet: re-export the public API so scripts can use
`from wordle import feedback, playGame, encode`.)*

### Dependency graph & suggested build order

```
feedback.py   (leaf — nothing imports it except others)
vocabulary.py (leaf)
features.py   (leaf)
game.py       → uses feedback.py, vocabulary.py
dataset.py    → uses feedback.py, vocabulary.py, features.py
entropy.py    → uses feedback.py, vocabulary.py   [deferred]
__init__.py   → package marker (empty; no re-exports yet)
```

**Build order:** ✅ `feedback.py` fixed → ✅ `vocabulary.py` tidied → ✅ `game.py` built
(both bugs fixed) → ✅ `features.py` built & tested → ✅ `dataset.py` built & tested
(2 bugs fixed) → ✅ `tests/` written (77 passing) →
**next: Attempt 1 (§7) — the ML deps (§4) are ✅ installed and pinned.**

---

## 7. Attempt 1 — predefined models (scikit-learn)

### 7.1 Task
Train classifiers: `features (evidence) → predicted answer word (index)` over `answers`.

### 7.2 Setup
- ✅ ML stack installed and pinned (§4): `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`.
- Suggested file layout — **new** files; nothing inside `src/wordle/` changes:
  ```
  src/attempt1/
  ├── trainCompare.py   # build datasets, train models, print table + plots
  └── solve.py          # the picker + solve-rate harness
  ```
  Run with `PYTHONPATH=src .venv/bin/python src/attempt1/trainCompare.py`.

### 7.3 Models to try (good spread to compare)
1. **Multinomial Logistic Regression** (linear, interpretable baseline).
2. **k-Nearest Neighbors** (distance-based; no training).
3. **Random Forest** (tree ensemble).
4. **Linear SVM** (margin-based) — optionally **RBF SVM**.
5. *Optional:* Gradient Boosting, or scikit-learn's `MLPClassifier`.

### 7.4 Step-by-step roadmap

1. **Choose the probes.** Start with 2 fixed probes, e.g. `PROBES = ("crane", "slate")`.
   Define them **once** and reuse for training *and* solving.
2. **Generate one dataset per probe count.** To measure accuracy vs. evidence you need three
   datasets — 1, 2 and 3 probes — **sharing the same seed** so the secrets line up:
   ```python
   data = {n: generate(PROBES[:n], nTrain=1500, nTest=400, seed=42) for n in (1, 2, 3)}
   ```
   Keep `nTest` identical across them; only the probe count changes.
3. **Convert to NumPy.** scikit-learn wants arrays, not lists:
   ```python
   X = np.array(ds.xTrain); y = np.array(ds.yTrain)
   ```
   Assert the shape: `X.shape[1] == nProbes * 5`. A silent length mismatch is the #1 way this
   pipeline breaks.
4. **Establish a baseline *before* any model.** Compute the majority-class accuracy and a
   uniform-random guess (≈ 1/2315 ≈ 0.04%). Without this you cannot tell whether a model
   learned anything real.
5. **Train each model** on `(X_train, y_train)`, then score with `model.score(X_test, y_test)`.
6. **Build the comparison table** — rows = models, columns = 1 / 2 / 3 probes, cells = top-1
   accuracy. Format it with `pandas.DataFrame` so it's readable.
7. **Plot it** — probe count on the x-axis, accuracy on the y-axis, one series per model.
   Save to `src/attempt1/accuracy.png`. The expected shape is **accuracy rising with probes** —
   that's the checkpoint in §12.
8. **Inspect disagreements** — pick test rows where model A and model B predict different
   answers and print `(evidence, modelA_pred, modelB_pred, truth)`. This is the literal
   "compare their predictions" goal, and where you learn *why* the models differ.
9. **Confusion summary for the best model.** With 2,315 classes a full confusion matrix is
   unreadable — instead report **top-5 accuracy** (how often the truth is in the model's top 5)
   and the most common wrong predictions.
10. **Turn it into a solver** (§7.6) and measure **solve rate** + **average guesses** across a
    few hundred test secrets, against the fixed-word baseline.

### 7.5 What to compare (this is the "train and compare predictions" goal)
- **Top-1 accuracy**, reported **per probe count** (1, 2, 3 probes) — accuracy should climb
  as evidence grows.
- A **comparison table** and a **bar/line plot**.
- Look at *where* models disagree (predicted answer differs) — a small disagreement table.
- A **confusion summary** for the best model.

### 7.6 Turn the classifier into a solver
The classifier alone outputs a word; to *play Wordle* you plug it into `playGame` as a
**picker** — `playGame(secret, picker)` calls `picker(guesses)` once per turn.

The picker is a small **phase state machine** keyed on `len(guesses)`:
1. **Probe phase** (`len(guesses) < nProbes`) — return the fixed probes in order.
2. **Model phase** — encode the probe evidence and predict:
   `encode(guesses[:nProbes], nProbes)`.

⚠️ **The `guesses[:nProbes]` slice is required.** Without it, once the model's *own* guess is
appended the vector grows past the trained length (10 → 15 features) and the model rejects it.
Slicing keeps the feature length constant; later guesses are used only for filtering
candidates, not fed to the model. (This is the "no upper cap" caveat noted in §6.4.)

Also avoid re-guessing a word already played — rank with `predict_proba` and take the best
*unused* class (Random Forest and Logistic Regression support it; SVM needs `probability=True`).

Measure **solve rate** (fraction of games won) and **average guesses**, compared against a
trivial baseline (e.g., always guess the same fixed word).

### 7.7 Pitfalls to expect
- **Feature length must match.** Train and solve must use the same `nProbes`; the
  `guesses[:nProbes]` slice (§7.6) is what guarantees it.
- **2,315 classes is a lot.** k-NN and SVM get slow — start around ~1,500 training rows and
  raise it only if needed. Don't shrink the *test* set.
- **1-probe accuracy is inherently low.** Many answers share a colour pattern, so the ceiling
  is real. Report per-probe numbers rather than chasing impossible perfection.
- **A model trained on 10 features cannot score 15-feature input.** Always pair each dataset
  with its own model — never mix probe counts.
- **Consider scaling for k-NN / SVM.** Colour codes are small integers, but `StandardScaler`
  often helps distance-based models; try both and compare.
- **Fix `random_state=` on every estimator** so your tables don't change between runs.
- **`predict_proba` isn't universal.** SVM needs `probability=True` (slower) or you must fall
  back to `decision_function` for ranking.

### 7.8 Deliverables checklist
- [x] ML stack installed and pinned in `requirements.txt` (§4)
- [ ] `trainCompare.py` runs end-to-end
- [ ] Baseline numbers (majority class / random)
- [ ] Comparison table (models × probe counts)
- [ ] Accuracy plot saved
- [ ] Disagreement table
- [ ] Confusion summary / top-5 for the best model
- [ ] `solve.py` reports solve rate + average guesses vs. the fixed-word baseline

---

## 8. Attempt 2 — from-scratch model using vector calculus

Now rebuild the **same** attempt-1 task using only NumPy and your own math. This is where
Module 5 comes alive. **Rule for this attempt: `numpy` only — no scikit-learn, no PyTorch.**
Everything scikit-learn did for you (softmax, cross-entropy, the gradient, the optimizer) you
write by hand.

Suggested layout — new files; `src/wordle/` is untouched:
```
src/attempt2/
├── softmaxModel.py   # softmax, loss, gradient, training loop
└── checks.py         # finite-difference + directional-derivative verification
```

### 8.1 Setup: reuse the *same* data as Attempt 1
Use `generate()` with the **same probes, same seed, same sizes** so the two attempts are
directly comparable (§8.10):
```python
from wordle.dataset import generate
ds = generate(PROBES, nTrain=1500, nTest=400, seed=42)

X = np.array(ds.xTrain, dtype=float)   # (N, D)  D = nProbes * 5
y = np.array(ds.yTrain, dtype=int)     # (N,)    values 0..2314
```
Then one-hot encode the labels:
```python
K = len(ds.vocab.words)                # 2315 classes
Y = np.eye(K)[y]                       # (N, K)
```
Use **float64** throughout — the finite-difference check depends on it.

### 8.2 Model: multinomial logistic regression
- **Forward:** `logits = X · W + b`, then `P = softmax(logits)`, where
  `softmax(z)_k = exp(z_k) / Σ_j exp(z_j)`.
- **Loss:** cross-entropy, `L = −(1/N) Σ_n log(P_{n,y_n})`.

**Shapes to keep straight:**

| Thing | Shape |
|---|---|
| `X` | `(N, D)` — N samples, D = `nProbes × 5` features (10 for 2 probes) |
| `W` | `(D, K)` — weights |
| `b` | `(K,)` — bias |
| `logits`, `P`, `Y` | `(N, K)` |
| `∇_W L` | `(D, K)` — same shape as `W` |
| `∇_b L` | `(K,)` — same shape as `b` |

**Numerical stability (do this or you'll get `nan`):** `exp` of a large logit overflows. Subtract
the row max before exponentiating — it cancels in the ratio and leaves the result unchanged:
```python
z = logits - logits.max(axis=1, keepdims=True)
P = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
```

### 8.3 Derive the gradient by hand (this is the vector-calculus core)
Work through, on paper, why the gradient of `L` w.r.t. the weight matrix is

```
∇_W L = Xᵀ (P − Y) / N
∇_b L = Σₙ (P − Y) / N          (sum over the N rows, giving shape (K,))
```

where `Y` is the one-hot label matrix. To get there you will use:
- **Chain rule** (composite function: softmax ∘ linear map).
- **Jacobian of softmax** — `∂P_i / ∂z_j` has a tidy closed form
  (`P_i(δ_ij − P_j)`), which collapses to the elegant `P − Y` form for cross-entropy.
- **Gradient as a matrix** (your weight matrix `W` is a `features × classes` matrix, so
  the gradient is a matrix of partial derivatives).

Two independent routes, both worth doing:

- **Route A (fast, one line):** `L = −(1/N) Σ_n log(P[n, y_n])`; the `P − Y` form falls out.
- **Route B (brute force, instructive):** differentiate `−log P_{n,c}` w.r.t. every `z_{n,j}`,
  apply the softmax Jacobian, then chain through `z = XW + b`. You should land on the same
  `P − Y`. Doing B *once* is how you convince yourself A isn't magic. This is also the entire
  gradient-derivation checkpoint in §12.

### 8.4 Numerical gradient check (finite differences) — do this BEFORE training
Implement the gradient **explicitly** (no autodiff). Never trust it until it matches a
numerical estimate. Use the **central difference** (more accurate than forward):

```
∂L/∂θ_i ≈ ( L(θ + ε·e_i) − L(θ − ε·e_i) ) / (2ε)
```

```python
def gradCheck(lossFn, gradW, W, b, X, Y, eps=1e-5, nSamples=10):
    gW, gb = gradW(W, b, X, Y)          # analytic
    for _ in range(nSamples):           # check a few random entries, not all 10k+
        i, j = np.random.randint(W.shape[0]), np.random.randint(W.shape[1])
        orig = W[i, j]
        W[i, j] = orig + eps; lp = lossFn(W, b, X, Y)
        W[i, j] = orig - eps; lm = lossFn(W, b, X, Y)
        W[i, j] = orig
        num = (lp - lm) / (2 * eps)
        rel = abs(num - gW[i, j]) / (abs(num) + abs(gW[i, j]) + 1e-12)
        print(i, j, num, gW[i, j], rel)
```

**Pass criterion:** the **relative** error `‖g_num − g_ana‖ / (‖g_num‖ + ‖g_ana‖ + 1e-12)`
should be about `1e-6`–`1e-7` for `ε ≈ 1e-5` in float64. A relative error near 1 means the
derivation is wrong; a tiny-but-not-tiny error usually means the wrong `ε` or float32.
Remember to check **`b` too**, not just `W`.

### 8.5 Optimize with gradient descent
- Update `W ← W − lr · ∇_W L` (and `b ← b − lr · ∇_b L`) over epochs/mini-batches.
- Initialise `W` with small random values (e.g. `0.01 * np.random.randn(D, K)`) and `b` at zeros.
  Starting `W` at zero makes every class symmetric, so the gradient never breaks the tie.
- Plot **loss vs. iteration**; expect it to fall smoothly. A loss that jumps to `nan` means the
  learning rate is too high (or the softmax stability trick is missing).

### 8.6 Module-5 convergence check (directional derivative)
Before trusting gradient descent, verify the math that *justifies* it:
- The **directional derivative** of `L` at `W` in direction `d` is `∇_W L · d` (flatten both to
  vectors and take the dot product).
- The first-order (Taylor) prediction of the loss change after a small step is
  `ΔL ≈ lr · (∇_W L · d)` with `d = −∇_W L`.
- Check empirically that (a) the directional derivative along `−∇_W L` is **negative**
  (going downhill) and (b) the actual observed `ΔL` tracks the first-order prediction for
  small step sizes. This is Module 5's point that **the gradient is the direction of
  steepest ascent**, so `−gradient` is the best descent direction.

Practical test: for `lr` in `[1e-4, 1e-3, 1e-2]`, compute the predicted `ΔL` and the observed
`L(W − lr·g) − L(W)`, then print the ratio `observed / predicted`. It should approach **1** as
`lr → 0`. Deviation at large `lr` is the second-order term showing up — that's the lesson.

### 8.7 Step-by-step roadmap
1. **Implement `softmax`** (stable version) and `crossEntropy(P, Y)`.
2. **Implement `forward`** — `logits = X @ W + b` → `P`.
3. **Implement `gradient`** by hand, returning `(∇_W L, ∇_b L)`.
4. **Verify with §8.4** — relative error ≈ 1e-6. *Do not proceed until this passes.*
5. **Verify with §8.6** — directional derivative along `−∇` is negative and `ΔL` tracks the
   prediction.
6. **Train** with mini-batch gradient descent; record loss per epoch; plot the curve.
7. **Evaluate** top-1 (and top-5) accuracy on the same test split, per probe count.
8. **Compare to Attempt 1** (§8.10) and explain the gap.

### 8.8 Pitfalls to expect
- **`exp` overflow → `nan`.** Fixed by the max-subtraction in §8.2. This is the single most
  common failure.
- **Wrong `ε`.** Too large → truncation error; too small → floating-point noise. `1e-5` in
  float64 is the sweet spot. Never use float32 for the check.
- **Forgetting the `/N`.** Your analytic and numerical gradients will disagree by exactly `N`
  unless both include the mean. Be consistent about mean vs. sum.
- **Checking only `W`.** The bias gradient is a separate derivation — check it too.
- **Zero initialisation.** Symmetry means it never learns. Use small random `W`.
- **Learning rate.** Too high → divergent/`nan` loss; too low → looks like nothing happens.
- **2,315 classes × 10 features** is ~23k parameters — small enough for plain NumPy, but the
  full 1,500×10 by 10×2315 matmul is fine. Mini-batches keep memory and speed sane.
- **Accuracy should rise with probe count** exactly as in Attempt 1 — if it doesn't, suspect a
  data/shape bug rather than the math.

### 8.9 Optional stretch: a 1-hidden-layer network (richer Jacobian practice)
Extend to an MLP: `h = activation(X·W1+b1)`, `logits = h·W2+b2`. Backprop becomes a fuller
**chain rule / Jacobian** exercise. Compare to the plain softmax model.

### 8.10 Compare to Attempt 1
Run your from-scratch model on the **same test split and probe counts** and report its
accuracy beside the scikit-learn models. Small differences are expected; explain them.
Note that scikit-learn's `LogisticRegression` is solving the *same* optimization problem with a
better optimizer and possibly regularisation, so it is a fair, informative yardstick: if your
hand-written gradient descent lands in the same ballpark, your math is right.

### 8.11 Module-5 concept map (what each piece is *for*)
| Module-5 idea | Where it shows up here |
|---|---|
| Composite functions | `L = crossEntropy(softmax(XW + b))` — three composed maps |
| Chain rule | Splits the gradient into `∂L/∂P · ∂P/∂z · ∂z/∂W` |
| Jacobian | `∂P_i/∂z_j = P_i(δ_ij − P_j)` — the softmax Jacobian |
| Gradient as a matrix | `∇_W L` is `(D, K)`; a matrix of partial derivatives |
| Directional derivative | `∇L · d` — the convergence check in §8.6 |
| Steepest ascent | Why `−∇L` is the best descent direction (§8.6) |
| Backpropagation (stretch) | The chain rule composed layer-by-layer in the MLP (§8.9) |

---

## 9. Testing and acceptance checklist

### ✅ Test suite written — `tests/` (77 tests, all passing)

Run from the repo root with:
```bash
.venv/bin/python -m pytest tests/ -q
```
`tests/conftest.py` puts `src/` on `sys.path`, so no `PYTHONPATH` is needed.

| File | Tests | Covers |
|---|---|---|
| `tests/test_feedback.py` | 15 | the 3 documented cases from §6.1; repeated-letter rules (green-before-yellow, surplus letters gray, greens never downgraded); `candidatesMatching` |
| `tests/test_features.py` | 10 | exact length = `probes × 5`; `-1` padding; padding ≠ gray; purity; input not mutated |
| `tests/test_game.py` | 17 | `GameState` + `playGame`; the probes-then-model phase flow; **regression tests for both §6.3 bugs** |
| `tests/test_dataset.py` | 24 | counts; uniform feature length; **no-leakage assertions**; determinism; every row re-derives from its secret; `_splitBySecret` guard |
| `tests/test_vocabulary.py` | 11 | list sizes (14,855 / 2,315); caching; word↔index round-trip |

**Verified by mutation testing** — the suite genuinely catches the bugs it targets:
- Reintroducing the §6.5 leakage bug (`testSecrets` → `trainSecrets`) → **3 tests fail**,
  including `testNoSecretAppearsInBothTrainAndTest`.
- Removing the green-skip guard in `feedback.py` (§6.1 original bug) → **2 tests fail**.

### Remaining acceptance items
- [ ] `src/attempt1/trainCompare.py` runs end-to-end → comparison table + accuracy plot
      (roadmap §7.4, deliverables §7.8).
- [ ] `src/attempt1/solve.py` reports solve rate + average guesses vs. the baseline (§7.6).
- [ ] Attempt-2 accuracy reported side-by-side with Attempt 1 (§8.10).
- [ ] Final answer: a short write-up comparing all models and *what you learned from each*.

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

1. Environment + `requirements.txt` — ✅ done: venv, `pytest`, and the full ML stack installed & pinned (§4).
2. Word-list data — ✅ done (`src/data/`).
3. Fix `feedback.py` (repeated-letter bug) + write tests — ✅ bug fixed; ✅ tests written (§9).
4. Tidy `vocabulary.py` (dedupe, answers-based vocab, avoid shared mutable dict) — ✅ done.
5. `features.py` — ✅ built & tested (`encode`: color-only + `-1` padding; §6.4).
6. `game.py` — ✅ built & verified (`GameState` + `playGame`); both bugs fixed (§6.3).
7. `dataset.py` — ✅ built & tested (`generate`, `_playProbes`, `_splitBySecret`; 2 bugs fixed, §6.5).
8. Tests — ✅ 77 tests written & passing, incl. the no-leakage assertion (§9).
9. **Attempt 1** — ML stack ✅ installed (§4); follow the roadmap (§7.4): datasets per probe
   count → baseline → train & compare models → solver loop via the picker (§7.6) → report (§7.8).
10. **Attempt 2** — from-scratch softmax: implement → finite-difference check (§8.4) →
    directional-derivative check (§8.6) → gradient descent → compare to Attempt 1 (§8.10);
    optional MLP (§8.9).
11. Update `README.md` with the roadmap and your findings.

---

## 12. Learning checkpoints — you'll know it's working when…

- [x] `feedback()` passes a repeated-letter test you wrote yourself — **done**, `tests/test_feedback.py` (§9).
- [ ] Accuracy on the test set **increases** as you go from 1 → 2 → 3 probes.
- [ ] Your hand-written `∇_W L = Xᵀ(P − Y)/N` matches a finite-difference check (relative
      error ≈ 1e-6) — procedure in §8.4.
- [ ] The loss curve descends and the directional derivative along `−∇` is negative — §8.6.
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
