# Wordle Solver — A Self-Guided Learning Plan

This is a plan for **you to build yourself**. It explains the goal, the math you'll
use, the steps to follow, and how to verify your work — but deliberately does **not**
write the code for you. Implement each step on your own and learn by doing.

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

---

## 3. Environment setup

- Use a virtual environment, e.g. `python -m venv .venv`.
- Install: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `pytest`.
- Save the pinned versions to `requirements.txt` after a successful install
  (`pip freeze > requirements.txt`).
- **Note (verified on this machine):** your default `python3` is 3.14. scikit-learn /
  scipy wheels exist for it, but if install fails, fall back to `python3.13` — several
  are already installed (`python3.11`…`python3.14`). Don't spend time fighting installs;
  pick the newest version where `pip install scikit-learn` succeeds.

---

## 4. Data: where to get real Wordle word lists

Use the real NYT Wordle lists for fidelity. Both sources below were verified reachable
and give clean lowercase 5-letter words.

- **Full vocabulary** (`words`, includes answers + allowed guesses, **14,855 words**):
  `https://raw.githubusercontent.com/tabatkins/wordle-list/main/words`
- **Answer list** (the ~2,309 words the game actually picks from):
  `https://raw.githubusercontent.com/LaurentLessard/wordlesolver/main/solutions_nyt.txt`
  — note this file is stored as a **Python list literal** (a giant comma-separated,
  quoted, mostly-single-line blob), so parse it with `ast.literal_eval` and strip quotes,
  keeping only 5-letter lowercase entries.

**Your job:** write `download_wordlist.py` that fetches both, cleans them, and writes
`data/words.txt` and `data/answers.txt`. Commit the data files so later steps work
offline. Verify: `answers` ⊂ `words`, all length 5, all lowercase.

---

## 5. Shared core (build this first)

Write a small package (e.g. `src/wordle/`) with three pieces:

### 5.1 Game engine
- `feedback(guess, secret) -> tuple[int, ...]` returning per-position color codes, e.g.
  `0 = gray`, `1 = yellow`, `2 = green`. **Handle the repeated-letter rule carefully:**
  if a letter appears twice in your guess but once in the secret, only one position can be
  yellow and the other is gray. A common bug is miscounting duplicates — write tests for it.
- A function that, given a list of `(guess, pattern)` constraints, returns the set of
  candidate words still consistent with all feedback (filter the vocabulary).
- A `game_loop(guess_picker)` helper that plays a full round to solve-rate evaluation.

### 5.2 Feature encoding
Design a fixed-length feature vector for a game state, e.g. concatenate, for each probe,
the 5 per-position color codes (so 1 probe → 5 features, 2 probes → 10, …). You may add
per-letter presence features. The vector must be fixed-length so the model can train on it.

### 5.3 Dataset generation
- Sample secrets uniformly from `answers.txt`.
- Play a fixed number of **probes** (informative opening guesses — see §6) against each
  secret to produce the evidence.
- Output `X` (features) and `y` (answer index).
- **Train/test split by whole secrets — no leakage.** A secret that appears in training
  must not appear in testing, or your accuracy will be artificially high. Assert this.

---

## 6. The entropy baseline (plan.md "Phase 1") — your reference and probe source

`plan.md` originally framed a Phase-1 solver using **information theory (entropy)**. Keep
it, but as a *reference benchmark* and a *source of informative probes*, not the main focus.

- Compute, for a candidate guess, the **expected information** (entropy) it would reveal:
  simulate the feedback pattern of that guess against every possible answer, count how
  many answers fall into each pattern bucket, and compute
  `H = −Σ p_bucket · log₂(p_bucket)`. High entropy ⇒ the guess splits candidates well.
- Use a high-entropy opening guess (e.g. evaluate all words once and pick the max) as the
  fixed probe(s) for dataset generation.
- This solver also gives you a **baseline solve rate** to compare your ML solver against.

> Learning note: this is "which guess shrinks the candidate set fastest." Attempt 1's ML
> models instead *learn* which answer is most likely given evidence. Both are legitimate;
> comparing them is instructive.

---

## 7. Attempt 1 — predefined models (scikit-learn)

### 7.1 Task
Train classifiers: `features (evidence) → predicted answer word (index)`.

### 7.2 Models to try (good spread to compare)
1. **Multinomial Logistic Regression** (linear, interpretable baseline).
2. **k-Nearest Neighbors** (distance-based; no training).
3. **Random Forest** (tree ensemble).
4. **Linear SVM** (margin-based) — optionally **RBF SVM**.
5. *Optional:* Gradient Boosting, or scikit-learn's `MLPClassifier`.

### 7.3 What to compare (this is the "train and compare predictions" goal)
- **Top-1 accuracy**, reported **per probe count** (1, 2, 3 probes) — you should see
  accuracy climb as evidence grows.
- A **comparison table** and a **bar/line plot**.
- Look at *where* models disagree (their predicted answer differs) — a small
  disagreement table is very instructive.
- A **confusion summary** for the best model.

### 7.4 Turn the classifier into a solver
The classifier alone outputs a word; to *play Wordle* you use it inside a loop:
1. Start with the opening probes (their feedback is your initial evidence).
2. Predict the most likely answer; **guess it**.
3. Feed the new feedback back in as evidence; re-predict.
4. Repeat until correct or 6 guesses.

Measure **solve rate** (fraction of games won) and **average guesses**, and compare to the
entropy baseline (§6) and to a trivial baseline (e.g., always guess the same fixed word).

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
**finite differences** — perturb `W` and confirm `(L(W+ε) − L(W)) / ε ≈ ∇_W L` — this is a
rigorous, satisfying way to prove your derivation is right.

### 8.3 Optimize with gradient descent
- Update `W ← W − lr · ∇_W L` over epochs/mini-batches.
- Plot **loss vs. iteration**; expect it to fall smoothly.

### 8.4 Module-5 convergence check (directional derivative)
Before trusting gradient descent, verify the math that *justifies* it:
- Recall the **directional derivative** of `L` at `W` in direction `d` is `∇_W L · d`
  (inner product of gradient with the direction).
- The first-order (Taylor) prediction of the loss change after a small step is
  `ΔL ≈ lr · (∇_W L · d)` with `d = −∇_W L`.
- Check empirically that (a) the directional derivative along `−∇_W L` is **negative**
  (you're going downhill) and (b) the actual observed `ΔL` tracks the first-order
  prediction for small step sizes. This is exactly Module 5's point that **the gradient is
  the direction of steepest ascent**, so `−gradient` is the best descent direction.

### 8.5 Optional stretch: a 1-hidden-layer network (richer Jacobian practice)
Extend to an MLP: `h = activation(X·W1+b1)`, `logits = h·W2+b2`. Backprop becomes a fuller
**chain rule / Jacobian** exercise. Compare it to the plain softmax model.

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
- **Leakage** — split by whole secrets, not by individual rows.
- **≈2309 classes** — fine on CPU with a few thousand samples, but if training is slow,
  reduce the sample size or the candidate set; don't change the test set logic.
- **Ambiguous evidence** — with 1 probe many answers share a pattern, so accuracy is
  naturally capped. Report per-probe accuracy instead of chasing impossible perfection.
- **Determinism** — seed RNGs so results are reproducible.
- **Install issues** — prefer `python3.13`+ if 3.14 wheels are missing (see §3).

---

## 11. Suggested implementation order (for you)

1. Environment + `requirements.txt`.
2. `data/download_wordlist.py` → commit `data/words.txt`, `data/answers.txt`.
3. `src/wordle/`: game engine + `feedback()` + tests.
4. Entropy baseline (probe picker + reference solver).
5. `src/wordle/`: features + dataset generation + no-leakage split.
6. Attempt 1: train & compare scikit-learn models; solver loop; report.
7. Attempt 2: from-scratch softmax + gradient (with finite-difference check) +
   directional-derivative convergence check; optional MLP; comparison report.
8. Update `README.md` with the roadmap and your findings.

---

## 12. Learning checkpoints — you'll know it's working when…

- [ ] `feedback()` passes a repeated-letter test you wrote yourself.
- [ ] Accuracy on the test set **increases** as you go from 1 → 2 → 3 probes.
- [ ] Your hand-written `∇_W L = Xᵀ(P − Y)/N` matches a finite-difference check.
- [ ] The loss curve descends and the directional derivative along `−∇` is negative.
- [ ] Your from-scratch model's accuracy is in the same ballpark as scikit-learn's.
- [ ] You can explain, in one paragraph each: what a gradient is, what a Jacobian is,
      why `−∇` is the steepest-descent direction, and how the chain rule produces backprop.

Build each piece yourself, run it, and only move on when the checkpoint at that step holds.
That is the whole point of this project — the learning, not the finished code.
