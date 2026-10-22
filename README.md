# Wordle Solver — four attempts at a Wordle solver

A Wordle solver built around reusable, tested game rules, and a series of attempts to
solve the game by different means. The project asks a learnable question:

> **Given the evidence so far, is this candidate word consistent with every round?**

and then asks a harder one:

> **Having narrowed the field with the rules, which word should be guessed next?**

| Attempt | What it is | Status |
|---|---|---|
| 1 | supervised candidate-consistency scoring | ✅ [`src/attempt1/attempt1.md`](src/attempt1/attempt1.md) |
| 2 | reinforcement learning — the agent learns a strategy | ✅ [`src/attempt2/attempt2.md`](src/attempt2/attempt2.md) |
| 3 | information theory — *compute* the strategy | 📋 [`src/attempt3/attempt3.md`](src/attempt3/attempt3.md) |
| 4 | the Module-5 math, written from scratch in NumPy | 📋 `plan.md` §10 |

---

## Why the first framing did not work

The obvious formulation — 2,315-class classification of the secret word — does not work:

1. A held-out secret is an unseen class; the model can never output it. The measured test accuracy
   was exactly 0%.
2. With a uniformly chosen secret and deterministic feedback, every feedback-consistent candidate
   is equally likely. A classifier can learn the rules but cannot rank the true secret above another
   consistent candidate.

Attempt 1 was therefore reframed to **candidate-consistency scoring**: a binary model that scores
each of the 2,315 answers with "is this candidate consistent with every round?", which the solver
then combines with the exact feedback rules.

That model recovers the rule, and still cannot beat the rule-based candidate filter — because the
posterior is uniform over consistent candidates. The remaining headroom is **strategy, not
classification**, and that is what Attempts 2–4 are about.

---

## Results

### Attempt 1 — supervised candidate-consistency

Consistency model on held-out candidate pairs, rounds 1–5:

| Model | Mean F1 | Mean AUC |
|---|---|---|
| Histogram Gradient Boosting | **0.977** | **1.000** |
| Random Forest | 0.358 | 0.978 |
| Logistic Regression | 0.426 | 0.646 |
| Linear SVM | 0.419 | 0.646 |

Solver over 400 held-out games, fixed opener `crane`:

| Picker | Win rate | Avg guesses (wins) |
|---|---|---|
| model + rule mask | 97.8% | 3.91 |
| model-only (no mask) | 95.5% | 3.90 |
| random-consistent | 98.2% | 4.06 |
| fixed-opener + random-consistent | 97.5% | 3.83 |
| random-legal / fixed-word | 0.0% | — |

![Consistency skill vs rounds](src/attempt1/accuracy.png)

### Attempt 2 — reinforcement learning

A Gymnasium `Env` with legal-action masking, a fixed-width observation, a secret split that
keeps 400 answers held out, a reward of +1 for a win minus 0.05 per guess, one measurement
harness, the baseline policies, and three algorithms (REINFORCE, DQN, PPO) trained in PyTorch.

On the 400 held-out secrets, measured by the same harness:

| Policy | Win rate | Avg guesses (wins) |
|---|---|---|
| best opener `slate` + masked random | 99.0% | **3.77** |
| **REINFORCE (250k episodes)** | 97.8% | **3.85** |
| Attempt-1 solver (model + rule mask) | 97.8% | 3.90 |
| PPO (150k episodes) | 98.8% | 4.01 |
| masked random | 98.5% | 4.07 |
| DQN (150k episodes) | 95.8% | 4.34 |

So the agent **beats the random baseline and beats Attempt 1**, but it does not beat one
hand-picked opening word.

An exhaustive search over **all 2,315 openers** (1.2M simulated games) settles what "best"
means: `slate` at 3.779 guesses, against a median opener's 4.050 and the worst (`jazzy`)
at 4.605. **PPO converged onto `slate` — the top of that ranking — from reward alone.**
Neither REINFORCE (`actor`) nor DQN (`fight`) came close.

Two things explain the attempt:
- 144 of the 2,315 openers are within 0.05 guesses of the best, so the good openers are a
  large, tightly packed group rather than a single peak.
- The best openers are all built from the same common letters (18 of the top 20 contain
  `a`, 18 contain `e`; six of them are permutations of the same five letters). The
  information in an opener is in *which common letters it tests*, not their arrangement.

**The most important result is the generalisation gap.** Scored on three sets of secrets
(masked random is identical on all three, so the sets are equally hard):

| Policy | training 400 | validation 200 | held-out 400 |
|---|---|---|---|
| PPO | 3.590 | 3.985 | 4.037 |
| REINFORCE | 3.995 | 3.850 | 3.895 |
| DQN | 4.128 | 4.330 | 4.412 |

PPO's 0.45-guess advantage on the training secrets is memorisation, and it is no better
than random on the held-out ones. The reason is structural: the observation is the feedback
history, but the best next guess depends on the set of still-consistent candidates, which
the observation does not contain — so a learned policy can only memorise the training
distribution's average.

### Attempts 3 and 4 — planned

**Attempt 3 (`src/attempt3/attempt3.md`) is the information-theoretic attempt**: compute
the strategy instead of learning it. It carries the mathematics (the pattern distribution,
the mutual information a guess carries, the concave expected-guesses objective), every
measurement from Attempt 2 that bears on it, and a four-phase build plan.

**Attempt 4 (`plan.md` §10) is the from-scratch math attempt** — the Module-5 one: a binary
candidate-consistency logistic regression written in NumPy, with a hand-derived gradient
`Xᵀ(p − y)/N`, a finite-difference check, a directional-derivative check, and gradient
descent. **NumPy only: no scikit-learn, no PyTorch.** It was originally Attempt 3 and was
not changed in any way — only renumbered and moved from `src/attempt3/` to `src/attempt4/`,
because Attempt 2 measured that the opening-word decision is better computed than learned.

---

## Repository layout

```
src/
├── data/                 # words.txt (14,855 legal guesses), answers.txt (2,315 secrets)
├── wordle/               # shared core
│   ├── feedback.py       # feedback(), candidatesMatching()
│   ├── feedbackTable.py  # cached 2315x2315 feedback table + fast candidate filter
│   ├── vocabulary.py     # cached word lists + stable word <-> index mapping
│   ├── game.py           # GameState, checkGuess, playGameDetailed
│   ├── features.py       # legacy encode, encodeRounds, encodeLetterState
│   ├── dataset.py        # generate, generateVariable, generateClassComplete, evidence helpers
│   └── core.md           # module-by-module core reference
├── attempt1/
│   ├── features.py       # 250 candidate-relative features (no label leakage)
│   ├── common.py         # data rows, model registry, metrics, artifact persistence
│   ├── trainCompare.py   # baselines, training, tables, plot, analysis
│   ├── play.py           # interactive / --advise / --auto solver
│   └── attempt1.md       # full Attempt-1 write-up
├── attempt2/
│   ├── config.py         # every Attempt-2 constant (seed, split, reward, observation)
│   ├── splits.py         # train / validation / held-out secrets, with a no-leakage assertion
│   ├── observation.py    # evidence -> one fixed-width vector
│   ├── actionMask.py     # which of the 2,315 answers are still legal to guess
│   ├── wordleEnv.py      # the Gymnasium environment
│   ├── eval.py           # the one measurement harness every policy is scored with
│   ├── baselines.py      # the policies the learned agent has to beat
│   ├── smoke.py          # end-to-end smoke test
│   ├── policies.py       # the networks, and masked action selection
│   ├── rollout.py        # the episode loop all three algorithms share
│   ├── replayBuffer.py   # the replay buffer DQN learns from
│   ├── reinforce.py      # REINFORCE with a value baseline
│   ├── dqn.py            # masked DQN with a target network
│   ├── ppo.py            # clipped-surrogate PPO
│   ├── train.py          # the training CLI
│   ├── openerSearch.py   # exhaustive search for the best opening word
│   ├── experiment.py     # the final table and the figures
│   ├── play.py           # the play modes: interactive, --advise, --auto
│   └── attempt2.md       # full Attempt-2 write-up
└── attempt3/
    └── attempt3.md       # the information-theoretic attempt: the math and the plan
tests/                    # 316 tests
```

## Setup

Python 3.9 virtual environment with the pinned stack in `requirements.txt`, which includes the
Attempt-2 RL stack (`torch`, `gymnasium`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The data lives at `src/data/`; `vocabulary.py` resolves paths relative to the package.

## Usage

Run the tests:

```bash
.venv/bin/python -m pytest tests/ -q
```

### Attempt 1

```bash
# the full experiment: step-3 encoding checks, baselines, four models, tables, plot
PYTHONPATH=src .venv/bin/python src/attempt1/trainCompare.py

# play interactively - you type guess 1, the model plays the rest
PYTHONPATH=src .venv/bin/python src/attempt1/play.py

# ask for the next best word while you play - you enter each guess and its colours
PYTHONPATH=src .venv/bin/python src/attempt1/play.py --advise

# evaluate the solver and baselines over 400 games
PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --games 400

# test the model without the rule mask (the honest check of what it learned)
PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --model-only
```

Colours can be entered as `GYXXG`, `2 1 0 0 2`, or `green yellow grey grey green`.

### Attempt 2

```bash
# play: you type the first guess, the trained agent plays the rest
PYTHONPATH=src .venv/bin/python -m attempt2.play

# advise: the agent proposes each word, you give back the colours
PYTHONPATH=src .venv/bin/python -m attempt2.play --advise

# evaluate the agent against the baselines on the 400 held-out secrets
PYTHONPATH=src .venv/bin/python -m attempt2.play --auto --games 400

# a different agent: --algo reinforce | dqn | ppo
PYTHONPATH=src .venv/bin/python -m attempt2.play --algo ppo

# the baseline table on the held-out secrets
PYTHONPATH=src .venv/bin/python -m attempt2.baselines

# the whole experiment: every policy, plus the three figures
PYTHONPATH=src .venv/bin/python -m attempt2.experiment

# train an agent (about 27 minutes for 250k episodes)
PYTHONPATH=src .venv/bin/python -m attempt2.train --algo reinforce --episodes 250000 \
    --learning-rate 0.01 --entropy-start 0.2 --entropy-end 0.01

# search all 2,315 openers for the best one (about 1.2 million games)
PYTHONPATH=src .venv/bin/python -m attempt2.openerSearch --games 400 --refine-games 4000
```

The play modes need a trained agent in `src/attempt2/artifacts/`; if it is missing they say so
and print the command to train one.

`trainCompare.py` writes `model.joblib`, `results.json`, `accuracy.png` and, when the solver is
run, `playResults.json`. `attempt2.train` writes `<algo>-model.pt`, `<algo>-curve.csv` and
`<algo>-results.json`, and `attempt2.experiment` writes `comparison.png`,
`learning-curves.png`, `first-guess.png` and `experiment-results.json`. All of them are
generated artifacts and are not tracked.

## Core guarantees

- `feedback()` is pure and handles repeated letters with a green-first, yellow-second pass.
- `candidatesMatching(constraints, loadAnswers())` returns every word consistent with all
  evidence.
- `feedbackTable()` is a pure accelerator: the table is cross-checked against `feedback()`, and
  `candidatesMatchingFast()` returns exactly what `candidatesMatching()` returns. The original
  pure-Python functions are unchanged.
- An illegal action is impossible, not unlikely: the mask is applied to the logits before the
  softmax, so it has probability zero and no gradient.
- The held-out secrets are held out in code, not in comments: the environment refuses a secret
  from the wrong pool, and the evaluation harness refuses to score a policy on the wrong set.
- All imports are package-absolute; run scripts with `PYTHONPATH=src`.
- Naming is camelCase throughout.
- The legacy colour-only `encode` is untouched; the letter-aware `encodeRounds` /
  `encodeLetterState` sit beside it.
- `Dataset.width` derives the expected row width from the stored recipe, and `Dataset.split`
  records whether secrets or evidence were held out.

See [`src/wordle/core.md`](src/wordle/core.md) for the full core contract.
