# Wordle Solver — Supervised Candidate-Consistency

A Wordle solver and a supervised-learning experiment built around reusable, tested game
rules. The project asks a learnable question:

> **Given the evidence so far, is this candidate word consistent with every round?**

A model scores all 2,315 possible answers with that binary classifier, and the solver uses those
scores plus the exact feedback rules to play.

## Why this framing

The obvious formulation — 2,315-class classification of the secret word — does not work:

1. A held-out secret is an unseen class; the model can never output it. The measured test accuracy
   was exactly 0%.
2. With a uniformly chosen secret and deterministic feedback, every feedback-consistent candidate
   is equally likely. A classifier can learn the rules but cannot rank the true secret above another
   consistent candidate.

Attempt 1 was therefore reframed to **candidate-consistency scoring**. The full reasoning, data,
features, model comparison and limitations are in
[`src/attempt1/attempt1.md`](src/attempt1/attempt1.md).

## Results

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

The learned model recovers the consistency rule, but it does **not** beat the rule-based candidate
filter. That is expected: the posterior is uniform over consistent candidates. The remaining
headroom is strategy, not classification.

![Consistency skill vs rounds](src/attempt1/accuracy.png)

## Repository layout

```
src/
├── data/                 # words.txt (14,855 legal guesses), answers.txt (2,315 secrets)
├── wordle/               # shared core
│   ├── feedback.py       # feedback(), candidatesMatching()
│   ├── vocabulary.py     # cached word lists + stable word <-> index mapping
│   ├── game.py           # GameState, checkGuess, playGameDetailed
│   ├── features.py       # legacy encode, encodeRounds, encodeLetterState
│   ├── dataset.py        # generate, generateVariable, generateClassComplete, evidence helpers
│   └── core.md           # module-by-module core reference
└── attempt1/
    ├── features.py       # 250 candidate-relative features (no label leakage)
    ├── common.py         # data rows, model registry, metrics, artifact persistence
    ├── trainCompare.py   # baselines, training, tables, plot, analysis
    ├── play.py           # interactive / --auto solver
    └── attempt1.md       # full Attempt-1 write-up
tests/                    # 162 tests
```

## Setup

Python 3.9 virtual environment with the pinned stack in `requirements.txt`:

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

Run the full Attempt-1 experiment (step-3 encoding checks, baselines, four models, comparison
tables, plot, state-level analysis):

```bash
PYTHONPATH=src .venv/bin/python src/attempt1/trainCompare.py
```

Play interactively — you type the first guess, the model plays the rest:

```bash
PYTHONPATH=src .venv/bin/python src/attempt1/play.py
```

Evaluate the solver and baselines over 400 games:

```bash
PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --games 400
```

Test the model without the rule mask (the honest check of what it learned):

```bash
PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --model-only
```

`trainCompare.py` writes `model.joblib`, `results.json`, `accuracy.png` and, when the solver is
run, `playResults.json`. The trained model and result files are generated artifacts and are not
tracked.

## Core guarantees

- `feedback()` is pure and handles repeated letters with a green-first, yellow-second pass.
- `candidatesMatching(constraints, loadAnswers())` returns every word consistent with all
  evidence.
- All imports are package-absolute; run scripts with `PYTHONPATH=src`.
- Naming is camelCase throughout.
- The legacy colour-only `encode` is untouched; the letter-aware `encodeRounds` /
  `encodeLetterState` sit beside it.
- `Dataset.width` derives the expected row width from the stored recipe, and `Dataset.split`
  records whether secrets or evidence were held out.

See [`src/wordle/core.md`](src/wordle/core.md) for the full core contract.
