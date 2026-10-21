# The `wordle` core, explained

This document explains **what `src/wordle/` does, why it is built that way, and what every
caller may rely on**. It is the reference for Attempt 1 and later; if code and this document
disagree, the code is the bug and one of them must be fixed in the same change.

The core has **one job**: turn the rules of Wordle into small, pure, testable pieces that
training scripts and solvers can reuse. It contains no ML code, no plotting, and no CLI.

```
src/wordle/
├── __init__.py    # package marker
├── feedback.py    # the rules: colors, candidate filtering
├── feedbackTable.py # cached 2315x2315 feedback codes + fast filtering
├── vocabulary.py  # word lists + word <-> index mapping
├── game.py        # GameState, checkGuess, the play loop
├── features.py    # evidence -> numeric feature vector
└── dataset.py     # simulate games -> (X, y) + no-leakage split
```

---

## 1. The one-page mental model

```
src/data/words.txt   ──> loadWords()    (14,855 legal guesses)
src/data/answers.txt ──> loadAnswers()  (2,315 possible secrets)
                              │
                              └──> answersVocab()  ──> word <-> label index
                                                          │
secret ──> feedback(guess, secret) ──> (guess, pattern) ──┤
                                                          │
                                          evidence list   │
                                                          v
                              encodeRounds(evidence) ──> feature row x
                                                          │
                              generateVariable(...) ──────┘
                                       │
                                       v
                                  (X, y) Dataset
                                       │
                                       v
                                  model.fit / predict
                                       │
                                       v
             playGameDetailed(secret, picker)  <── picker(history) ── model
```

The two vocabularies are deliberately different:

- **`answers` (2,315)** is the **label space**. The model predicts one of these indices.
- **`words` (14,855)** is the **legal-guess space**. It is used by `checkGuess`, not by the
  model. Guessing an answer word is always legal.

Three data types travel through the core:

| Name | Shape / type | Meaning |
|---|---|---|
| evidence | `list[tuple[str, tuple[int, ...]]]` | `(guess, pattern)` pairs in play order |
| pattern | `tuple[int, int, int, int, int]` | one colour per position: `0` grey, `1` yellow, `2` green |
| feature row | `list[int]` | numbers a model can consume; layout defined by the encoder |

---

## 2. `feedback.py` — the rules

### 2.1 `feedback(guess, secret) -> tuple[int, ...]`

This is the most important function in the project. Every other module either calls it or
depends on its correctness.

Wordle's repeated-letter rule is the only subtle part. The implementation uses **two passes**:

1. **Green pass.** Walk positions left to right. If `guess[x] == secret[x]`, write `2` and
   decrement that letter's remaining count in a frequency table of the secret.
2. **Yellow pass.** Walk positions again. Skip positions already marked green. If the guessed
   letter still has a positive remaining count, write `1` and decrement it. Otherwise leave `0`.

Why the order matters: a green match has already "used up" one copy of that letter in the
secret, so a duplicate copy in the guess cannot also be yellow. Doing yellow first would let a
duplicate steal a copy that belongs to a later green.

Worked examples (all locked in by `tests/test_feedback.py`):

```
feedback("aaaba", "ababa") == (2, 0, 2, 2, 2)
feedback("error", "shear") == (1, 0, 0, 0, 2)
feedback("crane", "crate") == (2, 2, 2, 0, 2)
```

In `feedback("error", "shear")`, the two middle `r`s are grey but the final `r` is green. That
single fact is why "a grey `r` means `r` is absent" is **not** a valid shortcut — see
`encodeLetterState` below.

Contract:

- pure: same arguments always give the same tuple;
- exactly five values, each in `{0, 1, 2}`;
- it does **not** validate word length; callers at a boundary should use `game.checkGuess`.

Cost: `O(5)` time plus a small dictionary.

### 2.2 `candidatesMatching(constraints, vocab) -> list[str]`

`constraints` is evidence. A candidate word `w` is possible iff

```
feedback(guess, w) == pattern        for every (guess, pattern) in constraints
```

The function returns every word in `vocab` that survives. It is the consistent-candidate set
the solver uses to avoid wasting guesses on words its own evidence has eliminated.

Contract:

- `vocab` is any **iterable of words**. Pass `loadAnswers()` for the solver. **Do not** pass
  `answersVocab()` — that is a frozen dataclass, not an iterable of words.
- pure;
- returns a new list; the caller may mutate it.

Cost: `O(|vocab| * |constraints| * 5)`. That is fine for 2,315 answers and at most five
constraints, but it is the hot path of `play.py` and the RL environment, so every
(answer, answer) pair is precomputed in `feedbackTable.py` (§2.3).

### 2.3 `feedbackTable.py` — the same rules, precomputed

`feedback(guess, secret)` is pure Python. Attempt 2's RL loop calls it millions of times, so the
module precomputes all 2,315 × 2,315 answer/answer pairs as one `uint8` code per pair, where a
code is the pattern read as a base-3 number (position 0 most significant, `0..242`).

Contract:

- `patternCode(pattern)` / `decodePattern(code)` convert between a 5-colour pattern and its code;
  both reject malformed input with `ValueError`.
- `feedbackTable()` returns a `(2315, 2315)` `uint8` array, **row = guess index, column = secret
  index**. It is built lazily on first call and cached for the process (~1.2 s, ~5.4 MB).
- `feedbackCode(guess, secret)` uses the table when both words are answers, and falls back to
  `feedback()` otherwise.
- `candidatesMatchingFast(constraints, vocab=None)` returns **exactly** what
  `candidatesMatching(constraints, vocab or loadAnswers())` returns, in the same order. Guesses
  or vocabulary words that are legal but not answers still work through a per-word fallback.
- the original `feedback()` and `candidatesMatching()` are unchanged; this module only
  accelerates callers that opt in.

Computation is lazy, so importing the core stays cheap. A table cell must always equal
`patternCode(feedback(guess, secret))`; `tests/test_feedbackTable.py` checks whole columns,
including repeated-letter secrets.

---

## 3. `vocabulary.py` — words and stable labels

### 3.1 Loading

`_DATA_DIR` resolves to `src/data/` from the location of `vocabulary.py`:

```
src/wordle/vocabulary.py -> parent = src/wordle -> parent = src -> /data
```

So the data lives at `src/data/words.txt` and `src/data/answers.txt`. `_read` strips each line
and drops blanks.

`loadWords()` and `loadAnswers()` are `@cache`d: the files are read once per process and the
same tuple is returned every time. Each call returns a **tuple**, so callers cannot accidentally
mutate the cached list.

### 3.2 `Vocabulary`

`Vocabulary` is a frozen dataclass with:

| Field / method | Meaning |
|---|---|
| `words: tuple` | ordered words; the index is the label |
| `index: dict` | `word -> position` |
| `indexOf(word)` | label for a word |
| `wordAt(index)` | word for a label |

`answersVocab()` is cached and always built over `loadAnswers()`. There is exactly **one**
authoritative label mapping per process, shared by train and test. That is what keeps labels
stable across datasets and models.

**Caveat:** `frozen=True` stops reassignment of the fields, but the `index` dict itself is still
mutable. Treat it as read-only and access it through `indexOf` / `wordAt`. A caller that does
`answersVocab().index["crane"] = 0` would corrupt every later dataset in the process.

### 3.3 Camel-case convention

The project uses camelCase throughout: `loadWords`, `answersVocab`, `indexOf`, `wordAt`,
`encodeRounds`, `playGameDetailed`, `addGuess`. Keep it consistent.

---

## 4. `game.py` — the solve-time driver

### 4.1 `GameState`

Holds a private `_guesses` list of `(guess, pattern)` pairs and exposes it through the `guesses`
property. The property returns the internal list (not a copy), which is a deliberate
speed/simplicity tradeoff: **pickers must treat it as read-only.**

- `addGuess(guess, pattern)` appends one round. Call it once per turn, in play order.
- `isSolved(secret)` is `True` only when the **last** guess equals `secret`. The length check
  comes first, so an empty history returns `False` instead of raising `IndexError` (a bug that
  was fixed and is pinned by tests).

### 4.2 `GameResult`

A dataclass with `won`, `guessesUsed`, `history`, `secret`.

**Loss semantics matter:** on a loss `guessesUsed` is `0`, even though `len(history)` is
`maxGuesses`. This preserves the original `playGame` contract. If you need the number of turns
played, use `len(result.history)`.

### 4.3 `checkGuess(guess)`

Validates a guess at the boundary:

1. it is a `str`;
2. it has exactly 5 characters;
3. it is in `loadWords()`.

`loadWords()` is converted to a cached `frozenset` once, because the old code would have
rebuilt a 14,855-element set on every turn of every simulated game. Failure raises `ValueError`
with the offending value.

Why this exists: `feedback` assumes five letters. Without the check, a malformed guess produces
plausible-looking but meaningless colours, and the real bug only shows up much later as bad
accuracy.

### 4.4 `playGameDetailed(secret, gamePicker, maxGuesses=6) -> GameResult`

The only game loop:

```
gameState = GameState()
repeat up to maxGuesses:
    guess = checkGuess(gamePicker(gameState.guesses))
    pattern = feedback(guess, secret)
    gameState.addGuess(guess, pattern)
    if gameState.isSolved(secret): return win result
return loss result
```

The picker contract is simply:

```
gamePicker(history: list[tuple[str, tuple[int, ...]]]) -> str
```

It receives the raw guess list (not a `GameState`), so the same seam can drive a human prompt, a
random baseline, a trained model, or an RL agent. The picker is called before anything is played,
so on the first turn it sees `[]`.

### 4.5 `playGame(...) -> int`

A thin wrapper returning `result.guessesUsed`: guesses used on a win, `0` on a loss. It is kept
for the 17 existing tests and the documented original contract; all new code should prefer
`playGameDetailed` so it can report win rate, average guesses, and the final board.

---

## 5. `features.py` — evidence to numbers

### 5.1 Constants

| Constant | Value | Meaning |
|---|---|---|
| `ALPHABET` | `"abcdefghijklmnopqrstuvwxyz"` | letter index `a=0 ... z=25` |
| `LETTER_INDEX` | dict | letter -> index |
| `PADDING` | `-1` | "this slot holds no evidence" |
| `COMPACT_BLOCK` | `10` | 5 colours + 5 letter indices |
| `ONEHOT_BLOCK` | `135` | 5 colours + 5 × 26 one-hot letters |
| `LETTER_STATES` | `8` | never / eliminated / present / green@0..4 |
| `NEVER`, `ELIMINATED`, `PRESENT` | `0`, `1`, `2` | state slots |
| `GREEN_OFFSET` | `3` | `green@p` is state `3 + p` |

`PADDING = -1` is outside every real range (colours `0..2`, letter indices `0..25`, one-hot
bits `0/1`, state bits `0/1`). Using `0` would be ambiguous: it means grey for colours and
"bit off" for one-hot slots.

### 5.2 `encode(gameState, probes)` — the legacy colour-only encoder

Concatenates the five colour codes of each round and pads with `-1` to `probes * 5`.

It is intentionally **still here and unchanged**: `tests/test_features.py::testEncodeIgnoresTheGuessWordItself`
pins its colour-only behaviour, and the legacy `generate()` uses it. Do **not** use it for a
solver that chooses its own guesses. Two consequences of its original design:

- it discards the guessed words, so `(1,0,0,0,0)` after `crane` and after `stout` is the same
  vector although the facts are different;
- it pads but never truncates, so more evidence than `probes` yields a longer vector.

### 5.3 `roundBlockSize(letterMode)`

Returns `10` for `"compact"` and `135` for `"onehot"`; raises for any other mode.

Why a function instead of a hardcoded number: every caller must compute width as
`maxRounds * roundBlockSize(letterMode)` and assert it, rather than remembering `50` or `675`.
Hardcoded widths are how train/solve mismatches sneak in silently.

### 5.4 `encodeRounds(guesses, maxRounds=5, letterMode="compact")`

The Attempt-1 encoder. Each played round contributes one block:

```
compact: [c0 c1 c2 c3 c4 | l0 l1 l2 l3 l4]                    width 10
         colours           letter indices 0..25

onehot:  [c0 c1 c2 c3 c4 | 26 bits | 26 bits | ... ]           width 135
         colours           one 1 per letter position
```

Unplayed rounds are padded with `PADDING`, so the total length is always

```
maxRounds * roundBlockSize(letterMode)
```

That constant width is what lets **one** model be consulted at turn 2 (1 round), turn 3
(2 rounds), …, turn 6 (5 rounds).

Why colours **and** letters: colours say what the game told us; letters say what we asked.
Colours alone are ambiguous once different games use different guesses, which is exactly the
redesign in Attempt 1.

Contract:

- pure; does not mutate `guesses`;
- every `(guess, pattern)` is validated by `_checkRound` (five lowercase letters, five colours
  in `0/1/2`); malformed evidence raises `ValueError` instead of becoming plausible garbage;
- more than `maxRounds` rounds raises; callers slice the history first, e.g.
  `guesses[:maxRounds]`;
- padding is always a **tail**: once a block is padded, every later block is padded too.

The `-1` padding applies to **all** slots, including one-hot letter slots. A padded one-hot
block is therefore `[-1] * 135`, not all zeros. Detect padding from the five colour slots.

### 5.5 `encodeLetterState(guesses)` — the aggregate rival encoder

Instead of storing rounds side by side, this folds all evidence into **one state per letter**:
26 letters × 8 one-hot states = **208 features**, constant length, no padding.

The state for a letter, in precedence order:

1. **green@p** (`3 + p`) — it has produced a green at position `p`;
2. **present** — it has produced at least one yellow;
3. **eliminated** — it appeared in a guess but never produced yellow or green;
4. **never** — it has not appeared in any guess.

Green outranks yellow outranks eliminated because a confirmed fact must not be overwritten by
weaker evidence. The repeated-letter case proves it: `feedback("error", "shear")` gives greys
for the middle `r`s and a green at position 4; a naive "any grey means eliminated" rule would
delete a letter that is genuinely in the secret.

Why "always grey" is a sound test for "not in the secret": `feedback` awards yellow greedily
until the secret's supply of that letter runs out. Any letter that appears in a guess **and** is
in the secret must therefore earn at least one yellow or green. So a letter with only grey
sightings is genuinely absent.

**What encoding B deliberately loses** (this is why it is a rival, not a replacement):

- it records only the **first** green position seen for a letter, so a repeated letter that is
  green in two positions loses one;
- it keeps "present somewhere" but not "not at position p", so it discards the positional
  exclusions that yellow gives you;
- it does not count how many copies of a letter the secret contains;
- it has no padding concept, which is convenient, but it also cannot tell "two rounds" from
  "one round that happened to say the same thing".

The experiment is deliberately **measurement, not argument**: implement
both and compare accuracy at the same number of rounds.

---

## 6. `dataset.py` — simulation, labels, and the no-leakage split

### 6.1 `Dataset`

```python
@dataclass
class Dataset:
    xTrain: list
    yTrain: list
    xTest: list
    yTest: list
    vocab: object
    maxRounds: int = 0
    letterMode: str = "compact"
    encoder: str = "rounds"      # "colour" | "rounds" | "letterState"
    split: str = "secret"        # "secret" | "evidence"
```

`Dataset.width` derives the expected row width from the stored recipe:

| `encoder` | Width |
|---|---|
| `"colour"` | `5 * maxRounds` |
| `"rounds"` | `maxRounds * roundBlockSize(letterMode)` |
| `"letterState"` | `26 * LETTER_STATES` = 208 |

`split` records what was held out: `"secret"` (train and test words are disjoint) or
`"evidence"` (every class is in train, test rows are fresh evidence). See §6.8 for why the
distinction exists.

Why this exists: a dataset that does not describe its own width is a trap. The solver has to
rebuild a feature vector for the model; if it picked the wrong `maxRounds` or letter mode, the
row would have the wrong width — or, worse, the right width with the wrong meaning. Use
`dataset.width` and assert against it at solve time.

`repr(Dataset)` still prints the whole 2,315-word vocabulary; that is noisy but not incorrect.

### 6.2 `_playProbes(secret, probes)`

Returns `[(probe, feedback(probe, secret)), ...]` for a fixed probe list. Pure. Used only by the
legacy `generate()`; `generateVariable()` deliberately does not use fixed probes.

### 6.3 `_splitBySecret(answers, nTrain, seed)`

The most important guarantee in the file:

1. copy the answer tuple into a list;
2. shuffle the **copy** with a seeded local `Random(seed)`;
3. return `shuffled[:nTrain]`, `shuffled[nTrain:]`.

Disjointness is therefore automatic. **Split by whole secret, never by row.** If a secret
appeared in both halves, the model would be tested on words it had effectively memorised and
test accuracy would be meaningless.

`nTrain` must satisfy `0 < nTrain < len(answers)`; otherwise `ValueError`.

### 6.4 `generate(probes, nTrain, nTest, seed=None)` — legacy fixed-probe generator

One row per secret, using `_playProbes` + `encode`. It returns a `Dataset` with
`encoder="colour"`, so `dataset.width == 5 * len(probes)`.

It remains because the existing tests depend on it and it is the simplest possible generator.
It is **not** what Attempt 1 trains on: Attempt 1 scores candidate words from raw evidence with
`attempt1.features`, because a 2,315-class answer classifier is capped by the uniform posterior.

`testSecrets` is trimmed to `nTest` with `[:nTest]`; if you request more test rows than remain,
you silently get fewer. Choose `nTrain + nTest <= 2315`.

### 6.5 `_secretRng(seed, secretIndex)`

Each secret gets its own deterministic RNG:

```
Random((seed if seed is not None else 0) * 1000003 + secretIndex)
```

Why per-secret rather than one RNG for the whole run: it makes the **n-round evidence a prefix**
of the (n+1)-round evidence for the same secret. The 1-round and 5-round datasets then describe
the same games cut off earlier, which is what makes "accuracy vs rounds of evidence" a fair
comparison instead of five unrelated datasets.

### 6.6 `_buildEvidence(secret, rng, count, guessPicker)`

Builds `count` **distinct** `(guess, pattern)` rounds:

- default (`guessPicker is None`): draw uniformly from `loadAnswers()`;
- custom: call `guessPicker(roundIndex, rng, copyOfEvidenceSoFar)`.

Why pass the evidence: a picker can only choose consistently if it can see what was already
played. That is the hook for consistent-guess and on-policy data generation. The list is copied,
so the picker cannot corrupt the generator's internal state.

Why no repeats: a real solver masks words already played, so training evidence containing
repeats would describe games it can never play.

Why the attempt limit: a picker that keeps returning the same word would otherwise loop forever,
and an infinite loop during data generation looks like a hang. The limit is
`50 * count + 100` calls; exceeding it raises `ValueError`.

The default is **not** a perfect imitation of the solver: uniform random guesses are usually not
consistent with earlier feedback, whereas the solver only plays consistent words. The experiment notes document the residual mismatch and the on-policy upgrade path.

### 6.7 `generateVariable(...)` — varied-guess, secret-disjoint rows

```python
generateVariable(nRounds, nTrain, nTest, seed=None, maxRounds=5,
                 letterMode="compact", guessPicker=None) -> Dataset
```

For each train and test secret:

1. compute its **global** answer index (the label);
2. build a per-secret RNG with `_secretRng(seed, index)`;
3. build `nRounds` rounds of evidence with `_buildEvidence`;
4. encode with `encodeRounds(evidence, maxRounds, letterMode)`;
5. return a `Dataset` with `encoder="rounds"`.

`nRounds` must be in `[1, maxRounds]`. Every row has width `dataset.width`, so one model can
serve every turn.

**Duplication when pooling rounds:** calling `generateVariable(1, ...)` … `generateVariable(5, ...)`
with the same seed produces the same secrets five times; the 1-round row is a prefix of the
2-round row, and so on. That is intentional for the accuracy-vs-rounds experiment, but it means
a pooled training set has five rows per secret, not five times as many distinct secrets. No
leakage is introduced: train and test secrets remain disjoint.

### 6.8 `generateClassComplete(...)` — every class in train, fresh evidence in test

```python
generateClassComplete(nRounds, nTest, seed=None, maxRounds=5,
                      letterMode="compact", guessPicker=None, testSeed=None) -> Dataset
```

Why it exists: a classifier over 2,315 answer classes can only output classes it has learned. If
400 secrets are held out of training, every test label is an unseen class and top-1 accuracy is
identically zero, no matter how good the model is. So training covers **all** answers, and the test
set asks the model to predict the same words from **different evidence**.

- Training rows: one per answer, `nRounds` of evidence, label = global answer index.
- Test rows: `selectTestSecrets(nTest, seed)` (the shuffled tail), with evidence generated from
  `testSeed` (default `seed + 1000`).
- If a test row is byte-identical to the **same secret's** training row, the function raises
  `ValueError` and asks for a different `testSeed`. Identical rows for **different** secrets are
  allowed: that is genuine Wordle ambiguity, not leakage.
- `Dataset.split == "evidence"`.

This is the generator that fixed the structurally-zero-accuracy split. Attempt 1 itself went one
step further and scores candidates from raw evidence (`evidenceFor` + candidate features), because
even a class-complete answer classifier is capped by the uniform posterior.

### 6.9 Public helpers: `splitSecrets`, `selectTestSecrets`, `evidenceFor`

| Function | Purpose |
|---|---|
| `splitSecrets(nTrain, seed=None)` | Public wrapper over `_splitBySecret`; returns `(trainSecrets, testSecrets)`. |
| `selectTestSecrets(nTest, seed=None)` | Deterministic held-out subset (the shuffled tail), at most 2,315. |
| `evidenceFor(secret, nRounds, seed=None, maxRounds=5, guessPicker=None)` | Raw `[(guess, pattern), ...]` for one secret, using the same per-secret RNG. |

`evidenceFor` is what the candidate-scoring experiment and the RL environment should use: they
need evidence, not an encoded class row.

---

## 7. Invariants worth asserting in new code

1. **Labels:** every label is in `[0, 2315)` and `answersVocab().wordAt(label)` is the secret.
2. **Width:** `len(row) == dataset.width` for every row, and the solver rebuilds with the same
   `maxRounds`, `letterMode`, and encoder the dataset records.
3. **Padding is a tail:** once a round block is padded, no later block is real.
4. **Letters are carried:** two different guesses with the same colours encode differently under
   `encodeRounds`; they do **not** under the legacy `encode`.
5. **Leakage:** for `generate`/`generateVariable`, train and test secret sets are disjoint. For
   `generateClassComplete`, classes overlap by design but no test row copies the same secret's
   training row; rows for different secrets may coincide (ambiguity).
6. **Determinism:** same seed and same arguments produce identical datasets.
7. **Legality:** every guess passed to `feedback` inside `playGameDetailed` has passed
   `checkGuess` (5 letters, in `words`).
8. **Purity:** `feedback`, `candidatesMatching`, `encode*`, `_playProbes`, and
   `_splitBySecret` do not mutate their inputs.

---

## 8. A worked example

Secret `stare`, guesses `crane`, `stout`, `slate`:

| Round | Guess | Feedback | Decoded colours |
|---|---|---|---|
| 1 | `crane` | `(0, 1, 2, 0, 2)` | grey, yellow, green, grey, green |
| 2 | `stout` | `(2, 2, 0, 0, 0)` | green, green, grey, grey, grey |
| 3 | `slate` | `(2, 0, 2, 1, 2)` | green, grey, green, yellow, green |

Compact encoding of round 1:

```
[0, 1, 2, 0, 2,  c, r, a, n, e]
[0, 1, 2, 0, 2,  2, 17, 0, 13, 4]
```

The first ten features are exactly that. One-hot mode replaces the five letter indices with
five 26-bit blocks (one `1` each), so its first ten features are
`[0, 1, 2, 0, 2, 0, 0, 1, 0, 0]` (the `1` is `c`).

`encodeLetterState` over all three rounds yields:

```
a=green@2  c=eliminated  e=green@4  l=eliminated  n=eliminated
o=eliminated  r=present  s=green@0  t=green@1  u=eliminated
```

Check a few by hand:

- `a` is green at position 2, and no later round contradicts it, so `green@2`.
- `r` was yellow in round 1 (`crane` pos 1), and never green, so `present`.
- `c` appeared in `crane` at position 0 and was grey, so `eliminated`.
- `b`, `d`, `f`, … were never guessed, so `never` (not printed above).

The full step-3 verification, including these decoded rows, lives in
`src/attempt1/trainCompare.py`; run it with the command in §10.

---

## 9. How the pieces fit at solve time

`play.py` (Attempt 1) will wrap a trained model in a picker:

```
history == []          -> prompt the human for guess 1 (interactive) or use the fixed opener (auto)
history != []          -> x = encodeRounds(history, dataset.maxRounds, dataset.letterMode)
                          assert len(x) == dataset.width
                          scores = model.predict_proba([x])[0]   # or decision_function
                          legal = candidatesMatching(history, loadAnswers())
                          legal = [w for w in legal if w not in played]
                          guess = max(legal, key=lambda w: scores[answersVocab().indexOf(w)])
```

`playGameDetailed` then computes feedback, appends to the history, and repeats until a win or
six guesses. The picker never sees the secret.

---

## 10. Running and testing

From the repository root:

```bash
# the whole regression suite
.venv/bin/python -m pytest tests/ -q

# the Attempt-1 step-3 sanity checks (decodes a known game, replays generated rows)
PYTHONPATH=src .venv/bin/python src/attempt1/trainCompare.py
```

Small interactive check of the core:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from wordle.feedback import feedback
from wordle.features import encodeRounds
from wordle.dataset import generateVariable

print(feedback("crane", "stare"))

ds = generateVariable(3, nTrain=20, nTest=5, seed=42)
print(ds.width, len(ds.xTrain), len(ds.xTest))
print(ds.xTrain[0][:10])
PY
```

`tests/conftest.py` puts `src/` on `sys.path` for pytest, so pytest needs no `PYTHONPATH`.
A script run from inside `src/attempt1/` does need it, because Python puts that script's
directory — not `src/` — on the path.

---

## 11. Traps that have already cost time

- **Colour-only encoding for a self-driving solver.** It looks fine and then stalls near chance.
  Use `encodeRounds` or `encodeLetterState`.
- **Train/solve width mismatch.** Compute `dataset.width`; never hardcode 50 or 675.
- **Padding confused with grey.** `-1` is padding; `0` is grey. Padded one-hot slots are `-1`,
  not `0`.
- **`candidatesMatching(answersVocab(), ...)`.** Pass `loadAnswers()` (an iterable of words), not
  the `Vocabulary` object.
- **`playGame` returning `0`.** That is a loss, not zero guesses. Use `playGameDetailed().won` and
  `len(.history)`.
- **Repeated guesses in training.** The solver masks them; `_buildEvidence` rejects them.
- **Assuming random training evidence is exactly the solver's distribution.** It is close in
  letter/colour space but looser in consistency; see the experiment notes.
- **Pooling 1..5 round datasets and assuming 5× the distinct secrets.** It is the same secrets at
  five prefix lengths.
- **Mutating `answersVocab().index` or `GameState.guesses`.** Both are shared or internal; treat
  them as read-only.
- **Direct `feedback` on unvalidated input.** It assumes five letters. Validate at boundaries with
  `checkGuess` (legal guesses) or `_checkRound` (encoder evidence).
- **Silent test-set truncation.** `testSecrets[:nTest]` returns fewer rows if `nTest` is too
  large; keep `nTrain + nTest <= 2315`.

---

## 12. Extension points

- **Attempt 1 trainers** (`src/attempt1/trainCompare.py`): build mixed-length `X`/`y` from
  `generateVariable`, train estimators, evaluate per round, plot accuracy vs rounds.
- **Attempt 1 solver** (`src/attempt1/play.py`): wrap a model as a picker; use
  `candidatesMatching` and never-repeat masking; interactive and `--auto` modes.
- **Consistent / on-policy data:** pass a `guessPicker(roundIndex, rng, evidence)` to
  `generateVariable`. A consistent random picker is

  ```python
  from wordle.feedback import candidatesMatching
  from wordle.vocabulary import loadAnswers

  def consistentPicker(roundIndex, rng, evidence):
      played = {guess for guess, _ in evidence}
      pool = [w for w in candidatesMatching(evidence, loadAnswers()) if w not in played]
      return pool[rng.randrange(len(pool))]
  ```

  It can still draw the secret before the final round; the generator does not stop early, so a
  row may contain post-win evidence. Handle that explicitly if you use it for training.
- **`encodeLetterState` datasets:** the encoder is supported, but `generateVariable` currently
  always uses `encodeRounds`. Derive B rows by decoding A1 rows with the decoders in
  `trainCompare.py`, or add an `encoder` option to `generateVariable` when the experiment needs
  it.
- **Entropy / information gain:** deferred; its natural home is `src/wordle/entropy.py`, and it
  would use `loadWords()` for probe search.
- **RL environment (Attempt 2):** reuse `encodeRounds`, the answer-index action space, and
  `candidatesMatchingFast` from `feedbackTable.py` (or `candidatesMatching` for a slow reference).
  The 2,315 × 2,315 table is already built and cached.

---

## 13. Test map

| File | What it pins |
|---|---|
| `tests/test_feedback.py` | the three documented cases, repeated-letter rules, `candidatesMatching` |
| `tests/test_feedbackTable.py` | pattern codes round-trip, table columns match `feedback()`, fast filter matches `candidatesMatching` |
| `tests/test_vocabulary.py` | list sizes (14,855 / 2,315), caching, word↔index round-trip |
| `tests/test_game.py` | `GameState`, `playGame`, both historical bugs, picker contract |
| `tests/test_features.py` | legacy `encode` only: length, `-1` padding, purity, colour-only behaviour |
| `tests/test_featuresVariable.py` | `encodeRounds` / `encodeLetterState`: widths, padding, letter modes, malformed input |
| `tests/test_dataset.py` | legacy generator: counts, widths, no leakage, determinism, `_splitBySecret` |
| `tests/test_datasetVariable.py` | variable generator: counts, width, prefix property, no leakage, custom pickers |
| `tests/test_datasetClassComplete.py` | class-complete generator: every class in train, evidence-held-out split, no same-secret row copy, prefix property, determinism, leaky-evidence rejection |

Run them all after any core change. A failure in `test_features.py` usually means the legacy
`encode` was touched; a failure in `test_dataset*.py` usually means the split or width contract
changed.
