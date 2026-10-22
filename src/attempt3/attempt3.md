# Attempt 3 — information theory as the strategy

**Status:** 📋 **plan only.** No code has been written for this attempt. This file records
the mathematics, everything Attempt 2 measured that bears on it, and the build plan.

**It replaces the original Attempt 3** (from-scratch binary logistic regression on the
Module-5 math), which is deferred to Attempt 4. The reasoning: Attempt 2 measured exactly
how much of the remaining strategy headroom is reachable from reward alone, and the answer
was "the opening-word decision is worth about 0.3 guesses, and the signal to find it is
barely above the noise". An attempt that computes the decision instead of learning it
attacks the part of the headroom that the learning-based attempt could not reach.

---

## 1. Why an information-theoretic attempt

Attempt 1 learned the *rules* of Wordle and could not beat the rule filter, because with a
uniformly chosen secret **every feedback-consistent candidate is equally likely**. Attempt
2 tried to learn a *strategy* from reward alone and reached 3.85 average guesses on the
held-out secrets — better than random-consistent (4.07) and better than the Attempt-1
solver (3.90), but not better than one hand-picked opening word (3.75–3.80).

Attempt 2's own analysis says why, and it is not a bug:

| Quantity | Measured |
|---|---|
| Spread of opener quality across all 2,315 answers | 0.46 guesses (best 3.78, median 4.05, worst 4.61) |
| Noise in a single game's outcome | ~1.0 guesses |
| Openers within 0.05 guesses of the best | **144 of 2,315** |
| Episodes needed to sweep every opener once at ~100 samples each | ~230,000 |

The opening-word decision has a clear target but almost no resolution inside the target
group, so reward-driven learning is a poor tool for it. **Computing** the decision is not:
the entire action space can be searched exactly. And the deeper problem — deciding what to
guess *after* the opener — is a decision problem with an exact objective, not a learning
problem with a noisy one.

That is the case for this attempt: replace the search over strategies with a computation
over strategies.

---

## 2. The mathematics

### 2.1 The setup

Let \(\Omega\) be the answer set (\(|\Omega| = 2315\)) and let the secret be drawn
uniformly, \(P(S = s) = 1/|\Omega|\). After a history \(h = (g_1, p_1), \dots, (g_t, p_t)\)
of guesses and their colour patterns, the **feasible set** is

\[
C(h) = \bigl\{ s \in \Omega \;:\; p(g_i, s) = p_i \text{ for all } i \bigr\}.
\]

The posterior over the secret is **uniform on \(C(h)\)**:

\[
P(S = s \mid h) = \frac{1}{|C(h)|} \quad \text{for } s \in C(h), \qquad 0 \text{ otherwise}.
\]

This is the uniform-posterior theorem from Attempt 1, and it is the single most important
fact in the project: **no method can rank the true secret above another consistent
candidate from the evidence alone.** What can be optimised is *which consistent candidate to
guess*, and *which guess to make in order to shrink the consistent set fastest*.

### 2.2 What a guess carries

A guess \(g\) against a secret \(s\) produces a colour pattern \(p(g, s) \in \{0,1,2\}^5\),
so there are \(3^5 = 243\) possible patterns. Against a feasible set \(C\), write

\[
n_p = \bigl|\{ s \in C : p(g, s) = p \}\bigr|,
\qquad
P(p \mid g, C) = \frac{n_p}{|C|}.
\]

Because the pattern is a deterministic function of the secret, the information the guess
carries is the entropy of its own pattern distribution:

\[
I(g; S \mid C) = H\bigl(P(\cdot \mid g, C)\bigr)
= -\sum_{p} \frac{n_p}{|C|} \log_2 \frac{n_p}{|C|}
\quad \text{bits}.
\]

And the size of the feasible set afterwards is, in expectation,

\[
\mathbb{E}\bigl[\,|C'| \mid g, C\,\bigr]
= \sum_{p} \frac{n_p}{|C|}\, n_p
= \frac{1}{|C|} \sum_{p} n_p^{2}.
\]

Both quantities are computed from the precomputed feedback table: one histogram over 243
codes per candidate guess. **No simulation is involved** — the whole ranking of all 2,315
openers costs a few milliseconds.

### 2.3 The two criteria are related but not the same

Jensen's inequality gives

\[
H(P) \le \log_2 \mathbb{E}[n_p],
\]

so the entropy is an **upper bound** on the log of the expected remaining count, and the
gap between them measures how *non-uniform* the pattern distribution is. For the best
opener, `slate`:

\[
H = 5.856\ \text{bits}, \qquad
\mathbb{E}[|C'|] = 71.6, \qquad
\log_2 71.6 = 6.16\ \text{bits},
\]

a Jensen gap of 0.31 bits. Two openers with the same entropy can have very different
expected remaining counts, and vice versa — which is why the two criteria produce
different rankings (measured in §3.3).

### 2.4 The bit budget

\[
\underbrace{\log_2 |\Omega|}_{\text{the secret}} = \log_2 2315 = 11.18 \ \text{bits},
\qquad
\underbrace{\log_2 243}_{\text{one guess, at best}} = 7.92 \ \text{bits}.
\]

One guess cannot carry the secret (7.92 < 11.18). **Two guesses can** (15.85 > 11.18) —
but only on paper; see §3.4 for why the word constraint eats that.

The best opener actually extracts **5.88 bits**, i.e. 53% of the secret, and no guess
achieves more than 7.92. The remaining 5.3 bits have to come from later guesses, each of
which sees a smaller and therefore less informative feasible set.

### 2.5 The objective the game actually asks for

The game ends after six guesses. If a guess leaves a feasible set of size \(m\), and the
remaining guesses are made uniformly at random among the consistent candidates, the number
of further guesses needed is \(\mathbb{E}[\min(T, 5)]\) with \(T\) uniform on
\(\{1, \dots, m\}\):

\[
c(m) = \mathbb{E}[\min(T,5)] =
\begin{cases}
\dfrac{m+1}{2} & m < 5 \\[6pt]
5 - \dfrac{10}{m} & m \ge 5
\end{cases}
\]

and the expected number of guesses for an opener \(g\) is the concave functional

\[
\mathbb{E}[\text{guesses}] = 1 + \sum_{p} \frac{n_p}{|C|}\; c(n_p).
\]

**This concavity is the whole story of the disagreement between criteria.** For \(m \ge 5\),
\(c(m) = 5 - 10/m\) is concave and increasing, so the marginal cost of another candidate in
a big branch is small: those games are lost either way. An opener that produces **many
tiny branches and one huge branch** is therefore better than one that produces a flat
spread, *even if the flat spread has more information*. Entropy rewards flatness;
the game's objective rewards the other shape.

### 2.6 Why the letter-level theory works

Two effects, both measured:

**Distinct letters.** A guess with a repeated letter cannot distinguish as many outcomes,
because the colours of the repeated letter are constrained by the counts of that letter in
the secret. Measured mean pattern entropy:

| Distinct letters in the guess | n | Mean entropy |
|---|---|---|
| 2 | 1 | 2.40 bits |
| 3 | 57 | 3.41 bits |
| 4 | 691 | 4.13 bits |
| 5 | 1,566 | **4.77 bits** |

**Common letters.** Among the 1,566 words with five distinct letters, the correlation
between the guess's mean letter frequency and its pattern entropy is **0.80**.

Together these explain the empirical ranking without any search: the best openers are
permutations of the commonest letters. The top twenty are built from
\(\{a, e, t, l, s, c, r, n, p\}\), and six of them (`slate`, `least`, `slant`, `stale`,
`dealt`, `leant`) are permutations of the *same five letters* — because **the information
in an opening word is in which common letters it tests, not in their arrangement**.

---

## 3. Everything measured so far

### 3.1 The exhaustive opener ranking

`attempt2/openerSearch.py` plays every one of the 2,315 answers as the opening guess with
masked-random play afterwards (926,000 games), then re-measures the top 60 with 4,000
games each (240,000 games).

| Rank | Opener | Avg guesses | ± | Win rate |
|---|---|---|---|---|
| 1 | **`slate`** | **3.779** | 0.015 | 98.7% |
| 2 | `least` | 3.796 | 0.015 | 98.9% |
| 3 | `crate` | 3.822 | 0.015 | 98.8% |
| 4 | `slant` | 3.829 | 0.015 | 99.0% |
| 5 | `react` | 3.832 | 0.015 | 98.9% |
| 6 | `dealt` | 3.836 | 0.015 | 98.6% |

Across all 2,315 openers (first pass): best 3.750, 1st percentile 3.828, 10th percentile
3.915, **median 4.050**, 90th percentile 4.235, worst (`jazzy`) 4.605.

**The opening-word decision is worth about 0.30 guesses**, and 144 of the 2,315 openers are
within 0.05 guesses of the best.

### 3.2 The information-theoretic ranking of the same words

Computed in closed form from the feedback table — no simulation:

| Opener | entropy | E[remaining] | rank by entropy | avg guesses (4,000 games) | win rate |
|---|---|---|---|---|---|
| `raise` | 5.878 b | 61.0 | **1** | 3.885 | 98.2% |
| `slate` | 5.856 b | 71.6 | **2** | **3.816** | 99.0% |
| `crate` | 5.835 b | 72.9 | 3 | 3.822 | 98.4% |
| `irate` | 5.831 b | 63.8 | 4 | 3.869 | 98.1% |
| `trace` | 5.831 b | 74.0 | 5 | 3.836 | 98.5% |
| `arise` | 5.821 b | 63.7 | 6 | 3.885 | 97.8% |
| `crane` | 5.743 b | 78.7 | 12 | 3.846 | 98.2% |

Three things to take from this table:

1. **The information-theoretic screen is free** — milliseconds instead of 240,000 games —
   and it puts the good openers in the same place as the simulation.
2. **The criteria disagree.** `raise` has the most information and the fewest expected
   candidates left, and is *not* the best opener in the game. §2.5 explains why: the
   six-guess cap makes the objective concave, so the shape of the branch-size
   distribution matters more than its mean.
3. **The famous openers are missing from this project's answer list.** `soare`, `adieu`,
   `tares`, `lares` and `carle` all top published information-theoretic rankings — and all
   are in `words.txt` (the 14,855 legal guesses) but **not** in `answers.txt` (the 2,315
   possible secrets). Since the action space here is the answers only, they cannot be
   played at all. This is worth remembering before comparing any number with the
   literature.

### 3.3 Where PPO's choice sits

PPO, learning from reward alone, converged on `slate`: **rank 2 by information theory and
rank 1 by the game objective.** The agent found a word the classical analysis also puts at
the very top of the list.

### 3.4 Can two guesses identify the word? The budget says yes, the words say no

The budget: \(2 \times 7.92 = 15.85 > 11.18\) bits, so two guesses *should* suffice. The
measurement, with an **oracle** that knows the first branch and picks the best second word
for it (searched over all 2,315):

| After two guesses | Share of secrets |
|---|---|
| uniquely determined | **37%** |
| within 2 candidates | 47% |
| within 5 candidates | 54% |

And in the single biggest branch (221 candidates — the all-grey branch of `slate`), the
best possible second word leaves ~10 candidates on average and makes only **7%** of them
unique. A *perfect hash* would have made **97%** unique, because there are 243 patterns
for 221 candidates.

**Why the budget fails:** the second guess's letters overlap the first's, and the branch
has already eliminated those letters, so the second guess's pattern distribution over the
branch is skewed — it cannot act as a near-random hash. The information is nominally
available and practically unreachable.

Expected total guesses with the oracle second guess and random play afterwards: **3.4**.

### 3.5 What Attempt 2's agents did, and where they fell short

| Policy (400 held-out secrets) | Avg guesses |
|---|---|
| `slate` + REINFORCE's learned later play | 3.682 |
| `crane` + masked random | 3.752 |
| `slate` + masked random | 3.768 |
| REINFORCE, own opener (`actor`) + own play | 3.847 |
| PPO, own opener (`slate`) + own play | 4.013 |
| Attempt-1 solver | 3.900 |
| masked random | 4.056 |

REINFORCE learned a good *strategy* and a mediocre opener; PPO learned the best opener and
a *poor* strategy — its later greedy choices are worse than a random legal guess. Neither
is what we want, and the combination of the two is only marginally better than random on
the validation slice. A computed strategy should get both halves right at once.

### 3.6 The bar this attempt has to clear

Measured on the same 400 held-out secrets:

| To beat | Average guesses |
|---|---|
| `crane` + masked random (Attempt 1's opener) | 3.752 |
| `slate` + masked random (the best opener found) | 3.768 |
| `slate` + REINFORCE's learned play | 3.682 |

and the *information-theoretic* target: an opener that is simultaneously top-2 by entropy
and top-1 by the game objective — which `slate` already is.

---

## 4. The build plan

**Rules carried over from Attempt 2:** package-absolute imports, camelCase, one module per
concern, a test file per module, the same 400 held-out secrets and the same measurement
harness (`attempt2.eval.evaluate`) so the numbers are comparable, and no touching of the
core in `src/wordle/`.

### Phase A — the mathematics core

| # | Module | File | Purpose |
|---|---|---|---|
| A1 | `patterns` | `src/attempt3/patterns.py` | The pattern histogram over a feasible set; \(n_p\); the mutual information \(H(P(\cdot\mid g,C))\); the expected remaining count \(\mathbb{E}[|C'|]\); the Jensen gap |
| A2 | `objective` | `src/attempt3/objective.py` | The cost function \(c(m)\) and the concave functional \(\mathbb{E}[\text{guesses}]\); the exact expected-guess evaluator for a policy over a feasible-set tree |
| A3 | `letterStats` | `src/attempt3/letterStats.py` | The letter-frequency statistics that explain the ranking (§2.6) — used to *explain* results, and to sanity-check any hand-written guess |

**A1 is the important one.** Everything else consumes it. It must be implemented over the
precomputed feedback table (`wordle.feedbackTable`) so that a candidate set of any size
can be turned into a 243-bin histogram in one vectorised operation.

### Phase B — the pickers

| # | Module | File | Purpose |
|---|---|---|---|
| B1 | `entropyPicker` | `src/attempt3/entropyPicker.py` | Greedy per-turn maximiser of the pattern entropy. **This is the deferred `entropy.py` of the original plan (§6.6), finally written.** |
| B2 | `remainingPicker` | `src/attempt3/remainingPicker.py` | Greedy per-turn minimiser of \(\mathbb{E}[|C'|]\) |
| B3 | `pairPicker` | `src/attempt3/pairPicker.py` | A *fixed pair* of words chosen jointly to maximise the information of the pair — the "two words" idea from §3.4, done properly |

### Phase C — the adaptive strategy

| # | Module | File | Purpose |
|---|---|---|---|
| C1 | `adaptivePicker` | `src/attempt3/adaptivePicker.py` | Per-turn choice that maximises the **actual** objective — the expected number of guesses under §2.5, not a proxy. Greedy over the legal set, with the feasible set carried forward |
| C2 | `lookahead` | `src/attempt3/lookahead.py` | Two-ply: for each of the 243 first-round branches, the best second word, evaluated with the objective of §2.5 rather than with entropy |

**C1 is where the real work is.** The greedy criterion has to be the concave objective, not
the entropy, because §2.5 shows the two disagree — and that disagreement is exactly what
separates `slate` (3.816) from `raise` (3.885) in the game.

### Phase D — evaluation and write-up

| # | Module | File | Purpose |
|---|---|---|---|
| D1 | `experiment` | `src/attempt3/experiment.py` | The comparison table against Attempts 1 and 2 on the same 400 held-out secrets |
| D2 | `attempt3.md` | `src/attempt3/attempt3.md` | The results, replacing this plan's §3 with measured numbers |

---

## 5. Risks and open questions

- **The greedy criterion may be myopic.** Maximising the expected information at turn *t*
  is not the same as minimising the expected number of guesses; the two differ exactly
  where the concavity of §2.5 bites. The evaluation has to decide which proxy is being
  used, and the answer has to be measured, not argued.
- **Compute.** Evaluating one guess over a feasible set is one histogram over 243 bins from
  the feedback table, but a full turn needs that for every legal candidate (up to 2,315)
  and a two-ply search needs it for every (guess, branch) pair. This has to be vectorised
  from the start; a Python loop per candidate will not finish.
- **The action space.** The action space here is the 2,315 answers. The classic analyses
  use the full legal-guess list, where `soare` and `adieu` are available and rank first.
  Any comparison with published numbers has to state which list was used.
- **The uniform posterior still holds.** No computation can rank the true secret above
  another consistent candidate. What can be computed is *which consistent candidate is
  worth guessing* and *which guess shrinks the field fastest* — a strategy, not a
  prediction. The write-up must be careful not to imply otherwise.
- **The ceiling.** The opening-word decision is worth ~0.30 guesses and the whole opener
  spread is 0.86. If the strategy's real headroom is mostly in the *later* turns, then the
  information-theoretic picker has to be good at those turns too — and there the feasible
  sets are small, the entropy is low, and the differences between candidate guesses are
  correspondingly small.
- **Honest expectation.** The entropy screen is free and the two-guess analysis says the
  ceiling for a two-word strategy is ~37% identification. A realistic target is therefore
  an average in the **3.4–3.6** range with the best opener and a good adaptive follow-up
  — better than the 3.68 the two Attempt-2 agents achieved together, but not a dramatic
  win.

---

## 6. What would make this attempt succeed where Attempt 2 did not

1. **Compute what is worth computing.** The opening-word decision has an exact answer;
   reward-driven learning found it only by luck (PPO) or not at all (the others).
2. **Optimise the real objective.** The objective is the concave expected-guess functional
   of §2.5, not the entropy. Every published ranking that uses entropy is ranking a proxy.
3. **Keep the two decisions separate.** The opener and the follow-up strategy are
   independent optimisations: the opener can be searched exhaustively (it is 2,315
   candidates), and the follow-up can be searched per branch (243 branches × 2,315
   candidates). Attempt 2's failure was that one agent had to learn both at once, from
   noise.
