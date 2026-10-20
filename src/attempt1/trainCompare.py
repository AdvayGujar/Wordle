"""Attempt 1 - supervised candidate-consistency scoring.

WHAT CHANGED FROM THE ORIGINAL PLAN
The plan framed this as 2,315-class answer prediction. That cannot work: with a uniformly
chosen secret and deterministic feedback, every feedback-consistent candidate is equally
likely, so no classifier can rank the true secret above the others. The learnable task is
binary and candidate-relative:

    (evidence, candidate word) -> is this candidate consistent with every round?

`features.py` builds the candidate-relative vector; `common.py` builds the rows, trains the
binary models, scores candidates and persists the chosen model.

STEP 3 STILL LIVES HERE. The decoders below turn an encoded row back into words + colours,
and the checks prove the round encoding really carries the letters. They are the same
checks as before and run before any model.

STEPS 4-9
    4  baselines (always-consistent / always-inconsistent / random)
    5  train each binary model on the pooled mixed-length rows
    6  comparison table - F1 and AUC x rounds of evidence
    7  skill-vs-rounds plot -> src/attempt1/accuracy.png
    8  disagreement table for two models
    9  state-level top-1/top-5 and common false positives/negatives

Run it with:
    PYTHONPATH=src .venv/bin/python src/attempt1/trainCompare.py

WHY PYTHONPATH=src: running a file inside src/attempt1/ puts THAT directory on sys.path,
not src/, so "from wordle..." would fail without it (see the package import convention).
"""

import os
import tempfile
from collections import Counter

import numpy as np

# Set the cache before importing the GUI machinery; the default ~/.matplotlib is not
# writable in this environment and the warning is noise in an otherwise reproducible run.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "wordle-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attempt1.common import (
    MAX_ROUNDS,
    MODEL_FILE,
    PLOT_FILE,
    RESULTS_FILE,
    TRAIN_ROUNDS,
    consistencyMetrics,
    defaultThreshold,
    candidateMatrix,
    fitModel,
    majorityBaselines,
    modelFactories,
    positiveScores,
    saveArtifact,
    saveJson,
    testDataByRound,
    trainingData,
)
from attempt1.features import FEATURE_WIDTH, featureDescription
from wordle.dataset import generateVariable
from wordle.features import (
    ALPHABET,
    LETTER_STATES,
    PADDING,
    encodeLetterState,
    encodeRounds,
    roundBlockSize,
)
from wordle.feedback import candidatesMatching, feedback
from wordle.vocabulary import loadAnswers


# The three colour codes feedback() returns. Only used to print readable output.
COLOR_NAMES = ("grey", "yellow", "green")

# The names of the 8 states in an encodeLetterState block, in slot order.
LETTER_STATE_NAMES = ["never", "eliminated", "present"] + ["green@%d" % position
                                                           for position in range(5)]


# ---------------------------------------------------------------------------
# STEP 3a - the decoders: numbers back to evidence
# ---------------------------------------------------------------------------

def isPaddingBlock(block):
    """True when this round block was never actually played.

    WHY the check looks at the COLOUR slots only: encodeRounds() pads the whole tail of
    the vector with PADDING, so a padded round has -1 in every slot. Testing the letters
    instead would have to distinguish -1 from a malformed one-hot slot; the colours are
    the one part whose layout is the same in both modes, so keying off them is
    unambiguous.
    """
    return all(value == PADDING for value in block[:5])


def decodeRound(block, letterMode = "compact"):
    """Reverse ONE round block back into (guess, pattern).

    The two block layouts:

        compact:  [c0 c1 c2 c3 c4 | l0 l1 l2 l3 l4]          10 slots, letters 0-25
        onehot:   [c0 c1 c2 c3 c4 | 26 bits | 26 bits | ...] 135 slots, one 1 per letter

    WHY this raises on a padding block instead of decoding it: in "compact" mode every
    -1 would index ALPHABET from the end and silently produce "zzzzz"; in "onehot" mode
    the slot sum is -26, not 1. Both are nonsense, and the loud error points at the caller
    that failed to stop at the padding, which is the real bug.
    """
    if isPaddingBlock(block):
        raise ValueError("this block is padding, not a played round: %r" % (list(block),))

    pattern = tuple(block[:5])

    if letterMode == "compact":
        guess = "".join(ALPHABET[index] for index in block[5:10])

    else:
        letters = []

        for position in range(5):
            start = 5 + position * len(ALPHABET)
            slot = list(block[start:start + len(ALPHABET)])

            if sum(slot) != 1:
                raise ValueError(
                    "onehot letter slot for position %d has %d active bit(s), expected 1"
                    % (position, sum(slot))
                )

            letters.append(ALPHABET[slot.index(1)])

        guess = "".join(letters)

    return guess, pattern


def decodeRounds(vector, maxRounds = 5, letterMode = "compact"):
    """Reverse a whole encoded row back into [(guess, pattern), ...].

    WHY the width check comes first: a row with the wrong width means the caller and the
    encoder disagree about maxRounds or letterMode. Decoding it would "work" and quietly
    slice the evidence in the wrong places, which is exactly the train/solve mismatch
    the experiment notes warn about.

    WHY "evidence after padding" raises: padding is only ever a tail. A real block that
    appears after a padded one means the padding logic itself is broken, and returning a
    shorter list would hide that.
    """
    block = roundBlockSize(letterMode)
    expected = maxRounds * block

    if len(vector) != expected:
        raise ValueError(
            "row has %d features, expected maxRounds * roundBlockSize(%r) = %d * %d = %d"
            % (len(vector), letterMode, maxRounds, block, expected)
        )

    rounds = []
    paddingSeen = False

    for start in range(0, expected, block):
        chunk = list(vector[start:start + block])

        if isPaddingBlock(chunk):
            paddingSeen = True
            continue

        if paddingSeen:
            raise ValueError(
                "found real evidence at feature %d, after a padded round; padding must "
                "only ever be a tail" % start
            )

        rounds.append(decodeRound(chunk, letterMode))

    return rounds


def decodeLetterState(vector):
    """Reverse encodeLetterState() into {letter: stateName}.

    WHY this decoder exists: encoding A (encodeRounds) should be compared against
    encoding B (encodeLetterState) by measurement. Being able to read B
    back is what lets you see whether it is actually recording "green at position 3" and
    not quietly dropping a fact.
    """
    block = LETTER_STATES

    if len(vector) != len(ALPHABET) * block:
        raise ValueError(
            "letter-state vector has %d features, expected 26 * %d = %d"
            % (len(vector), block, len(ALPHABET) * block)
        )

    states = {}

    for index, letter in enumerate(ALPHABET):
        slot = list(vector[index * block:(index + 1) * block])

        if sum(slot) != 1:
            raise ValueError(
                "letter %r has %d active states, expected exactly 1" % (letter, sum(slot))
            )

        states[letter] = LETTER_STATE_NAMES[slot.index(1)]

    return states


# ---------------------------------------------------------------------------
# STEP 3a helper - readable output, so the evidence can actually be eyeballed
# ---------------------------------------------------------------------------

def buildEvidence(secret, guesses):
    """PLAY a game: pair each guess with the colours feedback() returns against secret.

    WHY this is a function rather than inline code: it is the one place that turns "a
    game" into the (guess, pattern) list every other function consumes, so training data
    and the live solver can be pointed at the same definition.
    """
    return [(guess, feedback(guess, secret)) for guess in guesses]


def formatEvidence(rounds, maxRounds = 5):
    """Render decoded rounds as one readable line per round, padding included."""
    lines = []

    for number in range(maxRounds):
        if number < len(rounds):
            guess, pattern = rounds[number]
            colors = ", ".join(COLOR_NAMES[color] for color in pattern)
            lines.append("  round %d: %s  [%s]" % (number + 1, guess, colors))
        else:
            lines.append("  round %d: (not played - padded with %d)" % (number + 1, PADDING))

    return "\n".join(lines)


def formatLetterState(states):
    """Only show the letters that are not still 'never', to keep the output readable."""
    parts = []
    for letter in ALPHABET:
        if states[letter] != "never":
            parts.append("%s=%s" % (letter, states[letter]))
    return "  " + "  ".join(parts)


# ---------------------------------------------------------------------------
# STEP 3b - the checks
# ---------------------------------------------------------------------------

def checkRoundTrip():
    """PART A: encode a known game, decode it, demand an exact match.

    WHY this is the core of step 3: it is the only thing that proves the 50 (or 675)
    numbers really carry the guessed WORDS and not just the colours. If the letters were
    dropped or the order were wrong, the decoded evidence would not equal the input.
    """
    secret = "stare"
    guesses = ["crane", "stout", "slate", "abide", "plumb"]
    cases = 0

    for letterMode in ("compact", "onehot"):
        width = 5 * roundBlockSize(letterMode)

        for nRounds in (1, 2, 3, 4, 5):
            evidence = buildEvidence(secret, guesses[:nRounds])
            vector = encodeRounds(evidence, 5, letterMode)

            assert len(vector) == width, (
                "width changed: %d features for %r, expected %d" % (len(vector), letterMode, width)
            )

            decoded = decodeRounds(vector, 5, letterMode)

            assert decoded == evidence, (
                "round-trip failed for %s / %d rounds:\n  encoded %r\n  decoded %r"
                % (letterMode, nRounds, evidence, decoded)
            )
            assert len(decoded) == nRounds, "padding leaked into the decoded rounds"

            cases += 1

    return "%d encode/decode round-trips" % cases


def checkGeneratedDataset(nTrain = 20, nTest = 5, seed = 42):
    """PART B: decode rows straight out of generateVariable() and replay them.

    WHY this is stronger than PART A: PART A proves the encoder is self-consistent. This
    proves that a row the model will actually TRAIN on decodes into a game that really
    happened - the decoded guesses, replayed against the decoded secret with feedback(),
    reproduce the decoded colours exactly. If the dataset paired the wrong secret with a
    row, this is where it would show up.
    """
    answers = loadAnswers()
    answerSet = set(answers)
    checked = 0

    for letterMode in ("compact", "onehot"):
        width = 5 * roundBlockSize(letterMode)

        for nRounds in (1, 3, 5):
            dataset = generateVariable(nRounds, nTrain = nTrain, nTest = nTest, seed = seed,
                                       maxRounds = 5, letterMode = letterMode)

            assert dataset.width == width, (
                "dataset declares width %d but this mode needs %d"
                % (dataset.width, width)
            )

            for rows, labels in ((dataset.xTrain, dataset.yTrain),
                                 (dataset.xTest, dataset.yTest)):
                for row, label in zip(rows, labels):
                    assert len(row) == dataset.width, (
                        "dataset row is %d wide, expected %d" % (len(row), dataset.width)
                    )

                    decoded = decodeRounds(row, 5, letterMode)

                    assert len(decoded) == nRounds, (
                        "row for %d rounds decoded to %d rounds" % (nRounds, len(decoded))
                    )

                    secret = dataset.vocab.wordAt(label)
                    replayed = buildEvidence(secret, [guess for guess, _ in decoded])

                    assert decoded == replayed, (
                        "row does not replay against its own label:\n  decoded  %r\n  replayed %r"
                        % (decoded, replayed)
                    )

                    for guess, _ in decoded:
                        assert guess in answerSet, (
                            "decoded guess %r is not an answer word" % (guess,)
                        )

                    checked += 1

    return "%d generated rows decoded and replayed" % checked


def checkRoundPrefixes(seed = 42):
    """PART C: n rounds of evidence is a PREFIX of n+1 rounds for the same secret.

    WHY this matters now rather than later: the accuracy-vs-rounds comparison uses
    rounds of evidence, and that comparison is only fair if round 1, round 2, ... describe
    the SAME games cut off at different points. dataset._secretRng() gives one RNG per
    secret precisely to guarantee it; this check makes the assumption explicit before any
    accuracy number depends on it.
    """
    comparisons = 0

    for letterMode in ("compact", "onehot"):
        block = roundBlockSize(letterMode)

        for nRounds in (1, 2, 3, 4):
            short = generateVariable(nRounds, nTrain = 10, nTest = 3, seed = seed,
                                     maxRounds = 5, letterMode = letterMode)
            long = generateVariable(nRounds + 1, nTrain = 10, nTest = 3, seed = seed,
                                    maxRounds = 5, letterMode = letterMode)

            for shortRow, longRow in zip(short.xTrain, long.xTrain):
                assert shortRow[:nRounds * block] == longRow[:nRounds * block], (
                    "the %d-round row is not a prefix of the %d-round row (%s)"
                    % (nRounds, nRounds + 1, letterMode)
                )
                comparisons += 1

    return "%d prefix comparisons" % comparisons


def checkLetterState(secret = "stare", guesses = ("crane", "stout", "slate")):
    """PART D: the rival encoding B records the facts encodeRounds records.

    The checks are the facts that must hold no matter what:
      - a green in the evidence => that letter's state is a green state
      - a yellow                => PRESENT, or green (green outranks yellow)
      - guessed and not in the secret => ELIMINATED
      - never guessed                 => NEVER
      - guessed and in the secret     => PRESENT or green (never ELIMINATED)
    """
    evidence = buildEvidence(secret, guesses)
    vector = encodeLetterState(evidence)

    assert len(vector) == len(ALPHABET) * LETTER_STATES == 208, (
        "letter-state width drifted: %d" % len(vector)
    )

    states = decodeLetterState(vector)

    for guess, pattern in evidence:
        for position, color in enumerate(pattern):
            letter = guess[position]
            state = states[letter]

            if color == 2:
                assert state.startswith("green@"), (
                    "%s is green at %d but its state is %r" % (letter, position, state)
                )
            elif color == 1:
                assert state == "present" or state.startswith("green@"), (
                    "%s is yellow at %d but its state is %r" % (letter, position, state)
                )

    guessed = set("".join(guesses))

    for letter in ALPHABET:
        state = states[letter]

        if letter not in guessed:
            assert state == "never", "%s was never guessed but its state is %r" % (letter, state)
        elif letter in secret:
            assert state == "present" or state.startswith("green@"), (
                "%s is in the secret but its state is %r" % (letter, state)
            )
        else:
            assert state == "eliminated", (
                "%s is not in the secret but its state is %r" % (letter, state)
            )

    return "%d letters checked" % len(ALPHABET)


# ---------------------------------------------------------------------------
# main - print the evidence, then run every check
# ---------------------------------------------------------------------------

def showTheEvidence():
    """The dump the sanity-check step asks for: decoded evidence, round by round."""
    secret = "stare"
    guesses = ["crane", "stout", "slate"]

    print("A known game: secret = %r, guesses = %r" % (secret, guesses))
    print()

    evidence = buildEvidence(secret, guesses)

    for letterMode in ("compact", "onehot"):
        block = roundBlockSize(letterMode)
        vector = encodeRounds(evidence, 5, letterMode)
        decoded = decodeRounds(vector, 5, letterMode)

        print("--- %s (%d features, %d per round) ---" % (letterMode, len(vector), block))
        print(formatEvidence(decoded, 5))
        print("  first 10 features: %r" % (vector[:10],))
        print()

    states = decodeLetterState(encodeLetterState(evidence))
    print("--- letter state (208 features) ---")
    print(formatLetterState(states))
    print()


# ---------------------------------------------------------------------------
# Steps 4-9: the candidate-consistency experiment
# ---------------------------------------------------------------------------

def meanMetric(perRound, key):
    return float(np.mean([perRound[n][key] for n in TRAIN_ROUNDS]))


def printBaselines(tests):
    print("=" * 78)
    print("STEP 4 - baselines (no model)")
    print("=" * 78)

    for nRounds in TRAIN_ROUNDS:
        _, labels, _, _, _ = tests[nRounds]
        baselines = majorityBaselines(labels)
        print(
            "  n=%d  consistent=%.3f  always-negative F1=%.3f  "
            "always-positive F1=%.3f  random AUC=%.3f"
            % (
                nRounds,
                float(np.mean(labels)),
                baselines["alwaysNegative"]["f1"],
                baselines["alwaysPositive"]["f1"],
                baselines["random"]["auc"],
            )
        )
    print()


def trainAll(tests):
    print("=" * 78)
    print("STEPS 5-6 - train binary consistency models on pooled rounds")
    print("=" * 78)

    features, labels = trainingData()
    print("  training rows: %d, features: %d, positive rate: %.3f"
          % (features.shape[0], features.shape[1], float(np.mean(labels))))
    print("  %s" % featureDescription())
    print()

    results = {}

    for name, factory in modelFactories().items():
        model, seconds = fitModel(factory, features, labels)
        threshold = defaultThreshold(model)
        perRound = {}

        for nRounds in TRAIN_ROUNDS:
            testFeatures, testLabels, _, _, _ = tests[nRounds]
            scores = positiveScores(model, testFeatures)
            perRound[nRounds] = consistencyMetrics(testLabels, scores, threshold)

        results[name] = {
            "model": model,
            "seconds": seconds,
            "threshold": threshold,
            "perRound": perRound,
        }

        print(
            "  %-24s fit %6.1fs  mean F1 %.3f  mean AUC %.3f  threshold %.2f"
            % (
                name,
                seconds,
                meanMetric(perRound, "f1"),
                meanMetric(perRound, "auc"),
                threshold,
            )
        )

    print()
    return results


def printComparison(results):
    print("=" * 78)
    print("STEP 6 - comparison table (rows: model, columns: rounds of evidence)")
    print("=" * 78)

    print("  F1 (higher is better; positives become rarer as rounds grow)")
    print("  " + "model".ljust(24) + "".join(("n=%d" % n).rjust(9) for n in TRAIN_ROUNDS)
          + "mean".rjust(9))
    for name, result in results.items():
        cells = "".join(("%9.3f" % result["perRound"][n]["f1"]) for n in TRAIN_ROUNDS)
        print("  " + name.ljust(24) + cells + ("%9.3f" % meanMetric(result["perRound"], "f1")))

    print()
    print("  AUC (threshold-free; the honest model-skill comparison)")
    print("  " + "model".ljust(24) + "".join(("n=%d" % n).rjust(9) for n in TRAIN_ROUNDS)
          + "mean".rjust(9))
    for name, result in results.items():
        cells = "".join(("%9.3f" % result["perRound"][n]["auc"]) for n in TRAIN_ROUNDS)
        print("  " + name.ljust(24) + cells + ("%9.3f" % meanMetric(result["perRound"], "auc")))

    print()
    print("  Note: F1 falls partly because consistent candidates become rare by round 5.")
    print("  AUC is the fairer 'did the model learn the rule?' number.")
    print()


def plotComparison(results):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for name, result in results.items():
        f1 = [result["perRound"][n]["f1"] for n in TRAIN_ROUNDS]
        auc = [result["perRound"][n]["auc"] for n in TRAIN_ROUNDS]
        axes[0].plot(TRAIN_ROUNDS, f1, marker="o", label=name)
        axes[1].plot(TRAIN_ROUNDS, auc, marker="o", label=name)

    for axis, title in zip(axes, ("Consistency F1", "Consistency AUC")):
        axis.set_title(title)
        axis.set_xlabel("rounds of evidence")
        axis.set_xticks(TRAIN_ROUNDS)
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("score")
    figure.suptitle("Attempt 1 - can a model learn the feedback-consistency rule?")
    figure.tight_layout()
    figure.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("  plot saved: %s" % PLOT_FILE)
    print()


def bestModelName(results):
    return max(results, key=lambda name: meanMetric(results[name]["perRound"], "auc"))


def stateLevelAnalysis(name, model, tests, maxStates=100):
    """Score every answer for a sample of states: can the model find a consistent word?"""
    print("=" * 78)
    print("STEP 9 - state-level candidate ranking for the best model (%s)" % name)
    print("=" * 78)

    answers = loadAnswers()
    threshold = defaultThreshold(model)
    summary = {}

    for nRounds in TRAIN_ROUNDS:
        states = tests[nRounds][2][:maxStates]
        top1Consistent = top1Secret = top5Secret = 0
        candidateCounts = []

        for secret, evidence in states:
            scores = positiveScores(model, candidateMatrix(evidence))
            order = np.argsort(-scores)
            top1 = answers[order[0]]
            top5 = {answers[index] for index in order[:5]}
            consistent = set(candidatesMatching(evidence, answers))

            top1Consistent += int(top1 in consistent)
            top1Secret += int(top1 == secret)
            top5Secret += int(secret in top5)
            candidateCounts.append(len(consistent))

        count = len(states)
        summary[nRounds] = {
            "top1Consistent": top1Consistent / count,
            "top1Secret": top1Secret / count,
            "top5Secret": top5Secret / count,
            "meanCandidates": float(np.mean(candidateCounts)),
            "randomTop1Secret": float(np.mean([1.0 / c for c in candidateCounts])),
            "randomTop5Secret": float(np.mean([min(5, c) / c for c in candidateCounts])),
        }

        row = summary[nRounds]
        print(
            "  n=%d  top1-consistent %.3f | top1-secret %.3f (random %.3f) | "
            "top5-secret %.3f (random %.3f) | mean candidates %.1f"
            % (
                nRounds,
                row["top1Consistent"],
                row["top1Secret"],
                row["randomTop1Secret"],
                row["top5Secret"],
                row["randomTop5Secret"],
                row["meanCandidates"],
            )
        )

    print()
    print("  The model's real job is the first number: find SOME consistent word.")
    print("  top1-secret is capped by candidate ambiguity and should track the random")
    print("  column when the model is doing the consistency job correctly.")
    print()
    return summary


def disagreementTable(results, tests, nRounds=5, limit=8):
    print("=" * 78)
    print("STEP 8 - disagreement table (two strongest models, n=%d)" % nRounds)
    print("=" * 78)

    names = sorted(results, key=lambda item: meanMetric(results[item]["perRound"], "auc"),
                   reverse=True)[:2]

    if len(names) < 2:
        print("  need two models; none to compare")
        print()
        return

    first, second = names
    firstModel = results[first]["model"]
    secondModel = results[second]["model"]
    firstThreshold = results[first]["threshold"]
    secondThreshold = results[second]["threshold"]

    features, labels, states, candidates, groups = tests[nRounds]
    firstPred = positiveScores(firstModel, features) >= firstThreshold
    secondPred = positiveScores(secondModel, features) >= secondThreshold
    different = np.where(firstPred != secondPred)[0]

    print("  comparing %s vs %s; %d/%d sampled rows disagree"
          % (first, second, len(different), len(labels)))
    print()

    for index in different[:limit]:
        group = int(groups[index])
        secret, evidence = states[group]
        print("  candidate=%s  truth=%s  %s=%s  %s=%s"
              % (
                  candidates[index],
                  "consistent" if labels[index] else "inconsistent",
                  first,
                  "consistent" if firstPred[index] else "inconsistent",
                  second,
                  "consistent" if secondPred[index] else "inconsistent",
              ))
        print(formatEvidence(evidence, MAX_ROUNDS))

    if len(different) == 0:
        print("  no disagreements in the sampled rows")
    print()


def topErrors(results, tests, name, limit=5):
    print("=" * 78)
    print("STEP 9 - most common mistakes for %s" % name)
    print("=" * 78)

    model = results[name]["model"]
    threshold = results[name]["threshold"]

    for nRounds in TRAIN_ROUNDS:
        features, labels, _, candidates, _ = tests[nRounds]
        predictions = (positiveScores(model, features) >= threshold).astype(int)
        falsePositive = Counter(
            candidates[index] for index in range(len(labels))
            if labels[index] == 0 and predictions[index] == 1
        )
        falseNegative = Counter(
            candidates[index] for index in range(len(labels))
            if labels[index] == 1 and predictions[index] == 0
        )

        print("  n=%d  false positives %d, false negatives %d"
              % (nRounds, sum(falsePositive.values()), sum(falseNegative.values())))
        print("    FP words: %r" % (falsePositive.most_common(limit),))
        print("    FN words: %r" % (falseNegative.most_common(limit),))

    print()


def serialisableResults(results, best, stateSummary):
    payload = {"best": best, "models": {}, "stateLevel": {}}

    for name, result in results.items():
        payload["models"][name] = {
            "seconds": float(result["seconds"]),
            "threshold": float(result["threshold"]),
            "perRound": {
                str(nRounds): {key: float(value) for key, value in metrics.items()}
                for nRounds, metrics in result["perRound"].items()
            },
        }

    for nRounds, metrics in stateSummary.items():
        payload["stateLevel"][str(nRounds)] = {
            key: float(value) for key, value in metrics.items()
        }

    return payload


def runExperiment():
    tests = testDataByRound()
    printBaselines(tests)

    results = trainAll(tests)
    printComparison(results)
    plotComparison(results)

    best = bestModelName(results)
    model = results[best]["model"]
    print("  best model by mean AUC: %s" % best)
    print()

    stateSummary = stateLevelAnalysis(best, model, tests)
    disagreementTable(results, tests)
    topErrors(results, tests, best)

    saveArtifact(model, best, {
        "meanAuc": meanMetric(results[best]["perRound"], "auc"),
        "meanF1": meanMetric(results[best]["perRound"], "f1"),
        "perRound": {
            str(n): {key: float(value) for key, value in metrics.items()}
            for n, metrics in results[best]["perRound"].items()
        },
    }, results[best]["threshold"])
    saveJson(RESULTS_FILE, serialisableResults(results, best, stateSummary))

    print("=" * 78)
    print("Artifacts")
    print("=" * 78)
    print("  model:   %s" % MODEL_FILE)
    print("  results: %s" % RESULTS_FILE)
    print("  plot:    %s" % PLOT_FILE)
    print()
    print("Next: PYTHONPATH=src .venv/bin/python src/attempt1/play.py")
    print("      PYTHONPATH=src .venv/bin/python src/attempt1/play.py --auto --games 400")
    print()


def main():
    print("=" * 78)
    print("ATTEMPT 1 - supervised candidate-consistency scoring")
    print("=" * 78)
    print()

    print("=" * 78)
    print("STEP 3 - sanity-check the round encoding")
    print("=" * 78)
    print()

    showTheEvidence()

    print("Checks:")

    checks = [
        ("known game survives encode -> decode", checkRoundTrip),
        ("generateVariable rows decode + replay", checkGeneratedDataset),
        ("n rounds is a prefix of n+1 rounds", checkRoundPrefixes),
        ("letter-state encoding keeps the facts", checkLetterState),
    ]

    for name, check in checks:
        detail = check()
        print("  PASS  %-41s %s" % (name, detail))

    print()
    print("ALL CHECKS PASSED - the encoding really carries the letters.")
    print("  Candidate feature width: %d" % FEATURE_WIDTH)
    print()

    runExperiment()


if __name__ == "__main__":
    main()
