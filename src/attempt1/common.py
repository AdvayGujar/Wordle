"""Shared experiment plumbing for Attempt 1.

WHAT ATTEMPT 1 LEARNED TO BE
The original plan framed this as 2,315-class answer prediction. That cannot work: with a
uniform secret and deterministic feedback, every feedback-consistent candidate is equally
likely, so no classifier can rank the true secret above the others. The learnable task is
therefore binary and candidate-relative:

    (evidence, candidate word) -> is this candidate consistent with every round?

`features.py` builds the candidate-relative vector; this module builds the rows, trains the
binary models, scores candidates, and persists the chosen model for `play.py`.
"""

import json
import time
from pathlib import Path
from random import Random

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from attempt1.features import FEATURE_WIDTH, candidateFeatures
from wordle.dataset import evidenceFor, splitSecrets
from wordle.feedback import candidatesMatching
from wordle.vocabulary import loadAnswers

SEED = 42
MAX_ROUNDS = 5
TRAIN_ROUNDS = (1, 2, 3, 4, 5)

# 600 training secrets and 400 disjoint test secrets. Features are candidate-relative, so
# the model can score a word it has never seen as a truth.
N_TRAIN_SECRETS = 600
N_TEST_SECRETS = 400

# Balanced sampling: up to 20 consistent (positive) and 20 inconsistent (negative)
# candidates per state. The candidate set is often huge after one round and tiny after five.
POSITIVES_PER_STATE = 20
NEGATIVES_PER_STATE = 20

DATA_SEED = 2024

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "model.joblib"
RESULTS_FILE = HERE / "results.json"
PLAY_RESULTS_FILE = HERE / "playResults.json"
PLOT_FILE = HERE / "accuracy.png"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def trainSecrets():
    train, _ = splitSecrets(N_TRAIN_SECRETS, SEED)
    return train


def testSecrets():
    _, test = splitSecrets(N_TRAIN_SECRETS, SEED)
    return test[:N_TEST_SECRETS]


def statesFor(secrets, nRounds, seed = SEED, guessPicker = None):
    """[(secret, evidence), ...] for each secret, with nRounds of shared evidence."""
    return [
        (secret, evidenceFor(secret, nRounds, seed, MAX_ROUNDS, guessPicker))
        for secret in secrets
    ]


def _draw(rng, population, count):
    count = min(count, len(population))
    return rng.sample(list(population), count) if count else []


def candidateRows(states, rng, positivesPerState = POSITIVES_PER_STATE,
                  negativesPerState = NEGATIVES_PER_STATE):
    """Build (X, y, group, candidate) rows from states.

    Positives are feedback-consistent candidates; negatives are sampled from the rest. A
    group is one state, and the candidate list is kept so callers can print a disagreement
    or an error example instead of a wall of numbers.
    """
    answers = loadAnswers()
    rows, labels, groups, candidates = [], [], [], []

    for group, (_, evidence) in enumerate(states):
        consistent = candidatesMatching(evidence, answers)
        consistentSet = set(consistent)
        negatives = [word for word in answers if word not in consistentSet]

        for candidate in _draw(rng, consistent, positivesPerState):
            rows.append(candidateFeatures(evidence, candidate))
            labels.append(1)
            groups.append(group)
            candidates.append(candidate)

        for candidate in _draw(rng, negatives, negativesPerState):
            rows.append(candidateFeatures(evidence, candidate))
            labels.append(0)
            groups.append(group)
            candidates.append(candidate)

    return (
        np.asarray(rows, dtype = float),
        np.asarray(labels, dtype = int),
        np.asarray(groups),
        candidates,
    )


def trainingData():
    """Pooled rows across all evidence lengths, so one model serves every turn."""
    features, labels = [], []
    secrets = trainSecrets()

    for nRounds in TRAIN_ROUNDS:
        rows, rowLabels, _, _ = candidateRows(
            statesFor(secrets, nRounds), Random(DATA_SEED + nRounds)
        )
        features.append(rows)
        labels.append(rowLabels)

    return np.vstack(features), np.concatenate(labels)


def testDataByRound():
    """{nRounds: (X, y, states, candidates, groups)} on the disjoint test secrets."""
    secrets = testSecrets()
    result = {}

    for nRounds in TRAIN_ROUNDS:
        states = statesFor(secrets, nRounds)
        rows, labels, groups, candidates = candidateRows(
            states, Random(DATA_SEED + 100 + nRounds)
        )
        result[nRounds] = (rows, labels, states, candidates, groups)

    return result


def candidateMatrix(evidence):
    """Feature rows for all 2,315 answers, in vocabulary order."""
    return np.asarray(
        [candidateFeatures(evidence, candidate) for candidate in loadAnswers()],
        dtype = float,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def modelFactories():
    """The model spread compared in the table.

    HGB is the strong learner for these piecewise comparisons; the forest is the ensemble
    baseline; the linear models show how far an additive boundary gets. All are binary and
    all expose either predict_proba or decision_function.
    """
    return {
        "histGradientBoosting": lambda: HistGradientBoostingClassifier(
            max_iter = 200,
            learning_rate = 0.1,
            max_leaf_nodes = 31,
            l2_regularization = 1.0,
            class_weight = "balanced",
            early_stopping = False,
            random_state = SEED,
        ),
        "randomForest": lambda: RandomForestClassifier(
            n_estimators = 200,
            class_weight = "balanced_subsample",
            n_jobs = -1,
            random_state = SEED,
        ),
        "logisticRegression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter = 500,
                class_weight = "balanced",
                random_state = SEED,
            ),
        ),
        "linearSvm": lambda: make_pipeline(
            StandardScaler(),
            LinearSVC(
                C = 0.1,
                class_weight = "balanced",
                max_iter = 5000,
                random_state = SEED,
            ),
        ),
    }


def fitModel(factory, X, y):
    """Fit one model and return (model, seconds)."""
    model = factory()
    start = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - start


def positiveScores(model, X):
    """One score per row, higher = more likely consistent."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities.ravel()

    scores = np.asarray(model.decision_function(X)).ravel()
    return scores


def defaultThreshold(model):
    """probability models split at 0.5; margin models split at 0."""
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def consistencyMetrics(yTrue, scores, threshold = 0.5):
    """Threshold-free and thresholded metrics for the positive (consistent) class."""
    predictions = (scores >= threshold).astype(int)

    return {
        "precision": precision_score(yTrue, predictions, zero_division = 0),
        "recall": recall_score(yTrue, predictions, zero_division = 0),
        "f1": f1_score(yTrue, predictions, zero_division = 0),
        "balancedAccuracy": balanced_accuracy_score(yTrue, predictions),
        "auc": roc_auc_score(yTrue, scores),
        "positiveRate": float(np.mean(yTrue)),
    }


def majorityBaselines(yTrue):
    """What a constant predictor gets: always-negative and always-positive."""
    positiveRate = float(np.mean(yTrue))
    alwaysPositiveF1 = 2 * positiveRate / (1 + positiveRate) if positiveRate else 0.0

    return {
        "alwaysNegative": {"f1": 0.0, "balancedAccuracy": 0.5},
        "alwaysPositive": {"f1": alwaysPositiveF1, "balancedAccuracy": 0.5},
        "random": {"auc": 0.5},
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def saveArtifact(model, name, metrics, threshold = 0.5):
    payload = {
        "model": model,
        "modelName": name,
        "featureWidth": FEATURE_WIDTH,
        "maxRounds": MAX_ROUNDS,
        "threshold": threshold,
        "seed": SEED,
        "metrics": metrics,
    }
    joblib.dump(payload, MODEL_FILE)
    return MODEL_FILE


def loadArtifact():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "no trained model at %s; run trainCompare.py first" % MODEL_FILE
        )
    return joblib.load(MODEL_FILE)


def saveJson(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent = 2)
    return path
