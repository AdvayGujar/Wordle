from dataclasses import dataclass
from random import Random

from wordle.feedback import feedback
from wordle.features import encode
from wordle.vocabulary import answersVocab, loadAnswers

@dataclass
class Dataset:
    xTrain: list
    yTrain: list
    xTest: list
    yTest: list
    vocab: object

def _playProbes(secret, probes):
    evidence = []

    for probe in probes:
        pattern = feedback(probe, secret)
        evidence.append((probe, pattern))

    return evidence

def _splitBySecret(answers, nTrain, seed):
    if not 0 < nTrain < len(answers):
        raise ValueError(f"nTrain must be in (0, {len(answers)})")

    shuffled = list(answers)
    Random(seed).shuffle(shuffled)

    return shuffled[:nTrain], shuffled[nTrain:]

def generate(probes, nTrain, nTest, seed = None):
    answers = loadAnswers()
    vocab = answersVocab()

    trainSecrets, testSecrets = _splitBySecret(answers, nTrain, seed)
    testSecrets = testSecrets[:nTest]

    xTrain, yTrain = [], []
    for secret in trainSecrets:
        evidence = _playProbes(secret, probes)      # ← this function
        xTrain.append(encode(evidence, len(probes)))   # features.py
        yTrain.append(vocab.indexOf(secret))

    xTest, yTest = [], []
    for secret in testSecrets:
        evidence = _playProbes(secret, probes)      # ← this function
        xTest.append(encode(evidence, len(probes)))   # features.py
        yTest.append(vocab.indexOf(secret))

    return Dataset(xTrain, yTrain, xTest, yTest, vocab)
