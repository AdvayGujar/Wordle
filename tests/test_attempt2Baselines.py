"""Tests for the Attempt-2 baseline policies.

Each baseline has to be what its name says, because the whole comparison in the report is
built on the difference between them.
"""

import pytest

from attempt2.baselines import (
    attempt1SolverPolicy,
    baselinePolicies,
    fixedOpenerRandomPolicy,
    fixedWordPolicy,
    maskedRandomPolicy,
    unmaskedRandomPolicy,
)
from attempt2.eval import evaluate
from attempt2.splits import heldOutSecrets
from wordle.vocabulary import answersVocab


def _state(history=(), mask=None):
    """A minimal PolicyState for testing a policy on its own."""
    import numpy as np

    if mask is None:
        mask = np.ones(len(answersVocab().words), dtype=bool)

    return type("State", (), {
        "observation": np.zeros(60, dtype=np.float32),
        "mask": mask,
        "history": history,
    })()


def testMaskedRandomOnlyPlaysLegalActions():
    policy = maskedRandomPolicy(42)
    secrets = heldOutSecrets()[:20]

    result = evaluate(policy, secrets, name="masked random")

    assert result.illegalActions == 0
    assert result.wins > 0


def testUnmaskedRandomPlaysIllegalActions():
    policy = unmaskedRandomPolicy(42)
    secrets = heldOutSecrets()[:10]

    result = evaluate(policy, secrets, name="unmasked random")

    # It ignores the mask, so almost every guess is a word the feedback eliminated.
    assert result.illegalActions > 0


def testFixedWordAlwaysPlaysTheSameWord():
    policy = fixedWordPolicy("crane")
    index = answersVocab().indexOf("crane")

    assert policy(_state()) == index
    assert policy(_state(history=(("crane", (0, 0, 0, 0, 0)),))) == index


def testFixedOpenerOpensWithTheChosenWord():
    policy = fixedOpenerRandomPolicy("crane", 42)
    index = answersVocab().indexOf("crane")

    assert policy(_state()) == index


def testTheBaselineListHasEveryPolicy():
    names = [name for name, _ in baselinePolicies(42)]

    assert len(names) == 5
    assert any("masked random" == name for name in names)
    assert any("Attempt-1" in name for name in names)


def testTheAttempt1SolverIsWiredToTheAttempt2Interface():
    # The trained Attempt-1 model is a generated artifact, so it may not exist in a
    # fresh checkout; skip rather than fail when it is missing.
    try:
        policy = attempt1SolverPolicy()
    except FileNotFoundError:
        pytest.skip("no trained Attempt-1 model; run attempt1/trainCompare.py first")

    action = policy(_state())

    assert 0 <= action < len(answersVocab().words)
    assert answersVocab().wordAt(action) == "crane"
