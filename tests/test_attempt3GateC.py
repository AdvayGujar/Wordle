"""Tests for the Gate C paired comparison.

`gateC.py` reports a paired per-secret difference with a 95% interval, because a 0.02-guess
gap between averages means nothing on its own. That summary is the only logic in the
harness, so it is tested here with synthetic results - no games are played.

What is pinned:

- the difference is `right - left`, per secret;
- a loss costs the full guess budget, so losing early is not rewarded;
- secrets are matched by name, not by position.
"""

import math

import pytest

from attempt2.eval import EvaluationResult
from attempt3.gateC import pairedSummary


def _result(name, perGame):
    wins = sum(1 for _, won, _ in perGame if won)
    return EvaluationResult(
        name=name,
        games=len(perGame),
        wins=wins,
        losses=len(perGame) - wins,
        winRate=wins / len(perGame),
        avgGuessesOnWins=0.0,
        avgGuessesAll=0.0,
        guessDistribution={},
        firstGuessCounts={},
        illegalActions=0,
        perGame=tuple(perGame),
    )


def testTheDifferenceIsPerSecretAndCountsBetterAndWorse():
    left = _result("left", [("a", True, 3), ("b", False, 6), ("c", True, 2)])
    right = _result("right", [("a", True, 2), ("b", True, 4), ("c", True, 2)])

    summary = pairedSummary(left, right)

    assert summary["meanDifference"] == pytest.approx(-1.0)
    assert summary["standardError"] == pytest.approx(1.0 / math.sqrt(3))
    assert summary["better"] == 2
    assert summary["worse"] == 0
    assert summary["same"] == 1


def testALossCostsTheFullGuessBudget():
    # Losing in 3 is not cheaper than winning in 6: both cost the budget.
    left = _result("left", [("a", False, 3)])
    right = _result("right", [("a", True, 6)])

    summary = pairedSummary(left, right)

    assert summary["meanDifference"] == 0.0
    assert summary["same"] == 1


def testSecretsAreMatchedByNameNotByPosition():
    left = _result("left", [("a", True, 3), ("b", True, 2)])
    reordered = _result("right", [("b", True, 2), ("a", True, 3)])

    summary = pairedSummary(left, reordered)

    assert summary["meanDifference"] == 0.0
    assert summary["same"] == 2


def testTheSummaryNamesBothPolicies():
    left = _result("left", [("a", True, 3)])
    right = _result("right", [("a", True, 3)])

    summary = pairedSummary(left, right)

    assert summary["reference"] == "left"
    assert summary["other"] == "right"
