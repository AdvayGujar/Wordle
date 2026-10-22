"""Tests for the Attempt-2 evaluation harness.

The harness is the single source of every number in the report, so what matters is that it
counts correctly, that it refuses to score a policy on the wrong secrets, and that it
reports the cost of a policy that ignores the mask instead of hiding it.
"""

import pytest

from attempt2.eval import (
    EvaluationResult,
    PolicyState,
    evaluate,
    firstGuessTable,
    formatResultsTable,
)
from attempt2.splits import heldOutSecrets, trainSecrets
from wordle.vocabulary import answersVocab


def _index(word):
    return answersVocab().indexOf(word)


def _oraclePolicy(secret):
    """A policy that plays the secret on the first turn. It cheats, which is the point."""
    def policy(state):
        return _index(secret)

    return policy


def testAPolicyThatPlaysTheSecretWinsInOneGuess():
    secret = heldOutSecrets()[0]

    result = evaluate(_oraclePolicy(secret), [secret], name="oracle")

    assert result.games == 1
    assert result.wins == 1
    assert result.winRate == 1.0
    assert result.avgGuessesOnWins == 1.0
    assert result.avgGuessesAll == 1.0
    assert result.guessDistribution == {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    assert result.firstGuessCounts == {secret: 1}
    assert result.illegalActions == 0


def testALossCountsAsSixGuessesInTheAverage():
    secret = heldOutSecrets()[0]
    wrong = _index("crane") if secret != "crane" else _index("slate")

    result = evaluate(lambda state: wrong, [secret], name="wrong")

    assert result.wins == 0
    assert result.losses == 1
    assert result.winRate == 0.0
    # A loss uses all six guesses, so the average over all games is 6.
    assert result.avgGuessesAll == 6.0
    assert result.avgGuessesOnWins == 0.0
    assert result.guessDistribution[6] == 0


def testTheMaskedPolicyNeverPlaysAnIllegalAction():
    secrets = heldOutSecrets()[:20]

    def policy(state):
        legal = [i for i, allowed in enumerate(state.mask) if allowed]
        return legal[0]

    result = evaluate(policy, secrets, name="first legal")

    assert result.illegalActions == 0
    assert result.games == 20


def testIllegalActionsAreCountedAndStillPlayed():
    secret = heldOutSecrets()[0]
    wrong = _index("crane") if secret != "crane" else _index("slate")

    result = evaluate(lambda state: wrong, [secret], name="wrong")

    # The guess was eliminated by the feedback, but the harness plays it anyway and
    # reports it, instead of quietly rescuing the policy. The first guess is legal -
    # with no evidence yet every answer is - so only the five repeats are illegal.
    assert result.illegalActions == 5


def testEvaluationOnATrainingSecretIsRefusedInEvalMode():
    secret = trainSecrets()[0]

    with pytest.raises(ValueError, match="not in the eval secret pool"):
        evaluate(lambda state: 0, [secret], name="leaky")


def testEvaluationOnAHeldOutSecretIsRefusedInTrainMode():
    secret = heldOutSecrets()[0]

    with pytest.raises(ValueError, match="not in the train secret pool"):
        evaluate(lambda state: 0, [secret], name="leaky", mode="train")


def testEvaluationOnTrainingSecretsWorksInTrainMode():
    secrets = trainSecrets()[:5]

    result = evaluate(lambda state: 0, secrets, name="train mode", mode="train")

    assert result.games == 5


def testAnOutOfRangeActionIsRejectedLoudly():
    secret = heldOutSecrets()[0]

    with pytest.raises(ValueError, match="not an answer index"):
        evaluate(lambda state: 99999, [secret], name="broken")


def testTheResultRecordsEveryGame():
    secrets = heldOutSecrets()[:3]

    result = evaluate(lambda state: 0, secrets, name="fixed")

    assert len(result.perGame) == 3
    for (secret, won, guesses), expected in zip(result.perGame, secrets):
        assert secret == expected
        assert won in (True, False)
        assert 1 <= guesses <= 6


def testThePolicySeesTheObservationTheMaskAndTheHistory():
    secret = heldOutSecrets()[0]
    seen = []

    def policy(state):
        seen.append(state)
        return _index(secret)

    evaluate(policy, [secret], name="observer")

    assert len(seen) == 1
    state = seen[0]
    assert isinstance(state, PolicyState)
    assert state.observation.shape == (60,)
    assert state.mask.shape == (2315,)
    assert state.mask.dtype.name == "bool"
    assert state.history == ()
    assert state.mask.all()


def testTheFormatTableShowsTheNumbers():
    secret = heldOutSecrets()[0]
    result = evaluate(_oraclePolicy(secret), [secret], name="oracle")

    table = formatResultsTable([result])

    assert "oracle" in table
    assert "100.0%" in table
    assert "| 1:1 |" in table


def testTheFirstGuessTableRanksByCount():
    table = firstGuessTable({"crane": 5, "slate": 9, "moire": 1}, limit=2)

    assert table.index("slate") < table.index("crane")
    assert "moire" not in table


def testAnEmptyEvaluationDoesNotDivideByZero():
    result = evaluate(lambda state: 0, [], name="nothing")

    assert result.games == 0
    assert result.winRate == 0.0
    assert result.avgGuessesAll == 0.0
