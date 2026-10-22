"""Tests for the Attempt-2 play modes.

The play module is the only part of Attempt 2 that talks to a human, so these tests cover
the two things that could go wrong there: the environment accepting a guess the real game
would refuse, and the agent's advice depending on the secret (it must not).
"""

import builtins

import pytest

from attempt2.actionMask import maskFor
from attempt2.eval import PolicyState
from attempt2.observation import observationFor
from attempt2.play import loadAgent, render, renderTurn, suggestionFor
from attempt2.wordleEnv import EVAL_MODE, WordleEnv
from wordle.vocabulary import answersVocab, loadAnswers, loadWords


def _policy():
    """The trained agent, or a skip when there is none.

    The model is a generated artifact, so a fresh checkout may not have one. The play
    modes cannot work without it, and the tests that need it should say so rather than
    fail.
    """
    try:
        return loadAgent("reinforce")
    except SystemExit:
        pytest.skip("no trained Attempt-2 agent; run attempt2.train --algo reinforce first")


def _env(secret):
    env = WordleEnv(mode=EVAL_MODE)
    env.reset(seed=1, options={"secret": secret})
    return env


def testPlayWordGivesTheSameColoursAsTheRules():
    secret = heldOut()[0]
    env = _env(secret)

    _, _, solved, truncated, info = env.playWord("crane")

    from wordle.feedback import feedback

    assert info["guess"] == "crane"
    assert env.history[0][1] == feedback("crane", secret)
    assert solved is (secret == "crane")
    assert truncated is False


def testPlayWordAcceptsAWordThatIsNotAnAnswer():
    # "soare" is a legal guess but not one of the 2,315 answers, so it has no row in the
    # feedback table. A human may still play it.
    assert "soare" in loadWords()
    assert "soare" not in answersVocab().index

    env = _env(heldOut()[0])
    env.playWord("soare")

    from wordle.feedback import feedback

    assert env.history[0][0] == "soare"
    assert env.history[0][1] == feedback("soare", env.secret)


def testTheMaskStillWorksAfterANonAnswerGuess():
    env = _env(heldOut()[0])
    env.playWord("soare")

    mask = env.action_mask()

    assert mask.shape == (len(loadAnswers()),)
    assert mask.any()


def testPlayWordRejectsAnIllegalWord():
    env = _env(heldOut()[0])

    with pytest.raises(ValueError):
        env.playWord("zzzzz")


def testPlayWordRejectsARepeat():
    env = _env(heldOut()[0])
    env.playWord("crane")

    with pytest.raises(ValueError, match="already played"):
        env.playWord("crane")


def testPlayWordRefusesAfterTheGameIsOver():
    secret = heldOut()[0]
    env = _env(secret)
    env.playWord(secret)

    with pytest.raises(RuntimeError, match="episode is already over"):
        env.playWord("slate")


def testTheSuggestionDependsOnlyOnTheHistory():
    # The advisor mode must never see the secret: the same history has to produce the same
    # suggestion, whatever the secret happens to be.
    history = [("crane", (0, 0, 2, 0, 0))]
    policy = _policy()

    first = suggestionFor(policy, history)
    second = suggestionFor(policy, history)

    assert first == second
    assert first in answersVocab().index


def testTheSuggestionIsALegalAnswer():
    policy = _policy()
    history = [("crane", (0, 0, 2, 0, 0))]

    suggestion = suggestionFor(policy, history)

    mask = maskFor(history)
    assert mask[answersVocab().indexOf(suggestion)]


def testTheInteractiveGameRunsToCompletion(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *args: "crane")

    from attempt2.play import playInteractive

    policy = _policy()
    playInteractive(policy, "reinforce")


def testTheBoardRendersWithoutFailing(capsys):
    history = [("crane", (0, 1, 2, 0, 0))]

    render(history)
    renderTurn(history, 1)

    output = capsys.readouterr().out
    assert "CRANE" in output


def heldOut():
    from attempt2.splits import heldOutSecrets

    return heldOutSecrets()
