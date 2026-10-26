"""Tests for the Attempt-3 play modes.

The play module is the only part of Attempt 3 that talks to a human, so these tests cover
the things that could go wrong there: a strategy that proposes a word the game would refuse,
an advisor that repeats itself, or colours no secret could produce being believed.

Unlike Attempt 2's play modes, nothing here needs a trained artifact: the strategies are
deterministic code, so the tests always run.
"""

import builtins

import pytest

from attempt2.splits import heldOutSecrets
from attempt3 import play
from wordle.feedback import feedback
from wordle.vocabulary import answersVocab


def testTheStrategyRegistryHasEveryPolicy():
    assert sorted(play.STRATEGIES) == ["b1", "b2", "b3", "c1", "c1-noprobe", "c2"]


def testAnUnknownStrategyFailsLoudly():
    with pytest.raises(ValueError, match="unknown strategy"):
        play.strategyFor("nope")


def testTheNextWordIsALegalAnswerAndNeverRepeats():
    history = [("raise", (1, 0, 0, 0, 0))]
    _, pickGuess = play.strategyFor("c1")

    word = play.nextWord(pickGuess, history)

    assert word in answersVocab().index
    assert word != "raise"


def testEveryStrategyIsAFunctionOfTheHistoryAlone():
    # Same history, same answer - for every strategy. No secret is reachable from here.
    history = [("raise", (1, 0, 0, 0, 0))]

    for name in play.STRATEGIES:
        _, pickGuess = play.strategyFor(name)

        assert play.nextWord(pickGuess, history) == play.nextWord(pickGuess, history)


def testTheAdvisorProposesTheFirstWordBeforeAnyColours(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda *args: "quit")
    label, pickGuess = play.strategyFor("c1")

    with pytest.raises(SystemExit):
        play.advise(pickGuess, label)

    output = capsys.readouterr().out
    assert "The strategy's word:" in output


def testTheAdvisorLoopSolvesAGame(monkeypatch, capsys):
    """The strategy opens, the human reports the colours, and it finishes the game."""
    secret = heldOutSecrets()[0]
    suggestions = []
    realNextWord = play.nextWord

    def recordingNextWord(pickGuess, history):
        word = realNextWord(pickGuess, history)
        suggestions.append(word)
        return word

    monkeypatch.setattr(play, "nextWord", recordingNextWord)

    def fakeInput(prompt):
        return " ".join(str(color) for color in feedback(suggestions[-1], secret))

    monkeypatch.setattr(builtins, "input", fakeInput)

    label, pickGuess = play.strategyFor("c1")
    play.advise(pickGuess, label)

    output = capsys.readouterr().out
    assert "SOLVED" in output
    assert suggestions[-1] == secret


def testTheAdvisorRejectsColoursThatCannotHappen(monkeypatch, capsys):
    """A pattern no secret could produce must be refused, not believed."""
    attempts = []

    def fakeInput(prompt):
        attempts.append(1)
        # Four greens then a yellow is impossible: with the first four positions fixed,
        # the last letter can only be green or grey.
        return "quit" if len(attempts) > 1 else "2 2 2 2 1"

    monkeypatch.setattr(builtins, "input", fakeInput)

    label, pickGuess = play.strategyFor("c1")

    with pytest.raises(SystemExit):
        play.advise(pickGuess, label)

    output = capsys.readouterr().out
    assert "No answer is consistent with those colours" in output


def testTheInteractiveGameRunsToCompletion(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda *args: "crane")

    label, pickGuess = play.strategyFor("c1")
    play.playInteractive(pickGuess, label)

    output = capsys.readouterr().out
    assert "CRANE" in output
    assert "WON" in output or "LOST" in output


def testTheAutoModePrintsASpotCheckTable(monkeypatch, capsys):
    play.playAuto("b1", 2)

    output = capsys.readouterr().out
    assert "spot check on 2 held-out secrets" in output
    assert "B1 entropy" in output
    assert "masked random (A2)" in output
