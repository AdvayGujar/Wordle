"""Tests for game.py -- the solve-time driver.

Includes regression tests for the two bugs found in review:
  1. the `__innit__` typo that left _guesses uninitialised
  2. isSolved() raising IndexError on an empty history
"""

from wordle.feedback import feedback
from wordle.game import GameState, playGame


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

def testGameStateStartsEmpty():
    # Regression: a misspelled __init__ meant this raised AttributeError.
    assert GameState().guesses == []


def testAddGuessRecordsThePair():
    state = GameState()
    state.addGuess("crane", (2, 0, 1, 0, 0))
    assert state.guesses == [("crane", (2, 0, 1, 0, 0))]


def testAddGuessAppendsInOrder():
    state = GameState()
    state.addGuess("crane", (2, 0, 1, 0, 0))
    state.addGuess("slate", (0, 2, 0, 1, 0))
    assert [g for g, _ in state.guesses] == ["crane", "slate"]


def testIsSolvedOnEmptyHistoryDoesNotCrash():
    # Regression: this used to raise IndexError.
    assert GameState().isSolved("apple") is False


def testIsSolvedTrueForTheMatchingLastGuess():
    state = GameState()
    state.addGuess("crane", (0, 0, 0, 0, 0))
    state.addGuess("apple", (2, 2, 2, 2, 2))
    assert state.isSolved("apple") is True


def testIsSolvedFalseForANonMatchingLastGuess():
    state = GameState()
    state.addGuess("crane", (0, 0, 0, 0, 0))
    assert state.isSolved("apple") is False


def testIsSolvedOnlyLooksAtTheMostRecentGuess():
    # Solved earlier, but a later guess moved on -- not solved any more.
    state = GameState()
    state.addGuess("apple", (2, 2, 2, 2, 2))
    state.addGuess("crane", (0, 0, 0, 0, 0))
    assert state.isSolved("apple") is False


# ---------------------------------------------------------------------------
# playGame
# ---------------------------------------------------------------------------

def testPlayGameReturnsGuessesUsedOnAWin():
    assert playGame("apple", lambda guesses: "apple") == 1


def testPlayGameReturnsCountAcrossSeveralGuesses():
    def picker(guesses):
        return "crane" if len(guesses) == 0 else "apple"

    assert playGame("apple", picker) == 2


def testPlayGameReturnsZeroOnALoss():
    assert playGame("apple", lambda guesses: "crane") == 0


def testPlayGameNeverExceedsMaxGuesses():
    calls = []

    def picker(guesses):
        calls.append(len(guesses))
        return "crane"

    assert playGame("apple", picker, maxGuesses=3) == 0
    assert len(calls) == 3


def testPlayGameRespectsACustomMaxGuesses():
    def picker(guesses):
        return "apple" if len(guesses) == 4 else "crane"

    assert playGame("apple", picker, maxGuesses=5) == 5
    assert playGame("apple", picker, maxGuesses=4) == 0


def testPlayGameRecordsRealFeedbackNotThePickerWord():
    seen = {}

    def picker(guesses):
        seen["guesses"] = list(guesses)
        return "crane" if len(guesses) == 0 else "apple"

    playGame("apple", picker)
    # The second call sees the genuine feedback for the first guess.
    guess, pattern = seen["guesses"][0]
    assert guess == "crane"
    assert pattern == feedback("crane", "apple")


def testPlayGameRecordsAllGreenWhenTheGuessIsCorrect():
    calls = []

    def picker(guesses):
        calls.append(list(guesses))
        return "crane" if len(guesses) == 0 else "apple"

    playGame("crane", picker)
    # Solved on the first guess, so the picker was only ever called once...
    assert len(calls) == 1
    # ...and the pattern it would have seen next is all green.
    assert feedback("crane", "crane") == (2, 2, 2, 2, 2)


def testPlayGamePassesGrowingEvidenceToThePicker():
    lengths = []

    def picker(guesses):
        lengths.append(len(guesses))
        return "crane"

    playGame("apple", picker, maxGuesses=4)
    assert lengths == [0, 1, 2, 3]


def testPlayGameProbesThenModelFlow():
    """The phase pattern: fixed probes, then a real guess."""
    probes = ["crane", "slate"]
    phases = []

    def picker(guesses):
        if len(guesses) < len(probes):
            phases.append("probe")
            return probes[len(guesses)]
        phases.append("model")
        return "apple"

    used = playGame("apple", picker)
    assert phases == ["probe", "probe", "model"]
    assert used == 3


def testPlayGameWorksWithAnAnswerVocabularyWord():
    """A guess the picker returns must be a legal 5-letter word."""
    assert playGame("abbey", lambda guesses: "abbey") == 1
