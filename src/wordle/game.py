from dataclasses import dataclass
from functools import cache

from wordle.feedback import feedback
from wordle.vocabulary import loadWords

class GameState:
    """Holds the evidence built up during one game, as (guess, pattern) pairs.

    WHY: the picker needs to see everything played so far to decide the next guess, and
    the encoder needs both halves of each pair - the word AND its colours. Keeping them
    together is what lets a solver use its own previous guesses as evidence.
    """

    def __init__(self):
        self._guesses = []
    
    @property
    def guesses(self):
        return self._guesses

    def addGuess(self, guess, pattern):
        """Append one round of evidence. WHY: one call per turn, in play order, so the
        history is a faithful record of the game and can be replayed into the encoder."""
        self._guesses.append((guess, pattern))

    def isSolved(self, secret):
        """True when the most recent guess was the secret.

        WHY only the last guess: once a guess matches, the game is over. Checking the whole
        history would wrongly report "solved" after a later, non-matching guess.
        The length check comes first so an empty history returns False instead of raising
        IndexError (a bug that was previously found here).
        """
        if len(self._guesses) > 0 and self._guesses[-1][0] == secret:
            return True
        else:
            return False

@dataclass
class GameResult:
    """Outcome of a finished game.

    WHY: playGame() reported only an int (guesses used, or 0 on a loss). Measuring a win
    rate, an average guess count, or printing the final board all need more than that, and
    re-simulating the game just to recover the history would be wasteful and error-prone.
    """
    won: bool
    guessesUsed: int
    history: list
    secret: str

@cache
def _legalWords():
    """The set of legal guesses, built once.

    WHY CACHED: loadWords() returns a 14,855-entry tuple. Rebuilding a set from it on every
    turn of every game would dominate the runtime of a few hundred simulated games.
    """
    return frozenset(loadWords())

def checkGuess(guess):
    """Reject an illegal guess loudly instead of silently producing junk feedback.

    WHY: feedback() assumes a 5-letter word. A model or a human supplying anything else
    would previously have been fed straight into it, producing meaningless colours and a
    confusing failure much later. This points the error at its source.
    """
    if not isinstance(guess, str) or len(guess) != 5:
        raise ValueError("guess must be a 5-letter word, got %r" % (guess,))
    if guess not in _legalWords():
        raise ValueError("guess %r is not in the legal word list" % (guess,))
    return guess

def playGameDetailed(secret, gamePicker, maxGuesses = 6):
    """Run a full game and return the GameResult.

    WHY: this is the loop that actually plays Wordle - ask the picker, colour the guess,
    record the evidence, stop on a win. It is driving a picker rather than hardcoding a
    strategy, so the same driver runs a random baseline, a trained model, or a human.
    GameResult exists so callers can report win rate, average guesses and the board,
    which the int-returning playGame() could not.
    """
    gameState = GameState()

    for _ in range(maxGuesses):
        guess = checkGuess(gamePicker(gameState.guesses))

        pattern = feedback(guess, secret)

        gameState.addGuess(guess, pattern)

        if gameState.isSolved(secret):
            return GameResult(True, len(gameState.guesses), list(gameState.guesses), secret)

    return GameResult(False, 0, list(gameState.guesses), secret)

def playGame(secret, gamePicker, maxGuesses = 6):
    """Original driver, kept as the int-returning wrapper.

    WHY KEPT: 17 existing tests and the documented contract (guesses used on a win, 0 on a
    loss) depend on it. It now delegates, so there is only one game loop to maintain and
    it inherits the illegal-guess check.
    """
    return playGameDetailed(secret, gamePicker, maxGuesses).guessesUsed
