from feedback import feedback

class GameState:
    def __init__(self):
        self._guesses = []
    
    @property
    def guesses(self):
        return self._guesses

    def addGuess(self, guess, pattern):
        self._guesses.append((guess, pattern))

    def isSolved(self, secret):
        if len(self._guesses) > 0 and self._guesses[-1][0] == secret:
            return True
        else:
            return False

def playGame(secret, gamePicker, maxGuesses = 6):
    gameState = GameState()

    for _ in range(maxGuesses):
        guess = gamePicker(gameState.guesses)

        pattern = feedback(guess, secret)

        gameState.addGuess(guess, pattern)

        if gameState.isSolved(secret):
            return len(gameState.guesses)

    return 0
