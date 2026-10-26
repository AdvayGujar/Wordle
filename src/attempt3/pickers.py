"""The game state the Phase-B pickers share, and the deterministic tie-break.

WHY THIS MODULE EXISTS: `attempt3.md` §4.3 lists one module per picker, but all three need
the same two things before they can score anything:

1. **The feasible set C** - the answers still consistent with every round of feedback.
2. **The guess pool** - the answers that have not been played yet.

Both are functions of the history alone, never of the secret. The rule that computes C
already exists in `attempt2.actionMask.maskFor`, and reimplementing it here would create a
second copy of the one rule the whole project depends on. So this module converts a history
into index arrays and nothing else.

WHY THE POOL IS ALL UNPLAYED ANSWERS, NOT JUST C: a probe is a legal move, and §4.1.2
settled that the pickers may play one. Restricting the pool to C would silently turn every
picker into the no-probe ablation, which is a different row of the table.

WHY THE TIE-BREAK LIVES HERE: §2.7 fixes it in one sentence - prefer a candidate, then the
smaller expected remaining count, then the lower index - and every picker must obey the same
one. Written once, a tie cannot be resolved two different ways in two ablations, which would
make their scores incomparable for a reason that has nothing to do with the criteria.

WHY `maskFor` IS SAFE TO REUSE: it excludes words already played. During a live game that
costs nothing, because a word that was played and did not win cannot be the secret - its own
feedback rules it out - so C already excludes it. The extra exclusion and C agree exactly.
"""

from dataclasses import dataclass

import numpy as np

from attempt2.actionMask import maskFor
from wordle.vocabulary import answersVocab, loadAnswers


@dataclass(frozen=True)
class PickingState:
    """The feasible set and the playable guesses for one turn, both as answer indices.

    `candidates` is sorted ascending and non-empty in a live game. `guesses` is every
    answer not yet played, also sorted ascending, and is always a superset of `candidates`.
    """

    candidates: np.ndarray
    guesses: np.ndarray

    @property
    def isSolved(self):
        """True when exactly one candidate is left, i.e. the answer is known."""
        return self.candidates.size == 1


def stateFor(history):
    """Build the picking state from a history of `(guess word, colour pattern)` rounds.

    The history may contain a guess that is not an answer (a human's guess), because
    `maskFor` handles that case through the core filter. The returned arrays are fresh, so
    a caller cannot corrupt a later turn by editing them.
    """
    rounds = tuple(history)
    candidates = np.flatnonzero(maskFor(rounds)).astype(np.int64, copy=False)

    index = answersVocab().index
    played = [index[word] for word, _ in rounds if word in index]

    playable = np.ones(len(loadAnswers()), dtype=bool)
    if played:
        playable[np.asarray(played, dtype=np.int64)] = False

    return PickingState(
        candidates=candidates,
        guesses=np.flatnonzero(playable).astype(np.int64, copy=False),
    )


def bestByScore(scores, guesses, candidates, expectedRemaining):
    """The index of the smallest score, broken by the §2.7 tie-break.

    The keys, in order of precedence:

    1. smallest `scores`;
    2. a candidate beats a probe;
    3. smallest `expectedRemaining` (the expected size of C after the guess);
    4. the lower answer index.

    `scores` may be `-entropy` for a maximiser. The arrays are validated because a silent
    length mismatch would pair a word with another word's score and pick a nonsense guess.
    """
    scoresArray = np.asarray(scores, dtype=np.float64)
    guessesArray = np.asarray(guesses)
    candidatesArray = np.asarray(candidates)
    remaining = np.asarray(expectedRemaining, dtype=np.float64)

    if scoresArray.ndim != 1 or scoresArray.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional array")

    if not (
        guessesArray.shape == scoresArray.shape == remaining.shape
    ):
        raise ValueError(
            "scores, guesses and expectedRemaining must have the same length, got %r"
            % (scoresArray.shape,)
        )

    if candidatesArray.size == 0:
        raise ValueError("candidates must contain at least one answer index")

    if not np.all(np.isfinite(scoresArray)) or not np.all(np.isfinite(remaining)):
        raise ValueError("scores and expectedRemaining must be finite")

    isCandidate = np.isin(guessesArray, candidatesArray)

    # np.lexsort sorts by the LAST key first, so this reads right-to-left as:
    # score, then candidate, then expected remaining, then index.
    order = np.lexsort((guessesArray, remaining, ~isCandidate, scoresArray))
    return int(guessesArray[order[0]])
