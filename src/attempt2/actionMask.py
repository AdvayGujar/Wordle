"""Which of the 2,315 answers are still legal to guess, given the evidence.

WHY THIS IS THE MODULE THAT DECIDES WHETHER ATTEMPT 2 WORKS: without masking, the agent
has to rediscover from scratch two things that are free information - "do not repeat a
guess" and "do not guess a word the feedback already ruled out". Most of the training
signal would go into learning rules it was never meant to learn, and the policy would spend
its capacity on bookkeeping instead of strategy. With masking, the legal set is often a
handful of words by turn 4 and the problem becomes tractable (plan.md §8.3).

WHY THE MASK IS A FUNCTION OF THE HISTORY ALONE: it reads no secret and uses no randomness,
so the same history always produces the same mask. That is what keeps the environment
deterministic and free of the "non-stationarity with masking" pitfall in plan.md §8.8.

THE MASK IS A HARD CONSTRAINT, NOT A PENALTY: an illegal action is not "unlikely", it is
impossible. The policy applies the mask to the logits before the softmax (see
`attempt2.policies`), so an illegal action has probability exactly zero and the gradient
never pushes towards it.
"""

from functools import cache

import numpy as np

from wordle.feedbackTable import candidatesMatchingFast, feedbackTable, patternCode
from wordle.vocabulary import answersVocab, loadAnswers

# WHY CACHED: the index maps are rebuilt from 2,315 words on every call otherwise, and the
# mask is rebuilt once per turn of every game.
@cache
def _answerIndex():
    return answersVocab().index


def maskFor(history):
    """Boolean mask over the answer vocabulary: True where the answer is legal to guess.

    A word is legal when it is consistent with every round of feedback so far AND has not
    already been played.

    WHY THE TABLE IS USED DIRECTLY: the mask is needed on every turn of every episode, and
    an RL run makes millions of turns. Reading the precomputed table row by row turns the
    whole mask into a handful of vectorised comparisons over 2,315 booleans, instead of a
    Python loop over the consistent words. The rule is unchanged - the same table and the
    same `patternCode` the core's `candidatesMatchingFast` uses - and `tests/` pins the
    result against that core function.
    """
    answers = loadAnswers()
    index = _answerIndex()
    table = feedbackTable()

    mask = np.ones(len(answers), dtype=bool)

    for guess, pattern in history:
        position = index.get(guess)
        if position is None:
            # A guess that is not an answer has no row in the table, so the core filter
            # has to check it word by word. The agent only ever plays answers, so this
            # path exists for callers that pass hand-written evidence.
            mask = np.zeros(len(answers), dtype=bool)
            for word in candidatesMatchingFast(history, answers):
                mask[index[word]] = True
            break

        mask &= table[position] == patternCode(pattern)

    for guess, _ in history:
        position = index.get(guess)
        if position is not None:
            mask[position] = False

    return mask


def legalActions(history):
    """The legal action indices, in ascending order."""
    return [int(action) for action in np.flatnonzero(maskFor(history))]
