"""The policies the learned agent has to beat.

Every one of these is a function of the same `PolicyState` the learned policy will use, so
the comparison in the report is between *strategies*, not between different interfaces.

THE BAR IS "MASKED RANDOM", NOT "RANDOM": with the legal-action mask in place, a uniform
choice among the legal answers already wins ~100% of games (measured in Phase A). That
number is the real baseline. A policy that ignores the mask - the "unmasked random" one
below - is included precisely to show what the mask is worth, and it is expected to do
badly; it plays words the feedback has already eliminated and wastes guesses on them.

WHY THE ATTEMPT-1 SOLVER IS IMPORTED RATHER THAN REIMPLEMENTED: Attempt 1's pickers and
its trained model are the thing Attempt 2 has to be compared against. Copying them would
mean comparing Attempt 2 against a paraphrase of Attempt 1. Importing them means the
comparison row in the report is genuinely the Attempt-1 solver.
"""

import numpy as np

from attempt1.common import loadArtifact
from attempt1.play import modelFromArtifact, pickWithModel
from wordle.vocabulary import answersVocab


def maskedRandomPolicy(seed):
    """Uniform choice among the legal actions. This is the bar.

    WHY THIS IS THE IMPORTANT BASELINE: it uses the same information the agent has - the
    feedback - and the same rule set, and it still wins nearly every game. Any claimed
    improvement has to be measured against *this*, not against a policy that plays
    nonsense words.
    """
    rng = np.random.default_rng(seed)

    def policy(state):
        legal = np.flatnonzero(state.mask)
        return int(rng.choice(legal))

    return policy


def unmaskedRandomPolicy(seed):
    """Uniform choice among ALL answers, ignoring the mask. The ablation.

    WHY IT IS HERE: it shows what the mask is worth. Almost every guess it makes is a word
    the feedback has already eliminated, so it burns guesses and loses games. The harness
    counts every such guess in `illegalActions`.
    """
    rng = np.random.default_rng(seed)

    def policy(state):
        return int(rng.integers(0, state.mask.size))

    return policy


def fixedWordPolicy(word):
    """Always guess the same word.

    WHY IT IS HERE: the control. It never adapts, so it should never win - it shows what
    the rules alone achieve when the strategy does nothing.
    """
    index = answersVocab().indexOf(word)

    def policy(state):
        return index

    return policy


def fixedOpenerRandomPolicy(firstGuess, seed):
    """One chosen opening word, then masked random.

    WHY IT IS HERE: it isolates the value of the *opening* move from the value of the rest
    of the game. If a learned agent beats this baseline, it has learned more than "open
    well" - and if it does not, the opener is the whole story.
    """
    rng = np.random.default_rng(seed)
    opener = answersVocab().indexOf(firstGuess)

    def policy(state):
        if not state.history:
            return opener

        legal = np.flatnonzero(state.mask)
        return int(rng.choice(legal))

    return policy


def attempt1SolverPolicy(artifact=None, firstGuess="crane", mask=True):
    """The Attempt-1 supervised solver, driven through the Attempt-2 interface.

    WHY AN ADAPTER: Attempt 1's `pickWithModel` takes the history and returns a word;
    Attempt 2's harness wants an action index. The adapter is the whole difference - the
    model, the features and the rule mask are Attempt 1's own code.

    `firstGuess` defaults to `crane`, the opener Attempt 1's own solver evaluation used,
    so the comparison row is the solver as it was actually measured.
    """
    artifact = loadArtifact() if artifact is None else artifact
    model = modelFromArtifact(artifact)

    def policy(state):
        history = list(state.history)
        word = pickWithModel(model, history, mask=mask, firstGuess=firstGuess)
        return answersVocab().indexOf(word)

    return policy


def baselinePolicies(seed):
    """Every baseline, in the order the report should list them.

    WHY A SINGLE LIST: the report's baseline table and the code that produces it must not
    be able to disagree about which baselines exist.
    """
    return [
        ("masked random", maskedRandomPolicy(seed)),
        ("masked random, fixed opener 'crane'",
         fixedOpenerRandomPolicy("crane", seed + 1)),
        ("unmasked random (no mask)", unmaskedRandomPolicy(seed + 2)),
        ("fixed word ('crane' every turn)", fixedWordPolicy("crane")),
        ("Attempt-1 solver (model + rule mask)", attempt1SolverPolicy()),
    ]


def main():
    """Measure every baseline on the held-out secrets and print the table.

    Run it:

        PYTHONPATH=src .venv/bin/python -m attempt2.baselines
        PYTHONPATH=src .venv/bin/python -m attempt2.baselines --games 100
    """
    import argparse

    from attempt2 import config
    from attempt2.eval import evaluate, firstGuessTable, formatResultsTable
    from attempt2.splits import heldOutSecrets

    parser = argparse.ArgumentParser(description="Measure the Attempt-2 baselines")
    parser.add_argument("--games", type=int, default=config.N_TEST_SECRETS,
                        help="how many held-out secrets to play")
    arguments = parser.parse_args()

    secrets = heldOutSecrets()[:arguments.games]
    print("held-out secrets: %d" % len(secrets))
    print()

    results = [
        evaluate(policy, secrets, name=name)
        for name, policy in baselinePolicies(config.SEED)
    ]

    print(formatResultsTable(results))
    print()
    print("Opening guesses, masked random:")
    print(firstGuessTable(results[0].firstGuessCounts, limit=10))


if __name__ == "__main__":
    main()
