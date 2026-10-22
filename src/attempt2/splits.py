"""The train / held-out split of the answer list.

WHY THIS EXISTS: an RL agent learns by PLAYING, so the only thing standing between a real
held-out evaluation and a fake one is whether the environment can ever sample an evaluated
secret during training. That is a one-line property of the secret list, and it is far too
important to leave implicit (plan.md §8.6).

WHY IT REUSES THE CORE SPLIT: `wordle.dataset.splitSecrets` already splits by whole SECRET
and is already tested. A second split protocol here would be a second definition of
"held out", and the two would drift apart.

The split is deterministic: the same seed always produces the same 1,915 training secrets
and the same 400 held-out secrets, so every algorithm, every rerun and the Attempt-1
comparison are evaluated on exactly the same words.
"""

from functools import cache

from attempt2 import config
from wordle.dataset import splitSecrets

# WHY CACHED: the split is a shuffle of 2,315 words and every env, baseline and evaluation
# needs the same two lists. Rebuilding it per call would be pure waste.
@cache
def _split():
    return splitSecrets(config.N_TRAIN_SECRETS, config.SEED)


def trainSecrets():
    """The secrets the agent is allowed to play during training."""
    return _split()[0]


def heldOutSecrets():
    """The held-out secrets. The environment samples these ONLY in evaluation mode."""
    return _split()[1]


def trainingSecrets():
    """The secrets the agent trains on: the training half minus the validation slice."""
    return trainSecrets()[:-config.N_VALIDATION_SECRETS]


def validationSecrets():
    """The validation slice: training secrets the agent does NOT train on.

    WHY A SLICE OF THE TRAINING HALF: the held-out 400 may be read once, for the final
    table. The learning curve, the hyper-parameters and the decision to stop training all
    have to be decided on something else, or the final number is a number that was tuned
    on the test set.
    """
    return trainSecrets()[-config.N_VALIDATION_SECRETS:]


def assertNoLeakage(train=None, test=None):
    """Raise ValueError if the two halves overlap, or if either is the wrong size.

    WHY AN ASSERTION AND NOT A COMMENT: the leak this guards against is invisible in the
    results - a contaminated run looks exactly like a good one. Attempt 1 pins the same
    property in `tests/test_dataset.py`; this is the Attempt-2 copy of that guard.
    """
    train = trainSecrets() if train is None else list(train)
    test = heldOutSecrets() if test is None else list(test)

    if len(train) != config.N_TRAIN_SECRETS:
        raise ValueError(
            "expected %d training secrets, got %d" % (config.N_TRAIN_SECRETS, len(train))
        )

    if len(test) != config.N_TEST_SECRETS:
        raise ValueError(
            "expected %d held-out secrets, got %d" % (config.N_TEST_SECRETS, len(test))
        )

    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(
            "leak: %d secret(s) are in both halves, e.g. %r"
            % (len(overlap), sorted(overlap)[0])
        )

    return True
