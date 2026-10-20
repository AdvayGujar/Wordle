from dataclasses import dataclass
from random import Random

from wordle.feedback import feedback
from wordle.features import (
    ALPHABET,
    LETTER_STATES,
    encode,
    encodeRounds,
    roundBlockSize,
)
from wordle.vocabulary import answersVocab, loadAnswers

# The encoders a Dataset row can have been produced by. `width` needs to know which one,
# because the three have different width formulas.
ENCODINGS = ("colour", "rounds", "letterState")

# How train/test were separated.
#   "secret"   - disjoint secrets (the original rule; train classes and test classes differ)
#   "evidence" - all classes in train, fresh evidence in test (needed for a classifier)
SPLITS = ("secret", "evidence")

@dataclass
class Dataset:
    """Train/test features, labels, the label vocabulary, and the encoding recipe.

    WHY maxRounds, letterMode and encoder are stored: a dataset that does not describe its
    own width is a trap. The solver has to rebuild a feature vector for the model, and if
    it guessed the wrong maxRounds or letter mode the rows would have the wrong width - or,
    worse, the right width with the wrong meaning. Storing the recipe makes the dataset
    self-describing.

    WHY encoder: generate() uses the legacy colour-only encode(), while generateVariable()
    uses encodeRounds(). Their widths follow different formulas (5 * probes versus
    maxRounds * roundBlockSize), so maxRounds alone cannot describe both.

    WHY split: "secret" means train and test secrets are disjoint; "evidence" means every
    class appears in train and test rows come from fresh evidence sequences. A classifier
    over 2,315 classes cannot predict a class it never saw, so Attempt 1 needs "evidence".
    """

    xTrain: list
    yTrain: list
    xTest: list
    yTest: list
    vocab: object
    maxRounds: int = 0
    letterMode: str = "compact"
    encoder: str = "rounds"
    split: str = "secret"

    @property
    def width(self):
        """Expected number of features per row, derived from the stored recipe.

        WHY: the solver asserts its rebuilt vectors against this before predicting. A
        train/solve width mismatch is the quiet failure mode the experiment notes warn
        about, so make the expected width computable instead of remembered.
        """
        if self.encoder == "colour":
            return 5 * self.maxRounds
        if self.encoder == "rounds":
            return self.maxRounds * roundBlockSize(self.letterMode)
        if self.encoder == "letterState":
            return len(ALPHABET) * LETTER_STATES
        raise ValueError("encoder must be one of %r, got %r" % (ENCODINGS, self.encoder))

def _playProbes(secret, probes):
    """Play a FIXED list of probes against one secret and return (probe, pattern) pairs.

    WHY: this is the original fixed-probe generator's workhorse, and it stays because
    generate() and its tests still use it. generateVariable() deliberately does NOT use
    it - fixed probes are exactly what caused the train/solve distribution shift.
    """
    evidence = []

    for probe in probes:
        pattern = feedback(probe, secret)
        evidence.append((probe, pattern))

    return evidence

def _splitBySecret(answers, nTrain, seed):
    """Split the answer list into disjoint train/test secrets.

    WHY THIS IS THE MOST IMPORTANT FUNCTION HERE: the split is by whole SECRET, never by
    row. If a secret appeared in both halves, the model would effectively be tested on
    words it had memorised and test accuracy would be meaningless. Every other guarantee
    in this file rests on this one.
    """
    if not 0 < nTrain < len(answers):
        raise ValueError(f"nTrain must be in (0, {len(answers)})")

    shuffled = list(answers)
    Random(seed).shuffle(shuffled)

    return shuffled[:nTrain], shuffled[nTrain:]

def generate(probes, nTrain, nTest, seed = None):
    """Original generator: one row per secret, evidence from a FIXED probe list.

    WHY STILL HERE: 24 existing tests depend on it, and it remains the simplest way to
    build a dataset. It is NOT what Attempt 1 trains on, because the solver supplies its
    own guesses at solve time and this produces evidence from someone else's - see
    generateVariable for the fix.
    """
    answers = loadAnswers()
    vocab = answersVocab()

    trainSecrets, testSecrets = _splitBySecret(answers, nTrain, seed)
    testSecrets = testSecrets[:nTest]

    xTrain, yTrain = [], []
    for secret in trainSecrets:
        evidence = _playProbes(secret, probes)      # this function
        xTrain.append(encode(evidence, len(probes)))   # features.py
        yTrain.append(vocab.indexOf(secret))

    xTest, yTest = [], []
    for secret in testSecrets:
        evidence = _playProbes(secret, probes)      # this function
        xTest.append(encode(evidence, len(probes)))   # features.py
        yTest.append(vocab.indexOf(secret))

    return Dataset(xTrain, yTrain, xTest, yTest, vocab, len(probes), "compact", "colour")

def _secretRng(seed, secretIndex):
    """A deterministic RNG for one secret.

    WHY PER-SECRET rather than one RNG for the whole run: it makes n rounds a PREFIX of
    n+1 rounds for the same secret. The 1-round dataset and the 5-round dataset then
    describe the same games, merely cut off earlier - which is what makes "accuracy vs
    rounds of evidence" a fair comparison rather than five unrelated datasets.
    """
    return Random((seed if seed is not None else 0) * 1000003 + secretIndex)

def _buildEvidence(secret, rng, count, guessPicker):
    """Build `count` DISTINCT rounds of (guess, pattern) evidence against one secret.

    WHY evidence is built incrementally: a custom picker needs everything played so far
    to choose a sensible next guess. The default picker does not, but passing the growing
    list is what makes a *consistent* or *on-policy* picker possible later without
    changing this module again.

    WHY the default draws uniformly from answers: it is simple, reproducible, and keeps
    the generator honest about what "random training evidence" means (see the distribution
    note in the experiment documentation). It is NOT a perfect imitation of the solver, which only
    plays feedback-consistent words; a custom picker can close that gap.

    WHY a real solver never repeats a guess: the solver masks words already played, so
    training evidence containing repeats would describe games it can never actually play.

    WHY the attempt limit: a guessPicker that keeps returning the same word would
    otherwise loop here FOREVER, and an infinite loop during data generation looks like
    a hang rather than a bug in the picker. Fail loudly instead.
    """
    answers = loadAnswers()
    evidence = []
    seen = set()
    attempts = 0
    limit = 50 * count + 100

    while len(evidence) < count:
        attempts += 1

        if attempts > limit:
            raise ValueError(
                "guessPicker produced only %d distinct guess(es) but %d were needed; "
                "it must return distinct words (%d attempts)"
                % (len(evidence), count, attempts)
            )

        if guessPicker is None:
            guess = answers[rng.randrange(len(answers))]
        else:
            guess = guessPicker(len(evidence), rng, list(evidence))

        if guess in seen:
            continue

        seen.add(guess)
        evidence.append((guess, feedback(guess, secret)))

    return evidence

def generateVariable(nRounds, nTrain, nTest, seed = None, maxRounds = 5,
                     letterMode = "compact", guessPicker = None):
    """Evidence from VARIED guesses, not fixed probes - the Attempt 1 generator.

    WHY THIS EXISTS: generate() builds evidence from a fixed probe list, but the solver
    supplies its OWN guesses at solve time. Training on one and solving with the other
    asks the model about letter/colour combinations it never saw, accuracy collapses, and
    nothing in the code looks wrong. Drawing varied guesses puts the letter/colour
    combinations back inside the training space.

    WHY this is not a perfect imitation: default guesses are drawn uniformly from answers,
    while the solver only plays words consistent with the feedback. That residual mismatch
    is documented in the experiment notes; a custom guessPicker can narrow it.

    WHY maxRounds: every row is padded to maxRounds * roundBlockSize(letterMode), so one
    model can be asked at any point in the game. nRounds controls how much evidence a
    given dataset contains, which is what makes the accuracy-vs-rounds plot possible.

    WHY guessPicker: the default draws random answers, but a custom picker can supply
    consistent guesses or a model's own predictions - which is how you generate on-policy
    data later. Its signature is guessPicker(roundIndex, rng, evidence), where evidence is
    the [(guess, pattern), ...] list built so far.
    """
    if not 1 <= nRounds <= maxRounds:
        raise ValueError("nRounds must be in [1, %d], got %r" % (maxRounds, nRounds))

    answers = loadAnswers()
    vocab = answersVocab()

    trainSecrets, testSecrets = _splitBySecret(answers, nTrain, seed)
    testSecrets = testSecrets[:nTest]

    def encodeSecrets(secrets):
        rows, labels = [], []

        for secret in secrets:
            index = vocab.indexOf(secret)
            rng = _secretRng(seed, index)
            evidence = _buildEvidence(secret, rng, nRounds, guessPicker)

            rows.append(encodeRounds(evidence, maxRounds, letterMode))
            labels.append(index)

        return rows, labels

    xTrain, yTrain = encodeSecrets(trainSecrets)
    xTest, yTest = encodeSecrets(testSecrets)

    return Dataset(xTrain, yTrain, xTest, yTest, vocab, maxRounds, letterMode, "rounds")

def selectTestSecrets(nTest, seed = None):
    """Pick the deterministic subset of secrets used as the class-complete test set.

    WHY the LAST nTest after shuffling: this is the same held-out tail _splitBySecret would
    have produced, so the test secrets stay stable if the protocol changes. They are still
    present in the class-complete TRAINING set (every class must be learnable); what is held
    out is the evidence sequence, not the word.
    """
    answers = loadAnswers()

    if not 0 <= nTest <= len(answers):
        raise ValueError("nTest must be in [0, %d]" % len(answers))

    shuffled = list(answers)
    Random(seed).shuffle(shuffled)

    return shuffled[len(shuffled) - nTest:]

def splitSecrets(nTrain, seed = None):
    """Public wrapper around _splitBySecret for callers outside this module.

    WHY: Attempt 2 and the candidate-scoring experiment need disjoint train/test secrets
    without reaching into a private helper. The guarantee is unchanged.
    """
    return _splitBySecret(loadAnswers(), nTrain, seed)

def evidenceFor(secret, nRounds, seed = None, maxRounds = 5, guessPicker = None):
    """Build nRounds of (guess, pattern) evidence for one secret.

    WHY: callers that need raw evidence (the candidate-scoring feature builder, RL, the
    solver) should not duplicate _buildEvidence + _secretRng plumbing. Using the same
    per-secret RNG keeps n rounds a prefix of n+1 rounds.
    """
    if not 1 <= nRounds <= maxRounds:
        raise ValueError("nRounds must be in [1, %d], got %r" % (maxRounds, nRounds))

    index = answersVocab().indexOf(secret)
    return _buildEvidence(secret, _secretRng(seed, index), nRounds, guessPicker)

def generateClassComplete(nRounds, nTest, seed = None, maxRounds = 5,
                          letterMode = "compact", guessPicker = None, testSeed = None):
    """Attempt-1 generator: every class in train, fresh evidence in test.

    WHY THIS REPLACES A SECRET-SPLIT FOR CLASSIFICATION: a model over 2,315 answer classes
    can only output classes it has learned. If 400 secrets were held out of training, every
    test label would be an unseen class and top-1 accuracy would be identically zero - no
    matter how good the model is. So training covers ALL 2,315 secrets, and the test set
    asks the model to predict the same words from DIFFERENT evidence sequences.

    WHY the test rows use a different seed: they must be unseen examples, not copies of the
    training rows. The function raises if a test row is byte-identical to the training row
    for the same secret, because that would be leakage. Identical rows for DIFFERENT
    secrets are allowed (and are genuine Wordle ambiguity).

    WHY testSeed is separate: it lets the test evidence be regenerated/offset without
    changing the training rows. Attempt 1 uses SEED + 1000 unless told otherwise.
    """
    if not 1 <= nRounds <= maxRounds:
        raise ValueError("nRounds must be in [1, %d], got %r" % (maxRounds, nRounds))

    answers = loadAnswers()
    vocab = answersVocab()
    testSecrets = selectTestSecrets(nTest, seed)

    if testSeed is None:
        testSeed = (seed if seed is not None else 0) + 1000

    xTrain, yTrain = [], []
    trainRowByLabel = {}

    for index, secret in enumerate(answers):
        rng = _secretRng(seed, index)
        evidence = _buildEvidence(secret, rng, nRounds, guessPicker)
        row = encodeRounds(evidence, maxRounds, letterMode)

        xTrain.append(row)
        yTrain.append(index)
        trainRowByLabel[index] = tuple(row)

    xTest, yTest = [], []

    for secret in testSecrets:
        index = vocab.indexOf(secret)
        rng = _secretRng(testSeed, index)
        evidence = _buildEvidence(secret, rng, nRounds, guessPicker)
        row = encodeRounds(evidence, maxRounds, letterMode)

        if tuple(row) == trainRowByLabel[index]:
            raise ValueError(
                "test evidence for %r is identical to its training evidence; pass a "
                "different testSeed (currently %r)" % (secret, testSeed)
            )

        xTest.append(row)
        yTest.append(index)

    return Dataset(xTrain, yTrain, xTest, yTest, vocab, maxRounds, letterMode,
                   "rounds", "evidence")
