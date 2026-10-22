"""Tests for the Attempt-2 Gymnasium environment.

The environment is the only component that can leak the held-out secrets, so the most
important tests here are the ones that watch it sample secrets.
"""

import numpy as np
import pytest
from gymnasium import spaces

from attempt2 import config
from attempt2.splits import heldOutSecrets, trainSecrets
from attempt2.wordleEnv import EVAL_MODE, TRAIN_MODE, WordleEnv
from wordle.vocabulary import answersVocab, loadAnswers


def _index(word):
    return answersVocab().indexOf(word)


def testSpacesMatchTheDesign():
    env = WordleEnv()

    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == len(loadAnswers())
    assert env.observation_space.shape == (60,)
    assert env.observation_space.dtype == np.float32


def testUnknownModeIsRejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        WordleEnv(mode="cheating")


def testResetReturnsAnObservationInTheObservationSpace():
    env = WordleEnv()
    observation, info = env.reset(seed=0)

    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["round"] == 0
    assert info["candidates"] == len(loadAnswers())


def testTrainingNeverSamplesAHeldOutSecret():
    env = WordleEnv(mode=TRAIN_MODE)
    heldOut = set(heldOutSecrets())

    for seed in range(50):
        env.reset(seed=seed)
        assert env.secret not in heldOut


def testEvaluationOnlySamplesHeldOutSecrets():
    env = WordleEnv(mode=EVAL_MODE)
    training = set(trainSecrets())

    for seed in range(50):
        env.reset(seed=seed)
        assert env.secret not in training


def testTheTwoModesSampleDifferentPools():
    trainEnv = WordleEnv(mode=TRAIN_MODE)
    evalEnv = WordleEnv(mode=EVAL_MODE)

    trainSecretsSeen = set()
    evalSecretsSeen = set()

    for seed in range(50):
        trainEnv.reset(seed=seed)
        trainSecretsSeen.add(trainEnv.secret)
        evalEnv.reset(seed=seed)
        evalSecretsSeen.add(evalEnv.secret)

    assert trainSecretsSeen.isdisjoint(evalSecretsSeen)


def testTheSameSeedStartsTheSameGame():
    first = WordleEnv()
    second = WordleEnv()

    assert first.reset(seed=7)[0].tolist() == second.reset(seed=7)[0].tolist()
    assert first.secret == second.secret


def testDifferentSeedsGiveDifferentSecrets():
    env = WordleEnv()
    seen = set()

    for seed in range(50):
        env.reset(seed=seed)
        seen.add(env.secret)

    assert len(seen) > 40


def testGuessingTheSecretWins():
    env = WordleEnv()
    env.reset(seed=1)
    action = _index(env.secret)

    observation, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["solved"] is True
    assert info["guess"] == env.secret
    # +1 for the win, minus the cost of the one guess that was played.
    assert reward == pytest.approx(config.WIN_REWARD + config.STEP_PENALTY)


def testSixWrongGuessesLose():
    env = WordleEnv()
    env.reset(seed=2)

    action = _index("crane") if env.secret != "crane" else _index("slate")
    reward = 0.0
    terminated = truncated = False

    for turn in range(config.MAX_GUESSES):
        _, reward, terminated, truncated, info = env.step(action)

    assert terminated is False
    assert truncated is True
    assert info["round"] == config.MAX_GUESSES
    # No win, no loss bonus: the return is just the step penalties.
    assert reward == pytest.approx(config.LOSS_REWARD + config.STEP_PENALTY)


def testTheEpisodeRecordsEveryRound():
    env = WordleEnv()
    env.reset(seed=3)

    env.step(_index("crane"))

    assert env.guessesUsed == 1
    assert env.history[0][0] == "crane"
    assert len(env.history[0][1]) == 5


def testTheMaskShrinksAndNeverIncludesThePlayedWord():
    env = WordleEnv()
    env.reset(seed=4)

    action = _index("crane")
    env.step(action)
    mask = env.action_mask()

    assert mask.shape == (len(loadAnswers()),)
    assert not mask[action]
    assert mask.sum() < len(loadAnswers())


def testAnIllegalActionIsRejectedLoudly():
    env = WordleEnv()
    env.reset(seed=5)

    with pytest.raises(ValueError, match="action must be an answer index"):
        env.step(len(loadAnswers()))

    with pytest.raises(ValueError, match="action must be an answer index"):
        env.step(-1)


def testResetCanPlayASpecificSecret():
    env = WordleEnv()
    secret = trainSecrets()[0]

    env.reset(seed=0, options={"secret": secret})

    assert env.secret == secret


def testAForcedSecretIsCheckedAgainstTheModePool():
    # The leak guard has to hold for forced secrets too, not just for sampled ones.
    trainEnv = WordleEnv(mode=TRAIN_MODE)
    evalEnv = WordleEnv(mode=EVAL_MODE)

    with pytest.raises(ValueError, match="not in the train secret pool"):
        trainEnv.reset(seed=0, options={"secret": heldOutSecrets()[0]})

    with pytest.raises(ValueError, match="not in the eval secret pool"):
        evalEnv.reset(seed=0, options={"secret": trainSecrets()[0]})


def testAForcedSecretMustBeAnAnswer():
    env = WordleEnv()

    with pytest.raises(ValueError, match="not in the train secret pool"):
        env.reset(seed=0, options={"secret": "notaword"})


def testSteppingAfterTheEpisodeEndsIsRejectedLoudly():
    env = WordleEnv()
    env.reset(seed=6)
    env.step(_index(env.secret))

    with pytest.raises(RuntimeError, match="episode is already over"):
        env.step(_index("crane"))


def testTheEnvironmentPassesTheGymnasiumChecker():
    from gymnasium.utils.env_checker import check_env

    check_env(WordleEnv(), skip_render_check=True)
