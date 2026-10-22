"""The episode loop that all three algorithms share.

WHY ONE LOOP: the three algorithms differ in their *loss*, not in how a game is played. If
each of them had its own episode loop, a difference in the reported learning curves could
be a difference in the loop - a different observation, a different mask, a different
reward - rather than a difference in the algorithm. So the loop lives here, once, and the
algorithms supply only the update rule.

WHY A "CHOOSER" CALLBACK: each algorithm needs different extra numbers from the moment it
picks an action - REINFORCE needs the log-probability and the value, DQN needs the Q value.
The chooser returns the action plus those extras, so the loop can collect them without
knowing which algorithm is driving.
"""

from dataclasses import dataclass, field

from attempt2.eval import PolicyState


@dataclass
class Choice:
    """What a chooser decided, plus whatever the algorithm needs to remember."""

    action: int
    extras: dict = field(default_factory=dict)


@dataclass
class Transition:
    """One step of one episode."""

    observation: object
    mask: object
    action: int
    reward: float
    nextObservation: object
    nextMask: object
    done: bool
    extras: dict = field(default_factory=dict)


@dataclass
class Episode:
    """One finished game."""

    transitions: list
    totalReward: float
    won: bool
    guessesUsed: int
    firstGuess: str


def collectEpisode(env, chooser, seed=None):
    """Play one game and return what happened.

    `chooser(state, generator) -> Choice` is called once per turn. The generator is passed
    through so that the sampling in the chooser comes from the one seeded stream rather
    than from a fresh random source each turn.
    """
    observation, _ = env.reset(seed=seed)

    transitions = []
    totalReward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        mask = env.action_mask()
        state = PolicyState(observation, mask, tuple(env.history))

        choice = chooser(state, None)
        action = int(choice.action)

        nextObservation, reward, terminated, truncated, _ = env.step(action)

        transitions.append(Transition(
            observation=observation,
            mask=mask,
            action=action,
            reward=reward,
            nextObservation=nextObservation,
            nextMask=env.action_mask(),
            # WHY BOTH ENDS COUNT AS "DONE": the game is over either way - the guess was
            # the secret, or the six guesses ran out - so there is no future value to
            # bootstrap from in either case. This is not the usual "a time limit is not a
            # terminal state" situation: here the limit IS the end of the game.
            done=terminated or truncated,
            extras=choice.extras,
        ))

        observation = nextObservation
        totalReward += reward

    return Episode(
        transitions=transitions,
        totalReward=totalReward,
        won=terminated,
        guessesUsed=env.guessesUsed,
        firstGuess=env.history[0][0],
    )


def episodeStream(env, chooser, episodes, seedBase=0):
    """Yield `(index, Episode)` for `episodes` episodes.

    WHY A SEEDED SEQUENCE: every episode gets its own seed, so a run is reproducible from
    the start seed alone and the environment's secret choice is not correlated with the
    policy's exploration.
    """
    for index in range(episodes):
        yield index, collectEpisode(env, chooser, seed=seedBase + index)


def discountedReturns(rewards, discount):
    """The discounted return of each step, computed backwards.

    WHY THE DISCOUNT IS NOT 1: with a step penalty, an undiscounted return makes a long
    game almost as good as a short one and the agent has little reason to hurry. A discount
    below 1 makes an early win worth more than a late one.
    """
    returns = []
    accumulated = 0.0

    for reward in reversed(rewards):
        accumulated = reward + discount * accumulated
        returns.append(accumulated)

    returns.reverse()
    return returns
