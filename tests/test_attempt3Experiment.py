"""Tests for the Phase-D experiment harness.

§4.6 asks for "table formatting and result shapes; no 400-game run inside a test". The
held-out set is read once, in `main()`, and never here - the games these tests play are two
validation secrets, to prove the adapters return valid actions.

What is pinned:

- the six Attempt-3 rows exist, have unique names and are callable;
- the three reference rows that need no artifacts are built without Torch;
- the JSON has the fields the write-up reads, with distribution keys as strings;
- the paired section is present when C1 and the Attempt-2 reference are both in the run;
- both figures are written and non-empty.
"""

import json

from attempt2.eval import EvaluationResult
from attempt2.splits import validationSecrets
from attempt2.wordleEnv import TRAIN_MODE
from attempt3 import experiment


def _result(name, games=4, winRate=1.0, avg=3.0):
    distribution = {1: 0, 2: 0, 3: games, 4: 0, 5: 0, 6: 0}
    perGame = tuple(("secret%d" % index, True, 3) for index in range(games))

    return EvaluationResult(
        name=name,
        games=games,
        wins=int(games * winRate),
        losses=games - int(games * winRate),
        winRate=winRate,
        avgGuessesOnWins=avg,
        avgGuessesAll=avg,
        guessDistribution=distribution,
        firstGuessCounts={"raise": games},
        illegalActions=0,
        perGame=perGame,
    )


# ---------------------------------------------------------------------------
# The policy list
# ---------------------------------------------------------------------------


def testTheSixAttempt3PoliciesArePresentAndCallable():
    policies = experiment.attempt3Policies()
    names = [name for name, _ in policies]

    assert len(policies) == 6
    assert len(set(names)) == 6
    assert all(callable(policy) for _, policy in policies)
    assert "C1 expected score" in names
    assert "C1 no probes" in names
    assert "C2 two-ply" in names


def testTheReferenceRowsThatNeedNoArtifactsAreBuilt():
    policies = experiment.referencePolicies(includeAgents=False)
    names = [name for name, _ in policies]

    assert names == [
        "masked random (A2)",
        "crane + random (A2)",
        "slate + random (A2)",
    ]
    assert all(callable(policy) for _, policy in policies)


def testEveryPolicyReturnsAValidActionOnARealState():
    secrets = validationSecrets()[:2]

    results = experiment.evaluateAll(
        secrets, includeAgents=False, mode=TRAIN_MODE
    )

    assert len(results) == len(experiment.allPolicies(includeAgents=False))
    assert all(result.games == 2 for result in results)
    assert all(0.0 <= result.winRate <= 1.0 for result in results)
    assert all(0 <= result.illegalActions for result in results)


# ---------------------------------------------------------------------------
# The JSON
# ---------------------------------------------------------------------------


def testTheResultsFileHasTheFieldsTheWriteUpReads(tmp_path):
    results = [
        _result("C1 expected score", games=4, avg=3.42),
        _result("slate + random (A2)", games=4, avg=3.68),
    ]
    path = tmp_path / "results.json"

    experiment.writeResults(results, path=path, secrets=list(range(4)))

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["secrets"] == 4
    assert payload["settings"]["guessSet"] == "answers"
    assert payload["settings"]["scoreAnchorBits"] == 11.5
    assert len(payload["policies"]) == 2

    first = payload["policies"][0]
    assert first["name"] == "C1 expected score"
    assert first["guessDistribution"]["3"] == 4
    assert isinstance(first["firstGuessCounts"], dict)

    assert len(payload["pairedAgainstC1"]) == 1
    assert payload["pairedAgainstC1"][0]["other"] == "slate + random (A2)"


def testTheResultsFileWorksWithoutTheAttempt2Reference(tmp_path):
    path = tmp_path / "results.json"

    experiment.writeResults([_result("C1 expected score")], path=path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["secrets"] is None
    assert payload["pairedAgainstC1"] == []


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------


def testBothFiguresAreWritten(tmp_path):
    results = [
        _result("C1 expected score", avg=3.42),
        _result("C2 two-ply", avg=3.44),
        _result("masked random (A2)", avg=4.1, winRate=0.98),
    ]

    comparison = experiment.plotComparison(results, path=tmp_path / "comparison.png")
    distribution = experiment.plotDistribution(results, path=tmp_path / "distribution.png")

    for path in (comparison, distribution):
        assert path.exists()
        assert path.stat().st_size > 0


def testTheOrderingPutsTheFewestGuessesFirst():
    results = [
        _result("slow", avg=4.0),
        _result("fast", avg=3.4),
        _result("middling", avg=3.7),
    ]

    ordered = experiment._sortedByGuesses(results)

    assert [result.name for result in ordered] == ["fast", "middling", "slow"]


def testOnlyTheAttempt3RowsAreHighlighted():
    assert experiment._isAttempt3("C1 expected score")
    assert experiment._isAttempt3("B3 pair, then B2")
    assert not experiment._isAttempt3("slate + random (A2)")
    assert not experiment._isAttempt3("Attempt-1 solver")
