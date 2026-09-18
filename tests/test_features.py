"""Tests for features.py.

The point of `encode` is a *fixed-length* numeric vector, so most of these
assert exact length and content rather than just "it ran".
"""

from wordle.features import encode


def testEncodeSingleProbeHasFiveFeatures():
    assert encode([("crane", (2, 0, 1, 0, 0))], 1) == [2, 0, 1, 0, 0]


def testEncodeConcatenatesProbePatternsInOrder():
    probes = [
        ("crane", (2, 0, 1, 0, 0)),
        ("slate", (0, 2, 0, 1, 0)),
    ]
    assert encode(probes, 2) == [2, 0, 1, 0, 0, 0, 2, 0, 1, 0]


def testEncodePadsWithMinusOneUpToRequestedLength():
    probes = [
        ("crane", (2, 0, 1, 0, 0)),
        ("slate", (0, 2, 0, 1, 0)),
    ]
    result = encode(probes, 3)
    assert result == [2, 0, 1, 0, 0, 0, 2, 0, 1, 0, -1, -1, -1, -1, -1]


def testEncodeLengthIsAlwaysProbesTimesFive():
    for nProbes in (1, 2, 3, 4, 5):
        probes = [("crane", (2, 2, 2, 2, 2))] * nProbes
        assert len(encode(probes, nProbes)) == nProbes * 5
        assert len(encode(probes[:1], nProbes)) == nProbes * 5


def testEncodeWithNoEvidenceIsAllPadding():
    assert encode([], 2) == [-1] * 10


def testEncodePaddingIsDistinguishableFromGray():
    """-1 must not collide with a real gray code (0)."""
    padded = encode([("crane", (0, 0, 0, 0, 0))], 2)
    assert padded == [0, 0, 0, 0, 0, -1, -1, -1, -1, -1]


def testEncodeProducesOnlyValidColorCodesAndPadding():
    result = encode([("crane", (2, 1, 0, 1, 2))], 3)
    assert all(code in (-1, 0, 1, 2) for code in result)


def testEncodeIsPure():
    probes = [("crane", (2, 0, 1, 0, 0)), ("slate", (0, 2, 0, 1, 0))]
    assert encode(probes, 2) == encode(probes, 2)


def testEncodeDoesNotMutateItsInput():
    probes = [("crane", (2, 0, 1, 0, 0))]
    before = list(probes)
    encode(probes, 3)
    assert probes == before


def testEncodeIgnoresTheGuessWordItself():
    """Colour-only encoding: the same pattern from different words matches."""
    assert encode([("crane", (1, 2, 0, 0, 0))], 1) == encode(
        [("slate", (1, 2, 0, 0, 0))], 1
    )
