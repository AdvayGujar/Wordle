"""Tests for vocabulary.py -- word lists and the word<->index mapping."""

from wordle.vocabulary import Vocabulary, answersVocab, loadAnswers, loadWords


def testLoadWordsReturnsTheFullLegalGuessList():
    assert len(loadWords()) == 14855


def testLoadAnswersReturnsThePossibleSecrets():
    assert len(loadAnswers()) == 2315


def testEveryAnswerIsAlsoALegalGuess():
    assert set(loadAnswers()).issubset(set(loadWords()))


def testAllWordsAreLowercaseAndFiveLetters():
    for word in loadWords():
        assert len(word) == 5
        assert word.islower()


def testLoadsAreCachedToTheSameObject():
    assert loadWords() is loadWords()
    assert loadAnswers() is loadAnswers()


def testAnswersVocabIsCached():
    assert answersVocab() is answersVocab()


def testVocabIndexRoundTrips():
    vocab = answersVocab()
    for i in (0, 1, 100, len(vocab.words) - 1):
        assert vocab.indexOf(vocab.wordAt(i)) == i


def testVocabCoversEveryAnswerExactlyOnce():
    vocab = answersVocab()
    assert len(vocab.words) == len(loadAnswers())
    assert len(vocab.index) == len(loadAnswers())
    assert set(vocab.index) == set(loadAnswers())


def testKnownFirstAndLastAnswers():
    vocab = answersVocab()
    assert vocab.wordAt(0) == "aback"
    assert vocab.indexOf("aback") == 0


def testVocabIsFrozenAgainstAttributeRebinding():
    import pytest

    vocab = answersVocab()
    with pytest.raises(Exception):
        vocab.words = ("nope",)


def testVocabularyCanBeBuiltOverAnArbitraryWordList():
    vocab = Vocabulary(words=("apple", "crane"), index={"apple": 0, "crane": 1})
    assert vocab.indexOf("crane") == 1
    assert vocab.wordAt(0) == "apple"
