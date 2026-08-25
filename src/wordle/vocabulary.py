from pathlib import Path
from functools import cache
from dataclasses import dataclass

@dataclass(frozen=True)
class Vocabulary:
    words: tuple
    index: dict

    def indexOf(self, word):
        return self.index[word]

    def wordAt(self, index):
        return self.words[index]

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def _read(name):
    with open(_DATA_DIR / name) as f:
        return tuple(line.strip() for line in f if line.strip())

#This function return a tuple of all searchable words
@cache
def loadWords():
    return _read("words.txt")
#print(loadWords())

#This function returns a tuple of all the answer words
@cache
def loadAnswers():
    return _read("answers.txt")

#This function returns an object of Vocabulary class that lets the caller access the index of a word or a word at some index
@cache
def answersVocab() -> Vocabulary:
    ans = loadAnswers()
    return Vocabulary(words=ans, index={w: i for i, w in enumerate(ans)})
