from pathlib import Path
from functools import cache

#This function converts the words in text file to a tuple of strings.
@cache
def words():
    vocabularyDir = Path(__file__).resolve().parent

    filePath = vocabularyDir.parent / "data" / "words.txt"

    with open(filePath, "r") as file:
        wordsTuple = tuple(line.strip() for line in file if line.strip())

    return wordsTuple

#print(words())

#This function converts the answers in text file to a tuple of strings.
@cache
def answers():
    vocabularyDir = Path(__file__).resolve().parent

    filePath = vocabularyDir.parent / "data" / "answers.txt"

    with open(filePath, "r") as file:
        answersTuple = tuple(line.strip() for line in file if line.strip())

    return answersTuple

#print(answers())
#print(type(answers()[0]))
