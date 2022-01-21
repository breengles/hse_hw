#!/usr/bin/env python

from src.evaluation import evaluate
from src.spellchecker import NGramVecSpeller, Speller

if __name__ == "__main__":
    with open("data/test.txt", "r") as test_file:
        test_data = [line.strip().split("\t") for line in test_file.readlines()]

    checker = NGramVecSpeller()
    checker.fit((2, 2))
    res = evaluate(checker, test_data)
    print(f"\n{res}")

    checker = Speller()
    res = evaluate(checker, test_data)
    print(f"\n{res}")
