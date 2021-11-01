#!/usr/bin/env python

from src.evaluation import evaluate
from src.spellchecker import SpellChecker

if __name__ == "__main__":
    checker = SpellChecker()

    with open("data/test.txt", "r") as test_file:
        test_data = [line.strip().split("\t") for line in test_file.readlines()]

    for k in (1, 3, 5, 10):
        res = evaluate(checker, test_data, k=k)
        print(f"Accuracy @{k}: {res[0]:0.2f}")
        print(f"Median place in suggestion: {res[1]}")
