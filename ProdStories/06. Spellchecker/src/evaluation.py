from typing import List, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from src.spellchecker import SpellChecker


def metric(word: str, suggestions: List[str], k: int = 10) -> Tuple[bool, Union[int, None]]:
    top_k = word in suggestions[:k]
    mean_place = suggestions.index(word) + 1 if word in suggestions else None
    return top_k, mean_place


def evaluate(sc: SpellChecker, test_data: List[List[str]], k: int = 10) -> Tuple[float, int]:
    top_k = 0.0
    mean_places = []

    for misspelled_word, gt_word in tqdm(test_data):
        misspelled_word, gt_word = misspelled_word.lower(), gt_word.lower()
        suggestions = sc.suggest(misspelled_word)
        top_k_, mean_place_ = metric(gt_word, suggestions, k=k)

        if mean_place_ is not None:
            mean_places.append(mean_place_)

        top_k += top_k_

    top_k /= len(test_data)

    return top_k, np.median(mean_places).item()
