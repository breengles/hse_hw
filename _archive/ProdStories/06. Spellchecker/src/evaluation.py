from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from src.spellchecker import Speller


def metric(word: str, suggestions: List[str], k: int = 10) -> Tuple[bool, Union[int, None]]:
    top_k = word in suggestions[:k]
    mean_place = suggestions.index(word) + 1 if word in suggestions else None
    return top_k, mean_place


def evaluate(sc: Speller, test_data: List[List[str]], top_k: Tuple[int] = (1, 3, 5, 10)) -> Dict[int, float]:
    results = {}
    results[-1] = []

    mean_places = []
    for k in top_k:
        results[k] = 0.0

    for misspelled_word, gt_word in tqdm(test_data):
        misspelled_word, gt_word = misspelled_word.lower(), gt_word.lower()
        suggestions = sc.suggest(misspelled_word, n_candidates=max(top_k))

        for k in top_k:
            results[-1] = []

            top_k_, mean_place_ = metric(gt_word, suggestions, k=k)

            if mean_place_ is not None and k == max(top_k):
                mean_places.append(mean_place_)

            results[k] += top_k_

    for k in top_k:
        results[k] /= len(test_data)

    results[-1] = np.median(mean_places)

    return results
