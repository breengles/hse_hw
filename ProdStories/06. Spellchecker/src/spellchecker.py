from collections import defaultdict
from csv import reader
from typing import List, Union, DefaultDict

import numpy as np
from pyphonetics import RefinedSoundex
from spylls.hunspell import Dictionary
from textdistance import damerau_levenshtein, jaro, needleman_wunsch, editex


class SpellChecker:
    def __init__(self, vocab_path: str = "en_US", word_freq_path: str = "data/unigram_freq.csv") -> None:
        self.vocab = Dictionary.from_files(vocab_path)
        self.word_freq: defaultdict = self.__read_word_freq_file(word_freq_path)

        # see https://openbase.com/python/pyphonetics
        # its method `distance` provides distance between two phonetic repr
        self.rs = RefinedSoundex()

    def suggest(self, word: str, n_gram: bool = True) -> List[Union[str, None]]:
        if n_gram:
            suggestions = self.vocab.suggester.ngram_suggestions(word, set())
        else:
            suggestions = self.vocab.suggest(word)

        suggestions = np.array([x for x in suggestions])

        if len(suggestions) == 0:
            return [None]

        features = [self.__features(word, suggested_word) for suggested_word in suggestions]

        return suggestions[self.__ranking(features)].tolist()

    def __features(self, word: str, suggested_word: str) -> List[float]:
        damerau_levenshtein_distance = damerau_levenshtein(word, suggested_word)
        needleman_wunsch_distance = needleman_wunsch.normalized_distance(word, suggested_word)
        jaro_distance = jaro.normalized_distance(word, suggested_word)

        phonetic_distance_rs = self.rs.distance(word, suggested_word)
        phonetic_distance_editex = editex(word, suggested_word)

        freq = self.word_freq[suggested_word]

        return [
            (phonetic_distance_rs + phonetic_distance_editex) / 2,
            damerau_levenshtein_distance,
            needleman_wunsch_distance,
            jaro_distance,
            1 - freq,
        ]

    @staticmethod
    def __ranking(features: List[List[float]]) -> np.ndarray:
        return np.argsort(np.mean(features, axis=1))

    @staticmethod
    def __read_word_freq_file(word_freq_path: str) -> DefaultDict[str, float]:
        word_freq = defaultdict(lambda: 0.0)
        total_counts = 0

        with open(word_freq_path, "r") as f:
            freqs = reader(f, delimiter=",")
            next(freqs)  # skipping header
            for word, freq in freqs:
                ifreq = int(freq)
                total_counts += ifreq
                word_freq[word] = ifreq

        for word in word_freq.keys():
            word_freq[word] /= total_counts  # relative frequency for the provided file

        return word_freq
