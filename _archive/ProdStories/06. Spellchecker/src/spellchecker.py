from collections import Counter, defaultdict
from csv import reader
from typing import DefaultDict, List, Union

import numpy as np
from pyphonetics import RefinedSoundex
from sklearn.feature_extraction.text import CountVectorizer
from spylls.hunspell import Dictionary
from textdistance import damerau_levenshtein, editex, jaro, needleman_wunsch


class Speller:
    def __init__(self, vocab_path: str = "en_US", word_freq_path: str = "data/unigram_freq.csv") -> None:
        self.vocab = Dictionary.from_files(vocab_path)
        self.word_freq: defaultdict = self.read_word_freq_file(word_freq_path)

        # https://openbase.com/python/pyphonetics
        # метод `distance` предоставляет расстояние Левенштейна между фонетическими представлениями
        self.rs = RefinedSoundex()

    def _get_suggestions(self, word):
        suggestions = self.vocab.suggester.ngram_suggestions(word, set())
        return np.unique([x for x in suggestions])

    def suggest(self, word: str, n_candidates: int = 10) -> List[Union[str, None]]:
        """
        Предсказание спеллера
        :param word: целевое слово, для которого нужно найти исправления
        :param n_candidates: максимальное кол-во кандидатов для исправления
        :return: List[str]: список кандидатов для исправления
        """
        suggestions = self._get_suggestions(word)

        if len(suggestions) == 0:
            return [None]

        features = [self._features(word, suggested_word) for suggested_word in suggestions]

        return suggestions[self._ranking(features)][:n_candidates].tolist()

    def _features(self, word: str, suggested_word: str) -> List[float]:
        """
        Сбор признаков для двух слов по расстояниям: Дамера-Левенштейн, Нидлман-Вунш, Джаро и два фонетических
        :param word: целевое слово, для которого нужно найти исправления
        :param suggested_word: предложенное слово на отбор
        :return: List[float]: список признаков
        """
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
    def _ranking(features: List[List[float]]) -> np.ndarray:
        return np.argsort(np.mean(features, axis=1))

    @staticmethod
    def read_word_freq_file(word_freq_path: str) -> DefaultDict[str, float]:
        word_freq = defaultdict(lambda: 0.0)
        total_counts = 0

        with open(word_freq_path, "r") as f:
            freqs = reader(f, delimiter=",")
            next(freqs)  # пропускаем header
            for word, freq in freqs:
                ifreq = int(freq)
                total_counts += ifreq
                word_freq[word] = ifreq

        for word in word_freq.keys():
            word_freq[word] /= total_counts  # относительные частоты в датасете

        return word_freq


class NGramVecSpeller(Speller):
    def fit(self, ngram_range=(2, 2)):
        """
        Подгонка спеллера
        """
        self.words_list = np.unique([word.stem.lower() for word in self.vocab.dic.words])

        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=ngram_range, binary=True)
        encoded_words = self.vectorizer.fit_transform(self.words_list).tocoo()

        self.index = defaultdict(set)

        # строим словарь, отображающий идентификатор нграммы в множество термов
        for i in zip(encoded_words.row, encoded_words.col):
            self.index[i[1]].add(i[0])

        return self

    def _get_suggestions(self, word):
        char_ngrams_list = self.vectorizer.transform([word]).tocoo().col

        counter = Counter()

        for token_id in char_ngrams_list:
            for word_id in self.index[token_id]:
                counter[word_id] += 1

        suggestions = np.array([self.words_list[suggest[0]] for suggest in counter.most_common(n=20)])

        return suggestions
