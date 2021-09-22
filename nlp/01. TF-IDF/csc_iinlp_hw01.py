#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Домашнее задание #1

    К сожалению, у нас нет "истинных" "наиболее описательных" слов для датасета.
    В этом задании от вас требуется
    1) задать правильную формулу TF.IDF в функции calc_token_weight,
    2) передать в неё правильные параметры.

    Формула -- почти такая же, как на первом слайде о tf-idf.
    Единственное отличие -- надо увеличить idf на единицу.

    В этом задании не очень много смысла, но это домашнее задание ознакомительное --
    чтобы у всех сдающих было всё настроено + чтобы мы отладили процесс сдачи работ и поняли,
    сколько времени у преподавателей уходит на проверку.

    CSC, IINLP-2021
"""

import math
import re

from sklearn import datasets
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def count_tokens(documents):
    """
    Строим словари (хэшмэпы, ассоциативные массивы)
    разных частот для последующего вычисления tf-idf

    Мы не используем никаких трюков и считаем явно.

    :param documents: сырые тексты
    :return: кортеж-тройка:
    docs_with_token,            # словарь: docs_with_token[токен] = число содержащих его документов
    token_hits_count_in_doc,    # словарь: token_hits_count_in_doc[документ][токен] = сколько раз токен встретился в данном документе
    token_count_in_doc          # словарь: token_count_in_doc[документ] = сколько всего токенов в документе
    """

    token_pattern = re.compile(r"(?u)\b\w\w+\b")

    docs_with_token = {}
    token_hits_count_in_doc = []
    token_count_in_doc = []

    for doc in documents:

        # частоты токенов в этом документе
        # NB! здесь бывает удобно использовать defaultdict (как и для docs_with_token)
        token_dictionary = {}

        token_count = 0

        # NB! здесь и далее для подсчёта частот различных элементов
        # пригодится бы Counter из collections -- рекомендуем посмотреть
        # в сети примеры использования, если не знали о его существовании
        for token in token_pattern.findall(doc):

            token = token.lower()

            # отбрасываем стоп-слова
            if token in ENGLISH_STOP_WORDS:
                continue

            # считаем частоты отдельных токенов
            if token in token_dictionary:
                token_dictionary[token] += 1.0
            else:
                token_dictionary[token] = 1.0

            token_count += 1.0

        token_hits_count_in_doc.append(token_dictionary)
        token_count_in_doc.append(token_count)

        # для каждого токена ведём учёт числа документов,
        # в которых он встретился
        for token in token_dictionary:

            if token in docs_with_token:
                docs_with_token[token] += 1.0
            else:
                docs_with_token[token] = 1.0

    return docs_with_token, token_hits_count_in_doc, token_count_in_doc


def calc_token_weight(document_count, docs_with_token, tokens_in_doc, tokens_total):
    """
    Функция, вычисляющая tf-idf, для которого всё подготовлено.
    Формула -- почти та же, что на первом слайде про tf-idf, но к idf надо прибавить единицу.
    Используйте math.log.

    :param document_count: общее число документов
    :param docs_with_token: число документов
    :param tokens_in_doc: сколько раз токен встретился в документе
    :param tokens_total: общее число токенов в данном документе
    :return: tf.idf
    """

    # YOUR CODE HERE
    tf = tokens_in_doc / tokens_total

    # log(...) + 1 heuristic
    idf = math.log(document_count / docs_with_token) + 1

    return tf * idf


if __name__ == "__main__":

    # "истинные" наиболее представительные слова в первых 10 документах
    reference = [
        "cheaper supply orbits reach economies c5hcbo alike allen ground repairstation",
        "reston overhead wrap allen leadership fee integration dennis centers nasa",
        "revolt grasp alaska files acad3 prograsm geta cshow autocad 124722",
        "list rankings krumenaker 71160 2356 source larry compuserve traffic unsubscribed",
        "class foundation banquet teachers dinner teaching studies space lichtenberg planetary",
        "access muc hicksville flaking expecting redundancy navstar bird pat digex",
        "lehigh children abominable tfv0 ucdavis wealth starving capital games dan",
        "processors silicon lower slower ssrt higher access future hjistorical germanium",
        "wpi maverick giaquinto worcester novice 2280 01609 information shuttles periodicals",
        "accelerations henry toronto acceleration andrew immersion generalizes endured efpx7ws00uh7qaop1s zoology",
    ]

    CHECKED_DOCS = len(reference)
    CHECKED_TOP = len(reference[0].split(" "))

    # один из стандартных наборов данных для классификации текстов
    newsgroups = datasets.fetch_20newsgroups(subset="all", categories=["sci.space"])

    # тексты
    documents = newsgroups.data

    # общее число документов в наборе
    docs_count = len(documents)

    # см. документацию к функции
    docs_with_token, token_hits_count_in_doc, token_count_in_doc = count_tokens(documents)

    precision_accumulator = 0.0

    # посчитаем топ по tf-idf для первых десяти документов
    for n_doc in range(CHECKED_DOCS):
        token_cnt = token_count_in_doc[n_doc]

        token_weights = {}

        for token, cnt in token_hits_count_in_doc[n_doc].items():
            token_weights[token] = calc_token_weight(
                docs_count,
                # YOUR CODE HERE
                # используйте уже вычисленные значения, записанные в словари
                # число документов, содержащих token
                docs_with_token=docs_with_token[token],
                # сколько раз встретился токен token в документе n_doc
                tokens_in_doc=cnt,
                # сколько токенов в документе n_doc
                tokens_total=token_cnt,
            )

        # сортируем по tf-idf
        desc_tfidf_tokens = sorted(token_weights.items(), key=lambda x: (x[1], x[0]), reverse=True)
        tfidf_top = list(map(lambda x: x[0], desc_tfidf_tokens))[:CHECKED_TOP]

        # можно распечатать и посмотреть, отличается ли порядок (не должен)
        # print(" ".join(tfidf_top))
        # print(reference[n_doc])

        # вычисляем что-то вроде Precision@10
        tfidf_top_set = set(tfidf_top)
        in_reference = len(tfidf_top_set.intersection(set(reference[n_doc].split(" "))))
        precision = in_reference / CHECKED_TOP

        print(n_doc, "Precision: %.1f%%" % (precision * 100.0))

        precision_accumulator += precision

    # необходимое, но не достаточное условие: у правильного решения здесь должно быть ровно 100%
    print("Avg precision: %.2f%%" % (float(precision_accumulator) / CHECKED_DOCS * 100.0))
