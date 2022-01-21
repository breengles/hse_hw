import numpy as np
from .constants import UNK


def bd(data):
    outliers = (data.bd < 5) | (data.bd > 90)
    child = (data.bd >= 10) & (data.bd < 15)
    teenager = (data.bd >= 15) & (data.bd < 20)
    young = (data.bd >= 20) & (data.bd < 30)
    middle = (data.bd >= 30) & (data.bd < 40)
    quiteold = (data.bd >= 40) & (data.bd < 50)
    prettyold = (data.bd >= 50) & (data.bd < 60)
    old = (data.bd >= 60) & (data.bd <= 90)

    data.loc[outliers, "bd"] = UNK
    data.loc[child, "bd"] = "child"
    data.loc[teenager, "bd"] = "teenager"
    data.loc[young, "bd"] = "young"
    data.loc[middle, "bd"] = "middle"
    data.loc[quiteold, "bd"] = "quiteold"
    data.loc[prettyold, "bd"] = "prettyold"
    data.loc[old, "bd"] = "old"

    return data.astype({"bd": "category"})


def gender(data):
    data.gender = data.gender.cat.add_categories(UNK).fillna(UNK)

    return data


def time(data):
    data["registration_init_year"] = data.registration_init_time.apply(lambda x: x.year).astype("category")
    data["registration_init_month"] = data.registration_init_time.apply(lambda x: x.month).astype("category")
    data["registration_init_day"] = data.registration_init_time.apply(lambda x: x.day).astype("category")

    data["expiration_year"] = data.expiration_date.apply(lambda x: x.year).astype("category")
    data["expiration_month"] = data.expiration_date.apply(lambda x: x.month).astype("category")
    data["expiration_day"] = data.expiration_date.apply(lambda x: x.day).astype("category")

    data.drop(columns=["registration_init_time", "expiration_date"], inplace=True)

    return data
