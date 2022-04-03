import numpy as np
from .constants import UNK


SEPS = ("|", "&", "and", "feat", "\\", "/", ";")


def count(value, seps=SEPS):
    if value == UNK:
        return 0

    return 1 + sum(map(value.count, seps))


def artist(data):
    data.artist_name = data.artist_name.cat.add_categories(UNK).fillna(UNK)
    data["artist_count"] = data.artist_name.apply(count).astype(np.int32)

    return data


def composer(data):
    data.composer = data.composer.cat.add_categories(UNK).fillna(UNK)
    data["composer_count"] = data.composer.apply(count).astype(np.int32)

    return data


def lyricist(data):
    data.lyricist = data.lyricist.cat.add_categories(UNK).fillna(UNK)
    data["lyricist_count"] = data.lyricist.apply(count).astype(np.int32)

    return data


def genre(data):
    data.genre_ids = data.genre_ids.cat.add_categories(UNK).fillna(UNK)
    data["genre_count"] = data.genre_ids.apply(count).astype(np.int32)

    return data


def isrc(data):
    def parse(value):
        country = value[:2]
        reg_id = value[2:5]
        reg_year = value[5:7]
        uniq_id = value[7:]

        return country, reg_id, int(reg_year), uniq_id

    def year(value):
        if value == UNK:
            return 0

        return parse(value)[2]

    def country(value):
        if value == UNK:
            return UNK

        return parse(value)[0]

    data.isrc = data.isrc.fillna(UNK)
    data["isrc_year"] = data.isrc.apply(year).astype("category")
    data["isrc_country"] = data.isrc.apply(country).astype("category")

    data.drop(columns=["isrc"], inplace=True)

    return data
