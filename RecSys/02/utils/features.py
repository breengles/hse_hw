from .handlers import songs as hs
from .handlers import user as hu


def member(data):
    data = hu.bd(data)
    data = hu.gender(data)
    data = hu.time(data)

    return data


def song(data):
    data = hs.artist(data)
    data = hs.composer(data)
    data = hs.lyricist(data)
    data = hs.genre(data)
    data = hs.isrc(data)

    return data
