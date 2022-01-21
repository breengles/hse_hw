import pandas as pd
import numpy as np


UNK = np.nan


def members(path):
    data = pd.read_csv(
        path,
        dtype={
            "msno": "category",
            "city": "category",
            "bd": np.int32,
            "gender": "category",
            "registered_via": "category",
        },
        parse_dates=["registration_init_time", "expiration_date"],
    )

    return data


def songs(path):
    data = pd.read_csv(
        path,
        dtype={
            "song_id": "category",
            "song_length": np.int32,
            "genre_ids": "category",
            "artist_name": "category",
            "composer": "category",
            "lyricist": "category",
            "language": "category",
        },
    )

    return data


def extra_info(path):
    data = pd.read_csv(
        path,
        dtype={"song_id": "category", "name": "category", "isrc": "string"},
    )

    return data


def train(path):
    data = pd.read_csv(
        path,
        dtype={
            "msno": "category",
            "song_id": "category",
            "source_system_tab": "category",
            "source_screen_name": "category",
            "source_type": "category",
            "target": np.int32,
        },
    )
    return data
