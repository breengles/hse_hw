import pandas as pd
from pandas_profiling import ProfileReport


for d in ["members.csv", "song_extra_info.csv", "songs.csv", "test.csv", "train.csv"]:
    data = pd.read_csv(f"data/{d}")
    profile = ProfileReport(data, minimal=True)
    profile.to_file(f"{d[:-4]}.html")
