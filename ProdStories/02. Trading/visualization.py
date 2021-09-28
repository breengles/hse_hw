import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visualize(data: pd.DataFrame):
    plt.figure(figsize=(12, 12))

    unique_colors = np.random.random((len(data.session_id.unique()), 3))

    for c, ses in enumerate(sorted(data.session_id.unique())):
        d = data[data.session_id == ses]

        plt.plot(d.time, d.price_norm, color=unique_colors[c], label=f"Session: {ses}")

        plt.xlabel("Time")
        plt.ylabel("Price (norm)")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()
