from genericpath import exists
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from typing import Tuple
import os


class Logger:
    def __init__(self, params):
        self.params = params
        self.history = {}

    def log(self, key, value):
        try:
            self.history[key].append(value)
        except KeyError:
            self.history[key] = [value]

    def save_params(self, file_path: str, mode: str = "a+"):
        with open(file_path, mode) as f:
            yaml.dump(self.params, f, indent=4)

    def save(self, file_path: str, mode: str = "w+", dir_path: str = "log/"):
        os.makedirs(dir_path, exist_ok=True)
        pd.DataFrame(self.history).to_csv(dir_path + file_path, mode=mode, index=False)

    def plot(
        self,
        x: str,
        y: str,
        std: str = None,
        y_solved: float = None,
        size: Tuple[int, int] = (12, 8),
        title: str = None,
        label: str = None,
        x_label: str = None,
        y_label: str = None,
        alpha: float = 0.5,
    ):
        fig, ax = plt.subplots(figsize=size)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        x_ = np.array(self.history[x])
        y_ = np.array(self.history[y])
        std_ = np.array(self.history[std]) if std is not None else 0

        if y_solved is not None:
            plt.hlines(y_solved, np.min(x_), np.max(x_), colors="r", label="Solved")

        plt.plot(x_, y_, label=label)
        plt.fill_between(x_, y_ - std_, y_ + std_, alpha=alpha)

        plt.legend()
        plt.show()
