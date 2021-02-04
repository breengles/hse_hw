import numpy as np

from typing import List
from dataclasses import dataclass
from dataclasses import field
import pandas as pd
from tabulate import tabulate
from time import time


@dataclass
class Logger:
    start_time: float
    tol: float
    lam: float
    time: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    rk: List[float] = field(default_factory=list)
    call_count: List[int] = field(default_factory=list)
    num_iter: List[int] = field(default_factory=list)
    nonzero: List[int] = field(default_factory=list)

    def add(self, num_iter, call_count, entropy, rk, w):
        self.num_iter.append(num_iter)
        self.entropy.append(entropy[0])
        self.call_count.append(call_count)
        self.time.append(time() - self.start_time)
        self.rk.append(rk.ravel()[0])
        self.nonzero.append(np.sum(w >= 1e-15))
        
    def get_data(self):
        logs = [self.num_iter, self.call_count, self.time, self.lam, self.nonzero, self.entropy, self.rk, self.tol]
        names = ["num_iter", "oracle_calls", "time", "lambda", "nonzero", "entropy", "rk", "tol"]
        return {name: arr for name, arr in zip(names, logs)}

    @property
    def best(self):
        return pd.DataFrame({
            "num_iter": self.num_iter[-1],
            "oracle_calls": self.call_count[-1],
            "time": self.time[-1],
            "lambda": self.lam,
            "nonzero": self.nonzero[-1],
            "entropy": self.entropy[-1],
            "rk": self.rk[-1],
            "tol": self.tol
        }, index=[0])

    @property
    def dataframe(self):
        return pd.DataFrame.from_dict(self.get_data())

    @property
    def table(self):
        table = [self.dataframe.columns.values.tolist()] + self.dataframe.values.tolist()
        return tabulate(table,
                       headers="firstrow",
                       tablefmt="github",
                       floatfmt=["", "", ".0e", ".2e", ".2e", ".4f", "", "", ".4f", ".1e"])
