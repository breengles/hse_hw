import numpy as np

from typing import List
from dataclasses import dataclass
from dataclasses import field
import pandas as pd


@dataclass
class Logger:
    opt_method: str
    linesearch_method: str
    start_time: float
    tol: float
    c1: float
    c2: float
    hf_criterion: str = None
    history_size: int = None
    time: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    rk: List[float] = field(default_factory=list)
    call_count: List[int] = field(default_factory=list)
    num_iter: List[int] = field(default_factory=list)
    

    def add(self, num_iter, entropy, call_count, time, rk):
        self.num_iter.append(num_iter)
        self.entropy.append(entropy[0])
        self.call_count.append(call_count)
        self.time.append(time - self.start_time)
        self.rk.append(rk.ravel()[0])
        

    def get_data(self):
        logs = [self.opt_method, self.linesearch_method, self.tol, self.c1, self.c2, self.history_size, self.hf_criterion, self.entropy, self.num_iter, self.call_count, self.time, self.rk]
        names = ["OptMethod", "LineSearch", "tol", "c1", "c2", "history_size", "hf_criterion", "entropy", "num_iter", "oracle_calls", "time", "rk"]
        return {name: arr for name, arr in zip(names, logs)}

    def get_best(self):
        return [self.opt_method, self.linesearch_method, self.tol, self.c1, self.c2, self.history_size, self.hf_criterion, self.entropy[-1], self.num_iter[-1], self.call_count[-1], self.time[-1], self.rk[-1]]

    @property
    def best(self):
        return pd.DataFrame({
            "OptMethod": self.opt_method,
            "LineSearch": self.linesearch_method,
            "tol": self.tol,
            "c1": self.c1,
            "c2": self.c2,
            "history_size": self.history_size,
            "hf_criterion": self.hf_criterion,
            "entropy": self.entropy[-1], 
            "num_iter": self.num_iter[-1], 
            "oracle_calls": self.call_count[-1], 
            "time": self.time[-1],
            "rk": self.rk[-1]  
        }, index=[0])

    @property
    def dataframe(self):
        return pd.DataFrame.from_dict(self.get_data())

