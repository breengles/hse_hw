from oracle import Oracle, generate_dataset, make_oracle
import numpy as np
import scipy, scipy.linalg, scipy.sparse, scipy.sparse.linalg
import linesearch
from time import time
from typing import Tuple
from logger import Logger
import pandas as pd
from tabulate import tabulate


def prox(x, c):
    return np.sign(x) * np.maximum(np.abs(x) - c, 0)


def optimize(w0: np.ndarray, oracle: Oracle, lambda_: float = 0.1, L0: float = 1.0,
             max_iter: int = 10000, tol: float = 1e-8, max_iter_nesterov: int = 10000, log=True):
    w = w0.copy()
    L = L0
    oracle.reset_calls()
    fk, g = oracle.fuse_value_grad(w)

    logger = None
    if log:
        logger = Logger(time(), tol, lambda_)
    
    for i in range(max_iter):
        for _ in range(max_iter_nesterov):
            alpha = 1 / L
            y = prox(w - alpha * g, lambda_ * alpha)
            fkk = oracle.value(y)
            yw = y - w
            ywyw = yw.T @ yw
            if fkk <= fk + g.T @ yw + (L / 2) * ywyw:
                break
            L *= 2
        w = y
        crit = ywyw * L ** 2
        if log:
            logger.add(i + 1, oracle.calls, fkk + lambda_ * np.linalg.norm(w, ord=1), crit, w)
        if crit <= tol:
            break
        L /= 2
        fk, g = oracle.fuse_value_grad(w)
    return w, logger if log else w


if __name__ == "__main__":
    max_iter = 10000
    oracle = make_oracle("a1a.libsvm")
    w0 = np.ones((oracle.m, 1))
    
    w, log = optimize(w0, oracle, max_iter=max_iter,lambda_=0)
    print(log.best)
