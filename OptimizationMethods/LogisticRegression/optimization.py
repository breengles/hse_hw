from oracle import Oracle, generate_dataset, make_oracle
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import linesearch
from time import time
from typing import Tuple
from logger import Logger
import pandas as pd
from tabulate import tabulate
from collections import deque


def get_line_search(line_search_method: str) -> linesearch.LineSearch:
    method = line_search_method.strip()
    if method == "golden_section":
        return linesearch.GoldenLineSearch()
    elif method == "brent":
        return linesearch.BrentLineSearch()
    elif method == "dbrent":
        return linesearch.DBrentLineSearch()
    elif method == "armijo":
        return linesearch.Armijo()
    elif method == "wolfe":
        return linesearch.Wolfe()
    elif method == "nesterov":
        return linesearch.Nesterov()
    else:
        raise NotImplementedError


def check_armijo(w, d, oracle, alpha=1, c1=0.25):
    fw, dfw = oracle.fuse_value_grad(w)
    dfwd = dfw.T @ d
    return oracle.value(w + alpha * d) <= fw + c1 * alpha * dfwd


def correcting_hessian(H, eps):
    return H + eps * np.identity(H.shape[0]), eps * 2


def direction_and_grad(w: np.ndarray, oracle: Oracle, optimization_method, buffer: deque = None, gamma0: float = 1,
                       max_iter: int = 10000, tol: float = 1e-8, hf_criterion: str = "num2006"):
    method = optimization_method.strip()
    if method in ("gd", "gradient_descent"):
        g = oracle.grad(w)
        return -g, g
    
    elif method in ("n", "newton"):
        eps = 1e-8  # ? tune
        g, H = oracle.fuse_grad_hessian(w)
        while True:
            try:
                return scipy.linalg.cho_solve(scipy.linalg.cho_factor(H), -g), g
            except np.linalg.LinAlgError:
                H, eps = correcting_hessian(H, eps)
                
    elif method in ("cg", "conjugate_gradient"):
        g = oracle.grad(w)
        z = np.zeros_like(g)
        r, d = g, -g

        g_norm = np.linalg.norm(g)

        if hf_criterion == "num2006":
            eps = min(0.5, np.sqrt(g_norm)) * g_norm
        elif hf_criterion == "tol":
            eps = tol
        elif hf_criterion == "norm":
            eps = g_norm
        else:
            raise NotImplementedError

        for i in range(max_iter):
            Hd = oracle.hessian_vec_product(w, d)
            dHd = d.T @ Hd
            rr = r.T @ r
            if dHd <= 0:
                if i == 0:
                    z = -g
                break

            alpha = rr / dHd
            z += alpha * d
            r += alpha * Hd
            if np.linalg.norm(r) < eps:
                break

            beta = r.T @ r / rr
            d = beta * d - r
        return z, g
    
    elif method in ("lbfgs", "l-bfgs"):
        q0 = oracle.grad(w)
        q = q0.copy()
        alphas = []
        for rho, s, y in reversed(buffer):
            alpha = rho * s.T @ q
            alphas.append(alpha)
            q -= alpha * y
        alphas.reverse()
        
        if buffer:
            _, s, y = buffer[-1]
            gamma = s.T @ y / (y.T @ y)
        else:
            gamma = gamma0
            
        q *= gamma
        for i in range(len(buffer)):
            rho, s, y = buffer[i]
            alpha = alphas[i]
            beta = rho * y.T @ q
            q += s * (alpha - beta)
        return -q, q0
            
    else:
        raise NotImplementedError


def _gradient_descent(w0: np.ndarray, d0: np.ndarray, oracle: Oracle, line_search: callable, max_iter_alpha=10000,
                      c1=0.25, c2=0.9):
    alpha, _, _ = line_search(w0=w0, d=d0, oracle=oracle, max_iter=max_iter_alpha, c1=c1, c2=c2)
    if alpha is not None:
        w = w0 + d0 * alpha
        d, g = direction_and_grad(w, oracle, "gradient_descent")
    return w, d, g


def _newton(w0: np.ndarray, d0: np.ndarray, g0: np.ndarray, oracle: Oracle, line_search: callable, tol1: float = 1e-8,
            max_iter_alpha: int = 10000, c1: float = 0.25, c2: float = 0.9):
    cnst = 1e3
    if np.linalg.norm(d0) >= cnst:
        d0 = d0 / np.linalg.norm(d0)
        
    if check_armijo(w0, d0, oracle, c1=c1):
        alpha = 1
    else:
        alpha, _, _ = line_search(w0=w0, d=d0, oracle=oracle, max_iter=max_iter_alpha, c1=c1, c2=c2)
    
    w = w0 + d0 * alpha
    d, g = direction_and_grad(w, oracle, "newton")
    return w, d, g


def _conjugate_gradient(w0: np.ndarray, d0: np.ndarray, g0: np.ndarray, oracle: Oracle, line_search: callable,
                        max_iter_gc: int = 10000, max_iter_alpha: int = 10000, c1: float = 0.25, c2: float = 0.9,
                        hf_criterion: str = "num2006"):
    cnst = 1e3
    if np.linalg.norm(d0) >= cnst:
        d0 = d0 / np.linalg.norm(d0)
        
    if check_armijo(w0, d0, oracle, c1=c1):
        alpha = 1
    else:
        alpha, _, _ = line_search(w0=w0, d=d0, oracle=oracle, max_iter=max_iter_alpha, c1=c1, c2=c2)
        
    w = w0 + alpha * d0
    d, g = direction_and_grad(w, oracle, "conjugate_gradient", max_iter=max_iter_gc, hf_criterion=hf_criterion)
    return w, d, g


def _lbfgs(w0: np.ndarray, d0: np.ndarray, g0: np.ndarray, oracle: Oracle, buffer: deque, line_search: callable, max_iter_alpha: int = 10000, c1: float = 1e-4, c2: float = 0.9, gamma0: float = 1):
    alpha, _, _ = line_search(w0, d0, g0, oracle, max_iter=max_iter_alpha, c1=c1, c2=c2)    
    w = w0 + alpha * d0
    d, g = direction_and_grad(w, oracle, "lbfgs", buffer, gamma0=gamma0)
    
    s = w - w0
    y = g - g0
    rho = 1 / (y.T @ s)
    buffer.append((rho, s, y))
    
    return w, d, g


def optimize(start_point: np.ndarray, oracle: Oracle, optimization_method: str, line_search_method: str,
             history_size: int = 100, tol=1e-8, max_iter=10000, max_iter_alpha=10000, gamma0=1,
             hf_criterion: str = "num2006", c1=0.25, c2=0.9, output_log=False):
    log = None
    g0g0 = None
    buffer = None
    
    t_start = time()
    opt_method = optimization_method.strip()
    oracle.reset_calls()

    if opt_method in ("lbfgs", "l-bfgs"):
        line_search_method = "wolfe"
        buffer = deque(maxlen=history_size)
    if line_search_method:
        line_search = get_line_search(line_search_method)
        
    w = np.copy(start_point)

    d, g = direction_and_grad(w, oracle, optimization_method, hf_criterion=hf_criterion, buffer=buffer, gamma0=gamma0)
    gg = g.T @ g
    tol1 = tol * gg

    if output_log:
        if optimization_method == "conjugate_gradient":
            log = Logger(opt_method=optimization_method, linesearch_method=line_search_method, start_time=t_start, tol=tol, c1=c1, c2=c2, hf_criterion=hf_criterion)
        elif optimization_method == "lbfgs":
            log = Logger(opt_method=optimization_method, linesearch_method=line_search_method, start_time=t_start, tol=tol, c1=c1, c2=c2, history_size=history_size)
        else:
            log = Logger(opt_method=optimization_method, linesearch_method=line_search_method, start_time=t_start, tol=tol, c1=c1, c2=c2, hf_criterion=None, history_size=None)
        g0g0 = gg

    num_iter = 0
    for _ in range(max_iter):
        num_iter += 1

        if output_log:
            fw = oracle.value(w)
            oracle.calls -= 1
            log.add(num_iter, fw, oracle.calls, time(), gg / g0g0)

        if gg <= tol1:
            break

        if opt_method in ("gd", "gradient_descent"):
            w, d, g = _gradient_descent(w, d, oracle, line_search, max_iter_alpha=max_iter_alpha, c1=c1, c2=c2)
        elif opt_method in ("n", "newton"):
            w, d, g = _newton(w, d, g, oracle, line_search, tol1=tol1, max_iter_alpha=max_iter_alpha, c1=c1, c2=c2)
        elif opt_method in ("cg", "conjugate_gradient"):
            w, d, g = _conjugate_gradient(w, d, g, oracle, line_search)
        elif opt_method in ("lbfgs", "l-bfgs"):
            w, d, g = _lbfgs(w, d, g, oracle, buffer, line_search, c1=c1, c2=c2, gamma0=gamma0)
        gg = g.T @ g

    return w, oracle.value(w), log


if __name__ == "__main__":
    # max_iter = 10000
    # oracle = make_oracle("a1a.libsvm")
    # w0 = np.ones((oracle.m, 1)) * 0

    # _, _, log = optimize(w0, oracle, "lbfgs", None, output_log=True, history_size=50, c1=1e-4, gamma0=1)
    # print(log.best)
    
    # optimize_lbfgs(oracle, w0, buffer_size=50, verbose=True)
    
    # to_df = []
    # for hsize in range(10, 101, 10):
    #     _, _, log = optimize(w0, oracle, "lbfgs", None, output_log=True, history_size=hsize, c1=1e-4, gamma0=1)
    #     to_df.append(log.dataframe)
        
    max_iter = 10000
    oracle = make_oracle("a1a.libsvm")
    w0 = np.zeros((oracle.m, 1))

    to_df = []
    linesearch_methods = ["golden_section", "brent", "dbrent", "armijo", "wolfe"]
    for ls in linesearch_methods:
        if ls == "armijo":
            c1 = 0.25
            c2 = None
        elif ls == "wolfe":
            c1 = 1e-4
            c2 = 0.9
        elif ls == "nesterov":
            c1 = c2 = 2
        else:
            c1 = 0.25
            c2 = None
        _, _, log = optimize(w0, oracle, "conjugate_gradient", ls, max_iter=max_iter, output_log=True, c1=c1, c2=c2)
        to_df.append(log.dataframe)

        df = pd.concat(to_df, ignore_index=True)
        # df.to_csv("data/hf_exp.csv")
    
    table = [df.columns.values.tolist()] + df.values.tolist()
    print(tabulate(table, headers="firstrow", tablefmt="github", floatfmt=["", "", ".0e", ".2e", ".2e", ".4f", "", "", ".4f", ".1e"]))