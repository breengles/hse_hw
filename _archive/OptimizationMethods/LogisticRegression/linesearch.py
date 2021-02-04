import numpy as np
from scipy.optimize import bracket, line_search


class LineSearch:
    def __init__(self, init_a: float = 0.0, init_b: float = 1.0):
        self.oracle_calls = 0
        self.init_a = init_a
        self.init_b = init_b

    def __call__(self, w0: np.ndarray, d: np.ndarray, oracle, *args, tol: float = 1e-8, max_iter: int = 10000,
                 c1: float = 0.25, c2: float = 0.9, **kwargs):
        a, b, c = self._bracket(lambda x: oracle.value(w0 + x * d), self.init_a, self.init_b)
        return self._estimate_step(w0, d, oracle, *args, brack=(a, b, c), tol=tol, max_iter=max_iter, **kwargs)

    def _bracket(self, f: callable, xa: float = 0, xb: float = 1):
        a, b, c, _, _, _, funcalls = bracket(f, xa, xb)
        self.oracle_calls += funcalls
        return a, b, c

    def _estimate_step(self, *args, **kwargs) -> tuple:
        raise NotImplementedError


class GoldenLineSearch(LineSearch):
    def __init__(self, *args, **kwargs):
        super(GoldenLineSearch, self).__init__(*args, **kwargs)

    def _estimate_step(self, w0, d, oracle, brack: tuple = None, tol: float = 1e-8, max_iter: int = 10000) -> tuple:
        ax, bx, cx = brack
        phi = (np.sqrt(5) - 1) / 2
        Phi = 1 - phi
        x0 = ax
        x3 = cx
        if abs(cx - bx) > abs(bx - ax):
            x1 = bx
            x2 = bx + Phi * (cx - bx)
        else:
            x2 = bx
            x1 = bx - Phi * (bx - ax)
        f1 = oracle.value(w0 + x1 * d)
        f2 = oracle.value(w0 + x2 * d)
        num_f_calls = 2
        num_iter = 0
        for _ in range(max_iter):
            num_iter += 1
            if abs(x3 - x0) < tol * (abs(x1) + abs(x2)):
                break
            if f2 < f1:
                x0, x1, x2 = x1, x2, phi * x2 + Phi * x3
                f1, f2 = f2, oracle.value(w0 + x2 * d)
            else:
                x3, x2, x1 = x2, x1, phi * x1 + Phi * x0
                f2, f1 = f1, oracle.value(w0 + x1 * d)
            num_f_calls += 1
        if f1 < f2:
            x_min = x1
            f_min = f1
        else:
            x_min = x2
            f_min = f2
        return x_min, f_min, num_iter


class BrentLineSearch(LineSearch):
    def __init__(self, *args, **kwargs):
        super(BrentLineSearch, self).__init__(*args, **kwargs)

    def __call__(self, w0: np.ndarray, d: np.ndarray, oracle, *args, tol: float = 1e-8, max_iter: int = 10000,
                 **kwargs):
        return self._estimate_step(w0, d, oracle, *args, tol=tol, max_iter=max_iter, **kwargs)

    def _estimate_step(self, w0, d, oracle, *args, tol: float = 1e-8, max_iter: int = 10000, **kwargs) -> tuple:
        from scipy.optimize import brent
        x_min, f_min, num_iter, _ = brent(lambda x: oracle.value(w0 + x * d),
                                            brack=(self.init_a, self.init_b), tol=tol, maxiter=max_iter,
                                            full_output=True)
        return x_min, f_min, num_iter


class DBrentLineSearch(LineSearch):
    def __init__(self, *args, **kwargs):
        super(DBrentLineSearch, self).__init__(*args, **kwargs)

    def _estimate_step(self, w0: np.ndarray, dir: np.ndarray, oracle, brack: tuple = None, tol: float = 1e-8,
                       max_iter: int = 10000) -> tuple:
        if not brack:
            raise ValueError("Empty bracket")

        ax, _, cx = brack

        a = ax if ax < cx else cx
        b = ax if ax > cx else cx

        if a > b:
            a, b = b, a

        # len of current previous and current intervals
        current_d = d = b - a

        # x -- best, w -- 2nd best, v -- 3rd best
        x = w = v = 0.5 * (a + b)  # init values
        val = oracle.fuse_value_grad(w0 + x * dir)  # init oracle call
        f_x = f_w = f_v = val[0]
        df_x = df_w = df_v = val[1].T @ dir
        num_iter = 0
        for _ in range(max_iter):
            num_iter += 1
            xm = 0.5 * (a + b)
            tol1 = tol * (abs(x) + 1e-1)
            tol2 = 2 * tol1
            # some tricky condition (see e.g. OptMeth ML lecture 2016 MSU (quasi-empirical)
            if abs(x - xm) + 0.5 * (b - a) <= tol2:
                break

            if abs(current_d) > tol1:
                # init out-of-bound d1 and d2
                d1 = d2 = 2 * (b - a)
                # try to fit parabola via 1st-order derivative
                if not np.isclose(x, w) and not np.isclose(df_w, df_x):
                    d1 = (w - x) * df_x / (df_x - df_w)
                if not np.isclose(x, v) and not np.isclose(df_v, df_x):
                    d2 = (v - x) * df_x / (df_x - df_v)

                u1 = x + d1
                u2 = x + d2

                # check if obtained points with parabola fitting are actually good enough
                is_u1_ok = (a - u1) * (u1 - b) > 0 >= df_x * d1
                is_u2_ok = (a - u2) * (u2 - b) > 0 >= df_x * d2

                prev_d, current_d = current_d, d

                if is_u1_ok or is_u2_ok:
                    if is_u1_ok and is_u2_ok:
                        # take the smallest interval
                        d = d1 if abs(d1) < abs(d2) else d2
                    elif is_u1_ok:
                        d = d1
                    else:
                        d = d2
                    if abs(d) <= abs(0.5 * prev_d):
                        u = x + d
                        if u - a < tol2 or b - u < tol2:
                            d = np.copysign(tol1, xm - x)
                    else:
                        current_d = a - x if df_x >= 0.0 else b - x
                        d = 0.5 * current_d
                else:
                    current_d = a - x if df_x >= 0.0 else b - x
                    d = 0.5 * current_d
            else:
                current_d = a - x if df_x >= 0 else b - x
                d = 0.5 * current_d

            if abs(d) >= tol1:
                u = x + d
                f_u, df_u = oracle.fuse_value_grad(w0 + u * dir)
                df_u = df_u.T @ dir
            else:
                u = x + np.copysign(tol1, d)
                f_u, df_u = oracle.fuse_value_grad(w0 + u * dir)
                df_u = df_u.T @ dir
                if f_u > f_x:
                    break

            if f_u <= f_x:
                if u >= x:
                    a = x
                else:
                    b = x
                v = w
                f_v = f_w
                df_v = df_w
                w = x
                f_w = f_x
                df_w = df_x
                x = u
                f_x = f_u
                df_x = df_u
            else:
                if u < x:
                    a = u
                else:
                    b = u
                if f_u <= f_w or np.isclose(w, x):
                    v = w
                    f_v = f_w
                    df_v = df_w
                    w = u
                    f_w = f_u
                    df_w = df_u
                elif f_u < f_v or np.isclose(v, x) or np.isclose(v, w):
                    v = u
                    f_v = f_u
                    df_v = df_u

        return x, f_x, num_iter


class Armijo(LineSearch):
    def __init__(self, *args, **kwargs):
        super(Armijo, self).__init__(*args, **kwargs)

    def __call__(self, w0, d, oracle, max_iter: int = 10000, c1: float = 0.25, alpha: float = None, **kwargs):
        if alpha:
            a = b = c = alpha
        else:
            a, b, c = self._bracket(lambda x: oracle.value(w0 + x * d), self.init_a, self.init_b)
            
        return self._estimate_step(w0, d, oracle, brack=(a, b, c), max_iter=max_iter, c1=c1)

    def _estimate_step(self, w0, d, oracle, brack: tuple = None, max_iter: int = 10000, c1: float = 0.25) -> tuple:
        ax, bx, cx = brack
        x = max(ax, bx, cx)
        f0, df0 = oracle.fuse_value_grad(w0)
        df0 = df0.T @ d
        fx = f0
        
        num_iter = 0
        for i in range(max_iter):
            num_iter = i + 1
        
            if fx <= f0 + c1 * x * df0:
                break
        
            x /= 2
            fx = oracle.value(w0 + x * d)
        return x, fx, num_iter


class Wolfe(LineSearch):
    def __init__(self, *args, **kwargs):
        super(Wolfe, self).__init__(*args, **kwargs)

    def __call__(self, w0: np.ndarray, d: np.ndarray, g: np.ndarray, oracle, max_iter: int = 10000, c1: float = 1e-4, c2: float = 0.9,
                 **kwargs):
        num_iter = None
        alpha, _, _, fx, *_ = line_search(oracle.value, lambda x: oracle.grad(x).reshape(1, -1), w0, d, g.T,
                                          maxiter=max_iter, c1=c1, c2=c2)
        if not alpha:  # if Wolfe did not converge
            alpha, fx, _ = Armijo()(w0, d, oracle, max_iter, c1=c1)  # ? adjust c1

        return alpha, fx, num_iter


class Nesterov(LineSearch):
    def __init__(self, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.alpha = 1

    def __call__(self, w0: np.ndarray, d: np.ndarray, oracle, tol: float = 1e-8, max_iter: int = 10000, c1: float = 2.,
                 c2: float = 2., **kwargs):
        fk, fkk = oracle.value(w0), oracle.value(w0 + self.alpha * d)

        num_iter = 0
        for i in range(max_iter):
            num_iter = i + 1

            if fkk <= fk - 0.5 * self.alpha * d.T @ d:
                break

            self.alpha /= c1
            fkk = oracle.value(w0 + self.alpha * d)

        self.alpha *= c2
        return self.alpha / c2, fkk, num_iter
