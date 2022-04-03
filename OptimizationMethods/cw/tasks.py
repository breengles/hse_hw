import numpy as np
from sklearn import datasets


class Task:
    def __init__(self, n):
        self.n = n

    def f(self, x):
        raise NotImplementedError

    def df(self, x):
        raise NotImplementedError
    
    def ddf(self, x):
        raise NotImplementedError

    def numgra(self, x: np.ndarray, eps: float = 1e-8):
        grad = np.zeros(shape=(self.n, 1))
        t = np.zeros(shape=(self.n, 1))
        for i in range(self.n):
            t[i] = eps
            grad[i] = (self.f(x + t) - self.f(x - t)) / (2 * eps)
            t[i] = 0
        return grad

    def numhes(self, x: np.ndarray, eps: float = 1e-8):
        hessian = np.zeros(shape=(self.n, self.n))
        p = np.zeros(shape=(self.n, 1))
        for i in range(self.n):
            p[i] = eps
            hessian[i, :] = (self.df(x + p) - self.df(x - p)).ravel() / (2 * eps)
            p[i] = 0
        return hessian


class Task0(Task):
    def __init__(self, n):
        super().__init__(n)
        self.y = np.random.uniform(size=(self.n, 1))
        self.name = "<x, y>"

    def f(self, x):
        return (x.T @ self.y).ravel()[0]

    def df(self, x):
        return self.y

    def ddf(self, x):
        return np.zeros(shape=(self.n, self.n))


class Task41(Task):
    def __init__(self, n):
        super().__init__(n)
        self.A = datasets.make_spd_matrix(self.n)
        self.name = "4.1"

    def f(self, x):
        return 0.5 * np.linalg.norm(x @ x.T - self.A, ord="fro") ** 2

    def df(self, x):
        return 2 * (x @ x.T @ x - self.A @ x)

    def ddf(self, x):
        return 2 * ((x.T @ x).ravel()[0] * np.eye(self.n) + 2 * x @ x.T - self.A)


class Task42(Task):
    def __init__(self, n):
        super().__init__(n)
        self.A = datasets.make_spd_matrix(self.n)
        self.name = "4.2"

    def f(self, x):
        return ((self.A @ x).T @ x / (x.T @ x)).ravel()[0]

    def df(self, x):
        xx = (x.T @ x).ravel()[0]
        return 2 / xx * (self.A - x @ (self.A @ x).T / xx) @ x

    def ddf(self, x):
        xx = (x.T @ x).ravel()[0]
        return (
            2
            / xx ** 2
            * (
                xx * self.A
                - 2 * self.A @ x @ x.T
                - np.eye(self.n) * ((self.A @ x).T @ x).ravel()[0]
                - 2 * x @ (self.A @ x).T
                + 4 / xx * ((self.A @ x).T @ x).ravel()[0] * x @ x.T
            )
        )


class Task43(Task):
    def __init__(self, n):
        super().__init__(n)
        self.name = "4.3"

    def f(self, x):
        xx = (x.T @ x).ravel()[0]
        return np.power(xx, xx)

    def df(self, x):
        xx = (x.T @ x).ravel()[0]
        return 2 * np.power(xx, xx) * (np.log(xx) + 1) * x

    def ddf(self, x):
        xx = (x.T @ x).ravel()[0]
        return 2 * np.power(xx, xx) * (2 * ((np.log(xx) + 1)** 2 + 1 / xx) * x @ x.T + (np.log(xx) + 1) * np.eye(self.n))
        

class Task61(Task):
    def __init__(self):
        super().__init__(2)
        self.name = "6.1"

    def f(self, x):
        return 2 * x[0] ** 2 + x[1] ** 2 * (x[0] ** 2 - 2)

    def df(self, x):
        grad = np.zeros(shape=(2, 1))
        grad[0] = 4 * x[0] + 2 * x[1] ** 2 * x[0]
        grad[1] = 2 * x[0] ** 2 * x[1] - 4 * x[1]
        return grad

    def ddf(self, x):
        hessian = np.zeros(shape=(2, 2))
        hessian[0, 0] = 4 + 2 * x[1] ** 2
        hessian[0, 1] = hessian[1, 0] = 4 * x[0] * x[1]
        hessian[1, 1] = 2 * x[0] ** 2 - 4
        return hessian


class Task62(Task):
    def __init__(self, lam):
        super().__init__(2)
        self.lam = lam
        self.name = "6.2"

    def f(self, x):
        return (1 - x[0]) ** 2 + self.lam * (x[1] - x[0] ** 2) ** 2

    def df(self, x):
        grad = np.zeros(shape=(2, 1))
        grad[0] = 2 * (x[0] - 1 - 2 * self.lam * x[0] * x[1] + 2 * self.lam * x[0] ** 3)
        grad[1] = 2 * self.lam * (x[1] - x[0] ** 2)
        return grad

    def ddf(self, x):
        hessian = np.zeros(shape=(2, 2))
        hessian[0, 0] = 2 * (1 + 6 * self.lam * x[0] ** 2 - 2 * self.lam * x[1])
        hessian[0, 1] = hessian[1, 0] = -4 * self.lam * x[0]
        hessian[1, 1] = 2 * self.lam
        return hessian
