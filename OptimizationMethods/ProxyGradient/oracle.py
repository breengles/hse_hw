import numpy as np
import scipy
import scipy.sparse
import scipy.special
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer


class Oracle:
    def __init__(self, X, Y, eps=1e-12):
        self.X = X
        self.Y = Y
        self.n = self.X.shape[0]
        self.m = self.X.shape[1]
        self.eps = eps
        self.calls = 0

    def __sigmoid(self, x):
        #pylint: disable=no-member
        return scipy.special.expit(x).reshape(-1, 1)

    def __log(self, x):
        return np.log(x + self.eps)

    def reset_calls(self):
        self.calls = 0

    def add_calls(self, num_calls):
        self.calls += num_calls

    def value(self, w):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        return -1 / self.n * (self.Y.T @ self.__log(sigma) + (1 - self.Y).T @ self.__log(1 - sigma))[0]

    def grad(self, w):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        return 1 / self.n * self.X.T @ (sigma - self.Y)

    def hessian(self, w):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        sigma = (sigma * (1 - sigma)).T
        if scipy.sparse.issparse(self.X):
            return (1 / self.n * self.X.T.multiply(sigma) @ self.X).A
        else:
            return (1 / self.n * self.X.T * sigma @ self.X)

    def hessian_vec_product(self, w, d, eps: float = 1e-8):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        sigma = (sigma * (1 - sigma)).reshape(-1, 1)
        step1 = self.X @ d
        step2 = sigma * step1
        step3 = self.X.T @ step2
        return step3 / self.n

    def fuse_value_grad(self, w):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        value = -1 / self.n * (self.Y.T @ self.__log(sigma) + (1 - self.Y).T @ self.__log(1 - sigma))
        grad = 1 / self.n * self.X.T @ (sigma - self.Y)
        return value, grad

    def fuse_value_grad_hessian(self, w):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        value = -1 / self.n * (self.Y.T @ self.__log(sigma) + (1 - self.Y).T @ self.__log(1 - sigma))
        grad = 1 / self.n * self.X.T @ (sigma - self.Y)
        sigma = (sigma * (1 - sigma)).T
        if scipy.sparse.issparse(self.X):
            hessian = self.X.T.multiply(sigma)
            hessian = hessian.A
        else:
            hessian = self.X.T * sigma
        hessian = 1 / self.n * hessian @ self.X
        return value, grad, hessian

    def fuse_grad_hessian(self, w):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)
        grad = 1 / self.n * self.X.T @ (sigma - self.Y)
        sigma = (sigma * (1 - sigma)).T
        if scipy.sparse.issparse(self.X):
            hessian = self.X.T.multiply(sigma)
            hessian = hessian.A
        else:
            hessian = self.X.T * sigma
        hessian = 1 / self.n * hessian @ self.X
        return grad, hessian

    def fuse_value_grad_hessian_vec_product(self, w, d):
        self.calls += 1
        sigma = self.__sigmoid(self.X @ w)

        value = -1 / self.n * (self.Y.T @ self.__log(sigma) + (1 - self.Y).T @ self.__log(1 - sigma))

        grad = 1 / self.n * self.X.T @ (sigma - self.Y)

        sigma = (sigma * (1 - sigma)).reshape(-1, 1)
        step1 = self.X @ d
        step2 = sigma * step1
        hessian_vec_product = self.X.T @ step2 / self.n
        
        return value, grad, hessian_vec_product
    

def generate_dataset(w: np.ndarray, size: int = 1000):
    w_size = w.shape[0] - 1
    X = np.random.standard_normal(size=(size, w_size))
    X = np.hstack((X, np.ones(size).reshape(-1, 1)))
    Y = (X @ w >= 0).astype(int).reshape(-1, 1)
    return X, Y


def make_oracle(data_path: str = None, size: int = 1000, w: np.ndarray = None, eps: float = 1e-12):
    if data_path:
        X, Y = load_svmlight_file(data_path)
        X = scipy.sparse.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
        Y = LabelBinarizer().fit_transform(Y)
    else:
        X, Y = generate_dataset(w, size)

    return Oracle(X, Y, eps)
