import numpy as np
from sklearn.metrics import mean_squared_error as mse
from logger import Logger


class MatrixFactorization:
    def __init__(self, factors, iterations=100, lr=1e-4, weight_decay=0, verbose=False, save_every=None) -> None:
        self.logger = Logger({"factors": factors, "iterations": iterations, "lr": lr, "weight_decay": weight_decay})

        self.factors = factors
        self.iterations = iterations
        self.lr = lr
        self.weight_decay = weight_decay

        self.verbose = verbose

        if save_every is None:
            self.save_every = iterations // 100
            if self.save_every == 0:
                self.save_every = 1
        else:
            self.save_every = save_every

        self.R = None
        self.U = None
        self.I = None

    def similar_items(self, item_id):
        idx = item_id - 1
        I_norms = np.linalg.norm(self.I, axis=1)
        cosine_sim = self.I @ self.I[idx].T / ((I_norms * I_norms[idx]))
        return np.argsort(cosine_sim)[-10:] + 1

    def recommend(self, user_id):
        pred = self.I @ self.U[user_id - 1].T
        return np.argsort(pred)[-10:] + 1

    def _init_matrices(self):
        self.U = np.random.uniform(0, 1 / np.sqrt(self.factors), size=(self.R.shape[0], self.factors))
        self.I = np.random.uniform(0, 1 / np.sqrt(self.factors), size=(self.R.shape[1], self.factors))

    def _calc_rmse(self):
        y_pred = (self.U @ self.I.T)[self.R.nonzero()]
        y_true = self.R.data
        return np.sqrt(mse(y_pred, y_true))

    def fit(self, R):
        raise NotImplementedError


class SVDS(MatrixFactorization):
    def fit(self, R):
        self.R = R
        self._init_matrices()

        rows, cols = R.nonzero()
        samples = [(i, j, v) for i, j, v in zip(rows, cols, R.data)]

        for it in range(self.iterations):
            for u, i, v in np.random.permutation(samples):
                loss = self.U[u] @ self.I[i] - v
                self.U[u] -= self.lr * loss * self.I[i]
                self.I[i] -= self.lr * loss * self.U[u]

            if (it + 1) % self.save_every == 0:
                rmse_ = self._calc_rmse()

                self.logger.log("iter", it + 1)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {it + 1: 4d} | RMSE: {rmse_:0.4f}")

        return self


class SVD(MatrixFactorization):
    def fit(self, R):
        self.R = R
        self._init_matrices()

        msk = (R > 0).astype(int).toarray()

        for i in range(self.iterations):
            loss = np.asarray((self.U @ self.I.T - R)) * msk

            self.U -= self.lr * (loss @ self.I + self.weight_decay * self.U)
            self.I -= self.lr * (loss.T @ self.U + self.weight_decay * self.I)

            if (i + 1) % self.save_every == 0:
                rmse_ = self._calc_rmse()
                loss_ = np.linalg.norm(loss)

                self.logger.log("iter", i + 1)
                self.logger.log("loss", loss_)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {i + 1: 4d} | RMSE: {rmse_:0.4f}")

        return self


class ALS(MatrixFactorization):
    def fit(self, R):
        self.R = R
        self._init_matrices()
