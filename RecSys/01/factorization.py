import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.special import expit
from logger import Logger
from utils import roc_auc
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity


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

        self.kdtree = None

    def similar_items(self, item_id, n=10):
        # _, nn = self.kdtree.query([self.I[item_id]], n)
        # return nn.flatten() + 1

        sims = cosine_similarity(self.I)[item_id]
        return np.argsort(sims)[-n:][::-1]

    def recommend(self, user_id, n=10):
        pred = self.I @ self.U[user_id].T
        return np.argsort(pred)[-n:]

    def _init_matrices(self, R):
        self.R = R
        self.U = np.random.uniform(0, 1 / np.sqrt(self.factors), size=(self.R.shape[0], self.factors))
        self.I = np.random.uniform(0, 1 / np.sqrt(self.factors), size=(self.R.shape[1], self.factors))

        self.U_bias = np.zeros(R.shape[0])
        self.I_bias = np.zeros(R.shape[1])

    def _calc_rmse(self):
        y_pred = (self.U @ self.I.T)[self.R.nonzero()]
        y_true = self.R.data
        return np.sqrt(mse(y_pred, y_true))

    def fit(self, R):
        raise NotImplementedError


class SVDS(MatrixFactorization):
    def fit(self, R):
        self._init_matrices(R)

        data_bias = np.mean(R)

        rows, cols = R.nonzero()
        samples = [(i, j, v) for i, j, v in zip(rows, cols, R.data)]

        for it in range(self.iterations):
            for u, i, v in np.random.permutation(samples):
                loss = self.U[u] @ self.I[i] - v + self.U_bias[u] + self.I_bias[i] + data_bias

                self.U[u] -= self.lr * (loss * self.I[i] + self.weight_decay * self.U[u])
                self.I[i] -= self.lr * (loss * self.U[u] + self.weight_decay * self.I[i])

                self.U_bias[u] -= self.lr * (loss + self.weight_decay * self.U_bias[u])
                self.I_bias[i] -= self.lr * (loss + self.weight_decay * self.I_bias[i])
                # data_bias -= self.lr * loss

            if (it + 1) % self.save_every == 0:
                rmse_ = self._calc_rmse()

                self.logger.log("iter", it + 1)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {it + 1: 4d} | RMSE: {rmse_:0.5E}")

        self.kdtree = KDTree(self.I)

        return self


class SVD(MatrixFactorization):
    def fit(self, R):
        self._init_matrices(R)

        msk = (R > 0).astype(int).toarray()

        for i in range(self.iterations):
            loss = np.asarray(self.U @ self.I.T - R) * msk

            self.U -= self.lr * (loss @ self.I + self.weight_decay * self.U)
            self.I -= self.lr * (loss.T @ self.U + self.weight_decay * self.I)

            if (i + 1) % self.save_every == 0:
                rmse_ = self._calc_rmse()
                loss_ = np.linalg.norm(loss)

                self.logger.log("iter", i + 1)
                self.logger.log("loss", loss_)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {i + 1: 4d} | RMSE: {rmse_:0.5E}")

        self.kdtree = KDTree(self.I)
        return self


class ALS(MatrixFactorization):
    def __update_user(self):
        for u in range(self.R.shape[0]):
            r = self.R[u].toarray().flatten()

            msk = r != 0

            if msk.sum() == 0:
                # otherwise it will choose array with size of 0 in 1st dim
                continue

            Is = self.I[msk]

            self.U[u] = np.linalg.inv(
                np.einsum("ij,ik->jk", Is, Is) + self.weight_decay * np.eye(self.factors)
            ) @ np.sum(r[msk].reshape(-1, 1) * Is, axis=0)

    def __update_item(self):
        for i in range(self.R.shape[1]):
            r = self.R[:, i].toarray().flatten()
            msk = r != 0

            if msk.sum() == 0:
                continue

            Us = self.U[msk]

            self.I[i] = np.linalg.inv(
                np.einsum("ij,ik->jk", Us, Us) + self.weight_decay * np.eye(self.factors)
            ) @ np.sum(r[msk].reshape(-1, 1) * Us, axis=0)

    def fit(self, R):
        self._init_matrices(R)

        for it in range(self.iterations):
            self.__update_user()
            self.__update_item()

            # for idx, u in enumerate(self.U):
            #     u = (
            #         np.linalg.inv(self.I.T @ self.I + self.weight_decay * np.eye(self.factors))
            #         @ self.I.T
            #         @ R[idx].toarray().T
            #     )

            # for idx, i in enumerate(self.I):
            #     i = (
            #         np.linalg.inv(self.U.T @ self.U + self.weight_decay * np.eye(self.factors))
            #         @ self.U.T
            #         @ R[:, idx].toarray()
            #     )

            if (it + 1) % self.save_every == 0:
                rmse_ = self._calc_rmse()

                self.logger.log("iter", it + 1)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {it + 1: 4d} | RMSE: {rmse_:0.5E}")

        self.kdtree = KDTree(self.I)


class BPR(MatrixFactorization):
    def __step(self, u, i, j):
        user_u = self.U[u]
        item_i = self.I[i]
        item_j = self.I[j]

        r_uij = np.sum(user_u * (item_i - item_j), axis=1)

        sigmoid = np.tile(expit(r_uij), (self.factors, 1)).T

        grad_u = sigmoid * (item_j - item_i) + self.weight_decay * user_u
        grad_i = sigmoid * -user_u + self.weight_decay * item_i
        grad_j = sigmoid * user_u + self.weight_decay * item_j
        self.U[u] -= self.lr * grad_u
        self.I[i] -= self.lr * grad_i
        self.I[j] -= self.lr * grad_j

    def __sample(self, n_users, n_items, batch_size, indices, indptr):
        sampled_pos_items = np.zeros(batch_size, dtype=int)
        sampled_neg_items = np.zeros(batch_size, dtype=int)
        sampled_users = np.random.choice(n_users, size=batch_size, replace=False)

        for idx, user in enumerate(sampled_users):
            if indptr[user] == indptr[user + 1]:
                continue

            pos_items = indices[indptr[user] : indptr[user + 1]]

            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items

    def fit(self, R, batch_size=32):
        self._init_matrices(R)

        indptr = R.indptr
        indices = R.indices

        for it in range(self.iterations):
            users, items_pos, items_neg = self.__sample(R.shape[0], R.shape[1], batch_size, indices, indptr)
            self.__step(users, items_pos, items_neg)

            if (it + 1) % self.save_every == 0:
                auc = roc_auc(R, self)

                self.logger.log("iter", it + 1)
                self.logger.log("AUC", auc)

                if self.verbose:
                    print(f"Iter: {it + 1: 6d} | AUC: {auc:0.5f}")

        return self

    def _predict_user(self, user):
        return self.U[user] @ self.I.T
