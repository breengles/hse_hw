import random
from itertools import islice

import numpy as np
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree

from logger import Logger
from utils import rmse, roc_auc


class MatrixFactorization:
    def __init__(
        self, factors, iterations=100, lr=1e-4, weight_decay=0, verbose=False, save_every=None, seed=42
    ) -> None:
        self.logger = Logger(
            {"factors": factors, "iterations": iterations, "lr": lr, "weight_decay": weight_decay, "seed": seed}
        )

        np.random.seed(seed)
        random.seed(seed)

        self.factors = factors
        self.iterations = iterations
        self.lr = lr
        self.weight_decay = weight_decay

        self.verbose = verbose

        if save_every is None:
            self.save_every = iterations // 10
            if self.save_every == 0:
                self.save_every = 1
        else:
            self.save_every = save_every

        self.R = None
        self.U = None
        self.I = None

    def predict_user(self, user):
        return self.I @ self.U[user].T

    def similar_items(self, item_id, n=10):
        sims = cosine_similarity(self.I)[item_id]
        return np.argsort(sims)[-(n + 1) : -1][::-1]

    def recommend(self, user, n=10):
        already_liked = set(self.R.getrow(user).indices)
        rec = np.argsort(self.predict_user(user))[::-1]
        return list(islice((r for r in rec if r not in already_liked), n))

    def _init_matrices(self, R):
        self.R = R
        self.U = np.random.normal(scale=1 / np.sqrt(self.factors), size=(self.R.shape[0], self.factors))
        self.I = np.random.normal(scale=1 / np.sqrt(self.factors), size=(self.R.shape[1], self.factors))

    def fit(self, R):
        raise NotImplementedError


class SVDS(MatrixFactorization):
    def fit(self, R):
        self._init_matrices(R)

        rows, cols = R.nonzero()
        samples = [(i, j, v) for i, j, v in zip(rows, cols, R.data)]

        for it in range(self.iterations):
            for u, i, v in np.random.permutation(samples):  # very stochastic xD
                loss = self.U[u] @ self.I[i] - v

                self.U[u] -= self.lr * (loss * self.I[i] + self.weight_decay * self.U[u])
                self.I[i] -= self.lr * (loss * self.U[u] + self.weight_decay * self.I[i])

            if (it + 1) % self.save_every == 0 and self.verbose:
                rmse_ = rmse(self)

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

            if (i + 1) % self.save_every == 0 and self.verbose:
                rmse_ = rmse(self)
                loss_ = np.linalg.norm(loss)

                self.logger.log("iter", i + 1)
                self.logger.log("loss", loss_)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {i + 1: 4d} | RMSE: {rmse_:0.5E}")

        return self


class ALS(MatrixFactorization):
    def __update_user(self):
        II = self.I.T @ self.I
        for i in range(self.U.shape[0]):
            self.U[i] = (np.linalg.inv(II + self.E) @ self.I.T @ self.R[i].T).flatten()

    def __update_item(self):
        UU = self.U.T @ self.U
        for j in range(self.I.shape[0]):
            self.I[j] = (np.linalg.inv(UU + self.E) @ self.U.T @ self.R[:, j]).flatten()

    def fit(self, R):
        self._init_matrices(R)

        self.E = self.lr * np.eye(self.factors)

        for it in range(self.iterations):
            self.__update_user()
            self.__update_item()

            if (it + 1) % self.save_every == 0 and self.verbose:
                rmse_ = rmse(self)

                self.logger.log("iter", it + 1)
                self.logger.log("rmse", rmse_)

                if self.verbose:
                    print(f"Iter: {it + 1: 4d} | RMSE: {rmse_:0.5E}")

        return self


class BPR(MatrixFactorization):
    def __step(self, u, i, j):
        user_u = self.U[u]
        item_i = self.I[i]
        item_j = self.I[j]

        r_uij = np.sum(user_u * (item_i - item_j), axis=1)
        sigmoid = expit(-r_uij, keepdims=True)

        self.U[u] -= self.lr * (sigmoid * (item_j - item_i) + self.weight_decay * user_u)
        self.I[i] -= self.lr * (-sigmoid * user_u + self.weight_decay * item_i)
        self.I[j] -= self.lr * (sigmoid * user_u + self.weight_decay * item_j)

    def __sample(self, batch_size):
        n_users, n_items = self.R.shape

        indptr = self.R.indptr
        indices = self.R.indices

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

        for it in range(self.iterations):
            users, items_pos, items_neg = self.__sample(batch_size)
            self.__step(users, items_pos, items_neg)

            if (it + 1) % self.save_every == 0 and self.verbose:
                auc, auc2 = roc_auc(self)

                self.logger.log("iter", it + 1)
                self.logger.log("AUC", auc)
                self.logger.log("AUC2", auc2)

                if self.verbose:
                    print(f"Iter: {it + 1: 6d} | AUC: {auc:0.5f} | AUC2: {auc2:0.5f}")

        return self


class WARP(MatrixFactorization):
    def __L(self, k):
        return np.sum(1 / np.arange(1, k + 1))

    def __sample(self):
        n_users, n_items = self.R.shape

        user = np.random.choice(n_users)

        indptr = self.R.indptr
        indices = self.R.indices

        if indptr[user] == indptr[user + 1]:
            return

        items_pos = indices[indptr[user] : indptr[user + 1]]
        items_neg = np.setdiff1d(np.arange(n_items), items_pos)

        item_pos = np.random.choice(items_pos)
        score_pos = self.U[user] @ self.I[item_pos]

        np.random.shuffle(items_neg)
        for trial, item_neg in enumerate(items_neg, start=1):
            score_neg = self.U[user] @ self.I[item_neg]

            if score_neg >= score_pos - 1:
                return user, item_pos, item_neg, self.__L(items_neg.shape[0] // trial), trial

    def fit(self, R):
        self._init_matrices(R)

        for it in range(self.iterations):
            sample = self.__sample()

            if sample is None:
                continue

            user, item_pos, item_neg, rank, trial = sample

            upd_user = self.lr * (rank * (self.I[item_neg] - self.I[item_pos]) + self.weight_decay * self.U[user])
            upd_pos = self.lr * (-rank * self.U[user] + self.weight_decay * self.I[item_pos])
            upd_neg = self.lr * (rank * self.U[user] + self.weight_decay * self.I[item_neg])

            if trial == 1:
                upd_user *= 10
                upd_pos *= 10
                upd_neg *= 10

            self.U[user] -= upd_user
            self.I[item_pos] -= upd_pos
            self.I[item_neg] -= upd_neg

            if (it + 1) % self.save_every == 0 and self.verbose:
                auc, auc2 = roc_auc(self)

                self.logger.log("iter", it + 1)
                self.logger.log("AUC", auc)
                self.logger.log("AUC2", auc2)

                if self.verbose:
                    print(f"Iter: {it + 1: 6d} | AUC: {auc:0.5f} | AUC2: {auc2:0.5f}")
