from itertools import islice
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.special import expit
from logger import Logger
from utils import roc_auc, roc_auc_2
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
            for u, i, v in np.random.permutation(samples):  # very stochastic xD
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

        return self


class ALS(MatrixFactorization):
    def __update_user(self):
        E = self.lr * np.eye(self.factors)
        II = self.I.T @ self.I
        for i in range(self.U.shape[0]):
            self.U[i] = (np.linalg.inv(II + E) @ self.I.T @ self.R[i].T).flatten()

    def __update_item(self):
        E = self.lr * np.eye(self.factors)
        UU = self.U.T @ self.U
        for j in range(self.I.shape[0]):
            self.I[j] = (np.linalg.inv(UU + E) @ self.U.T @ self.R[:, j]).flatten()

    def fit(self, R):
        self._init_matrices(R)

        for it in range(self.iterations):
            self.__update_user()
            self.__update_item()

            if (it + 1) % self.save_every == 0:
                rmse_ = self._calc_rmse()

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

        sigmoid = np.tile(expit(-r_uij), (self.factors, 1)).T

        upd_u = sigmoid * (item_j - item_i) + self.weight_decay * user_u
        upd_i = -sigmoid * user_u + self.weight_decay * item_i
        upd_j = sigmoid * user_u + self.weight_decay * item_j
        self.U[u] -= self.lr * upd_u
        self.I[i] -= self.lr * upd_i
        self.I[j] -= self.lr * upd_j

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
                auc, auc2 = roc_auc(R, self)

                self.logger.log("iter", it + 1)
                self.logger.log("AUC", auc)
                self.logger.log("AUC2", auc2)

                if self.verbose:
                    print(f"Iter: {it + 1: 6d} | AUC: {auc:0.5f} | AUC2: {auc2:0.5f}")

        return self


class WARP(MatrixFactorization):
    def __sample(self):
        n_users, n_items = self.R.shape

        user = np.random.choice(n_users)
        items_pos = self.R[user].nonzero()[1]

        if items_pos.shape[0] == 0:
            return

        items_neg = np.arange(n_items)[
            ~np.isin(np.arange(n_items), items_pos)
        ]  # kinda trick to extract negative items?
        np.random.shuffle(items_neg)

        item_pos = np.random.choice(items_pos)
        score_pos = self.U[user] @ self.I[item_pos]

        for trial, item_neg in enumerate(items_neg, start=1):
            score_neg = self.U[user] @ self.I[item_neg]
            margin = 1 - score_pos + score_neg
            if margin > 0:
                self.__num_of_trials += trial
                return user, item_pos, item_neg, trial, items_neg.shape[0]

            self.__correct_cnt += score_pos > score_neg

        # in case we haven't found any violating negative item
        # we still need to update number of trials made
        self.__num_of_trials += trial

    def __step(self):
        self.__correct_cnt = self.__num_of_trials = 0
        for _ in range(self.R.shape[0]):
            sample = self.__sample()
            if sample is None:
                continue

            user, item_pos, item_neg, num_of_trials, items_neg_total = sample
            loss = np.log(np.floor(items_neg_total / num_of_trials))

            upd_user = self.lr * (loss * (self.I[item_neg] - self.I[item_pos]) + self.weight_decay * self.U[user])
            upd_pos = self.lr * (-loss * self.U[user] + self.weight_decay * self.I[item_pos])
            upd_neg = self.lr * (loss * self.U[user] + self.weight_decay * self.I[item_neg])

            if num_of_trials == 1:
                upd_user *= 10
                upd_pos *= 10
                upd_neg *= 10

            self.U[user] -= upd_user
            self.I[item_pos] -= upd_pos
            self.I[item_neg] -= upd_neg

        return self.__correct_cnt / self.__num_of_trials

    def fit(self, R):
        self._init_matrices(R)

        for it in range(self.iterations):
            acc = self.__step()

            if (it + 1) % self.save_every == 0:
                auc, auc2 = roc_auc(R, self)

                self.logger.log("iter", it + 1)
                self.logger.log("AUC", auc)
                self.logger.log("AUC2", auc2)
                self.logger.log("acc", acc)

                if self.verbose:
                    print(f"Iter: {it + 1: 6d} | AUC: {auc:0.5f} | AUC2: {auc2:0.5f} | ACC: {acc:0.5f}")

        return self
