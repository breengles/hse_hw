import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from catboost import CatBoostRanker, Pool


class CatBoost:
    def __init__(self, iterations) -> None:
        self.model = CatBoostRanker(
            iterations=iterations,
            task_type="GPU",
            custom_metric=["NDCG", "QueryAUC:type=Ranking"],
            metric_period=10,
            train_dir="YetiRank/",
        )

    @staticmethod
    def pool(data):
        X = data.drop(columns=["target", "song_id", "msno"], axis=1)
        y = data.target.to_numpy()
        q = data.msno.cat.codes.to_numpy()

        categorical_features = X.select_dtypes(["category"]).columns.to_numpy()
        return Pool(data=X, label=y, group_id=q, cat_features=categorical_features, has_header=True)

    def fit(self, data):
        self.model.fit(self.pool(data))
        return self

    def predict(self, data):
        return self.model.predict(self.pool(data))


class Embeddings:
    def __init__(self, embedding_dim, window=5, min_count=5, seed=42) -> None:
        self.embedding_dim = embedding_dim
        self.user_encoder = LabelEncoder()
        self.w2v = Word2Vec(vector_size=embedding_dim, window=window, min_count=min_count, seed=seed)

        self.users = None
        self.user_embeddings = None

    def fit(self, data: pd.DataFrame, epochs=10):
        # items
        print("Fitting items...")
        sessions = dict(data.groupby("msno").song_id.apply(list))

        seqs = [x for x in sessions.values() if len(x) > 0]

        self.w2v.build_vocab(seqs)
        self.w2v.train(seqs, total_examples=self.w2v.corpus_count, epochs=epochs)

        # users
        print("Fitting users...")
        pos = dict(data[data.target == 1].groupby("msno").song_id.apply(list))
        self.user_encoder.fit(list(pos.keys()))
        self.users = set(self.user_encoder.classes_)

        self.user_embeddings = np.zeros((len(self.users), self.embedding_dim))
        for user, items in tqdm(pos.items()):
            positives = [item for item in items if item in self.w2v.wv]

            if len(positives) > 0:
                encoded_user = self.user_encoder.transform([user])[0]
                self.user_embeddings[encoded_user] = self.w2v.wv[positives].mean(0)

        return self

    def predict(self, data):
        scores = np.zeros(len(data))

        users = data.msno.to_numpy()
        items = data.song_id.to_numpy()

        msk = np.array([user in self.users and item in self.w2v.wv for user, item in zip(users, items)])

        embeddings_user = self.user_embeddings[self.user_encoder.transform(users[msk])]
        embeddings_item = self.w2v.wv[items[msk]]

        scores[msk] = (embeddings_user * embeddings_item).sum(1)

        return scores

    def fit_predict(self, data, epochs=10):
        self.fit(data, epochs=epochs)
        return self.predict(data)


class Stack:
    def __init__(self, catboost_model: CatBoost, embedding_model: Embeddings) -> None:
        self.catboost = catboost_model
        self.embedding = embedding_model

    def fit(self, data, epochs=10):
        # appending scores inplace so we won't get high mem usage
        scores = self.embedding.fit_predict(data, epochs=epochs)
        data["scores"] = scores

        self.catboost.fit(data)
        data.drop("scores", axis=1, inplace=True)

    def predict(self, data):
        scores = self.embedding.predict(data)
        data["scores"] = scores

        outs = self.catboost.predict(data)
        data.drop("scores", axis=1, inplace=True)
        return outs

    def get_emb_score(self, data):
        return self.embedding.predict(data)
