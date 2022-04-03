import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel


class Lda:
    def __init__(self, corpus, dictionary, num_topics, seed=42):
        self.corpus = corpus
        self.dictionary = dictionary
        self.model = LdaModel(corpus, num_topics, random_state=seed, id2word=dictionary)

    def get_vis(self, corpus=None):
        if corpus is None:
            corpus = self.corpus

        return pyLDAvis.gensim_models.prepare(self.model, corpus, self.dictionary, mds="mmds")

    def predict(self, data):
        corpus = [self.dictionary.doc2bow(text) for text in data]
        vis = self.get_vis(corpus)

        return self.model.get_document_topics(corpus), vis

    def print_major_topic(self):
        for idx, topic in self.model.print_topics(-1):
            print(f"Topic: {idx + 1} \nWords: {topic}")
