import gensim
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


class Dataset:
    _STOPWORDS = set(stopwords.words("english") + ["`", ":", "'", ",", ".", "I'm", "``", "'s", "@", "n't"])

    def __init__(self, path, versions=None):
        self.versions = versions

        raw_data = pd.read_json(path, lines=True).rename(columns={"Affected versions": "versions"})
        raw_data.summary = raw_data.summary.astype("string")
        raw_data.description = raw_data.description.astype("string")

        self.raw_data = raw_data

        if versions is not None:
            self.raw_data = self.get_data_by_versions(versions)

        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

        self.dictionaries = {
            "summary": self._construct_dictionary(self.raw_data.summary),
            "description": self._construct_dictionary(self.raw_data.description),
        }

        self.summary = self.raw_data.summary.map(self._preprocess)
        self.description = self.raw_data.description.map(self._preprocess)

        self.corpuses = {
            "summary": [self.dictionaries["summary"].doc2bow(text) for text in self.summary],
            "description": [self.dictionaries["description"].doc2bow(text) for text in self.description],
        }

    def _construct_dictionary(self, raw_data):
        data = raw_data.map(self._preprocess)

        dictionary = gensim.corpora.Dictionary(data)
        dictionary.filter_extremes()

        return dictionary

    def _filter_versions(self, value, versions=("2020.2", "2020.3", "2021.1", "2021.2", "2021.3")):
        for ver in value:
            if ver in versions:
                return True
        return False

    def _get_version_mask(self, versions):
        return self.raw_data.versions.apply(lambda x: self._filter_versions(x, versions))

    def get_data_by_versions(self, versions):
        msk = self._get_version_mask(versions)
        return self.raw_data[msk]

    def get_summary_by_versions(self, versions):
        msk = self._get_version_mask(versions)
        return self.summary[msk]

    def get_description_by_versions(self, versions):
        msk = self._get_version_mask(versions)
        return self.description[msk]

    def get_summary_corpus_by_version(self, versions):
        msk = self._get_version_mask(versions)
        return [self.dictionaries["summary"].doc2bow(text) for text in self.summary[msk]]

    def get_description_corpus_by_version(self, versions):
        msk = self._get_version_mask(versions)
        return [self.dictionaries["description"].doc2bow(text) for text in self.description[msk]]

    def _lemmatize(self, text):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos="v"))

    def _preprocess(self, value):
        text = str(value)
        tokens = simple_preprocess(text)

        filtered = [self._lemmatize(w) for w in tokens if not w in self._STOPWORDS and len(w) > 3]
        return filtered
