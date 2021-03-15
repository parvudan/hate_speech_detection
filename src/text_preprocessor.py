import re
import emoji
import string
import numpy as np
import pandas as pd
# from normalise import normalise
import multiprocessing as mp
from sklearn.base import TransformerMixin, BaseEstimator
import spacy
from spacy.tokens import Doc
from typing import Dict


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    - Converting to lower case
    - Tokenizing
    - Removing punctuation and URL links
    - Removing stop words
    - Lemmatization
    - Removing emojis (english) and numbers (you can leave emojis in if your model can read them in, and if you plan to do emoji analysis)
    """

    def __init__(self,
                 nlp: spacy.Language = None,
                 variety: str = 'BrE',
                 user_abbrevs: Dict = None,
                 n_jobs: int = 1):
        """
        Adapted from:
        https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a

        variety - format of date (AmE - american type, BrE - british format)
        user_abbrevs - dict of user abbreviations mappings (from normalise package)
        n_jobs - parallel jobs to run
        """

        self._nlp = nlp
        self.variety = variety
        self.user_abbrevs = user_abbrevs or {}
        self.n_jobs = n_jobs

    @property
    def nlp(self):
        if self._nlp is None:
            # load spacy language vectors
            self._nlp = spacy.load('en_core_web_lg')
        return self._nlp

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        # cores = mp.cpu_count()
        cores = 2
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        pool.close()
        pool.join()

        return data

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text: str):
        normalized_text = self._normalize(text)

        removed_links = self._remove_links(normalized_text)
        removed_emojis = self._remove_emojis(removed_links)
        removed_numbers = self._remove_numbers(removed_emojis)
        removed_punct = self._remove_punct(removed_numbers)
        removed_spaces = self._remove_multiple_spaces(removed_punct)

        doc = self.nlp(removed_spaces)
        removed_stop_words = self._remove_stop_words(doc)

        return self._lemmatize(removed_stop_words)

    def _normalize(self, text: str):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text.lower()

    @staticmethod
    def _remove_links(s: str):
        return re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)

    @staticmethod
    def _remove_numbers(s: str):
        return re.sub(r'\d*', '', s)

    @staticmethod
    def _remove_punct(s: str):
        return ''.join([c if c not in string.punctuation else ' ' for c in s])

    @staticmethod
    def _remove_stop_words(doc: Doc):
        return [t for t in doc if t.text.strip() and not t.is_stop]

    @staticmethod
    def _remove_emojis(s: str):
        return ''.join([c for c in s if c not in emoji.UNICODE_EMOJI['en']])

    @staticmethod
    def _remove_multiple_spaces(s: str):
        return re.sub(r'\s{2,}', ' ', s)

    @staticmethod
    def _lemmatize(doc: Doc):
        return ' '.join([t.lemma_ for t in doc])
