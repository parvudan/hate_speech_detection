import gensim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras.preprocessing as kp
from sklearn.base import TransformerMixin, BaseEstimator
from typing import Iterable, List, Optional


class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 common_terms_list: List[str] = None,
                 ngrams_min_count: int = 1,
                 grams_join: str = ' '):
        """
        :parameter
            :param grams_join: string - '_' (new_york), ' ' (new york)
            :param common_terms_list: list - ['of','with','without','and','or','the','a']
            :param ngrams_min_count: int - ignore all words with total collected count lower than this value
        """

        self._corpus:  Optional[Iterable[str]] = None  # Iterable cleaned text

        self.common_terms_list = common_terms_list
        if common_terms_list is None:
            self.common_terms_list = ['of', 'with', 'without', 'and', 'or', 'the', 'a']

        self.ngrams_min_count = ngrams_min_count
        self.grams_join = grams_join

        self.df_ngrams: Optional[pd.DataFrame] = None
        self.nlp: Optional[gensim.models.word2vec.Word2Vec] = None
        self.ngrams_detector_list: Optional[List[gensim.models.phrases.Phraser]] = None
        self.fitted_tokenizer: Optional[kp.text.Tokenizer] = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        self._corpus = X.copy()

        # create unigrams
        unigrams_list = self._create_unigrams()

        # create ngram detectors
        self._create_ngrams_detectors(unigrams_list)
        self._plot_ngrams()

        # apply ngram detectors
        processed_corpus = self._apply_ngrams(unigrams_list)

        # create input for lstm (sequences of tokens)
        X_train, dic_vocabulary = self._text_tokenizer(processed_corpus, top=None, oov='NaN', maxlen=15)
        self._plot_heatmap(X_train)

        # train Word2Vec from scratch
        avg_len = np.max([len(text_list) for text_list in self._corpus]) / 2
        self._fit_w2v(processed_corpus, window=avg_len)

        embeddings = self._vocabulary_embeddings(dic_vocabulary)

        return X_train, embeddings

    # region ngrams
    def _create_ngrams_detectors(self, corpus: List[List[str]], grams_join=' ', min_count=5):
        """
        Train common bigrams and trigrams detectors with gensim
        :parameter
            :param corpus: list - dtf['text']
            :param grams_join: string - '_' (new_york), ' ' (new york)
            :param lst_common_terms: list - ['of','with','without','and','or','the','a']
            :param min_count: int - ignore all words with total collected count lower than this value
        :return
            list with n-grams models and dataframe with frequency
        """

        # fit ngram models
        bigrams_detector = gensim.models.phrases.Phrases(corpus,
                                                         delimiter=grams_join.encode(),
                                                         common_terms=self.common_terms_list,
                                                         min_count=min_count,
                                                         threshold=min_count * 2)
        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)

        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[corpus],
                                                          delimiter=grams_join.encode(),
                                                          common_terms=self.common_terms_list,
                                                          min_count=min_count,
                                                          threshold=min_count * 2)
        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

        # plot
        self.df_ngrams = pd.DataFrame([
            {
                'word': grams_join.join([gram.decode() for gram in k]),
                'freq': v
            } for k, v in trigrams_detector.phrasegrams.items()
        ])
        self.df_ngrams['ngrams'] = self.df_ngrams['word'].apply(lambda x: x.count(grams_join) + 1)
        self.df_ngrams = self.df_ngrams.sort_values(['ngrams', 'freq'], ascending=[True, False])

        if self.ngrams_detector_list is None:
            self.ngrams_detector_list = [bigrams_detector, trigrams_detector]

    def _create_unigrams(self) -> List[List[str]]:
        """
        Create unigrams or 1-grams list from corpus
        """

        ngrams = 1
        # create list of n-grams
        unigrams_list = []
        for string in self._corpus:
            lst_words = string.strip().split()
            lst_grams = [self.grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
            unigrams_list.append(lst_grams)

        return unigrams_list

    def _apply_ngrams(self, unigrams_list) -> List[List[str]]:
        """ Apply the bigrams and trigrams detectors on the unigrams list """
        processed_corpus = list(unigrams_list)
        # detect common bi-grams and tri-grams
        if self.ngrams_detector_list:
            for detector in self.ngrams_detector_list:
                processed_corpus = list(detector[processed_corpus])

        return processed_corpus

    def _plot_ngrams(self):

        figsize = (10, 7)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='freq', y='word', hue='ngrams', dodge=False, ax=ax,
                    data=self.df_ngrams.groupby('ngrams')['ngrams', 'freq', 'word'].head(10))
        ax.set(xlabel=None, ylabel=None, title='Most frequent words')
        ax.grid(axis='x')
        plt.show()

    # endregion

    def _text_tokenizer(self, corpus: List[List[str]], top=None, oov=None, maxlen=None):
        """
        Transforms the corpus into an array of sequences of idx (tokenizer) with same length (padding).
        :parameter
            :param corpus: processed corpus
            :param top: num - if given the tokenizer keeps only top important words
            :param oov: string - how to encode words not in vocabulary (ex. 'NAN')
            :param maxlen: num - dimensionality of the vectors, if None takes the max length in corpus
        :return
            If training: matrix of sequences, tokenizer, dic_vocabulary. Else matrix of sequences only.
        """

        print('--- tokenization ---')
        # bow with keras to get text2tokens without creating the sparse matrix
        # train
        if self.fitted_tokenizer is None:
            # if None it creates one with fit and transorm (train set), if given it transforms only (test set)
            tokenizer = kp.text.Tokenizer(num_words=top,
                                          lower=True,
                                          split=' ',
                                          char_level=False,
                                          oov_token=oov,
                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
            tokenizer.fit_on_texts(corpus)
            self.fitted_tokenizer = tokenizer

        dic_vocabulary = self.fitted_tokenizer.word_index
        print(len(dic_vocabulary), 'words')

        # transform
        lst_text2seq = self.fitted_tokenizer.texts_to_sequences(corpus)

        # padding sequence (from [1,2],[3,4,5,6] to [0,0,1,2],[3,4,5,6])
        print('--- padding to sequence ---')
        X = kp.sequence.pad_sequences(lst_text2seq, maxlen=maxlen, padding='post', truncating='post')
        print(X.shape[0], 'sequences of length', X.shape[1])

        return X, dic_vocabulary

    @staticmethod
    def _plot_heatmap(data):
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.heatmap(data == 0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
        plt.show()

    # region word2vec
    def _fit_w2v(self, corpus: List[List[str]], size=300, window=20, sg=1, epochs=30):
        """
        Fits the Word2Vec model from gensim.
        :parameter
            :param min_count: num - ignores all words with total frequency lower than this
            :param size: num - dimensionality of the vectors
            :param window: num - ( x x x ... x  word  x ... x x x)
            :param sg: num - 1 for skip-grams, 0 for CBOW
        :return
            the nlp model
        """

        nlp = gensim.models.word2vec.Word2Vec(corpus,
                                              size=size,
                                              window=window,
                                              min_count=self.ngrams_min_count,
                                              sg=sg,
                                              iter=epochs)
        self.nlp = nlp

    def _vocabulary_embeddings(self, dic_vocabulary):
        """
        Embeds a vocabulary of unigrams with gensim w2v.
        :parameter
            :param dic_vocabulary: dict - {'word':1, 'word':2, ...}
            :param nlp: gensim model
        :return
            Matrix and the nlp model
        """

        embeddings = np.zeros((len(dic_vocabulary) + 1, self.nlp.vector_size))
        for word, idx in dic_vocabulary.items():
            # update the row with vector
            try:
                embeddings[idx] = self.nlp[word]
            # if word not in model then skip and the row stays all zeros
            except:
                pass

        print('vocabulary mapped to', embeddings.shape[0], 'vectors of size', embeddings.shape[1])
        return embeddings
    # endregion
