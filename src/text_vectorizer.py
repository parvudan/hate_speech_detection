import gensim
import numpy as np
import pandas as pd
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

        self._corpus:  Optional[Iterable[str]] = None
        self._processed_corpus: Optional[List[List[str]]] = None
        self.common_terms_list = common_terms_list
        if common_terms_list is None:
            self.common_terms_list = ['of', 'with', 'without', 'and', 'or', 'the', 'a']

        self.ngrams_min_count = ngrams_min_count
        self.grams_join = grams_join

        self.dtf_ngrams: Optional[pd.DataFrame] = None
        self.nlp: Optional[gensim.models.word2vec.Word2Vec] = None
        self.ngrams_detector_list: Optional[List[gensim.models.phrases.Phraser]] = None
        self.fitted_tokenizer: Optional[kp.text.Tokenizer] = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        self._corpus = X.copy()

        # create unigrams
        self._utils_preprocess_ngrams()
        # unigrams_list = self._make_unigrams()

        # create ngram detectors
        self._create_ngrams_detectors()

        # apply ngram detectors
        self._utils_preprocess_ngrams()

        # create input for lstm (sequences of tokens)
        dic_seq = self._text2seq(top=None, oov='NaN', maxlen=15)
        X_train, tokenizer, dic_vocabulary = dic_seq['X'], dic_seq['tokenizer'], dic_seq['dic_vocabulary']

        # train Word2Vec from scratch
        avg_len = np.max([len(text_list) for text_list in self._corpus]) / 2

        nlp = self._fit_w2v(min_count=1, size=300, window=avg_len, sg=0, epochs=30)

        embeddings = self._vocabulary_embeddings(dic_vocabulary, nlp)

        return X_train, embeddings

    def _create_ngrams_detectors(self, grams_join=' ', min_count=5):
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
        bigrams_detector = gensim.models.phrases.Phrases(self._processed_corpus,
                                                         delimiter=grams_join.encode(),
                                                         common_terms=self.common_terms_list,
                                                         min_count=min_count,
                                                         threshold=min_count * 2)
        bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)

        trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[self._processed_corpus],
                                                          delimiter=grams_join.encode(),
                                                          common_terms=self.common_terms_list,
                                                          min_count=min_count,
                                                          threshold=min_count * 2)
        trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

        # plot
        self.dtf_ngrams = pd.DataFrame([
            {
                'word': grams_join.join([gram.decode() for gram in k]),
                'freq': v
            } for k, v in trigrams_detector.phrasegrams.items()
        ])
        self.dtf_ngrams['ngrams'] = self.dtf_ngrams['word'].apply(lambda x: x.count(grams_join) + 1)
        self.dtf_ngrams = self.dtf_ngrams.sort_values(['ngrams', 'freq'], ascending=[True, False])

        if self.ngrams_detector_list is None:
            self.ngrams_detector_list = [bigrams_detector, trigrams_detector]

    def _make_unigrams(self):
        ngrams = 1
        # create list of n-grams
        unigrams_list = []
        for string in self._corpus:
            lst_words = string.strip().split()
            lst_grams = [self.grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
            unigrams_list.append(lst_grams)

        return unigrams_list

    def _utils_preprocess_ngrams(self, ngrams=1):
        """
        Create a list of lists of grams with gensim:
            [ ['hi', 'my', 'name', 'is', 'Tom'],
              ['what', 'is', 'yours'] ]
        :parameter
            :param corpus: list - dtf['text']
            :param ngrams: num - ex. 'new', 'york'
            :param grams_join: string - '_' (new_york), ' ' (new york)
            :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
        :return
            lst of lists of n-grams
        """

        # create list of n-grams
        lst_corpus = []
        for string in self._corpus:
            lst_words = string.strip().split()
            lst_grams = [self.grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
            lst_corpus.append(lst_grams)

        # detect common bi-grams and tri-grams
        if self.ngrams_detector_list:
            for detector in self.ngrams_detector_list:
                lst_corpus = list(detector[lst_corpus])

        self._processed_corpus = lst_corpus

    def _text2seq(self, fitted_tokenizer=None, top=None, oov=None, maxlen=None):
        """
        Transforms the corpus into an array of sequences of idx (tokenizer) with same length (padding).
        :parameter
            :param corpus: list - dtf['text']
            :param ngrams: num - ex. 'new', 'york'
            :param grams_join: string - '_' (new_york), ' ' (new york)
            :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
            :param fitted_tokenizer: keras tokenizer - if None it creates one with fit and transorm (train set), if given it transforms only (test set)
            :param top: num - if given the tokenizer keeps only top important words
            :param oov: string - how to encode words not in vocabulary (ex. 'NAN')
            :param maxlen: num - dimensionality of the vectors, if None takes the max length in corpus
            :param padding: string - 'pre' for [9999,1,2,3] or 'post' for [1,2,3,9999]
        :return
            If training: matrix of sequences, tokenizer, dic_vocabulary. Else matrix of sequences only.
        """

        print('--- tokenization ---')
        # bow with keras to get text2tokens without creating the sparse matrix
        # train
        if self.fitted_tokenizer is None:
            tokenizer = kp.text.Tokenizer(num_words=top,
                                          lower=True,
                                          split=' ',
                                          char_level=False,
                                          oov_token=oov,
                                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
            tokenizer.fit_on_texts(self._processed_corpus)
            self.fitted_tokenizer = tokenizer
        else:
            tokenizer = self.fitted_tokenizer

        dic_vocabulary = tokenizer.word_index
        print(len(dic_vocabulary), 'words')

        # transform
        lst_text2seq = tokenizer.texts_to_sequences(self._processed_corpus)

        # padding sequence (from [1,2],[3,4,5,6] to [0,0,1,2],[3,4,5,6])
        print('--- padding to sequence ---')
        X = kp.sequence.pad_sequences(lst_text2seq, maxlen=maxlen, padding='post', truncating='post')
        print(X.shape[0], 'sequences of length', X.shape[1])

        return {'X': X, 'tokenizer': tokenizer, 'dic_vocabulary': dic_vocabulary} if fitted_tokenizer is None else X

    def _fit_w2v(self, min_count=1, size=300, window=20, sg=1, epochs=30):
        """
        Fits the Word2Vec model from gensim.
        :parameter
            :param ngrams: num - ex. 'new', 'york'
            :param grams_join: string - '_' (new_york), ' ' (new york)
            :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
            :param min_count: num - ignores all words with total frequency lower than this
            :param size: num - dimensionality of the vectors
            :param window: num - ( x x x ... x  word  x ... x x x)
            :param sg: num - 1 for skip-grams, 0 for CBOW
        :return
            lst_corpus and the nlp model
        """

        nlp = gensim.models.word2vec.Word2Vec(self._processed_corpus, size=size, window=window, min_count=min_count, sg=sg, iter=epochs)
        return nlp.wv

    def _vocabulary_embeddings(self, dic_vocabulary, nlp=None):
        """
        Embeds a vocabulary of unigrams with gensim w2v.
        :parameter
            :param dic_vocabulary: dict - {'word':1, 'word':2, ...}
            :param nlp: gensim model
        :return
            Matrix and the nlp model
        """

        embeddings = np.zeros((len(dic_vocabulary) + 1, nlp.vector_size))
        for word, idx in dic_vocabulary.items():
            # update the row with vector
            try:
                embeddings[idx] = nlp[word]
            # if word not in model then skip and the row stays all zeros
            except:
                pass

        print('vocabulary mapped to', embeddings.shape[0], 'vectors of size', embeddings.shape[1])
        return embeddings
