"""
- Converting to lower case
- Tokenizing
- Removing punctuation and URL links
- Removing stop words
- Lemmatization
- Removing emojis (english) and numbers (you can leave emojis in if your model can read them in, and if you plan to do emoji analysis)
"""

import re
import emoji
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from normalise import normalise
import multiprocessing as mp
import string
from sklearn import manifold
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import gensim

import spacy
nlp = spacy.load("en_core_web_lg")


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        """
        Adapted from:
        https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a

        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
            4. Lemmatization

        variety - format of date (AmE - american type, BrE - british format)
        user_abbrevs - dict of user abbreviations mappings (from normalise package)
        n_jobs - parallel jobs to run
        """

        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

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

    def _preprocess_text(self, text):
        normalized_text = self._normalize(text)
        # doc = nlp(normalized_text)

        removed_links = self._remove_links(normalized_text)
        removed_emojis = self._remove_emojis(removed_links)
        removed_numbers = self._remove_numbers(removed_emojis)
        removed_punct = self._remove_punct(removed_numbers)
        removed_spaces = self._remove_multiple_spaces(removed_punct)

        doc = nlp(removed_spaces)
        removed_stop_words = self._remove_stop_words(doc)

        return self._lemmatize(removed_stop_words)

    def _normalize(self, text):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text.lower()

    def _remove_links(self, s):
        return re.sub(r'https?:\/\/.*[\r\n]*', '', s, flags=re.MULTILINE)

    def _remove_numbers(self, s):
        return re.sub(r'\d*', '', s)

    def _remove_punct(self, s):
        return ''.join([c if c not in string.punctuation else ' ' for c in s])

    def _remove_stop_words(self, doc):
        return [t for t in doc if t.text.strip() and not t.is_stop]

    def _remove_emojis(self, s):
        return ''.join([c for c in s if c not in emoji.UNICODE_EMOJI['en']])

    def _remove_multiple_spaces(self, s):
        return re.sub(r'\s{2,}', ' ', s)

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])


def visualize_lengths(data, title):
    """ Visualizing lengths of tokens in each tweet """
    lengths = [len(i) for i in data]
    plt.figure(figsize=(13, 6))
    plt.hist(lengths, bins=40)
    plt.title(title)
    plt.show()


def plot_categories(data):
    fig, ax = plt.subplots()
    fig.suptitle("categories", fontsize=12)

    data = data.replace({'class': {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}})
    data["class"].reset_index().groupby("class").count().sort_values(by="index").plot(kind="barh", legend=False,ax=ax).grid(axis='x')
    plt.show()


def plot_sparse_matrix(data):
    sns.heatmap(data.todense()[:, np.random.randint(0, data.shape[1], 100)] == 0, vmin=0, vmax=1, cbar=False).set_title(
        'Sparse Matrix Sample')
    plt.show()


def plot_3d_pca(nlp, word='ugly'):
    fig = plt.figure()

    # word embedding
    tot_words = [word] + [tupla[0] for tupla in
                          nlp.most_similar(word, topn=20)]
    X = nlp[tot_words]

    # pca to reduce dimensionality from 300 to 3
    pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
    X = pca.fit_transform(X)

    ## create dtf
    dtf_ = pd.DataFrame(X, index=tot_words, columns=["x", "y", "z"])
    dtf_["input"] = 0
    dtf_["input"].iloc[0:1] = 1

    # plot 3d
    from mpl_toolkits.mplot3d import Axes3D

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dtf_[dtf_["input"] == 0]['x'],
               dtf_[dtf_["input"] == 0]['y'],
               dtf_[dtf_["input"] == 0]['z'], c="black")
    ax.scatter(dtf_[dtf_["input"] == 1]['x'],
               dtf_[dtf_["input"] == 1]['y'],
               dtf_[dtf_["input"] == 1]['z'], c="red")
    ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[],
           yticklabels=[], zticklabels=[])

    for label, row in dtf_[["x", "y", "z"]].iterrows():
        x, y, z = row
        ax.text(x, y, z, s=label)

    plt.show()


def select_interesting_features(training_data):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
    vectorizer.fit(training_data)
    X_vec_train = vectorizer.transform(training_data)
    plot_sparse_matrix(X_vec_train)

    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.9
    dtf_features = pd.DataFrame()

    for cat in np.unique(training_data):
        chi2, p = feature_selection.chi2(X_vec_train, training_data == cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat}))
        dtf_features = dtf_features.sort_values(["y", "score"], ascending=[True, False])
        dtf_features = dtf_features[dtf_features["score"] > p_value_limit]

    selected_feature_names = dtf_features["feature"].unique().tolist()

    for cat in np.unique(training_data):
        print("# {}:".format(cat))
        print("  . selected features:", len(dtf_features[dtf_features["y"] == cat]))
        print("  . top features:", ",".join(dtf_features[dtf_features["y"] == cat]["feature"].values[:10]))
        print(" ")

    vectorizer = TfidfVectorizer(vocabulary=selected_feature_names)
    vectorizer.fit(training_data)
    filtered_training_data = vectorizer.transform(training_data)
    # dic_vocabulary = vectorizer.vocabulary_
    plot_sparse_matrix(filtered_training_data)
    return filtered_training_data


def word2vec_tokenizer(corpus):
    # create list of lists of unigrams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i + 1])
                     for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)

    # # detect bigrams and trigrams
    # bigrams_detector = gensim.models.phrases.Phrases(lst_corpus,
    #                                                  delimiter=" ".encode(), min_count=5, threshold=10)
    # bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    # trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
    #                                                   delimiter=" ".encode(), min_count=5, threshold=10)
    # trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    # fit w2v
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=300,
                                          window=8, min_count=1, sg=1, iter=30)


    import keras.preprocessing as kprocessing
    # tokenize text
    tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ',
                                           oov_token="NaN",
                                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(lst_corpus)
    dic_vocabulary = tokenizer.word_index

    # create sequence
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)
    # padding sequence
    X_train = kprocessing.sequence.pad_sequences(lst_text2seq,
                                                 maxlen=15, padding="post", truncating="post")

    sns.heatmap(X_train == 0, vmin=0, vmax=1, cbar=False)
    plt.show()

    i = 0

    # list of text: ["I like this", ...]
    len_txt = len(corpus.iloc[i].split())
    print("from: ", corpus.iloc[i], "| len:", len_txt)

    # sequence of token ids: [[1, 2, 3], ...]
    len_tokens = len(X_train[i])
    print("to: ", X_train[i], "| len:", len(X_train[i]))

    # vocabulary: {"I":1, "like":2, "this":3, ...}
    print("check: ", corpus.iloc[i].split()[0],
          " -- idx in vocabulary -->",
          dic_vocabulary[corpus.iloc[i].split()[0]])

    print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")

    # start the matrix (length of vocabulary x vector size) with all 0s
    embeddings = np.zeros((len(dic_vocabulary) + 1, 300))
    for word, idx in dic_vocabulary.items():
        # update the row with vector
        try:
            embeddings[idx] = nlp[word]
        # if word not in model then skip and the row stays all 0s
        except:
            pass

    return embeddings


if __name__ == '__main__':
    df = pd.read_csv('t_davidson_hate_speech_and_offensive_language.csv')
    # plot_categories(df)

    sentences = df['tweet']
    # y = df['class'].values
    mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    y = df[['class']]
    y = y.replace({'class': mapping})['class']
    # classes = df[['hate_speech', 'offensive_language', 'neither']].values
    # y = np.zeros_like(classes)
    # y[np.arange(len(classes)), classes.argmax(1)] = 1

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.05, random_state=1000)

    clf = Pipeline(steps=[
        ('normalize', TextPreprocessor(n_jobs=0)),
        # ('features', TfidfVectorizer(analyze='word', ngram_range=(1, 1), sublinear_tf=True))
    ])

    sentences_train_cleaned = clf.fit_transform(sentences_train)
    sentences_test_cleaned = clf.fit_transform(sentences_test)

    # X_train_filtered = select_interesting_features(X_train)

    # emb = word2vec_tokenizer(X_train)

    from model_lstm.word2vec_utils import create_ngrams_detectors, text2seq, fit_w2v, plot_w2v, vocabulary_embeddings

    lst_common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    lst_ngrams_detectors, dtf_ngrams = create_ngrams_detectors(corpus=sentences_train_cleaned,
                                                               lst_common_terms=lst_common_terms, min_count=2)

    # create input for lstm (sequences of tokens)
    dic_seq = text2seq(corpus=sentences_train_cleaned, lst_ngrams_detectors=lst_ngrams_detectors,
                       top=None, oov="NaN", maxlen=15)

    X_train, tokenizer, dic_vocabulary = dic_seq["X"], dic_seq["tokenizer"], dic_seq["dic_vocabulary"]

    # Preprocess Test with the same tokenizer
    X_test = text2seq(corpus=sentences_test_cleaned, lst_ngrams_detectors=lst_ngrams_detectors,
                      fitted_tokenizer=tokenizer, maxlen=X_train.shape[1])

    # Or train Word2Vec from scratch
    avg_len = np.max([len(text.split()) for text in sentences_train_cleaned]) / 2

    lst_corpus, nlp = fit_w2v(corpus=sentences_train_cleaned, lst_ngrams_detectors=lst_ngrams_detectors,
                              min_count=1, size=300, window=avg_len, sg=0, epochs=30)

    # plot_w2v(lst_words=['nigga'], nlp=nlp, plot_type="3d", top=200, annotate=True, figsize=(10, 5))
    # plot_w2v(lst_words=None, nlp=nlp, plot_type="2d", annotate=False, figsize=(20, 10))

    embeddings = vocabulary_embeddings(dic_vocabulary, nlp)

    from model_lstm.model_arch import model_builder, fit_dl_classif, evaluate_multi_classif

    model = model_builder(X_train, y_train, embeddings)

    model, predicted_prob, predicted = fit_dl_classif(X_train, y_train, X_test, encode_y=True,
                                                      model=model, epochs=10, batch_size=256)

    evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15, 5))

    print('Done!')
