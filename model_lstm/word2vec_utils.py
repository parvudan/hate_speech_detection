import gensim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras.preprocessing
import sklearn.manifold as manifold


def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    """
    Create a list of lists of grams with gensim:
        [ ["hi", "my", "name", "is", "Tom"],
          ["what", "is", "yours"] ]
    :parameter
        :param corpus: list - dtf["text"]
        :param ngrams: num - ex. "new", "york"
        :param grams_join: string - "_" (new_york), " " (new york)
        :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :return
        lst of lists of n-grams
    """

    # create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)

    # detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return lst_corpus


def create_ngrams_detectors(corpus, grams_join=" ", lst_common_terms=[], min_count=5):
    """
    Train common bigrams and trigrams detectors with gensim
    :parameter
        :param corpus: list - dtf["text"]
        :param grams_join: string - "_" (new_york), " " (new york)
        :param lst_common_terms: list - ["of","with","without","and","or","the","a"]
        :param min_count: int - ignore all words with total collected count lower than this value
    :return
        list with n-grams models and dataframe with frequency
    """

    def plot_ngrams():
        figsize = (10, 7)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                    data=dtf_ngrams.groupby('ngrams')["ngrams", "freq", "word"].head(10))
        ax.set(xlabel=None, ylabel=None, title="Most frequent words")
        ax.grid(axis="x")
        plt.show()

    # fit models
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=1, grams_join=grams_join)
    bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, delimiter=grams_join.encode(),
                                                     common_terms=lst_common_terms,
                                                     min_count=min_count, threshold=min_count * 2)
    bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], delimiter=grams_join.encode(),
                                                      common_terms=lst_common_terms,
                                                      min_count=min_count, threshold=min_count * 2)
    trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

    # plot
    dtf_ngrams = pd.DataFrame([{"word": grams_join.join([gram.decode() for gram in k]), "freq": v} for k, v in
                               trigrams_detector.phrasegrams.items()])
    dtf_ngrams["ngrams"] = dtf_ngrams["word"].apply(lambda x: x.count(grams_join) + 1)
    dtf_ngrams = dtf_ngrams.sort_values(["ngrams", "freq"], ascending=[True, False])

    plot_ngrams()

    return [bigrams_detector, trigrams_detector], dtf_ngrams


def vocabulary_embeddings(dic_vocabulary, nlp=None):
    """
    Embeds a vocabulary of unigrams with gensim w2v.
    :parameter
        :param dic_vocabulary: dict - {"word":1, "word":2, ...}
        :param nlp: gensim model
    :return
        Matric and the nlp model
    """

    # nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    assert nlp is not None

    embeddings = np.zeros((len(dic_vocabulary)+1, nlp.vector_size))
    for word, idx in dic_vocabulary.items():
        # update the row with vector
        try:
            embeddings[idx] =  nlp[word]
        # if word not in model then skip and the row stays all zeros
        except:
            pass

    print("vocabulary mapped to", embeddings.shape[0], "vectors of size", embeddings.shape[1])
    return embeddings


def fit_w2v(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], min_count=1, size=300, window=20, sg=1, epochs=30):
    """
    Fits the Word2Vec model from gensim.
    :parameter
        :param corpus: list - dtf["text"]
        :param ngrams: num - ex. "new", "york"
        :param grams_join: string - "_" (new_york), " " (new york)
        :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
        :param min_count: num - ignores all words with total frequency lower than this
        :param size: num - dimensionality of the vectors
        :param window: num - ( x x x ... x  word  x ... x x x)
        :param sg: num - 1 for skip-grams, 0 for CBOW
    :return
        lst_corpus and the nlp model
    """

    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors)
    nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=size, window=window, min_count=min_count, sg=sg, iter=epochs)
    return lst_corpus, nlp.wv


def embedding_w2v(x, nlp=None, value_na=0):
    """
    Creates a feature matrix (num_docs x vector_size)
    :parameter
        :param x: string or list
        :param nlp: gensim model
        :param value_na: value to return when the word is not in vocabulary
    :return
        vector or matrix
    """

    # nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    assert nlp is not None

    null_vec = [value_na] * nlp.vector_size

    # single word --> vec (size,)
    if (type(x) is str) and (len(x.split()) == 1):
        X = nlp[x] if x in nlp.vocab.keys() else null_vec

    # list of words --> matrix (n, size)
    elif (type(x) is list) and (type(x[0]) is str) and (len(x[0].split()) == 1):
        X = np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in x])

    # list of lists of words --> matrix (n mean vectors, size)
    elif (type(x) is list) and (type(x[0]) is list):
        lst_mean_vecs = []
        for lst in x:
            lst_mean_vecs.append(np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in lst]
                                          ).mean(0))
        X = np.array(lst_mean_vecs)

    # single text --> matrix (n words, size)
    elif (type(x) is str) and (len(x.split()) > 1):
        X = np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in x.split()])

    # list of texts --> matrix (n mean vectors, size)
    else:
        lst_mean_vecs = []
        for txt in x:
            lst_mean_vecs.append(np.array([nlp[word] if word in nlp.vocab.keys() else null_vec for word in txt.split()]
                                          ).mean(0))
        X = np.array(lst_mean_vecs)

    return X


def text2seq(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], fitted_tokenizer=None, top=None, oov=None,
             maxlen=None):
    """
    Transforms the corpus into an array of sequences of idx (tokenizer) with same length (padding).
    :parameter
        :param corpus: list - dtf["text"]
        :param ngrams: num - ex. "new", "york"
        :param grams_join: string - "_" (new_york), " " (new york)
        :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
        :param fitted_tokenizer: keras tokenizer - if None it creates one with fit and transorm (train set), if given it transforms only (test set)
        :param top: num - if given the tokenizer keeps only top important words
        :param oov: string - how to encode words not in vocabulary (ex. "NAN")
        :param maxlen: num - dimensionality of the vectors, if None takes the max length in corpus
        :param padding: string - "pre" for [9999,1,2,3] or "post" for [1,2,3,9999]
    :return
        If training: matrix of sequences, tokenizer, dic_vocabulary. Else matrix of sequences only.
    """

    def plot_heatmap():
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.heatmap(X == 0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
        plt.show()

    print("--- tokenization ---")

    # detect common n-grams in corpus
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join,
                                         lst_ngrams_detectors=lst_ngrams_detectors)

    # bow with keras to get text2tokens without creating the sparse matrix
    # train
    if fitted_tokenizer is None:
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=top, lower=True, split=' ', char_level=False, oov_token=oov,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(lst_corpus)
    else:
        tokenizer = fitted_tokenizer

    dic_vocabulary = tokenizer.word_index
    print(len(dic_vocabulary), "words")

    # transform
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    # padding sequence (from [1,2],[3,4,5,6] to [0,0,1,2],[3,4,5,6])
    print("--- padding to sequence ---")
    X = keras.preprocessing.sequence.pad_sequences(lst_text2seq, maxlen=maxlen, padding="post", truncating="post")
    print(X.shape[0], "sequences of length", X.shape[1])

    plot_heatmap()

    return {"X": X, "tokenizer": tokenizer, "dic_vocabulary": dic_vocabulary} if fitted_tokenizer is None else X


def plot_w2v(lst_words=None, nlp=None, plot_type="2d", top=20, annotate=True, figsize=(10, 5)):
    """
    Plot words in vector space (2d or 3d).
    :parameter
        :param lst_words: list - ["donald trump","china", ...]. If None, it plots the whole vocabulary
        :param nlp: gensim model
        :param plot_type: string - "2d" or "3d"
        :param top: num - plot top most similar words (only if lst_words is given)
        :param annotate: bool - include word text
    """

    # nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    assert nlp is not None

    fig = plt.figure(figsize=figsize)
    if lst_words is not None:
        fig.suptitle("Word: " + lst_words[0], fontsize=12) if len(lst_words) == 1 else fig.suptitle(
            "Words: " + str(lst_words[:5]), fontsize=12)
    else:
        fig.suptitle("Vocabulary")
    try:
        # word embedding
        tot_words = lst_words + [tupla[0] for tupla in
                                 nlp.most_similar(lst_words, topn=top)] if lst_words is not None else list(
            nlp.vocab.keys())
        X = nlp[tot_words]

        # pca
        pca = manifold.TSNE(perplexity=40, n_components=int(plot_type[0]), init='pca')
        X = pca.fit_transform(X)

        # create dtf
        columns = ["x", "y"] if plot_type == "2d" else ["x", "y", "z"]
        dtf = pd.DataFrame(X, index=tot_words, columns=columns)
        dtf["input"] = 0
        if lst_words is not None:
            dtf["input"].iloc[0:len(lst_words)] = 1  # <--this makes the difference between vocabulary and input words

        # plot 2d
        if plot_type == "2d":
            ax = fig.add_subplot()
            sns.scatterplot(data=dtf, x="x", y="y", hue="input", legend=False, ax=ax, palette={0: 'black', 1: 'red'})
            ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], yticks=[], yticklabels=[])
            if annotate is True:
                for i in range(len(dtf)):
                    ax.annotate(dtf.index[i], xy=(dtf["x"].iloc[i], dtf["y"].iloc[i]),
                                xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

        # plot 3d
        elif plot_type == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dtf[dtf["input"] == 0]['x'], dtf[dtf["input"] == 0]['y'], dtf[dtf["input"] == 0]['z'], c="black")
            ax.scatter(dtf[dtf["input"] == 1]['x'], dtf[dtf["input"] == 1]['y'], dtf[dtf["input"] == 1]['z'], c="red")
            ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], yticklabels=[], zticklabels=[])
            if annotate is True:
                for label, row in dtf[["x", "y", "z"]].iterrows():
                    x, y, z = row
                    ax.text(x, y, z, s=label)

        plt.show()

    except Exception as e:
        print("--- got error ---")
        print(e)
        # word = str(e).split("'")[1]
        # print("maybe you are looking for ... ")
        # print([k for k in list(nlp.vocab.keys()) if 1 - nltk.jaccard_distance(set(word), set(k)) > 0.7])
