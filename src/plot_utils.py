import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import manifold


def plot_categories(data: pd.DataFrame):
    fig, ax = plt.subplots()
    fig.suptitle("categories", fontsize=12)

    data = data.replace({'class': {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}})
    data["class"].reset_index().groupby("class").count().sort_values(by="index").plot(kind="barh", legend=False,ax=ax).grid(axis='x')
    plt.show()


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
