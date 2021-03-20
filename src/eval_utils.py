import os
import re
import keras
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.preprocessing as sp
import keras.preprocessing as kp
import keras.backend as kb

from src.classifier_model import LSTMClassifierModel
from src.text_vectorizer import TextVectorizer


def evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15, 5)):
    """
    Evaluates a model performance.
    :parameter
        :param y_test: array
        :param predicted: array
        :param predicted_prob: array
        :param figsize: tuple - plot setting
    """

    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    # Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    # Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    # Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    # Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()


def explain_observation(observation_dict: dict, nlp_dict: dict, top=5, figsize=(25, 5)) -> str:
    # check true value and predicted value
    print("True:", observation_dict['label'],
          "--> Pred:", observation_dict['predicted'],
          "| Prob:", round(np.max(observation_dict['prediction_prob']), 2))

    # preprocess input
    lst_corpus = []
    for s in [re.sub(r'[^\w\s]', '', observation_dict['text'].lower().strip())]:
        lst_words = s.split()
        lst_grams = [' '.join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)

    lst_corpus = list(nlp_dict['bigrams_detector'][lst_corpus])
    lst_corpus = list(nlp_dict['trigrams_detector'][lst_corpus])

    X_instance = kp.sequence.pad_sequences(nlp_dict['tokenizer'].texts_to_sequences(lst_corpus),
                                           maxlen=int(nlp_dict['model'].input.shape[1]), padding="post",
                                           truncating="post")

    # get attention weights
    layer = [layer for layer in nlp_dict['model'].layers if "attention" in layer.name][0]
    func = kb.function([nlp_dict['model'].input], [layer.output])
    weights = func(X_instance)[0]
    weights = np.mean(weights, axis=2).flatten()

    # rescale weights, remove null vector, map word-weight
    weights = sp.MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(weights).reshape(-1, 1)).reshape(-1)
    weights = [weights[n] for n, idx in enumerate(X_instance[0]) if idx != 0]
    dic_word_weight = {word: weights[n] for n, word in enumerate(lst_corpus[0]) if word in nlp_dict['tokenizer'].word_index.keys()}

    # plot
    if len(dic_word_weight) > 0:
        dtf = pd.DataFrame.from_dict(dic_word_weight, orient='index', columns=["score"])
        dtf.sort_values(by="score", ascending=True).tail(top).plot(kind="barh", legend=False, figsize=figsize).grid(
            axis='x')
        plt.show()
    else:
        print("--- No word recognized ---")

    # return html visualization (yellow:255,215,0 | blue:100,149,237)
    text = []
    for word in lst_corpus[0]:
        weight = dic_word_weight.get(word)
        if weight is not None:
            text.append(
                '<b><span style="background-color:rgba(100,149,237,' + str(weight) + ');">' + word + '</span></b>')
        else:
            text.append(word)
    text = ' '.join(text)

    try:
        from IPython.core.display import display, HTML
        display(HTML(text))
    except:
        pass
    return text


def export_all(classifier: LSTMClassifierModel, vectorizer: TextVectorizer, folder_out_path: str):
    def handle_exist(path, file_name):
        if os.path.exists(os.path.join(path, file_name)):
            print(f'{path} already contains a file named "{file_name}"!')
            return True
        return False

    os.makedirs(folder_out_path, exist_ok=True)

    # save keras model
    if not handle_exist(folder_out_path, 'hate_classifier.h5'):
        model_out_path = os.path.join(folder_out_path, 'hate_classifier.h5')
        classifier.model.save(model_out_path)

    # save label mapping
    if not handle_exist(folder_out_path, 'y_labels.pickle'):
        y_labels_out_path = os.path.join(folder_out_path, 'y_labels.pickle')
        with open(y_labels_out_path, 'wb') as handle:
            pickle.dump(classifier.dic_y_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save tokenizer
    if not handle_exist(folder_out_path, 'tokenizer.pickle'):
        tokenizer_out_path = os.path.join(folder_out_path, 'tokenizer.pickle')
        with open(tokenizer_out_path, 'wb') as handle:
            pickle.dump(vectorizer.fitted_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save ngram detectors
    if not handle_exist(folder_out_path, 'ngrams_detectors.pickle'):
        detectors_out_path = os.path.join(folder_out_path, 'ngrams_detectors.pickle')
        with open(detectors_out_path, 'wb') as handle:
            pickle.dump(vectorizer.ngrams_detector_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
