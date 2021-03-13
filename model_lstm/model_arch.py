import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers, models


def model_builder(X_train, y_train, embeddings):
    # Embedding network with Bi-LSTM and Attention layers (for attention explainer)
    x_in = layers.Input(shape=(X_train.shape[1],))

    # embedding
    x = layers.Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], weights=[embeddings],
                         input_length=X_train.shape[1], trainable=False)(x_in)
    # attention

    x = layers.Attention()([x, x])
    # 2 layers of bidirectional lstm
    x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2))(x)

    # final dense layers
    x = layers.Dense(64, activation='relu')(x)
    y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
    # y_out = layers.Dense(y_train.shape[1], activation='softmax')(x)

    # compile
    model = models.Model(x_in, y_out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def utils_plot_keras_training(training):
    """
    Plot loss and metrics of keras training.
    """

    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

    # training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()

    # validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_' + metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()


def fit_dl_classif(X_train, y_train, X_test, encode_y=False, dic_y_mapping=None, model=None, weights=None, epochs=10,
                   batch_size=256):
    """
    Fits a keras classification model.
    :parameter
        :param dic_y_mapping: dict - {0:"A", 1:"B", 2:"C"}. If None it calculates.
        :param X_train: array of sequence
        :param y_train: array of classes
        :param X_test: array of sequence
        :param model: model object - model to fit (before fitting)
        :param weights: array of weights - like embeddings
    :return
        model fitted and predictions
    """

    # encode y
    if encode_y is True:
        dic_y_mapping = {n: label for n, label in enumerate(np.unique(y_train))}
        inverse_dic = {v: k for k, v in dic_y_mapping.items()}
        y_train = np.array([inverse_dic[y] for y in y_train])
    print(dic_y_mapping)

    # model
    if model is None:
        # params
        n_features, embeddings_dim = weights.shape
        max_seq_lenght = X_train.shape[1]

        # neural network
        x_in = layers.Input(shape=(X_train.shape[1],))
        x = layers.Embedding(input_dim=n_features, output_dim=embeddings_dim, weights=[weights],
                             input_length=max_seq_lenght, trainable=False)(x_in)
        x = layers.Attention()([x, x])
        x = layers.Bidirectional(layers.LSTM(units=X_train.shape[1], dropout=0.2))(x)
        x = layers.Dense(units=64, activation='relu')(x)
        y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)

        # compile
        model = models.Model(x_in, y_out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

    # train
    # verbose = 0 if epochs > 1 else 1
    verbose = 1
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose,
                         validation_split=0.3)
    if epochs > 1:
        utils_plot_keras_training(training)

    # test
    predicted_prob = model.predict(X_test)
    predicted = [dic_y_mapping[np.argmax(pred)] if encode_y else np.argmax(pred) for pred in predicted_prob]

    return training.model, predicted_prob, predicted


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
