import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.base import ClassifierMixin, BaseEstimator


class LSTMClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, encode_y: bool = False, model=None, epochs: int = 10, batch_size: int = 256):
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = model
        self.encode_y = encode_y
        self.dic_y_mapping = None

    @property
    def model(self):
        return self._model

    def _compile_model(self, X_train, y_train, embeddings):
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

        # compile
        model = models.Model(x_in, y_out)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        self._model = model

    @staticmethod
    def _plot_keras_training(training: 'keras training history'):
        """
        Plot loss and metrics of keras training.
        """

        metrics = [k for k in training.history.keys() if ('loss' not in k) and ('val' not in k)]
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

        # training
        ax[0].set(title='Training')
        ax11 = ax[0].twinx()
        ax[0].plot(training.history['loss'], color='black')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax11.plot(training.history[metric], label=metric)
        ax11.set_ylabel("Score", color='steelblue')
        ax11.legend()

        # validation
        ax[1].set(title='Validation')
        ax22 = ax[1].twinx()
        ax[1].plot(training.history['val_loss'], color='black')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax22.plot(training.history['val_' + metric], label=metric)
        ax22.set_ylabel('Score', color='steelblue')
        plt.show()

    def train(self, X, y):
        verbose = 1
        training = self.model.fit(x=X, y=y,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  shuffle=True,
                                  verbose=verbose,
                                  validation_split=0.2)

        if self.epochs > 1:
            self._plot_keras_training(training)

    def fit(self, X, y):

        (X_train, weights), y_train = X, y
        if self.encode_y is True:
            self.dic_y_mapping = {n: label for n, label in enumerate(np.unique(y))}
            inverse_dic = {v: k for k, v in self.dic_y_mapping.items()}
            y_train = np.array([inverse_dic[i] for i in y])

        print(self.dic_y_mapping)

        if self.model is None:
            self._compile_model(X_train, y_train, weights)

        self.train(X_train, y_train)

        return self

    def predict(self, X):
        predicted_prob = self.model.predict(X)
        predicted = [self.dic_y_mapping[np.argmax(pred)] if self.encode_y else np.argmax(pred) for pred in predicted_prob]

        return predicted_prob, predicted
