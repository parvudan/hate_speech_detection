import numpy as np
from keras import layers, models
from sklearn.base import ClassifierMixin, BaseEstimator


class LSTMClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, encode_y=False, batch_size=256, epochs=10):
        self._model = None
        self.batch_size = batch_size
        self.epochs = epochs
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

    def _encode_labels(self, y):
        self.dic_y_mapping = {n: label for n, label in enumerate(np.unique(y))}
        inverse_dic = {v: k for k, v in self.dic_y_mapping.items()}
        y_train = np.array([inverse_dic[i] for i in y])
        return y_train

    def train(self, X, y):
        verbose = 1
        training = self.model.fit(x=X, y=y,
                                  batch_size=self.batch_size, epochs=self.epochs,
                                  shuffle=True, verbose=verbose,
                                  validation_split=0.3)

    def fit(self, X, y):
        (X_train, weights), y_train = X, y

        if self.encode_y is True:
            y_train = self._encode_labels(y)
            print(self.dic_y_mapping)

        if self.model is None:
            self._compile_model(X_train, y_train, weights)

        self.train(X_train, y_train)

        return self

    def predict(self, X):
        predicted_prob = self.model.predict(X)
        predicted = [self.dic_y_mapping[np.argmax(pred)] if self.encode_y else np.argmax(pred) for pred in predicted_prob]

        return predicted_prob, predicted
