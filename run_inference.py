import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from src.text_preprocessor import TextPreprocessor
from src.text_vectorizer import TextVectorizer
from src.classifier_model import LSTMClassifierModel

from typing import Tuple, List, Union


def load_text_processing_pipeline(folder_path):
    with open(os.path.join(folder_path, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(os.path.join(folder_path, 'ngrams_detectors.pickle'), 'rb') as handle:
        ngrams_detector_list = pickle.load(handle)

    # initialize sklearn preprocessing pipeline
    nltk_stop_words = pd.read_csv('../data/nltk_stop_words.csv', header=None).iloc[0].tolist()
    pipeline = Pipeline(steps=[
        ('normalize', TextPreprocessor(custom_stop_words=nltk_stop_words + ['rt', 'amp', 'ai', 'nt', 'na'])),
        ('features', TextVectorizer(fitted_tokenizer=tokenizer, ngrams_detector_list=ngrams_detector_list)),
    ])

    return pipeline


def load_keras_model(folder_path):
    loaded_model = keras.models.load_model(os.path.join(folder_path, 'hate_classifier.h5'))

    with open(os.path.join(folder_path, 'y_labels.pickle'), 'rb') as handle:
        y_label_mapping = pickle.load(handle)

    classifier = LSTMClassifierModel(model=loaded_model, classes_mapping=y_label_mapping)

    return classifier


def run_inference(input_str: str) -> Tuple[np.array, List[Union[int, str]]]:
    preprocess_pipeline = load_text_processing_pipeline('saved_model/hate_classifier_v2')
    model = load_keras_model('saved_model/hate_classifier_v2')

    # preprocess test vector
    X_test, _ = preprocess_pipeline.transform(pd.Series(user_input))

    # run inference
    inference = model.predict(X_test)

    return inference


if __name__ == '__main__':

    # load data
    while True:
        user_input = input(r'Type in a sentence (or \q to quit): ')
        if user_input == r'\q':
            break

        predicted_prob, predicted_class = run_inference(user_input)
        if predicted_class[0] == 'offensive_language' and predicted_prob.max() < 0.8:
            predicted_class[0] = 'neither (man)'

        print(f'\nclass = "{predicted_class[0]}" ({predicted_prob.max()})')
