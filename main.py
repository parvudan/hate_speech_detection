import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.text_preprocessor import TextPreprocessor
from src.text_vectorizer import TextVectorizer
from src.classifier_model import LSTMClassifierModel

from src.eval_utils import evaluate_multi_classif, explain_observation
from src.plot_utils import plot_categories, plot_w2v


if __name__ == '__main__':
    # load data
    df = pd.read_csv('data/t_davidson_hate_speech_and_offensive_language.csv', index_col=0)
    plot_categories(df)

    sentences = df['tweet']
    mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    y = df[['class']]
    y = y.replace({'class': mapping})['class']

    # split into train and test (the val set will be taken from `sentences_train` during training)
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state=1000)

    # initialize sklearn preprocessing pipeline
    nltk_stop_words = pd.read_csv('data/nltk_stop_words.csv', header=None).iloc[0].tolist()
    preprocess_pipeline = Pipeline(steps=[
        ('normalize', TextPreprocessor(custom_stop_words=nltk_stop_words + ['rt', 'amp', 'ai', 'nt', 'na'])),
        ('features', TextVectorizer()),
    ])

    # initialize sklearn classifier pipeline
    classifier_pipeline = Pipeline(steps=[
        ('preprocessing', preprocess_pipeline),
        ('classifier', LSTMClassifierModel(encode_y=True, epochs=15, batch_size=128))
    ])

    # train model
    classifier_pipeline.fit(sentences_train, y_train)

    # plot word clouds
    nlp = classifier_pipeline['preprocessing']['features'].nlp
    plot_w2v(lst_words=['nigga'], nlp=nlp, plot_type="3d", top=20, annotate=True, figsize=(10, 5))
    plot_w2v(lst_words=None, nlp=nlp.wv, plot_type="2d", annotate=False, figsize=(20, 10))

    # preprocess test vector
    X_test, _ = preprocess_pipeline.fit_transform(sentences_test)

    # run inference
    predicted_prob, predicted = classifier_pipeline['classifier'].predict(X_test)

    evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15, 5))

    observation_dict = {
        'text': sentences_test.iloc[0],
        'label': y_test.iloc[0],
        'predicted': predicted[0],
        'prediction_prob': predicted_prob[0]
    }

    nlp_dict = {
        'bigrams_detector': preprocess_pipeline['features'].ngrams_detector_list[0],
        'trigrams_detector': preprocess_pipeline['features'].ngrams_detector_list[1],
        'tokenizer': preprocess_pipeline['features'].fitted_tokenizer,
        'model': classifier_pipeline['classifier'].model
    }

    explain_observation(observation_dict, nlp_dict)

    print('Done!')
