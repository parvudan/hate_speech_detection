import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.text_preprocessor import TextPreprocessor
from src.text_vectorizer import TextVectorizer
from src.classifier_model import LSTMClassifierModel
from model_lstm.model_arch import evaluate_multi_classif


if __name__ == '__main__':
    # load data
    df = pd.read_csv('../hate_speech/data/t_davidson_hate_speech_and_offensive_language.csv')

    sentences = df['tweet']
    mapping = {0: 'hate_speech', 1: 'offensive_language', 2: 'neither'}
    y = df[['class']]
    y = y.replace({'class': mapping})['class']

    # split into train and test (the val set will be taken from `sentences_train` during training)
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state=1000)

    # initialize sklearn preprocessing pipeline
    preprocess_pipeline = Pipeline(steps=[
        ('normalize', TextPreprocessor(n_jobs=0)),
        ('features', TextVectorizer()),
    ])

    # initialize sklearn classifier pipeline
    classifier_pipeline = Pipeline(steps=[
        ('preprocessing', preprocess_pipeline),
        ('classifier', LSTMClassifierModel(encode_y=True))
    ])

    # train model
    classifier_pipeline.fit(sentences_train, y_train)

    # preprocess test vector
    X_test, _ = preprocess_pipeline.fit_transform(sentences_test)

    # run inference
    predicted_prob, predicted = classifier_pipeline['classifier'].predict(X_test)

    evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15, 5))

    print('Done!')
