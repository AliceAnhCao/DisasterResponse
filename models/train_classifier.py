#Import libraries
import sys
import pandas as pd
pd.set_option('display.max_columns', 100)

import os
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine


def load_data(database_filepath):
    '''
    Function to load data from database into dataframe
    Input: 
        database_filepath: database filename for sqlite database with (.db) file type
        
    Output:
        X: messages (input variable)
        Y: categories of the messages (output variable)
        category_names: category name for y
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    '''
    Function to tokenize text message
    Input:
        text: raw text
    
    Output:
        clean_tokens: tokenized messages
    '''
    url_reg = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detect_url = re.findall(url_reg, text_data)
    for url in detect_url:
        text_data = text_data.replace(url, "urlPlaceHolder")
    
    tokens = word_tokenize(text_data)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model(clf = AdaBoostClassifier()):
    '''
    Function to train the model with cleaned data using classifier algorithms
    INPUT:
        clf: classifier model (the default value is 'AdaBoostClassifier()'
    
    OUTPUT:
        cv = Model pipeline after performing grid search
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(clf))
    ])
    
    parameters = {
        'clf__estimator__learning_rate':[0.5, 1.0],
        'clf__estimator__n_estimators':[10,30]
    
    }
        
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=3) 
    
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Funtion to evaluate classifier model with test data
    
    INPUT:
        model: Trained model
        X_test: Input data- test messages
        Y_test: categories for test messages
        category_names: category name for y
        
    OUTPUT:
        evaluation metrics for model
    '''
    y_predict = model.predict(X_test)
    
    # classification_report with test data
    print(classification_report(y_test.values,       y_predict, target_names=y.columns.values))
    
    
    


def save_model(model, model_filepath):
    '''
    Function to save final model
    INPUT:
        model: ML model
        model_filepath: location to save the model
    '''
    # save model in pickle file
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()