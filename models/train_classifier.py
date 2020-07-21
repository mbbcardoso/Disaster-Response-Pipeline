import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """ Reads database table into dataframe
    
    INPUT
    database_filepath - path for the database
    
    OUTPUT
    X - features dataframe (messages)
    Y - target dataframe (categories)
    category_names - list of category names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('LabeledMessages', con = engine)
    
    category_names = df.columns[4:]
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, category_names


def tokenize(text):
    """ Tokenizes text
    
    INPUT
    text - text to tokenize
    
    OUTPUT
    clean_tokens - tokenized text
    """
    
    # normalizing
    norm_text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenizing
    tokens = word_tokenize(norm_text)
    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """ Builds model using pipeline and grid search cross-validation
    
    INPUT
    None
    
    OUTPUT
    cv - model trained using pipeline and optimized with GridSearchCV
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    parameters =  {'clf__estimator__min_samples_leaf': [2, 4],
    'clf__estimator__min_samples_split': [5, 10]}

    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates classification performance for each category
    
    INPUT
    model - the classification model to use for predictions
    X_test - the testing dataset with features
    Y_test - the testing dataset with targets
    category_names - list of category names
    """
    # predict 
    y_pred = model.predict(X_test)
    # report classification performance
    print(classification_report(y_pred, Y_test.values, target_names=category_names))


def save_model(model, model_filepath):
    """ Saves model in pickle file
    
    INPUT
    model - model to be saved
    model_filepath - path for pickle
    
    OUTPUT
    None
    """
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)


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