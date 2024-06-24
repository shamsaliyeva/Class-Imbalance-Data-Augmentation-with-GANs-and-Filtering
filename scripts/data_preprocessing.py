# scripts/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_variable='Churn'):
    """
    Preprocesses the dataset by splitting it into features and target variable,
    and performing train-test split.

    Parameters:
    - df: pandas DataFrame, the dataset
    - target_variable: str, the name of the target variable column

    Returns:
    - X_train: pandas DataFrame, features for training
    - X_test: pandas DataFrame, features for testing
    - y_train: pandas Series, target variable for training
    - y_test: pandas Series, target variable for testing
    """
    X = df.drop(target_variable, axis=1)  # Features (all columns except the target variable)
    y = df[target_variable]  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def encode_categorical(X):
    """
    Encode categorical variables in the dataset.
    
    Parameters:
    - X: pandas DataFrame, features for encoding
    
    Returns:
    - X: pandas DataFrame, data with encoded categorical variables
    - encoded_labels: dict, mapping of original labels to encoded values for each column
    """
    
    encoded_labels = {}

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoded_labels[col] = {num: label for num, label in zip(range(len(le.classes_)), le.classes_)}

    return X, encoded_labels
