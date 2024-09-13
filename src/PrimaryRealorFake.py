# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


if __name__ == '__main__':
    # # Importing dataset
    # combined = pd.read_csv("first_1000_rows_location_lat_long.csv")
    # # Checking dataset
    # print(f"first 10 rows of combined dataset:\n{combined.head(10)}")

    # Using the full combined dataset before extracting locations
    full_combined = pd.read_csv("Combined.csv")

    # Taking a random sample of 1000 rows
    combined = full_combined.sample(n=1000, random_state=1)

    # Removing unwanted columns
    combined.drop(["subject", "date", "title"], axis=1, inplace=True)
    print(f"combined dataset after dropping unwanted columns:\n {combined}")

    # Check for missing values in the text column
    print(f"Missing values in text column before handling: {combined['text'].isnull().sum()}")

    # Check for missing values in the text column and replacing them
    combined['text'].fillna('Missing text', inplace=True)

    # # Check for missing values in the location column
    # print(f"Missing values in location column before handling: {combined['location'].isnull().sum()}")
    #
    # # drop rows with missing values
    # combined['location'].dropna()

    # Test train data split
    X_train, X_test, y_train, y_test = train_test_split(combined.text, combined.label, test_size=0.2, random_state=1)

    # Different types of classifier model implementation
    cv = CountVectorizer()
    cv_1 = cv.fit_transform(X_train)

    rf = RandomForestClassifier()
    rf.fit(cv_1, y_train)
    y_pred_randomForest = rf.predict(cv.transform(X_test))

    # Classification report testing
    print(classification_report(y_test, y_pred_randomForest))

    print(f"{confusion_matrix(y_test, y_pred_randomForest)}\n")

    print(accuracy_score(y_test, y_pred_randomForest))
