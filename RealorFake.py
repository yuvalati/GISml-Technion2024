import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def objective(trial):
    with mlflow.start_run(nested=True):
        # Importing dataset
        combined = pd.read_csv("1000_rows_with_lat_long_2.csv", encoding='latin1')
        # Checking dataset
        # print(f"first 10 rows of combined dataset:\n{combined.head(10)}")

        # Using the full combined dataset before extracting locations
        # full_combined = pd.read_csv("Combined.csv")

        # Taking a random sample of 1000 rows
        # combined = full_combined.sample(n=1000, random_state=1)

        # Check for missing values in the text column
        # print(f"Missing values in text column before handling: {combined['text'].isnull().sum()}\n")

        # Removing unwanted columns
        combined.drop(["subject", "date", "title"], axis=1, inplace=True)
        # print(f"combined dataset after dropping unwanted columns:\n {combined}")

        # Check for missing values in the text column and replacing them
        combined['text'].fillna('Missing text', inplace=True)

        # Test train data split
        X_train, X_test, y_train, y_test = train_test_split(combined.text, combined.label, test_size=0.2,
                                                            random_state=1)

        # transform text data to numeric data
        cv = CountVectorizer()
        cv_1 = cv.fit_transform(X_train)

        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)

        # Classifier model implementation
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(cv_1, y_train)
        y_pred_randomForest = rf.predict(cv.transform(X_test))

        accuracy = accuracy_score(y_test, y_pred_randomForest)

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        # Classification report testing
        print(classification_report(y_test, y_pred_randomForest))

        print(f"{confusion_matrix(y_test, y_pred_randomForest)}\n")

        print(accuracy_score(y_test, y_pred_randomForest))

        return accuracy


study = optuna.create_study(direction='maximize', study_name='Real or Fake')
study.optimize(objective, n_trials=10)

print("\nBest trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
