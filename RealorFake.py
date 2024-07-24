import mlflow
import optuna
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def objective(trial):
    with mlflow.start_run(nested=True):
        combined = pd.read_csv("first_1000_rows_location_lat_long.csv", encoding='latin1')
        combined.drop(["subject", "date", "title"], axis=1, inplace=True)
        combined['text'].fillna('Missing text', inplace=True)
        combined['location'].dropna()

        X_train, X_test, y_train, y_test = train_test_split(combined.text, combined.label, test_size=0.2,
                                                            random_state=1)

        cv = CountVectorizer()
        cv_1 = cv.fit_transform(X_train)

        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)

        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(cv_1, y_train)
        y_pred = rf.predict(cv.transform(X_test))

        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
