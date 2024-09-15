import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import optuna
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Step 1: Load the dataset directly
df = pd.read_csv('combined.csv')  # Adjust with your actual file path


# Step 2: Preprocess the data
def preprocess_data(df):
    X = df['text']  # Use 'text' column for the article content
    y = df['label']  # Use 'label' column for the target (fake/real)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Step 3: Preprocess the dataset
X_train, X_test, y_train, y_test = preprocess_data(df)

# Step 4: Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# To track error rates for plotting
errors = []


# Step 5: Model training and hyperparameter optimization function
def objective(trial):
    mlflow.start_run()

    # Hyperparameters suggested by Optuna
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)

    # Random Forest model with suggested hyperparameters
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        ))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate error (1 - accuracy)
    error = 1 - accuracy
    errors.append(error)  # Track error for plotting

    # Logging results with MLflow
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('min_samples_split', min_samples_split)
    mlflow.log_metric('accuracy', accuracy)

    mlflow.sklearn.log_model(pipeline, 'model')
    mlflow.end_run()

    return accuracy


# Step 6: Suppress Optuna's logging output and set a single TQDM progress bar
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress unnecessary Optuna logs

n_trials = 10  # Define the number of trials you want to run

# Wrapping Optuna's `optimize` function inside a single tqdm progress bar
with tqdm(total=n_trials, desc="Optimization progress", unit="trial") as pbar:
    def objective_with_progress(trial):
        accuracy = objective(trial)  # Run the original objective function
        pbar.update(1)  # Update the progress bar after each trial
        return accuracy


    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_progress, n_trials=n_trials)

# Step 7: Best parameters and accuracy
print(f'Best trial: {study.best_trial.params}')
print(f'Best accuracy: {study.best_value}')

# Step 8: Plotting the error across trials
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='b')
plt.title('Error over Trials')
plt.xlabel('Trial Number')
plt.ylabel('Error (1 - Accuracy)')
plt.grid(True)
plt.show()
