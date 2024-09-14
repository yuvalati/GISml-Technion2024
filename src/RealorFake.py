import mlflow
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# Function to visualize the loss and accuracy over trials
def plot_metrics(metrics):
    plt.figure(figsize=(10, 6))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(metrics['accuracy'], label="Accuracy")
    plt.title('Model Accuracy Over Trials')
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(metrics['loss'], label="Log Loss", color='red')
    plt.title('Model Loss Over Trials')
    plt.xlabel('Trial')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Ask the user once which dataset to use (small or full)
dataset_choice = input("Enter 'small' for 1000 rows or 'full' for full dataset: ").strip().lower()


# The objective function for Optuna
def objective(trial):
    with mlflow.start_run(nested=True):
        # Use dataset choice to select between the small or full dataset
        if dataset_choice == 'small':
            dataset_path = "1000_rows_with_lat_long_2.csv"  # Small dataset (1000 rows)
        else:
            dataset_path = "Combined.csv"  # Full dataset (~40,000 rows)

        # Importing dataset
        combined = pd.read_csv(dataset_path, encoding='latin1')

        # Removing unwanted columns
        combined.drop(["subject", "date", "title"], axis=1, inplace=True)

        # Check for missing values in the text column and replace them
        combined['text'].fillna('Missing text', inplace=True)

        # Test train data split
        X_train, X_test, y_train, y_test = train_test_split(combined.text, combined.label, test_size=0.2,
                                                            random_state=1)

        # Transform text data to numeric data
        cv = CountVectorizer()
        cv_1 = cv.fit_transform(X_train)

        # Suggest hyperparameters for optimization
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32)

        # Classifier model implementation
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rf.fit(cv_1, y_train)
        y_pred_randomForest = rf.predict(cv.transform(X_test))
        y_pred_proba = rf.predict_proba(cv.transform(X_test))

        # Calculate accuracy and log loss
        accuracy = accuracy_score(y_test, y_pred_randomForest)
        loss = log_loss(y_test, y_pred_proba)

        # Log parameters and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("loss", loss)

        return accuracy, loss


# Tracking accuracy and loss for each trial
accuracy_over_trials = []
loss_over_trials = []

# Initialize Optuna study
study = optuna.create_study(direction='maximize', study_name='Real or Fake')

# Add a tqdm progress bar to visualize the process
n_trials = 10  # You can set this to any number of trials you want
with tqdm(total=n_trials) as pbar:
    def wrapped_objective(trial):
        accuracy, loss = objective(trial)
        accuracy_over_trials.append(accuracy)
        loss_over_trials.append(loss)
        pbar.update(1)  # Update the progress bar after each trial
        return accuracy


    # Run Optuna optimization
    study.optimize(wrapped_objective, n_trials=n_trials)

# Visualize accuracy and loss over trials
plot_metrics({"accuracy": accuracy_over_trials, "loss": loss_over_trials})

# Display the best trial
print("\nBest trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Display final classification report and confusion matrix for the best trial
if dataset_choice == 'small':
    dataset_path = "1000_rows_with_lat_long_2.csv"
else:
    dataset_path = "Combined.csv"

combined = pd.read_csv(dataset_path, encoding='latin1')
combined.drop(["subject", "date", "title"], axis=1, inplace=True)
combined['text'].fillna('Missing text', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(combined.text, combined.label, test_size=0.2, random_state=1)

cv = CountVectorizer()
cv_1 = cv.fit_transform(X_train)

# Use the best parameters from the best trial
best_rf = RandomForestClassifier(n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'])
best_rf.fit(cv_1, y_train)
y_pred_best = best_rf.predict(cv.transform(X_test))

# Assign predictions to a new column in the dataframe
combined.loc[X_test.index, 'predictions'] = y_pred_best

# Save the CSV with predictions
output_csv_path = dataset_choice + "_with_predictions.csv"
combined.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")

# Print final classification report and confusion matrix
print("\nFinal classification report for the best trial:")
print(classification_report(y_test, y_pred_best))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_best))
