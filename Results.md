## Results of Hyperparameter Tuning with Optuna (Trial Numbers: 10 to 50)

This document presents the results of running **Optuna** for hyperparameter tuning on the Random Forest model. The number of trials varies between **10 to 50**:
### Run 1: **10 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 74
  - `max_depth`: 20
  - `min_samples_split`: 16
- **Best Accuracy**: 99.3%
- **Time Taken**: 5 minutes

![10 Trials - Accuracy over Time](path_to_graph_10_trials.png)

---

### Run 2: **20 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 88
  - `max_depth`: 15
  - `min_samples_split`: 12
- **Best Accuracy**: 99.2%
- **Time Taken**: 7 minutes

![20 Trials - Accuracy over Time](path_to_graph_20_trials.png)

---

### Run 3: **30 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 110
  - `max_depth`: 18
  - `min_samples_split`: 14
- **Best Accuracy**: 99.4%
- **Time Taken**: 10 minutes

![30 Trials - Accuracy over Time](path_to_graph_30_trials.png)

---

### Run 4: **40 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 105
  - `max_depth`: 19
  - `min_samples_split`: 16
- **Best Accuracy**: 99.5%
- **Time Taken**: 12 minutes

![40 Trials - Accuracy over Time](path_to_graph_40_trials.png)

---

### Run 5: **50 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 130
  - `max_depth`: 22
  - `min_samples_split`: 10
- **Best Accuracy**: 99.6%
- **Time Taken**: 15 minutes

![50 Trials - Accuracy over Time](path_to_graph_50_trials.png)
