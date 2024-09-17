## Results of Hyperparameter Tuning with Optuna (Trial Numbers: 10 to 50)

This document presents the results of running **Optuna** for hyperparameter tuning on the Random Forest model. The number of trials varies between **10 to 50**:
### Run 1: **10 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 58
  - `max_depth`: 18
  - `min_samples_split`: 6
- **Best Accuracy**: 98.9%
- **Time Taken**: 2.5 minutes


![Screenshot 2024-09-17 at 19 33 29](https://github.com/user-attachments/assets/5a16d931-8ffd-48ac-bec2-eef093a58f1f)

---

### Run 2: **20 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 74
  - `max_depth`: 20
  - `min_samples_split`: 16
- **Best Accuracy**: 99.3%
- **Time Taken**: 6 minutes

![Screenshot 2024-09-17 at 18 44 33](https://github.com/user-attachments/assets/d9b7fd64-c631-4f26-b551-c19342db0115)

---

### Run 3: **30 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 110
  - `max_depth`: 18
  - `min_samples_split`: 14
- **Best Accuracy**: 99.4%
- **Time Taken**: 10 minutes




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
