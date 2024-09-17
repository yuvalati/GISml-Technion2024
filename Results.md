## Results of Hyperparameter Tuning with Optuna (Trial Numbers: 10 to 50)

This document presents the results of running **Optuna** for hyperparameter tuning on the Random Forest model. The number of trials varies between **10 to 50**:
### Run 1: **10 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 58
  - `max_depth`: 18
  - `min_samples_split`: 6
- **Best Accuracy**: 98.9%
- **Time Taken**: 2.5 minutes

![plot1-10trials](https://github.com/user-attachments/assets/00c30391-9363-4432-8595-0a168457da6c)


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

![plot2-30trials](https://github.com/user-attachments/assets/945346b1-4016-406a-a73d-bdd0d58dff3f)

---

### Run 4: **40 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 269
  - `max_depth`: 19
  - `min_samples_split`: 13
- **Best Accuracy**: 99.2%
- **Time Taken**: 12 minutes

![Screenshot 2024-09-17 at 20 22 35](https://github.com/user-attachments/assets/57f5848c-d024-4e53-a08a-be749631bd66)


---

### Run 5: **50 Trials**

- **Best Hyperparameters**:
  - `n_estimators`: 190
  - `max_depth`: 19
  - `min_samples_split`: 13
- **Best Accuracy**: 99.2%
- **Time Taken**: 15 minutes

![Screenshot 2024-09-17 at 20 04 52](https://github.com/user-attachments/assets/8a62bee5-a822-4276-8931-73470f404ef1)

