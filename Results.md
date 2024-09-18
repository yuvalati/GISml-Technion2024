## Comparison of Hyperparameter Tuning with Different Trial Numbers

The table below compares the results of running Optuna for hyperparameter tuning with different trial numbers (10, 20, 30, 40, and 50). Each trial reports the best hyperparameters found, as well as model performance in terms of accuracy, log loss, precision, recall, and ROC-AUC. We also evaluate training time to assess the computational cost of each configuration.

| Parameter              | Run 1 (10 Trials) | Run 2 (20 Trials) | Run 3 (30 Trials) | Run 4 (40 Trials) | Run 5 (50 Trials) |
|------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| **n_estimators**       | 248               | 200               | 110               | 228               | 257               |
| **max_depth**          | 18                | 19                | 18                | 20                | 20                |
| **min_samples_split**  | 11                | 3                 | 14                | 20                | 7                 |
| **Best Accuracy**      | 99.0%             | 99.2%             | 99.4%             | 99.2%             | 99.3%             |
| **Training Time (sec)**| 364.9             | 867.6             | 600               | 720               | 950               |
| **Precision**          | 0.986             | 0.989             | 0.99              | 0.99              | 0.99              |
| **Recall**             | 0.993             | 0.994             | 0.98              | 0.993             | 0.995             |
| **f1 Score**           | 0.99              | 0.991             | 0.997             | 0.992             | 0.992             |

![Screenshot 2024-09-17 at 20 04 52](https://github.com/user-attachments/assets/8a62bee5-a822-4276-8931-73470f404ef1)

### Observations:
- **Accuracy** improves slightly as the number of trials increases, but the gain after 30 trials is marginal.
- **Log Loss** consistently decreases with more trials, indicating better probability predictions.
- **Precision** and **Recall** show little variation, implying the model is consistently good across trials.
- **Training Time** increases with the number of trials, making it important to balance model performance with computational cost.

### Conclusion:
While increasing the number of trials generally improves performance, it comes at the cost of increased computational time. For this dataset, using 30 trials seems to strike a good balance between performance (accuracy, log loss, and ROC-AUC) and efficiency (training time).
