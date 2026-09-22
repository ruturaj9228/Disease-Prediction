# Model Comparison Results

Here are the results of the rigorous evaluation for the four models on the Kaggle dataset.

| Model         | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Mean CV F1 (5-fold) |
|---------------|---------------|----------------------|-------------------|---------------|---------------------|
| SVM           | 1.0           | 1.0                  | 1.0               | 1.0           | 1.0                 |
| Naive Bayes   | 1.0           | 1.0                  | 1.0               | 1.0           | 1.0                 |
| Random Forest | 0.976         | 0.988                | 0.988             | 0.984         | 1.0                 |
| Decision Tree | 0.976         | 0.988                | 0.988             | 0.984         | 1.0                 |

## Key Takeaways

1. **Perfect Score for SVM & Naive Bayes**: Both Linear SVM and Naive Bayes achieved a perfect 1.0 F1 score on the test set and during cross-validation. This is common for this specific dataset since the symptoms are perfectly correlated binary indicators for the diseases.
2. **Programmatic Selection**: As implemented in `ml/predict.py`, the system programmatically reads the CSV output of these results and selects the best model. In this case, **SVM** will be used as the production model for inference because it achieved the top score.
3. **Artifacts Generated**: The `ml/outputs/` directory now contains the trained `.joblib` models, label encoder, symptom vocabulary, the raw `model_comparison.csv`, and confusion matrices `.png` for all four models for your final report.

Are you ready to approve this so we can move to **Phase 2: FastAPI Backend**?
