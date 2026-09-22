import os
import pandas as pd
import numpy as np

def run_diagnostics():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    
    # 1. Load raw datasets
    train_df = pd.read_csv(os.path.join(raw_dir, 'Training.csv'))
    test_df = pd.read_csv(os.path.join(raw_dir, 'Testing.csv'))
    
    # Clean up empty columns just like in preprocess
    if 'Unnamed: 133' in train_df.columns:
        train_df = train_df.drop('Unnamed: 133', axis=1)
    if 'Unnamed: 133' in test_df.columns:
        test_df = test_df.drop('Unnamed: 133', axis=1)
        
    train_df = train_df.dropna(subset=['prognosis'])
    test_df = test_df.dropna(subset=['prognosis'])
    
    # Check exact row overlap
    # We can do this by merging
    merged = test_df.merge(train_df, how='inner')
    exact_duplicates = len(merged)
    pct_exact = (exact_duplicates / len(test_df)) * 100
    print(f"--- 1. Row Overlap Check ---")
    print(f"Testing set size: {len(test_df)}")
    print(f"Exact row duplicates in Training set: {exact_duplicates} ({pct_exact:.2f}%)")
    
    # Symptom vector only check
    X_train = train_df.drop('prognosis', axis=1)
    X_test = test_df.drop('prognosis', axis=1)
    
    merged_symptoms = X_test.merge(X_train, how='inner')
    symptom_duplicates = len(merged_symptoms.drop_duplicates())
    # Actually, to find how many test rows are in train (ignoring prognosis)
    overlap_idx = X_test.apply(tuple, 1).isin(X_train.apply(tuple, 1))
    print(f"Test rows with identical symptom vector in Training set: {overlap_idx.sum()} ({(overlap_idx.sum() / len(test_df))*100:.2f}%)")
    print()
    
    # 2. Symptom-pattern uniqueness per disease
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"--- 2. Symptom-Pattern Uniqueness per Disease ---")
    disease_patterns = combined_df.groupby('prognosis').apply(lambda x: len(x.drop('prognosis', axis=1).drop_duplicates())).reset_index(name='unique_patterns')
    
    print(disease_patterns.to_string(index=False))
    print(f"Average unique patterns per disease: {disease_patterns['unique_patterns'].mean():.2f}")
    
    print("\n--- 3. Confusion Matrix Review (from test set predictions) ---")
    # To get exact confusions, let's just do a quick predict for RF and DT
    out_dir = os.path.join(base_dir, 'outputs')
    import joblib
    
    X_test_proc = pd.read_csv(os.path.join(out_dir, 'X_test.csv'))
    y_test_proc = pd.read_csv(os.path.join(out_dir, 'y_test.csv')).values.ravel()
    le = joblib.load(os.path.join(out_dir, 'label_encoder.joblib'))
    
    for model_name, prefix in [("Random Forest", "random_forest"), ("Decision Tree", "decision_tree")]:
        model = joblib.load(os.path.join(out_dir, f'model_{prefix}.joblib'))
        preds = model.predict(X_test_proc)
        
        errors = (preds != y_test_proc)
        if np.any(errors):
            print(f"\n{model_name} Errors:")
            err_actual = le.inverse_transform(y_test_proc[errors])
            err_pred = le.inverse_transform(preds[errors])
            for a, p in zip(err_actual, err_pred):
                print(f"  Actual: {a} -> Predicted: {p}")
        else:
            print(f"\n{model_name} Errors: None")

if __name__ == "__main__":
    run_diagnostics()
