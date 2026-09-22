import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

def evaluate_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'outputs')
    
    print("Loading datasets and models...")
    X_train = pd.read_csv(os.path.join(out_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(out_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(out_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(out_dir, 'y_test.csv')).values.ravel()
    
    label_encoder = joblib.load(os.path.join(out_dir, 'label_encoder.joblib'))
    classes = label_encoder.classes_
    
    model_names = ['Decision Tree', 'Random Forest', 'Naive Bayes', 'SVM']
    
    results = []
    
    for name in model_names:
        print(f"\nEvaluating {name}...")
        filename = f"model_{name.replace(' ', '_').lower()}.joblib"
        model = joblib.load(os.path.join(out_dir, filename))
        
        # 5-fold CV on training data for macro F1
        print("  Running 5-fold Cross-Validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        mean_cv_f1 = np.mean(cv_scores)
        
        # Predict on holdout test set
        print("  Evaluating on test set...")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results.append({
            'Model': name,
            'Test Accuracy': acc,
            'Test Macro Precision': prec,
            'Test Macro Recall': rec,
            'Test Macro F1': f1,
            'Mean CV F1 (5-fold)': mean_cv_f1
        })
        
        # Confusion matrix
        print("  Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"))
        plt.close()
        
    # Save comparison table
    results_df = pd.DataFrame(results)
    # Sort by Test Macro F1
    results_df = results_df.sort_values(by='Test Macro F1', ascending=False)
    
    csv_path = os.path.join(out_dir, 'model_comparison.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nEvaluation complete. Results saved to {csv_path}")
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
