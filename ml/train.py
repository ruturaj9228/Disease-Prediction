import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
import joblib

def train_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'outputs')
    
    print("Loading preprocessed training data...")
    X_train = pd.read_csv(os.path.join(out_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(out_dir, 'y_train.csv')).values.ravel()
    
    print("Initializing models...")
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': BernoulliNB(),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    print("Training models...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        filename = f"model_{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, os.path.join(out_dir, filename))
        print(f"  Saved {name} to {filename}")

    print("All models trained and saved!")

if __name__ == "__main__":
    train_models()
