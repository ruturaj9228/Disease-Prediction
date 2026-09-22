import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    out_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(raw_dir, 'Training.csv'))
    test_df = pd.read_csv(os.path.join(raw_dir, 'Testing.csv'))
    
    # Drop trailing empty columns if present
    if 'Unnamed: 133' in train_df.columns:
        train_df = train_df.drop('Unnamed: 133', axis=1)
    if 'Unnamed: 133' in test_df.columns:
        test_df = test_df.drop('Unnamed: 133', axis=1)
        
    print("Handling nulls/duplicates...")
    # Drop any rows with missing targets
    train_df = train_df.dropna(subset=['prognosis'])
    test_df = test_df.dropna(subset=['prognosis'])
    
    # Fill any null symptoms with 0
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Separate features and target
    X_train = train_df.drop('prognosis', axis=1)
    y_train_raw = train_df['prognosis']
    
    X_test = test_df.drop('prognosis', axis=1)
    y_test_raw = test_df['prognosis']
    
    # Ensure column order matches
    X_test = X_test[X_train.columns]
    
    print("Encoding labels...")
    le = LabelEncoder()
    # Fit on both train and test to ensure all labels are captured
    le.fit(pd.concat([y_train_raw, y_test_raw]))
    
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    # Save preprocessed features and labels
    print("Saving preprocessed data and artifacts...")
    X_train.to_csv(os.path.join(out_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(out_dir, 'X_test.csv'), index=False)
    
    pd.DataFrame(y_train, columns=['prognosis']).to_csv(os.path.join(out_dir, 'y_train.csv'), index=False)
    pd.DataFrame(y_test, columns=['prognosis']).to_csv(os.path.join(out_dir, 'y_test.csv'), index=False)
    
    # Save the label encoder
    joblib.dump(le, os.path.join(out_dir, 'label_encoder.joblib'))
    
    # Save the symptom vocabulary (feature names)
    symptoms = list(X_train.columns)
    joblib.dump(symptoms, os.path.join(out_dir, 'symptoms_vocab.joblib'))
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()
