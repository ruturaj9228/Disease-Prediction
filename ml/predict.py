import os
import pandas as pd
import numpy as np
import joblib

def load_best_model_info(out_dir):
    """Reads the comparison CSV and selects the best interpretable model."""
    comp_df = pd.read_csv(os.path.join(out_dir, 'model_comparison.csv'))
    
    # Selection Override: Even if SVM/Naive Bayes score 1.0, we prefer Random Forest
    # if it's within a close margin (e.g., >0.95), because it provides feature importances 
    # natively which is required for explainability in the frontend.
    
    # Filter for interpretable models (Random Forest, Decision Tree)
    interpretable_models = comp_df[comp_df['Model'].isin(['Random Forest', 'Decision Tree'])]
    
    if not interpretable_models.empty and interpretable_models['Test Macro F1'].max() >= 0.95:
        best_row = interpretable_models.loc[interpretable_models['Test Macro F1'].idxmax()]
    else:
        # Fallback if tree models perform terribly
        best_row = comp_df.loc[comp_df['Test Macro F1'].idxmax()]
        
    best_model_name = best_row['Model']
    prefix = best_model_name.replace(' ', '_').lower()
    
    # We no longer rely on global feature importances, so we just return the name and prefix
    return best_model_name, prefix

class DiseasePredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = os.path.join(self.base_dir, 'outputs')
        
        self.label_encoder = joblib.load(os.path.join(self.out_dir, 'label_encoder.joblib'))
        self.symptoms_vocab = joblib.load(os.path.join(self.out_dir, 'symptoms_vocab.joblib'))
        
        best_name, prefix = load_best_model_info(self.out_dir)
        print(f"Loaded Best Model: {best_name}")
        
        model_filename = f"model_{prefix}.joblib"
        self.model = joblib.load(os.path.join(self.out_dir, model_filename))
        
        # Compute symptom frequencies per disease for explainability
        print("Computing per-class symptom frequencies for explainability...")
        X_train = pd.read_csv(os.path.join(self.out_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(self.out_dir, 'y_train.csv'))['prognosis']
        
        # Group by disease class and compute mean (frequency of 1s)
        freq_df = X_train.groupby(y_train).mean()
        
        # Map back to disease names
        self.disease_symptom_freq = {}
        for cls_idx, row in freq_df.iterrows():
            disease_name = self.label_encoder.inverse_transform([cls_idx])[0]
            # Create a dict of {symptom: frequency}
            self.disease_symptom_freq[disease_name] = row.to_dict()
        print("Frequencies computed successfully.")
        
    def get_all_symptoms(self):
        return self.symptoms_vocab
        
    def predict(self, user_symptoms):
        """
        user_symptoms: list of symptom strings
        Returns top 3 diseases, confidences, and top contributing symptoms
        """
        # Create a feature vector of zeros
        feature_vector = np.zeros(len(self.symptoms_vocab))
        
        # Set 1 for the symptoms present
        for symptom in user_symptoms:
            if symptom in self.symptoms_vocab:
                idx = self.symptoms_vocab.index(symptom)
                feature_vector[idx] = 1
                
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_vector)[0]
        else:
            # Fallback if probability not supported (though our models support it)
            pred = self.model.predict(feature_vector)[0]
            probs = np.zeros(len(self.label_encoder.classes_))
            probs[pred] = 1.0
            
        # Get top 3
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_classes = self.label_encoder.inverse_transform(top3_idx)
        top3_probs = probs[top3_idx]
        
        results = []
        for cls, prob in zip(top3_classes, top3_probs):
            # Calculate contributing symptoms FOR THIS DISEASE
            cls_freqs = self.disease_symptom_freq.get(cls, {})
            
            # Filter to only the symptoms the user actually provided
            user_symptom_freqs = []
            for s in user_symptoms:
                if s in cls_freqs:
                    user_symptom_freqs.append((s, cls_freqs[s]))
            
            # Sort by frequency descending
            user_symptom_freqs.sort(key=lambda x: x[1], reverse=True)
            
            # Top 3 (or fewer) symptoms that actually occurred for this disease in the training set
            # We filter out symptoms that have a 0.0 frequency for this disease
            top_symptoms = [x[0] for x in user_symptom_freqs if x[1] > 0][:3]
            
            results.append({
                "disease": cls,
                "confidence": float(prob * 100),
                "contributing_symptoms": top_symptoms
            })
            
        return results

if __name__ == "__main__":
    predictor = DiseasePredictor()
    sample_symptoms = ["high_fever", "cold_hands_and_feets", "throat_irritation", "headache", "pain_behind_the_eyes"]
    print(f"\nPredicting for symptoms: {sample_symptoms}")
    res = predictor.predict(sample_symptoms)
    for r in res:
        print(f"\nDisease: {r['disease']} ({r['confidence']:.2f}%)")
        print(f"Contributing Symptoms: {r['contributing_symptoms']}")
