import os
import pandas as pd
import numpy as np
import joblib

def load_best_model_info(out_dir):
    """Reads the comparison CSV and returns the best model's filename prefix and whether it supports feature importances."""
    comp_df = pd.read_csv(os.path.join(out_dir, 'model_comparison.csv'))
    
    # The first row is the best one (since we sorted by Test Macro F1 descending in evaluate.py)
    # But just in case, let's explicitly find it
    best_row = comp_df.loc[comp_df['Test Macro F1'].idxmax()]
    best_model_name = best_row['Model']
    
    prefix = best_model_name.replace(' ', '_').lower()
    
    # Tree-based models support feature_importances_ natively in sklearn
    supports_importance = best_model_name in ['Decision Tree', 'Random Forest']
    
    return best_model_name, prefix, supports_importance

class DiseasePredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = os.path.join(self.base_dir, 'outputs')
        
        self.label_encoder = joblib.load(os.path.join(self.out_dir, 'label_encoder.joblib'))
        self.symptoms_vocab = joblib.load(os.path.join(self.out_dir, 'symptoms_vocab.joblib'))
        
        best_name, prefix, self.supports_importance = load_best_model_info(self.out_dir)
        print(f"Loaded Best Model: {best_name}")
        
        model_filename = f"model_{prefix}.joblib"
        self.model = joblib.load(os.path.join(self.out_dir, model_filename))
        
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
        
        # Feature importances (global, not local, since we use simple tree/RF)
        # Note: Local interpretability (e.g., SHAP) is better, but global is a simple approximation
        # for which symptoms matter most. We'll return the highest importance symptoms that the user ACTUALLY has.
        top_symptoms_for_prediction = []
        if self.supports_importance and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Find which of the USER'S symptoms have the highest importance globally
            user_symptom_indices = [self.symptoms_vocab.index(s) for s in user_symptoms if s in self.symptoms_vocab]
            if user_symptom_indices:
                user_importances = [(self.symptoms_vocab[i], importances[i]) for i in user_symptom_indices]
                user_importances.sort(key=lambda x: x[1], reverse=True)
                top_symptoms_for_prediction = [x[0] for x in user_importances[:3]]
                
        results = []
        for cls, prob in zip(top3_classes, top3_probs):
            results.append({
                "disease": cls,
                "confidence": float(prob * 100),
                "contributing_symptoms": top_symptoms_for_prediction
            })
            
        return results

if __name__ == "__main__":
    predictor = DiseasePredictor()
    sample_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
    print(f"\nPredicting for symptoms: {sample_symptoms}")
    res = predictor.predict(sample_symptoms)
    for r in res:
        print(f"{r['disease']}: {r['confidence']:.2f}%")
