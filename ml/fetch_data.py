import os
import shutil
import kagglehub

def fetch_dataset():
    print("Downloading dataset from Kaggle...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
        print("Path to dataset files:", path)
        
        # Determine target directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(script_dir, 'data', 'raw')
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy CSV files to target directory
        for file in os.listdir(path):
            if file.endswith('.csv'):
                src = os.path.join(path, file)
                dst = os.path.join(target_dir, file)
                shutil.copy2(src, dst)
                print(f"Copied {file} to {target_dir}")
                
        print("Dataset successfully fetched and placed in ml/data/raw/")
    except Exception as e:
        print(f"Failed to fetch dataset: {e}")
        print("Please manually place the Kaggle dataset CSV files in ml/data/raw/ before proceeding.")

if __name__ == "__main__":
    fetch_dataset()
