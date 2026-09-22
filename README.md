# AI-Based Disease Prediction from Symptoms

A full-stack academic project that uses machine learning to predict diseases based on a given set of symptoms. The system includes a Python ML pipeline for training and evaluation, a FastAPI backend to serve predictions, and a modern React + Vite frontend for user interaction.

## Project Structure

- `ml/`: The machine learning pipeline, data processing, and model training scripts.
- `backend/`: The FastAPI server that exposes the prediction endpoints.
- `frontend/`: The React application with a Vite build system.

---

## 1. Dataset & ML Pipeline (`ml/`)

### Dataset Source
This project uses the "Disease Prediction Using Machine Learning" dataset by `kaushil268` on Kaggle (132 binary symptoms mapping to 41 diseases).
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **Data Placement**: If you don't use the automatic fetch script, place the downloaded `Training.csv` and `Testing.csv` into `ml/data/raw/`. No synthetic fallback dataset was used in this final build.

### Running the ML Pipeline
1. Navigate to the root directory and install requirements:
   ```bash
   pip install -r ml/requirements.txt
   ```
2. (Optional) Run the data fetcher if you don't have the CSVs manually placed:
   ```bash
   python ml/fetch_data.py
   ```
3. Preprocess the data (encodes labels, handles nulls):
   ```bash
   python ml/preprocess.py
   ```
4. Train the models (Decision Tree, Random Forest, Naive Bayes, SVM):
   ```bash
   python ml/train.py
   ```
5. Evaluate the models (5-fold CV, Test set metrics, Confusion Matrices):
   ```bash
   python ml/evaluate.py
   ```
   This generates a `model_comparison.csv` and confusion matrix PNGs in `ml/outputs/`.

*Note: Based on programmatic selection and the need for explainability (feature importances), the Random Forest model is selected as the production model by `predict.py`.*

---

## 2. Backend API (`backend/`)

A FastAPI service that loads the trained model and serves predictions over HTTP.

### Running the Backend
1. Install requirements:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Start the Uvicorn server (ensure you run this from the project root):
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
   ```
   The API will be available at `http://127.0.0.1:8001`. You can view the auto-generated OpenAPI docs at `http://127.0.0.1:8001/docs`.

---

## 3. Frontend Application (`frontend/`)

A modern, responsive React web app built with Vite, TailwindCSS, and React Query.

### Running the Frontend
1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   The application will be available at `http://localhost:5173`.

## Deployment

This project is prepared for easy deployment (e.g., Backend to Render, Frontend to Vercel).

### Backend (Render / Heroku)
The FastAPI backend requires the trained model artifacts (`.joblib` files) and precomputed CSVs to operate. These have explicitly been excluded from `.gitignore` so they are committed to your repository and available to the cloud provider on startup.

**Environment Variables:**
- `FRONTEND_URL`: Set this to the production URL of your frontend (e.g., `https://my-disease-app.vercel.app`). The backend will automatically add this to the CORS allowed origins.

> **Note on Free Tiers:** If you deploy the backend on a free tier (like Render), the instance will spin down after 15 minutes of inactivity. The first request after it spins down will cause a "cold start," which may take 30-50 seconds to respond while the server boots up and loads the machine learning models. A `/health` endpoint is provided for service monitoring.

### Frontend (Vercel / Netlify)
The React + Vite frontend accesses the backend via an environment variable.

**Environment Variables:**
- `VITE_API_URL`: Set this to your deployed backend URL (e.g., `https://my-backend-app.onrender.com`). If not provided, it falls back to `http://127.0.0.1:8002` for local development.

---

## Limitations

> **Academic Demo Only**
> This project is designed strictly as an academic engineering demonstration. **It is not a clinical diagnostic tool.** Consult a qualified healthcare professional for any real medical symptoms or concerns.

- **Dataset Realism & Combinatorial Nature**: The dataset features perfectly correlated binary indicators for target diseases. There are only about ~7.4 unique symptom patterns per disease on average, and 97.6% of the test set rows exist identically in the training set. The near-perfect scores achieved by the models reflect the deterministic/rules-based nature of this specific dataset rather than true clinical predictive power.
- **Missing Clinical Features**: The model operates purely on a binary presence/absence of symptoms. It does not account for critical clinical dimensions such as symptom severity, duration, onset sequence, patient demographics (age, sex, history), or lab results.
- **Synthetic Fallback**: *No synthetic fallback dataset was used in this final build.* The actual Kaggle dataset CSVs were successfully acquired and utilized.
