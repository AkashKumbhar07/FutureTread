# utils/ensemble.py

import joblib
import numpy as np
import os
from utils.train_model import retrain_model

def load_all_models():
    if not os.path.exists("models/rf_model.pkl") or not os.path.exists("models/log_model.pkl"):
        print("[INFO] Model(s) missing. Retraining...")
        retrain_model()

    rf = joblib.load("models/rf_model.pkl")
    log = joblib.load("models/log_model.pkl")
    return {"Random Forest": rf, "Logistic Regression": log}

def predict_with_ensemble(models, X):
    preds = np.array([model.predict(X) for model in models.values()])
    vote = np.round(np.mean(preds, axis=0)).astype(int)
    return vote
