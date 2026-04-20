"""
shared.py  – central loader (fixes the 30-vs-N feature mismatch)

The model was trained on selected_features (saved via joblib).
We always slice X to those features BEFORE calling predict / predict_proba.
"""

import os, joblib, pandas as pd, numpy as np, streamlit as st
from sklearn.preprocessing import StandardScaler

DATA_PATH    = "data/preprocessed_churn.csv"
MODEL_PATH   = "models/final_churn_model.pkl"
FEATURES_PATH= "models/selected_features.pkl"

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource(show_spinner=False)
def load_model():
    model    = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)      # list saved by research_model.py
    return model, features

@st.cache_data(show_spinner=False)
def get_Xy():
    df = load_data()
    X  = df.drop("Churn", axis=1)
    y  = df["Churn"]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, y, scaler
def safe_predict_proba(X_input):
    model, features = load_model()

    import pandas as pd
    import numpy as np

    # Convert numpy → DataFrame automatically
    if isinstance(X_input, np.ndarray):
        X_input = pd.DataFrame(X_input, columns=features)

    cols = [c for c in features if c in X_input.columns]

    return model.predict_proba(X_input[cols])[:, 1], cols
def safe_predict(X_input):
    probs, cols = safe_predict_proba(X_input)
    return (probs >= 0.5).astype(int)

# ── Result CSVs (pre-computed) ────────────────────────────────────────────────
RESULTS = "results"

def _read(name, **kw):
    p = os.path.join(RESULTS, name)
    if os.path.exists(p):
        return pd.read_csv(p, **kw)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def ablation_df():      return _read("ablation.csv")
@st.cache_data(show_spinner=False)
def model_comp_df():    return _read("model_comparison.csv")
@st.cache_data(show_spinner=False)
def budget_df():        return _read("budget_analysis.csv")
@st.cache_data(show_spinner=False)
def stability_df():     return _read("stability.csv")
@st.cache_data(show_spinner=False)
def threshold_df():     return _read("threshold_comparison.csv")
@st.cache_data(show_spinner=False)
def feature_count_df(): return _read("feature_count.csv")
@st.cache_data(show_spinner=False)
def shap_df():          return _read("shap_geo_overlap.csv")