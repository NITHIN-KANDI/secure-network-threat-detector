import os
import pandas as pd
import joblib
import streamlit as st
import numpy as np
import random

# ─── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = r"C:\Users\dorai\OneDrive\Documents\Documents\SEM6\Computer Security\Project_cs\IDS-binary-classification"
RAW_CSV       = os.path.join(BASE_DIR, "data", "raw",   "UNSW_NB15_training-set.csv")
PREPROCESSOR  = os.path.join(BASE_DIR, "data", "processed", "preprocessor.pkl")
PCA_MODEL     = os.path.join(BASE_DIR, "data", "processed", "pca.pkl")
MODELS_DIR    = os.path.join(BASE_DIR, "src", "models_binary")

MODEL_FILES = {
    "Logistic Regression": "lr_model.pkl",
    "Random Forest":       "rf_model.pkl",
    "XGBoost":             "xgb_model.pkl",
    "LightGBM":            "lgb_model.pkl",
    "Voting Ensemble":     "voting_ensemble.pkl"
}

# ─── LOAD RAW SCHEMA FOR UI ────────────────────────────────────────────────────
df_schema   = pd.read_csv(RAW_CSV)
feature_cols = [c for c in df_schema.columns if c not in ("label","attack_cat","row_hash")]

# Precompute numeric ranges & categorical options
numeric_ranges     = {}
numeric_is_int     = {}
categorical_options = {}

for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df_schema[col]):
        lo = df_schema[col].min()
        hi = df_schema[col].max()
        numeric_ranges[col]    = (lo, hi)
        numeric_is_int[col]    = df_schema[col].dtype.kind in "iu"
    else:
        categorical_options[col] = df_schema[col].dropna().unique().tolist()

# ─── STREAMLIT UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="IDS Predictor", layout="centered")
st.title("IDS Single-Sample Prediction")

# Model selector
model_choice = st.selectbox("Select a model", list(MODEL_FILES.keys()))

# Randomize button
if st.button("Random Sample"):
    for col in feature_cols:
        if col in numeric_ranges:
            lo, hi = numeric_ranges[col]
            if numeric_is_int[col]:
                st.session_state[col] = random.randint(int(lo), int(hi))
            else:
                st.session_state[col] = random.uniform(lo, hi)
        else:
            st.session_state[col] = random.choice(categorical_options[col])

st.write("#### Enter feature values:")
user_input = {}

for col in feature_cols:
    key = f"input_{col}"
    if col in numeric_ranges:
        lo, hi = numeric_ranges[col]
        if numeric_is_int[col]:
            default = st.session_state.get(col, int((lo + hi) // 2))
            val = st.number_input(
                col,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default),
                key=key
            )
        else:
            default = st.session_state.get(col, float((lo + hi) / 2.0))
            val = st.number_input(
                col,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                key=key
            )
    else:
        opts    = categorical_options[col]
        default = st.session_state.get(col, opts[0])
        val     = st.selectbox(col, opts, index=opts.index(default), key=key)

    user_input[col] = val

# Predict button
if st.button("Predict"):
    st.subheader("Input Sample")
    st.json(user_input)

    sample_df = pd.DataFrame([user_input])

    # 1) Preprocessing
    try:
        preproc = joblib.load(PREPROCESSOR)
        Xp = preproc.transform(sample_df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # 2) PCA
    try:
        pca = joblib.load(PCA_MODEL)
        Xp = pca.transform(Xp)
    except Exception as e:
        st.error(f"PCA error: {e}")
        st.stop()

    # 3) Load model
    model_path = os.path.join(MODELS_DIR, MODEL_FILES[model_choice])
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

    # 4) Inference
    try:
        pred_label = model.predict(Xp)[0]
        pred_proba = model.predict_proba(Xp)[0,1]
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    label_str = "ATTACK" if pred_label == 1 else "NORMAL"

    st.subheader("Prediction Result")
    st.write(f"**Model:** {model_choice}")
    st.write(f"**Label:** {label_str}")
    st.write(f"**Attack Probability:** {pred_proba:.4f}")
