import streamlit as st
import pandas as pd
import pickle
import json
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="POAG Phenotype Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_xgb_model.pkl"
FEATURE_PATH = BASE_DIR / "model" / "feature_columns.json"

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open(FEATURE_PATH, "r") as f:
        return json.load(f)

model = load_model()
feature_columns = load_features()

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(df):
    df = df.copy()

    df.columns = df.columns.str.replace(" ", "_") \
                           .str.replace("(", "", regex=False) \
                           .str.replace(")", "", regex=False)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    binary_cols = [
        "Smoking_History",
        "Alcoholic_History",
        "FH_Glaucoma",
        "Hypertension",
        "Cardiac",
        "Hyperlipidemia"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(0)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align: center;'>POAG Clinical Phenotype Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-driven stratification: Diabetic vs Non-diabetic POAG</h4>", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("Workflow")

section = st.sidebar.radio(
    "Navigate",
    [
        "Model Info",
        "Patient Profile",
        "Ocular Measurements",
        "Systemic Factors",
        "Prediction"
    ]
)

# -------------------------------
# SESSION STATE
# -------------------------------
if "data" not in st.session_state:
    st.session_state.data = {}

# -------------------------------
# MODEL INFO
# -------------------------------
if section == "Model Info":

    st.subheader("Model Information")

    st.markdown("""
    **Model:** XGBoost Classifier  
    **Task:** DM-POAG vs Non-DM POAG  
    **Dataset:** 664 patients  
    """)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", "0.61")
    c2.metric("Precision", "0.65")
    c3.metric("Recall", "0.60")
    c4.metric("F1 Score", "0.63")
    c5.metric("ROC-AUC", "0.63")

    st.info("Moderate discrimination; useful for phenotype stratification.")

# -------------------------------
# PATIENT PROFILE
# -------------------------------
elif section == "Patient Profile":

    st.subheader("Patient Profile")

    age = st.number_input("Age", 0, 100, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol", ["Yes", "No"])

    height = st.number_input("Height (cm)", 100, 220, 165)
    weight = st.number_input("Weight (kg)", 30, 150, 70)

    st.session_state.data.update({
        "Age": age,
        "Gender": gender,
        "Smoking_History": smoking,
        "Alcoholic_History": alcohol,
        "Height": height,
        "Weight": weight
    })

# -------------------------------
# OCULAR
# -------------------------------
elif section == "Ocular Measurements":

    st.subheader("Ocular Measurements")

    iop_re = st.number_input("IOP RE", 0.0, 50.0, 15.0)
    iop_le = st.number_input("IOP LE", 0.0, 50.0, 16.0)

    cct_re = st.number_input("CCT RE", 300.0, 700.0, 523.0)
    cct_le = st.number_input("CCT LE", 300.0, 700.0, 523.0)

    st.session_state.data.update({
        "RE_Intra_occular_Pressure (IOP-GAT)": iop_re,
        "LE_Intra_occular_Pressure (IOP-GAT)": iop_le,
        "RE_Central_Corneal_Thickness(CCT)": cct_re,
        "LE_Central_Corneal_Thickness(CCT)": cct_le
    })

# -------------------------------
# SYSTEMIC
# -------------------------------
elif section == "Systemic Factors":

    st.subheader("Systemic Risk Factors")

    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    cardiac = st.selectbox("Cardiac", ["Yes", "No"])
    hyperlipidemia = st.selectbox("Hyperlipidemia", ["Yes", "No"])
    fh = st.selectbox("Family History (Glaucoma)", ["Yes", "No"])

    st.session_state.data.update({
        "Hypertension": hypertension,
        "Cardiac": cardiac,
        "Hyperlipidemia": hyperlipidemia,
        "FH_Glaucoma": fh
    })

# -------------------------------
# PREDICTION
# -------------------------------
elif section == "Prediction":

    st.subheader("Prediction")

    data = st.session_state.data

    if len(data) == 0:
        st.warning("Enter patient data in previous steps.")
    else:
        height = data.get("Height", 165)
        weight = data.get("Weight", 70)

        bmi = weight / ((height / 100) ** 2)

        input_df = pd.DataFrame([{
            **data,
            "BMI": bmi
        }])

        df = preprocess(input_df)

        prob = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]

        st.metric("P(DM-POAG)", f"{prob:.3f}")

        if pred == 1:
            st.error("Diabetic-associated POAG")
        else:
            st.success("Non-diabetic POAG")

        # -------------------------------
        # SHAP BAR PLOT
        # -------------------------------
        st.markdown("### Feature Importance (SHAP)")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, df, plot_type="bar", show=False)
        st.pyplot(fig)