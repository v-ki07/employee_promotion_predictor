
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Employee Promotion Predictor",
    page_icon="🏆",
    layout="wide"
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏆 Employee Promotion Eligibility Predictor")
st.markdown("### Group 6 - Workplace Fairness & AI Transparency Project")
st.markdown("""
This tool predicts whether an employee is likely to be promoted
based on their performance data. It uses a **fair, transparent AI model**
explained using SHAP values.
""")
st.divider()

# ─────────────────────────────────────────
# LOAD AND TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_and_train():

    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "hr_data_clean.csv")

    df = pd.read_csv(csv_path)

    # Encode text columns
    df_model = df.copy()

    # Split features and target
    X = df_model.drop(["employee_id","is_promoted"], axis=1)
    y = df_model["is_promoted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE balancing
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    return model, encoders, X.columns.tolist()

with st.spinner("⏳ Loading and training model... (~30 seconds on first load)"):
    model, encoders, feature_names = load_and_train()

st.success("✅ Model ready! Fill in the employee details on the left.")
st.divider()

# ─────────────────────────────────────────
# SIDEBAR — EMPLOYEE INPUT
# ─────────────────────────────────────────
st.sidebar.header("👤 Enter Employee Details")

department = st.sidebar.selectbox("Department", [
    "Sales & Marketing","Operations","Technology",
    "Analytics","Finance","HR","Legal","Procurement","R&D"])

education = st.sidebar.selectbox("Education Level", [
    "Bachelor's","Master's & above","Below Secondary"])

gender = st.sidebar.radio("Gender", ["Female","Male"])

recruitment_channel = st.sidebar.selectbox("Recruitment Channel", [
    "sourcing","other","referred"])

no_of_trainings = st.sidebar.slider(
    "Number of Trainings Completed", min_value=1, max_value=10, value=1)

age = st.sidebar.slider("Age", min_value=20, max_value=60, value=30)

previous_year_rating = st.sidebar.select_slider(
    "Previous Year Rating (1-5)",
    options=[1.0, 2.0, 3.0, 4.0, 5.0], value=3.0)

length_of_service = st.sidebar.slider(
    "Years of Service", min_value=1, max_value=30, value=5)

awards_won = st.sidebar.radio("Won an Award?", ["No","Yes"])

avg_training_score = st.sidebar.slider(
    "Average Training Score", min_value=40, max_value=100, value=60)

region = st.sidebar.selectbox(
    "Region", [f"region_{i}" for i in range(1, 35)])

# ─────────────────────────────────────────
# ENCODE USER INPUT
# ─────────────────────────────────────────
def encode_input():
    dept_enc = encoders["department"].transform([department])[0]
    edu_enc  = encoders["education"].transform([education])[0]
    gen_enc  = encoders["gender"].transform(
                ["f" if gender == "Female" else "m"])[0]
    rec_enc  = encoders["recruitment_channel"].transform(
                [recruitment_channel])[0]
    reg_enc  = encoders["region"].transform([region])[0]
    aw_enc   = 1.0 if awards_won == "Yes" else 0.0

    return pd.DataFrame([[
        dept_enc, reg_enc, edu_enc, gen_enc, rec_enc,
        no_of_trainings, age, previous_year_rating,
        length_of_service, aw_enc, avg_training_score
    ]], columns=feature_names)

# ─────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Employee Summary")
    summary = {
        "Department": department, "Education": education,
        "Gender": gender, "Age": age,
        "Years of Service": length_of_service,
        "Previous Year Rating": previous_year_rating,
        "Avg Training Score": avg_training_score,
        "Trainings Completed": no_of_trainings,
        "Award Won": awards_won,
        "Recruitment Channel": recruitment_channel,
        "Region": region
    }
    for key, val in summary.items():
        st.write(f"**{key}:** {val}")

with col2:
    st.subheader("🎯 Prediction Result")

    if st.button("🔮 Predict Promotion Eligibility",
                 use_container_width=True):
        input_df = encode_input()
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("## ✅ LIKELY TO BE PROMOTED")
        else:
            st.error("## ❌ NOT LIKELY TO BE PROMOTED")

        st.metric("Promotion Probability", f"{probability:.1%}")
        st.progress(float(probability))

        st.info("""
        **🔍 Fairness Note:** This prediction is based primarily on
        performance metrics. Gender and education have the lowest
        influence on this model's decisions.
        """)

        st.divider()
        st.subheader("🧠 Why This Prediction? (SHAP Explanation)")

        with st.spinner("Computing SHAP explanation..."):
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(input_df)
            shap_promo = shap_vals[:, :, 1]

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#ff4b4b" if v < 0 else "#00c853"
                      for v in shap_promo[0]]
            ax.barh(feature_names, shap_promo[0], color=colors)
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on promotion probability)")
            ax.set_title("Feature Impact on This Prediction")
            plt.tight_layout()
            st.pyplot(fig)

            st.caption("""
            🟢 Green bars = pushes TOWARD promotion |
            🔴 Red bars = pushes AWAY from promotion |
            Longer bar = stronger influence
            """)

st.divider()
st.caption(
    "Group 6 | Employee Performance Prediction | "
    "Powered by Random Forest + SHAP"
)
