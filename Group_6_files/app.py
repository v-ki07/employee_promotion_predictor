import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Employee Promotion Predictor",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Employee Promotion Eligibility Predictor")
st.markdown("### Group 6 - Workplace Fairness & AI Transparency Project")
st.divider()

# ─────────────────────────────────────────
# LOAD AND TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_and_train():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "hr_data_clean.csv")

    df = pd.read_csv(csv_path)

    # ---- ENCODING ----
    df_model = df.copy()

    text_cols = ["department", "region", "education", "gender", "recruitment_channel"]
    encoders = {}

    for col in text_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    # ---- FEATURES / TARGET ----
    X = df_model.drop(["employee_id", "is_promoted"], axis=1)
    y = df_model["is_promoted"]

    # make sure everything is numeric + clean
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # ---- SPLIT ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- SMOTE ----
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # ---- MODEL ----
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    return model, encoders, X.columns.tolist()

with st.spinner("Training model..."):
    model, encoders, feature_names = load_and_train()

st.success("Model ready!")

# ─────────────────────────────────────────
# SIDEBAR INPUT
# ─────────────────────────────────────────
st.sidebar.header("Employee Details")

department = st.sidebar.selectbox("Department", [
    "Sales & Marketing","Operations","Technology","Analytics","Finance","HR","Legal","Procurement","R&D"
])

education = st.sidebar.selectbox("Education Level", [
    "Bachelor's","Master's & above","Below Secondary"
])

gender = st.sidebar.radio("Gender", ["Female","Male"])

recruitment_channel = st.sidebar.selectbox("Recruitment Channel", [
    "sourcing","other","referred"
])

no_of_trainings = st.sidebar.slider("Trainings", 1, 10, 1)
age = st.sidebar.slider("Age", 20, 60, 30)

previous_year_rating = st.sidebar.select_slider(
    "Rating", options=[1.0,2.0,3.0,4.0,5.0], value=3.0
)

length_of_service = st.sidebar.slider("Years of Service", 1, 30, 5)

awards_won = st.sidebar.radio("Award", ["No","Yes"])

avg_training_score = st.sidebar.slider("Training Score", 40, 100, 60)

region = st.sidebar.selectbox("Region", [f"region_{i}" for i in range(1, 35)])

# ─────────────────────────────────────────
# ENCODE INPUT
# ─────────────────────────────────────────
def encode_input():

    dept = encoders["department"].transform([department])[0]
    edu = encoders["education"].transform([education])[0]
    gen = encoders["gender"].transform(["f" if gender == "Female" else "m"])[0]
    rec = encoders["recruitment_channel"].transform([recruitment_channel])[0]
    reg = encoders["region"].transform([region])[0]
    aw = 1 if awards_won == "Yes" else 0

    return pd.DataFrame([[ 
        dept, reg, edu, gen, rec,
        no_of_trainings, age, previous_year_rating,
        length_of_service, aw, avg_training_score
    ]], columns=feature_names)

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Summary")
    st.write({
        "Department": department,
        "Education": education,
        "Gender": gender,
        "Age": age
    })

with col2:
    st.subheader("Prediction")

    if st.button("Predict"):
        input_df = encode_input()

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("LIKELY TO BE PROMOTED")
        else:
            st.error("NOT LIKELY TO BE PROMOTED")

        st.metric("Probability", f"{probability:.2%}")
        st.progress(float(probability))

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_vals = shap_values[1][0]

        fig, ax = plt.subplots()
        ax.barh(feature_names, shap_vals)
        ax.set_title("Feature Impact (SHAP)")
        st.pyplot(fig)

st.caption("Group 6 Project - Random Forest + SHAP")
