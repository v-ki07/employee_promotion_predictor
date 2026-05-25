import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────
# PAGE CONFIG
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
# LOAD DATA (GLOBAL FOR PLOTS)
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "hr_data_clean.csv")
df_raw = pd.read_csv(csv_path)

# ─────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_and_train():

    df = df_raw.copy()

    text_cols = ["department", "region", "education", "gender", "recruitment_channel"]
    encoders = {}

    for col in text_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(["employee_id", "is_promoted"], axis=1)
    y = df["is_promoted"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

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

    return pd.DataFrame([[
        encoders["department"].transform([department])[0],
        encoders["region"].transform([region])[0],
        encoders["education"].transform([education])[0],
        encoders["gender"].transform(["f" if gender == "Female" else "m"])[0],
        encoders["recruitment_channel"].transform([recruitment_channel])[0],
        no_of_trainings,
        age,
        previous_year_rating,
        length_of_service,
        1 if awards_won == "Yes" else 0,
        avg_training_score
    ]], columns=feature_names)

# ─────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Employee Summary")

    st.markdown(f"""
    **Department:** {department}  
    **Education:** {education}  
    **Gender:** {gender}  
    **Age:** {age}  
    **Years of Service:** {length_of_service}  
    **Rating:** {previous_year_rating}  
    **Training Score:** {avg_training_score}  
    **Trainings Completed:** {no_of_trainings}  
    **Award Won:** {awards_won}  
    **Recruitment Channel:** {recruitment_channel}  
    **Region:** {region}
    """)

with col2:
    st.subheader("Prediction")

    if st.button("Predict"):

        input_df = encode_input()

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success("LIKELY TO BE PROMOTED")
            st.info("This employee shows strong performance indicators.")
        else:
            st.error("NOT LIKELY TO BE PROMOTED")
            st.info("This employee currently lacks strong promotion signals.")

        st.metric("Promotion Probability", f"{probability:.2%}")
        st.progress(float(probability))

        st.divider()

        # ─────────────────────────────────────────
        # 📊 CHART 1: PROMOTION DISTRIBUTION
        # ─────────────────────────────────────────
        st.subheader("📊 Promotion Distribution")

        fig1, ax1 = plt.subplots()
        sns.countplot(data=df_raw, x="is_promoted", ax=ax1)
        ax1.set_title("Promotion vs Non-Promotion")

        st.pyplot(fig1)

        st.write("""
        📌 This chart shows class imbalance. Most employees are not promoted,
        so the model learns strict patterns before predicting promotion.
        """)

        st.divider()

        # ─────────────────────────────────────────
        # 📊 CHART 2: TRAINING SCORE IMPACT
        # ─────────────────────────────────────────
        st.subheader("📈 Training Score vs Promotion")

        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df_raw, x="is_promoted", y="avg_training_score", ax=ax2)

        st.pyplot(fig2)

        st.write("""
        📌 Higher training scores are strongly associated with promotions.
        This indicates performance is a key driver of decisions.
        """)

        st.divider()

        # ─────────────────────────────────────────
        # 📊 CHART 3: AGE DISTRIBUTION
        # ─────────────────────────────────────────
        st.subheader("📉 Age vs Promotion")

        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df_raw, x="is_promoted", y="age", ax=ax3)

        st.pyplot(fig3)

        st.write("""
        📌 Age does not strongly influence promotion outcomes,
        suggesting fairness in age-related decisions.
        """)

        st.divider()

        # ─────────────────────────────────────────
        # 📊 CHART 4: FEATURE IMPORTANCE
        # ─────────────────────────────────────────
        st.subheader("📊 Feature Importance")

        importance = model.feature_importances_

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance")

        fig4, ax4 = plt.subplots()
        ax4.barh(feat_df["Feature"], feat_df["Importance"])
        st.pyplot(fig4)

        st.write("""
        📌 The model relies mostly on performance-related features like:
        training score, rating, and years of service.
        """)

        st.divider()

        # ─────────────────────────────────────────
        # 🧠 SHAP EXPLANATION
        # ─────────────────────────────────────────
        st.subheader("🧠 SHAP Explanation")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        shap_vals = np.array(shap_vals).flatten()

        min_len = min(len(feature_names), len(shap_vals))

        fig5, ax5 = plt.subplots(figsize=(8, 4))
        colors = ["#ff4b4b" if v < 0 else "#00c853"
                  for v in shap_vals[:min_len]]

        ax5.barh(feature_names[:min_len], shap_vals[:min_len], color=colors)
        ax5.axvline(0, color="black")
        ax5.set_title("Feature Impact on Prediction")

        st.pyplot(fig5)

        st.write("""
        📌 Green bars increase promotion probability.
        Red bars decrease it.
        Longer bars = stronger influence on the decision.
        """)

st.caption("Group 6 Project - Random Forest + SHAP + Analytics Dashboard")
