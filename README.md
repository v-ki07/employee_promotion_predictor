# Employee Performance Prediction System

This project is an AI-powered employee promotion prediction system developed to explore fairness, transparency, and explainable artificial intelligence in workplace decision-making.

The system predicts whether an employee is likely to be promoted based on workplace performance data while also analyzing possible bias in historical promotion decisions. To improve transparency, SHAP and LIME were integrated to explain how the model arrives at its predictions.

This project was developed by Group 6 for CSC 322: Computer Science Innovation and New Technology.

---
---

# 🎥 Demo Video

You can watch the full project demonstration here:

[YouTube Demo](https://youtu.be/L1bXutq6uRY)

---

# 🌍 Live Streamlit App

Access the deployed application here:

[Streamlit App](https://employeepromotionpredictor-dk2hmrdkzx8qjb7dpttfts.streamlit.app/)

---

# 📄 Project Documentation

View the full project report here:

[Project Report](Group_6_files/Group6_Employee_Performance_Report.docx)

---
# 📌 Project Objectives

The main goals of this project were to:

- Build a machine learning model capable of predicting employee promotion eligibility
- Detect possible bias in workplace promotion data
- Improve transparency using explainable AI techniques
- Develop a simple interactive web application using Streamlit

---

# 🚀 Features

- Employee promotion prediction
- Interactive Streamlit interface
- SHAP explainability visualizations
- LIME local explanations
- Bias and fairness analysis
- Class imbalance handling using SMOTE
- Real-time prediction probability score
- Human-readable prediction explanations

---

# 🧠 Technologies Used

- Python
- Streamlit
- Scikit-learn
- SHAP
- LIME
- Pandas
- NumPy
- Matplotlib
- imbalanced-learn (SMOTE)

---

# 📊 Dataset Information

The project uses the HR Analytics Employee Promotion dataset containing over 31,000 employee records across different departments and regions.

The dataset includes features such as:
- Department
- Education
- Gender
- Age
- Number of trainings
- Previous year rating
- Length of service
- Awards won
- Average training score

Target Variable:
- `is_promoted`
  - 1 = Promoted
  - 0 = Not Promoted

---

# ⚙️ Machine Learning Workflow

## 1. Data Preprocessing
- Missing value treatment
- Label encoding for categorical features
- Feature selection
- Train-test split

## 2. Handling Class Imbalance
SMOTE (Synthetic Minority Oversampling Technique) was used to address the imbalance between promoted and non-promoted employees.

## 3. Model Training
A Random Forest Classifier was trained on the processed dataset.

## 4. Explainability
- SHAP was used for both global and local explanations
- LIME was used for individual prediction explanations

---

# 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 92% |
| Weighted F1 Score | 0.90 |
| Precision (Promoted Class) | 0.54 |
| Recall (Promoted Class) | 0.25 |

---

# 🔍 Fairness & Bias Analysis

The project also focused heavily on fairness analysis.

Some important findings:
- Gender had very low influence on predictions
- Previous year rating and training score were the strongest predictors
- Department-level disparities existed in historical promotion data
- Intersectional bias patterns were identified across some departments

The goal was not just prediction accuracy, but responsible and transparent AI.

---

# 🌐 Streamlit Application

The Streamlit application allows users to:
- Enter employee details
- Receive instant promotion predictions
- View prediction probability
- See SHAP explanation charts
- Understand which features influenced the prediction


# 👥 Team Members

## Group 6

- Victoria Onu (Group Lead)
- Ugbede Favour
- Yilkudi Joseph
- Bakpet Shepherd
- Benjamin Annick
- Uzodinma Valentine
- Odomo Leader
- Kadiya Denzel
- James Jesse
- Lukman Isyaku
- John Idabor
- Elijah Eleribe
- Lori Scofied
- Sadiq Abdulhameed
- Idahosa Success

---

# 📚 References

- Breiman, L. (2001). Random Forests
- Lundberg & Lee (2017). SHAP
- Ribeiro et al. (2016). LIME
- Chawla et al. (2002). SMOTE

---

# 🔮 Future Improvements

Possible future improvements include:
- Adding advanced fairness metrics
- Improving recall for promoted employees
- Using more advanced ensemble models
- Deploying with cloud infrastructure
- Adding authentication and HR dashboards

---

# 🙏 Acknowledgements

Special thanks to:
- Dr. Felix Uloko
- Veritas University Abuja
- All Group 6 team members
