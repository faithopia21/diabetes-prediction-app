import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Diabetes Risk Prediction App", layout="wide")

# LOAD OR TRAIN MODEL
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return pipeline, acc, cm, report, X

model, acc, cm, report, feature_data = train_model()

st.title("🩺 Diabetes Risk Prediction App")
st.markdown("A machine learning tool that predicts diabetes risk using clinical indicators.")

st.success(f"Model Accuracy: {acc:.2f}")

st.sidebar.header("Patient Input Features")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 33)

input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]],
                          columns=[
                              "Pregnancies", "Glucose", "BloodPressure",
                              "SkinThickness", "Insulin", "BMI",
                              "DiabetesPedigreeFunction", "Age"
                          ])


prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("High Risk of Diabetes Detected")
    else:
        st.success("Low Risk of Diabetes Detected")

with col2:
    st.subheader("Probability")
    st.write(f"No Diabetes: {proba[0]:.2f}")
    st.write(f"Diabetes: {proba[1]:.2f}")

# PROBABILITY VISUALIZATION
st.subheader("Risk Probability Visualization")

fig, ax = plt.subplots()
ax.bar(["No Diabetes", "Diabetes"], proba)
ax.set_ylim(0, 1)
st.pyplot(fig)

# FEATURE IMPORTANCE (EXPLANATION)
st.subheader("Model Explanation (Feature Influence)")

coeff = model.named_steps["model"].coef_[0]
features = feature_data.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Impact": coeff
}).sort_values(by="Impact", key=abs, ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.markdown("""
### Interpretation:
- Positive values increase diabetes risk
- Negative values reduce risk
- Glucose, BMI, and Age are usually strongest drivers
""")

# MODEL EXPLANATION
st.subheader("Model Insight")

st.write("""
The model evaluates risk based on:
- Glucose level (strongest indicator)
- BMI and Age
- Blood pressure and insulin levels
- Genetic predisposition (DPF)
""")

st.write(f"Confidence: {max(proba)*100:.2f}%")

# DOWNLOAD REPORT
def generate_healthcare_pdf(data_dict, prediction, proba):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, 760, "DIABETES RISK ASSESSMENT REPORT")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, 745, "AI-Based Clinical Decision Support System")
    pdf.drawString(50, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pdf.line(50, 720, 550, 720)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, 700, "Patient Clinical Summary")

    pdf.setFont("Helvetica", 10)

    y = 680
    for key, value in data_dict.items():
        pdf.drawString(50, y, f"{key}: {value}")
        y -= 18

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y - 10, "Model Prediction")

    pdf.setFont("Helvetica", 10)
    result = "HIGH RISK OF DIABETES" if prediction == 1 else "LOW RISK OF DIABETES"

    pdf.drawString(50, y - 30, f"Prediction: {result}")
    pdf.drawString(50, y - 50, f"Probability (No Diabetes): {proba[0]:.2f}")
    pdf.drawString(50, y - 70, f"Probability (Diabetes): {proba[1]:.2f}")

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y - 100, "Clinical Interpretation")

    pdf.setFont("Helvetica", 10)

    if prediction == 1:
        interpretation = (
            "The model indicates elevated risk factors associated with diabetes. "
            "Clinical follow-up and confirmatory diagnostic tests are recommended."
        )
    else:
        interpretation = (
            "The model indicates low risk based on current indicators. "
            "Continue routine monitoring and healthy lifestyle practices."
        )

    pdf.drawString(50, y - 120, interpretation[:90])
    pdf.drawString(50, y - 135, interpretation[90:180])

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(50, 50, "Disclaimer: This report is generated by an AI system and is not a medical diagnosis.")
    pdf.drawString(50, 35, "It should be used for educational and decision-support purposes only.")

    pdf.save()
    buffer.seek(0)
    return buffer

data_dict = input_df.iloc[0].to_dict()

pdf_file = generate_healthcare_pdf(data_dict, pred, proba)

st.download_button(
    label="Download Clinical PDF Report",
    data=pdf_file,
    file_name="diabetes_clinical_report.pdf",
    mime="application/pdf"
)

report = pd.DataFrame({
    "Feature": input_data.columns,
    "Value": input_data.values[0]
})

report["Prediction"] = "Diabetes" if prediction == 1 else "No Diabetes"

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Report (CSV)",
    data=csv,
    file_name="diabetes_report.csv",
    mime="text/csv"
)



# MEDICAL DISCLAIMER
st.warning("""
⚠️ Disclaimer: This tool is for educational purposes only.
It is not a medical diagnosis. Always consult a healthcare professional.
""")