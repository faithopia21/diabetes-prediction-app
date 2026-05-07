import streamlit as st
import pandas as pd
import joblib

# Load model
model = 
joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Image (make sure banner.png exists in same folder)
st.image("banner.png", use_container_width=True)

st.title("Diabetes Prediction App")

st.markdown("""
This app uses **machine learning (Logistic Regression)** to predict diabetes risk.
""")

st.write("Enter patient details below:")

# Input UI
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose", 0, 200, 120)
    blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 120, 33)

# Convert to dataframe (VERY IMPORTANT ORDER MATCHES TRAINING DATA)
input_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age
]], columns=[
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
])

# Prediction
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]

st.subheader("Result")

if prediction == 1:
    st.error("High risk of Diabetes")
else:
    st.success("Low risk of Diabetes")

st.subheader("Probability")
st.write(f"No Diabetes: {proba[0]:.2f}")
st.write(f"Diabetes: {proba[1]:.2f}")