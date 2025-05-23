import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.image("your_banner.png", use_container_width=True)
st.title("Diabetes Prediction App")
st.markdown("""
This app uses **machine learning** to predict the likelihood of having **diabetes** based on health data.
""")

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.write("Enter the patient's health data below:")

# User input widgets
def user_input_features():
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=140, value=70)
    skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=900, value=79)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age', min_value=1, max_value=120, value=33)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
diabetes_label = {0: 'No Diabetes', 1: 'Diabetes'}
st.write(diabetes_label[prediction[0]])

st.subheader('Prediction Probability')
st.write(f"Probability of no diabetes: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of diabetes: {prediction_proba[0][1]:.2f}")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', 0, 20, step=1)
    glucose = st.slider('Glucose', 0, 200, 120)
    blood_pressure = st.slider('Blood Pressure', 0, 140, 70)

with col2:
    skin_thickness = st.slider('Skin Thickness', 0, 100, 20)
    insulin = st.slider('Insulin', 0, 900, 80)
    BMI = st.number_input('BMI', 0.0, 70.0, 24.0)
    diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    age = st.number_input('Age', 1, 100, 33)
