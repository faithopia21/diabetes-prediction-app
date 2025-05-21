import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

st.title("Diabetes Prediction App")

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
