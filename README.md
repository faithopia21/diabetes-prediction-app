# Diabetes Risk Prediction App

A machine learning web application that predicts diabetes risk based on clinical indicators. Built with Python and Streamlit, the app allows users to input patient health data and receive an instant risk assessment with probability scores and a downloadable clinical report.

## Project Overview

This project applies logistic regression to the Pima Indians Diabetes Dataset to classify patients as high or low risk for diabetes. It is part of a broader portfolio focused on applying data science to health challenges.

**Model accuracy: 75%** 

## Dataset

- Source: Pima Indians Diabetes Dataset via Plotly's public dataset repository
- 768 patient records, 8 clinical features
- Target variable: Diabetes diagnosis (0 = No, 1 = Yes)

## Features

- Glucose level (strongest predictor)
- BMI
- Age
- Blood pressure
- Insulin level
- Skin thickness
- Diabetes pedigree function
- Number of pregnancies

## Key Findings

- Glucose level, BMI, and age are the strongest predictors of diabetes risk in this dataset
- The model achieves 75% accuracy on the held-out test set using logistic regression with standard scaling
- Class imbalance exists in the dataset (approximately 65% negative, 35% positive cases), which affects recall for the positive class

## App Features

- Interactive sidebar for entering patient clinical values
- Instant risk classification (High Risk / Low Risk)
- Probability scores for both outcomes
- Feature importance chart showing which indicators drive the prediction
- Downloadable PDF clinical report
- Downloadable CSV summary

## Tools Used

- Python
- pandas, NumPy
- scikit-learn (Logistic Regression, StandardScaler, Pipeline)
- matplotlib
- Streamlit
- ReportLab (PDF generation)

## Project Structure

```
diabetes-prediction-app/
│
├── data/
│   └── about_dataset.md
├── notebooks/
│   └── diabetes_analysis.py
├── visuals/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── risk_probability_chart.png
├── app.py
├── diabetes_model.pkl
├── requirements.txt
└── README.md
```

## How to Run

1. Clone this repository
2. Navigate to the project folder
3. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the app:

```bash
streamlit run app.py
```

## Limitations

- The dataset is based on Pima Indian women only, which limits generalisability across populations
- Zero values in columns like Glucose and BloodPressure likely represent missing data and have not been fully addressed in this version
- This tool is for educational purposes only and is not a medical diagnostic instrument

## Author

Faith Olaniyi