# Dataset Information

## Source

Pima Indians Diabetes Dataset, accessed via Plotly's public dataset repository.

Original source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), UCI Machine Learning Repository.

URL: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

## Description

The dataset contains health records from 768 female patients of Pima Indian heritage, aged 21 and older. It was originally collected to study diabetes prevalence in this population.

## Features

| Column | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg / height in m²) |
| DiabetesPedigreeFunction | Genetic risk score based on family history |
| Age | Age in years |
| Outcome | Diabetes diagnosis (0 = No, 1 = Yes) |

## Data Quality Notes

Several columns contain zero values that are biologically impossible and represent missing or unrecorded data:

| Column | Zero Count | Percentage |
|---|---|---|
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

These zeros were replaced with NaN and imputed using column medians prior to analysis.

## Limitations

- Data is limited to Pima Indian women aged 21 and older
- Results may not generalise to other populations or demographic groups
- High proportion of missing insulin and skin thickness values reduces reliability of those features