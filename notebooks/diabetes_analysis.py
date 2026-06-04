import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# LOAD DATA

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# DATA QUALITY CHECK

print("\nMissing values:")
print(df.isnull().sum())

# In this dataset, zero values in certain columns are biologically impossible
# and represent missing or unrecorded data, not true zeros.
# Affected columns: Glucose, BloodPressure, SkinThickness, Insulin, BMI

zero_check_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("\nZero counts in biologically invalid columns:")
for col in zero_check_cols:
    zero_count = (df[col] == 0).sum()
    pct = (zero_count / len(df)) * 100
    print(f"  {col}: {zero_count} zeros ({pct:.1f}%)")

# Replace zeros with NaN to acknowledge them as missing
df[zero_check_cols] = df[zero_check_cols].replace(0, np.nan)

# Impute missing values with column median
# Median is preferred over mean here because some columns (Insulin)
# have high skew and outliers
df[zero_check_cols] = df[zero_check_cols].fillna(df[zero_check_cols].median())

print("\nAfter cleaning, missing values:")
print(df.isnull().sum())

print("\nBasic statistics after cleaning:")
print(df.describe().round(2))

# EXPLORATORY DATA ANALYSIS

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
visuals_dir = os.path.join(base_dir, "visuals")
os.makedirs(visuals_dir, exist_ok=True)

# Class distribution
print("\nOutcome distribution:")
print(df["Outcome"].value_counts())
print(df["Outcome"].value_counts(normalize=True).round(2))

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "correlation_heatmap.png"), dpi=150)
plt.close()
print("\nCorrelation heatmap saved.")

# Distribution of key features by outcome
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
key_features = ["Glucose", "BMI", "Age", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"]

for ax, feature in zip(axes.flatten(), key_features):
    df[df["Outcome"] == 0][feature].hist(
        ax=ax, alpha=0.6, label="No Diabetes", bins=20, color="steelblue"
    )
    df[df["Outcome"] == 1][feature].hist(
        ax=ax, alpha=0.6, label="Diabetes", bins=20, color="tomato"
    )
    ax.set_title(feature)
    ax.legend()

plt.suptitle("Feature Distributions by Diabetes Outcome", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "feature_distributions.png"), dpi=150)
plt.close()
print("Feature distribution chart saved.")

# MODEL TRAINING

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline ensures scaling is applied consistently to train and test sets
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# EVALUATION

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Diabetes", "Diabetes"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "confusion_matrix.png"), dpi=150)
plt.close()
print("Confusion matrix saved.")

# Feature importance (model coefficients after scaling)
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": pipeline.named_steps["model"].coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(coeff_df.to_string(index=False))

plt.figure(figsize=(8, 5))
colors = ["tomato" if c > 0 else "steelblue" for c in coeff_df["Coefficient"]]
plt.barh(coeff_df["Feature"], coeff_df["Coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Coefficients (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "feature_importance.png"), dpi=150)
plt.close()
print("Feature importance chart saved.")

# SAVE MODEL

joblib.dump(pipeline, os.path.join(base_dir, "diabetes_model.pkl"))
print("\nModel pipeline saved to root as diabetes_model.pkl")