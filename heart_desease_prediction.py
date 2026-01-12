"""
===========================================================
HEART DISEASE PREDICTION USING MACHINE LEARNING
Author: College Project Expo
Description:
This project predicts whether a person has heart disease
using multiple Machine Learning algorithms and selects
the best performing model.
===========================================================
"""

# =========================================================
# 1. IMPORT REQUIRED LIBRARIES
# =========================================================

import pandas as pd               # For data handling
import numpy as np                # For numerical operations
import matplotlib.pyplot as plt   # For data visualization
import seaborn as sns             # For advanced plots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings


# =========================================================
# 2. LOAD THE DATASET
# =========================================================
# This dataset contains medical information of patients

df = pd.read_csv("./Heart_desease_prdition.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print(df.head())


# =========================================================
# 3. DROP UNNECESSARY COLUMNS
# =========================================================
# 'id' column does not help in prediction

df.drop(columns=["id"], inplace=True)


# =========================================================
# 4. HANDLE MISSING VALUES
# =========================================================
# Missing data can affect model performance

# Numerical columns → fill with median (robust to outliers)
num_cols = ["trestbps", "chol", "thalch", "oldpeak"]
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical / discrete columns → fill with most frequent value
cat_cols = ["fbs", "restecg", "exang", "slope", "ca", "thal"]
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# =========================================================
# 5. TARGET VARIABLE TRANSFORMATION
# =========================================================
# Convert multi-class target into binary classification
# 0 → No heart disease
# 1 → Heart disease present

df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)


# =========================================================
# 6. ENCODE CATEGORICAL FEATURES
# =========================================================
# ML models only understand numbers

encoder = LabelEncoder()
categorical_features = ["sex", "dataset", "cp", "restecg", "slope", "thal"]

for col in categorical_features:
    df[col] = encoder.fit_transform(df[col])


# =========================================================
# 7. DATA VISUALIZATION (FOR PROJECT EXPO)
# =========================================================

# Distribution of heart disease
plt.figure(figsize=(5,4))
sns.countplot(x="num", data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()


# =========================================================
# 8. SPLIT DATA INTO FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["num"])   # Input features
y = df["num"]                  # Output label

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# 9. FEATURE SCALING
# =========================================================
# Scaling improves performance of distance-based models

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================================
# 10. TRAIN MULTIPLE MACHINE LEARNING MODELS
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

accuracy_results = {}

print("\nModel Training Started...\n")

for name, model in models.items():
    model.fit(X_train, y_train)              # Train model
    y_pred = model.predict(X_test)           # Predict on test data
    acc = accuracy_score(y_test, y_pred)     # Calculate accuracy
    accuracy_results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")


# =========================================================
# 11. MODEL COMPARISON GRAPH
# =========================================================

plt.figure(figsize=(8,5))
plt.bar(accuracy_results.keys(), accuracy_results.values())
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.title("Machine Learning Model Comparison")
plt.show()


# =========================================================
# 12. SELECT BEST MODEL
# =========================================================

best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

print("\n===================================")
print("BEST MODEL:", best_model_name)
print("ACCURACY:", accuracy_results[best_model_name])
print("===================================")


# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_best))


# =========================================================
# 15. FEATURE IMPORTANCE (FOR RANDOM FOREST)
# =========================================================

if best_model_name == "Random Forest":
    importances = best_model.feature_importances_
    features = X.columns

    feature_importance = pd.Series(importances, index=features)
    feature_importance.sort_values(ascending=False).plot(
        kind="bar",
        figsize=(10,5)
    )
    plt.title("Feature Importance in Heart Disease Prediction")
    plt.ylabel("Importance Score")
    plt.show()


# =========================================================
# 16. FINAL MESSAGE
# =========================================================

print("\nProject Execution Completed Successfully!")
print("This system can assist doctors in early heart disease detection.")


# =========================================================
# 17. INTERACTIVE HEART DISEASE PREDICTION SYSTEM
# =========================================================
# This system asks user to enter patient medical details
# and predicts heart disease using trained ML model.

print("\n================ HEART DISEASE PREDICTION SYSTEM ================\n")

# ---------------------------------------------------------
# FUNCTION TO TAKE NUMERIC INPUT SAFELY
# ---------------------------------------------------------
def get_numeric_input(prompt, dtype=float):
    while True:
        try:
            return dtype(input(prompt))
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

# ---------------------------------------------------------
# STEP 1: TAKE USER INPUT
# ---------------------------------------------------------

print("Please enter the following patient medical details:\n")

age = get_numeric_input("Age (years): ", int)
sex = get_numeric_input("Sex (1 = Male, 0 = Female): ", int)
dataset = get_numeric_input("Dataset (0 = Cleveland, 1 = Other): ", int)
cp = get_numeric_input("Chest Pain Type (0–3): ", int)
trestbps = get_numeric_input("Resting Blood Pressure (mm Hg): ")
chol = get_numeric_input("Serum Cholesterol (mg/dl): ")
fbs = get_numeric_input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No): ", int)
restecg = get_numeric_input("Resting ECG (0–2): ", int)
thalch = get_numeric_input("Maximum Heart Rate Achieved: ")
exang = get_numeric_input("Exercise Induced Angina (1 = Yes, 0 = No): ", int)
oldpeak = get_numeric_input("ST Depression (oldpeak): ")
slope = get_numeric_input("Slope of ST Segment (0–2): ", int)
ca = get_numeric_input("Number of Major Vessels (0–3): ", int)
thal = get_numeric_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): ", int)

# ---------------------------------------------------------
# STEP 2: CREATE DATAFRAME FROM USER INPUT
# ---------------------------------------------------------

user_input = {
    "age": age,
    "sex": sex,
    "dataset": dataset,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

user_df = pd.DataFrame([user_input])

print("\nEntered Patient Data:")
print(user_df)

# ---------------------------------------------------------
# STEP 3: APPLY SAME SCALING
# ---------------------------------------------------------

user_scaled = scaler.transform(user_df)

# ---------------------------------------------------------
# STEP 4: MAKE PREDICTION
# ---------------------------------------------------------

prediction = best_model.predict(user_scaled)
probability = best_model.predict_proba(user_scaled)

# ---------------------------------------------------------
# STEP 5: DISPLAY RESULT
# ---------------------------------------------------------

print("\n================ PREDICTION RESULT ================\n")

if prediction[0] == 1:
    print("⚠️ HEART DISEASE DETECTED")
    print(f"Risk Probability: {probability[0][1] * 100:.2f}%")
else:
    print("✅ NO HEART DISEASE DETECTED")
    print(f"Confidence Level: {probability[0][0] * 100:.2f}%")

print("\n===================================================")
