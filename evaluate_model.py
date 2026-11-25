
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_excel(
    BASE_DIR / "irrigation_prediction.csv.xlsx"
)

df = df.drop_duplicates()
df = df.dropna(subset=["Irrigation_Need"])

X = df.drop(columns=["Irrigation_Need"])
y = df["Irrigation_Need"]

# Load trained model
model = joblib.load(
    BASE_DIR / "model" / "irrigation_model.pkl"
)

X = X[model.feature_names_in_]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Predict
predictions = model.predict(X_test)

# Evaluate performance
print("MODEL PERFORMANCE EVALUATION")

print("\nAccuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))

print("\nClasses:", model.classes_)
