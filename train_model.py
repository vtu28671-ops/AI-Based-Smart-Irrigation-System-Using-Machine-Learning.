
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_excel("irrigation_prediction.csv.xlsx")

# Clean dataset
df = df.drop_duplicates()
df = df.dropna(subset=["Irrigation_Need"])

# Separate input and output
X = df.drop(columns=["Irrigation_Need"])
y = df["Irrigation_Need"]

# Identify feature types
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Random Forest model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight={"High": 8, "Low": 1, "Medium": 1}
    ))
])

# Train model
model.fit(X_train, y_train)

# Verify number of trees
print("Number of Trees:",
      model.named_steps["classifier"].n_estimators)

# Evaluate model
predictions = model.predict(X_test)

print("\nModel Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/irrigation_model.pkl")

print("\nModel trained and saved successfully!")
