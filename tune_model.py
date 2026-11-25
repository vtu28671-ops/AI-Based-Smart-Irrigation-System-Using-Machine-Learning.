
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

# Load dataset
df = pd.read_excel(BASE_DIR / "irrigation_prediction.csv.xlsx")
df = df.drop_duplicates()
df = df.dropna(subset=["Irrigation_Need"])

X = df.drop(columns=["Irrigation_Need"])
y = df["Irrigation_Need"]

# Split into training, validation and testing data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25, random_state=42, stratify=y_train_val
)

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# Test different High-class weights
weights = [
    "balanced_subsample",
    {"High": 3, "Low": 1, "Medium": 1},
    {"High": 5, "Low": 1, "Medium": 1},
    {"High": 8, "Low": 1, "Medium": 1}
]

for weight in weights:

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=weight
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print("\n================================")
    print("CLASS WEIGHT:", weight)
    print("Accuracy:",
          round(accuracy_score(y_val, predictions) * 100, 2), "%")

    print(classification_report(
        y_val, predictions, zero_division=0
    ))

print("\nMODEL COMPARISON COMPLETED")
