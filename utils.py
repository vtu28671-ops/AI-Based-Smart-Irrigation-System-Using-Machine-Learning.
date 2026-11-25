"""Helper functions for the AI-Based Smart Irrigation System.

This module has no Streamlit code, so every function can be unit-tested
(see test_utils.py).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "irrigation_prediction.csv.xlsx"
MODEL_FILE = BASE_DIR / "model" / "irrigation_model.pkl"
TARGET = "Irrigation_Need"
CLASS_ORDER = ["Low", "Medium", "High"]
HIGH_WARN = 0.30  # show a caution when P(High) is at least this value

# Units / help text shown on the input form
FEATURE_INFO = {
    "Soil_pH": ("Soil pH", "Valid range: 0 to 14."),
    "Soil_Moisture": ("Soil Moisture (%)", "Valid range: 0 to 100."),
    "Organic_Carbon": ("Organic Carbon", "Must not be negative."),
    "Electrical_Conductivity": ("Electrical Conductivity", "Must not be negative."),
    "Temperature_C": ("Temperature (°C)", "Air temperature in degrees Celsius."),
    "Humidity": ("Humidity (%)", "Valid range: 0 to 100."),
    "Rainfall_mm": ("Rainfall (mm)", "Must not be negative."),
    "Sunlight_Hours": ("Sunlight Hours (h)", "Valid range: 0 to 24."),
    "Wind_Speed_kmh": ("Wind Speed (km/h)", "Must not be negative."),
    "Field_Area_hectare": ("Field Area (hectare)", "Must be greater than 0."),
    "Previous_Irrigation_mm": ("Previous Irrigation (mm)", "Must not be negative."),
}

CLASS_MESSAGES = {
    "High": ("error", "High irrigation need predicted. Review soil moisture and "
                      "recent rainfall, and plan irrigation based on field conditions."),
    "Medium": ("warning", "Medium irrigation need predicted. Monitor soil moisture "
                          "and consider irrigation according to crop requirements."),
    "Low": ("success", "Low irrigation need predicted. Avoid unnecessary irrigation "
                       "and continue monitoring field conditions."),
}

WHATIF_FEATURES = ["Soil_Moisture", "Rainfall_mm", "Temperature_C",
                   "Humidity", "Wind_Speed_kmh", "Previous_Irrigation_mm"]


# ---------------------------------------------------------------- loading
def load_data():
    df = pd.read_excel(DATA_FILE)
    df = df.drop_duplicates().dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    return df.reset_index(drop=True)


def load_model():
    return joblib.load(MODEL_FILE)


def label_for(column):
    return FEATURE_INFO.get(column, (column.replace("_", " "), ""))[0]


def help_for(column):
    return FEATURE_INFO.get(column, ("", ""))[1] or None


# ------------------------------------------------------------ input records
def default_record(df, features):
    """Median for numeric columns, most common value for categories."""
    rec = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            rec[col] = round(float(df[col].median()), 2)
        else:
            rec[col] = str(df[col].mode().iloc[0])
    return rec


def sample_record(df, features, cls):
    """First real dataset record of the given class (Low / Medium / High)."""
    row = df[df[TARGET] == cls].iloc[0]
    rec = {}
    for col in features:
        v = row[col]
        rec[col] = round(float(v), 2) if pd.api.types.is_numeric_dtype(df[col]) else str(v)
    return rec


def validate_inputs(d):
    errors = []
    if not 0 <= d["Soil_pH"] <= 14:
        errors.append("Soil pH must be between 0 and 14.")
    if not 0 <= d["Soil_Moisture"] <= 100:
        errors.append("Soil Moisture must be between 0 and 100.")
    if not 0 <= d["Humidity"] <= 100:
        errors.append("Humidity must be between 0 and 100.")
    if not 0 <= d["Sunlight_Hours"] <= 24:
        errors.append("Sunlight Hours must be between 0 and 24.")
    if d["Organic_Carbon"] < 0:
        errors.append("Organic Carbon cannot be negative.")
    if d["Electrical_Conductivity"] < 0:
        errors.append("Electrical Conductivity cannot be negative.")
    if d["Rainfall_mm"] < 0:
        errors.append("Rainfall cannot be negative.")
    if d["Wind_Speed_kmh"] < 0:
        errors.append("Wind Speed cannot be negative.")
    if d["Field_Area_hectare"] <= 0:
        errors.append("Field Area must be greater than 0.")
    if d["Previous_Irrigation_mm"] < 0:
        errors.append("Previous Irrigation cannot be negative.")
    return errors


# --------------------------------------------------------------- prediction
def predict(model, inputs):
    frame = pd.DataFrame([inputs], columns=model.feature_names_in_)
    label = str(model.predict(frame)[0])
    proba = model.predict_proba(frame)[0]
    probs = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return label, {c: probs.get(c, 0.0) for c in CLASS_ORDER}


def percentile_of(series, value):
    return float((series <= value).mean() * 100)


def class_comparison(df, inputs, label, features):
    """Your value vs the average of the predicted class, plus percentile."""
    rows = []
    cls_df = df[df[TARGET] == label]
    for f in features:
        rows.append({
            "Feature": label_for(f),
            "Your value": round(float(inputs[f]), 2),
            "Average for %s-need records" % label: round(float(cls_df[f].mean()), 2),
            "Percentile in dataset": round(percentile_of(df[f], inputs[f]), 1),
        })
    return pd.DataFrame(rows)


def categorical_context(df, inputs, columns=("Crop_Growth_Stage", "Mulching_Used")):
    """Share of Low / Medium / High among dataset records with the same value."""
    rows = []
    for c in columns:
        sub = df[df[c].astype(str) == str(inputs[c])]
        share = sub[TARGET].value_counts(normalize=True) * 100
        row = {"Feature": label_for(c), "Your value": str(inputs[c]),
               "Records": int(len(sub))}
        for k in CLASS_ORDER:
            row["%% %s" % k] = round(float(share.get(k, 0.0)), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def recommendation_tips(df, inputs, label):
    """Data-driven notes. Every statement is computed from the dataset."""
    tips = []
    for f, name in [("Soil_Moisture", "soil moisture"), ("Rainfall_mm", "rainfall")]:
        pct = percentile_of(df[f], inputs[f])
        if pct <= 25:
            tips.append("Your %s is in the lowest quarter of the dataset (percentile %.0f)." % (name, pct))
        elif pct >= 75:
            tips.append("Your %s is in the highest quarter of the dataset (percentile %.0f)." % (name, pct))
    hi = df[df[TARGET] == "High"]["Soil_Moisture"].mean()
    lo = df[df[TARGET] == "Low"]["Soil_Moisture"].mean()
    if label == "High" or percentile_of(df["Soil_Moisture"], inputs["Soil_Moisture"]) <= 25:
        tips.append("In this dataset, High-need records have an average soil moisture of "
                    "%.1f against %.1f for Low-need records." % (hi, lo))
    mulch = df.groupby("Mulching_Used")[TARGET].apply(lambda s: (s == "Low").mean() * 100)
    if "Yes" in mulch.index and "No" in mulch.index and str(inputs["Mulching_Used"]) == "No":
        tips.append("In this dataset, %.0f%% of records with mulching are Low need, against "
                    "%.0f%% without mulching." % (mulch["Yes"], mulch["No"]))
    return tips[:4]


def whatif_curve(model, inputs, feature, df, points=25):
    lo, hi = float(df[feature].min()), float(df[feature].max())
    values = np.linspace(lo, hi, points)
    frame = pd.DataFrame([inputs] * points, columns=model.feature_names_in_)
    frame[feature] = values
    proba = model.predict_proba(frame)
    out = pd.DataFrame(proba, columns=[str(c) for c in model.classes_])
    out = out[CLASS_ORDER]
    out.index = np.round(values, 2)
    out.index.name = label_for(feature)
    return out


# ------------------------------------------------------ model-level numbers
def grouped_importance(model):
    names = model.named_steps["preprocessing"].get_feature_names_out()
    imp = model.named_steps["classifier"].feature_importances_
    cols = sorted(model.feature_names_in_, key=len, reverse=True)
    grouped = {}
    for n, v in zip(names, imp):
        base = n.split("__", 1)[1]
        key = next((c for c in cols if base == c or base.startswith(c + "_")), base)
        grouped[key] = grouped.get(key, 0.0) + float(v)
    return pd.Series(grouped).sort_values(ascending=False)


def make_split(df, model):
    X = df[list(model.feature_names_in_)]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def _metrics(y_true, y_pred):
    rep = pd.DataFrame(classification_report(
        y_true, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)).T
    rep = rep.loc[CLASS_ORDER + ["accuracy", "macro avg", "weighted avg"]]
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASS_ORDER),
                      index=["Actual " + c for c in CLASS_ORDER],
                      columns=["Pred " + c for c in CLASS_ORDER])
    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "high_recall": float(rep.loc["High", "recall"]) * 100,
        "report": rep, "confusion": cm,
    }


def evaluate_model(model, df):
    _, X_te, _, y_te = make_split(df, model)
    out = _metrics(y_te, model.predict(X_te))
    out["n_test"] = int(len(y_te))
    return out


def evaluate_decision_tree(model, df):
    X_tr, X_te, y_tr, y_te = make_split(df, model)
    num = X_tr.select_dtypes(include="number").columns
    cat = X_tr.select_dtypes(exclude="number").columns
    pre = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median"))]), num),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    tree = Pipeline([("preprocessing", pre),
                     ("classifier", DecisionTreeClassifier(random_state=42))])
    tree.fit(X_tr, y_tr)
    return _metrics(y_te, tree.predict(X_te))
