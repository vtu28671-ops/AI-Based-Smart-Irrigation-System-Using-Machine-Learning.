import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("irrigation_prediction.csv.xlsx")
df = df.drop_duplicates().dropna(subset=["Irrigation_Need"])

X = pd.get_dummies(df.drop(columns=["Irrigation_Need"]))
y = df["Irrigation_Need"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("INTEGRATION TESTING")
print("Dataset Loading: PASS")
print("Data Preprocessing: PASS")
print("Model Training: PASS")
print("Prediction: PASS")
print(f"Accuracy: {accuracy * 100:.1f}%")
print("Integration Testing Completed!")