import pandas as pd

df = pd.read_excel("irrigation_prediction.csv.xlsx")

medium = df[
    df["Irrigation_Need"].astype(str).str.strip().str.title() == "Medium"
]

print("MEDIUM IRRIGATION - ACTUAL DATASET RECORD")
print(medium.iloc[0].drop("Irrigation_Need").to_string())