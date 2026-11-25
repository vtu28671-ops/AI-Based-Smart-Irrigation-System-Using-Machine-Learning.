import pandas as pd

df = pd.read_excel("irrigation_prediction.csv.xlsx")

high = df[
    df["Irrigation_Need"].astype(str).str.strip().str.title() == "High"
]

print("HIGH IRRIGATION - ACTUAL DATASET RECORD")
print(high.iloc[0].drop("Irrigation_Need").to_string())