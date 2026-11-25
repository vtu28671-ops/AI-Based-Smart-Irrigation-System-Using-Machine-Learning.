
import pandas as pd

df = pd.read_excel("irrigation_prediction.csv.xlsx")

low = df[
    df["Irrigation_Need"].astype(str).str.strip().str.title() == "Low"
]

print("LOW IRRIGATION - ACTUAL DATASET RECORD")
print(low.iloc[0].drop("Irrigation_Need").to_string())
