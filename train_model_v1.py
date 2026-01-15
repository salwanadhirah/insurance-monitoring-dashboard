import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("insurance.csv")

X = df[["age", "bmi", "children"]]
y = df["charges"]

model_v1 = LinearRegression()
model_v1.fit(X, y)

joblib.dump(model_v1, "charges_model_v1.pkl")
print("Baseline model saved.")
