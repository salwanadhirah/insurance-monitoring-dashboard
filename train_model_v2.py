import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

df = pd.read_csv("insurance.csv")

X = df[["age", "bmi", "children", "smoker", "region"]]
y = df["charges"]

preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["smoker", "region"])
], remainder="passthrough")

model_v2 = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model_v2.fit(X, y)

joblib.dump(model_v2, "charges_model_v2.pkl")
print("Improved model saved.")
