import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("train.csv")

# Features: Pclass, Sex (encoded), Age
df["Sex"] = df["Sex"].map({"male":0, "female":1})
df["Age"].fillna(df["Age"].median(), inplace=True)

X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

model_v1 = LogisticRegression(max_iter=200)
model_v1.fit(X, y)

joblib.dump(model_v1, "titanic_model_v1.pkl")
print("Baseline model saved.")