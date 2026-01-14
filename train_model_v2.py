import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("train.csv")

df["Sex"] = df["Sex"].map({"male":0, "female":1})
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

model_v2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=300))
])

model_v2.fit(X, y)
joblib.dump(model_v2, "titanic_model_v2.pkl")
print("Improved model saved.")