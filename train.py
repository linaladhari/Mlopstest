import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import sys
import json

data_path = sys.argv[1]
model_path = sys.argv[2]

df = pd.read_csv(data_path)
X = df.drop("target", axis=1)
y = df["target"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, model_path)

# simple metrics
metrics = {"score": model.score(X, y)}
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f)
