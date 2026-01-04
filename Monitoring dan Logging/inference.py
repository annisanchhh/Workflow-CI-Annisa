import mlflow
import pandas as pd

# =====================
# PAKSA LOCAL MLflow
# =====================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# =====================
# GANTI RUN_ID
# =====================
RUN_ID = "30c165101847412f9c6acd6907e5d7b4"

model_uri = f"runs:/{RUN_ID}/model"
model = mlflow.sklearn.load_model(model_uri)

# =====================
# DATA INFERENCE
# =====================
data = pd.DataFrame([{
    "feature_0": 0.5,
    "feature_1": -1.2,
    "feature_2": 0.3,
    "feature_3": 2.0,
    "feature_4": 1.1
}])

prediction = model.predict(data)
print("Prediction:", prediction)
