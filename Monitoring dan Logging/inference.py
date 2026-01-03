import mlflow
import mlflow.sklearn
import pandas as pd

# ===============================
# SET TRACKING URI (LOCAL)
# ===============================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# GANTI dengan RUN_ID milikmu
RUN_ID = "785b3c936c55461ea8f0be5af97cfc45"

# ===============================
# LOAD MODEL DARI MLFLOW
# ===============================
model = mlflow.sklearn.load_model(
    f"runs:/{RUN_ID}/model"
)

print("Model loaded successfully")

# ===============================
# DATA CONTOH (HARUS SAMA FEATURE)
# ===============================
data = pd.DataFrame([{
    "feature_0": 0.5,
    "feature_1": -1.2,
    "feature_2": 0.3,
    "feature_3": 2.1,
    "feature_4": -0.7
}])

# ===============================
# PREDICTION
# ===============================
prediction = model.predict(data)

print("Prediction result:", prediction)
