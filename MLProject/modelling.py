# MLProject/modelling.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ambil environment variable dari GitHub Actions
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not MLFLOW_URI or not MLFLOW_USER or not MLFLOW_PASSWORD:
    raise ValueError("MLflow environment variables not set!")

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_URI)

# Kalau DagsHub, login menggunakan username/password
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

# Buat dataset dummy (bisa ganti dengan CSV kamu)
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1), df["target"], test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():
    # Buat model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi & evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Log parameter, metric, model
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("Training done! Model logged to MLflow.")
print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))
print("MLFLOW_TRACKING_USERNAME:", os.getenv("MLFLOW_TRACKING_USERNAME"))
print("MLFLOW_TRACKING_PASSWORD:", "******" if os.getenv("MLFLOW_TRACKING_PASSWORD") else None)
