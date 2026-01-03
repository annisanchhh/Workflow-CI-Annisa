import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ambil environment variable MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

print("=== MLflow Environment Variables ===")
print("MLFLOW_TRACKING_URI:", MLFLOW_URI)
print("MLFLOW_TRACKING_USERNAME:", MLFLOW_USER)
print("MLFLOW_TRACKING_PASSWORD:", "******" if MLFLOW_PASSWORD else None)
print("===================================")

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
    if MLFLOW_USER and MLFLOW_PASSWORD:
        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD
    print("Using remote MLflow tracking")
else:
    print("Using local MLflow tracking")


mlflow.set_tracking_uri(MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

# Dataset dummy
X, y = make_regression(
    n_samples=100,
    n_features=5,
    noise=0.1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df["target"] = y

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1),
    df["target"],
    test_size=0.2,
    random_state=42
)

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Mean Squared Error: {mse}")

print("Training done! Model logged to MLflow.")
