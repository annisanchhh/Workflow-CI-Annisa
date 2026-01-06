import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# =====================
# SET EXPERIMENT
# =====================
mlflow.set_experiment("workflow-ci-annisa")

# =====================
# DATA
# =====================
X, y = make_regression(
    n_samples=100,
    n_features=5,
    noise=0.1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["target"] = y

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("target", axis=1),
    df["target"],
    test_size=0.2,
    random_state=42
)

# =====================
# TRAIN & LOG MODEL
# =====================
with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print("RUN_ID:", run.info.run_id)
