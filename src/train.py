import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import os

# Set MLflow tracking URI (can be a local directory, a database, or a remote server)
# For this example, we'll use a local 'mlruns' directory.
# In a real-world scenario, you might point this to an MLflow server.
# If you have an MLflow tracking server, you would set it like:
# os.environ["MLFLOW_TRACKING_URI"] = "http://your-mlflow-server:5000"

# Explicitly set the tracking URI for local tracking.
# If MLFLOW_TRACKING_URI is not set, MLflow defaults to ./mlruns.
# However, explicitly setting it is good practice, especially in CI/CD.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("Iris_Logistic_Regression_Experiment")


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters to log
    hyperparameters = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    }

    with mlflow.start_run():
        model = LogisticRegression(**hyperparameters)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log hyperparameters
        mlflow.log_params(hyperparameters)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Log the model (Scikit-learn flavor)
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        # You can still save it with joblib if you prefer a separate .pkl file
        joblib.dump(model, "model.pkl")
        mlflow.log_artifact("model.pkl") # Log the .pkl file as an artifact