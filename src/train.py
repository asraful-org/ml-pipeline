import joblib
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn # Required for mlflow.sklearn.log_model

# --- Azure ML SDK Imports for MLflow Tracking URI ---
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential # For authentication


import numpy as np

# --- Azure ML Workspace Details (FILL THESE IN!) ---
# You need to replace these placeholders with your actual Azure ML workspace details.
# These can be found in the Azure portal or by using 'az ml workspace show'.
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.environ.get("AZURE_RESOURCE_GROUP_NAME")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE_NAME")

# --- MLflow Configuration ---
# 1. Connect to Azure ML workspace to get the MLflow Tracking URI
try:
    # Authenticate to Azure. DefaultAzureCredential tries various methods
    # (environment variables, managed identity, VS Code, Azure CLI, etc.).
    # InteractiveBrowserCredential is good for local interactive use if others fail.
    # Consider using InteractiveBrowserCredential() for local dev for a browser popup login
    # or DefaultAzureCredential() for more automated environments (like CI/CD, if properly configured).
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
        workspace_name=WORKSPACE_NAME
    )
    # Retrieve the MLflow tracking URI from the connected workspace
    azure_mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(azure_mlflow_tracking_uri)
    print(f"MLflow Tracking URI set to Azure ML: {mlflow.get_tracking_uri()}")

    # Set the experiment name. MLflow will create it if it doesn't exist within Azure ML.
    mlflow.set_experiment("Iris_Logistic_Regression_Azure_AML")

except Exception as e:
    print(f"Could not connect to Azure ML or retrieve MLflow URI: {e}")
    print("Defaulting to local MLflow tracking ('mlruns/' directory).")
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Iris_Logistic_Regression_Local")


print(f"Final MLflow Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters for the Logistic Regression model
    hyperparameters = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    }
    registered_model_name = "IrisLogisticRegressionModel"

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        print(f"MLflow Run ID: {run_id}")
        print(f"MLflow Experiment ID: {experiment_id}")

        # Initialize and train the Logistic Regression model
        model = LogisticRegression(**hyperparameters)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # --- MLflow Logging ---
        print("Logging parameters...")
        mlflow.log_params(hyperparameters)

        print("Logging metrics...")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Metrics for Run ID: {run_id}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

       #  print("Logging model with mlflow.sklearn.log_model...")
       # mlflow.sklearn.log_model(model, "logistic_regression_model")

        model_artifact_path = "logistic_regression_model_artifact"
        print(f"Logging model to run artifacts at path: {model_artifact_path}...")
        mlflow.sklearn.log_model(
            sk_model=model, # Use sk_model for sklearn flavor (or pytorch_model for PyTorch, etc.)
            artifact_path=model_artifact_path,
        )
        print("Model logged as artifact.")


        # 2. Construct the run_uri for the logged model
        # This URI points to the model artifact within the MLflow Tracking Server
        run_uri = f"runs:/{run_id}/{model_artifact_path}"
        print(f"Constructed run_uri for model: {run_uri}")

        # 3. Register the model from the run_uri to the MLflow Model Registry
        # The second argument is the NAME of the model in the Model Registry
        print(f"Registering model '{registered_model_name}' to MLflow Model Registry...")
        registered_model = mlflow.register_model(
            model_uri=run_uri,
            name=registered_model_name # This is the unique name of the model in the registry
        )
        print(f"Model registered. Version: {registered_model.version}")






        joblib_model_path = "model.pkl"
        print(f"Saving model with joblib to {joblib_model_path}...")
        joblib.dump(model, joblib_model_path)

        print("Model saved successfully. Verifying integrity...")



    try:
        # Attempt to load the model immediately after saving
        with open(joblib_model_path, 'rb') as f:
            loaded_model = joblib.load(f)
        print("Model integrity verified successfully at source.")
        # You could even run a quick prediction with a dummy input
        # if loaded_model is not None and hasattr(loaded_model, 'predict'):
        dummy_input = np.array([[5.1, 3.5, 1.4, 0.2]]) # Adjust for your model's expected input
        _ = loaded_model.predict(dummy_input)
        print("Model can make predictions.")

    except Exception as e:
        print(f"ERROR: Model integrity check failed at source: {e}")
        # In a real pipeline, you might want to exit with an error here
        # to prevent a corrupted model from proceeding.
        exit(1) # Fail the script if model is corrupted at creation



    print(f"Logging {joblib_model_path} as an MLflow artifact...")

    mlflow.log_artifact(joblib_model_path)

    print("\nTraining and logging complete.")
    print(f"View MLflow UI via Azure ML Studio: https://ml.azure.com/")
    print("Navigate to 'Jobs' or 'Experiments' to see your runs.")
