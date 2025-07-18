# inference_api/app.py
import os
import numpy as np
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc # Important for loading MLflow-logged models

# --- Azure ML SDK Imports for MLflow Tracking URI ---
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential # For authentication

app = Flask(__name__)

# --- Configuration for MLflow Model Registry ---
# These should be set as environment variables in your Dockerfile or deployment environment
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP_NAME = os.environ.get("AZURE_RESOURCE_GROUP_NAME")
WORKSPACE_NAME = os.environ.get("AZURE_ML_WORKSPACE_NAME")

# Define the name and version of the model in MLflow Model Registry
# You can make these environment variables too if you need to switch models/versions frequently
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "IrisLogisticRegressionModel")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "latest") # Or a specific version number, e.g., "1"

model = None # Initialize model as None
mlflow_tracking_uri = None # Initialize tracking URI

# --- Model Loading from MLflow Model Registry ---
# This block runs once when the Flask application starts
try:
    if not all([SUBSCRIPTION_ID, RESOURCE_GROUP_NAME, WORKSPACE_NAME]):
        raise ValueError("Azure ML workspace details (Subscription ID, Resource Group, Workspace Name) not found in environment variables.")

    # Initialize MLClient to get the MLflow tracking URI
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
        workspace_name=WORKSPACE_NAME
    )

    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow Tracking URI set to Azure ML: {mlflow.get_tracking_uri()}")

    # Construct the MLflow Model Registry URI
    # This URI format points to a model in the registry by name and version/stage
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
    print(f"Attempting to load model from MLflow Model Registry: {model_uri}")

    # Load the model using mlflow.pyfunc.load_model
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{REGISTERED_MODEL_NAME}' version '{MODEL_VERSION}' loaded successfully from MLflow Model Registry.")

except Exception as e:
    print(f"Error loading model from Azure ML MLflow Model Registry: {e}")
    print("Please ensure environment variables (AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP_NAME, AZURE_ML_WORKSPACE_NAME, REGISTERED_MODEL_NAME, MODEL_VERSION) are set correctly and the model exists.")
    # In a production scenario, you might want to raise an exception here
    # or implement a retry mechanism. For now, we'll let 'model' remain None.


@app.route('/')
def home():
    return "ML Model Inference API. Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Server might be misconfigured or failed to load model from MLflow Registry.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)

        # Expected input: A dictionary with a 'features' key containing a list of feature vectors
        # Example: {"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}
        features = data.get('features')
        if not isinstance(features, list) or not all(isinstance(f, list) for f in features):
            return jsonify({'error': 'Invalid input format. Expecting JSON with a "features" key containing a list of feature vectors (e.g., {"features": [[val1, val2,...], [val3, val4,...]]}).'}), 400

        # Convert to numpy array for prediction if the model expects it
        # mlflow.pyfunc.load_model often returns a PyFunc model that can handle various inputs,
        # but for scikit-learn, numpy array is typical.
        input_array = np.array(features)

        # Make prediction
        # Use model.predict() directly for classification
        predictions = model.predict(input_array).tolist()

        # Try to get probabilities if the model supports it
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_array).tolist()
        elif hasattr(model.unwrap_python_model(), 'predict_proba'): # For some MLflow pyfunc wrappers
            probabilities = model.unwrap_python_model().predict_proba(input_array).tolist()


        response = {'predictions': predictions}
        if probabilities is not None:
            response['probabilities'] = probabilities

        return jsonify(response)

    except ValueError as ve:
        return jsonify({'error': f'Invalid input data for model prediction: {ve}. Ensure numerical data and correct dimensionality.'}), 400
    except Exception as e:
        # Log the full exception for debugging in production
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

if __name__ == '__main__':
    # When running locally for development
    # In production, use a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5001, debug=True)