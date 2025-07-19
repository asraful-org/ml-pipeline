import os
import numpy as np # Added numpy for array conversion
from flask import Flask, request, jsonify
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
import mlflow.pyfunc # Important for loading MLflow-logged models

app = Flask(__name__)

# It's good practice to initialize these from environment variables
# which are passed from your GitHub Actions workflow.
# Use sensible defaults for local testing if env vars are not set.
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP_NAME = os.getenv("AZURE_RESOURCE_GROUP_NAME")
AZURE_ML_WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE_NAME")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "IrisLogisticRegressionModel")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest") # This will now be the dynamic version

# Initialize model as None. This will be populated by load_model().
model = None

@app.before_first_request
def load_model():
    """
    Loads the machine learning model from Azure ML Workspace.
    This function runs once when the Flask app starts.
    """
    global model
    print("Attempting to load model...")

    # Ensure all required Azure ML environment variables are set
    if not all([AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP_NAME, AZURE_ML_WORKSPACE_NAME]):
        error_msg = "Error: Azure ML workspace details (Subscription ID, Resource Group, Workspace Name) not found in environment variables. Model cannot be loaded."
        print(error_msg)
        # In a production scenario, it's critical to fail early if essential config is missing
        raise ValueError(error_msg)

    try:
        # Initialize MLClient to connect to Azure ML Workspace
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=AZURE_SUBSCRIPTION_ID,
            resource_group_name=AZURE_RESOURCE_GROUP_NAME,
            workspace_name=AZURE_ML_WORKSPACE_NAME
        )
        print(f"Connected to Azure ML Workspace: {AZURE_ML_WORKSPACE_NAME}")

        # Set MLflow tracking URI to Azure ML
        mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"MLflow Tracking URI set to Azure ML: {mlflow.get_tracking_uri()}")

        # Construct the model URI for MLflow to load the model
        # This URI format points to a model in the registry by name and version/stage
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
        print(f"Attempting to load model from MLflow Model Registry: {model_uri}")

        # Load the model using mlflow.pyfunc.load_model
        # This will download the model from Azure ML and load it.
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print(f"Model '{REGISTERED_MODEL_NAME}' version '{MODEL_VERSION}' loaded successfully from Azure ML!")

    except Exception as e:
        print(f"Error loading model from Azure ML MLflow Model Registry: {e}")
        print("Please ensure environment variables (AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP_NAME, AZURE_ML_WORKSPACE_NAME, REGISTERED_MODEL_NAME, MODEL_VERSION) are set correctly and the model exists.")
        model = None # Ensure model is None if loading fails
        # Re-raise the exception to indicate a critical startup failure
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives input data, makes predictions using the loaded model,
    and returns the predictions.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Server might be misconfigured or failed to load model from MLflow Registry.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)

        # Expected input: A dictionary with a 'data' key containing a list of feature vectors
        # Example: {"data": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}
        features = data.get('data')
        if not isinstance(features, list) or not all(isinstance(f, list) for f in features):
            return jsonify({'error': 'Invalid input format. Expecting JSON with a "data" key containing a list of feature vectors (e.g., {"data": [[val1, val2,...], [val3, val4,...]]}).'}), 400

        # Convert to numpy array for prediction if the model expects it
        input_array = np.array(features)

        # Make prediction
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
        app.logger.error(f"Invalid input data for model prediction: {ve}", exc_info=True)
        return jsonify({'error': f'Invalid input data for model prediction: {ve}. Ensure numerical data and correct dimensionality.'}), 400
    except Exception as e:
        # Log the full exception for debugging in production
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during prediction.'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify if the application is running and model is loaded.
    Updated to include model information.
    """
    status = "healthy" if model is not None else "unhealthy"
    response_data = {"status": status, "model_loaded": model is not None}

    if model is not None:
        response_data["model_info"] = {
            "name": REGISTERED_MODEL_NAME,
            "version": MODEL_VERSION,
            "source": "Azure ML Model Registry"
        }
    else:
        response_data["model_info"] = "Model not loaded"

    return jsonify(response_data), 200

if __name__ == '__main__':
    # Use PORT environment variable, default to 5001
    port = int(os.getenv('PORT', 5001))
    # In production, use a WSGI server like Gunicorn. debug=True should be avoided in production.
    app.run(host='0.0.0.0', port=port)