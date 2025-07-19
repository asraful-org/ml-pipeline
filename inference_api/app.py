import os
import numpy as np
from flask import Flask, request, jsonify
import pickle # Import pickle for loading local model

app = Flask(__name__)

# --- Configuration for Model ---
# If your model.pkl is bundled in the Docker image,
# it will be loaded directly from the file system.
# No dynamic versioning or Azure ML connection needed for this approach.
REGISTERED_MODEL_NAME = "IrisLogisticRegressionModel" # Static name for health check
MODEL_VERSION = "local_bundled" # Indicate it's a local, bundled model

# Initialize model as None. This will be populated by the direct loading attempt.
model = None

# --- Model Loading (Original/Simplified Approach) ---
# This block attempts to load the model directly when the Flask application starts.
# It runs only once at startup.
try:
    # Assuming model.pkl is located in the same directory as app.py (i.e., /app inside the container)
    with open("model.pkl", 'rb') as f:
        model = pickle.load(f)
    print("Model 'model.pkl' loaded successfully from local file system.")
except FileNotFoundError:
    print("Error: model.pkl not found. Model will not be loaded.")
    # In a production scenario, you might want to raise a critical error here
    # to prevent the container from starting if the model is essential.
except Exception as e:
    print(f"Error loading model from local file system: {e}")
    print("Please ensure model.pkl exists and is a valid pickle file.")
    model = None


@app.route('/')
def home():
    return "ML Model Inference API. Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives input data, makes predictions using the loaded model,
    and returns the predictions.
    """
    if model is None:
        # Updated error message to reflect local model loading
        return jsonify({'error': 'Model not loaded. Server might be misconfigured or model.pkl is missing/corrupt.'}), 500

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
        elif hasattr(getattr(model, 'unwrap_python_model', None)(), 'predict_proba'): # More robust check for MLflow pyfunc wrappers
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
    Updated to include model information for a locally bundled model.
    """
    status = "healthy" if model is not None else "unhealthy"
    response_data = {"status": status, "model_loaded": model is not None}

    if model is not None:
        response_data["model_info"] = {
            "name": REGISTERED_MODEL_NAME,
            "version": MODEL_VERSION,
            "source": "Local/Bundled model.pkl" # Updated source
        }
    else:
        response_data["model_info"] = "Model not loaded"

    return jsonify(response_data), 200

if __name__ == '__main__':
    # Use PORT environment variable, default to 5001
    port = int(os.getenv('PORT', 5001))
    # In production, use a WSGI server like Gunicorn. debug=True should be avoided in production.
    app.run(host='0.0.0.0', port=port)
