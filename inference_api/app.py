import os
import numpy as np
from flask import Flask, request, jsonify
import pickle
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration for Model
REGISTERED_MODEL_NAME = "IrisLogisticRegressionModel"
MODEL_VERSION = "local_bundled"

# Initialize model as None
model = None

def load_model():
    """Load the model from the local file system."""
    global model
    try:
        with open("model.pkl", 'rb') as f:
            model = joblib.load(f)

        logger.info("Model 'model.pkl' loaded successfully from local file system.")
        return True
    except FileNotFoundError:
        logger.error("Error: model.pkl not found. Model will not be loaded.")
        return False
    except Exception as e:
        logger.error(f"Error loading model from local file system: {e}")
        return False

# Load model at startup
model_loaded_successfully = load_model()

if not model_loaded_successfully:
    logger.warning("App starting without model. Some endpoints may not function properly.")

@app.route('/')
def home():
    return jsonify({
        "message": "ML Model Inference API",
        "endpoints": {
            "/predict": "POST - Make predictions",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        },
        "model_status": "loaded" if model is not None else "not_loaded"
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed model information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    model_details = {
        "name": REGISTERED_MODEL_NAME,
        "version": MODEL_VERSION,
        "source": "Local/Bundled model.pkl",
        "type": type(model).__name__
    }
    # Add model-specific details if available
    if hasattr(model, 'n_features_in_'):
        model_details["expected_features"] = model.n_features_in_
    if hasattr(model, 'classes_'):
        model_details["classes"] = model.classes_.tolist()

    return jsonify(model_details)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the loaded model."""
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Server might be misconfigured or model.pkl is missing/corrupt.'
        }), 503

    try:
        # Validate content type
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400

        # Get data from the POST request
        data = request.get_json()
        if data is None:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400

        # Validate input format
        features = data.get('data')
        if not features:
            return jsonify({
                'error': 'Missing "data" key in JSON payload'
            }), 400

        if not isinstance(features, list):
            return jsonify({
                'error': 'Data must be a list of feature vectors'
            }), 400

        # Validate feature vectors
        for i, feature_vector in enumerate(features):
            if not isinstance(feature_vector, list):
                return jsonify({
                    'error': f'Feature vector at index {i} must be a list of numbers'
                }), 400

            # Check for non-numeric values
            try:
                [float(x) for x in feature_vector]
            except (ValueError, TypeError):
                return jsonify({
                    'error': f'Feature vector at index {i} contains non-numeric values'
                }), 400

        # Convert to numpy array
        try:
            input_array = np.array(features, dtype=float)
        except Exception as e:
            return jsonify({
                'error': f'Failed to convert input to numpy array: {str(e)}'
            }), 400

        # Validate input dimensions
        if hasattr(model, 'n_features_in_') and input_array.shape[1] != model.n_features_in_:
            return jsonify({
                'error': f'Expected {model.n_features_in_} features, got {input_array.shape[1]}'
            }), 400

        # Make prediction
        predictions = model.predict(input_array)

        # Convert numpy types to Python types for JSON serialization
        predictions = [int(p) if isinstance(p, (np.integer, np.int32, np.int64))
                       else float(p) if isinstance(p, (np.floating, np.float32, np.float64))
        else p for p in predictions.tolist()]

        response = {'predictions': predictions}

        # Add probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(input_array).tolist()
                response['probabilities'] = probabilities
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")

        return jsonify(response)

    except ValueError as ve:
        logger.error(f"Invalid input data for model prediction: {ve}")
        return jsonify({
            'error': f'Invalid input data: {str(ve)}'
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred during prediction'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    status = "healthy" if model is not None else "unhealthy"
    response_data = {
        "status": status,
        "model_loaded": model is not None,
        "service": "ML Inference API"
    }

    if model is not None:
        response_data["model_info"] = {
            "name": REGISTERED_MODEL_NAME,
            "version": MODEL_VERSION,
            "source": "Local/Bundled model.pkl"
        }

    status_code = 200 if model is not None else 503
    return jsonify(response_data), status_code

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/predict', '/health', '/model-info']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed for this endpoint'
    }), 405

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)