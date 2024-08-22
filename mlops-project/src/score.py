from azureml.core.model import Model
import json
import numpy as np
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global model
    try:
        # Load the model from Azure ML
        model_path = Model.get_model_path('logistic-regression-model')
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def run(raw_data):
    try:
        # Parse the input data
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
            input_data = np.array(data['data'])
        elif isinstance(raw_data, np.ndarray):
            input_data = raw_data
        else:
            raise TypeError(f"Expected JSON string or ndarray, got {type(raw_data)}")

        # Make a prediction using the model
        prediction = model.predict(input_data)
        
        # Return the prediction
        return json.dumps({'prediction': prediction.tolist()})
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
