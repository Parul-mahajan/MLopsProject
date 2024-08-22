import pickle
import pandas as pd
import numpy as np
import json
import sys

def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def run(raw_data):
    # Load the model
    model = load_model('outputs/model.pkl')
    
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

if __name__ == "__main__":
    # Example usage
    input_data = json.dumps({
        'data': [
            [5.1, 3.5, 1.4, 0.2]  # Example values for the Iris dataset
        ]
    })
    predictions = run(input_data)
    print(predictions)
