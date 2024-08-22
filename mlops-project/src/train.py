from azureml.core import Workspace, Experiment,  Environment
from azureml.core.compute import ComputeTarget
from azureml.core.model import Model
from sklearn.datasets import load_iris
from azureml.core.model import InferenceConfig
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np


# Load workspace configuration from a file
ws = Workspace.from_config(path='mlops-project/src/config.json')
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model as a pickle file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


# Upload the model to Azure ML
model = Model.register(workspace=ws, model_path='model.pkl', model_name='logistic-regression-model')

print(f"Model registered: {model.name}, version: {model.version}")
