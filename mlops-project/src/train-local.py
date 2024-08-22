from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the model as a pickle file
os.makedirs('outputs', exist_ok=True)
model_path = 'outputs/model.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model saved to {model_path}")
