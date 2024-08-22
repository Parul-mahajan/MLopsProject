# MLOps Project

This project demonstrates a complete MLOps workflow using Azure Machine Learning. The workflow includes creating infrastructure, training a model, scoring the model, and deploying the model to an Azure Kubernetes Service (AKS) cluster.



### Files Description

- **config.json**: Configuration file for Azure ML workspace.
- **deploy.py**: Script to deploy the trained model to an AKS cluster.
- **environment.yml**: Conda environment specification file.
- **requirements.txt**: Python dependencies file.
- **score.py**: Script for scoring the model.
- **scoring.py**: Additional scoring script (if needed).
- **train-compute.py**: Script to train the model on a compute cluster.
- **train-deploy.yml**: YAML file for Azure DevOps pipeline to automate training and deployment.
- **train-local.py**: Script to train the model locally.
- **train.py**: Script to train the model and register it in Azure ML.

## Workflow

### 1. Creating Infrastructure

The infrastructure is defined and managed using Azure resources. The Azure Machine Learning workspace is created to manage the machine learning lifecycle. The workspace provides a centralized place to work with all artifacts, datasets, models, and deployments.


### 2. Training the Model

- **Local Training**: Use [`train-local.py`] to train the model locally.
- **Cluster Training**: Use [`train-compute.py`] to train the model on an Azure ML compute cluster.

The training script ([`train.py`]loads the Iris dataset, trains a Logistic Regression model, and registers the trained model in the Azure ML workspace.

### 3. Scoring the Model

The [`score.py`] script is used to score the model. It contains the logic for model inference.

### 4. Deploying the Model

The [`deploy.py`] script deploys the trained model to an AKS cluster. It sets up the environment, defines the inference configuration, and deploys the model as a web service.

## Azure DevOps Pipeline

The [`train-deploy.yml`] file defines the Azure DevOps pipeline. It includes the following steps:

1. **Set up Python Environment**: Installs necessary Python packages.
2. **Train Model on Compute Cluster**: Executes [`train-compute.py`] to train the model.
3. **Deploy Model to AKS**: Executes [`deploy.py`] to deploy the model to AKS.



### Setup

1. Clone the repository.
2. Configure the Azure ML workspace by updating [`config.json`]
3. Set up the Azure DevOps pipeline using [`train-deploy.yml`].

### Running the Pipeline

Trigger the Azure DevOps pipeline to automate the training and deployment process.

## Conclusion

This project provides a comprehensive MLOps workflow using Azure Machine Learning and Azure DevOps. It automates the process of training, scoring, and deploying machine learning models, ensuring a streamlined and efficient workflow.