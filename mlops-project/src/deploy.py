from azureml.core import Workspace, Model, Environment, ComputeTarget
from azureml.core.webservice import AksWebservice
from azureml.core.compute import AksCompute
from azureml.core.model import InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies

# Connect to the Azure ML workspace
ws = Workspace.from_config(path='<CONFIG_PATH>')

# Load the model
model = Model(ws, '<MODEL_NAME>')
print("Model loaded")

# Define environment from YAML file
env = Environment.from_conda_specification(name='myenv', file_path='<ENV_SPEC_PATH>')
env.register(workspace=ws) 
print("workspace loaded")

# Define inference configuration
inference_config = InferenceConfig(entry_script='mlops-project/src/score.py', environment=env)
print("infer loaded")

# Get the AKS cluster
aks_name = '<AKS_CLUSTER_NAME>'

# Check if the cluster exists
if aks_name not in ws.compute_targets:
    print(f"Creating new AKS cluster '{aks_name}'...")
    aks_config = AksCompute.provisioning_configuration()
    aks_target = AksCompute.create(ws, aks_name, aks_config)
    aks_target.wait_for_completion(show_output=True)
else:
    aks_target = ws.compute_targets[aks_name]
    print(f"AKS cluster '{aks_name}' found")

# Define deployment configuration
deployment_config = AksWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service_name = '<SERVICE_NAME>'

service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    deployment_target=aks_target
)
service.wait_for_deployment(show_output=True)

print(f"Service deployed at: {service.scoring_uri}")