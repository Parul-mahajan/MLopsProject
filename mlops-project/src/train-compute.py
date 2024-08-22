from azureml.core import Workspace, Experiment, ComputeTarget, Environment, ScriptRunConfig
from azureml.core.runconfig import RunConfiguration

# Initialize workspace with DefaultAzureCredential
# ws = Workspace.from_config(auth=DefaultAzureCredential())

# Print workspace details

# Connect to the Azure ML workspace
ws = Workspace.from_config(path='<CONFIG_PATH>')
print(f"Workspace: {ws.name}")

# Define the experiment
experiment_name = '<EXPERIMENT_NAME>'
experiment = Experiment(ws, name=experiment_name)

# Define the compute target
compute_target = ComputeTarget(workspace=ws, name='<COMPUTE_TARGET_NAME>')

# Define the environment
env = Environment.from_conda_specification(name='myenv', file_path='<ENV_SPEC_PATH>')

# Set up the RunConfiguration
run_config = RunConfiguration()
run_config.target = compute_target
run_config.environment = env

# Define script config without including Workspace object
src = ScriptRunConfig(source_directory='<SOURCE_DIRECTORY>',
                      script='train.py',
                      run_config=run_config)

# Submit the experiment
run = experiment.submit(src)
run.wait_for_completion(show_output=True)

print(f"Run ID: {run.id}")