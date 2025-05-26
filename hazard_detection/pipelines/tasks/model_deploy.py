import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from clearml import Task, Model
from enigmaai.config import Project, ConfigFactory
from enigmaai import model
from enigmaai.model import DeploymentError

"""
Deoploy model to FastAPI repo for inferencing.
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Deployment", 
                task_type=Task.TaskTypes.inference)

params = {
    'pub_model_id': '',          # deploy a specific published model
    'pub_model_name': 'yolo11n', # deploy the latest published model
}

task.connect(params)
task.execute_remotely(queue_name=project.get('queue-default'))
task_params = task.get_parameters()

pub_model_id = task_params["General/pub_model_id"]
pub_model_name = task_params["General/pub_model_name"]

# check model availability
if (not pub_model_id) and (not pub_model_name):
    raise ValueError("Missing model. Please provide model_id or model_name.")

if pub_model_id:
     pub_model = Model(model_id=pub_model_id)  # raise error by default if not found
elif pub_model_name:    
    server_models = Model.query_models(model_name=pub_model_name, only_published=True)
    if server_models: 
        pub_model = server_models[0]
        
if not pub_model:
    raise ValueError("Error fetching model. Please ensure model_id or model_name is available from the server")

pub_model_path = pub_model.get_local_copy(raise_on_error=True)

print(f"Downloaded model name: {pub_model.name} id:{pub_model.id} to: {pub_model_path}")

# setup github info for deployment and inferencing
repo_name = project.get('endpoint-repo-name')
repo_branch = project.get('endpoint-repo-branch')
repo_path = project.get('endpoint-repo-path')

try:
    model.deploy_model(pub_model_path, repo_name, repo_branch, repo_path)
    task.set_parameter("deployed_model_id", pub_model.id)
    print(f"Deployed model from: {pub_model_path} to {repo_name}/{repo_path} branch {repo_branch}")    
except DeploymentError as e:    
    print(f"Deployment Error: {e.message}: {pub_model.name} id:{pub_model.id} to: {pub_model_path}")  
except Exception as e:    
    print(f"Unkown Error: {e.message}: {pub_model.name} id:{pub_model.id} to: {pub_model_path}")  
    

