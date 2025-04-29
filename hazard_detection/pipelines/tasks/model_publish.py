import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from clearml import Task, Model
import os
from enigmaai.config import Project, ConfigFactory

"""
Publis a specific draft model on the server.
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Publishing", 
                task_type=Task.TaskTypes.qc)

params = {
    'draft_model_id': '',      # specific version of the dataset. if provided, ignore dataset_name
}

task.connect(params)
task.execute_remotely(queue_name="default")
task_params = task.get_parameters()
print("model_publish params=", task_params)

draft_model_id = task_params['General/draft_model_id']
    
# no model provided for publishing
if not draft_model_id:
    task.mark_completed(status_message="No draft model provided for publishing.")
    exit(0)
    
# fetch the specific draft model   
draft_model = Model(model_id=draft_model_id)    
print(f"Found draft model name:{draft_model.name} id:{draft_model.id}")

# publish the model
draft_model.publish()
print(f"Publishing new model name:{draft_model.name} id:{draft_model.id}")

# verify model publication
server_models = Model.query_models(model_name=draft_model, only_published=True)
if not server_models: 
    raise Exception(f"Error publishing draft model name:{draft_model.name} id:{draft_model.id}. Please check the logs.")  

pub_model = server_models[0]    # get the latest publised model
if pub_model.id != draft_model.id:
    raise Exception(f"Error publishing draft model name:{draft_model.name} id:{draft_model.id}. Please check the logs.")  

print(f"Draft model successfully published. New published model name:{pub_model.name} id:{pub_model.id}")

# show output    
print("pub_model_project:", project_name)
print("pub_model_id:", pub_model.id)
print("pub_model_name:", pub_model.name)
print("pub_model_variant:", pub_model.name)

# task output info
task.set_parameter("pub_model_project", project_name)
task.set_parameter("pub_model_id", pub_model.id)
task.set_parameter("pub_model_name", pub_model.name)
task.set_parameter("pub_model_variant", pub_model.name)