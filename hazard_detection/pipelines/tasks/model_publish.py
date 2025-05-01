import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from clearml import Task, Model
import os
from enigmaai.config import Project, ConfigFactory

"""
Publish a specific draft model from the server if it meets the performance requirements. 
The model can be in Draft for Published state. If it is already in Published state, the model will 
not be published again. Refer to Model.publish() API.

This task combined model validation and publishing. The input (or draft) model is validated against
performance metrics. If it passes, the model gets published, otherwise do nothing.
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Publishing", 
                task_type=Task.TaskTypes.qc)

params = {
    'draft_model_id': '',      # specific version of the model to publish
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

# validate and publish model
if draft_model.published:
    print(f"Model already published with name:{draft_model.name} id:{draft_model.id}")
else: 
    # get validation metrics requirements
    valid_recall = project.get("validation-metrics-recall")
    print(f"Validating with performance requirement Recall >= {valid_recall}" )

    # get model eval recall metric
    draft_recall = draft_model.get_metadata("validation-metrics-recall")
    if not draft_recall:
        raise Exception(f"Draft model name:{draft_model.name} id:{draft_model.id} has not been evaluated. Please evaluate using the Model Evaluation task before publishing.")

    draft_recall = float(draft_recall)
    
    # validate and publish model
    if draft_recall >= valid_recall: 
        print(f"Publishing draft model name:{draft_model.name} id:{draft_model.id}")
        draft_model.publish()
        print(f"Draft model successfully published. New published model name:{draft_model.name} id:{draft_model.id}")
    else: 
        print(f"Draft model name:{draft_model.name} id:{draft_model.id} does not meet model validation requirements. Model NOT published.")
   
# show output    
print("pub_model_project:", project_name)
print("pub_model_id:", draft_model.id)
print("pub_model_name:", draft_model.name)
print("pub_model_variant:", draft_model.name)

# task output info
task.set_parameter("pub_model_project", project_name)
task.set_parameter("pub_model_id", draft_model.id)
task.set_parameter("pub_model_name", draft_model.name)
task.set_parameter("pub_model_variant", draft_model.name)