import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from clearml import Task, Dataset, Model, Logger
from pathlib import Path
import yaml
import numpy as np
import tempfile
from ultralytics import YOLO
from enigmaai.config import Project, ConfigFactory
from enigmaai import util

"""
For hazard detection, accident prevention is paramount. To detect an obstacle is more important
than the what type of object is the obstacle. Therefore, Recall performance metric will be 
used over accuracty to evaluate the model. The newly trained (in draft) is compared to the current 
live (published) model. If the Recall score is higher, the newly train model return as the best model

Other metrics, both from model training (such as latency) and outside (such is resource demands), 
are not considered at this stage. 

Metrics Recall at IoU=0.50:0.95 (TP / (TP + FN))

This task will compare two models using the same dataset for evaluation. 
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Evaluation", 
                task_type=Task.TaskTypes.qc)

"""
One of eval_dataset_id or eval_dataset_name must be provided to load evaluation dataset
"""
params = {
    'eval_dataset_id': '',      # specific version of the dataset. if provided, ignore dataset_name
    'eval_dataset_name': '',    # latest registered dataset. used if dataset_id is empty
    'draft_model_id': '',       # the unpublished model to evaluate 
    'pub_model_name': 'yolo11n',# the published model name (also variant) for comparison
    'eval_args': ''             # string format of dictionary of hyperparameters for YOLO.val()
}

task.connect(params)
task.execute_remotely(queue_name=project.get('queue-gpu'))
task_params = task.get_parameters()
print("model_eval params=", task_params)

eval_dataset_id = task_params['General/eval_dataset_id']
eval_dataset_name = task_params["General/eval_dataset_name"]
draft_model_id = task_params['General/draft_model_id']
pub_model_name = task_params["General/pub_model_name"]
eval_args_str = task_params["General/eval_args"]

# no eval dataset provided
if (not eval_dataset_id) and (not eval_dataset_name):
    task.mark_completed(status_message="No dataset provided for evaluation.")
    exit(0)
    
# no model provided for evaluation
if not draft_model_id:
    raise ValueError("Missing new/draft model. Please provide draft_model_id.")
    
# Mandatory input param
if not pub_model_name:
    raise ValueError("Missing model. Please provide pub_model_name.")

# Mandatory input param
if not eval_args_str:
    raise ValueError("Missing eval configurations. Please provide eval_args_str.")

eval_args = yaml.safe_load(eval_args_str)

# use temp directory for output
working_dir = Path(tempfile.mkdtemp()) / project_name
working_dir.mkdir(parents=True, exist_ok=True)    
print("Working temp directory at:", working_dir)

# fetch the specific model for evaluation    
draft_model = Model(model_id=draft_model_id)    
print(f"Found draft model name:{draft_model.name} id:{draft_model.id}")

# load dataset
if eval_dataset_id:  # get specific dataset
    server_dataset = Dataset.get(dataset_id=eval_dataset_id, alias="eval")
elif eval_dataset_name: # get the latest registered dataset
    server_dataset = Dataset.get(dataset_name=eval_dataset_name, dataset_project=project_name, only_completed=True,  alias="eval")

dataset_path = server_dataset.get_local_copy()

# check if data.yaml exist
source_data_yaml = Path(dataset_path) / 'data.yaml'   # use original yaml, only modify paths
if not source_data_yaml.exists():
    raise FileNotFoundError(f"{source_data_yaml} does not exist.")

# modify the paths and save to new dataset
with open(source_data_yaml, "r") as file:
    # load existing yaml data
    data_yaml = yaml.safe_load(file)
    data_yaml["train"] = f"{dataset_path}/images"   # NOT used
    data_yaml["val"] = f"{dataset_path}/images"     # NOT used
    data_yaml["test"] = f"{dataset_path}/images"    # USED!
    
    # new yaml with the right paths  
    data_yaml_path = working_dir / 'data.yaml'        
    with open(data_yaml_path, "w") as file:
        yaml.dump(data_yaml, file, default_flow_style=False)
    
print("YAML file created at: ", data_yaml_path)

# eval input params: args + other data
eval_args = eval_args.copy() # copy to prevent altering original by reference
eval_args["data"] = str(data_yaml_path)
eval_args["name"] = pub_model_name
eval_args["device"] = util.get_device_name()
eval_args["project"] = str(working_dir)
    
# evaluate the draft model    
draft_model_path = draft_model.get_local_copy(raise_on_error=True)
print(f"Downloaded draft model name: {draft_model.name} id:{draft_model.id} to: {draft_model_path}")
draft_yolo_model = YOLO(draft_model_path)
draft_metrics = draft_yolo_model.val(**eval_args)
draft_recall = draft_metrics.box.mr
draft_model.set_metadata("validation-metrics-recall", draft_recall) # used for validation
draft_model.report_scalar("Evaluation Metrics", "draft", draft_recall, 0)

# upload results reference for report analysis 
task.upload_artifact(name=f"{pub_model_name}_eval_config", artifact_object=eval_args)
task.upload_artifact(name=f"{pub_model_name}_draft_metrics", artifact_object=draft_metrics)
    
# fetch the published best model
server_models = Model.query_models(model_name=pub_model_name, only_published=True)
if not server_models:     
    best_model = draft_model # best published model not found, use first draft as best
    print(f"No published model found, use draft as the best model name:{draft_model.name} id:{draft_model.id}")
else:    
    # best published model found
    pub_model = server_models[0] # get the most recent one
    task.set_parameter("pub_model_id", pub_model.id)
    print(f"Found published model name:{pub_model.name} id:{pub_model.id}")  

    """
    Compare the model performance using a common dataset.
    """ 
    # evaluate the published model
    pub_model_path = pub_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded published model name: {pub_model.name} id:{pub_model.id} to: {pub_model_path}")
    pub_yolo_model = YOLO(pub_model_path)
    pub_metrics = pub_yolo_model.val(**eval_args)
    pub_recall = pub_metrics.box.mr
    pub_model.set_metadata("validation-metrics-recall", pub_recall) #  # used for validation 
    # NOTE: can not report metrics on published models, use metadata due to scalar inaccessible using SDK
    
    # log task scalar metrics
    logger = task.get_logger()
    logger.report_scalar("Evaluation Metrics", "recall (draft)", draft_recall, 0)
    logger.report_scalar("Evaluation Metrics", "recall (published)", pub_recall, 0)
    print("keys=", draft_metrics.keys)
    print("draft_metrics=", draft_metrics.mean_results(), " recall=", draft_recall)
    print("pub_metrics=", pub_metrics.mean_results(), " recall=", pub_recall)

    # compare and select the best model
    best_model = pub_model if pub_recall > draft_recall else draft_model    
    
    # upload results reference for report analysis 
    task.upload_artifact(name=f"{pub_model_name}_pub_metrics", artifact_object=pub_metrics)
    
# check the best model
if best_model.id == draft_model.id: 
    print(f"Draft model is the best model name:{best_model.name} id:{best_model.id}")
else: # new model not better, nothing to publish
    print(f"Existing published model is the best name:{best_model.name} id:{best_model.id}.")

# show output    
print("best_model_project:", project_name)
print("bets_model_id:", best_model.id)
print("best_model_name:", best_model.name)
print("best_model_variant:", best_model.name)

"""
Data analysis and visualisation
"""
# dataset analysis and visualisation
class_names = data_yaml.get("names")
labels_dir = dataset_path + "/labels/"
class_dist = util.class_dist(labels_dir, class_names)
task.get_logger().report_histogram (
    title="Class Distribution",
    series="Evaluation",
    values=np.array(class_dist),
    iteration=0,
    xlabels=class_names,
    xaxis="Class",
    yaxis="Count"
)

# task output info
task.set_parameter("best_model_project", project_name)
task.set_parameter("best_model_id", best_model.id)
task.set_parameter("best_model_name", best_model.name)
task.set_parameter("best_model_variant", best_model.name)