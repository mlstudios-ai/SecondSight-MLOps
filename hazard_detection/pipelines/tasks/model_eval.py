import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from clearml import Task, Dataset, Model
from pathlib import Path
import os
import shutil
import tempfile
from ultralytics import YOLO
from enigmaai import util
from enigmaai.config import Project, Config, ConfigFactory

"""
For hazard detection, accident prevention is paramount. To detect an obstacle is more important
than the what type of object is the obstacle. Therefore, Recall performance metric will be 
used over accuracty to evaluate the model. The newly trained (in draft) is compared to the current 
live (published) model. If the Recall score is higher, the newly train model will be published for 
inferencing.

Other metrics, both from model training (such as latency) and outside (such is resource demands), 
are not considered at this stage. 

This task will compare two models using the same dataset for evaluation. 
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Evaluation", 
                task_type=Task.TaskTypes.testing)

"""
One of test_dataset_id or test_dataset_name must be provided to load test dataset
"""
params = {
    'test_dataset_id': '',      # specific version of the dataset. if provided, ignore dataset_name
    'test_dataset_name': '',    # latest registered dataset. used if dataset_id is empty
    'draft_model_id': '',       # the unpublished model to evaluate 
    'pub_model_name': '',       # the published model name (also variant)
}

task.connect(params)
task.execute_remotely(queue_name="training")
task_params = task.get_parameters()
print("model_eval params=", task_params)

test_dataset_id = task_params['General/test_dataset_id']
test_dataset_name = task_params["General/test_dataset_name"]
draft_model_id = task_params['General/draft_model_id']
pub_model_name = task_params["General/pub_model_name"]

# no test dataset provided
if not test_dataset_id and not test_dataset_name:
    task.mark_completed(status_message="No dataset provided for evaluation.")
    exit(0)
    
# no model provided for evaluation
if not draft_model_id:
    task.mark_completed(status_message="No model provided for evaluation.")
    exit(0)
    
# Mandatory input param
if not pub_model_name:
    raise ValueError("Missing model. Please provide pub_model_name.")

# fetch the specific model for evaluation    
draft_model = Model(model_id=draft_model_id)    
print(f"Found draft model name:{draft_model.name} id:{draft_model.id}")

# fetch the published best model
server_models = Model.query_models(project_name=project_name, model_name=pub_model_name, only_published=True)
if not server_models:     
    best_model = draft_model # best published model not found, use first draft as best
    print(f"No published model found, use draft as the best model name:{draft_model.name} id:{draft_model.id}")
else:    
    # best published model found
    pub_model = server_models[0] # get the most recent one
    print(f"Found published model name:{pub_model.name} id:{pub_model.id}")  

    """
    Compare the model performance using a common dataset.
    Use metrics Recall at IoU=0.50:0.95 (TP / (TP + FN))
    """ 
        
    # load dataset
    if test_dataset_id:  # get specific dataset
        server_dataset = Dataset.get(dataset_id=test_dataset_id, alias="test")
    elif test_dataset_name: # get the latest registered dataset
        server_dataset = Dataset.get(dataset_name=test_dataset_name, dataset_project=project_name, only_completed=True,  alias="test")

    dataset_path = server_dataset.get_local_copy()

    # use temp directory for output
    working_dir = Path(tempfile.mkdtemp()) / project_name
    working_dir.mkdir(parents=True, exist_ok=True)    
    print("Working temp directory at:", working_dir)

    # contruct YAML config file
    data_yaml_path = working_dir / 'data.yaml'
    classes = ['hole', 'pole', 'stairs', 'bottle', 'rock']
    with open(data_yaml_path, 'w') as f: 
        f.write(f"train: {dataset_path}/images\n")   # Not used
        f.write(f"val: {dataset_path}/images\n")     # Not used
        f.write(f"test: {dataset_path}/images\n")    # Used
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n")
        
    print("YAML file created at: ", data_yaml_path)

    device_name = util.get_device_name()
        
    # evaluate the draft model    
    draft_model_path = draft_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded draft model name: {draft_model.name} id:{draft_model.id} to: {draft_model_path}")
    draft_yolo_model = YOLO(draft_model_path)
    draft_metrics = draft_yolo_model.val(data=data_yaml_path, split="test", imgsz=640, 
                                         batch=16, conf=0.25, iou=0.6, device=device_name)
    draft_recall = draft_metrics.box.map

    # evaluate the published model
    pub_model_path = pub_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded published model name: {pub_model.name} id:{pub_model.id} to: {pub_model_path}")
    pub_yolo_model = YOLO(pub_model_path)
    pub_metrics = pub_yolo_model.val(data=data_yaml_path, split="test", imgsz=640, 
                                     batch=16, conf=0.25, iou=0.6, device=device_name)
    pub_recall = pub_metrics.box.map
    
    # show metrics for comparision
    print("keys=", draft_metrics.keys)
    print("draft_metrics=", draft_metrics.mean_results(), " recall=", draft_recall)
    print("pub_metrics=", pub_metrics.mean_results(), " recall=", pub_recall)

    # compare and select the best model
    best_model = pub_model if pub_recall > draft_recall else draft_model
    
# publish the best model
if best_model.id == draft_model.id: # publish new model
    best_model.publish()
    print(f"Published new model name:{best_model.name} id:{best_model.id}")
else: # new model not better, nothing to publish
    print(f"Existing published model name:{best_model.name} id:{best_model.id} is already the best, nothing to publish.")

# show output    
print("best_model_project:", project_name)
print("bets_model_id:", best_model.id)
print("best_model_name:", best_model.name)
print("best_model_variant:", best_model.name)

# task output info
task.set_parameter("best_model_project", project_name)
task.set_parameter("best_model_id", best_model.id)
task.set_parameter("best_model_name", best_model.name)
task.set_parameter("best_model_variant", best_model.name)

task.flush()
if os.path.exists(working_dir.parent): 
        shutil.rmtree(working_dir.parent) # clean up output temp dir

task.mark_completed(status_message=f"Best evaluated model name:{best_model.name} id:{best_model.id}")