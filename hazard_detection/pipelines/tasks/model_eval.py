import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from clearml import Task, Dataset, Model
from pathlib import Path
import yaml
import os
import shutil
import tempfile
import torch
from ultralytics import YOLO
from enigmaai.config import Project, Config, ConfigFactory

"""
Constraints and requirements from the Business and data understanding phase will shape this phase. 
For example, the application domain’s model evaluation metrics, might include performance metrics, 
robustness, fairness, scalability, interpretability, model complexity degree, and model resource 
demand. It is suggested to evaluate the models on at least six complementary properties. Besides a 
performance metric, soft measures such as robustness, explainability, latency amongst others must 
be evaluated. The measures can be weighted differently depending on the application. In practical 
applications, explainability or robustness might be valued more than accuracy. Additionally, the 
model’s fairness or privacy might have to be assessed and mitigated. 

Performance requirements:
Recall
Latency
Memory (or resource) consumption

Here we test the recall on a common test dataset. The model complexity, interpretability etc is fine 
as the model is very small and straght forward. 
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Evaluation", 
                task_type=Task.TaskTypes.testing)

params = {
    'test_dataset_id': '',                                  # specific dataset for testing
    'test_dataset_name': 'test_dataset',                         # name of the dataset for testing
    'draft_model_id': 'f359bbd5cbe148f18f69702ef50704e2',   # the unpublished model to evaluate 
    'pub_model_name': 'yolo11n',                            # the published model name (also variant)
}


test_dataset_id = params['test_dataset_id']
test_dataset_name = params["test_dataset_name"]
draft_model_id = params['draft_model_id']
pub_model_name = params["pub_model_name"]

task.connect(params)
# task.execute_remotely(queue_name="default")

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
    # best published model not found, use draft (first train) as best
    # published draft model as first 
    best_model = draft_model    
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

    # device check and selection
    device_name = "mps"
    if torch.cuda.is_available():
        device_name = "cuda"
        print(f"CUDA is available on device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available(): #and torch.backends.mps.is_built():
        device_name = "mps"
        print("MPS is available (Apple Silicon GPU) with this version of PyTorch")
    else:
        print("No GPU available. Using CPU instead.")
        
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
if best_model.id != pub_model.id: # publish new model
    best_model.publish()
    print(f"Published new model name:{best_model.name} id:{best_model.id}")
else: # new model not better, nothing to publish
    print(f"Existing model name:{best_model.name} id:{best_model.id} is already the best, nothing to publish.")

# show output    
print("best_model_project:", project_name)
print("bets_model_id:", best_model.id)
print("best_model_name:", best_model.name)
print("best_model_variant:", best_model.name)

# task output info
task.set_parameter("best_model_project", project_name)
task.set_parameter("bets_model_id", best_model.id)
task.set_parameter("best_model_name", best_model.name)
task.set_parameter("best_model_variant", best_model.name)

task.mark_completed(status_message=f"Best evaluated model name:{best_model.name} id:{best_model.id}")