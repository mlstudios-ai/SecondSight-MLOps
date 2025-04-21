from clearml import Task, Dataset, Model
from pathlib import Path
import yaml
import os
import shutil
import tempfile
import torch
from ultralytics import YOLO
# from enigmaai.config import Project, Config, ConfigFactory

"""
Train YOLOv11 model using the latest split dataset stored on ClearML server.
THe dataset needs to be in the following structure:

data.yaml
train/images/
train/labels/
val/images/
val/labels/
test/images/
test/labels/

IMPORTANT: The dataset will be downloaded to a cached directory and will NOT be copied into 
local_working_directory. This is to preserve the built-in caching mechanism of 
Dataset.get_local_copy(). Caching keep tracks of changes in datdaset and only download if 
there is a new version to prevent repeat downloads, especially with a large dataset.

NOTE: this is for training only, model evaluation task with compare and register the best model for deployment.
"""

# NOT WORKING: setup.py not running on execute_remotely, hence can not import enigmaai package
# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"
Task.add_requirements('numpy', '==1.26.4')
# Task.add_requirements('pytorch', '')

task = Task.init(project_name=project_name, 
                task_name="Model Training", 
                task_type=Task.TaskTypes.training)

params = {
    'dataset_id': '',                       # specific version of the dataset
    'dataset_name': '',              # latest registered dataset
    'model_id': '',                         # specific version of the model 
    'model_variant': '',             # base model variant from ultralytics if no model given
    'hyperparameters':  {}                  # dictionary of hyperparameters for training
}

# logger = task.get_logger()
task.connect(params)
# task.execute_remotely(queue_name="training")

dataset_id = params['dataset_id']
dataset_name = params['dataset_name']
model_id = params['model_id']
model_variant = params["model_variant"]
model_name = model_variant
hyperparameters = params["hyperparameters"]

# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to train on.")
    exit(0)
    
# Mandatory input param
if not model_variant:
    raise ValueError("Missing model variant. Please provide model_variant.")

if not hyperparameters:
    raise ValueError("Missing hyperparameters. Please provide hyperparameters for YOLO.train().")

"""
Prepare dataset.
"""

if dataset_id:  # get specific dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, alias=model_variant)
elif dataset_name: # get the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True,  alias=model_variant)

extract_path = server_dataset.get_local_copy()
print(f"Downloaded dataset name: {server_dataset.name} id:{server_dataset.id} to: {extract_path}")

"""
Model training.
"""

# use temp director for output
working_dir = Path(tempfile.mkdtemp()) / project_name
working_dir.mkdir(parents=True, exist_ok=True)    
print("Working temp directory at:", working_dir)

# contruct YAML config file
data_yaml_path = working_dir / 'data.yaml'
classes = ['hole', 'pole', 'stairs', 'bottle', 'rock']
with open(data_yaml_path, 'w') as f:
    f.write(f"train: {extract_path}/train/images\n")
    f.write(f"val: {extract_path}/val/images\n")
    f.write(f"test: {extract_path}/test/images\n")  
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")
    
print("YAML file created at: ", data_yaml_path)

# device check and selection
device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
    print(f"CUDA is available on device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device_name = "mps"
    print("MPS is available (Apple Silicon GPU)")
else:
    print("No GPU available. Using CPU instead.")

# select input model
# default download from repo if model_id or model_name is not provided
input_model_path = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_variant}.pt"
if model_id:        # get the specific model
    server_model = Model(model_id=model_id)    
    input_model_path = server_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded model name: {server_model.name} id:{server_model.id} to: {input_model_path}")
elif model_name:    # get the latest from Model Registry
    server_models = Model.query_models(project_name=project_name, model_name=model_name, only_published=True)
    if server_models:
        server_model = server_models[0]
        input_model_path = server_model.get_local_copy(raise_on_error=True)
        print(f"Downloaded model name: {server_model.name} id:{server_model.id} to: {input_model_path}")        
    else:
        print (f"No registered model found with name '{model_name}'. Using {model_variant} base model from Ultralytics.")

# training input params: hyp + other data
train_args = hyperparameters.copy() # copy to prevent altering original by reference
train_args["data"] = str(data_yaml_path)
train_args["name"] = model_variant
train_args["device"] = device_name
train_args["project"] = str(working_dir)

task.connect(train_args)

print(f"Loading {model_variant} model from {input_model_path}")
model = YOLO(input_model_path)

print(f"Training {model_variant} model using {device_name}.")
results = model.train(**train_args)

# upload results reference for report analysis 
task.upload_artifact(name=f"{model_variant}_hyp_config", artifact_object=hyperparameters)
result_file = working_dir / model_variant / "results.csv"
task.upload_artifact(name="results", artifact_object=result_file)
task.flush() 

# output info
output_model_id = task.models.output[-1].id
output_model_name = task.models.output[-1].name
task.set_parameter("output_model_project", project_name)
task.set_parameter("output_model_id", output_model_id)
task.set_parameter("output_model_name", output_model_name)
task.set_parameter("output_model_variant", model_variant)

# # shutil.rmtree(working_dir) # clean up output temp files

task.mark_completed(status_message=f"Completed training {model_variant} output:{output_model_name} id:{output_model_id}")
