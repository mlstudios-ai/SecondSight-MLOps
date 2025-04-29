import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
    
from clearml import Task, Dataset, Model
from pathlib import Path
import yaml
import tempfile
from ultralytics import YOLO
from enigmaai import util
from enigmaai.config import Project, ConfigFactory

"""
Train YOLOv11 model using the split dataset stored on ClearML server.
The dataset needs to be in the following structure:

data.yaml
train/images/*.jpg
train/labels/*.txt
val/images/*.jpg
val/labels/*.txt
test/images/
test/labels/

NOTE: The dataset will be downloaded to a cached directory and will NOT be copied into 
local_working_directory. This is to preserve the built-in caching mechanism of 
Dataset.get_local_copy(). Caching keep tracks of changes in datdaset and only download if 
there is a new version to prevent repeat downloads, especially with a large dataset.

NOTE: this is for training only and will only use train and val dataset. A sepearate 
test dataset on the server will be used for model evaluation task to compare and register 
the best model for deployment.
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Training", 
                task_type=Task.TaskTypes.training)

"""
One of dataset_id or dataset_name must be provided to load dataset
model_id is optional, if not provided, use model_variant to load latest publised model
"""
params = {
    'dataset_id': '',               # specific version of the dataset. if provided, ignore dataset_name
    'dataset_name': '',             # latest registered dataset. used if dataset_id is empty
    'model_id': '',                 # load specific version of the model 
    'model_variant': '',            # base model variant from ultralytics. if model_id is empty, also used to load latest version
    'model_hyps': ''                # string format of dictionary of hyperparameters for YOLO.train()
}

task.connect(params)
task.execute_remotely(queue_name="training")
task_params = task.get_parameters()
print("model_train params=", task_params)

dataset_id = task_params['General/dataset_id']
dataset_name = task_params['General/dataset_name']
model_id = task_params['General/model_id']
model_variant = task_params["General/model_variant"]
model_name = model_variant
model_hyps_str = task_params["General/model_hyps"]
        
# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to train on.")
    exit(0)
    
# Mandatory input param
if not model_variant:
    raise ValueError("Missing model variant. Please provide model_variant.")

# Mandatory input param
if not model_hyps_str:
    raise ValueError("Missing hyperparameters. Please provide model_hyps for YOLO.train().")

model_hyps = yaml.safe_load(model_hyps_str)

"""
Prepare dataset.
"""

if dataset_id:  # get specific dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, alias=model_variant)
elif dataset_name: # get the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, 
                                 only_completed=True,  alias=model_variant)

extract_path = server_dataset.get_local_copy()
print(f"Downloaded dataset name: {server_dataset.name} id:{server_dataset.id} to: {extract_path}")

"""
Model training.
"""

# use temp directory for output
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

device_name = util.get_device_name()

# default download from repo
input_model_path = project.get("base-model-url").format(model_variant)

# select input model
if model_id:        # get the specific model
    server_model = Model(model_id=model_id)    
    input_model_path = server_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded model name: {server_model.name} id:{server_model.id} to: {input_model_path}")
elif model_name:    # get the latest from Model Registry, if not found, use default ultralytics base model
    server_models = Model.query_models(model_name=model_name, only_published=True)
    if server_models:
        server_model = server_models[0]
        input_model_path = server_model.get_local_copy(raise_on_error=True)
        print(f"Downloaded model name: {server_model.name} id:{server_model.id} to: {input_model_path}")        
    else:
        print (f"No registered model found with name '{model_name}'. Using {model_variant} base model from Ultralytics.")

# training input params: hyp + other data
train_args = model_hyps.copy() # copy to prevent altering original by reference
train_args["data"] = str(data_yaml_path)
train_args["name"] = model_variant
train_args["device"] = device_name
train_args["project"] = str(working_dir)

task.connect(train_args)

print(f"Training {model_variant} model using {device_name}.")
model = YOLO(input_model_path)
results = model.train(**train_args)

# upload results reference for report analysis 
task.upload_artifact(name=f"{model_variant}_hyp_config", artifact_object=model_hyps)
result_file = working_dir / model_variant / "results.csv"
task.upload_artifact(name="results", artifact_object=result_file)
result_model = working_dir / model_variant / "weights" / "best.pt"
task.upload_artifact(name=f"{model_variant}", artifact_object=result_model)
        
# output info
output_model = task.models.output[0] # get the most recent model. there should be only 1
task.set_parameter("output_model_project", project_name)
task.set_parameter("output_model_id", output_model.id)
task.set_parameter("output_model_name", output_model.name)
task.set_parameter("output_model_variant", model_variant)