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

NOTE: this is for training only, model evaluation task with compare and register the best model.
"""

# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"
Task.add_requirements('numpy', '==1.26.4')
Task.add_requirements('pytorch', '')

task = Task.init(project_name=project_name, 
                task_name="Model Training", 
                task_type=Task.TaskTypes.training)

params = {
    'dataset_id': '',                       # specific version of the dataset
    'dataset_name': 'dataset',              # latest dataset
    'model_id': '',                         # specific version of the model artifact id
    'model_name': '',                       # latest model artifact name
    'model_variant': 'yolo11n',             # base model variant from ultralytics if no model artifact given
    'hyperparameters':  {}                  # dictionaryo of hyperparameters
}

# TODO: check params and download model from ClearML

# logger = task.get_logger()
task.connect(params)
# task.execute_remotely(queue_name="training")


"""
Prepare dataset.
"""

dataset_id = params['dataset_id']
dataset_name = params['dataset_name']
model_id = params['model_id']
model_name = params['model_name']
model_variant = params["model_variant"]

if dataset_id: # get specific dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, dataset_project=project_name, alias=model_variant)
elif dataset_name: # get the latest
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, alias=model_variant)
else:
    raise ValueError("Missing dataset. Please provide dataset_id or dataset_name.")

extract_path = server_dataset.get_local_copy()

print("Downloaded dataset to: ", extract_path)


# """
# Model training.
# """

# use temp director for output before uploading to ClearML
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

# train the model
device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
    print(f"CUDA is available on device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device_name = "mps"
    print("MPS is available (Apple Silicon GPU)")
else:
    print("No GPU available. Using CPU instead.")
    
# get the previous trained model. If not found, download the original YOLO pretrained variants.
if model_id:        # get the specific model
    server_model = Model(model_id=model_id)
elif model_name:    # get the latest from Model Registry
    model = Model.query_models(project_name=project_name, model_name=model_name, only_published=True)
elif model_variant: # download base model from Ultralytics repo
    model = YOLO(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_variant}.pt")
else:
    raise ValueError("Missing model. Please provide model_artifact_id, model_artifact_name, or model_variant")

print(f"Training {model_variant} model using {device_name}.")

hyperparameters = params["hyperparameters"]

# temporary use file for testing
if not hyperparameters: # use default config from github repo 
    variant_train_config = f"./{model_variant}_hyp_config.yaml"
    if not os.path.exists(variant_train_config):    
        with open(variant_train_config, "r") as file:
            hyperparameters = yaml.safe_load(file)

hyperparameters["data"] = str(data_yaml_path)
hyperparameters["name"] = model_variant
hyperparameters["device"] = device_name
hyperparameters["project"] = str(working_dir)

task.connect(hyperparameters)

results = model.train(**hyperparameters)

# upload results for report analysis reference
result_file = working_dir / model_variant / "results.csv"
task.upload_artifact(name="results", artifact_object=result_file)
task.flush() 

task.set_parameter("output_model_project", project_name)
task.set_parameter("output_model_id", task.models.output[-1].id)
task.set_parameter("output_model_name", task.models.output[-1].name)
task.set_parameter("output_model_variant", model_variant)

# # shutil.rmtree(working_dir) # clean up output temp files

task.mark_completed(status_message=f"Completed training {model_variant} with model output '{task.get_parameter("output_model_name")} ({task.get_parameter("output_model_id")})'.")
