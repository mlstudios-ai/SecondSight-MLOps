from clearml import Task, Dataset
from pathlib import Path
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
"""

# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"

task = Task.init(project_name=project_name, 
                task_name="Model Training", 
                task_type=Task.TaskTypes.training)

params = {
    'dataset_id': '',                       # specific version of the dataset
    'dataset_name': 'dataset',              # latest dataset
    'model_artifact_id': '',                # specific version of the model
    'model_artifact_name': '',              # latest model
    'model_variant': 'yolo11n',             # base model variant from ultralytics if no model artifact given
}

# TODO: check params and download model from ClearML

# logger = task.get_logger()
task.add_requirements('numpy', '==1.26.4')
task.connect(params)
task.execute_remotely(queue_name="training")


"""
Prepare dataset.
"""

# download dataset from ClearML server
dataset_id = params['dataset_id']
dataset_name = params['dataset_name']
model_variant = params["model_variant"]

if dataset_id: # get specific dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, dataset_project=project_name, alias=model_variant)
elif dataset_name: # get the latest
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, alias=model_variant)
else:
    raise ValueError("Missing param dataset_id and dataset_name. Provide at least one.")

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

model = YOLO(f"https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/{model_variant}.pt")

print(f"Training {model_variant} model using {device_name}.")

args = dict(data=data_yaml_path, epochs=2, project=working_dir, device=device_name)
task.connect(args)

# # Step 5: Initiating Model Training
results = model.train(**args)

# model_saved_path = working_dir / model_variant / "train/weights/best.pt"
# task.upload_artifact(name=model_variant, artifact_object=model_saved_path)
# task.flush() 

# shutil.rmtree(working_dir) # clean up temp output

# results = model.train(
#     data=str(data_yaml_path),
#     epochs=2,
#     imgsz=640,
#     batch=8,
#     lr0=0.0003,
#     warmup_epochs=3,
#     device=device_name,
#     name=model_variant,
#     project=str(working_dir),   # custom output path
#     patience=10,
#     verbose=True,
#     plots=True,
#     augment=True,
#     mosaic=0,
#     mixup=0,
#     degrees=5,
#     translate=0.05,
#     scale=0.1,
#     shear=0.0,
#     hsv_h=0.005,             # lower color jitter
#     hsv_s=0.3,
#     hsv_v=0.2
#     )

# model_saved_path = working_dir / model_variant / "train/weights/best.pt"
# task.upload_artifact(name=model_variant, artifact_object=model_saved_path)
# task.flush() 

# all uploaded finised, clean up temp output dir
# shutil.rmtree(working_dir) # clean up temp output
