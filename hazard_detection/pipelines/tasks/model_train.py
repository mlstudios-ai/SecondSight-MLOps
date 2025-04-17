from clearml import Task, Dataset, StorageManager
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

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

task = Task.init(project_name="Hazard Detection", 
                task_name="Model Training", 
                task_type=Task.TaskTypes.data_processing,
                reuse_last_task_id=True)

params = {
    'dataset_id': '',                       # specific version of the dataset
    'dataset_name': 'dataset',              # latest dataset
    'model_artifact_id': '',                # specific version of the model
    'model_artifact_name': '',              # latest model
    'model_variant': 'yolo11s',             # base model variant from ultralytics if no model artifact given
    'local_working_directory': '/Users/jasper/Anna/Uni/UTS/MAI/Subjects/AIS/experiments/training/yolov11s/'  # full path local directory to train and save outputs 
}

# TODO: check params and download model from ClearML
# logger = task.get_logger()
task.connect(params)
task.execute_remotely(queue_name="training")

"""
Prepare datase.
"""

# download dataset from ClearML server
dataset_id = params['dataset_id']
dataset_name = params['dataset_name']

if dataset_id: # get specific dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, dataset_project="Hazard Detection")
elif dataset_name: # get the latest
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project="Hazard Detection")
else:
    raise ValueError("Missing param dataset_id and dataset_name. Provide at least one.")

extract_path = server_dataset.get_local_copy()

print("Downloaded dataset to: ", extract_path)


"""
Model training.
"""

# create local working directory
working_dir = params['local_working_directory']
if not working_dir:
    raise ValueError("Missing param local_working_directory.")

working_dir = Path(working_dir)
print("Trying to create working directory: ", working_dir)

if working_dir.exists(): # DO NOT overwrite existing dir
    raise FileExistsError(f"Directory already exists: {working_dir}")

working_dir.mkdir(parents=True)
print("Working directory created.")

# contruct YAML config file
data_yaml_path = working_dir / 'data.yaml'
classes = ['hole', 'pole', 'stairs', 'bottle', 'rock']
with open(data_yaml_path, 'w') as f:
    f.write(f"train: {extract_path}train/images\n")
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

model_variant = params["model_variant"]
model = YOLO(f"{model_variant}.pt")

# Step 3: Loading the YOLO11 Model

# Step 4: Setting Up Training Arguments
# task.connect(args)

# Step 5: Initiating Model Training
# results = model.train(**args)

print(f"Training {model_variant} model using {device_name}.")

results = model.train(
    data=str(data_yaml_path),
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.0003,
    warmup_epochs=3,
    device=device_name,
    name="cvat_v4_smoother",
    project=str(working_dir),   # custom output path
    patience=10,
    verbose=True,
    plots=True,
    augment=True,
    mosaic=0,
    mixup=0,
    degrees=5,
    translate=0.05,
    scale=0.1,
    shear=0.0,
    hsv_h=0.005,             # lower color jitter
    hsv_s=0.3,
    hsv_v=0.2
    )