import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset, StorageManager
from enigmaai.config import Project, Config, ConfigFactory

def clean_dataset_file_stems(images_path, labels_path):
    """
    Removed unlabelled images and labels without images.
    
    Args:
        images_dir (Path): Path to the folder containing YOLO image files.
        labels_dir (Path): Path to the folder containing YOLO label (.txt) files.
    
    Returns:
        [str]: Valid image file name prefix that has corresponding labels
    """
    images_dir = Path(images_path)
    labels_dir = Path(labels_path)

    # Get image and label stems
    image_files = {f.stem: f for f in images_dir.glob("*.[jp][pn]g")}
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}

    # Calculate mismatches
    image_stems = set(image_files.keys())
    label_stems = set(label_files.keys())

    missing_labels = image_stems - label_stems

    if missing_labels:
        image_stems -= missing_labels
        print(f"Missing labels for the following images: {missing_labels}")
    else:
        print("All images have corresponding labels.")
    
    return list(image_stems)
    
"""
Split dataset into train, val, test. Base dataset can be downloaded from ClearML server 
or remote URL ZIP file according to task parameter values. 

The dataset should be in the following structure:

data.yaml
images/
labels/
"""

project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Split Base Dataset", 
                task_type=Task.TaskTypes.data_processing)

params = {
    'base_dataset_id': '',
    'base_dataset_name': '',
    'base_dataset_url': '',
    'random_state': 42,
    'val_size': 0.3,
    'test_size': 0.0,
}

# logger = task.get_logger()
task.connect(params)
task.execute_remotely(queue_name="default")

dataset_name = "dataset" # name for uploading the output dataset

# download dataset 
base_dataset_id = params['base_dataset_id']
base_dataset_name = params['base_dataset_name']
base_dataset_url = params['base_dataset_url']

# validate task input params
if not base_dataset_id and not base_dataset_name and not base_dataset_url:
    task.mark_completed(status_message="No dataset provided. Nothing to process.")
    exit(0)

if base_dataset_id: 
    # download the specific dataset from ClearML Server   
    server_dataset = Dataset.get(dataset_id=base_dataset_id)
    extract_path = server_dataset.get_local_copy()
    print(f"Downloaded dataset name:{server_dataset.name} id:{server_dataset.id} to: {extract_path}")
elif base_dataset_name: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=base_dataset_name, dataset_project=project_name, only_completed=True)
    extract_path = server_dataset.get_local_copy()          
    print(f"Downloaded dataset name:{server_dataset.name} id:{server_dataset.id} to: {extract_path}")
elif base_dataset_url: 
    # download from remote URL
    extract_path = StorageManager.get_local_copy(remote_url=base_dataset_url,
                                                 name="base_dataset",
                                                 cache_context="hd",
                                                 force_download=True)    
    if extract_path is None:
        raise FileNotFoundError("404", f"Found not found at URL {base_dataset_url}")    
    print(f"Downloaded dataset from:{base_dataset_url} to: {extract_path}") 
else: # link not provided
     raise ValueError("Missing param dataset_url")

extract_path = Path(extract_path)

"""
Split dataset to train, val, test
"""

# get image file prefix that has corresponding labels
clean_file_stems = clean_dataset_file_stems(extract_path / "images", extract_path / "labels")
print("clean_file_stems:", len(clean_file_stems))

# split sizes
val_size = float(params['val_size'])
test_size = float(params['test_size'])
random_state = int(params['random_state'])

# split train, val, test sets according task params
train_stems, val_stems = train_test_split(clean_file_stems, 
                                         test_size = val_size + test_size, 
                                         random_state=random_state)

if test_size > 0:
    val_stems, test_stems = train_test_split(val_stems, 
                                            test_size=test_size, 
                                            random_state=random_state) 
else:
    test_stems = []

"""
Move files to train, val, test folders
"""

# base destination path for split dataset
dest_path = Path(extract_path / dataset_name)
if os.path.exists(dest_path): 
        shutil.rmtree(dest_path) # remove old data
        
os.makedirs(dest_path)

# train set
train_path = dest_path / "train"
train_images = train_path / "images"
os.makedirs(train_images, exist_ok=True)
train_labels = train_path / "labels"
os.makedirs(train_labels, exist_ok=True)

for stem in train_stems:
    image = f"{stem}.jpg" # TODO: make it geneneric to other image formats
    label = f"{stem}.txt"
    shutil.move(extract_path / f"images/{image}", train_images / image)
    shutil.move(extract_path / f"labels/{label}", train_labels / label)
     
# validation set
val_path = dest_path / "val"
val_images = val_path / "images"
os.makedirs(val_images, exist_ok=True)
val_labels = val_path / "labels"
os.makedirs(val_labels, exist_ok=True)

for stem in val_stems:
    image = f"{stem}.jpg"
    label = f"{stem}.txt"
    shutil.move(extract_path / f"images/{image}", val_images / image)
    shutil.move(extract_path / f"labels/{label}", val_labels / label)

# test set
test_path = dest_path / "test"
test_images = test_path / "images"
os.makedirs(test_images, exist_ok=True)
test_labels = test_path / "labels"
os.makedirs(test_labels, exist_ok=True)

for stem in test_stems:
    image = f"{stem}.jpg"
    label = f"{stem}.txt"
    shutil.move(extract_path / f"images/{image}", test_images / image)
    shutil.move(extract_path / f"labels/{label}", test_labels / label)

# contruct YAML config file
data_yaml_path = dest_path / 'data.yaml'
classes = ['hole', 'pole', 'stairs', 'bottle', 'rock']
with open(data_yaml_path, 'w') as f:
    f.write(f"train: ./{train_path.name}/images\n")
    f.write(f"val: ./{val_path.name}/images\n")
    f.write(f"test: ./{test_path.name}/images\n")  
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

# upload dataset to ClearML server
dataset = Dataset.create(
    dataset_project=project_name, dataset_name=dataset_name
)

dataset.add_files(path=dest_path)

print('Uploading dataset in the background')

dataset.upload()
dataset.finalize()

print('Done')
print("output_dataset_project", dataset.project)
print("output_dataset_id", dataset.id)
print("output_dataset_name", dataset.name)

task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)

# TODO: log data visualisation

