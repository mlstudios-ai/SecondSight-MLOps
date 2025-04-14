import zipfile
import tempfile
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset, StorageManager

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
Extract files from zipped YOLO dataset artifact and upload as Dataset for training, tracking, and analysis. 
Options to use remote URL link to the zipped file.
The dataset should be in the following structure:

data.yaml
images/
labels/

The extracted dataset will be in train/val/test ration using the 'test' param setting.
"""

task = Task.init(project_name="Hazard Detection", 
                task_name="Extract ZIP and Upload Dataset", 
                task_type=Task.TaskTypes.data_preprocessing)

args = {
    'dataset_name': 'dataset_zip',
    'dataset_url': 'https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip',
    'random_state': 42,
    'val_size': 0.15,
    'test_size': 0.15,
}

logger = task.get_logger()
task.connect(args)
task.execute_remotely(queue_name="default")

# download dataset zip file and extract into temp dir
# if args['dataset_task_id']: # download from ClearML Server   
#     dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
#     zip_dataset = dataset_upload_task.artifacts['dataset'].get_local_copy()
if args['dataset_name']: # download from ClearML Server   
    server_dataset = Dataset.get(dataset_name=args['dataset_name'], dataset_project="Hazard Detection")
    zip_dataset = server_dataset.get_local_copy()
    
    # TODO: test if it clears cache copy 
    # TODO: configure custom cache dir
       
    # extract zip artifact into temp
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_dataset, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
    extract_path = temp_dir
    
elif args['dataset_url']: # download from remote URL
    extract_path = StorageManager.get_local_copy(remote_url=args['dataset_url'],
                                                 cache_context="hd_dataset",
                                                 force_download=True)
    if dataset_path is None:
        raise FileNotFoundError("404", f"Found not found at URL {args['dataset_url']}") 
    
else: # link not provided
     raise ValueError("Missing dataset link")

if extract_path is None:
    raise FileNotFoundError("404", f"Found not found at URL {args['dataset_url']}") 

extract_path += "/mini/"

print("Dataset extracted to: ", extract_path)

"""
Split dataset to train, val, test
"""
# get image file prefix that has corresponding labels
clean_file_stems = clean_dataset_file_stems(extract_path + "/images", extract_path + "/labels")

# split sizes
val_size = args['val_size']
test_size = args['test_size']
random_state = args['random_state']

print("clean_file_stems:", len(clean_file_stems))
# split train, val, test sets according task args
train_stems, remaining_stems = train_test_split(clean_file_stems, 
                                         test_size = val_size + test_size, 
                                         random_state=random_state)

val_stems, test_stems = train_test_split(remaining_stems, 
                                         test_size=test_size/(val_size + test_size), 
                                         random_state=random_state) 


"""
Move files to train, val, test folders
"""

# base dest path for split dataset
dest_path = Path(extract_path + "dataset")
os.makedirs(dest_path, exist_ok=True)

# train set
train_path = dest_path / "train"
train_images = train_path / "images"
os.makedirs(train_images, exist_ok=True)
train_labels = train_path / "labels"
os.makedirs(train_labels, exist_ok=True)

for stem in train_stems:
    image = f"{stem}.jpg" # TODO: make it geneneric to other image formats
    label = f"{stem}.txt"
    shutil.move(extract_path + f"images/{image}", train_images / image)
    shutil.move(extract_path + f"labels/{label}", train_labels / label)
     

# validation set
val_path = dest_path / "val"
val_images = val_path / "images"
os.makedirs(val_images, exist_ok=True)
val_labels = val_path / "labels"
os.makedirs(val_labels, exist_ok=True)

for stem in val_stems:
    image = f"{stem}.jpg"
    label = f"{stem}.txt"
    shutil.move(extract_path + f"images/{image}", val_images / image)
    shutil.move(extract_path + f"labels/{label}", val_labels / label)

# test set
test_path = dest_path / "test"
test_images = test_path / "images"
os.makedirs(test_images, exist_ok=True)
test_labels = test_path / "labels"
os.makedirs(test_labels, exist_ok=True)

for stem in test_stems:
    image = f"{stem}.jpg"
    label = f"{stem}.txt"
    shutil.move(extract_path + f"images/{image}", test_images / image)
    shutil.move(extract_path + f"labels/{label}", test_labels / label)

data_yaml_path = dest_path / 'data.yaml'
classes = ['hole', 'pole', 'stairs', 'bottle', 'rock']
with open(data_yaml_path, 'w') as f:
    f.write(f"train: {train_path}\n")
    f.write(f"val: {val_path}\n")
    f.write(f"test: {test_path}\n")  
    f.write(f"nc: {len(classes)}\n")
    f.write(f"names: {classes}\n")

dataset = Dataset.create(
    dataset_project="Hazard Detection", dataset_name="dataset"
)

# add all elements to dataset
dataset.add_files(path=dest_path)

dataset.upload()
dataset.finalize()
