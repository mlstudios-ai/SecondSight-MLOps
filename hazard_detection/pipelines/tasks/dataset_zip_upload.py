import argparse
import os
from clearml import Task, Dataset, StorageManager

"""
Upload zipped YOLO dataset file from remote URL to ClearML server as an artifact. 
The zipped file needs to contain the YAML file and assets in the following structure:

data.yaml
images/
labels/
"""

task = Task.init(project_name="Hazard Detection", 
                task_name="Upload ZIP Dataset", 
                task_type=Task.TaskTypes.data_processing)

params = {
    'dataset_url': 'https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip'
}

task.connect(params)

print("params=", params)


task.execute_remotely(queue_name="default")

dataset_url = params['dataset_url']

if not dataset_url:
    raise ValueError("No dataset URL provided!")

hazard_dataset = StorageManager.get_local_copy(remote_url=dataset_url)

print("Downloading to: ", hazard_dataset)

if hazard_dataset is None:
    # StorageManage can not find or access the file, assume file not 
    # found (404 http status code)
    raise FileNotFoundError("404", f"Found not found at URL {dataset_url}") 

# task.upload_artifact('dataset', artifact_object=hazard_dataset)
dataset = Dataset.create(
    dataset_project="Hazard Detection", dataset_name="dataset_mini_zip"
)

dataset.add_files(path=hazard_dataset)

print('uploading artifacts in the background')

# we are done
print('Done')
