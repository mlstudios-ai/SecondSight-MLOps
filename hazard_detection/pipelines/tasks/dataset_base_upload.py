import argparse
import os
import sys
from clearml import Task, Dataset, StorageManager
from enigmaai.config import Project, Config, ConfigFactory

"""
Upload zipped YOLO dataset file from remote URL, extract and upload to ClearML server. 
The zipped file needs to contain the YAML file and assets in the following structure:

data.yaml
images/
labels/
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Upload Base Dataset", 
                task_type=Task.TaskTypes.data_processing,
                reuse_last_task_id=True)

params = {
    'dataset_url': 'https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip'
}

task.connect(params)
task.execute_remotely(queue_name="default")

dataset_url = params['dataset_url']

if not dataset_url:
    raise ValueError("Missing dataset url")

# download zip dataset from remote url and extract to local disk
hazard_dataset = StorageManager.get_local_copy(remote_url=dataset_url,
                                                name="base_dataset",
                                                cache_context="hd_zip_dataset",
                                                force_download=True)

if hazard_dataset is None:
    # Error: Assume file not found (404 http status code)
    raise FileNotFoundError("404", f"Found not found at URL {dataset_url}") 

print("Downloaded to: ", hazard_dataset)

# upload dataset to ClearML server
dataset = Dataset.create(
    dataset_project=project_name, dataset_name="base_dataset"
)

dataset.add_files(path=hazard_dataset)

print('Uploading dataset in the background')

dataset.upload()
dataset.finalize()

# TODO: log data visualisation