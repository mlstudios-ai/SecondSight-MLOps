import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import shutil
from clearml import Task, Dataset, StorageManager
from enigmaai.config import Project, ConfigFactory

"""
Upload zipped YOLO dataset file from remote URL, extract and upload to ClearML server. 
The zipped file needs to contain the YAML file and assets in the following structure:

data.yaml
images/
labels/
"""

project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Upload Base Dataset", 
                task_type=Task.TaskTypes.data_processing)

params = {
    'dataset_url': '',                      # url of a zip file to download from
    'output_dataset_name': 'base_dataset',    # name for output dataset to upload
}

task.connect(params)
task.execute_remotely(queue_name="default")
task_params = task.get_parameters()
print("dataset_base_upload params=", task_params)

dataset_url = task_params['General/dataset_url']
dataset_name = task_params['General/output_dataset_name']

# validate task input params
if not dataset_url:
    task.mark_completed(status_message="No dataset URL provided. Nothing to upload.")
    exit(0)
    
# Mandatory input param if dataset_url is not empty
if not dataset_name:
    raise ValueError("Missing a dataset name to upload to. Please provide output_dataset_name.")

# download zip dataset from remote url and extract to local disk
StorageManager.set_cache_file_limit(project.get("storage-cache-limit"))
dataset_path = StorageManager.get_local_copy(remote_url=dataset_url,
                                                name=dataset_name,
                                                extract_archive=True,
                                                cache_context=dataset_name,
                                                force_download=True)

if dataset_path is None:
    # Error: Assume file not found (404 http status code)
    raise FileNotFoundError("404", f"Found not found at URL {dataset_url}") 

print("Downloaded to: ", dataset_path)

# upload dataset to ClearML server
dataset = Dataset.create(
    dataset_project=project_name, dataset_name=dataset_name
)

dataset.add_files(path=dataset_path)

print('Uploading base dataset in the background')

dataset.upload()
dataset.finalize()

task.flush()
if os.path.exists(dataset_path): 
        shutil.rmtree(dataset_path) # clean up
        
task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)

# TODO: log data visualisation