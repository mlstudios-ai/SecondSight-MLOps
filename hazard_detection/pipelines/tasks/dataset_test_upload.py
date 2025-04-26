import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from clearml import Task, Dataset, StorageManager
from enigmaai.config import Project, Config, ConfigFactory

"""
Upload zipped YOLO test dataset file from remote URL, extract and upload to ClearML server. 
The zipped file needs to contain the YAML file and assets in the following structure:

data.yaml
images/
labels/

This dataset is used for evaluate best model in the Model Evaluation task.
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Upload Evaluation Dataset", 
                task_type=Task.TaskTypes.data_processing,
                reuse_last_task_id=True)

params = {
    'dataset_url': ''    
}

task.connect(params)
task.execute_remotely(queue_name="default")

dataset_name = "test_dataset"
dataset_url = params['dataset_url']

# validate task input params
if not dataset_url:
    task.mark_completed(status_message="No dataset URL provided. Nothing to upload.")
    exit(0)

# download zip dataset from remote url and extract to local disk
dataset_path = StorageManager.get_local_copy(remote_url=dataset_url,
                                                name=dataset_name,
                                                cache_context="dataset",
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

print('Uploading test dataset in the background')

dataset.upload()
dataset.finalize()

task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)

# TODO: log data visualisation