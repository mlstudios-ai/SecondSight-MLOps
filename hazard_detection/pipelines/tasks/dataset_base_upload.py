from clearml import Task, Dataset, StorageManager
# from enigmaai.config import Project, Config, ConfigFactory

"""
Upload zipped YOLO dataset file from remote URL, extract and upload to ClearML server. 
The zipped file needs to contain the YAML file and assets in the following structure:

data.yaml
images/
labels/
"""

# NOT WORKING: setup.py not running on execute_remotely, hence can not import enigmaai package
# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"

task = Task.init(project_name=project_name, 
                task_name="Upload Base Dataset", 
                task_type=Task.TaskTypes.data_processing,
                reuse_last_task_id=True)

params = {
    'dataset_url': ''
}

task.connect(params)
task.execute_remotely(queue_name="default")

dataset_url = params['dataset_url']

# validate task input params
if not dataset_url:
    task.mark_completed(status_message="No dataset URL provided. Nothing to upload.")
    exit(0)

# download zip dataset from remote url and extract to local disk
hazard_dataset = StorageManager.get_local_copy(remote_url=dataset_url,
                                                name="base_dataset",
                                                cache_context="zip_dataset",
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

task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)

# TODO: log data visualisation