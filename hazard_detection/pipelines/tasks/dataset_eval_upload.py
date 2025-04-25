from clearml import Task, Dataset, StorageManager
# from enigmaai.config import Project, Config, ConfigFactory

"""
Upload zipped YOLO test dataset file from remote URL, extract and upload to ClearML server. 
The zipped file needs to contain the YAML file and assets in the following structure:

images/
labels/
"""

# NOT WORKING: setup.py not running on execute_remotely, hence can not import enigmaai package
# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"

task = Task.init(project_name=project_name, 
                task_name="Upload Evaluation Dataset", 
                task_type=Task.TaskTypes.data_processing,
                reuse_last_task_id=True)

params = {
    # 'dataset_url': ''    
    'dataset_url': 'https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/test_dataset.zip'
}

task.connect(params)
task.execute_remotely(queue_name="default")

dataset_url = params['dataset_url']

# validate task input params
if not dataset_url:
    task.mark_completed(status_message="No dataset URL provided. Nothing to upload.")
    exit(0)

# download zip dataset from remote url and extract to local disk
dataset_path = StorageManager.get_local_copy(remote_url=dataset_url,
                                                name="test_dataset",
                                                cache_context="dataset",
                                                force_download=True)

if dataset_path is None:
    # Error: Assume file not found (404 http status code)
    raise FileNotFoundError("404", f"Found not found at URL {dataset_url}") 

print("Downloaded to: ", dataset_path)

# upload dataset to ClearML server
dataset = Dataset.create(
    dataset_project=project_name, dataset_name="test_dataset"
)

dataset.add_files(path=dataset_path)

print('Uploading test dataset in the background')

dataset.upload()
dataset.finalize()

task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)

# TODO: log data visualisation