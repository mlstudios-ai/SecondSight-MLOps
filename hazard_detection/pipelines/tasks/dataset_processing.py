import zipfile
import tempfile
import pickle
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset, StorageManager

"""
Extract files from zipped YOLO dataset artifact and upload as Dataset for training, tracking, and analysis. 
Options to use remote URL link to the zipped file.
The dataset should be in the following structure:

data.yaml
images/
labels/

The extracted dataset will be in train/test ration using the 'test' param setting.
"""

task = Task.init(project_name="Hazad Detection", 
                task_name="Extract and Split Dataset", 
                task_type=Task.TaskTypes.data_processing, 
                reuse_last_task_id=True)
args = {
    'dataset_task_id': '',
    'dataset_url': '',
    'random_state': 42,
    'test_size': 0.2,
}

task.connect(args)

task.execute_remotely(queue_name="default")

if args['dataset_task_id']: # download from ClearML Server   
    dataset_upload_task = Task.get_task(task_id=args['dataset_task_id'])
    zip_dataset = dataset_upload_task.artifacts['dataset'].get_local_copy()
    
    # extract zip artifact into temp
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_dataset, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
    dataset_path = temp_dir
    
elif args['dataset_url']: # download from remote URL
    dataset_path = StorageManager.get_local_copy(remote_url=args['dataset_url'], extract_archive=True)
    if dataset_path is None:
        raise FileNotFoundError("404", f"Found not found at URL {args['dataset_url']}") 
    
else: # link not provided
     raise ValueError("Missing dataset link")

 dataset = pickle.load(open(dataset_path, 'rb'))

# "process" data
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args['test_size'], random_state=args['random_state'])

# upload processed data
print('Uploading process dataset')
task.upload_artifact('X_train', X_train)
task.upload_artifact('X_test', X_test)
task.upload_artifact('y_train', y_train)
task.upload_artifact('y_test', y_test)


dataset = Dataset.create(
    dataset_project="Hazard Detection", dataset_name="hd_yolov8_dataset"
)

# TODO: Extract file
dataset.add_files(path=hazard_dataset)

dataset.upload()
dataset.finalize()
