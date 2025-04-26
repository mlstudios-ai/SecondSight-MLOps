import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pathlib import Path
import yaml
from clearml import Task
from clearml.automation import PipelineController
from enigmaai.config import Project, Config, ConfigFactory

"""


"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(name="Train YOLOv11 Model", 
                          project=project_name, 
                          add_pipeline_tags=False)


# pipe.set_default_execution_queue("default")

""" 
STEP 1: Load initial dataset
"""

# intial dataset to download. If none provided, task will complete without upload
# base_dataset_url = project.get("base-dataset-url")
base_dataset_url = ""
pipe.add_parameter("base_dataset_url", base_dataset_url, "(Optional) URL to the final dataset.")

def pre_upload_callback(pipeline, node, param_override) -> bool:    
    print("Cloning Task id={} with parameters: {}".format(node.base_task_id, param_override))
    
    return True

def post_upload_callback(pipeline, node) -> None:               
    print("Completed Task id={}".format(node.executed))
    
    return

pipe.add_step(
    name="load_base_dataset",
    base_task_project=project_name,
    base_task_name="Upload Base Dataset",
    parameter_override={"General/dataset_url": "${pipeline.base_dataset_url}"},
    pre_execute_callback=pre_upload_callback,
    post_execute_callback=post_upload_callback
)

""" 
STEP 2: Dataset processing
"""

# processing starting dataset for pipeline
# it will get dataset_id from step 1, if not provided, this will be used
pipe.add_parameter("base_dataset_id", "", "Overitten if previous task is not skipped.")
pipe.add_parameter("base_dataset_name", "", "Used only if base_dataset_id is empty.")
pipe.add_parameter("random_state", 42, "Specify random state for consistent training")
pipe.add_parameter("val", 0.30, "Validation split. Percentage of entire dataset.")

def pre_processing_callback(pipeline, node, param_override) -> bool:
    print("Cloning Task id={} with parameters: {}".format(
        node.base_task_id, param_override))
    
    return True

def post_processing_callback(pipeline, node) -> None:
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    
    return

pipe.add_step(
    name="dataset_processing",
    parents=["load_base_dataset"],
    base_task_project=project_name,
    base_task_name="Split Base Dataset",
    parameter_override={
        "General/base_dataset_id": (
            "${load_base_dataset.parameters.General/output_dataset_id}"
            if pipe.get_parameters()["base_dataset_url"] # url not provided, no base dataset upload
            else "${pipeline.base_dataset_id}"), 
        "General/base_dataset_name": "${pipeline.base_dataset_name}",
        "General/random_state": pipe.get_parameters()["random_state"],
        "General/val_size": pipe.get_parameters()["val"],
        "General/test_size": 0.0,
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback,
)

""" 
STEP 3: Model training
"""
   
def load_hyp_config(model_variant) -> dict:
    hyp_config_file = f"{model_variant}_hyp_config.yaml"
    hyp_config_path = Path(__file__).parent / hyp_config_file
    print("hyp_config_path=", hyp_config_path.resolve())
    if hyp_config_path.exists():    
        with open(hyp_config_path, "r") as file:
            hyperparameters = yaml.safe_load(file)
    
    return hyperparameters

# model training settings
pipe.add_parameter("train_dataset_id", "", "Overitten if previous task is not skipped.")
pipe.add_parameter("train_dataset_name", "dataset", "Used only if train_dataset_id is empty.")
pipe.add_parameter("model_id", "", "(Optional) Pre-trained mode. If not provided, use default based on model_variant")
pipe.add_parameter("model_variant", "yolo11n", "YOLOv11 model variant to train. Saved as model_name.")
pipe.add_parameter("model_hyps", "", "Dictionary of YOLO.train() input params. Defaults from model variant config file")

def pre_training_callback(pipeline, node, param_override) -> bool:   
    # param validation check
    model_variant = param_override["General/model_variant"]
    if not model_variant:
        raise ValueError(f"Missing model_variant param value.")
    
    # add default hyp config if none provided     
    if not param_override["General/model_hyps"]:
        hyps_data = load_hyp_config(model_variant)
        hyps = yaml.dump(hyps_data) 
        param_override["General/model_hyps"] = hyps
    
    print("Cloning Task id={} with parameters: {}".format(
        node.base_task_id, param_override))
    
    return True
            
def post_training_callback(pipeline, node) -> None:
    print("Completed Task id={}".format(node.executed))
    
    return

pipe.add_step(
    name="model_training",
    parents=["dataset_processing"],
    base_task_project=project_name,
    base_task_name="Model Training",
    parameter_override={
        "General/dataset_id": (
            "${dataset_processing.parameters.General/output_dataset_id}"
            if pipe.get_parameters()["base_dataset_url"] and pipe.get_parameters()["base_dataset_name"]
            else "${pipeline.train_dataset_id}"), # no base dataset download and no output    
        "General/dataset_name": "${pipeline.train_dataset_name}",     
        "General/model_id": "${pipeline.model_id}",     
        "General/model_variant": "${pipeline.model_variant}",
        "General/model_hyps": "${pipeline.model_hyps}",
    },
    pre_execute_callback=pre_training_callback,
    post_execute_callback=post_training_callback,
)

pipe.add_step(
    name="model_evaluation",
    parents=["model_training"],
    base_task_project=project_name,
    base_task_name="Model Evaluation",
    parameter_override={
        "General/test_dataset_name": "test_dataset",
        "General/draft_model_id": "${model_training.parameters.General/output_model_id}",
        "General/pub_model_name": "yolo11n",
    }
)

# pipe.add_step(
#     name="model_deployment",
#     parents=["model_evaluation"],
#     base_task_project=project_name,
#     base_task_name="Model Deployment",
#     parameter_override={
#         "General/dataset_id": "${dataset_processing.parameters.General/output_dataset_id}",
#         "General/model_variant": "${pipeline.model_variant}",
#         "General/model_hyps": "${pipeline.model_hyps}",
#     },
#     pre_execute_callback=pre_training_callback,
#     post_execute_callback=post_training_callback,
# )

# for debugging purposes use local jobs
pipe.start_locally(run_pipeline_steps_locally=True)

# Starting the pipeline (in the background)
# pipe.start()

print("done")