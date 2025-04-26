import os
from pathlib import Path
import yaml
from clearml import Task
from clearml.automation import PipelineController

# from sys import platform
# if  platform == "darwin": # MacOSX MPS platform dependencies for torchvision
#     print("Detected MacOSX platform.")
#     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

"""


"""

project_name = "Detection"

# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(name="Train YOLOv11 Model", 
                          project=project_name, 
                          add_pipeline_tags=False)


pipe.set_default_execution_queue("default")

""" 
STEP 1: Load initial dataset
"""

# intial dataset to download. If none provided, task will complete without upload
pipe.add_parameter(
    "base_dataset_url",
    # "https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip",
    # "https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/dev_dataset.zip",
    "",
    "(Optional) URL to the final dataset."
)

def pre_upload_callback(pipeline, node, param_override) -> bool:
    # if no dataset url provided, there will be no output databset_id.
    # assign it to empty so the task can exit safely and allow the pipeline to continue
    if not param_override["General/dataset_url"]:
        node.job.task.set_parameter("output_dataset_id", "")
        print("No output_dataset_id, assigned to empty string")
    
    print("Cloning Task id={} with parameters: {}".format(
        node.base_task_id, param_override))
    
    return

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
pipe.add_parameter("base_dataset_id", "", "(Optional) Will be updated after previous upload task if dataset_url is set") 
pipe.add_parameter("base_dataset_name", "dataset", "(Optional) latest registered dataset.")
pipe.add_parameter("random_state", 42, "Specify random state for consistent training")
pipe.add_parameter("val", 0.30, "Validation split. Percentage of entire dataset.")
pipe.add_parameter("test", 0.0, "Test split. Percentage of entire dataset.")

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
    base_task_name="Split Dataset",
    parameter_override={
        "General/base_dataset_id": "${load_base_dataset.parameters.General/output_dataset_id}",
        "General/base_dataset_name": pipe.get_parameters()["base_dataset_name"],
        "General/random_state": pipe.get_parameters()["random_state"],
        "General/val_size": pipe.get_parameters()["val"],
        "General/test_size": pipe.get_parameters()["test"],
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback,
)

""" 
STEP 3: Model training
"""
   
def load_hyp_config(model_variant) -> dict:
    """ 
    Load hyperparameters from a config file. Filename format {model_variant}_hyp_config.yaml.

    Args:
        model_variant (_type_): variant of the model: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

    Returns:
        dict: Hyperparameters.
    """
    hyp_config_file = f"{model_variant}_hyp_config.yaml"
    hyp_config_path = Path(__file__).parent / hyp_config_file
    print("hyp_config_path=", hyp_config_path.resolve())
    if hyp_config_path.exists():    
        with open(hyp_config_path, "r") as file:
            hyperparameters = yaml.safe_load(file)
    
    return hyperparameters

# model training settings
pipe.add_parameter("model_variant", "yolo11n", "YOLOv11 model variant to train. Saved as model_name.")
hyps_data = load_hyp_config(pipe.get_parameters()["model_variant"])
hyps = yaml.dump(hyps_data) # convert to string to easy passing parameter values
pipe.add_parameter("model_hyps", hyps, "Dictionary of YOLO.train() input params. Defaults from model variant config file")

def pre_training_callback(pipeline, node, param_override) -> bool:     
    print("Cloning Task id={} with parameters: {}".format(
        node.base_task_id, param_override))
    
    # param validation check
    model_variant = param_override["General/model_variant"]
    if not model_variant:
        raise ValueError(f"Missing model_variant param value.")
    
    model_hyps = param_override["General/model_hyps"]
    if not model_hyps:
        raise ValueError(f"Missing param model_hyps value.")
    
    return True
            
def post_training_callback(pipeline, node) -> None:
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    
    return

# # TESTING
# pipe.add_step(
#     name="model_training",
#     # parents=["dataset_processing"],
#     base_task_project=project_name,
#     base_task_name="Model Training",
#     parameter_override={
#         "General/dataset_id": "f7ef54810a544aa0b2377cdf27f8c600",
#         "General/model_variant": "${pipeline.model_variant}",
#         "General/model_hyps": "${pipeline.model_hyps}",
#     },
#     pre_execute_callback=pre_training_callback,
#     post_execute_callback=post_training_callback,
# )

# # pipe.add_step(
# #     name="model_training",
# #     parents=["dataset_processing"],
# #     base_task_project=project_name,
# #     base_task_name="Model Training",
# #     parameter_override={
# #         "General/dataset_id": "${dataset_processing.parameters.General/output_dataset_id}",
# #         "General/model_variant": "${pipeline.model_variant}",
# #         "General/model_hyps": "${pipeline.model_hyps}",
# #     },
# #     pre_execute_callback=pre_training_callback,
# #     post_execute_callback=post_training_callback,
# # )

# # pipe.add_step(
# #     name="model_evaluation",
# #     parents=["model_training"],
# #     base_task_project=project_name,
# #     base_task_name="Model Evaluation",
# #     parameter_override={
# #         "General/test_dataset_name": "test_dataset",
# #         "General/draft_model_id": "${model_training.parameters.General/output_model_id}",
# #         "General/pub_model_name": "yolo11n",
# #     }
# # )

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
pipe.start_locally()

# Starting the pipeline (in the background)
# pipe.start()

print("done")