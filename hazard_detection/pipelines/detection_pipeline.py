import sys
import os
sys.path.remove("/Users/jasper/Anna/Uni/UTS/MAI/Subjects/AIS/project/SecondSight-API/api/src")
sys.path.remove("/Users/jasper/Anna/Uni/UTS/MAI/Subjects/AIS/project/EnigmaAI/hazard_detection/pipelines")
# print("-----")
# print(sys.path)
# print("-----")

print(sys.path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pathlib import Path
import yaml
from clearml import Task
from clearml.automation import PipelineController
from enigmaai.config import Project, ConfigFactory

"""
YOLOv11 model end-to-end MLOps pipeline. The pipeline is designed to cater for flexibilities of different needs at 
different point of the pipeline. 

Sometimes user may want to execute a task for specific purposes and then continue with the pipeline. Some steps can 
be skipped if the minimum parameters are not provided as specified in the pipeline parameter descriptions (refer to 
the corresponding tasks for more info).

Pipeline parameter settings can be set in each new run for various purposes as follow:

1. End-to-end from downloading base dataset from remote URL to model publising.
2. Skip step 1 URL download, use existing base datatset, and start from step 2: dataset processing
3. Skip steps 1 & 2, use processed dataset and start from step 3: model training
4. Skip steps 1, 2, & 3, use existing model as the new model for evaluation, starts from step 4: model evaluation

The above scenarios are designed to reduce execution time, resources requirements, duplications, and allows adjustment 
to various circumstances. Note that you can not skip model evaluation - this leads to publishing the model. If this is 
not a desire behaviour, use the task Model Evaluation from the WebUI instead of the pipeline.

IMPORTANT: by default, it will use the base_dataset and eval_dataset existing on the server, presuming they are already 
uploaded. If those datasets are not uploaded, please put in the base_dataset_url and/or eval_dataset_url accordingly.
Alternatively, before running the pipeline with default settings, upload the dataset using the following tasks from
the ClearML WebUI:

Upload Base Dataset - upload base dataset. This will trigger default pipeline to run in CD phase (NOT IMPLEMENTED)
Upload Evaluation Dataset - upload base dataset. This will trigger default pipeline to run in CD phase (NOT IMPLEMENTED)
"""

# get project configurations
project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')
pipeline_name = "YOLOv11 Pipeline"

# Connecting ClearML with the current pipeline, from here on everything is logged automatically
pipe = PipelineController(name=pipeline_name, 
                          project=project_name, 
                          add_pipeline_tags=False)

pipe.set_default_execution_queue("default")

""" 
STEP 1.1: Load base dataset
"""

# intial dataset to download. If none provided, task will complete without upload
# base_dataset_url = ""       # default for no uploading
# base_dataset_name = ""      # default for no preprocessing if base_dataset_url is also empty
base_dataset_url = project.get("base-dataset-url")
base_dataset_name = "base_dataset"
# base_dataset_url = "/Users/jasper/Anna/Uni/UTS/MAI/Subjects/AIS/project/Datasets/mini"
# base_dataset_name = "base_update_dataset"
pipe.add_parameter("base_dataset_url", base_dataset_url, "(Optional) URL to the final dataset.")
pipe.add_parameter("base_dataset_name", base_dataset_name, "Name of the dataset to upload to the server. Also used for the next step.")

def pre_base_upload_callback(pipeline, node, param_override) -> bool:    
    print("Cloning upload_base_dataset id={}".format(node.base_task_id))    
    return True

def post_base_upload_callback(pipeline, node) -> None:   
    print("Completed upload_base_dataset id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="upload_base_dataset",
    base_task_project=project_name,
    base_task_name="Upload Base Dataset",
    parameter_override={
        "General/dataset_url": "${pipeline.base_dataset_url}",
        "General/output_dataset_name": "${pipeline.base_dataset_name}"
        },
    pre_execute_callback=pre_base_upload_callback,
    post_execute_callback=post_base_upload_callback
)

""" 
STEP 2: Dataset processing
"""

# processing starting dataset for pipeline
# it will get dataset_id from step 1, if not provided, this will be used
split_dataset_name = "dataset"  # default to use the latest preprocessed dataset
# split_dataset_name = "update_dataset"
pipe.add_parameter("base_dataset_id", "", "(Optional) Overitten if previous task is not skipped. If empty, use the lastest of base_dataset_name")
pipe.add_parameter("split_random_state", 42, "Specify random state for consistent training")
pipe.add_parameter("split_val_size", 0.20, "Validation split. Percentage of entire dataset.")
pipe.add_parameter("split_dataset_name", split_dataset_name, "Name of the dataset to uppload the outout to the server. Also used for the next step.")

def pre_processing_callback(pipeline, node, param_override) -> bool:
    print("Cloning dataset_processing id={}".format(node.base_task_id))    
    return True

def post_processing_callback(pipeline, node) -> None:
    print("Completed dataset_processing id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="dataset_processing",
    parents=["upload_base_dataset"],
    base_task_project=project_name,
    base_task_name="Split Base Dataset",
    parameter_override={
        "General/base_dataset_id": (
            "${upload_base_dataset.parameters.General/output_dataset_id}"
            if pipe.get_parameters()["base_dataset_url"] # url not provided, no base dataset upload
            else "${pipeline.base_dataset_id}"), 
        "General/base_dataset_name": "${pipeline.base_dataset_name}",
        "General/output_dataset_name": pipe.get_parameters()["split_dataset_name"],
        "General/random_state": pipe.get_parameters()["split_random_state"],
        "General/val_size": pipe.get_parameters()["split_val_size"],
        "General/test_size": 0.0
    },
    pre_execute_callback=pre_processing_callback,
    post_execute_callback=post_processing_callback
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
pipe.add_parameter("model_dataset_id", "", "(Optional) Overitten if previous task is not skipped. If set, ignore model_dataset_name")
pipe.add_parameter("model_id", "", "(Optional) Pre-trained model from the server. If not provided, use default based on model_name")
pipe.add_parameter("model_name", "yolo11n", "(Optional) Latest pre-trained model from the server. If not provided, use default based on model_variant")
pipe.add_parameter("model_variant", "yolo11n", "YOLOv11 model variant from Ultralytics. Also saved as model_name for future updates.")
pipe.add_parameter("model_hyps", "", "Dictionary of YOLO.train() input params. Defaults from model variant config file")

def pre_training_callback(pipeline, node, param_override) -> bool:  
    print("Cloning model_training id={}".format(node.base_task_id))    
     
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
    print("Completed model_training id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="model_training",
    parents=["dataset_processing"],
    base_task_project=project_name,
    base_task_name="Model Training",
    parameter_override={
        "General/dataset_id": (
            "${dataset_processing.parameters.General/output_dataset_id}"
            if pipe.get_parameters()["base_dataset_url"] 
                or pipe.get_parameters()["base_dataset_id"]
                or pipe.get_parameters()["base_dataset_name"]
            else "${pipeline.model_dataset_id}"), # no output from previous steps    
        "General/dataset_name": "${pipeline.split_dataset_name}", 
        "General/model_id": "${pipeline.model_id}",   
        "General/model_name": "${pipeline.model_name}",       
        "General/model_variant": "${pipeline.model_variant}",
        "General/model_hyps": "${pipeline.model_hyps}"
    },
    pre_execute_callback=pre_training_callback,
    post_execute_callback=post_training_callback
)


""" 
STEP 4: Model hyperparameter optimisation
"""
# model optimisation settings
pipe.add_parameter("hpo_min_batch", 6, "Minimum batch size of HPO range")
pipe.add_parameter("hpo_max_batch", 8, "Maximum batch size of HPO range")
pipe.add_parameter("hpo_min_weight_decay", 1e-5, "Minimum batch size of HPO range")
pipe.add_parameter("hpo_max_weight_decay", 1e-5, "Maximum batch size of HPO range")
pipe.add_parameter("total_max_jobs", 3, "Total maximum job for the optimization process")
pipe.add_parameter("max_job_iter", 5 , "Number of iteration per job ‘iterations’ for the specified objective")
pipe.add_parameter("concurrent_tasks", 5 , "Nuber of concurrent experiments running")

def pre_hpo_callback(pipeline, node, param_override) -> bool:  
    print("Cloning model_hpo id={}".format(node.base_task_id))    
    
    print("Cloning Task id={} with parameters: {}".format(
        node.base_task_id, param_override))
    
    return True
            
def post_hpo_callback(pipeline, node) -> None:
    print("Completed model_hpo id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="model_hpo",
    parents=["model_training"],
    base_task_project=project_name,
    base_task_name="Model HPO",
    parameter_override={
        "General/base_task_id": "${model_training.id}",   
        "General/hpo_min_batch": "${pipeline.hpo_min_batch}",       
        "General/hpo_max_batch": "${pipeline.hpo_max_batch}",
        "General/hpo_min_weight_decay": "${pipeline.hpo_min_weight_decay}",
        "General/hpo_max_weight_decay": "${pipeline.hpo_max_weight_decay}",
        "General/total_max_jobs": "${pipeline.total_max_jobs}",
        "General/max_job_iter": "${pipeline.max_job_iter}",
        "General/concurrent_tasks": "${pipeline.concurrent_tasks}"
    },
    pre_execute_callback=pre_hpo_callback,
    post_execute_callback=post_hpo_callback
)

""" 
STEP 1.2: Upload eval dataset
"""
# intial dataset to download. If none provided, task will complete without upload
eval_dataset_url = project.get("eval-dataset-url")
# eval_dataset_url = ""       # default for no uploading
pipe.add_parameter("eval_dataset_url", eval_dataset_url, "(Optional) URL to the evaluation dataset.")
pipe.add_parameter("eval_dataset_name", "eval_dataset", "Name of the dataset to upload to the server. Also used for the next step.")

def pre_eval_upload_callback(pipeline, node, param_override) -> bool:    
    print("Cloning upload_eval_dataset id={}".format(node.base_task_id))    
    return True

def post_eval_upload_callback(pipeline, node) -> None:   
    print("Completed upload_eval_dataset id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="upload_eval_dataset",
    base_task_project=project_name,
    base_task_name="Upload Evaluation Dataset",
    parameter_override={
        "General/dataset_url": "${pipeline.eval_dataset_url}",
        "General/output_dataset_name": "${pipeline.eval_dataset_name}"},
    pre_execute_callback=pre_eval_upload_callback,
    post_execute_callback=post_eval_upload_callback
)

"""
STEP 5: Model Evaluation
"""

def load_eval_config(model_variant) -> dict:
    eval_config_file = f"{model_variant}_eval_config.yaml"
    eval_config_path = Path(__file__).parent / eval_config_file
    print("eval_config_path=", eval_config_path.resolve())
    if eval_config_path.exists():    
        with open(eval_config_path, "r") as file:
            eval_confg = yaml.safe_load(file)
    
    return eval_confg

pipe.add_parameter("eval_dataset_id", "", "(Optional) Overitten if previous task is not skipped. If set, ignore eval_dataset_name")
pipe.add_parameter("eval_args", "", "Dictionary of YOLO.val() input params. Defaults from model variant config file")

def pre_eval_callback(pipeline, node, param_override) -> bool:    
    print("Cloning model_evaluation id={}".format(node.base_task_id))      # param validation check
    model_variant = pipe.get_parameters()["model_variant"]
    if not model_variant:
        raise ValueError(f"Missing model_variant param value.")
    
    # add default eval config if none provided     
    if not param_override["General/eval_args"]:
        eval_args = load_eval_config(model_variant)
        eval_args_str = yaml.dump(eval_args) 
        param_override["General/eval_args"] = eval_args_str
    
    return True

def post_eval_callback(pipeline, node) -> None:   
    print("Completed model_evaluation id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="model_evaluation",
    parents=["model_hpo", "upload_eval_dataset"],
    base_task_project=project_name,
    base_task_name="Model Evaluation",
    parameter_override={
        "General/eval_dataset_id":  (
            "${upload_eval_dataset.parameters.General/output_dataset_id}"
            if pipe.get_parameters()["eval_dataset_url"] 
            else "${pipeline.model_dataset_id}"), # no eval dataset upload
        "General/eval_dataset_name": "${pipeline.eval_dataset_name}",
        "General/draft_model_id": "${model_hpo.parameters.General/best_model_id}",
        "General/pub_model_name": "${pipeline.model_variant}",
        "General/eval_args": "${pipeline.eval_args}"
    },
    pre_execute_callback=pre_eval_callback,
    post_execute_callback=post_eval_callback
)

"""
STEP 6: Model Publishing
"""

def pre_pub_callback(pipeline, node, param_override) -> bool:
    print("Cloning model_publishing id={}".format(node.base_task_id))    
    return True

def post_pub_callback(pipeline, node) -> None:
    print("Completed model_publishing id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_step(
    name="model_publishing",
    parents=["model_evaluation"],
    base_task_project=project_name,
    base_task_name="Model Publishing",
    parameter_override={
        "General/draft_model_id": "${model_evaluation.parameters.General/best_model_id}"
    },
    pre_execute_callback=pre_pub_callback,
    post_execute_callback=post_pub_callback
)


"""
STEP 7: Model Deployment
"""
def pre_deploy_callback(pipeline, node, param_override) -> bool:
    print("Cloning model_deploy id={}".format(node.base_task_id))    
    return True

def post_deploy_callback(pipeline, node) -> None:
    print("Completed model_deploy id={} {}".format(node.base_task_id, node.executed))    
    return

pipe.add_parameter("pub_model_name", "yolo11n", "Deploy the lastest published model to endpoint")

pipe.add_step(
    name="model_deployment",
    parents=["model_publishing"],
    base_task_project=project_name,
    base_task_name="Model Deployment",
    parameter_override={
        "General/pub_model_name": "${pipeline.pub_model_name}"
    },
    pre_execute_callback=pre_deploy_callback,
    post_execute_callback=post_deploy_callback
)

remote_execution = project.get("pipeline-remote-execution")

if remote_execution:
    print(f"Executing '{pipeline_name}' pipeline remotely")
    pipe.start(queue=project.get('queue-gpu'))
else:
    print(f"Executing '{pipeline_name}' pipeline locally")
    pipe.start_locally(run_pipeline_steps_locally=True)

print("done")