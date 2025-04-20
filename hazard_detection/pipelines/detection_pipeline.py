from clearml import Task
from clearml.automation import PipelineController

project_name = "Detection"

# if not hyperparameters: # use default config from github repo 
#     variant_train_config = f"./{model_variant}_train_config.yaml"
#     if not os.path.exists(variant_train_config):    
#         with open("config.yaml", "r") as file:
#             hyperparameters = yaml.safe_load(file)
#     else:
#         raise ValueError("Missing training hyperparameter.")
    
    
def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print(
        "Cloning Task id={} with parameters: {}".format(
            a_node.base_task_id, current_param_override
        )
    )
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print("Completed Task id={}".format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(name="Train YOLOv11 Model", 
                          project=project_name, 
                          add_pipeline_tags=False)

pipe.add_parameter(
    "base_dataset_url",
    "https://raw.githubusercontent.com/vanilla-ai-ml/large_datasets/main/mini.zip",
    "(Optional) URL to the final dataset. Used as default unless base_dataset_id or base_dataset_name is provided"
)

# pipe.add_parameter(
#     "model_variant",
#     "",
#     "YOLOv11 model variant to train. Stored as model name."
# )

# pipe.add_parameter(
#     "base_model_url",
#     "",
#     "(Optional) URL to the final dataset. Used as default unless base_dataset_id or base_dataset_name is provided"
# )

# pipe.set_default_execution_queue("training")

# pipe.add_step(
#     name="load_base_dataset",
#     base_task_project=project_name,
#     base_task_name="Upload Base Dataset"
# )

# pipe.add_step(
#     name="dataset_preprocessing",
#     base_task_project=project_name,
#     base_task_name="Split Dataset"
# )

# pipe.add_step(
#     name="train_model",
#     base_task_project=project_name,
#     base_task_name="Model Training"
# )


# for debugging purposes use local jobs
pipe.start_locally()

# Starting the pipeline (in the background)
# pipe.start()

print("done")