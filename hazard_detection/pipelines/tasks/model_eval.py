from clearml import Task, Dataset, Model
# from pathlib import Path
# import yaml
# import os
# import shutil
# import tempfile
# import torch
# from ultralytics import YOLO
# from enigmaai.config import Project, Config, ConfigFactory

"""
Compare to the published model. If performance is better, override published model as the best one.
"""

# NOT WORKING: setup.py not running on execute_remotely, hence can not import enigmaai package
# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"
Task.add_requirements('numpy', '==1.26.4')
# Task.add_requirements('pytorch', '')

task = Task.init(project_name=project_name, 
                task_name="Model Evaluation", 
                task_type=Task.TaskTypes.testing)

# TODO: check if there is a registered model, if none, register the input model.
# TODO: if the registered model id is the same, nothing to eval, exit.

# TODO: compare 2 models and publish the best one

params = {
    'model_id': '6cd8d49e57a244aba4b9751128d7b783',     # the unpublished model to evaluate 
    'model_variant': 'yolo11n',                         # the model variant of the published mode, same as model name
}


model_id = params['model_id']
model_variant = params["model_variant"]

task.connect(params)
# task.execute_remotely(queue_name="default")

# validate task input params
if not model_id:
    task.mark_completed(status_message="No model provided. Nothing to compare.")
    exit(0)
    
# Mandatory input param
if not model_variant:
    raise ValueError("Missing model variant. Please provide model_variant.")

# fetch the specific model for evaluation    
draft_model = Model(model_id=model_id)    
print(f"Found draft model name:{draft_model.name} id:{draft_model.id}")

# fetch the published best model
model_name = model_variant
server_models = Model.query_models(project_name=project_name, model_name=model_name, only_published=True)
if server_models:
    # best published model found
    published_model = server_models[0]
    best_model = published_model
    print(f"Found published model:{published_model.name} id:{published_model.id}")        
else:
    # best published model not found, use draft (first train) as best
    best_model = draft_model
    print (f"No published model found with name '{model_name}'. Draft model set as best.")

# compare and evaluate, best model gets published
if draft_model.id == best_model.id:
    # published draft model as first publication
    print(f"Publish first model name:{draft_model.name} id:{draft_model.id}")
    draft_model.publish()
else:
    # fetch metrics for both models and compare
    draft_task = Task.get_task(task_id=draft_model.task)
    best_task = Task.get_task(task_id=best_model.task)
    
    # print("draft_task scalars:", draft_task.get_all_reported_scalars())
    # print("best_task scalars:", best_task.get_all_reported_scalars())
    
    
