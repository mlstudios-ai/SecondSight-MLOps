from clearml import Task, Dataset, Model
from pathlib import Path
import yaml
import os
import shutil
import tempfile
import torch
from ultralytics import YOLO
# from enigmaai.config import Project, Config, ConfigFactory

"""
Train YOLOv11 model using the latest split dataset stored on ClearML server.
THe dataset needs to be in the following structure:

data.yaml
train/images/
train/labels/
val/images/
val/labels/
test/images/
test/labels/

IMPORTANT: The dataset will be downloaded to a cached directory and will NOT be copied into 
local_working_directory. This is to preserve the built-in caching mechanism of 
Dataset.get_local_copy(). Caching keep tracks of changes in datdaset and only download if 
there is a new version to prevent repeat downloads, especially with a large dataset.

NOTE: this is for training only, model evaluation task with compare and register the best model for deployment.
"""

# NOT WORKING: setup.py not running on execute_remotely, hence can not import enigmaai package
# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
project_name="Detection"
Task.add_requirements('numpy', '==1.26.4')
# Task.add_requirements('pytorch', '')

task = Task.init(project_name=project_name, 
                task_name="Model Training", 
                task_type=Task.TaskTypes.training)