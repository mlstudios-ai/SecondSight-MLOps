from clearml import Task, Dataset, Model
from pathlib import Path
import os
import shutil
import yaml
from enigmaai import util
from enigmaai.config import Project, ConfigFactory

project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="Model Evaluation", 
                task_type=Task.TaskTypes.qc)