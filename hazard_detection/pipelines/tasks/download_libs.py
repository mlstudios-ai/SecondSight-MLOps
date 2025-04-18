from clearml import Task
# import torch
# from ultralytics import YOLO
# from enigmaai.config import Project, Config, ConfigFactory

"""
Downloads the library dependecies for the remote worker. 
Run this on all remote workers so that other tasks can
run smoothly instead of hanging trying to download dependcies
at the same time.
"""

# get project configurations
# project = ConfigFactory.get_config(Project.HAZARD_DETECTION)
# project_name = project.get('project-name')
# project_name="Detection"

task = Task.init(project_name="Detection", 
                task_name="Download Library Dependencies", 
                task_type=Task.TaskTypes.custom)

task.execute_remotely(queue_name="default")

print("Downloaded packages to virutal env complete")
