import torch
from ultralytics import YOLO

"""
Execute this so that it downloads the library dependecies 
for the remote worker. 
"""
task = Task.init(project_name="Hazard Detection", 
                task_name="Download Dependencies", 
                task_type=Task.TaskTypes.training)

task.execute_remotely(queue_name="training")

print("Downloaded packages to virutal env complete")
