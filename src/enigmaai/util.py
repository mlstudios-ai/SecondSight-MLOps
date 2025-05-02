import os
from typing import List
import torch

def get_device_name() -> str:
    """
    Check if current systam has GPU support.

    Returns:
        str: device name depending of GPU availability
    """
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.backends.mps.is_available(): #and torch.backends.mps.is_built():
        device_name = "mps"
        
    return device_name

def class_dist(labels_dir: List[str], classes: List[str]) -> List[int]:
    """
    Get the class distribution of a dataset used in yolo or other compatible format.

    Args:
        label_path (List[str]): absolute file directory for the labels
        classes (List[str]): the classes in order for distribution counts

    Returns:
        List[int]: Class counts in the order of `classes` parameter
    """
    class_counts = [0] * len(classes)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file)) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    return class_counts
    