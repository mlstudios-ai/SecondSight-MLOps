import os
from typing import List
import torch
from pathlib import Path

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
    labels_dir = Path(labels_dir)
    class_counts = [0] * len(classes)
    
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        for label_file in label_files:
            with open(label_file, "r") as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    
    return class_counts
    