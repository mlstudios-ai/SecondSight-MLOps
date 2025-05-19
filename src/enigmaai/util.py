import os
from typing import List
import torch
from pathlib import Path
from dotenv import load_dotenv
import base64
from github import Github, InputGitTreeElement, GithubException

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
    
def deploy_model(file_path: str, repo_name: str, repo_ref: str, repo_path: str):
    # Load .env file
    load_dotenv()
    
    try:
        # Authenticate with a personal access token
        g = Github(os.getenv("ENDPOINT_GITHUB_ACCESS_TOKEN"))
        repo = g.get_user().get_repo(repo_name)

        with open(file_path, "rb") as f:
            content = f.read()
            encoded_content = base64.b64encode(content).decode()

        # Get the latest commit
        master_ref = repo.get_git_ref(repo_ref)
        master_sha = master_ref.object.sha
        base_tree = repo.get_git_tree(master_sha)

        # Create a git tree element
        element = InputGitTreeElement(repo_path, '100644', 'blob', encoded_content)
        tree = repo.create_git_tree([element], base_tree)
        parent = repo.get_git_commit(master_sha)
        commit = repo.create_git_commit("Add model file", tree, [parent])
        master_ref.edit(commit.sha)
        
    except GithubException as e:
        print(f"GitHub error: {e.data['message']}")
        return False
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False