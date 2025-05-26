import os
from dotenv import load_dotenv
import base64
from github import Github, InputGitTreeElement, GithubException

class DeploymentError(Exception):
    """ Error deploying model to endpoint url for inferencing. """
    def __init__(self, message):
        self.message = message
        super().__init__(message)
    
def deploy_model(file_path: str, repo_name: str, repo_branch: str, repo_path: str):
    """Check a file from local to github repo. Please set github access key in .env file.

    Args:
        file_path (str): Local file path
        repo_name (str): GitHub repository name
        repo_branch (str): GitHub branch. e.g main
        repo_path (str): path from the root folder in the repo

    Raises:
        DeploymentError: Errors that prevent deploying model into GitHub.
    """
    load_dotenv()
    
    try:
        # Authenticate with a personal access token
        g = Github(os.getenv("ENDPOINT_GITHUB_ACCESS_TOKEN"))        
        repo = g.get_user().get_repo(repo_name)
        print("user:", g.get_user(), "repo:", repo)
        
        with open(file_path, "rb") as f:
            content = f.read()
            
        try: # File exists, update it
            file = repo.get_contents(repo_path, ref=repo_branch)            
            repo.update_file(
                path=repo_path,
                message="Update model file",
                content=content,
                sha=file.sha,
                branch=repo_branch
            )
            print(f"Updated {repo_path}")
        except Exception: # File does not exist, create it            
            repo.create_file(
                path=repo_path,
                message="Add model file",
                content=content,
                branch=repo_branch
            )
            print(f"Updated {repo_path}")
    except GithubException as e:
        raise DeploymentError(f"GitHub error: {e.data['message']}")
    
    except Exception as e:
        raise DeploymentError(f"Unexpected error: {e}")