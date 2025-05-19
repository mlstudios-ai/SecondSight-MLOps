import os
from dotenv import load_dotenv
import base64
from github import Github, InputGitTreeElement, GithubException

class DeploymentError(Exception):
    """ Error deploying model to endpoint url for inferencing. """
    pass
    
def deploy_model(file_path: str, repo_name: str, repo_branch: str, repo_path: str):
    load_dotenv()
    
    try:
        # Authenticate with a personal access token
        g = Github(os.getenv("ENDPOINT_GITHUB_ACCESS_TOKEN"))
        
        print("user:", g)
        repo = g.get_user().get_repo(repo_name)

        print("repo:", repo)
        
        with open(file_path, "rb") as f:
            content = f.read()
            encoded_content = base64.b64encode(content).decode()

        blob = repo.create_git_blob(encoded_content, encoding="base64")
        
        # Get the latest commit
        main_ref = repo.get_git_ref(repo_branch)
        main_sha = main_ref.object.sha
        base_tree = repo.get_git_tree(main_sha)

        # Create a git tree element
        element = InputGitTreeElement(repo_path, '100644', 'blob', sha=blob.sha)
        tree = repo.create_git_tree([element], base_tree)
        parent = repo.get_git_commit(main_sha)
        commit = repo.create_git_commit("Add model file", tree, [parent])
        main_ref.edit(commit.sha)
        
    except GithubException as e:
        raise DeploymentError(f"GitHub error: {e.data['message']}")
    
    except Exception as e:
        raise DeploymentError(f"Unexpected error: {e}")