# Second Sight by EnigmaAI

This project consists 4 components.

## Cloning the entire project fom GitHub 
```sh 
git clone https://{github_username}:{github_access_token}@github.com/{github_username}/EnigmaAI.git
   ```

## Libaray dependencies
1. List all top level libraries in `requirements.txt` for top level.
2. Run `pip install -r requirements.txt` to install top level dependencies

## Hazard Detection component

### Install libary dependencies
1. List all libraries in `hazard_detection/requirements.txt`.
2. Run `pip install -r hazard_detection/requirements.txt` to install

## Image Description component
Submodules to train VLM model for descring and image or video frame from detected hazard geared for visual impairment.

Model output will be integrated into the application in API or UI (iOS app) components

For more detail, refer to submodules README.md including setup instructions.

### Install libary dependencies
1. List all libraries in `image_description/requirements.txt`.
2. Run `pip install -r image_description/requirements.txt` to install

## API component
API to serve the model inferencing endpoint for the application. This is a FastAPI project.

### Install libaray dependencies
1. List all libraries in `api/requirements.txt`.
2. Run `pip install -r api/requirements.txt` to install

## iOS component
UI component for iOS platform. This is the final application using all other componenets to serve its functionalities for a blind person. Please require to Requirment and Design documentations for more details.

done in swift







