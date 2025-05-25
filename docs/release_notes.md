# Release Notes:
Product: SecondSight - Live hazard detection for visual impairment <br>
## Version 0.2
Release Date: 4 May, 2025<br>

### Overview
We have completed end-to-end pipeline from data upload to model deployment as part of the MLOps CI/CD. Building from the previous version where you can configure the pipeline and tasks to suit your needs, this version now added hyperparameter optimisation and model deployment for on-deviced and remote API endpoint inferencing.

We also added Pull Request release to automatically trigger retraining of the model.

For on-device inferencing, our model uses the pipeline YOLO11 Nano output model deployed to FastAPI then converted to CoreML for iOS compatibility. You can now download the iOS protoype for use in live detection. This is a standalone mode - no internet connection required.

Please refer to https://github.com/vanilla-ai-ml/SecondSight for your build. NOTE: permission required to access repo.

For remote API inferencing, we host it on FastAPI framework and use for scene description. The model is a custom fine tuned model for vision impairment. You can now build the API from https://github.com/vanilla-ai-ml/SecondSight-API and use for inferencing the scene description model.r

#### Hazard Detection Pipeline
There are minimal updates from Hazard Detection Pipeline apart from the HPO and Mode Deployment tasks. For this version, we focus on system integration and the completion of scene description model pipeline below. 

#### Scene Description Pipeline
The Scene Description Pipeline is for training Vision encoder-Language decoder model architecture using google/vit-base-patch16-224-in21k for vision and distilgpt2 for language components. The pipeline begins with mapping the images with annotations for both train and test sets, to publishing the model after training and fine-tuning hyperparameters. Similar to the Hazard Detection Pipeline, it is designed to be flexible.

VLMPipeline: end-to-end vision‐language MLOps orchestration using ClearML
This pipeline orchestrates the full life cycle of a vision‐language (VLM) image‐captioning model include hyperparameter optimization, providing flexible
entry points so you can run only the steps you need

Flexible Execution Modes (Just as YOLOv11 Pipeline)
Depending on which pipeline parameters supplied, it can skip any of the first three steps and pick up later:
- **Full run**: steps 1→9 
- **Skip raw download**: start at step 2  
- **Skip prep + labeling**: start at step 3  
- **Skip prep + training**: start at step 4 (directly evaluate an existing model)
> **Note:** Model evaluation (step 8) cannot be skipped it would auto‐publish whatever draft model provided.
All steps log their task IDs and parameters to the console, so you can trace exactly how each subtask was cloned
and run. Simply override the pipeline parameters at launch time to adjust behavior, queues, or hyperparameter ranges.

### New Features 
- Optimised YOLO11 Nano model for hazard detection
- UI application and model integration
- Completion of VML pipeline. Now the machine learning is fully automated.

### Improvements
- Hyperparameter optimised models
- Auto deployment
- Refined UI design

### Bug Fixes
- Cache error on File Not Found

### Known Issues
No load testing on scene description model on FastAPI
Limited UI testing for protoyping
Voice Synthesizer has some multi threading issues where repeat speech quickly can cause malfunction of the synthesizer.

##3 Installation
Please refer to `/README.md`.

### Coming Soon
- Model registration and serving
- Model deployment and inferencing
- Application UI for hazard detection.

## Version 0.1<br>
Release Date: 4 May, 2025<br>

### Overview
We are pround to release our machine learning pipelines on ClearML for automation. The automation project can be configurated to run in different environments in isolation, removing chance of conflict. This is done by simply give it a project name in the project config. This feature enables production workflow environment such as dev, staging, UAT, and production environments.

The pipelines are geared towards many of machine learning processes through configurable experiements and orchestration. Each ClearML tasks are highly configurable and can be executed indpendently for various experiments. These tasks can be orchestrated in a pipeline for a sequence of machine learning operations, levaraging the way the tasks are designed.

The Hazard Detection pipeline is for training YOLOv11 models from end-to-end. That is from dataset upload to model publishing. The pipeline is designed to cater for flexibilities of different needs at different point of the model training process. 

Sometimes user may want to execute a task for specific purposes and then continue with the pipeline. Some steps can be skipped if the minimum parameters are not provided as specified in the pipeline parameter descriptions (refer to the corresponding tasks for more info).

The YOLOv11 Pipeline is highly configurable via parameter settings can be set in each new run for various purposes as follow:

1. End-to-end from downloading base dataset from remote URL to model publising.
2. Skip step 1 URL download, use existing base datatset, and start from step 2: dataset processing
3. Skip steps 1 & 2, use processed dataset and start from step 3: model training
4. Skip steps 1, 2, & 3, use existing model as the new model for evaluation, starts from step 4: model evaluation

The above scenarios are designed to reduce execution time, resources requirements, duplications, and allows adjustment 
to various circumstances. Note that you can not skip model evaluation - this leads to publishing the model. If this is 
not a desire behaviour, use the task Model Evaluation from the WebUI instead of the pipeline.

The possibilities are endless!

IMPORTANT: by default, it will use the base_dataset and eval_dataset existing on the server, presuming they are already 
uploaded. If those datasets are not uploaded, please put in the base_dataset_url and/or eval_dataset_url accordingly.
Alternatively, before running the pipeline with default settings, upload the dataset using the following tasks from
the ClearML WebUI:

Upload Base Dataset - upload base dataset. This will trigger default pipeline to run in CD phase (Coming soon!)
Upload Evaluation Dataset - upload base dataset. This will trigger default pipeline to run in CD phase (Coming soon!)

For more information on how to use YOLOv11 pipeline, please refer to the User Guide on `/docs/user_guide.md`

Our Scene Description pipeline works in similar fashion. It uses the same design principle but adapated for fine tuning VLM model for scene description geared towards vision impairment. This pipeple requires a complex set of tasks combining vision and language. 

It takes on a common dataset from Hazard Detection for fune tuning and inline with the same vision and objectives. The annotation and dataset are automatically generated without needing of manual work. This pipleline can handle end to end similar to the YOLOv11 pipeline, that is from dataset upload to model publishing.

### New Features 
- Model selection of Yolov11 for best performance in mobile devices
- ClearML tasks for running various experiements independently.
- Pipeline automation for various CI operations and model training
- Each step contains data visualisation and performance visualisation for indepth analysis
- Configurable project. New changes can be made without changing the code.

### Improvements
- Optimised YOLO and VLM model
- Custom dataset curated specific for visual impairment
- Training now can be automated
- Added visualisation for better analysis

### Bug Fixes
None

### Known Issues
Scene Description pipeline not yet configurable. This feature is coming soon.

### Installation
Please refer to `/README.md`.

### Coming Soon
- Model registration and serving
- Model deployment and inferencing
- Application UI for hazard detection.