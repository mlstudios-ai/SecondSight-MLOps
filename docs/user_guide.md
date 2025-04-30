# User Guide on MLOps using ClearML

## Hazard Detection
There are 6 tasks and 1 pipeline to orchestrate the machine learning operation process using YOLOV11 model for
detecting hazard objects as part of the CI/CD pipelines.

### ClearML Tasks
There are 6 tasks descibed as follow:
- **Upload Evaluation Dataset** - uploads a dataset to evaluate the newly trained model and the trained model for comparisons
- **Upload Base Dataset** - uploads a full dataset, used for splitting and training later
- **Split Base Dataset** - split the full dataset into train and val. 
- **Model Training** - train the YOLO model using the split dataset. Test split is not used.
- **Model Evaluation** - compare Recall performance metrics with the published model and select the best model
- **Model Publishing** - publish the best model

### ClearML Pipeline
There is one pipeline but this pipeline can be used for multipurposes using different parameter configurations. 
- **YOLOv11 Pipeline** This pipeline uses the tasks for orchestrating sequential execution to complete machine learning operations from 
dataset upload to model publishing.

### MLOps Processes
The following steps are used for an end-to-end pipeline in the **YOLOv11 Pipeline**.

### 1. Edit the `project_config.yaml` to set your project name and other settings
### 2. Start a ClearML Agent for the tasks and pipeline
1. Run `clearml-agent daemon --queue "default" --detached` for CPU tasks
2. Run `clearml-agent daemon --queue "training" --detached` for GPU tasks
3. Run `clearml-agent daemon --queue "{my_queue_name}" --detached` for pipeline unique for your machine only

### 3. Initialise all tasks:
1. Navigate to repo folder `hazard_detection/pipelines/tasks`.
2. Run ` python dataset_eval_upload.py; python dataset_base_upload.py; python dataset_base_split.py; python model_train.py; python model_eval.py; python model_publish.py `

### 4. Upload datasets using tasks on WebUI in STEP 1.1 Option 1 and STEP 1.2 Option 2.

### 5. Run the following to initialise the pipeline
This could take a while.
1. Navigate to repo project folder (ie, EnigmaAI folder). It must be in the root project folder.
2. Run `hazard_detection/pipelines/detection_pipeline.py`

### 6 Clone the **YOLOv11 Pipeline** for various purposes with the following settings

##### *<u>STEP</u> 1.1: Upload Base Dataset*

###### Option 1 (Preferred): Upload via **Upload Base Dataset** task
1. Clone the tasks and prefix with the agent name of your machine (any name will do as long as it is identifiable)
2. Edit and set the `databset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path})
3. Edit and set the `output_dataset_name` value, defaults to `base_dataset`. This name is used to associate the dataset upload on the server.
4. Right click on the cloned task and enqueue for execution. The upload should start
5. Verify upload on dataset with the output name or the default `base_dataset` on ClearML WebUI

###### Option 2 : Upload via pipeline parameter (Optional)
By default, base dataset is assume to be uploaded via a task. This step will compelete without upload if `base_dataset_url` is not set, hence, skipping the step. To skip, omit setting value for `base_dataset_url`.

To upload base dataset via the pipeline, follow these instructions:
1. Set the `base_dataset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path}). 
2. If `base_dataset_url` is set, the `base_dataset_name` must be set. This name is used to associate the dataset upload on the server. It is also used if `base_dataset_id` is not set in the  **Split Base Dataset** step to get the latest base dataset.

##### *<u>STEP</u> 2: Split Base Dataset* (Optional)
Depends on **Uploading Base Dataset**. This step need to be ran at least once to initialise the split dataset in the pipeline. After the first ru, the split dataset can be reused by omitting both of `base_dataset_id` and `base_dataset_name`. Without both parameters, the task will complete without execution, hence skipping the step. NOTE that `base_datset_id` will be overwritten if the **Split Base Dataset** task is executed with an output base dataset id.

To split the base dataset for training, at lease set ONE of the following values:
1. Set the `base_dataset_id` parameter to use a specific base dataset. If not set, it will get the latest from `base_dataset_name`. NOTE that if the base dataset is uploaded via pipeline, this step will be overwritting with the output of the previous task uplopad. 
2. Set the `split_val` and `split_test` for validation and test split in percentages.
3. If `base_dataset_id` is set, the `split_dataset_name` must be set. This name is used to associate the dataset upload on the server. It is also used if `base_dataset_id` is not set in the **Model Training** step to get the latest split dataset.

##### *<u>STEP</u> 3: Model Training* 
Depends on **Split Base Dataset**. This step can not be skipped and is the starting point of the default setting for the pipeline. 

To train the model:
1. Set at least ONE of the parameters `model_dataset_id` for specific dataset set or `split_dataset_name` for latest version of `dataset` used the model to train from
2. Set at least ONE of the parameters `model_id` for a specific model, `model_name` for the latest version of the model. e.g. `yolo11n`, or `model_variant` for variant of the model to be downloaded from Ultralytics repository.
3. Set `hyps` values as a string for hyperparameters of the model `YOLO.train()` method. If not set, it will use the configs in the file `{model_variant}_hyps_config.yaml`.


##### *<u>STEP</u> 1.2: Upload Evaluation Dataset*

###### Option 1 (Preferred): Upload via **Upload Evaluation Dataset** task
1. Clone the tasks and prefix with the agent name of your machine (any name will do as long as it is identifiable)
2. Edit and set the `eval_databset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path})
3. Edit and set the `output_dataset_name` value, defaults to `eval_dataset`. This is name is used to associate the dataset upload on the server.
4. Right click on the cloned task and enqueue for execution. The upload should start
5. Verify upload on dataset with the output name or the default `eval_dataset` on ClearML WebUI

###### Option 2 : Upload via pipeline parameter (Optional)
By default, eval dataset is assume to be uploaded via a task. This step will compelete without upload if `eval_dataset_url` is not set, hence, skipping the step. To skip, omit setting value for `eval_dataset_url`.

To upload evaluation dataset via the pipeline, follow these instructions:
1. Set the `eval_dataset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path}).  
2. If `eval_dataset_url` is set, the `eval_dataset_name` must be set. Default is `eval_dataset`. This name is used to associate the dataset upload on the server. It is also used if `eval_dataset_id` is not set in the **Model Evaluation** step to get the latest evaluation dataset.

##### *<u>STEP</u> 4: Model Evaluation* 
Depends on **Model Training** and **Upload Evaluation Dataset**. This step evaluates the newly trained model against the published model using the evaluation dataset to evaluate both models.

To evaluate the newly trained model from the previous step:
1. Set at leaset ONE of the parameters `eval_dataset_id` specific dataset, or `eval_dataset_name` for the latest version of dataset, defaults `eval_dataset` 
2. Set `eval_args` values as a string for input of the model `YOLO.val()` method. If not set, it will use the configs in the file `{model_variant}_eval_config.yaml`.

##### *<u>STEP</u> 5: Model Publishing* 
Depends on **Model Evaluation**. This published a model to the register. There are no parameters to be set in this step. The `draft_model_id` will be automatically set to the output of best model from the **Model Evaluation** step. If the best model is already published, this step will do nothing, otherwise it will be published to the register ready for serving.

### 6. Select your queue and hit RUN to execute the pipeline

### 7. Tag the new pipeline meaningfully

### MLOps Example Setup

#### End-to-end: Complete operations

#### Split and train: Use existing base dataset

#### Train and publish: Use all existing dataset

#### Update model: Upload addition dataset and retrain existing model




