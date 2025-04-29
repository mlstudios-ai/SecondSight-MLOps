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
### 2. Run the following to initiliase all the tasks:
` python dataset_eval_upload.py; python dataset_base_upload.py; python dataset_base_split.py; python model_train.py; python model_eval.py; python model_publish.py `
### 3. Start a ClearML Agent with a queue unique to your machine or worker. This is used for remote execution using your machine and only your machine

### 4. Create a new run from **YOLOv11 Pipelinet** set the following parameters to configure the pipeline for desired operations.

##### *<u>STEP</u> 1.1: Upload Base Dataset*

###### Option 1 (Preferred): Upload via **Upload Base Dataset** task
1. Clone the tasks and prefix with the agent name of your machine (any name will do as long as it is identifiable)
2. Edit and set the `databset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path})
3. Right click on the cloned task and enqueue for execution. The upload should start
4. Verify upload on dataset with name `base_dataset` on ClearML WebUI

###### Option 2 : Upload via pipeline parameter (Optional)
By default, base dataset is assume to be uploaded via a task. This step will compelete without upload if `base_dataset_url` is not set, hence, skipping the step. To skip, omit setting value for `base_dataset_url`.

To upload evaluation dataset via the pipeline, follow these instructions:
1. Set the `base_dataset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path}). 
2. Click RUN to execute the pipeline NOTE: this will run the entire pipeline. Please enter all your desired parameter values before RUN.

##### *<u>STEP</u> 1.2: Upload Evaluation Dataset*

###### Option 1 (Preferred): Upload via **Upload Evaluation Dataset** task
1. Clone the tasks and prefix with the agent name of your machine (any name will do as long as it is identifiable)
2. Edit and set the `eval_databset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path})
3. Right click on the cloned task and enqueue for execution. The upload should start
4. Verify upload on dataset with name `eval_dataset` on ClearML WebUI

###### Option 2 : Upload via pipeline parameter (Optional)
By default, eval dataset is assume to be uploaded via a task. This step will compelete without upload if `eval_dataset_url` is not set, hence, skipping the step. To skip, omit setting value for `eval_dataset_url`.

To upload evaluation dataset via the pipeline, follow these instructions:
1. Set the `eval_dataset_url` value to your zip file location. This can be a URL (https://) or local file (file://{full_dataset_path}). 

##### *<u>STEP</u> 2: Split Base Dataset* (Optional)
Depends on **Uploading Base Dataset**. This step need to be ran at least once to initialise the split dataset in the pipeline. After the first ru, the split dataset can be reused by omitting both of `base_dataset_id` and `base_dataset_name`. Without both parameters, the task will complete without execution, hence skipping the step.

To split the base dataset for training, at lease set ONE of the following values:
1. NOTE that if the base dataset is uploaded via pipeline, this step will be overwritting with the output of the previous task uplopad. If previous step is skipped, set the `base_dataset_id` parameter to use a specific base dataset. This takes priority over `base_dataset_name`.
2. Set the `base_dataset_name` to use the lastest version of the `base_dataset`. This will only be used if the previous upload tasks is skipped and `base_dataset_id` is not set.

##### *<u>STEP</u> 3: Model Training* 
Depends on **Split Base Dataset**. This step can not be skipped and is the starting point of the default setting for the pipeline. 

To train the model:
1. Set at lease ONE of the parameters `model_dataset_id` for specific dataset set or `model_dataset_name` for latest version of `dataset` used the model to train from
2. Set at least ONE of the parameters `model_id` for a specific model, `model_name` for the latest version of the model. e.g. `yolo11n`, or `model_variant` for variant of the model to be downloaded from Ultralytics repository.
3. Set `hyps` values as a string for hyperparameters of the model `YOLO.train()` method. This is a mandatory field.

##### *<u>STEP</u> 4: Model Evaluation* 
Depends on **Model Training** and **Upload Evaluation Dataset**. This step evaluates the newly trained model against the published model using the evaluation dataset to evaluate both models.

To evaluate the newly trained model from the previous step:
1. Set at leaset ONE of the parameters `eval_dataset_id` specific `eval_dataset`, or `eval_dataset_name` for the latest version of `eval_dataset` 
2. Set `eval_args` values as a string for input of the model `YOLO.val()` method. This is a mandatory field.

##### *<u>STEP</u> 5: Model Publishing* 
Depends on **Model Evaluation**. This published a model to the register. There are no parameters to be set in this step. The `draft_model_id` will be automatically set to the output of best model from the **Model Evaluation** step. If the best model is already published, this step will do nothing, otherwise it will be published to the register ready for serving.

### 5. Select your queue and hit RUN to execute the pipeline

### 6. Tag the new pipeline with a meaningful name





