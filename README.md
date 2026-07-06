# SecondSight MLOps

**ML Pipeline Automation for Hazard Detection & Scene Description**

This repository contains the **MLOps infrastructure** for SecondSight - an AI-powered assistive iOS application for individuals with visual impairment. The project implements end-to-end machine learning pipelines for training, evaluating, and deploying two core AI models: **YOLO v11n** for hazard detection and **distilled VLM** for scene description.

## Project Context

SecondSight provides real-time hazard detection and environmental scene description through:
- **On-device inference**: YOLO v11n CoreML model for hazard detection (5 classes: pole, vehicle, person, wheelchair, stroller)
- **Remote inference**: LLaVA 1.5-7B distilled model for scene description via FastAPI
- **Target platform**: iOS 17+ (iPhone 14 Pro / 15 Pro)
- **Performance targets**: ≥75% recall for detection, ≥0.47 CIDER for description, ≤300ms latency

---

## Table of Contents
- [MLOps Architecture](#mlops-architecture)
- [Pipeline Overview](#pipeline-overview)
  - [Hazard Detection Pipeline](#hazard-detection-pipeline)
  - [Scene Description Pipeline](#scene-description-pipeline)
- [Pipeline Orchestration](#pipeline-orchestration)
- [Model Training & Deployment](#model-training--deployment)
- [Performance Monitoring](#performance-monitoring)
- [Environment Setup](#environment-setup)
- [Running Pipelines](#running-pipelines)
- [Project Structure](#project-structure)
- [CI/CD Workflow](#cicd-workflow)

---

## System Architecture

The system architecture implements **component-based design** with automated pipelines for continuous model training, evaluation, and deployment.

![System Architecture](docs/images/SecondSight%20-%20Solution%20Design%20v0.1.png)

For application component, please see [SecondSight](https://github.com/mlstudios-ai/SecondSight)

For API component, please see [FastAPI API](https://github.com/mlstudios-ai/SecondSight-API)

### MLOps Stack:

#### **Orchestration & Experimentation**
- **ClearML**: Pipeline orchestration, experiment tracking, model versioning
- **Remote Agents**: Google Cloud compute for distributed task execution
- **Task Scheduling**: Automated pipeline triggers and dependencies

#### **Model Training**
- **Hazard Detection**: YOLO v11n training on custom hazard dataset (5 classes)
- **Scene Description**: Knowledge distillation from LLaVA 1.5-7B to lightweight VIT-GPT2 student model
- **HPO**: Hyperparameter optimization using ClearML orchestrated experiments

#### **Model Deployment**
- **On-device (iOS)**: PyTorch → CoreML conversion for hazard detection
- **Remote API**: FastAPI endpoint for scene description inference
- **Containerization**: Docker images deployed to AWS ECS
- **CI/CD**: GitHub Actions for automated model deployment

#### **Model Serving**
- **FastAPI**: REST API for model inference (both on-device and remote endpoints)
- **GitHub Repo**: Model artifacts published to GitHub for version control
- **AWS ECS**: Cloud hosting for scene description model (on-demand scaling)

---

## Pipeline Overview

### Hazard Detection Pipeline

Automated end-to-end pipeline for training and deploying **YOLO v11n Nano CoreML** models for on-device hazard detection (5 classes: pole, vehicle, person, wheelchair, stroller).

![Hazard Detection Pipeline](docs/images/detection_pipeline_info.png)

#### **Pipeline Tasks (ClearML Orchestration)**

| Task | Purpose | Runtime | Key Operations |
|------|---------|---------|----------------|
| **upload_base_dataset** | Data ingestion | - | Validation on annotations, upload from URL, EDA visualization |
| **dataset_process_split** | Data preparation | - | Configurable split ratio, format conversion, augmentation |
| **model_training** | Model training | 13:38m | Fine-tune from base YOLO v11n or existing checkpoint |
| **model_hpo** | Hyperparameter tuning | 22:35m | Grid/random search over hyperparameter ranges |
| **upload_eval_dataset** | Eval data ingestion | - | Separate held-out test set upload and validation |
| **model_evaluation** | Model validation | 01:19m | Compare new model with published baseline, compute metrics |
| **model_publishing** | Model registry | 14s | Validate recall ≥75%, publish best model to registry |
| **model_deployment** | Production deploy | 18s | Deploy to GitHub FastAPI repo, convert to CoreML |

#### **Key Features**
- **Automation**: Sequential ClearML tasks with dependency management
- **Remote Execution**: Tasks distributed across Google Cloud remote agents
- **Version Control**: All models tracked in ClearML registry with metrics
- **Validation Gates**: Publishing only if recall threshold (≥75%) is met
- **Model Format**: PyTorch → CoreML conversion for iOS deployment

#### **Configuration**
```python
# Pipeline orchestration via ClearML
pipeline = Pipeline(name="hazard_detection_pipeline")
pipeline.add_step(name="upload_base_dataset", ...)
pipeline.add_step(name="model_training", parents=["upload_base_dataset"], ...)
pipeline.start_remotely(queue="default")
```

---

### Scene Description Pipeline

Automated MLOps pipeline for **knowledge distillation** from LLaVA 1.5-7B teacher model to lightweight VIT-GPT2 student model for scene description generation.

![Scene Description Pipeline](docs/images/description_pipeline_info.png)

#### **Pipeline Tasks (ClearML Orchestration)**

| Task | Purpose | Runtime | Key Operations |
|------|---------|---------|----------------|
| **step1_desc_basedata_preparation** | Teacher dataset prep | 39s | Extract images from hazard detection upload, data annotation for VLM |
| **step2_desc_testdata_preparation** | Test dataset prep | - | Extract evaluation images from detection upload |
| **step3_desc_basecaption_generation** | Teacher inference | 10:54m | Generate captions using LLaVA 1.5-7B for distillation training data |
| **step4_desc_evalcaption_generation** | Teacher inference | 06:32m | Generate eval captions using LLaVA model for benchmark |
| **step5_desc_split_data** | Data preparation | 30s | Configurable train/val split for student model training |
| **step6_desc_model_training** | Student training | 04:03m | Fine-tune VIT-GPT2 student on teacher-generated captions |
| **step7_desc_model_hpo** | Hyperparameter tuning | 33:38m | HPO for student model learning rate, batch size, epochs |
| **step8_desc_model_evaluation** | Model validation | 01:25m | Evaluate student using CIDER, BLEU, ROUGE metrics |
| **step9_desc_model_publish** | Model registry | 11s | Validate CIDER ≥0.47, publish to ClearML registry |

#### **Knowledge Distillation Strategy**
- **Teacher Model**: LLaVA 1.5-7B (vision-language model) - high quality but computationally expensive
- **Student Model**: VIT-GPT2 (distilled) - lightweight for real-time inference
- **Distillation Process**: Student learns from teacher-generated captions, not direct mimicking
- **Benefits**: 10x faster inference, much smaller model size, maintains quality (CIDER ≥0.47)

#### **Evaluation Metrics**
- **CIDER**: Consensus-based Image Description Evaluation (primary metric, target ≥0.47)
- **BLEU**: N-gram precision for caption quality
- **ROUGE**: Recall-based evaluation for description completeness

#### **Deployment**
```python
# FastAPI inference endpoint
@app.post("/describe")
async def generate_description(image: UploadFile):
    # Load distilled VIT-GPT2 model
    # Generate description
    # Return JSON response
```

---

## Pipeline Orchestration

### ClearML Task Execution

All pipeline tasks are orchestrated using **ClearML** with remote agent execution on Google Cloud:

```python
# Example: Hazard Detection Pipeline Orchestration
from clearml import Pipeline

pipe = Pipeline(
    name="hazard_detection_pipeline",
    project="SecondSight/HazardDetection",
    version="1.0"
)

# Define tasks with dependencies
pipe.add_step(
    name="upload_base_dataset",
    base_task_project="SecondSight/Tasks",
    base_task_name="dataset_upload",
    parameter_override={"dataset_url": "${pipeline.dataset_url}"}
)

pipe.add_step(
    name="model_training",
    parents=["upload_base_dataset", "dataset_process_split"],
    base_task_project="SecondSight/Tasks",
    base_task_name="yolo_training",
    parameter_override={"epochs": 100, "batch_size": 16}
)

# Start pipeline on remote queue
pipe.start_remotely(queue="default")
```

### Task Configuration
- **Agents**: Google Cloud remote agents with GPU support
- **Queues**: Default queue for CPU tasks, GPU queue for training tasks
- **Caching**: Intermediate results cached in ClearML for reproducibility
- **Logging**: All metrics, artifacts, and logs tracked in ClearML UI

### Pipeline Triggers
- **Manual**: Via ClearML UI or Python SDK
- **Scheduled**: Cron-based triggers for periodic retraining
- **Event-driven**: Triggered on new dataset uploads or model registry updates

---

## Model Training & Deployment

### Hazard Detection Workflow

```bash
# 1. Dataset preparation
python hazard_detection/pipelines/tasks/dataset_base_upload.py
python hazard_detection/pipelines/tasks/dataset_base_split.py

# 2. Model training
python hazard_detection/pipelines/tasks/model_train.py \
  --epochs 100 --batch_size 16 --img_size 640

# 3. Hyperparameter optimization
python hazard_detection/pipelines/tasks/model_hpo.py \
  --param_ranges '{"lr": [0.001, 0.01], "batch_size": [8, 16, 32]}'

# 4. Model evaluation
python hazard_detection/pipelines/tasks/model_eval.py \
  --test_dataset <dataset_id> --model_id <trained_model_id>

# 5. Publishing (if recall ≥ 75%)
python hazard_detection/pipelines/tasks/model_publish.py \
  --model_id <best_model_id> --min_recall 0.75

# 6. Deployment to FastAPI
python hazard_detection/pipelines/tasks/model_deploy.py \
  --model_id <published_model_id> --deploy_target github
```

### Scene Description Workflow

```bash
# 1. Data extraction from hazard detection dataset
python image_description/pipelines/tasks/base_data_preparation.py

# 2. Teacher model caption generation (LLaVA 1.5-7B)
python image_description/pipelines/tasks/base_desc_generation.py \
  --teacher_model "llava-1.5-7b" --batch_size 8

# 3. Student model training (knowledge distillation)
python image_description/pipelines/tasks/desc_model_train.py \
  --student_model "vit-gpt2" --epochs 50 --learning_rate 5e-5

# 4. HPO for student model
python image_description/pipelines/tasks/desc_hpo.py

# 5. Evaluation (CIDER, BLEU, ROUGE)
python image_description/pipelines/tasks/desc_model_eval.py

# 6. Publishing (if CIDER ≥ 0.47)
python image_description/pipelines/tasks/desc_model_publish.py \
  --min_cider 0.47
```

### Model Registry
- **ClearML Model Registry**: All trained models tracked with metadata
- **Versioning**: Semantic versioning (v1.0, v1.1, v2.0)
- **Tagging**: Models tagged as `baseline`, `candidate`, `production`
- **Artifacts**: Model weights, config files, performance metrics stored

---

## Performance Monitoring

### Model Performance Metrics

#### Hazard Detection (YOLO v11n)
| Metric | Target | Achieved | Pipeline Task |
|--------|--------|----------|---------------|
| **Recall (Macro)** | ≥75% | **75.1%** | `model_evaluation` |
| **mAP50** | ≥80% | **0.817** | `model_evaluation` |
| **FPS (On-device)** | ≥15 FPS | **≥20 FPS** | Post-deployment validation |
| **Inference Latency** | ≤300ms | **~35ms** | `model_deploy` |
| **Model Size** | <50MB | **~25MB** | CoreML conversion |

#### Scene Description (Distilled VLM)
| Metric | Target | Achieved | Pipeline Task |
|--------|--------|----------|---------------|
| **CIDER Score** | ≥0.47 | **0.4686** | `desc_model_evaluation` |
| **BLEU-4** | - | Logged | `desc_model_evaluation` |
| **ROUGE-L** | - | Logged | `desc_model_evaluation` |
| **Inference Time** | <1s | Logged | FastAPI endpoint |
| **Caption Length** | <10 words | ✓ | Post-processing validation |

### Experiment Tracking
- **ClearML Dashboard**: Real-time metrics, loss curves, sample predictions
- **Hyperparameter Logging**: All hyperparameters logged per experiment
- **Model Comparison**: Side-by-side comparison of model versions
- **Artifact Storage**: Model checkpoints, training logs, evaluation reports

---

## Environment Setup

### Prerequisites
- **Python**: 3.8+ (recommended: 3.9 or 3.10)
- **ClearML Account**: Sign up at [clear.ml](https://clear.ml) for pipeline orchestration
- **Google Cloud** (optional): For remote agent execution
- **Git**: Version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/SecondSight-MLOps.git
   cd SecondSight-MLOps
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Top-level dependencies
   pip install -r requirements.txt
   
   # Hazard Detection pipeline dependencies
   pip install -r hazard_detection/requirements.txt
   
   # Scene Description pipeline dependencies
   pip install -r image_description/requirements.txt
   ```

4. **Configure ClearML**
   ```bash
   clearml-init
   # Follow prompts to enter API credentials from clear.ml
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

---

## Running Pipelines

### Option 1: Via ClearML UI
1. Navigate to ClearML web UI
2. Go to Pipelines → Create New Pipeline
3. Select `hazard_detection_pipeline` or `scene_description_pipeline`
4. Configure parameters and start execution

### Option 2: Via Python SDK

**Run Hazard Detection Pipeline:**
```bash
cd hazard_detection/pipelines
python detection_pipeline.py \
  --dataset_url "https://path/to/dataset.zip" \
  --epochs 100 \
  --batch_size 16 \
  --queue "default"
```

**Run Scene Description Pipeline:**
```bash
cd image_description/pipelines
python desc_pipeline.py \
  --teacher_model "llava-1.5-7b" \
  --student_model "vit-gpt2" \
  --queue "default"
```

### Option 3: Individual Task Execution

```bash
# Run individual pipeline task
cd hazard_detection/pipelines/tasks
python model_train.py --config configs/train_config.yaml
```

### Remote Agent Setup (Google Cloud)

```bash
# On Google Cloud VM with GPU
pip install clearml-agent
clearml-agent init
clearml-agent daemon --queue default --gpu
```

---

## Project Structure

```
SecondSight-MLOps/
├── .github/
│   └── workflows/                       # GitHub Actions CI/CD
│       ├── deploy_detection.yml         # Auto-deploy hazard detection model
│       └── deploy_description.yml       # Auto-deploy scene description model
├── docs/
│   └── images/                          # Architecture diagrams
├── hazard_detection/                    # Hazard Detection MLOps
│   ├── pipelines/
│   │   ├── detection_pipeline.py        # Main pipeline orchestration
│   │   ├── tasks/                       # ClearML pipeline tasks
│   │   │   ├── dataset_base_upload.py
│   │   │   ├── dataset_base_split.py
│   │   │   ├── model_train.py
│   │   │   ├── model_hpo.py
│   │   │   ├── model_eval.py
│   │   │   ├── model_publish.py
│   │   │   └── model_deploy.py
│   │   └── triggers/                    # Pipeline trigger configurations
│   ├── configs/                         # Training configurations
│   ├── requirements.txt
│   └── README.md
├── image_description/                   # Scene Description MLOps
│   ├── pipelines/
│   │   ├── desc_pipeline.py             # Main pipeline orchestration
│   │   ├── tasks/                       # ClearML pipeline tasks
│   │   │   ├── base_data_preparation.py
│   │   │   ├── base_desc_generation.py
│   │   │   ├── desc_data_split.py
│   │   │   ├── desc_model_train.py
│   │   │   ├── desc_hpo.py
│   │   │   ├── desc_model_eval.py
│   │   │   └── desc_model_publish.py
│   │   └── triggers/
│   ├── model_api_inferencing.py         # FastAPI inference server
│   ├── requirements.txt
│   └── README.md
├── notebooks/                           # Exploratory notebooks & POC
│   ├── data_exploration.py
│   └── model_experiments.ipynb
├── .env                                 # Environment variables (git-ignored)
├── .gitignore
├── requirements.txt                     # Core dependencies
├── setup.py                             # Package setup
└── README.md                            # This file
```

### Key Directories

- **`hazard_detection/pipelines/tasks/`**: Individual ClearML tasks for detection pipeline
- **`image_description/pipelines/tasks/`**: Individual ClearML tasks for description pipeline
- **`.github/workflows/`**: CI/CD automation for model deployment
- **`configs/`**: Training configurations, hyperparameter ranges
- **`notebooks/`**: Exploratory data analysis and POC experiments

For detailed documentation on each component:
- **Hazard Detection Pipeline**: [hazard_detection/README.md](hazard_detection/README.md)
- **Scene Description Pipeline**: [image_description/README.md](image_description/README.md)

---

## CI/CD Workflow

### GitHub Actions Integration

**Deployment Triggers:**
- Manual trigger via GitHub Actions UI
- Automatic on model publish to ClearML registry (webhook)
- Scheduled retraining (weekly/monthly)

**Deployment Pipeline:**
```yaml
# .github/workflows/deploy_detection.yml
name: Deploy Hazard Detection Model

on:
  workflow_dispatch:  # Manual trigger
  repository_dispatch:  # ClearML webhook trigger
    types: [model-published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Download model from ClearML
      - name: Convert PyTorch to CoreML
      - name: Publish to FastAPI GitHub repo
      - name: Build Docker image
      - name: Deploy to AWS ECS
```

### Continuous Integration
- **Linting**: `flake8`, `black` for code formatting
- **Testing**: Unit tests for pipeline tasks
- **Model Validation**: Automated evaluation on test set
- **Registry Checks**: Verify model meets performance thresholds before deployment

---

## Technology Stack

### MLOps & Orchestration
- **ClearML**: Experiment tracking, pipeline orchestration, model registry
- **Google Cloud**: Remote compute agents for distributed training
- **GitHub Actions**: CI/CD automation

### ML Frameworks
- **PyTorch**: Model training (YOLO v11n, VIT-GPT2)
- **Ultralytics YOLO**: Object detection framework
- **Transformers (HuggingFace)**: Vision-language models
- **CoreML Tools**: iOS model conversion

### Deployment & Serving
- **FastAPI**: REST API for model inference
- **Docker**: Containerization
- **AWS ECS**: Cloud deployment (on-demand scaling)
- **GitHub**: Model artifact repository

### Monitoring & Logging
- **ClearML**: Experiment tracking, metrics logging
- **TensorBoard**: Training visualization
- **Prometheus** (planned): Production metrics monitoring

---

## Team

**EnigmaAI** | **SecondSight v0.2**

| Role | Name | Focus Area |
|------|------|------------|
| **Tech Lead & MLOps** | Anna Huang | Pipeline automation, model deployment |
| **Product Owner** | Rozhin Vosoughi | Requirements, stakeholder management |
| **Solution Designer** | Kamatchi Gnanavel | Architecture design, system integration |
| **Data Scientist** | Zoe Lin | Model development, evaluation |

---

## References

### Frameworks & Libraries
- **YOLO v11**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **LLaVA 1.5**: [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- **ClearML**: [ClearML Documentation](https://clear.ml/docs)
- **CoreML**: [Apple CoreML Guide](https://developer.apple.com/documentation/coreml)

### Academic References
- Vision 2020 (2022). *2022-23 Pre-Budget Submission*. [Link](https://treasury.gov.au/sites/default/files/2022-03/258735_vision_2020_australia.pdf)
- WHO (2023). *Blindness and vision impairment Facts*

---

## License

This project is part of an academic assessment (UTS MAI - Advanced Intelligent Systems) for educational purposes.





