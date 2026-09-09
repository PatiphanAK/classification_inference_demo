We are building an undergraduate computer vision project called `detection_inference_demo`.

For this step, **ONLY work on the `notebooks/` directory.**

Do not create or modify `src/`, `deployment/`, TensorRT code, ONNX code, Knowledge Distillation code, or Jetson deployment code yet.

## Project Context

The overall project will eventually investigate an edge-oriented vehicle vision pipeline:

```text
Public Vehicle Dataset
        ↓
Model Training
        ↓
PyTorch Baseline Weight
        ↓
Video Inference
        ↓
Knowledge Distillation
        ↓
Student Model
        ↓
ONNX
        ↓
TensorRT
        ↓
Jetson Orin Nano
        ↓
Real-time Video Inference
```

The final inference use case is **video**, not single-image inference.

The eventual video pipeline will be approximately:

```text
Input Video
    ↓
Vehicle Detection
    ↓
Vehicle Crop
    ↓
Vehicle Classification
    ↓
Tracking / Frame Processing
    ↓
Annotated Output Video
```

However, we will build this incrementally.

---

# Current Phase: Notebook-based Experimentation

For now, create a clean notebook workflow for the machine learning experiments.

Use a public vehicle dataset.

The preferred dataset is:

**Stanford Cars / Cars196**

The dataset contains approximately:

* 16,185 images
* 196 vehicle classes
* 8,144 training images
* 8,041 test images

The task for the initial experiment is **fine-grained vehicle classification**.

The dataset itself must NOT be committed to Git.

The notebook should assume that the dataset has been downloaded separately and that its location can be configured.

Do not depend on `torchvision.datasets.StanfordCars(download=True)` because the original Stanford download URL is no longer reliably available.

---

# Notebook Structure

For now, create only:

```text
notebooks/
├── 01_training.ipynb
├── 02_evaluation.ipynb
└── 03_video_inference.ipynb
```

Do not implement later optimization notebooks yet.

The progression should be:

```text
01_training
      ↓
02_evaluation
      ↓
03_video_inference
```

---

# 01_training.ipynb

This notebook establishes the **PyTorch baseline model**.

Sections should be clearly separated with Markdown cells.

Recommended structure:

### 1. Environment

* Python version
* PyTorch version
* torchvision version
* CUDA availability
* GPU name
* random seed

### 2. Configuration

Define variables such as:

```text
DATASET_ROOT
IMAGE_SIZE
BATCH_SIZE
NUM_EPOCHS
LEARNING_RATE
NUM_WORKERS
NUM_CLASSES
DEVICE
```

Do not hard-code machine-specific paths.

### 3. Dataset

Load Stanford Cars.

Show:

* number of training samples
* number of test samples
* number of classes
* class names

Visualize several training examples.

### 4. Preprocessing

Implement reasonable:

* resize
* normalization
* training augmentation
* validation/test preprocessing

Keep the preprocessing configuration reusable because the same preprocessing will later be required during inference.

### 5. Model

Use a pretrained ImageNet classification model.

Start with a lightweight architecture suitable for future edge deployment.

A good initial choice is:

**EfficientNet-B0**

Replace the classification head with:

```text
196 classes
```

Keep the model definition simple.

### 6. Training

Implement a normal supervised training loop.

Track:

* training loss
* validation loss
* Top-1 accuracy
* Top-5 accuracy

Use the best validation metric to determine the best checkpoint.

### 7. Visualization

Plot:

* training loss
* validation loss
* Top-1 accuracy
* Top-5 accuracy

### 8. Save checkpoint

Save the best PyTorch checkpoint under:

```text
models/baseline/best.pt
```

The checkpoint should contain enough metadata to reproduce inference, including:

* model architecture
* model state dict
* class mapping
* preprocessing information
* training configuration
* evaluation metrics

Do NOT commit the actual `.pt` file to Git unless explicitly requested.

### 9. Sanity-check inference

Run the trained model on several test images.

Display:

```text
Ground Truth
Predicted Class
Confidence
```

This is only a sanity check.

The project's actual application-level inference will be video-based and will be implemented in `03_video_inference.ipynb`.

---

# 02_evaluation.ipynb

This notebook should focus on **evaluating the trained baseline**, not retraining it.

Load:

```text
models/baseline/best.pt
```

Evaluate the model on the test set.

Include:

* Top-1 accuracy
* Top-5 accuracy
* per-class metrics where practical
* confusion matrix
* representative correct predictions
* representative incorrect predictions

Also calculate basic model information:

* number of parameters
* model size
* inference latency on the current device if practical

Keep the evaluation results reproducible.

This notebook will later become the baseline against which the distilled and TensorRT versions are compared.

---

# 03_video_inference.ipynb

This notebook is specifically for **video inference**.

Do NOT treat image inference as the main use case.

The notebook should eventually implement:

```text
Video
 ↓
Frame
 ↓
Vehicle Detection
 ↓
Vehicle Crop
 ↓
Vehicle Classification
 ↓
Draw Results
 ↓
Output Video
```

For the first version, use an existing pretrained vehicle/object detector rather than training a detector from scratch.

The classifier trained in `01_training.ipynb` should be loaded from:

```text
models/baseline/best.pt
```

The notebook should support:

```text
INPUT_VIDEO
OUTPUT_VIDEO
```

and process the video frame by frame.

The output should be an annotated video showing detected vehicles and the predicted vehicle class.

Keep the detector and classifier conceptually separate.

For example:

```text
detector(frame)
    ↓
bounding boxes
    ↓
crop vehicle
    ↓
classifier(crop)
    ↓
class + confidence
    ↓
draw annotation
```

At this stage, basic frame-by-frame processing is sufficient.

Do not optimize for TensorRT yet.

---

# Important Scope Boundary

For this task, do NOT implement:

* Knowledge Distillation
* Teacher/Student training
* Quantization
* ONNX export
* TensorRT
* TensorRT engine generation
* Jetson optimization
* distributed inference
* production API
* Docker deployment

Those will be later phases.

The intended future progression is:

```text
Phase 1
Notebook Training
        ↓
Baseline Weight

Phase 2
Evaluation
        ↓
Baseline Metrics

Phase 3
Video Inference
        ↓
Baseline Video Pipeline

Phase 4
Knowledge Distillation
        ↓
Student Weight

Phase 5
ONNX + TensorRT
        ↓
Optimized Model

Phase 6
Jetson Orin Nano
        ↓
Real-time Video Benchmark
```

## Coding Rules

Before editing:

1. Inspect the existing repository.
2. Preserve existing work.
3. Only create/modify files under `notebooks/` for this task.
4. Do not add unnecessary dependencies unless required.
5. Keep notebooks readable and educational.
6. Use Markdown sections to explain what each experiment is doing.
7. Avoid hidden state between notebooks.
8. Each notebook should be understandable when opened independently.
9. Do not commit datasets, videos, generated outputs, or model weights.
10. Prefer a clean experimental workflow over premature software abstraction.

After implementation, report:

* notebooks created
* dataset setup required
* dependencies required
* how to run each notebook
* expected outputs
* assumptions
* any limitations
* recommended next phase
