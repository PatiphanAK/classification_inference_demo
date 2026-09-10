# Vehicle Classification and Inference

Vehicle classification and inference developed as a demonstration project
by undergraduate students from the School of Information Technology,
King Mongkut's Institute of Technology Ladkrabang (KMITL).

## Overview

This project demonstrates an end-to-end vehicle image classification
pipeline, covering model training, evaluation, knowledge distillation,
model export, and edge-device inference.

The project uses the Stanford Cars dataset to classify vehicles into
196 different classes.

## Pipeline

```text
Dataset
   │
   ▼
Training
   │
   ▼
Teacher Model
   │
   ▼
Knowledge Distillation
   │
   ▼
Student Model
   │
   ▼
ONNX Export
   │
   ▼
TensorRT FP16
   │
   ▼
NVIDIA Jetson Orin Nano
   │
   ▼
Real-time Inference
```
