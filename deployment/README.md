# Deployment: Distilled Student -> ONNX -> TensorRT -> Jetson Orin Nano

Phase 5 of the pipeline. Takes the student trained in `notebooks/04_distillation.ipynb`
and prepares it for real-time inference on the edge device. No training, no architecture
changes, no further tuning - this is the end-to-end demo.

```text
notebooks/weight/distillation/best.pt
        |
        | export_onnx.py
        v
deployment/models/distilled/best.onnx + metadata.json
        |
        | copy to Jetson
        v
best.engine
        |
        | infer_trt.py
        v
prediction + latency/FPS
```

## 1. Export (dev machine or Colab)

The deployment environment is managed with `uv`.

Requirements:

* `uv`
* `torch`
* `torchvision`
* `onnx` (optional, for the graph check)

Install/sync the deployment environment:

```bash
cd deployment
uv sync
```

Export the student checkpoint:

```bash
uv run python export_onnx.py
```

Or specify the checkpoint explicitly:

```bash
uv run python export_onnx.py \
  --checkpoint ../notebooks/weight/distillation/best.pt
```

Produces:

* `models/distilled/best.onnx` - inference graph, fixed input
  `1x3x224x224`, output `logits` (196), opset 17
* `models/distilled/metadata.json` - the **exact preprocessing**
  (image size, mean, std, interpolation) and the **class-index mapping**
  (`classes[i] = label i`), plus model/metrics information

The Jetson side reads the preprocessing and class mapping from `metadata.json`,
so the device does not need to guess these values.

If the TensorRT parser has compatibility issues with opset 17, re-export with:

```bash
uv run python export_onnx.py --opset 13
```

## 2. Validate (dev machine or Colab)

ONNX validation uses `onnxruntime`.

```bash
uv add onnxruntime
```

Validate against a test image:

```bash
uv run python validate_onnx.py --image path/to/car.jpg
```

Or pick a test image from the dataset:

```bash
uv run python validate_onnx.py \
  --dataset-root /content/data/stanford_cars
```

The validator compares:

* PyTorch prediction
* ONNX prediction
* maximum logit difference
* top-1 agreement
* top-5 agreement

Exit code `0` = PASS:

* same top-1 prediction
* logits within `1e-2`

## 3. Jetson Orin Nano

Target environment:

* Ubuntu 22.04
* ARM64
* JetPack 5.1.x or 6.x
* TensorRT / `trtexec` provided by JetPack

Verify TensorRT:

```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
which trtexec
```

Install `uv` if it is not already available, then sync the deployment
environment:

```bash
cd ~/object_track/deployment
uv sync
```

The Jetson-side Python dependencies are managed by `uv`.

If `pycuda`, `numpy`, or `pillow` are not already declared in
`deployment/pyproject.toml`, add them with:

```bash
uv add pycuda numpy pillow
```

> TensorRT itself is provided by the JetPack installation and should not be
> installed through `uv`.

### Copy deployment artifacts to the Jetson

From the machine containing the exported model:

```bash
scp deployment/models/distilled/best.onnx \
    deployment/models/distilled/metadata.json \
    orin_nano@<orin-ip>:~/object_track/deployment/
```

Copy the Jetson deployment scripts:

```bash
scp deployment/jetson/build_engine.sh \
    deployment/jetson/infer_trt.py \
    orin_nano@<orin-ip>:~/object_track/deployment/jetson/
```

Copy a test image:

```bash
scp one_test_image.jpg \
    orin_nano@<orin-ip>:~/object_track/deployment/
```

## 4. Build TensorRT Engine on Jetson

TensorRT engines are device-specific, so the engine must be built **on the
Jetson**.

```bash
cd ~/object_track/deployment
bash jetson/build_engine.sh models/distilled/best.onnx
```

FP16 is enabled by default.

To disable FP16 and build FP32:

```bash
PRECISION=fp32 bash jetson/build_engine.sh models/distilled/best.onnx
```

The resulting engine should be:

```text
models/distilled/best.engine
```

## 5. Run TensorRT Inference

Run inference using the generated engine and metadata:

```bash
cd ~/object_track/deployment

uv run python jetson/infer_trt.py \
  --engine models/distilled/best.engine \
  --metadata models/distilled/metadata.json \
  --image one_test_image.jpg
```

Expected output:

* predicted Stanford Cars class
* confidence
* top-5 predictions
* mean latency (ms/img)
* FPS

The latency benchmark runs 100 iterations and includes host-to-device and
device-to-host copies.

## Acceptance Checklist

| # | Criterion                         | Where                                       |
| - | --------------------------------- | ------------------------------------------- |
| 1 | Student checkpoint loads          | `export_onnx.py` (strict `load_state_dict`) |
| 2 | ONNX export succeeds              | `export_onnx.py` + ONNX checker             |
| 3 | ONNX prediction matches PyTorch   | `validate_onnx.py` (exit code 0)            |
| 4 | TensorRT engine created on Jetson | `build_engine.sh`                           |
| 5 | Valid Stanford Cars prediction    | `infer_trt.py` (class + confidence)         |
| 6 | Basic latency/FPS recorded        | `infer_trt.py`                              |

## End-to-End Flow

```text
04_distillation.ipynb
        |
        v
notebooks/weight/distillation/best.pt
        |
        | uv run python export_onnx.py
        v
deployment/models/distilled/
├── best.onnx
└── metadata.json
        |
        | validate_onnx.py
        v
ONNX == PyTorch
        |
        | copy to Jetson
        v
Jetson Orin Nano
        |
        | trtexec
        v
best.engine
        |
        | uv run python jetson/infer_trt.py
        v
Prediction + Confidence + Top-5 + Latency/FPS
```

Stop here - this completes the demo.

Distillation quality is judged with
`notebooks/02_evaluation.ipynb`
(`CHECKPOINT_PATH=../models/distilled/best.pt`).

This deployment folder only moves the student model through:

**PyTorch -> ONNX -> TensorRT -> Jetson inference**
