# Deployment: Distilled Student -> ONNX -> TensorRT -> Jetson Orin Nano

Phase 5 of the pipeline. Takes the student trained in `notebooks/04_distillation.ipynb`
and prepares it for real-time inference on the edge device. No training, no architecture
changes, no further tuning - this is the end-to-end demo.

```
models/distilled/best.pt                      (notebooks/04_distillation.ipynb)
        |  export_onnx.py                     (dev machine or Colab)
        v
models/distilled/best.onnx + metadata.json    (fixed input 1x3x224x224)
        |  copy to the Jetson (scp / rsync / USB)
        v
best.engine                                   (trtexec, built ON the Jetson)
        |  infer_trt.py
        v
prediction + latency/FPS
```

## 1. Export (dev machine or Colab)

Requirements: `torch`, `torchvision` (same versions as training ideally), `onnx`
(optional, for the graph check).

```bash
cd deployment
python3 export_onnx.py --checkpoint ../models/distilled/best.pt
```

Produces, next to the checkpoint:

- `best.onnx` - inference graph, fixed input `1x3x224x224`, output `logits` (196), opset 17
  (re-export with `--opset 13` if the device's TensorRT parser ever complains)
- `metadata.json` - the **exact preprocessing** (image size, mean, std, interpolation) and
  the **class-index mapping** (`classes[i]` = label i), plus model/metrics info. The Jetson
  side reads everything from here, so the device never guesses.

## 2. Validate (dev machine or Colab)

Requirements: `onnxruntime` (+ the export requirements).

```bash
python3 validate_onnx.py --image path/to/car.jpg
# or pick a test image from the dataset:
python3 validate_onnx.py --dataset-root /content/data/stanford_cars
```

Prints the PyTorch and ONNX predictions for the same image, the max logit difference, and
top-1/top-5 agreement. Exit code 0 = PASS (same top-1, logits within 1e-2).

## 3. Jetson Orin Nano (Ubuntu 22.04, ARM64)

Prerequisites:

- JetPack 5.1.x or 6.x - both ship TensorRT and `trtexec` preinstalled. Verify:

  ```bash
  python3 -c "import tensorrt; print(tensorrt.__version__)"
  which trtexec
  ```

- Python deps (TensorRT is already there):

  ```bash
  pip3 install pycuda numpy pillow
  ```

Copy to the device (from the machine with the exports):

```bash
scp models/distilled/best.onnx models/distilled/metadata.json jetson@<orin-ip>:~/
scp deployment/jetson/build_engine.sh deployment/jetson/infer_trt.py jetson@<orin-ip>:~/
scp one_test_image.jpg jetson@<orin-ip>:~/
```

Build the engine **on the device** (engines are device-specific - never build on x86 and
copy over):

```bash
bash build_engine.sh best.onnx           # fp16 by default; PRECISION=fp32 to disable
```

Run inference + latency/FPS:

```bash
python3 infer_trt.py --engine best.engine --metadata metadata.json --image one_test_image.jpg
```

Expected output: the predicted Stanford Cars class with confidence, the top-5 list, and
mean latency (ms/img) + FPS over 100 iterations (including host<->device copies).

## Acceptance checklist

| # | criterion | where |
|---|---|---|
| 1 | student checkpoint loads | `export_onnx.py` (strict `load_state_dict`) |
| 2 | ONNX export succeeds | `export_onnx.py` (+ onnx checker) |
| 3 | ONNX prediction matches PyTorch | `validate_onnx.py` (exit code 0) |
| 4 | TensorRT engine created on Jetson | `build_engine.sh` |
| 5 | valid Stanford Cars prediction | `infer_trt.py` (class + confidence) |
| 6 | basic latency/FPS recorded | `infer_trt.py` |

Stop here - this completes the demo. Distillation quality is judged with
`notebooks/02_evaluation.ipynb` (`CHECKPOINT_PATH=../models/distilled/best.pt`); this
folder only moves the student onto the device.
