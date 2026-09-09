#!/usr/bin/env python3
"""Minimal TensorRT inference for the distilled student - Jetson Orin Nano.

Runs the TensorRT engine built by build_engine.sh on one test image, prints the predicted
Stanford Cars class + confidence, then measures basic latency/FPS. That is all - this is
the end of the demo pipeline, not a benchmark suite.

Preprocessing is reimplemented here with numpy/PIL only (no torch/torchvision needed on
the device) and is bit-identical to training-time eval: PIL bilinear resize to
image_size x image_size -> /255 -> ImageNet normalization -> CHW. All of it comes from
metadata.json exported next to the ONNX model, so the device never guesses.

Works on JetPack 5 (TensorRT 8.x) and JetPack 6 (TensorRT 10.x).

Usage (on the Jetson):
    python3 infer_trt.py --engine best.engine --metadata metadata.json --image test.jpg
    python3 infer_trt.py --engine best.engine --metadata metadata.json --image test.jpg \
                         --warmup 20 --iters 100

Requires: tensorrt (preinstalled with JetPack), pycuda, numpy, pillow.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- preprocessing
def preprocess(image_path, metadata):
    """Image -> float32 1x3xHxW, identical to the notebooks' eval transform."""
    pp = metadata["preprocessing"]
    size = pp["image_size"]
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.asarray(pp["mean"], dtype=np.float32)) / np.asarray(pp["std"], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def softmax_np(logits):
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------- TensorRT glue
def load_trt(engine_path):
    """Deserialize the engine and return (engine, context, input_name, output_name, out_shape).

    tensorrt/pycuda are imported lazily so this file can be imported off-device
    (e.g. for preprocessing tests) without them installed.
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (creates the CUDA context)

    trt10 = int(trt.__version__.split(".")[0]) >= 10
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("failed to deserialize engine: %s" % engine_path)
    context = engine.create_execution_context()

    if trt10:  # TensorRT 10 (JetPack 6): tensor-name API
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        out_shape = tuple(engine.get_tensor_shape(outputs[0]))
    else:      # TensorRT 8 (JetPack 5): binding-index API
        inputs = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
        outputs = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
        out_shape = tuple(engine.get_binding_shape(engine.get_binding_index(outputs[0])))

    return trt, cuda, engine, context, inputs[0], outputs[0], out_shape


def run_inference(trt, cuda, context, input_name, output_name, out_shape, x):
    """One synchronous inference: H2D -> execute -> D2H."""
    d_input = cuda.mem_alloc(x.nbytes)
    output = np.empty(out_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    stream = cuda.Stream()

    trt10 = int(trt.__version__.split(".")[0]) >= 10
    if trt10:
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        execute = lambda: context.execute_async_v3(stream.handle)
    else:
        bindings = [0] * context.engine.num_bindings
        bindings[context.engine.get_binding_index(input_name)] = int(d_input)
        bindings[context.engine.get_binding_index(output_name)] = int(d_output)
        execute = lambda: context.execute_async_v2(bindings, stream.handle)

    cuda.memcpy_htod_async(d_input, x, stream)
    execute()
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    return output[0]


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True, help="TensorRT engine from build_engine.sh")
    parser.add_argument("--metadata", default="metadata.json", help="sidecar JSON from export_onnx.py")
    parser.add_argument("--image", required=True, help="one test image")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100, help="timed iterations for latency/FPS")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text())
    classes = metadata["classes"]
    print("engine   :", args.engine)
    print("model    :", metadata["model_arch"], "-", metadata["num_classes"], "classes")
    print("tensorrt : TensorRT", end=" ")
    import tensorrt as trt  # safe here: load_trt would fail anyway without it
    print(trt.__version__, "| arch:", platform.machine())
    print()

    trt, cuda, engine, context, input_name, output_name, out_shape = load_trt(args.engine)
    x = preprocess(args.image, metadata)
    expected = tuple(metadata["input_shape"])
    if x.shape != expected:
        raise SystemExit("input shape mismatch: engine wants %s, got %s" % (expected, x.shape))

    # --- single prediction (acceptance: a valid Stanford Cars prediction) ---
    logits = run_inference(trt, cuda, context, input_name, output_name, out_shape, x)
    probs = softmax_np(logits)
    pred = int(probs.argmax())
    print("image      :", args.image)
    print("prediction : %s (%.2f%% confidence)" % (classes[pred], probs[pred] * 100))
    print("top-5      :")
    for idx in np.argsort(probs)[::-1][:5]:
        print("   %5.2f%%  %s" % (probs[idx] * 100, classes[int(idx)]))
    print()

    # --- basic latency / FPS (acceptance: record numbers, nothing more) ---
    for _ in range(args.warmup):
        run_inference(trt, cuda, context, input_name, output_name, out_shape, x)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_inference(trt, cuda, context, input_name, output_name, out_shape, x)
    elapsed = time.perf_counter() - t0

    ms_per_img = elapsed / args.iters * 1000.0
    print("latency    : %.2f ms/img (mean over %d iters, incl. host<->device copies)" % (ms_per_img, args.iters))
    print("throughput : %.1f FPS (batch=1)" % (1000.0 / ms_per_img))
    print("precision  : as built by build_engine.sh (default fp16)")


if __name__ == "__main__":
    main()
