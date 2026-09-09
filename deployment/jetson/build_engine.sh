#!/usr/bin/env bash
# Build a TensorRT engine from the exported ONNX model - run ON the Jetson Orin Nano.
#
# TensorRT engines are device-specific (GPU chip, JetPack/TensorRT version): always build
# on the device you will run on, never copy an engine from an x86 machine.
#
# Usage:
#   bash build_engine.sh [onnx_file] [engine_file]
#   PRECISION=fp32 bash build_engine.sh        # default is fp16 (recommended on Orin)
#
# JetPack 5.1.x and 6.x both ship trtexec on PATH.
set -euo pipefail

ONNX="${1:-best.onnx}"
ENGINE="${2:-${ONNX%.onnx}.engine}"
PRECISION="${PRECISION:-fp16}"   # fp16 (default, uses Orin tensor cores) or fp32

if [ ! -f "$ONNX" ]; then
    echo "error: $ONNX not found - copy the exported ONNX (and metadata.json) to this device first" >&2
    exit 1
fi

FLAGS=(--onnx="$ONNX" --saveEngine="$ENGINE")
if [ "$PRECISION" = "fp16" ]; then
    FLAGS+=(--fp16)
fi

echo "building $ENGINE from $ONNX (precision: $PRECISION) ..."
trtexec "${FLAGS[@]}"
echo
echo "done: $ENGINE"
echo "next: python3 infer_trt.py --engine $ENGINE --metadata metadata.json --image <test.jpg>"
