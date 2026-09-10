#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX file using the Python API.

Use this when trtexec is not shipped with the JetPack TensorRT packages
(common on JetPack 6.x / TensorRT 10 for Jetson Orin Nano).

Usage:
    python3 build_engine.py [onnx_file] [engine_file]
    PRECISION=fp32 python3 build_engine.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import tensorrt as trt  # pyright: ignore[reportMissingImports]

PRECISION = os.environ.get("PRECISION", "fp16").lower()
WORKSPACE_GB = int(os.environ.get("WORKSPACE_GB", "1"))
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build(onnx_path: Path, engine_path: Path) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"parsing {onnx_path} ...")
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  parser error: {parser.get_error(i)}", file=sys.stderr)
            raise SystemExit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30
    )

    if PRECISION == "fp16":
        if not builder.platform_has_fast_fp16:
            print("warning: platform has no fast fp16 path", file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"building engine (precision: {PRECISION}, workspace: {WORKSPACE_GB} GB) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build failed - see TensorRT log above")

    engine_path.write_bytes(serialized)
    print(f"done: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_onnx = script_dir.parent / "models" / "distilled" / "best.onnx"

    onnx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_onnx
    engine_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else onnx_path.with_suffix(".engine")
    )

    if not onnx_path.exists():
        raise SystemExit(f"error: {onnx_path} not found")

    build(onnx_path, engine_path)


if __name__ == "__main__":
    main()
