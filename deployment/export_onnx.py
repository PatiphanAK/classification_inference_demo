#!/usr/bin/env python3
"""Export the distilled student checkpoint to ONNX for TensorRT deployment.

Loads the checkpoint written by notebooks/04_distillation.ipynb (same schema as the
baseline from 01), rebuilds the architecture from the metadata recorded inside it, and
exports:

    models/distilled/best.onnx      fixed input 1x3x224x224, logits output
    models/distilled/metadata.json  preprocessing + class mapping for the device side

No training, no architecture changes - the inference graph only. Run this on the machine
that has the checkpoint (dev machine or Colab); the engine itself is built later on the
Jetson with trtexec.

Usage:
    python3 export_onnx.py                                # defaults to models/distilled/best.pt
    python3 export_onnx.py --checkpoint path/to/best.pt
    python3 export_onnx.py --opset 13                     # if the device TRT parser complains
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import torch
from torch import nn


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    """Rebuild the architecture recorded in a project checkpoint (identical to the notebooks)."""
    arch = ckpt["model_arch"]
    num_classes = ckpt["num_classes"]

    if arch == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(weights=None)
        head = cast(nn.Linear, model.classifier[1])
        model.classifier[1] = nn.Linear(head.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        head = cast(nn.Linear, model.classifier[3])
        model.classifier[3] = nn.Linear(head.in_features, num_classes)
    else:
        raise ValueError(f"unknown architecture: {arch}")

    return model


def export_onnx(model: nn.Module, dummy: torch.Tensor, out_path: Path, opset: int) -> None:
    """Export with the fixed 1x3xHxW input the Jetson pipeline expects."""
    args: tuple[torch.Tensor, ...] = (dummy,)
    path = str(out_path)

    try:
        # Prefer the classic exporter: stable graph, no onnxscript dependency.
        torch.onnx.export(
            model,
            args,
            path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # torch without the dynamo kwarg -> plain legacy export
        torch.onnx.export(
            model,
            args,
            path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
        )
    except (RuntimeError, ImportError) as exc:
        # legacy exporter unavailable in this torch build -> fall back to dynamo
        print(f"legacy exporter failed ({exc}); retrying with the dynamo exporter")
        print("(if this fails with a missing module, run: pip install onnxscript)")
        torch.onnx.export(
            model,
            args,
            path,
            input_names=["input"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    project_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "notebooks" / "weight" / "distillation" / "best.pt",
        help="student checkpoint from 04_distillation.ipynb",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "deployment" / "models" / "distilled",
        help="directory for deployment artifacts",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset (17 works with TensorRT 8.5+; use 13 if needed)",
    )

    args = parser.parse_args()

    ckpt_path: Path = args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt_path} "
            "(run notebooks/04_distillation.ipynb first, or point --checkpoint at it)"
        )

    print(f"loading checkpoint : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    preprocessing = ckpt["preprocessing"]
    size = preprocessing["image_size"]

    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"architecture       : {ckpt['model_arch']} "
        f"({params_m:.2f} M params, {ckpt['num_classes']} classes)"
    )

    dummy = torch.randn(1, 3, size, size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "best.onnx"

    print(f"exporting          : {onnx_path} (input 1x3x{size}x{size}, opset {args.opset})")

    export_onnx(model, dummy, onnx_path, args.opset)

    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print("onnx checker       : PASS")
    except ImportError:
        print("onnx checker       : skipped (pip install onnx to enable)")

    metrics = ckpt.get("metrics", {})
    metadata = {
        "model_arch": ckpt["model_arch"],
        "num_classes": ckpt["num_classes"],
        "classes": ckpt["classes"],
        "preprocessing": preprocessing,
        "input_shape": [1, 3, size, size],
        "opset": args.opset,
        "dataset": ckpt.get("dataset", "stanford_cars"),
        "metrics": {k: metrics[k] for k in ("val_top1", "val_top5") if k in metrics},
        "source_checkpoint": str(ckpt_path),
        "torch_version": ckpt.get("torch_version", torch.__version__),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"saved {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"saved {meta_path} (classes + preprocessing for the device side)")
    print()
    print("next: validate with validate_onnx.py, then deploy (see deployment/README.md)")


if __name__ == "__main__":
    main()
