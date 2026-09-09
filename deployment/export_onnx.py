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
import argparse
import inspect
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


def build_model_from_checkpoint(ckpt):
    """Rebuild the architecture recorded in a project checkpoint (identical to the notebooks)."""
    arch = ckpt["model_arch"]
    if arch == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, ckpt["num_classes"])
    elif arch == "mobilenet_v3_small":
        from torchvision.models import mobilenet_v3_small
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, ckpt["num_classes"])
    else:
        raise ValueError("unknown architecture: " + str(arch))
    return model


def export_onnx(model, dummy, out_path, opset):
    """Export with the fixed 1x3xHxW input the Jetson pipeline expects."""
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
    )
    try:
        # Prefer the classic exporter: stable graph, no onnxscript dependency.
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        # torch without the dynamo kwarg
        torch.onnx.export(model, dummy, str(out_path), **kwargs)
    except Exception as exc:  # legacy exporter unavailable in this torch build
        print("legacy exporter failed (%s); retrying with the dynamo exporter" % exc)
        print("(if this fails with a missing module, run: pip install onnxscript)")
        torch.onnx.export(model, dummy, str(out_path), dynamo=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/distilled/best.pt",
                        help="student checkpoint from 04_distillation.ipynb")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset (17 works with TensorRT 8.5+; fall back to 13 if needed)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit("checkpoint not found: %s (run notebooks/04_distillation.ipynb first, "
                         "or point --checkpoint at it)" % ckpt_path)

    print("loading checkpoint :", ckpt_path)
    # weights_only=False: the checkpoint holds plain metadata alongside tensors and is
    # produced by our own notebooks (torch >= 2.6 defaults to weights_only=True, which
    # rejects even the stored torch_version object).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    preprocessing = ckpt["preprocessing"]
    size = preprocessing["image_size"]

    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])   # strict=True: schema must match exactly
    model.eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print("architecture       : %s (%.2f M params, %d classes)" % (ckpt["model_arch"], params_m, ckpt["num_classes"]))

    dummy = torch.randn(1, 3, size, size)
    onnx_path = ckpt_path.with_suffix(".onnx")
    print("exporting          : %s (input 1x3x%dx%d, opset %d)" % (onnx_path, size, size, args.opset))
    export_onnx(model, dummy, onnx_path, args.opset)

    # Sanity-check the exported graph if the onnx package is available.
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print("onnx checker       : PASS")
    except ImportError:
        print("onnx checker       : skipped (pip install onnx to enable)")

    # Sidecar metadata: everything the device needs to reproduce training-time inference
    # without the .pt checkpoint - exact preprocessing and the class-index mapping.
    metadata = {
        "model_arch": ckpt["model_arch"],
        "num_classes": ckpt["num_classes"],
        "classes": ckpt["classes"],                 # index -> class name
        "preprocessing": preprocessing,             # image_size / mean / std / interpolation
        "input_shape": [1, 3, size, size],
        "opset": args.opset,
        "dataset": ckpt.get("dataset", "stanford_cars"),
        "metrics": {k: ckpt["metrics"][k] for k in ("val_top1", "val_top5") if k in ckpt.get("metrics", {})},
        "source_checkpoint": str(ckpt_path),
        "torch_version": ckpt.get("torch_version", torch.__version__),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = ckpt_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print()
    print("saved %s (%.1f MB)" % (onnx_path, onnx_path.stat().st_size / 1e6))
    print("saved %s (classes + preprocessing for the device side)" % meta_path)
    print()
    print("next: validate with validate_onnx.py, then deploy (see deployment/README.md)")


if __name__ == "__main__":
    main()
