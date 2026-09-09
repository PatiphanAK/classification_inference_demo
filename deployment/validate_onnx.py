#!/usr/bin/env python3
"""Validate the exported ONNX student against the PyTorch checkpoint on one test image.

Loads one test image, preprocesses it exactly like the notebooks (PIL bilinear resize ->
ToTensor -> ImageNet normalization, taken from the checkpoint metadata), then runs BOTH
the PyTorch student and the ONNX model on the same input and compares:

    * predicted class + confidence (printed for both)
    * max |logit| difference
    * top-1 / top-5 agreement

Exit code 0 = PASS (same top-1 prediction, logits within tolerance), 1 = FAIL.

Usage:
    python3 validate_onnx.py --image path/to/car.jpg
    python3 validate_onnx.py --dataset-root /content/data/stanford_cars   # picks a test image
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export_onnx import build_model_from_checkpoint  # same rebuild logic as the exporter


def build_eval_transform(preprocessing):
    """Eval transform rebuilt from checkpoint metadata - identical to the notebooks."""
    from torchvision import transforms
    size = preprocessing["image_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(preprocessing["mean"], preprocessing["std"]),
    ])


def pick_test_image(root):
    """Pick the first test image from a Stanford Cars checkout (either layout)."""
    root = Path(root)
    if (root / "cars_annos.mat").exists():  # devkit layout
        from scipy.io import loadmat
        for a in loadmat(root / "cars_annos.mat")["annotations"].ravel():
            if bool(int(np.asarray(a["test"]).ravel()[0])):
                return root / str(np.asarray(a["relative_im_path"]).ravel()[0])
        raise FileNotFoundError("no labeled test images in " + str(root))
    test_dir = root / "test"                # class-folder layout
    if test_dir.is_dir():
        for class_dir in sorted(test_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                files = sorted(class_dir.glob(pattern))
                if files:
                    return files[0]
    raise FileNotFoundError("no test image found under %s - pass --image explicitly" % root)


def softmax_np(logits):
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/distilled/best.pt")
    parser.add_argument("--onnx", default=None, help="default: next to the checkpoint (.onnx)")
    parser.add_argument("--image", default=None, help="test image; default: first image of --dataset-root")
    parser.add_argument("--dataset-root", default=None, help="Stanford Cars root to pick a test image from")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="max |logit| difference for PASS")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx) if args.onnx else ckpt_path.with_suffix(".onnx")
    if not ckpt_path.exists():
        raise SystemExit("checkpoint not found: %s" % ckpt_path)
    if not onnx_path.exists():
        raise SystemExit("ONNX model not found: %s (run export_onnx.py first)" % onnx_path)

    import onnxruntime as ort

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own checkpoint
    metadata = json.loads(onnx_path.with_suffix(".json").read_text())
    classes = ckpt["classes"]
    assert classes == metadata["classes"], "class mapping in metadata.json does not match the checkpoint"

    # --- PyTorch side ---
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- input: one test image, preprocessing identical to training ---
    image_path = Path(args.image) if args.image else pick_test_image(args.dataset_root)
    from PIL import Image
    transform = build_eval_transform(ckpt["preprocessing"])
    tensor = transform(Image.open(image_path).convert("RGB"))

    with torch.no_grad():
        pt_logits = model(tensor[None]).numpy()[0]

    # --- ONNX side, same input ---
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: tensor[None].numpy()})[0][0]

    # --- compare ---
    pt_pred, onnx_pred = int(pt_logits.argmax()), int(onnx_logits.argmax())
    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    top5_agree = set(pt_logits.argsort()[-5:].tolist()) == set(onnx_logits.argsort()[-5:].tolist())

    print("image             :", image_path)
    print()
    print("PyTorch student   : %-45s %5.2f%%" % (classes[pt_pred], softmax_np(pt_logits)[pt_pred] * 100))
    print("ONNX model        : %-45s %5.2f%%" % (classes[onnx_pred], softmax_np(onnx_logits)[onnx_pred] * 100))
    print()
    print("max |logit| diff  : %.2e (tolerance %.0e)" % (max_diff, args.tolerance))
    print("top-1 agreement   :", pt_pred == onnx_pred)
    print("top-5 agreement   :", top5_agree)

    passed = (pt_pred == onnx_pred) and max_diff < args.tolerance
    print()
    print("VERDICT           :", "PASS - ONNX matches PyTorch" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
