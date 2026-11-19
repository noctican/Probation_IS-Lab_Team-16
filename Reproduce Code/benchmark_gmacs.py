"""Measure parameters and GMACs for every trained checkpoint.

CPU is enough: GMACs counts operations in the graph, not wall-clock time, so the
result is identical on CPU and GPU. Run from anywhere:

    python "Reproduce Code/benchmark_gmacs.py"
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "models"
RESULT_PATH = ROOT / "outputs" / "results" / "model_efficiency.csv"
NUM_CLASSES = 30
IMG_SIZE = 224
DEVICE = torch.device("cpu")


def build_torchvision(name: str) -> nn.Module:
    from torchvision import models

    model = getattr(models, name)(weights=None)
    head = model.classifier
    if isinstance(head, nn.Sequential):  # V3: [...Linear]; V2: [Dropout, Linear]
        head[-1] = nn.Linear(head[-1].in_features, NUM_CLASSES)
    else:
        model.classifier = nn.Linear(head.in_features, NUM_CLASSES)
    return model


def build_timm(name: str) -> nn.Module:
    import timm

    return timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)


# checkpoint prefix -> (builder, model id). Order matters only for readability.
FAMILIES = {
    "mobilenetv2": (build_torchvision, "mobilenet_v2"),
    "mobilenetv3small": (build_torchvision, "mobilenet_v3_small"),
    "mobilenetv3large_tv": (build_torchvision, "mobilenet_v3_large"),
    "mobilenetv3large": (build_timm, "mobilenetv3_large_100.ra4_e3600_r224_in1k"),
    "mobilenetv4convs": (build_timm, "mobilenetv4_conv_small.e2400_r224_in1k"),
    "mobilenetv4convsmall_in1k": (build_timm, "mobilenetv4_conv_small.e2400_r224_in1k"),
    "mobilevitxs": (build_timm, "mobilevit_xs.cvnets_in1k"),
    "starnet_s1_in1k": (build_timm, "starnet_s1.in1k"),
    "starnet_s2_in1k": (build_timm, "starnet_s2.in1k"),
    "efficientformerv2_s0_in1k": (build_timm, "efficientformerv2_s0.snap_dist_in1k"),
    "repvit_m0_9_in1k": (build_timm, "repvit_m0_9.dist_450e_in1k"),
}


def measure_gmacs(model: nn.Module) -> float:
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
    return counter.get_total_flops() / 2 / 1e9


def load_state_dict(path: Path):
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # torch < 2.0
        ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return {k.removeprefix("module."): v for k, v in state.items()}


def main() -> None:
    measured = {}
    for checkpoint in sorted(MODEL_DIR.glob("*_best.pt")):
        prefix = next(
            (p for p in sorted(FAMILIES, key=len, reverse=True)
             if checkpoint.name.startswith(p + "_")),
            None,
        )
        if prefix is None:
            continue
        build, model_id = FAMILIES[prefix]
        if model_id in measured:  # same architecture, one measurement is enough
            continue

        model = build(model_id).to(DEVICE)
        model.load_state_dict(load_state_dict(checkpoint))
        model.eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        gmacs = measure_gmacs(model)
        measured[model_id] = {
            "model": prefix,
            "model_id": model_id,
            "parameters_m": params,
            "gmacs": gmacs,
            "checkpoint": checkpoint.name,
        }
        print(f"{prefix:28s} {params:7.3f} M  {gmacs:7.3f} GMACs")

    if not measured:
        raise SystemExit(f"No *_best.pt under {MODEL_DIR}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(next(iter(measured.values()))))
        writer.writeheader()
        writer.writerows(measured.values())
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
