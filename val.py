#!/usr/bin/env python3
"""Canonical validation entry that reproduces the in-training protocol.

This script uses the validated training-aligned path rather than the
standard AutoBackend validation wrapper.
It does not instantiate ``AutoBackend`` and therefore does not fuse Conv-BN
layers.  More importantly, the rectangular validation loader is aligned to
the stride of the actual Detect head rather than a possibly stale top-level
``model.stride`` attribute stored in a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml
from torch.nn.modules.batchnorm import _BatchNorm

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import torch_safe_load
from ultralytics.utils.torch_utils import init_seeds, select_device


ROOT = Path(__file__).resolve().parent
DATASET_ALIASES = {
    "d4la": ("D4LA.yaml", "D4LA"),
    "m6doc": ("m6doc.yaml", "m6doc"),
    "hjdataset": ("HJDataset.yaml", "HJDataset"),
    "prima-lad": ("prima-lad.yaml", "prima-lad"),
    "prima_lad": ("prima-lad.yaml", "prima-lad"),
}
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map": "metrics/mAP50-95(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local checkpoint with the in-training protocol: "
            "rectangular batches using the true Detect stride, FP16, and no fusion."
        )
    )
    parser.add_argument("--data", required=True, help="Existing dataset YAML or supported alias.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="External dataset parent directory for alias mode (or DOC_LAYOUT_DATA_ROOT).",
    )
    parser.add_argument("--model", required=True, type=Path, help="Local .pt checkpoint path.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input size. Defaults to the value stored in the checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Validation batch size; the target DDP runs used 16 per validation process.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Training worker count. Validation uses twice this value, as DetectionTrainer does.",
    )
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--rect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use rectangular validation batches.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate in FP16. Pass --no-half for FP32.",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/train-aligned-val"))
    parser.add_argument("--name")
    parser.add_argument(
        "--tolerance-pp",
        type=float,
        default=0.01,
        help="Allowed mAP50-95 difference from the checkpoint metric, in percentage points.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if the checkpoint metric is not reproduced within tolerance.",
    )
    return parser.parse_args()


def resolve_data_config(data: str, dataset_root: Path | None) -> tuple[Path, Path | None]:
    candidate = Path(data).expanduser()
    if candidate.is_file():
        return candidate.resolve(), None
    alias = DATASET_ALIASES.get(data.lower())
    filename = alias[0] if alias else data
    if not filename.lower().endswith((".yaml", ".yml")):
        filename += ".yaml"
    config = ROOT / "ultralytics" / "cfg" / "datasets" / filename
    if not config.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {data!r}. "
            "Pass an existing YAML path or a supported alias."
        )
    if alias is None:
        return config, None
    configured_root = dataset_root
    if configured_root is None:
        env_root = os.environ.get("DOC_LAYOUT_DATA_ROOT")
        configured_root = Path(env_root).expanduser() if env_root else None
    if configured_root is None:
        # Default location: datasets/ in the project root (see README, Dataset Preparation).
        configured_root = ROOT / "datasets"
    return config, configured_root.resolve() / alias[1]


def materialize_data_config(
    data: str, dataset_root: Path | None
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    config_path, resolved_dataset_root = resolve_data_config(data, dataset_root)
    if resolved_dataset_root is None:
        return config_path, None
    if not resolved_dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {resolved_dataset_root}")
    temporary_dir = tempfile.TemporaryDirectory(prefix="doclayout_train_aligned_val_")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["path"] = str(resolved_dataset_root)
    temporary_config = Path(temporary_dir.name) / config_path.name
    temporary_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return temporary_config, temporary_dir


def checkpoint_metric(ckpt: Mapping[str, Any], key: str) -> float | None:
    metrics = ckpt.get("train_metrics")
    if not isinstance(metrics, Mapping) or key not in metrics:
        return None
    try:
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def tensor_values(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def true_detect_stride(model: torch.nn.Module) -> tuple[int, list[float], list[float]]:
    layers = getattr(model, "model", None)
    if layers is None or len(layers) == 0:
        raise TypeError("Checkpoint model has no detection layers.")
    head = layers[-1]
    if not hasattr(head, "stride"):
        raise TypeError(f"Final layer {type(head).__name__} has no stride attribute.")

    head_values = tensor_values(head.stride)
    outer_values = tensor_values(getattr(model, "stride", head.stride))
    if not head_values or any(value <= 0 or not value.is_integer() for value in head_values):
        raise ValueError(f"Invalid Detect strides: {head_values}")
    if hasattr(head, "nl") and len(head_values) != int(head.nl):
        raise ValueError(
            f"Detect stride count {len(head_values)} does not match nl={int(head.nl)}."
        )
    stride = max(int(max(head_values)), 32)
    return stride, head_values, outer_values


def label_loss_items(loss_items: torch.Tensor, prefix: str = "val") -> dict[str, float]:
    names = ("box_loss", "cls_loss", "dfl_loss")
    return {
        f"{prefix}/{name}": round(float(value), 5)
        for name, value in zip(names, loss_items)
    }


def shutdown_dataloader(loader: Any) -> None:
    """Stop persistent DataLoader workers before interpreter teardown."""
    iterator = getattr(loader, "iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def main() -> None:
    args = parse_args()
    checkpoint = args.model.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "The script will not fall back to an online download."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Training-aligned FP16 validation requires a CUDA GPU.")

    data_config, temporary_dir = materialize_data_config(args.data, args.dataset_root)
    dataloader = None
    try:
        ckpt, _ = torch_safe_load(str(checkpoint))
        if not isinstance(ckpt, Mapping):
            raise TypeError(f"Unsupported checkpoint structure: {type(ckpt).__name__}")
        train_args = ckpt.get("train_args")
        if not isinstance(train_args, Mapping):
            raise KeyError("Checkpoint has no train_args; exact loss/protocol restoration is unavailable.")

        seed = int(train_args.get("seed", 0))
        deterministic = bool(train_args.get("deterministic", True))
        init_seeds(seed + 1, deterministic=deterministic)  # rank-0 BaseTrainer convention

        imgsz = int(args.imgsz or train_args.get("imgsz", 1600))
        train_workers = int(args.workers if args.workers is not None else train_args.get("workers", 8))
        val_workers = train_workers * 2
        cache = train_args.get("cache", False)
        device = select_device(args.device, batch=args.batch_size)

        network = ckpt.get("ema") or ckpt.get("model")
        if network is None:
            raise KeyError(f"No EMA/model object found in checkpoint: {checkpoint}")
        network = network.to(device).float()
        network.args = SimpleNamespace(**dict(train_args))
        # Rebuild the loss from the restored train_args. This avoids reusing a
        # criterion that an unstripped checkpoint may carry from another device.
        network.criterion = None
        for parameter in network.parameters():
            parameter.requires_grad_(False)

        loader_stride, head_strides, outer_strides = true_detect_stride(network)
        if head_strides != outer_strides:
            print(
                "WARNING: checkpoint outer model.stride does not match the Detect head: "
                f"outer={outer_strides}, head={head_strides}. "
                f"The validation loader will use the true Detect stride {loader_stride}.",
                flush=True,
            )

        data = check_det_dataset(str(data_config))
        run_name = args.name or f"{checkpoint.stem}_{Path(str(args.data)).stem}"
        save_dir = args.project.expanduser().resolve() / run_name
        validator_args = dict(train_args)
        validator_args.update({
            "model": str(checkpoint),
            "data": str(data_config),
            "mode": "val",
            "task": "detect",
            "split": "val",
            "imgsz": imgsz,
            "batch": args.batch_size,
            "device": args.device,
            "workers": train_workers,
            "cache": cache,
            # DetectionTrainer stores rect=False in args but constructs the val
            # dataset with rect=True before later mutating validator.args.rect.
            "rect": False,
            "half": args.half,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "augment": False,
            "plots": False,
            "save": False,
            "save_json": False,
            "save_txt": False,
            "verbose": True,
        })
        validator = DetectionValidator(save_dir=save_dir, args=validator_args)
        dataset = build_yolo_dataset(
            validator.args,
            data[validator.args.split],
            args.batch_size,
            data,
            mode="val",
            rect=args.rect,
            stride=loader_stride,
        )
        dataloader = build_dataloader(
            dataset,
            args.batch_size,
            val_workers,
            shuffle=False,
            rank=-1,
        )
        validator.dataloader = dataloader

        trainer = SimpleNamespace(
            device=device,
            data=data,
            amp=args.half,
            ema=SimpleNamespace(ema=network),
            model=network,
            loss_items=torch.zeros(3, device=device),
            stopper=SimpleNamespace(possible_stop=False),
            epoch=0,
            epochs=1,
            label_loss_items=label_loss_items,
        )
        batch_norm_before = sum(isinstance(module, _BatchNorm) for module in network.modules())
        print(
            "Evaluation protocol: training validator branch, "
            f"imgsz={imgsz}, batch={args.batch_size}, dataset.rect={dataset.rect}, "
            f"FP16={args.half}, Conv-BN fusion=False, loader_stride={loader_stride}, "
            f"cache={cache}, seed={seed + 1}, deterministic={deterministic}, "
            f"conf={args.conf}, IoU={args.iou}, max_det={args.max_det}",
            flush=True,
        )
        results = validator(trainer=trainer)
        batch_norm_after = sum(isinstance(module, _BatchNorm) for module in network.modules())
        if batch_norm_after != batch_norm_before:
            raise RuntimeError(
                "BatchNorm module count changed during validation; Conv-BN fusion was not disabled."
            )

        precision = float(results[METRIC_KEYS["precision"]])
        recall = float(results[METRIC_KEYS["recall"]])
        map50 = float(results[METRIC_KEYS["map50"]])
        map75 = float(validator.metrics.box.map75)
        map_value = float(results[METRIC_KEYS["map"]])
        print(
            "Final metrics (%): "
            f"P={100.0 * precision:.3f}, R={100.0 * recall:.3f}, "
            f"mAP50={100.0 * map50:.3f}, mAP75={100.0 * map75:.3f}, "
            f"mAP50-95={100.0 * map_value:.3f}",
            flush=True,
        )

        attached_map = checkpoint_metric(ckpt, METRIC_KEYS["map"])
        aligned = None
        delta_pp = None
        if attached_map is not None:
            delta_pp = 100.0 * (map_value - attached_map)
            aligned = abs(delta_pp) <= args.tolerance_pp
            print(
                "Checkpoint alignment: "
                f"recorded={100.0 * attached_map:.3f}, delta={delta_pp:+.3f} pp, "
                f"tolerance={args.tolerance_pp:.3f} pp -> {'PASS' if aligned else 'FAIL'}",
                flush=True,
            )

        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "train_aligned_metrics.json"
        output_path.write_text(
            json.dumps(
                {
                    "model": str(checkpoint),
                    "data": str(args.data),
                    "protocol": {
                        "validator_path": "training",
                        "imgsz": imgsz,
                        "batch": args.batch_size,
                        "rect": bool(dataset.rect),
                        "half": args.half,
                        "fused": False,
                        "loader_stride": loader_stride,
                        "detect_strides": head_strides,
                        "checkpoint_outer_strides": outer_strides,
                        "cache": cache,
                        "seed": seed + 1,
                        "deterministic": deterministic,
                        "conf": args.conf,
                        "iou": args.iou,
                        "max_det": args.max_det,
                    },
                    "metrics": {
                        **{key: float(value) for key, value in results.items()},
                        "metrics/mAP75(B)": map75,
                    },
                    "checkpoint_attached_map50_95": attached_map,
                    "difference_pp": delta_pp,
                    "alignment_passed": aligned,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Metrics written to {output_path}", flush=True)

        if args.strict and aligned is False:
            raise SystemExit(2)
    finally:
        if dataloader is not None:
            shutdown_dataloader(dataloader)
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    main()
