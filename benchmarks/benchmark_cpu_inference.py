#!/usr/bin/env python3
"""
DocLayout-YOLO CPU Inference Benchmark
======================================
Benchmarks CPU inference optimizations as opt-in toggles:
  --channels-last    : Use NHWC memory format (typically ~1.3-1.4x speedup on CPU)
  --fuse             : Apply fuse_custom() recursively (safe, idempotent)

Usage:
    python benchmark_cpu_inference.py --model model.pt --image img.png
    python benchmark_cpu_inference.py --model model.pt --image img.png --channels-last
    python benchmark_cpu_inference.py --model model.pt --image img.png --channels-last --fuse
"""
import argparse
import gc
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from doclayout_yolo import YOLOv10

warnings.filterwarnings("ignore", category=UserWarning)


# ── CONFIG ─────────────────────────────────────────────
DEFAULT_IMGSZ = 1024
DEFAULT_CONF = 0.2
DEFAULT_DEVICE = "cpu"
N_WARMUP_FWD = 5
N_ITERS_FWD = 20
N_WARMUP_PRED = 2
N_ITERS_PRED = 5
# ───────────────────────────────────────────────────────


@contextmanager
def cache_flush():
    gc.collect()
    _ = torch.randn(20_000_000).mul_(2).sum().item()
    yield
    gc.collect()


def preprocess(path: str, size: int) -> torch.Tensor:
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size))
    return torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def recursive_fuse(model: nn.Module) -> int:
    """Recursively apply fuse_custom() / fuse() to all modules. Safe to call multiple times."""
    fused_count = 0
    for name, m in model.named_modules():
        if hasattr(m, "fuse_custom") and callable(m.fuse_custom):
            if not getattr(m, "fused", False):
                try:
                    m.fuse_custom()
                    fused_count += 1
                except Exception as e:
                    print(f"  [WARN] fuse_custom failed on {name}: {e}")
        elif hasattr(m, "fuse") and callable(m.fuse):
            if not getattr(m, "fused", False):
                try:
                    m.fuse()
                    fused_count += 1
                except Exception as e:
                    print(f"  [WARN] fuse failed on {name}: {e}")
    return fused_count


def benchmark_forward(model, x, n_warmup=5, n_iters=20, channels_last=False, device="cpu"):
    model = model.eval().to(device)
    x = x.to(device)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    with cache_flush():
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                _ = model(x)
        return (time.perf_counter() - t0) / n_iters * 1000


def benchmark_predict(model, path, imgsz, conf, n_warmup=2, n_iters=5, device="cpu"):
    for _ in range(n_warmup):
        _ = model.predict(path, imgsz=imgsz, conf=conf, device=device, verbose=False)
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = model.predict(path, imgsz=imgsz, conf=conf, device=device, verbose=False)
    return (time.perf_counter() - t0) / n_iters * 1000


def main():
    parser = argparse.ArgumentParser(description="DocLayout-YOLO CPU Inference Benchmark")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference size")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device (cpu/cuda)")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last (NHWC) memory format")
    parser.add_argument("--fuse", action="store_true", help="Apply recursive fuse_custom() (safe, idempotent)")
    parser.add_argument("--save", default=None, help="Optional path to save annotated result image")
    args = parser.parse_args()

    print("=" * 70)
    print("  DocLayout-YOLO CPU Inference Benchmark")
    print("=" * 70)
    print(f"PyTorch:  {torch.__version__}")
    print(f"Device:   {args.device} ({torch.get_num_threads()} threads)")
    print(f"Model:    {args.model}")
    print(f"Image:    {args.image}")
    print(f"Size:     {args.imgsz}")
    print(f"Opts:     channels_last={args.channels_last}, fuse={args.fuse}")
    print("=" * 70)

    x = preprocess(args.image, args.imgsz)

    # ── Load model ──
    print("\nLoading model...")
    model = YOLOv10(args.model)
    model = model.to(args.device)

    # ── Optional: recursive fuse_custom ──
    if args.fuse:
        print("Applying fuse_custom()...")
        n = recursive_fuse(model.model)
        print(f"  Fused {n} modules.")

    # ── Optional: channels_last ──
    if args.channels_last:
        print("Converting to channels_last (NHWC)...")
        x = x.to(memory_format=torch.channels_last)
        model.model = model.model.to(memory_format=torch.channels_last)

    # ── Benchmark raw forward ──
    print("\nBenchmarking raw forward pass...")
    t_fwd = benchmark_forward(
        model.model, x,
        n_warmup=N_WARMUP_FWD, n_iters=N_ITERS_FWD,
        channels_last=False,  # already applied above
        device=args.device,
    )
    print(f"  Raw forward: {t_fwd:.2f} ms")

    # ── Benchmark full predict pipeline ──
    print("\nBenchmarking full predict pipeline...")
    t_pred = benchmark_predict(
        model, args.image,
        imgsz=args.imgsz, conf=args.conf,
        n_warmup=N_WARMUP_PRED, n_iters=N_ITERS_PRED,
        device=args.device,
    )
    print(f"  Full predict: {t_pred:.2f} ms")

    # ── Optional: save result ──
    if args.save:
        print(f"\nSaving result to {args.save}...")
        result = model.predict(args.image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        result.save(args.save)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Raw forward:  {t_fwd:>10.2f} ms")
    print(f"  Full predict: {t_pred:>10.2f} ms")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
