"""Mede FPS de inferência em CPU (só o modelo, sem overlay) e grava docs/assets/fps.json.

Uso:
    python scripts/benchmark_fps.py --weights models/best.pt --images data/rod/test/images --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import cv2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, default=Path("docs/assets/fps.json"))
    args = p.parse_args(argv)

    from ultralytics import YOLO
    import torch

    model = YOLO(str(args.weights))
    paths = sorted(x for x in args.images.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
    paths = paths[: args.n + args.warmup]
    times = []
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        model.predict(img, imgsz=args.imgsz, conf=0.15, verbose=False, device="cpu")
        if i >= args.warmup:
            times.append(time.perf_counter() - t0)
    mean_ms = statistics.mean(times) * 1000
    result = {
        "weights": args.weights.name,
        "device": "cpu",
        "hardware": {"processor": platform.processor(), "cpu_count": os.cpu_count(), "system": platform.platform()},
        "torch": torch.__version__,
        "imgsz": args.imgsz,
        "n": len(times),
        "mean_ms": round(mean_ms, 1),
        "p95_ms": round(sorted(times)[int(0.95 * (len(times) - 1))] * 1000, 1),
        "fps": round(1000 / mean_ms, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
