"""Compara YOLO11n COCO (sem fine-tune) com o modelo treinado nas mesmas imagens, lado a lado.

Pessoas são desfocadas nos dois lados. Gera docs/assets/baseline/*.jpg e summary.md.

Uso:
    python scripts/compare_baseline.py --images img1.jpg img2.jpg ... --out docs/assets/baseline
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from caminho_livre_cv.config import load_barrier_config
from caminho_livre_cv.detect import DEFAULT_CONF, _make_predict, process_frame


def _title(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((36, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--baseline", type=Path, default=Path("yolo11n.pt"))
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--out", type=Path, default=Path("docs/assets/baseline"))
    args = p.parse_args(argv)

    cfg = load_barrier_config()
    base, tuned = _make_predict(args.baseline, 640), _make_predict(args.weights, 640)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = ["| Imagem | COCO (sem fine-tune) | Fine-tuned (ROD) |", "|---|---|---|"]
    for path in args.images:
        img = cv2.imread(str(path))
        if img is None:
            print(f"ignorada (não abriu): {path}")
            continue
        a, da = process_frame(img, base, cfg, args.conf)
        b, db = process_frame(img, tuned, cfg, args.conf)
        side = np.hstack([_title(a, "YOLO11n COCO (sem fine-tune)"), _title(b, "YOLO11n fine-tuned (ROD)")])
        cv2.imwrite(str(args.out / f"{path.stem}_compare.jpg"), side)
        fmt = lambda ds: ", ".join(f"{k} ×{v}" for k, v in Counter(d.class_name for d in ds).most_common()) or "—"
        rows.append(f"| {path.name} | {fmt(da)} | {fmt(db)} |")
    (args.out / "summary.md").write_text(
        f"# Baseline COCO × fine-tuned (conf ≥ {args.conf})\n\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
