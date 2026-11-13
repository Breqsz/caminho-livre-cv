"""Detecção em tempo real de barreiras de acessibilidade (webcam, vídeo ou pasta de imagens).

Uso:
    python -m caminho_livre_cv.detect --source 0                      # webcam
    python -m caminho_livre_cv.detect --source rua.mp4 --save out.mp4
    python -m caminho_livre_cv.detect --source pasta/ --save saida/ --no-show
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from caminho_livre_cv.barrier_map import Detection
from caminho_livre_cv.config import DEFAULT_CONFIG, BarrierConfig, load_barrier_config
from caminho_livre_cv.overlay import render
from caminho_livre_cv.privacy import blur_regions

PRIVACY_CONF = 0.15
DEFAULT_CONF = 0.35
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "Caminho Livre CV"

Predict = Callable[[np.ndarray, float], list[Detection]]


def results_to_detections(result, min_conf: float) -> list[Detection]:
    boxes = result.boxes
    dets = []
    for cls, conf, xyxy in zip(np.asarray(boxes.cls), np.asarray(boxes.conf), np.asarray(boxes.xyxy)):
        if float(conf) < min_conf:
            continue
        x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
        dets.append(Detection(result.names[int(cls)], float(conf), (x1, y1, x2, y2)))
    return dets


def process_frame(
    frame: np.ndarray,
    predict: Predict,
    cfg: BarrierConfig,
    conf: float = DEFAULT_CONF,
    blur: bool = True,
    fps: float | None = None,
) -> tuple[np.ndarray, list[Detection]]:
    all_dets = predict(frame, PRIVACY_CONF)
    if blur:
        frame = blur_regions(frame, [d.xyxy for d in all_dets if cfg.is_private(d.class_name)])
    shown = [d for d in all_dets if d.confidence >= conf]
    return render(frame, shown, cfg, fps), shown


def writer_fps(cap_fps: float) -> float:
    if cap_fps is None or math.isnan(cap_fps) or cap_fps <= 0:
        return 30.0
    return cap_fps


def _fail(msg: str) -> int:
    print(f"erro: {msg}", file=sys.stderr)
    return 2


def _image_frames(folder: Path) -> Iterator[tuple[str, np.ndarray]]:
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(str(p))
            if img is not None:
                yield p.name, img


def _make_predict(weights: Path, imgsz: int) -> Predict:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(frame: np.ndarray, min_conf: float) -> list[Detection]:
        result = model.predict(frame, conf=min_conf, imgsz=imgsz, verbose=False)[0]
        return results_to_detections(result, min_conf)

    return predict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detector de barreiras de acessibilidade em calçadas (Caminho Livre CV)")
    p.add_argument("--source", default="0", help="índice da webcam, arquivo de vídeo ou pasta de imagens")
    p.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save", type=Path, help="mp4 de saída (vídeo/webcam) ou pasta (imagens)")
    p.add_argument("--no-blur", action="store_true", help="desliga o desfoque de pessoas")
    p.add_argument("--no-show", action="store_true", help="não abre janela")
    args = p.parse_args(argv)

    if not args.weights.is_file():
        return _fail(f"pesos não encontrados: {args.weights}")
    is_cam = args.source.isdigit()
    src = Path(args.source)
    if not is_cam and not src.exists():
        return _fail(f"fonte não encontrada: {args.source}")

    cfg = load_barrier_config(args.config)
    predict = _make_predict(args.weights, args.imgsz)
    blur = not args.no_blur

    if not is_cam and src.is_dir():
        if args.save:
            args.save.mkdir(parents=True, exist_ok=True)
        for name, img in _image_frames(src):
            out, _ = process_frame(img, predict, cfg, args.conf, blur)
            if args.save:
                cv2.imwrite(str(args.save / name), out)
            if not args.no_show:
                cv2.imshow(WINDOW, out)
                if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                    break
        cv2.destroyAllWindows()
        return 0

    cap = cv2.VideoCapture(int(args.source) if is_cam else str(src))
    if not cap.isOpened():
        return _fail(f"não foi possível abrir a fonte: {args.source}")
    writer = None
    times: deque[float] = deque(maxlen=30)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            fps = len(times) / sum(times) if times else None
            out, _ = process_frame(frame, predict, cfg, args.conf, blur, fps)
            times.append(time.perf_counter() - t0)
            if args.save:
                if writer is None:
                    args.save.parent.mkdir(parents=True, exist_ok=True)
                    h, w = out.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(args.save), fourcc, writer_fps(cap.get(cv2.CAP_PROP_FPS)), (w, h))
                writer.write(out)
            if not args.no_show:
                cv2.imshow(WINDOW, out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
