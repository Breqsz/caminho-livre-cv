"""Desenha detecções, painel de contagem por categoria, FPS e aviso de verificação humana."""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from caminho_livre_cv.barrier_map import IGNORED, Detection, categorize, summarize
from caminho_livre_cv.config import BarrierConfig

DISCLAIMER = "Detecção automática — requer verificação humana"
HIDDEN_CATEGORIES = {IGNORED, "privacidade"}

CLASS_LABELS_PT = {
    "Bike": "Bicicleta", "Car": "Carro", "Motorcycle": "Moto", "Stairs": "Escada/degrau",
    "Electrical Pole": "Poste", "Dustbin": "Lixeira", "Manhole": "Bueiro/tampa", "Tree": "Árvore",
    "Guard rail": "Grade/guarda-corpo", "Pedestrian crosswalk": "Faixa de pedestre", "Bench": "Banco",
    "Traffic Cone": "Cone", "Fire hydrant": "Hidrante", "Traffic Barrel": "Barril de sinalização",
    "Plant Pot": "Vaso", "Electrical Box": "Caixa elétrica", "Chair": "Cadeira",
    "Bicycle Rack": "Paraciclo", "Person": "Pessoa",
}

_FONT_CANDIDATES = ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return bgr[2], bgr[1], bgr[0]


def render(frame: np.ndarray, dets: list[Detection], cfg: BarrierConfig, fps: float | None = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.6, min(w, h) / 720)
    fs = int(18 * scale)
    labels: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []

    for det in dets:
        key = categorize(det, cfg)
        if key in HIDDEN_CATEGORIES:
            continue
        color = cfg.category_for(det.class_name).color_bgr
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))
        text = f"{CLASS_LABELS_PT.get(det.class_name, det.class_name)} {det.confidence:.2f}"
        labels.append(((x1, max(0, y1 - fs - 4)), text, color))

    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _font(fs)
    for (x, y), text, color in labels:
        tb = draw.textbbox((x, y), text, font=font)
        draw.rectangle((tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2), fill=_rgb(color))
        draw.text((x, y), text, font=font, fill=(0, 0, 0))

    panel = [f"{label}: {n}" for label, n in summarize(dets, cfg).items()] or ["Nenhuma barreira detectada"]
    if fps is not None:
        panel.append(f"FPS: {fps:.1f}")
    pad, line_h = 8, fs + 6
    pw = max(draw.textlength(t, font=font) for t in panel) + 2 * pad
    draw.rectangle((0, 0, pw, len(panel) * line_h + pad), fill=(15, 15, 15))
    for i, t in enumerate(panel):
        draw.text((pad, pad // 2 + i * line_h), t, font=font, fill=(240, 240, 240))

    small = _font(max(12, int(fs * 0.8)))
    db = draw.textbbox((0, 0), DISCLAIMER, font=small)
    dh = db[3] - db[1] + 10
    draw.rectangle((0, h - dh, w, h), fill=(15, 15, 15))
    draw.text((pad, h - dh + 3), DISCLAIMER, font=small, fill=(255, 210, 0))

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
