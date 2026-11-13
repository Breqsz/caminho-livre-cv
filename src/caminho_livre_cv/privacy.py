"""Privacidade by design: desfoca regiões (pessoas) antes de exibir ou salvar qualquer frame."""
from __future__ import annotations

import cv2
import numpy as np


def blur_regions(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], ksize: int = 51) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    k = ksize if ksize % 2 == 1 else ksize + 1
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out
