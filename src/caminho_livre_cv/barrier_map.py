from __future__ import annotations

from dataclasses import dataclass

from caminho_livre_cv.config import BarrierConfig

IGNORED = "ignorado"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]


def categorize(det: Detection, cfg: BarrierConfig) -> str:
    cat = cfg.category_for(det.class_name)
    return cat.key if cat else IGNORED


def summarize(dets: list[Detection], cfg: BarrierConfig) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in dets:
        cat = cfg.category_for(det.class_name)
        if cat and cat.counted:
            counts[cat.key] = counts.get(cat.key, 0) + 1
    return {c.label: counts[c.key] for c in cfg.categories if c.key in counts}
