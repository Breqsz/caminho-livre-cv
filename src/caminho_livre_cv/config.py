from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "barriers.yaml"


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    color_bgr: tuple[int, int, int]
    classes: tuple[str, ...]
    counted: bool


@dataclass(frozen=True)
class BarrierConfig:
    categories: tuple[Category, ...]
    privacy_classes: frozenset[str]

    def category_for(self, class_name: str) -> Category | None:
        for cat in self.categories:
            if class_name in cat.classes:
                return cat
        return None


def load_barrier_config(path: Path = DEFAULT_CONFIG) -> BarrierConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cats = tuple(
        Category(
            key=c["key"],
            label=c["label"],
            color_bgr=tuple(c["color_bgr"]),
            classes=tuple(c["classes"]),
            counted=bool(c["counted"]),
        )
        for c in raw["categories"]
    )
    return BarrierConfig(categories=cats, privacy_classes=frozenset(raw.get("privacy_classes", [])))
