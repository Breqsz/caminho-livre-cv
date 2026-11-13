"""Baixa o ROD-Dataset (Hugging Face), gera o data.yaml do projeto e as estatísticas por classe.

Uso:
    python scripts/prepare_dataset.py --root data/rod --download
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

REPO_ID = "Abtinzandi/Obstacle-Detection-Dataset-YOLO"
SPLITS = ("train", "valid", "test")

# Ordem e nomes idênticos ao data.yaml original, exceto o typo "Teraffic Barrel" (id 20) corrigido.
CLASS_NAMES: list[str] = [
    "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign", "Electrical Pole", "Road",
    "Motorcycle", "Dustbin", "Dog", "Manhole", "Tree", "Guard rail", "Pedestrian crosswalk",
    "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant", "Traffic Barrel", "Plant Pot",
    "Electrical Box", "Chair", "Bicycle Rack",
]


def write_data_yaml(root: Path, out: Path) -> Path:
    data = {
        "path": str(Path(root).resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def class_stats(root: Path) -> dict[str, dict[str, int]]:
    stats = {name: {split: 0 for split in SPLITS} for name in CLASS_NAMES}
    for split in SPLITS:
        for label in (Path(root) / split / "labels").glob("*.txt"):
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    stats[CLASS_NAMES[int(parts[0])]][split] += 1
    return stats


def write_stats_csv(stats: dict[str, dict[str, int]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", *SPLITS, "total"])
        for i, name in enumerate(CLASS_NAMES):
            row = [stats[name][s] for s in SPLITS]
            w.writerow([i, name, *row, sum(row)])


def download(root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(root))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/rod"))
    p.add_argument("--download", action="store_true")
    p.add_argument("--stats-out", type=Path, default=Path("docs/assets/class_stats.csv"))
    args = p.parse_args(argv)
    if args.download:
        download(args.root)
    yaml_path = write_data_yaml(args.root, args.root / "data_cl.yaml")
    stats = class_stats(args.root)
    write_stats_csv(stats, args.stats_out)
    total = sum(sum(v.values()) for v in stats.values())
    print(f"data.yaml: {yaml_path}\ncaixas: {total}\nestatísticas: {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
