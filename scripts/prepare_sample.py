"""Seleciona e baixa a amostra ROD, preservando as divisões publicadas."""
from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Mapping
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REVISION = '103b38ccb6c80be269e5c7288e63b828d6f54e0b'
BASE = f'https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO/resolve/{REVISION}'
SIZES = {'train': 600, 'valid': 125, 'test': 150}
MINIMUM = {'train': 17, 'valid': 4, 'test': 5}
SEED = 42


def fetch(relpath: str, dest: Path) -> None:
    if dest.is_file():
        return
    host_path = relpath
    if relpath.startswith('train/'):
        split, kind, filename = relpath.split('/')
        shard = (int(Path(filename).stem.removeprefix('IMG_')) - 1) // 10_000
        host_path = f'{split}/{kind}/{shard}/{filename}'
    for attempt in range(5):
        try:
            response = requests.get(f'{BASE}/{host_path}', timeout=15)
            response.raise_for_status()
            temporary = dest.with_suffix('.part')
            temporary.write_bytes(response.content)
            temporary.replace(dest)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def choose(
    rows: list[dict[str, str]], split: str, randomizer: random.Random,
    origins: Mapping[str, str], used_origins: set[str],
) -> list[dict[str, str]]:
    candidates = [row for row in rows if row['split'] == split]
    randomizer.shuffle(candidates)
    candidates = [row for row in candidates
                  if origins[Path(row['image_filename']).stem] not in used_origins]
    selected = {}
    by_class = {i: [] for i in range(25)}
    for row in candidates:
        for label in row['classes_present'].split(','):
            if label:
                by_class[int(label)].append(row)
    for label in sorted(by_class, key=lambda i: len(by_class[i])):
        already = sum(row['image_filename'] in selected for row in by_class[label])
        remaining = (row for row in by_class[label] if row['image_filename'] not in selected)
        for _, row in zip(range(max(0, MINIMUM[split] - already)), remaining):
            selected[row['image_filename']] = row
    for row in candidates:
        if len(selected) >= SIZES[split]:
            break
        selected[row['image_filename']] = row
    if len(selected) != SIZES[split]:
        raise ValueError(f'{split}: esperadas {SIZES[split]} imagens, selecionadas {len(selected)}')
    return list(selected.values())


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepara a amostra reproduzível do ROD-Dataset.')
    parser.add_argument('--root', type=Path, default=Path('data/rod-sample'))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'images_manifest.csv'
    fetch('images_manifest.csv', manifest)
    rename_map = root / 'rename_map.csv'
    fetch('rename_map.csv', rename_map)
    with rename_map.open(newline='', encoding='utf-8') as source:
        origins = {row['new_stem']: row['old_stem'].split('.rf.', 1)[0]
                   for row in csv.DictReader(source)}
    with manifest.open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    randomizer = random.Random(SEED)
    items = []
    used_origins: set[str] = set()
    for split in SIZES:
        for sub in ('images', 'labels'):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        selected = choose(rows, split, randomizer, origins, used_origins)
        used_origins.update(origins[Path(row['image_filename']).stem] for row in selected)
        items.extend(selected)
    with (root / 'selection.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(items)
    files = [f"{row['split']}/{folder}/{row[key]}" for row in items
             for folder, key in [('images', 'image_filename'), ('labels', 'label_filename')]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(lambda path: fetch(path, root / path), files), 1):
            if index % 100 == 0:
                print(f'{index}/{len(files)} arquivos', flush=True)


if __name__ == '__main__':
    main()
