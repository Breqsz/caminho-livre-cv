"""Gera notebooks/train_colab.ipynb embutindo a versão atual de scripts/prepare_dataset.py.

O notebook é autocontido (não depende do repositório estar público). Rode de novo sempre
que prepare_dataset.py mudar — tests/test_notebook.py garante que estão sincronizados.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "scripts" / "prepare_dataset.py"
OUT = ROOT / "notebooks" / "train_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def build() -> dict:
    prepare_src = PREPARE.read_text(encoding="utf-8")
    cells = [
        md("""
# Caminho Livre CV — treino do YOLO11n (Google Colab)

GS FIAP 2026.2 · Applied Computer Vision · fine-tuning do YOLO11n no
[ROD-Dataset](https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO)
(24.326 imagens, 25 classes).

**Runtime:** Ambiente de execução → Alterar tipo → **GPU T4**. Checkpoints vão para o Google Drive,
então o treino sobrevive a uma desconexão (use a célula de retomada).
"""),
        code("!nvidia-smi\n%pip install -q ultralytics huggingface_hub"),
        code("""
from google.colab import drive
drive.mount('/content/drive')
RUN_DIR = '/content/drive/MyDrive/caminho-livre-cv/runs'
DATA_ROOT = '/content/rod'
import os; os.makedirs(RUN_DIR, exist_ok=True)
"""),
        md("## 1. Preparar o dataset\nCópia fiel de `scripts/prepare_dataset.py` do repositório."),
        code("%%writefile prepare_dataset.py\n" + prepare_src),
        code("!python prepare_dataset.py --root $DATA_ROOT --download --stats-out /content/class_stats.csv\n"
             "!ls $DATA_ROOT/train/images | wc -l; ls $DATA_ROOT/valid/images | wc -l; ls $DATA_ROOT/test/images | wc -l"),
        md("## 2. Treino\nYOLO11n a partir dos pesos COCO · `imgsz=640` · 50 épocas · early stopping `patience=15` · `seed=0`."),
        code("""
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(data=f'{DATA_ROOT}/data_cl.yaml', epochs=50, imgsz=640, patience=15, seed=0,
            project=RUN_DIR, name='y11n', exist_ok=True)
"""),
        md("### 2b. Retomar (só se o Colab desconectou)\nRode as células 1–3 e a preparação de novo, depois esta."),
        code("""
from ultralytics import YOLO
YOLO(f'{RUN_DIR}/y11n/weights/last.pt').train(resume=True)
"""),
        md("## 3. Avaliação no split **test** (não visto no treino)"),
        code("""
import json
from ultralytics import YOLO
best = YOLO(f'{RUN_DIR}/y11n/weights/best.pt')
m = best.val(data=f'{DATA_ROOT}/data_cl.yaml', split='test', imgsz=640, project=RUN_DIR, name='y11n_test', exist_ok=True)
names = best.names
per_class = {names[int(c)]: round(float(ap), 4) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
metrics = {
    'split': 'test', 'model': 'yolo11n', 'imgsz': 640,
    'map50': round(float(m.box.map50), 4), 'map50_95': round(float(m.box.map), 4),
    'precision': round(float(m.box.mp), 4), 'recall': round(float(m.box.mr), 4),
    'ap50_per_class': per_class,
    'speed_ms': m.speed,
}
json.dump(metrics, open(f'{RUN_DIR}/test_metrics.json', 'w'), indent=2, ensure_ascii=False)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
"""),
        md("## 4. Empacotar artefatos para o repositório"),
        code("""
import shutil, os
pack = '/content/artefatos'; os.makedirs(pack, exist_ok=True)
tr, te = f'{RUN_DIR}/y11n', f'{RUN_DIR}/y11n_test'
for src in [f'{tr}/weights/best.pt', f'{tr}/results.csv', f'{tr}/results.png', f'{tr}/args.yaml',
            f'{te}/confusion_matrix_normalized.png', f'{te}/BoxPR_curve.png', f'{te}/val_batch0_pred.jpg',
            f'{RUN_DIR}/test_metrics.json', '/content/class_stats.csv']:
    if os.path.exists(src):
        shutil.copy(src, pack)
    else:
        print('ausente:', src)
shutil.make_archive('/content/drive/MyDrive/caminho-livre-cv/artefatos', 'zip', pack)
print(os.listdir(pack))
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
