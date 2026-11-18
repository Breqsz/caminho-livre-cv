"""Treina e avalia o YOLO11n usando a amostra oficial do projeto."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

from scripts.prepare_dataset import write_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/rod-sample'))
    parser.add_argument('--weights', default='yolo11n.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--out', type=Path, default=Path('runs/amostra-rod'))
    parser.add_argument('--eval-only', action='store_true', help='avalia os pesos já treinados neste diretório')
    args = parser.parse_args()

    yaml_file = write_data_yaml(args.data, args.data / 'data_cl.yaml').resolve()
    out = args.out.resolve()
    if not args.eval_only:
        model = YOLO(args.weights)
        model.train(
            data=str(yaml_file), epochs=args.epochs, imgsz=args.imgsz, batch=16,
            workers=0, device=args.device, seed=42, optimizer='AdamW', lr0=0.002,
            close_mosaic=3, project=str(out), name='train', exist_ok=True, plots=True,
        )
    best = out / 'train/weights/best.pt'
    model = YOLO(str(best))
    metrics = model.val(
        data=str(yaml_file), split='test', imgsz=args.imgsz, batch=16, workers=0,
        device=args.device, project=str(out), name='test', exist_ok=True, plots=True,
    )
    box = metrics.box
    result = {
        'split': 'test',
        'images': len(list((args.data / 'test/images').glob('*.jpg'))),
        'imgsz': args.imgsz,
        'map50': round(float(box.map50), 4),
        'map50_95': round(float(box.map), 4),
        'precision': round(float(box.mp), 4),
        'recall': round(float(box.mr), 4),
        'classes': {
            model.names[int(class_id)]: {
                'ap50': round(float(ap), 4),
                'ap50_95': round(float(ap_all), 4),
                'precision': round(float(precision), 4),
                'recall': round(float(recall), 4),
            }
            for class_id, ap, ap_all, precision, recall in zip(
                box.ap_class_index, box.ap50, box.ap, box.p, box.r
            )
        },
    }
    (out / 'test_metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(best, out / 'best.pt')


if __name__ == '__main__':
    main()
