from pathlib import Path

import yaml

from scripts.prepare_dataset import CLASS_NAMES, class_stats, write_data_yaml


def _fake_dataset(root: Path) -> None:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "train" / "labels" / "IMG_00001.txt").write_text("0 .5 .5 .1 .1\n20 .5 .5 .1 .1\n")
    (root / "valid" / "labels" / "IMG_20816.txt").write_text("3 .5 .5 .2 .2\n")


def test_data_yaml_fixes_typo(tmp_path):
    _fake_dataset(tmp_path)
    out = write_data_yaml(tmp_path, tmp_path / "data_cl.yaml")
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["nc"] == 25
    assert "Traffic Barrel" in data["names"]
    assert "Teraffic Barrel" not in data["names"]
    assert data["names"] == CLASS_NAMES


def test_data_yaml_paths_absolute(tmp_path):
    _fake_dataset(tmp_path)
    data = yaml.safe_load(write_data_yaml(tmp_path, tmp_path / "d.yaml").read_text(encoding="utf-8"))
    assert Path(data["path"]).is_absolute()
    assert (data["train"], data["val"], data["test"]) == ("train/images", "valid/images", "test/images")


def test_class_stats_counts(tmp_path):
    _fake_dataset(tmp_path)
    stats = class_stats(tmp_path)
    assert stats["Bike"]["train"] == 1
    assert stats["Traffic Barrel"]["train"] == 1
    assert stats["Person"]["valid"] == 1
    assert stats["Person"]["train"] == 0
