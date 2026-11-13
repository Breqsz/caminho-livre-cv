import pytest

from caminho_livre_cv.barrier_map import IGNORED, Detection, categorize, summarize
from caminho_livre_cv.config import load_barrier_config
from scripts.prepare_dataset import CLASS_NAMES


@pytest.fixture(scope="module")
def cfg():
    return load_barrier_config()


def det(name, conf=0.9):
    return Detection(name, conf, (0, 0, 10, 10))


def test_every_dataset_class_is_mapped(cfg):
    missing = [n for n in CLASS_NAMES if cfg.category_for(n) is None]
    assert missing == []


def test_no_class_in_two_categories(cfg):
    seen = [c for cat in cfg.categories for c in cat.classes]
    assert len(seen) == len(set(seen))


def test_categorize_known(cfg):
    assert categorize(det("Traffic Cone"), cfg) == "calcada_obstruida"
    assert categorize(det("Stairs"), cfg) == "nivel"


def test_unknown_class_is_ignored(cfg):
    assert categorize(det("potted plant"), cfg) == IGNORED
    assert categorize(det("Teraffic Barrel"), cfg) == IGNORED


def test_summarize_counts_only_counted(cfg):
    dets = [det("Dustbin"), det("Dustbin"), det("Car"), *[det("Person")] * 3, det("Pedestrian crosswalk")]
    assert summarize(dets, cfg) == {
        "Calçada obstruída": 2,
        "Veículo — verificar se está na calçada": 1,
    }


def test_summarize_empty(cfg):
    assert summarize([], cfg) == {}


def test_privacy_classes(cfg):
    assert cfg.privacy_classes == frozenset({"Person"})
