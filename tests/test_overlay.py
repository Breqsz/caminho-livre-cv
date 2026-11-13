import numpy as np
import pytest

from caminho_livre_cv.barrier_map import Detection
from caminho_livre_cv.config import load_barrier_config
from caminho_livre_cv.overlay import DISCLAIMER, render


@pytest.fixture(scope="module")
def cfg():
    return load_barrier_config()


def black(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_disclaimer_text():
    assert DISCLAIMER == "Detecção automática — requer verificação humana"


def test_render_returns_same_shape_and_copy(cfg):
    f = black()
    out = render(f, [Detection("Dustbin", 0.9, (300, 200, 400, 300))], cfg, fps=12.3)
    assert out.shape == f.shape
    assert not f.any()


def test_render_draws_for_counted_class(cfg):
    out = render(black(), [Detection("Dustbin", 0.9, (300, 200, 400, 300))], cfg)
    assert out[250, 300].any() or out[250, 301].any()


def test_render_skips_ignored_class(cfg):
    out = render(black(), [Detection("Building", 0.9, (300, 200, 400, 300))], cfg)
    assert not out[190:310, 290:410].any()


def test_render_skips_person_box(cfg):
    out = render(black(), [Detection("Person", 0.9, (300, 200, 400, 300))], cfg)
    assert not out[190:310, 290:410].any()


def test_render_without_fps_ok(cfg):
    assert render(black(), [], cfg, fps=None).shape == (480, 640, 3)
