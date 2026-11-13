import math

import numpy as np
import pytest

from caminho_livre_cv import detect
from caminho_livre_cv.barrier_map import Detection
from caminho_livre_cv.config import load_barrier_config


@pytest.fixture(scope="module")
def cfg():
    return load_barrier_config()


def noise(h=320, w=320):
    return np.random.default_rng(1).integers(0, 256, (h, w, 3), dtype=np.uint8)


def fake_predict(dets):
    calls = []

    def predict(frame, min_conf):
        calls.append(min_conf)
        return [d for d in dets if d.confidence >= min_conf]

    predict.calls = calls
    return predict


PERSON_BOX = (200, 120, 260, 180)


def test_constants():
    assert detect.PRIVACY_CONF == 0.15
    assert detect.DEFAULT_CONF == 0.35


def test_low_conf_person_still_blurred(cfg):
    f = noise()
    pred = fake_predict([Detection("Person", 0.2, PERSON_BOX)])
    out, dets = detect.process_frame(f, pred, cfg, conf=0.35)
    x1, y1, x2, y2 = PERSON_BOX
    assert not np.array_equal(out[y1:y2, x1:x2], f[y1:y2, x1:x2])
    assert dets == []
    assert pred.calls == [detect.PRIVACY_CONF]


def test_no_blur_flag_keeps_person_region(cfg):
    f = noise()
    out, _ = detect.process_frame(f, fake_predict([Detection("Person", 0.9, PERSON_BOX)]), cfg, blur=False)
    x1, y1, x2, y2 = PERSON_BOX
    assert np.array_equal(out[y1:y2, x1:x2], f[y1:y2, x1:x2])


def test_conf_filter(cfg):
    low, high = Detection("Dustbin", 0.3, (10, 10, 50, 50)), Detection("Dustbin", 0.8, (60, 60, 90, 90))
    _, dets = detect.process_frame(noise(), fake_predict([low, high]), cfg, conf=0.35)
    assert dets == [high]


def test_writer_fps_fallback():
    assert detect.writer_fps(0) == 30.0
    assert detect.writer_fps(float("nan")) == 30.0
    assert detect.writer_fps(25) == 25


def test_main_missing_video_returns_2(tmp_path, capsys):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"")
    rc = detect.main(["--source", str(tmp_path / "nao_existe.mp4"), "--weights", str(weights), "--no-show"])
    assert rc == 2
    assert "não encontrad" in capsys.readouterr().err


def test_main_missing_weights_returns_2(tmp_path, capsys):
    rc = detect.main(["--source", "0", "--weights", str(tmp_path / "nada.pt"), "--no-show"])
    assert rc == 2
    assert "não encontrad" in capsys.readouterr().err


class _Boxes:
    def __init__(self, rows):
        self.cls = np.array([r[0] for r in rows], dtype=float)
        self.conf = np.array([r[1] for r in rows], dtype=float)
        self.xyxy = np.array([r[2] for r in rows], dtype=float)


class _Result:
    names = {0: "Bike", 9: "Dustbin"}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


def test_results_to_detections_converts_and_filters():
    res = _Result([(9, 0.9, (1.4, 2.6, 30.2, 40.9)), (0, 0.1, (0, 0, 5, 5))])
    assert detect.results_to_detections(res, 0.15) == [Detection("Dustbin", pytest.approx(0.9), (1, 3, 30, 41))]


def test_results_to_detections_empty():
    assert detect.results_to_detections(_Result([]), 0.15) == []


def test_lowercase_person_from_coco_weights_is_blurred(cfg):
    f = noise()
    out, _ = detect.process_frame(f, fake_predict([Detection("person", 0.9, PERSON_BOX)]), cfg)
    x1, y1, x2, y2 = PERSON_BOX
    assert not np.array_equal(out[y1:y2, x1:x2], f[y1:y2, x1:x2])
