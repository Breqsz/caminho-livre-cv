import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "train_colab.ipynb"


def test_notebook_embeds_current_prepare_script():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    embedded = [s for s in sources if s.startswith("%%writefile prepare_dataset.py\n")]
    assert len(embedded) == 1
    current = (ROOT / "scripts" / "prepare_dataset.py").read_text(encoding="utf-8")
    assert embedded[0].split("\n", 1)[1].strip() == current.strip()


def test_notebook_trains_spec_hyperparameters():
    text = NB.read_text(encoding="utf-8")
    for token in ("yolo11n.pt", "epochs=50", "imgsz=640", "patience=15", "seed=0", "split='test'"):
        assert token in text, token


def test_notebook_disables_hf_progress_bars_before_download():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    setup = next(i for i, s in enumerate(sources) if "drive.mount" in s)
    download = next(i for i, s in enumerate(sources) if "--download" in s and "!python" in s)
    assert "HF_HUB_DISABLE_PROGRESS_BARS" in sources[setup]
    assert setup < download
