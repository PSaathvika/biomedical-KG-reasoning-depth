import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_medmcqa(path=None):
    path = Path(path or ROOT / "data/medmcqa/dev.json")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_pubmedqa(path=None):
    path = Path(path or ROOT / "data/pubmedqa/pqa_labeled.json")
    with path.open(encoding="utf-8") as f:
        return json.load(f)
