from pathlib import Path
import json

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent


def download_medmcqa():
    out = ROOT / "medmcqa" / "dev.json"
    dataset = load_dataset("awinml/medmcqa", split="validation")
    rows = [dict(row) for row in dataset]
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} MedMCQA validation questions to {out}")


def download_pubmedqa():
    out = ROOT / "pubmedqa" / "pqa_labeled.json"
    dataset = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        trust_remote_code=True,
    )
    rows = [dict(row) for row in dataset]
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} PubMedQA labeled questions to {out}")


if __name__ == "__main__":
    download_medmcqa()
    download_pubmedqa()
