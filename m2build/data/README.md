# Datasets

Milestone 2 needs the biomedical question datasets so the same questions can later be passed to all methods.

## MedMCQA

Use the `awinml/medmcqa` Hugging Face mirror. It provides train/dev/test JSON files. The dev split is the labeled evaluation split used by the loader in this package. The full train file is large, so it is not committed to this repository.

Source: https://huggingface.co/datasets/awinml/medmcqa

## PubMedQA

Use the `qiaojin/PubMedQA` dataset. The labeled `pqa_labeled` configuration contains the 1,000 expert-annotated examples. The loader requests `trust_remote_code=True` for compatibility with the original dataset builder.

Source: https://huggingface.co/datasets/qiaojin/PubMedQA

Run:

```bash
python data/download_datasets.py
```

The script stores the downloaded files under `data/`. Do not commit large generated dataset copies unless the team explicitly decides to version them.
