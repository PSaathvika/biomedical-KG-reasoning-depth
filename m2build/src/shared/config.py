MODEL_NAME = "BioMistral/BioMistral-7B"
EMBEDDING_MODEL = "NeuML/biomedbert-base-embeddings"
DEFAULT_TOP_K = 5
DEFAULT_DEPTH = 1
VALID_DEPTHS = (1, 2, 3)


def validate_depth(depth):
    if depth not in VALID_DEPTHS:
        raise ValueError("depth must be 1, 2, or 3")
