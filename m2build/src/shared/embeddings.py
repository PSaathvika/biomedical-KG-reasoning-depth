from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class BiomedicalEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
