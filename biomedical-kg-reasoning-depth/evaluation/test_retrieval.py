import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from shared.embeddings import EmbeddingModel
from shared.vector_store import VectorStore


documents = [
    "Insulin helps regulate blood glucose levels.",
    "Hypertension is a major risk factor for cardiovascular disease.",
    "Hemoglobin carries oxygen in the blood."
]

embedding_model = EmbeddingModel()
document_embeddings = embedding_model.encode(documents)

vector_store = VectorStore(document_embeddings.shape[1])
vector_store.add(document_embeddings, documents)

query = "What regulates blood glucose?"
query_embedding = embedding_model.encode([query])[0]

results = vector_store.search(query_embedding, top_k=2)

for result in results:
    print(result)
