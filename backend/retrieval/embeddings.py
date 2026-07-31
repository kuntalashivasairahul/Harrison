from sentence_transformers import SentenceTransformer
import numpy as np

from backend.config import EMBEDDING_MODEL


model = SentenceTransformer(EMBEDDING_MODEL)


def embedding_dimension() -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def embed_text(text: str):
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(embedding).astype("float32")
