from sentence_transformers import SentenceTransformer
import numpy as np

# load MiniLM (small + fast + free)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    embedding = model.encode([text], convert_to_numpy=True)
    return np.array(embedding).astype("float32")
