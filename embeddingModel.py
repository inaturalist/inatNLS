import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, input_data, normalize_vectors=False):
        emb = self.model.encode(input_data)
        if normalize_vectors:
            emb = emb / np.linalg.norm(emb)

        return emb
