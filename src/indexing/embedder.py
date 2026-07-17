from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        # A heavier version: "BAAI/bge-code-v1"
        self.model = SentenceTransformer(model_name)

    def embed(self,texts,batch_size=4,normalize_embeddings=True,) -> np.ndarray:
        """
        Embed a list of text chunks.

        Parameters
        ----------
        texts : list[str]
        batch_size : int
        normalize_embeddings : bool

        Returns
        -------
        np.ndarray of shape (N, embedding_dim)
        """
        return self.model.encode(texts, batch_size=batch_size,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings)
