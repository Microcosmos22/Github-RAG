import faiss
import numpy as np
import json
from pathlib import Path

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:

    def __init__(self,
                 index_path=r"C:\Users\PC\Documents\Github-RAG\data\embeddings\faiss.index",
                 metadata_path=r"C:\Users\PC\Documents\Github-RAG\data\embeddings\metadata.json"):

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.index = None
        self.metadata = None

    def build(self, embeddings, metadata, indextype = "flat"):
        """
        FAISS only knows:

        nearest vector IDs:
        [534, 120, 999]
        """
        # Normalize embedding vectors of the snippets
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1] # 768 or 2*768
        self.indextype = indextype

        if indextype == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif indextype == "ivf":
            nlist = round(np.sqrt(len(embeddings)))  # number of clusters
            quantizer = faiss.IndexFlatIP(dim)

            self.index = faiss.IndexIVFFlat(quantizer,dim,
                nlist,faiss.METRIC_INNER_PRODUCT)

            # IVF needs training before adding vectors
            self.index.train(embeddings)

        self.index.add(embeddings.astype(np.float32))

        self.metadata = metadata

    def save(self):
        """ FAISS index

        ID 0  -> [0.23, -0.12, ...]
        ID 1  -> [0.41,  0.22, ...]
        ID 2  -> [-0.11, 0.54, ...]
        ...
        ID 9999 -> [...]
        """
        faiss.write_index(self.index,str(self.index_path))

        with open(self.metadata_path, "w") as f:
            print("Vectorstore saving embedding vectors for the text chunks in {self.metadata_path}")
            json.dump(self.metadata, f)

    def load(self):

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

    def search_ids(self, query_embedding, k=10, nprobe=10):

        if isinstance(self.index, faiss.IndexIVF): # IVF
            self.index.nprobe = nprobe

        scores, ids = self.index.search(query_embedding, k)

        return scores, ids

    def search(self, query_embedding, k=5):

        scores, ids = self.index.search(query_embedding.astype(np.float32),k)

        return [self.metadata[i] for i in ids[0]]

if __name__ == "__main__":
    # Load FAISS index
    index = faiss.read_index(
        "../../data/embeddings/faiss.index")

    # Load chunk metadata
    with open("../../data/embeddings/metadata.json",encoding="utf-8") as f:
        metadata = json.load(f)

    # Load the SAME embedding model
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    query = "Where is autograd implemented in PyTorch?"
    query_embedding = embedder.encode([query],normalize_embeddings=True,convert_to_numpy=True)

    # Search
    k = 5
    print("Index dimension:", index.d)
    print("Query shape:", query_embedding.shape)
    scores, ids = index.search(query_embedding.astype(np.float32),k)

    print("Nearest IDs:")
    print(ids[0])

    print("\nScores:")
    print(scores[0])

    # Recover original chunks
    for score, idx in zip(scores[0], ids[0]):

        print("="*80)
        print("Score:", score)
        print("ID:", idx)

        chunk = metadata[idx]

        print("File:", chunk["file"])
        print("Lines:", chunk["start_line"], "-", chunk["end_line"])
        print(chunk["text"][:500])
