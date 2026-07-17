"""
Final RAG pipeline

Main class:

Example:

rag.ask(
Why does torch.compile fail
with dynamic shapes?
)

Output:

According to torch/_dynamo.py ...

The issue happens because ...

Relevant files:
- torch/_dynamo/...
- issue #12345
"""

from src.indexing.chunker import CodeChunker
from src.indexing.embedder import Embedder
from src.indexing.vectorstore import VectorStore
from src.github.crawler import GithubCrawler

import json, os, requests
import numpy as np
from llama_cpp import Llama

class FlatRetriever:

    def __init__(self, llm, embedder = "BAAI/bge-base-en-v1.5", vectorstore = None):
        self.llm = llm
        self.embedder = Embedder(model_name=embedder)
        if vectorstore is None: # Use the standard
            self.store = VectorStore(index_path, metadata_path)

    def build_flatIP(self, chunks):
        """
        Chunks will be a list containing each:

        chunks.append({
            "text": "\n".join(lines[start:end]),
            "file": str(path.relative_to(self.input_dir)),
            "extension": path.suffix,
            "start_line": start + 1,
            "end_line": end
        })
        """
        print(f" Found {len(chunks)} chunks")

        texts = [c["text"] for c in chunks] # list of text sections

        """ Make batches of texts to prevent RAM exceeding. Then, EMBED using BAAI"""
        all_embeddings = []
        steps = 50

        for i in range(0, len(texts), steps):
            batch = texts[i:min(i+steps,len(texts))]
            emb = self.embedder.embed(batch,batch_size=4)
            all_embeddings.append(emb)

            embeddings = np.vstack(all_embeddings)
            # embeddings.shape == (10000, 768)

            self.store.build_flatIP(embeddings,chunks)

            self.store.save()

    def load_flatIP(self):
        self.store.load()

    def ask(self,question):

        context=self.retriever.search(question)

        prompt=f"""
        Answer using this context:

        {context}

        Question:
        {question}
        """

        return self.llm.generate(prompt)

if __name__ == "__main__":
    """
    goes thru all chunks, embeds them, creates vector database
    """

    with open("../data/chunks/chunks.json", "r") as f:
        chunks = json.load(f)

    llm = Llama(model_path=r"C:\Users\PC\Documents\Github-RAG\qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf",n_ctx=8192,n_threads=8)
    retriever = FlatRetriever(llm)
    retriever.build_flatIP(chunks[:100])

    query = retriever.embedder.embed(["How does torch.nn.Module work?"])

    results = retriever.store.search(query)
