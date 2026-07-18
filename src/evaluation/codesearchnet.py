from datasets import load_dataset
from src.retriever import FlatRetriever
from llama_cpp import Llama
import numpy as np
"""
We have a benchmark dataset, with queries (docstrings)
"""

import os
print(os.cpu_count())

llm = Llama(model_path=r"C:\Users\PC\Documents\Github-RAG\qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf",n_ctx=2048,n_threads=8, verbose=False)
retriever = FlatRetriever(llm)
dataset = load_dataset("code-search-net/code_search_net","python")
small_test = dataset["test"].select(range(1000))

snippets = []
queries = []
ground_truth = []

print(dataset["test"].column_names)

for idx, sample in enumerate(small_test):

    snippets.append(sample["whole_func_string"])
    # We use the docstring as the query
    queries.append(sample["func_documentation_string"])
    # Ground truth is that we know the ID of the corresponding snippets
    ground_truth.append(idx)

print("Lists for snippets, queries, GT ready\n\n\n\n")

embeddings = retriever.embedder.embed(snippets,batch_size=16)

print("Embeddings ready\n")

retriever.store.build_flatIP(embeddings,snippets)
retriever.store.save()

print("VectorStore ready\n\n")

def recall_at_k(retrieved,relevant):

    return int(relevant in retrieved)

recalls = []

for query, gt in zip(queries, ground_truth):

    query_embedding = retriever.embedder.embed([query],False)

    scores, ids = retriever.store.search_ids(
        query_embedding,
        k=10
    )

    retrieved = ids[0].tolist()

    recalls.append(recall_at_k(retrieved, gt))

print(f"Recall@10 = {np.mean(recalls):.3f}")
