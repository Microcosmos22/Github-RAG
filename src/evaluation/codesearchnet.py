from datasets import load_dataset
from src.retriever import DenseRetriever
from llama_cpp import Llama
import numpy as np
"""
We have a benchmark dataset, with queries (docstrings)
"""

def create_snippets_embeddings(dataset, save_or_load = "load"):

    snippets = []
    queries = []
    ground_truth = []

    for idx, sample in enumerate(dataset):
        snippets.append(sample["whole_func_string"])
        # We use the docstring as the query
        queries.append(sample["func_documentation_string"])
        # Ground truth is that we know the ID of the corresponding snippets
        ground_truth.append(idx)

    print("Lists for snippets, queries, GT ready\n\n\n\n")

    if save_or_load == "embed_and_save":
        embeddings = flatretriever.embedder.embed(snippets,batch_size=16)
        np.save(r"C:\Users\PC\Documents\Github-RAG\data\embeddings\snippets_embeddings.npy",embeddings)
    elif save_or_load == "load":
        embeddings = np.load(r"C:\Users\PC\Documents\Github-RAG\data\embeddings\snippets_embeddings.npy")
    return snippets, queries, ground_truth, embeddings

def recall_at_k(retrieved,relevant):

    return int(relevant in retrieved)

def searchids_computerecall(retriever, queries, ground_truth):
    recalls = []

    for query, gt in zip(queries, ground_truth):

        query_embedding = retriever.embedder.embed([query],False)

        scores, ids = retriever.store.search_ids(query_embedding,k=10,nprobe=20) # Whichever method
        retrieved = ids[0].tolist()

        recalls.append(recall_at_k(retrieved, gt))

    return recalls

if __name__ == "__main__":

    import os
    print(os.cpu_count())

    llm = Llama(model_path=r"C:\Users\PC\Documents\Github-RAG\qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf",n_ctx=2048,n_threads=8, verbose=False)
    flatretriever = DenseRetriever(llm,index_path = r"C:\Users\PC\Documents\Github-RAG\data\embeddings\flat_faiss.index",
                                   metadata_path = r"C:\Users\PC\Documents\Github-RAG\data\embeddings\flat_metadata.json",)
    ivfretriever = DenseRetriever(llm,index_path = r"C:\Users\PC\Documents\Github-RAG\data\embeddings\ivf_faiss.index",
                                  metadata_path = r"C:\Users\PC\Documents\Github-RAG\data\embeddings\ivf_metadata.json",)
    with open(path) as f:
        dataset = json.load(f)

    snippets, queries, ground_truth, embeddings = create_snippets_embeddings(dataset, 4000, "load")

    print("Embeddings ready and saved\n")

    flatretriever.store.build(embeddings,snippets, "flat")
    flatretriever.store.save()
    ivfretriever.store.build(embeddings,snippets, "ivf")
    ivfretriever.store.save()

    print("VectorStore ready\n\n")

    flat_recalls = searchids_computerecall(flatretriever, queries, ground_truth)
    ivf_recalls = searchids_computerecall(ivfretriever, queries, ground_truth)


    print(f"Flat Recall@10 = {np.mean(flat_recalls):.3f}")
    print(f"IVF Recall@10 = {np.mean(ivf_recalls):.3f}")

    llm.close()
