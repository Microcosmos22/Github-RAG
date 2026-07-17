from datasets import load_dataset
from src.retriever import FlatRetriever
from llama_cpp import Llama

llm = Llama(model_path=r"C:\Users\PC\Documents\Github-RAG\qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf",n_ctx=8192,n_threads=8)
retriever = FlatRetriever(llm)
dataset = load_dataset("code-search-net/code_search_net","python")

snippets = []
queries = []
ground_truth = []

for idx, sample in enumerate(dataset["test"]):

    snippets.append(sample["whole_func_string"])

    # We use the docstring as the query
    queries.append(sample["func_documentation_string"])
    # Ground truth is that we know the ID of the corresponding snippets
    ground_truth.append(idx)

# Make a semantic vector out of the snippet
embeddings = retriever.embedder.embed(
    snippets,
    batch_size=16
)

store.build(
    embeddings,
    snippets
)

query_embedding = embedder.embed([query])

scores, ids = store.search(
    query_embedding,
    k=10
)

def recall_at_k(retrieved,relevant):

    return int(relevant in retrieved)

recalls = []

for query, gt in zip(queries, ground_truth):

    retrieved = ...

    recalls.append(recall_at_k(retrieved,gt,))

print(np.mean(recalls))
