# GitHub RAG

## 1. Motivation

LLMs have enourmous reasoning capabilities at the cost that the original information, even though contained within the model, is embedded in the deep abstract layers. Retrieval-Augmented Generation (RAG) aims to overcome this shortage by providing context to the LLM in the form of a prompt:

```
prompt=f"""
Answer using this context:

{context}

Question:
{Original user question}
"""
```

## 2. Problem

Finding the correct context / code snippet $s_i$ from the user query $q$ is the main task of a RAG, and is done in embedding space. That is, we use the encoder to find the vectors $\eta_s^i$ and $\eta_q$ and search the snippets using either `FlatIndex` or `InverseFileIndex` to find the best `top_k` matching snippets: $Min_i\(\eta_s^i, \eta_q\)$. We usually search for the best 10 matching snippets $$\\{i\\}$$.

Apart from the Indexing method, we can use techniques like Multi-Query searching, where we find, using an LLM, four equivalent but differently worded queries $\eta_q^j$. We then find the best 10 matching snippets for each $j$, that is $$\\{i\\}^j$$, and compute the best 10 matches of this complete set $\\{i\\}_{multiquery}$. This is expected to give a better result than single-querying.

## 3. Methods

We use the labeled dataset `CodeSearchNet`, which contains *code* functions and their *annotation* which is the docstring.

Example function: 
```
def is_palindrome(s):
    """
    Checks whether a string is a palindrome.

    Args:
        s: The string to check.

    Returns:
        True if the string reads the same backwards, otherwise False.
    """
    return s == s[::-1]
```
And its docstring:
`Checks whether a string is a palindrome.`

### Benchmarking three retrieval methods with `CodeSearchNet`

First, we search the snippets using the docstring directly (this is cheating, but a reference/benchmark): `identical_query`.

Then, we search using an equivalent but differently worded query (real scenario simulation): `single_query`.
We create four similar queries using Llama and Qwen.

Lastly, we implement the multi-query, simulating how we will combine different queries: `multi_query`.

## 4. Results

Knowing the Ground-truth index of the snippets and their docstrings, we can compute the Recall for each of the six retrieving methods.
We use 540 augmented samples. IVF is meant to speed up the search of large databases while compromising accuracy/recall. We obtain:
```
                         Identical Query         Single-query           Multi-query

Flat Recall@10 =             0.961                  0.943                 0.972

IVF Recall@10 =              0.950                  0.930                 0.967
```
## 5. Discussion

We expected identical query to give the best results. However, some functions may be badly annotated, reducing recall. 
Single-query shows the **lowest recall** / capacity of finding the correct code.
Multi-query combines different querys of different wording, overcoming the badly phrased docstrings and finding the relevant context code snippets with **highest recall**.

Lastly, we see that the IVF method indeed compromises recall at the advantage of performing the fastest search.

## 6. Outlook

This benchmarking method will allow us to compare future Retrievers and techniques to maximize context quality, for example Hybrid retrieval.
Next necessary step in this project will be to chunk the downloaded Github repo (which we want to build a RAG for) respecting functions, classes, etc. rather than blocks of 100 lines.
Lastly, we wonder if the context needs multiple code snippets, how we could benchmark this scenario, since `CodeSearchNet` only contains single functions that dont necessarily work together.

