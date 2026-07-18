"""
Goal of this file is to create a new CodeSearchNet dataset where the original docstrings = query are augmented
into three different formulations that describe the code-snippet.

Remember for retrieval we will encode(query) != encode(docstrings) try to find the best match of these two vectors.
samples = [
    {
        "id": 0,
        "snippet": "...",
        "queries": [
            original_docstring,
            paraphrase1,
            paraphrase2,
            paraphrase3
        ]
    },
    ...
]"""


import json

dataset = load_dataset("code-search-net/code_search_net","python")
small_test = dataset["test"].select(range(onlyfirstN))

dataset = []

for idx, sample in enumerate(small_test):

    doc = sample["func_documentation_string"]

    alternatives = generate_queries(doc)   # returns list of 3 strings

    dataset.append({
        "id": idx,
        "snippet": sample["whole_func_string"],
        "queries": [doc] + alternatives
    })

with open(
    r".\codesearchnet_queries.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(dataset, f, indent=2)
