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


class RAGPipeline:


    def __init__(
        self,
        retriever,
        llm
    ):
        self.retriever=retriever
        self.llm=llm



    def ask(self,question):

        context=self.retriever.search(
            question
        )

        prompt=f"""
        Answer using this context:

        {context}

        Question:
        {question}
        """

        return self.llm.generate(prompt)
