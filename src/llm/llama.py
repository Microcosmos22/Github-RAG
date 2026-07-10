from llama_cpp import Llama

"""
Step 8: Create LLM wrapper
"""

class LocalLLM:

    def __init__(self,path):

        self.model=Llama(
            model_path=path
        )


    def generate(self,prompt):

        output=self.model(
            prompt,
            max_tokens=512
        )

        return output["choices"][0]["text"]
