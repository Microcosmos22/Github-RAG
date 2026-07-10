from sentence_transformers import SentenceTransformer

# This will automatically download the BAAI/bge-code-v1 model
model = SentenceTransformer("BAAI/bge-code-v1")

# Example usage
sentences = ["def hello_world():", "print('Hello, world!')"]
embeddings = model.encode(sentences)
print(embeddings.shape)

class Embedder:

    def __init__(self,model_name):
        self.model = load_model(model_name)


    def embed(self,texts):
        return vectors
