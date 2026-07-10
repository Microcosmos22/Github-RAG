"""A file like:
torch/cuda/__init__.py
may have 2000 lines.
The LLM cannot see everything.
You split:
File
class Tensor:
    ...
    function cuda():
    ...
becomes:

Chunk 1:
class Tensor...

Chunk 2:
function cuda...

OUTPUT

[
 {
  "text": "def cuda()...",
  "file": "torch/cuda.py",
  "line": 100
 }
]"""

class CodeChunker:

    def __init__(
        self,
        chunk_size=1000,
        overlap=200
    ):
        self.chunk_size=chunk_size
        self.overlap=overlap


    def split_file(self,path):
        pass
