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
    "text": "...",
    "file": "torch/cuda/memory.py",
    "start_line": 120,
    "end_line": 158
  },
  ...
]
"""

from pathlib import Path
import json


class CodeChunker:

    def __init__(self,input_dir="data/raw/repo_files",output_file="data/chunks/chunks.json",
                 chunk_size=100,overlap=20):

        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.extensions = {".py",".cpp",".cc",".c",".hpp",".h",".md",".rst"}

    def get_files(self):
        return [f for f in self.input_dir.rglob("*") if f.suffix in self.extensions]

    def chunk_file(self, path):

        lines = path.read_text(errors="ignore").splitlines()
        chunks = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(lines), step):
            end = min(start + self.chunk_size, len(lines))

            chunks.append({
                "text": "\n".join(lines[start:end]),
                "file": str(path.relative_to(self.input_dir)),
                "extension": path.suffix,
                "start_line": start + 1,
                "end_line": end
            })

        return chunks

    def chunk_repository(self):

        chunks = []

        for file in self.get_files():
            chunks.extend(self.chunk_file(file))

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.output_file.write_text(
            json.dumps(chunks, indent=2),
            encoding="utf-8"
        )

        print(f"Saved {len(chunks)} chunks.")

if __name__ == "__main__":
    chunker = CodeChunker()

    chunker.chunk_repository()
