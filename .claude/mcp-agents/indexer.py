#!/usr/bin/env python3
"""
Prism4D Codebase Indexer for RAG
Indexes all code, docs, and comments into LanceDB for retrieval.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass
import lancedb
from sentence_transformers import SentenceTransformer

# Configuration
PRISM_ROOT = Path(__file__).parent.parent.parent
DB_PATH = Path(__file__).parent / "prism_rag.lance"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good for code

# File patterns to index
PATTERNS = {
    "rust": ["**/*.rs"],
    "cuda": ["**/*.cu", "**/*.cuh"],
    "python": ["**/*.py"],
    "markdown": ["**/*.md"],
    "toml": ["**/Cargo.toml"],
    "shell": ["**/*.sh"],
}

# Directories to skip
SKIP_DIRS = {"target", "node_modules", ".git", "__pycache__", "*.lance"}


@dataclass
class CodeChunk:
    """A chunk of code with metadata for indexing."""
    file_path: str
    content: str
    chunk_type: str  # function, struct, module, comment, doc
    name: str  # function/struct name or section header
    line_start: int
    line_end: int
    language: str
    domain: str  # cuda, physics, ml, nhs, etc.


def detect_domain(file_path: str) -> str:
    """Detect which Prism4D domain a file belongs to."""
    path = file_path.lower()

    if "prism-gpu" in path or ".cu" in path:
        if "amber" in path:
            return "md-physics"
        if "dendritic" in path or "dqn" in path:
            return "ml-neural"
        if "quantum" in path:
            return "quantum"
        return "cuda-gpu"

    if "prism-nhs" in path:
        return "neuromorphic"

    if "prism-physics" in path or "prism-phases" in path:
        return "md-physics"

    if "prism-learning" in path or "prism-gnn" in path or "prism-fluxnet" in path:
        return "ml-neural"

    if "prism-lbs" in path or "prism-amber-prep" in path:
        return "bioinformatics"

    if "prism-io" in path or "prism-pipeline" in path:
        return "io-systems"

    if "prism-validation" in path or "bench" in path:
        return "validation"

    if "scripts" in path:
        return "python-scripts"

    return "general"


def chunk_rust_file(file_path: Path) -> Iterator[CodeChunk]:
    """Parse Rust file into semantic chunks."""
    content = file_path.read_text(errors="ignore")
    lines = content.split("\n")
    domain = detect_domain(str(file_path))
    rel_path = str(file_path.relative_to(PRISM_ROOT))

    # Pattern for function/impl/struct definitions
    patterns = [
        (r"^(pub\s+)?(async\s+)?fn\s+(\w+)", "function"),
        (r"^(pub\s+)?struct\s+(\w+)", "struct"),
        (r"^(pub\s+)?enum\s+(\w+)", "enum"),
        (r"^(pub\s+)?trait\s+(\w+)", "trait"),
        (r"^impl(?:<[^>]+>)?\s+(\w+)", "impl"),
        (r"^(pub\s+)?mod\s+(\w+)", "module"),
    ]

    current_chunk = []
    current_name = "module_header"
    current_type = "module"
    chunk_start = 0
    brace_depth = 0

    for i, line in enumerate(lines):
        # Track brace depth
        brace_depth += line.count("{") - line.count("}")

        # Check for new definition at top level
        if brace_depth <= 1:
            for pattern, chunk_type in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Yield previous chunk if substantial
                    if current_chunk and len("\n".join(current_chunk)) > 50:
                        yield CodeChunk(
                            file_path=rel_path,
                            content="\n".join(current_chunk),
                            chunk_type=current_type,
                            name=current_name,
                            line_start=chunk_start + 1,
                            line_end=i,
                            language="rust",
                            domain=domain,
                        )

                    # Start new chunk
                    current_chunk = []
                    current_name = match.group(match.lastindex) if match.lastindex else "unknown"
                    current_type = chunk_type
                    chunk_start = i
                    break

        current_chunk.append(line)

    # Yield final chunk
    if current_chunk and len("\n".join(current_chunk)) > 50:
        yield CodeChunk(
            file_path=rel_path,
            content="\n".join(current_chunk),
            chunk_type=current_type,
            name=current_name,
            line_start=chunk_start + 1,
            line_end=len(lines),
            language="rust",
            domain=domain,
        )


def chunk_cuda_file(file_path: Path) -> Iterator[CodeChunk]:
    """Parse CUDA file into semantic chunks."""
    content = file_path.read_text(errors="ignore")
    lines = content.split("\n")
    domain = detect_domain(str(file_path))
    rel_path = str(file_path.relative_to(PRISM_ROOT))

    # Pattern for CUDA kernels and functions
    patterns = [
        (r"^__global__\s+void\s+(\w+)", "kernel"),
        (r"^__device__\s+\w+\s+(\w+)", "device_function"),
        (r"^__host__\s+\w+\s+(\w+)", "host_function"),
        (r"^(static\s+)?(inline\s+)?\w+\s+(\w+)\s*\(", "function"),
    ]

    current_chunk = []
    current_name = "file_header"
    current_type = "header"
    chunk_start = 0
    brace_depth = 0

    for i, line in enumerate(lines):
        brace_depth += line.count("{") - line.count("}")

        if brace_depth <= 0:
            for pattern, chunk_type in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    if current_chunk and len("\n".join(current_chunk)) > 30:
                        yield CodeChunk(
                            file_path=rel_path,
                            content="\n".join(current_chunk),
                            chunk_type=current_type,
                            name=current_name,
                            line_start=chunk_start + 1,
                            line_end=i,
                            language="cuda",
                            domain=domain,
                        )

                    current_chunk = []
                    current_name = match.group(match.lastindex) if match.lastindex else "unknown"
                    current_type = chunk_type
                    chunk_start = i
                    break

        current_chunk.append(line)

    if current_chunk and len("\n".join(current_chunk)) > 30:
        yield CodeChunk(
            file_path=rel_path,
            content="\n".join(current_chunk),
            chunk_type=current_type,
            name=current_name,
            line_start=chunk_start + 1,
            line_end=len(lines),
            language="cuda",
            domain=domain,
        )


def chunk_python_file(file_path: Path) -> Iterator[CodeChunk]:
    """Parse Python file into semantic chunks."""
    content = file_path.read_text(errors="ignore")
    lines = content.split("\n")
    domain = detect_domain(str(file_path))
    rel_path = str(file_path.relative_to(PRISM_ROOT))

    patterns = [
        (r"^class\s+(\w+)", "class"),
        (r"^(async\s+)?def\s+(\w+)", "function"),
    ]

    current_chunk = []
    current_name = "module"
    current_type = "module"
    chunk_start = 0
    indent_level = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped) if stripped else indent_level

        if current_indent == 0 and stripped:
            for pattern, chunk_type in patterns:
                match = re.match(pattern, stripped)
                if match:
                    if current_chunk and len("\n".join(current_chunk)) > 30:
                        yield CodeChunk(
                            file_path=rel_path,
                            content="\n".join(current_chunk),
                            chunk_type=current_type,
                            name=current_name,
                            line_start=chunk_start + 1,
                            line_end=i,
                            language="python",
                            domain=domain,
                        )

                    current_chunk = []
                    current_name = match.group(match.lastindex) if match.lastindex else "unknown"
                    current_type = chunk_type
                    chunk_start = i
                    break

        current_chunk.append(line)
        if stripped:
            indent_level = current_indent

    if current_chunk and len("\n".join(current_chunk)) > 30:
        yield CodeChunk(
            file_path=rel_path,
            content="\n".join(current_chunk),
            chunk_type=current_type,
            name=current_name,
            line_start=chunk_start + 1,
            line_end=len(lines),
            language="python",
            domain=domain,
        )


def chunk_markdown_file(file_path: Path) -> Iterator[CodeChunk]:
    """Parse Markdown file into sections."""
    content = file_path.read_text(errors="ignore")
    lines = content.split("\n")
    domain = detect_domain(str(file_path))
    rel_path = str(file_path.relative_to(PRISM_ROOT))

    current_chunk = []
    current_name = "introduction"
    chunk_start = 0

    for i, line in enumerate(lines):
        if line.startswith("#"):
            if current_chunk and len("\n".join(current_chunk)) > 20:
                yield CodeChunk(
                    file_path=rel_path,
                    content="\n".join(current_chunk),
                    chunk_type="documentation",
                    name=current_name,
                    line_start=chunk_start + 1,
                    line_end=i,
                    language="markdown",
                    domain=domain,
                )

            current_chunk = []
            current_name = line.lstrip("#").strip()
            chunk_start = i

        current_chunk.append(line)

    if current_chunk and len("\n".join(current_chunk)) > 20:
        yield CodeChunk(
            file_path=rel_path,
            content="\n".join(current_chunk),
            chunk_type="documentation",
            name=current_name,
            line_start=chunk_start + 1,
            line_end=len(lines),
            language="markdown",
            domain=domain,
        )


def iter_files() -> Iterator[Path]:
    """Iterate over all indexable files in Prism4D."""
    for lang, patterns in PATTERNS.items():
        for pattern in patterns:
            for file_path in PRISM_ROOT.glob(pattern):
                # Skip excluded directories
                if any(skip in str(file_path) for skip in SKIP_DIRS):
                    continue
                yield file_path


def index_codebase():
    """Index entire Prism4D codebase into LanceDB."""
    print("🚀 Starting Prism4D codebase indexing...")

    # Initialize embedding model
    print("📦 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Collect all chunks
    print("📂 Parsing source files...")
    chunks = []

    chunkers = {
        ".rs": chunk_rust_file,
        ".cu": chunk_cuda_file,
        ".cuh": chunk_cuda_file,
        ".py": chunk_python_file,
        ".md": chunk_markdown_file,
    }

    for file_path in iter_files():
        suffix = file_path.suffix
        if suffix in chunkers:
            try:
                for chunk in chunkers[suffix](file_path):
                    chunks.append(chunk)
            except Exception as e:
                print(f"  ⚠️ Error parsing {file_path}: {e}")

    print(f"📊 Found {len(chunks)} code chunks")

    # Generate embeddings
    print("🧮 Generating embeddings...")
    texts = [f"{c.name}: {c.content[:1000]}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    # Prepare data for LanceDB
    print("💾 Writing to LanceDB...")
    data = []
    for chunk, embedding in zip(chunks, embeddings):
        data.append({
            "id": hashlib.md5(f"{chunk.file_path}:{chunk.line_start}".encode()).hexdigest(),
            "file_path": chunk.file_path,
            "content": chunk.content[:4000],  # Truncate very long chunks
            "chunk_type": chunk.chunk_type,
            "name": chunk.name,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "language": chunk.language,
            "domain": chunk.domain,
            "vector": embedding.tolist(),
        })

    # Create LanceDB table
    db = lancedb.connect(str(DB_PATH))

    # Drop existing table if exists
    try:
        db.drop_table("code_chunks")
    except:
        pass

    table = db.create_table("code_chunks", data)

    # Create index for fast retrieval
    table.create_index(num_partitions=16, num_sub_vectors=32)

    print(f"✅ Indexed {len(data)} chunks into {DB_PATH}")

    # Print domain distribution
    from collections import Counter
    domain_counts = Counter(c.domain for c in chunks)
    print("\n📈 Domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    index_codebase()
