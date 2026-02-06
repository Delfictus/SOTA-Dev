#!/usr/bin/env python3
"""
Prism4D RAG MCP Server
Provides semantic code search and domain-specific retrieval for Claude Code.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import lancedb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sentence_transformers import SentenceTransformer

# Configuration
DB_PATH = Path(__file__).parent / "prism_rag.lance"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Domain descriptions for agent routing
DOMAINS = {
    "cuda-gpu": "CUDA kernels, GPU optimization, PTX, Tensor Cores, sm_120 Blackwell",
    "md-physics": "Molecular dynamics, AMBER force fields, Langevin integrators, constraints",
    "ml-neural": "Machine learning, RL agents, GNNs, DendriticAgent, ONNX",
    "neuromorphic": "NHS engine, UV-aromatic coupling, spike detection, transfer entropy",
    "bioinformatics": "PDB parsing, topology generation, glycans, protein structure",
    "io-systems": "rkyv serialization, io_uring, async streaming, memory mapping",
    "validation": "Benchmarks, ATLAS dataset, Hit@K metrics, testing",
    "python-scripts": "Pipeline orchestration, visualization, data preparation",
    "quantum": "Quantum simulations, PIMC, quantum kernels",
    "general": "Core utilities, shared types, configuration",
}

# Initialize server
server = Server("prism-rag")

# Lazy-loaded resources
_model = None
_db = None
_table = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_table():
    global _db, _table
    if _table is None:
        _db = lancedb.connect(str(DB_PATH))
        _table = _db.open_table("code_chunks")
    return _table


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG tools."""
    return [
        Tool(
            name="search_code",
            description="Semantic search across Prism4D codebase. Returns relevant code chunks with file paths and line numbers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you're looking for"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Optional: filter by domain (cuda-gpu, md-physics, ml-neural, neuromorphic, bioinformatics, io-systems, validation, python-scripts, quantum)",
                        "enum": list(DOMAINS.keys())
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional: filter by language",
                        "enum": ["rust", "cuda", "python", "markdown"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="find_similar",
            description="Find code similar to a given snippet. Useful for finding related implementations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_snippet": {
                        "type": "string",
                        "description": "Code snippet to find similar implementations for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["code_snippet"]
            }
        ),
        Tool(
            name="get_domain_overview",
            description="Get an overview of a specific domain in the Prism4D codebase.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain to get overview for",
                        "enum": list(DOMAINS.keys())
                    }
                },
                "required": ["domain"]
            }
        ),
        Tool(
            name="find_function",
            description="Find a specific function, struct, or kernel by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of function/struct/kernel to find"
                    },
                    "chunk_type": {
                        "type": "string",
                        "description": "Optional: type of code element",
                        "enum": ["function", "struct", "kernel", "class", "impl", "trait"]
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="list_domains",
            description="List all domains in Prism4D with descriptions and chunk counts.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a RAG tool."""

    if name == "search_code":
        return await search_code(
            arguments["query"],
            arguments.get("domain"),
            arguments.get("language"),
            arguments.get("limit", 10)
        )

    elif name == "find_similar":
        return await find_similar(
            arguments["code_snippet"],
            arguments.get("limit", 5)
        )

    elif name == "get_domain_overview":
        return await get_domain_overview(arguments["domain"])

    elif name == "find_function":
        return await find_function(
            arguments["name"],
            arguments.get("chunk_type")
        )

    elif name == "list_domains":
        return await list_domains()

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_code(query: str, domain: str = None, language: str = None, limit: int = 10) -> list[TextContent]:
    """Semantic search across codebase."""
    model = get_model()
    table = get_table()

    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Build search query
    search = table.search(query_embedding).limit(limit * 2)  # Get extra for filtering

    results = search.to_list()

    # Apply filters
    if domain:
        results = [r for r in results if r["domain"] == domain]
    if language:
        results = [r for r in results if r["language"] == language]

    results = results[:limit]

    # Format output
    output = f"## Search Results for: {query}\n\n"

    if not results:
        output += "No results found. Try a different query or remove filters."
    else:
        for i, r in enumerate(results, 1):
            output += f"### {i}. {r['name']} ({r['chunk_type']})\n"
            output += f"**File:** `{r['file_path']}:{r['line_start']}-{r['line_end']}`\n"
            output += f"**Domain:** {r['domain']} | **Language:** {r['language']}\n"
            output += f"```{r['language']}\n{r['content'][:1500]}{'...' if len(r['content']) > 1500 else ''}\n```\n\n"

    return [TextContent(type="text", text=output)]


async def find_similar(code_snippet: str, limit: int = 5) -> list[TextContent]:
    """Find similar code to a snippet."""
    model = get_model()
    table = get_table()

    embedding = model.encode(code_snippet).tolist()
    results = table.search(embedding).limit(limit).to_list()

    output = "## Similar Code Chunks\n\n"

    for i, r in enumerate(results, 1):
        output += f"### {i}. {r['name']} ({r['chunk_type']})\n"
        output += f"**File:** `{r['file_path']}:{r['line_start']}`\n"
        output += f"```{r['language']}\n{r['content'][:1000]}\n```\n\n"

    return [TextContent(type="text", text=output)]


async def get_domain_overview(domain: str) -> list[TextContent]:
    """Get overview of a domain."""
    table = get_table()

    # Get all chunks for this domain
    results = table.search().where(f"domain = '{domain}'").limit(1000).to_list()

    output = f"## Domain Overview: {domain}\n\n"
    output += f"**Description:** {DOMAINS.get(domain, 'Unknown domain')}\n\n"
    output += f"**Total chunks:** {len(results)}\n\n"

    # Group by file
    files = {}
    for r in results:
        fp = r["file_path"]
        if fp not in files:
            files[fp] = []
        files[fp].append(r)

    output += f"**Files:** {len(files)}\n\n"

    # List key files
    output += "### Key Files\n"
    for fp, chunks in sorted(files.items(), key=lambda x: -len(x[1]))[:15]:
        names = [c["name"] for c in chunks[:5]]
        output += f"- `{fp}` ({len(chunks)} chunks): {', '.join(names)}{'...' if len(chunks) > 5 else ''}\n"

    # List important structs/functions
    important = [r for r in results if r["chunk_type"] in ("struct", "kernel", "trait", "class")]
    if important:
        output += "\n### Key Types/Kernels\n"
        for r in important[:20]:
            output += f"- **{r['name']}** ({r['chunk_type']}) in `{r['file_path']}:{r['line_start']}`\n"

    return [TextContent(type="text", text=output)]


async def find_function(name: str, chunk_type: str = None) -> list[TextContent]:
    """Find a specific function by name."""
    table = get_table()

    # Search by name
    where_clause = f"name = '{name}'"
    if chunk_type:
        where_clause += f" AND chunk_type = '{chunk_type}'"

    results = table.search().where(where_clause).limit(10).to_list()

    if not results:
        # Fall back to fuzzy search
        model = get_model()
        embedding = model.encode(f"function {name}").tolist()
        results = table.search(embedding).limit(5).to_list()
        results = [r for r in results if name.lower() in r["name"].lower()]

    output = f"## Results for: {name}\n\n"

    if not results:
        output += f"No exact match found for '{name}'. Try search_code for semantic search."
    else:
        for r in results:
            output += f"### {r['name']} ({r['chunk_type']})\n"
            output += f"**File:** `{r['file_path']}:{r['line_start']}-{r['line_end']}`\n"
            output += f"**Domain:** {r['domain']}\n"
            output += f"```{r['language']}\n{r['content']}\n```\n\n"

    return [TextContent(type="text", text=output)]


async def list_domains() -> list[TextContent]:
    """List all domains with stats."""
    table = get_table()

    output = "## Prism4D Domains\n\n"

    all_results = table.search().limit(10000).to_list()

    from collections import Counter
    domain_counts = Counter(r["domain"] for r in all_results)

    for domain, description in DOMAINS.items():
        count = domain_counts.get(domain, 0)
        output += f"### {domain}\n"
        output += f"**Chunks:** {count}\n"
        output += f"**Description:** {description}\n\n"

    return [TextContent(type="text", text=output)]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
