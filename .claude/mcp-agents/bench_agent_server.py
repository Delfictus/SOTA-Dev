#!/usr/bin/env python3
"""
Validation & Benchmarking MCP Server
Provides tools for running benchmarks, analyzing metrics, and validation testing.
"""

import asyncio
import subprocess
import json
import re
from pathlib import Path
from typing import Any
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

PRISM_ROOT = Path(__file__).parent.parent.parent
VALIDATION_DIR = PRISM_ROOT / "crates" / "prism-validation"
DATA_DIR = PRISM_ROOT / "data"

server = Server("bench-agent")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_benchmarks",
            description="List all available validation benchmarks and their status.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="run_benchmark",
            description="Run a specific benchmark and return results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "benchmark": {
                        "type": "string",
                        "description": "Name of the benchmark binary (e.g., 'atlas_validation')"
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments to pass"
                    }
                },
                "required": ["benchmark"]
            }
        ),
        Tool(
            name="analyze_results",
            description="Analyze benchmark results and compute metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "results_file": {
                        "type": "string",
                        "description": "Path to results JSON file"
                    }
                },
                "required": ["results_file"]
            }
        ),
        Tool(
            name="compute_hit_at_k",
            description="Compute Hit@K metrics from prediction results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "string",
                        "description": "JSON string of predictions [{target, predicted_sites, true_sites}]"
                    },
                    "k_values": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "K values to compute (default: [1, 3, 5, 10])"
                    }
                },
                "required": ["predictions"]
            }
        ),
        Tool(
            name="compare_methods",
            description="Compare two methods' benchmark results statistically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "method_a_results": {"type": "string", "description": "Path to method A results"},
                    "method_b_results": {"type": "string", "description": "Path to method B results"}
                },
                "required": ["method_a_results", "method_b_results"]
            }
        ),
        Tool(
            name="get_dataset_stats",
            description="Get statistics about a benchmark dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name (atlas, apo_holo, retrospective)"
                    }
                },
                "required": ["dataset"]
            }
        ),
        Tool(
            name="profile_performance",
            description="Profile performance of a binary and identify bottlenecks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "binary": {"type": "string", "description": "Name of the binary to profile"},
                    "duration": {"type": "integer", "description": "Profile duration in seconds (default: 30)"}
                },
                "required": ["binary"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "list_benchmarks":
        return await list_benchmarks()
    elif name == "run_benchmark":
        return await run_benchmark(arguments["benchmark"], arguments.get("args", ""))
    elif name == "analyze_results":
        return await analyze_results(arguments["results_file"])
    elif name == "compute_hit_at_k":
        return await compute_hit_at_k(
            arguments["predictions"],
            arguments.get("k_values", [1, 3, 5, 10])
        )
    elif name == "compare_methods":
        return await compare_methods(
            arguments["method_a_results"],
            arguments["method_b_results"]
        )
    elif name == "get_dataset_stats":
        return await get_dataset_stats(arguments["dataset"])
    elif name == "profile_performance":
        return await profile_performance(
            arguments["binary"],
            arguments.get("duration", 30)
        )
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def list_benchmarks() -> list[TextContent]:
    """List all validation benchmarks."""
    output = "## Available Benchmarks\n\n"

    # Find all validation binaries
    bin_dir = VALIDATION_DIR / "src" / "bin"
    if bin_dir.exists():
        bins = list(bin_dir.glob("*.rs"))
        output += f"**Validation binaries:** {len(bins)}\n\n"

        categories = {
            "atlas": [],
            "apo": [],
            "retro": [],
            "perf": [],
            "other": []
        }

        for b in bins:
            name = b.stem
            if "atlas" in name:
                categories["atlas"].append(name)
            elif "apo" in name or "holo" in name:
                categories["apo"].append(name)
            elif "retro" in name:
                categories["retro"].append(name)
            elif "perf" in name or "bench" in name:
                categories["perf"].append(name)
            else:
                categories["other"].append(name)

        for cat, bins in categories.items():
            if bins:
                output += f"### {cat.title()}\n"
                for b in sorted(bins):
                    output += f"- `{b}`\n"
                output += "\n"

    # Check for pre-built binaries
    release_dir = PRISM_ROOT / "target" / "release"
    if release_dir.exists():
        built = [f.name for f in release_dir.iterdir() if f.is_file() and not f.suffix]
        validation_bins = [b for b in built if "valid" in b or "bench" in b or "atlas" in b]
        if validation_bins:
            output += f"\n**Pre-built binaries:** {len(validation_bins)}\n"

    return [TextContent(type="text", text=output)]


async def run_benchmark(benchmark: str, args: str = "") -> list[TextContent]:
    """Run a benchmark."""
    output = f"## Running Benchmark: {benchmark}\n\n"

    binary_path = PRISM_ROOT / "target" / "release" / benchmark

    if not binary_path.exists():
        output += f"Binary not found at `{binary_path}`\n"
        output += "\nBuilding...\n"

        # Try to build
        try:
            proc = await asyncio.create_subprocess_exec(
                "cargo", "build", "--release", "--bin", benchmark,
                cwd=str(PRISM_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)

            if proc.returncode != 0:
                output += f"Build failed:\n```\n{stderr.decode()}\n```"
                return [TextContent(type="text", text=output)]

            output += "Build successful.\n\n"
        except asyncio.TimeoutError:
            output += "Build timed out after 5 minutes."
            return [TextContent(type="text", text=output)]

    # Run benchmark
    output += f"**Command:** `{binary_path} {args}`\n"
    output += f"**Started:** {datetime.now().isoformat()}\n\n"

    try:
        cmd = [str(binary_path)] + (args.split() if args else [])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PRISM_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

        output += "### Output\n```\n"
        output += stdout.decode()[:5000]
        if len(stdout) > 5000:
            output += "\n... (truncated)"
        output += "\n```\n"

        if stderr:
            output += "\n### Stderr\n```\n"
            output += stderr.decode()[:2000]
            output += "\n```\n"

        output += f"\n**Exit code:** {proc.returncode}\n"
        output += f"**Finished:** {datetime.now().isoformat()}\n"

    except asyncio.TimeoutError:
        output += "⚠️ Benchmark timed out after 10 minutes."

    return [TextContent(type="text", text=output)]


async def compute_hit_at_k(predictions_json: str, k_values: list[int]) -> list[TextContent]:
    """Compute Hit@K metrics."""
    output = "## Hit@K Analysis\n\n"

    try:
        predictions = json.loads(predictions_json)
    except json.JSONDecodeError:
        return [TextContent(type="text", text="Invalid JSON in predictions")]

    results = {}
    for k in k_values:
        hits = 0
        total = len(predictions)

        for pred in predictions:
            predicted = set(pred.get("predicted_sites", [])[:k])
            true = set(pred.get("true_sites", []))

            if predicted & true:  # Any overlap
                hits += 1

        results[k] = hits / total if total > 0 else 0

    output += "| K | Hit@K | Hits | Total |\n"
    output += "|---|-------|------|-------|\n"
    for k in k_values:
        hit_rate = results[k]
        hits = int(hit_rate * len(predictions))
        output += f"| {k} | {hit_rate:.1%} | {hits} | {len(predictions)} |\n"

    # Analysis
    output += "\n### Analysis\n"
    if results.get(1, 0) < 0.1:
        output += "- ⚠️ Hit@1 is very low (<10%) - ranking algorithm needs improvement\n"
    if results.get(5, 0) > 0.5:
        output += "- ✅ Hit@5 > 50% indicates good recall\n"

    gap = results.get(5, 0) - results.get(1, 0)
    if gap > 0.4:
        output += f"- 💡 Large gap between Hit@1 and Hit@5 ({gap:.1%}) suggests ranking issue, not detection issue\n"

    return [TextContent(type="text", text=output)]


async def analyze_results(results_file: str) -> list[TextContent]:
    """Analyze benchmark results file."""
    output = f"## Results Analysis: {results_file}\n\n"

    file_path = Path(results_file)
    if not file_path.is_absolute():
        file_path = PRISM_ROOT / file_path

    if not file_path.exists():
        return [TextContent(type="text", text=f"File not found: {results_file}")]

    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [TextContent(type="text", text=f"Invalid JSON: {e}")]

    # Generic analysis
    output += f"**Keys:** {list(data.keys())}\n\n"

    if isinstance(data, list):
        output += f"**Entries:** {len(data)}\n"

        # Try to extract common metrics
        if data and isinstance(data[0], dict):
            sample = data[0]
            output += f"**Sample entry keys:** {list(sample.keys())}\n"

    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (int, float)):
                output += f"- **{key}:** {value}\n"
            elif isinstance(value, list):
                output += f"- **{key}:** {len(value)} items\n"

    return [TextContent(type="text", text=output)]


async def compare_methods(method_a_results: str, method_b_results: str) -> list[TextContent]:
    """Compare two methods statistically."""
    output = "## Method Comparison\n\n"

    # This would load both result files and perform statistical comparison
    output += "**Method A:** " + method_a_results + "\n"
    output += "**Method B:** " + method_b_results + "\n\n"

    output += "⚠️ Full statistical comparison requires scipy. "
    output += "Install with: `pip install scipy`\n\n"

    output += "### Recommended Tests\n"
    output += "- **Paired t-test** for normally distributed metrics\n"
    output += "- **Wilcoxon signed-rank** for non-parametric comparison\n"
    output += "- **McNemar's test** for binary outcomes (hit/miss)\n"

    return [TextContent(type="text", text=output)]


async def get_dataset_stats(dataset: str) -> list[TextContent]:
    """Get dataset statistics."""
    output = f"## Dataset: {dataset}\n\n"

    dataset_paths = {
        "atlas": DATA_DIR / "atlas",
        "apo_holo": DATA_DIR / "apo_holo",
        "retrospective": DATA_DIR / "retrospective"
    }

    path = dataset_paths.get(dataset.lower())
    if not path:
        output += f"Unknown dataset. Available: {list(dataset_paths.keys())}"
        return [TextContent(type="text", text=output)]

    if not path.exists():
        output += f"Dataset not downloaded. Path: `{path}`\n"
        output += f"\nRun: `python scripts/download_{dataset}.py`"
        return [TextContent(type="text", text=output)]

    # Count files
    pdbs = list(path.glob("**/*.pdb"))
    jsons = list(path.glob("**/*.json"))

    output += f"**Location:** `{path}`\n"
    output += f"**PDB files:** {len(pdbs)}\n"
    output += f"**JSON files:** {len(jsons)}\n"

    # Get disk usage
    total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    output += f"**Total size:** {total_size / 1024 / 1024:.1f} MB\n"

    return [TextContent(type="text", text=output)]


async def profile_performance(binary: str, duration: int = 30) -> list[TextContent]:
    """Profile a binary's performance."""
    output = f"## Performance Profile: {binary}\n\n"

    output += "### Quick Profile Commands\n\n"

    output += "**CPU profiling (perf):**\n"
    output += f"```bash\nperf record -g ./target/release/{binary}\n"
    output += "perf report\n```\n\n"

    output += "**Flamegraph:**\n"
    output += f"```bash\ncargo flamegraph --bin {binary}\n```\n\n"

    output += "**GPU profiling (NVIDIA):**\n"
    output += f"```bash\nncu --set full ./target/release/{binary}\n"
    output += f"nsys profile ./target/release/{binary}\n```\n\n"

    output += "**Memory profiling:**\n"
    output += f"```bash\nheaptrack ./target/release/{binary}\n"
    output += f"valgrind --tool=massif ./target/release/{binary}\n```\n"

    return [TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
