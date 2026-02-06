#!/usr/bin/env python3
"""
CUDA/GPU Specialist MCP Server
Provides domain-specific tools for GPU kernel development and optimization.
"""

import asyncio
import subprocess
import re
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

PRISM_ROOT = Path(__file__).parent.parent.parent
KERNELS_DIR = PRISM_ROOT / "crates" / "prism-gpu" / "src" / "kernels"

server = Server("cuda-agent")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_kernels",
            description="List all CUDA kernels in Prism4D with their types (global/device/host).",
            inputSchema={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter pattern (e.g., 'amber', 'dendritic')"
                    }
                }
            }
        ),
        Tool(
            name="analyze_kernel",
            description="Analyze a CUDA kernel's resource usage (registers, shared memory, occupancy hints).",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_file": {
                        "type": "string",
                        "description": "Name of the .cu file (e.g., 'amber_bonded.cu')"
                    }
                },
                "required": ["kernel_file"]
            }
        ),
        Tool(
            name="find_tensor_core_usage",
            description="Find all Tensor Core (WMMA) operations in the codebase.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="check_memory_patterns",
            description="Analyze memory access patterns in a kernel for coalescing issues.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_file": {
                        "type": "string",
                        "description": "Name of the .cu file to analyze"
                    }
                },
                "required": ["kernel_file"]
            }
        ),
        Tool(
            name="get_ptx_info",
            description="Get PTX compilation info for a kernel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_name": {
                        "type": "string",
                        "description": "Name of the kernel (without extension)"
                    }
                },
                "required": ["kernel_name"]
            }
        ),
        Tool(
            name="suggest_optimizations",
            description="Suggest optimizations for a CUDA kernel based on common patterns.",
            inputSchema={
                "type": "object",
                "properties": {
                    "kernel_file": {
                        "type": "string",
                        "description": "Name of the .cu file"
                    }
                },
                "required": ["kernel_file"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "list_kernels":
        return await list_kernels(arguments.get("filter"))
    elif name == "analyze_kernel":
        return await analyze_kernel(arguments["kernel_file"])
    elif name == "find_tensor_core_usage":
        return await find_tensor_core_usage()
    elif name == "check_memory_patterns":
        return await check_memory_patterns(arguments["kernel_file"])
    elif name == "get_ptx_info":
        return await get_ptx_info(arguments["kernel_name"])
    elif name == "suggest_optimizations":
        return await suggest_optimizations(arguments["kernel_file"])
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def list_kernels(filter_pattern: str = None) -> list[TextContent]:
    """List all CUDA kernels."""
    output = "## CUDA Kernels in Prism4D\n\n"

    kernels = {"__global__": [], "__device__": [], "__host__": []}

    for cu_file in KERNELS_DIR.glob("*.cu"):
        if filter_pattern and filter_pattern.lower() not in cu_file.name.lower():
            continue

        try:
            content = cu_file.read_text()

            # Find kernel declarations
            for match in re.finditer(r'(__global__|__device__|__host__)\s+\w+\s+(\w+)\s*\(', content):
                kernel_type = match.group(1)
                kernel_name = match.group(2)
                kernels[kernel_type].append((cu_file.name, kernel_name))
        except:
            pass

    for ktype, klist in kernels.items():
        if klist:
            output += f"### {ktype} ({len(klist)})\n"
            for fname, kname in sorted(klist):
                output += f"- `{kname}` in `{fname}`\n"
            output += "\n"

    return [TextContent(type="text", text=output)]


async def analyze_kernel(kernel_file: str) -> list[TextContent]:
    """Analyze kernel resource usage."""
    file_path = KERNELS_DIR / kernel_file

    if not file_path.exists():
        return [TextContent(type="text", text=f"File not found: {kernel_file}")]

    content = file_path.read_text()
    output = f"## Kernel Analysis: {kernel_file}\n\n"

    # Count kernels
    global_kernels = re.findall(r'__global__\s+void\s+(\w+)', content)
    output += f"**Global kernels:** {len(global_kernels)}\n"
    for k in global_kernels:
        output += f"  - `{k}`\n"

    # Shared memory usage
    shared_decls = re.findall(r'__shared__\s+(\w+)\s+(\w+)\[([^\]]+)\]', content)
    if shared_decls:
        output += f"\n**Shared memory declarations:** {len(shared_decls)}\n"
        for dtype, name, size in shared_decls:
            output += f"  - `{dtype} {name}[{size}]`\n"

    # Syncthreads
    syncs = len(re.findall(r'__syncthreads\(\)', content))
    output += f"\n**__syncthreads() calls:** {syncs}\n"

    # Atomic operations
    atomics = re.findall(r'(atomic\w+)\(', content)
    if atomics:
        output += f"\n**Atomic operations:** {len(atomics)}\n"
        for a in set(atomics):
            output += f"  - `{a}` ({atomics.count(a)}x)\n"

    # WMMA usage
    wmma = len(re.findall(r'wmma::', content))
    if wmma:
        output += f"\n**Tensor Core (WMMA) ops:** {wmma}\n"

    # Line count
    lines = len(content.split('\n'))
    output += f"\n**Total lines:** {lines}\n"

    return [TextContent(type="text", text=output)]


async def find_tensor_core_usage() -> list[TextContent]:
    """Find all WMMA/Tensor Core usage."""
    output = "## Tensor Core (WMMA) Usage\n\n"

    for cu_file in KERNELS_DIR.glob("*.cu"):
        try:
            content = cu_file.read_text()
            if "wmma::" in content:
                output += f"### {cu_file.name}\n"

                # Find WMMA operations
                ops = re.findall(r'wmma::(\w+)', content)
                output += f"**Operations:** {', '.join(set(ops))}\n"

                # Find fragment types
                frags = re.findall(r'wmma::fragment<[^>]+>', content)
                if frags:
                    output += f"**Fragments:** {len(frags)}\n"
                    for f in set(frags):
                        output += f"  - `{f}`\n"

                output += "\n"
        except:
            pass

    return [TextContent(type="text", text=output)]


async def check_memory_patterns(kernel_file: str) -> list[TextContent]:
    """Analyze memory access patterns."""
    file_path = KERNELS_DIR / kernel_file

    if not file_path.exists():
        return [TextContent(type="text", text=f"File not found: {kernel_file}")]

    content = file_path.read_text()
    output = f"## Memory Pattern Analysis: {kernel_file}\n\n"

    issues = []
    suggestions = []

    # Check for strided access patterns
    if re.search(r'\[.*threadIdx\.x\s*\*', content):
        issues.append("Possible strided access (threadIdx.x * stride)")
        suggestions.append("Consider transposing data for coalesced access")

    # Check for shared memory bank conflicts
    if re.search(r'__shared__.*\[\d+\]\[(?:32|64|128)\]', content):
        issues.append("Possible shared memory bank conflicts (power-of-2 stride)")
        suggestions.append("Add +1 padding to inner dimension")

    # Check for uncoalesced global writes
    if re.search(r'[a-z_]+\[.*threadIdx\.y', content):
        issues.append("Global access indexed by threadIdx.y may be uncoalesced")
        suggestions.append("Reorder thread mapping so threadIdx.x indexes contiguous memory")

    # Check for register pressure indicators
    local_arrays = re.findall(r'(float|double|int)\s+\w+\[(\d+)\]', content)
    if local_arrays:
        for dtype, size in local_arrays:
            if int(size) > 16:
                issues.append(f"Large local array ({dtype}[{size}]) may spill to local memory")
                suggestions.append("Consider using shared memory or reducing array size")

    if issues:
        output += "### Potential Issues\n"
        for i in issues:
            output += f"- ⚠️ {i}\n"
        output += "\n### Suggestions\n"
        for s in suggestions:
            output += f"- 💡 {s}\n"
    else:
        output += "✅ No obvious memory access issues detected.\n"
        output += "\nNote: Profile with Nsight Compute for accurate analysis."

    return [TextContent(type="text", text=output)]


async def get_ptx_info(kernel_name: str) -> list[TextContent]:
    """Get PTX info for a kernel."""
    ptx_file = KERNELS_DIR / f"{kernel_name}.ptx"

    if not ptx_file.exists():
        # Try to find matching PTX
        matches = list(KERNELS_DIR.glob(f"*{kernel_name}*.ptx"))
        if matches:
            ptx_file = matches[0]
        else:
            return [TextContent(type="text", text=f"PTX not found for: {kernel_name}")]

    output = f"## PTX Info: {ptx_file.name}\n\n"

    stat = ptx_file.stat()
    output += f"**Size:** {stat.st_size:,} bytes\n"

    content = ptx_file.read_text()

    # Extract target
    target = re.search(r'\.target\s+(\w+)', content)
    if target:
        output += f"**Target:** {target.group(1)}\n"

    # Count functions
    funcs = re.findall(r'\.visible\s+\.entry\s+(\w+)', content)
    output += f"**Entry points:** {len(funcs)}\n"
    for f in funcs[:10]:
        output += f"  - `{f}`\n"
    if len(funcs) > 10:
        output += f"  - ... and {len(funcs) - 10} more\n"

    # Register usage (if available in PTX)
    regs = re.findall(r'\.reg\s+\.\w+\s+%\w+<(\d+)>', content)
    if regs:
        max_regs = max(int(r) for r in regs)
        output += f"**Max registers per thread:** ~{max_regs}\n"

    return [TextContent(type="text", text=output)]


async def suggest_optimizations(kernel_file: str) -> list[TextContent]:
    """Suggest optimizations for a kernel."""
    file_path = KERNELS_DIR / kernel_file

    if not file_path.exists():
        return [TextContent(type="text", text=f"File not found: {kernel_file}")]

    content = file_path.read_text()
    output = f"## Optimization Suggestions: {kernel_file}\n\n"

    suggestions = []

    # Check if using float vs double
    if "double" in content and "float" not in content:
        suggestions.append({
            "priority": "high",
            "category": "Precision",
            "suggestion": "Consider using float instead of double for 2x throughput",
            "details": "FP32 has 2x the throughput of FP64 on consumer GPUs"
        })

    # Check for Tensor Core opportunity
    if "wmma::" not in content and re.search(r'for.*for.*\+=', content):
        suggestions.append({
            "priority": "high",
            "category": "Tensor Cores",
            "suggestion": "Matrix multiply pattern detected - consider WMMA",
            "details": "Tensor Cores provide 8x+ speedup for FP16 matrix operations"
        })

    # Check for loop unrolling
    if re.search(r'for\s*\(.*<\s*\d+', content) and "#pragma unroll" not in content:
        suggestions.append({
            "priority": "medium",
            "category": "Loop Optimization",
            "suggestion": "Add #pragma unroll for small fixed-size loops",
            "details": "Compiler may not auto-unroll without hints"
        })

    # Check for shared memory
    if "__shared__" not in content and re.search(r'\[.*\+.*\].*\[.*\+.*\]', content):
        suggestions.append({
            "priority": "medium",
            "category": "Memory",
            "suggestion": "Consider shared memory tiling for data reuse",
            "details": "Shared memory is ~100x faster than global memory"
        })

    # Check for warp-level primitives
    if "__shfl" not in content and "__syncwarp" not in content:
        suggestions.append({
            "priority": "low",
            "category": "Warp Primitives",
            "suggestion": "Consider warp shuffle for intra-warp communication",
            "details": "__shfl_down_sync is faster than shared memory for reductions"
        })

    if suggestions:
        for s in suggestions:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}[s["priority"]]
            output += f"### {emoji} {s['category']}\n"
            output += f"**{s['suggestion']}**\n"
            output += f"_{s['details']}_\n\n"
    else:
        output += "✅ Kernel looks well-optimized! Consider profiling with Nsight Compute for micro-optimizations.\n"

    return [TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
