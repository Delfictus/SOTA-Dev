#!/usr/bin/env python3
"""
Prism4D Agent Orchestrator MCP Server
Coordinates multiple specialist agents and manages parallel execution.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from enum import Enum

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

PRISM_ROOT = Path(__file__).parent.parent.parent

server = Server("prism-orchestrator")


class Domain(Enum):
    CUDA_GPU = "cuda-gpu"
    MD_PHYSICS = "md-physics"
    ML_NEURAL = "ml-neural"
    NEUROMORPHIC = "neuromorphic"
    BIOINFORMATICS = "bioinformatics"
    IO_SYSTEMS = "io-systems"
    VALIDATION = "validation"
    PYTHON_SCRIPTS = "python-scripts"
    DEVOPS = "devops"


@dataclass
class AgentSpec:
    name: str
    domain: Domain
    description: str
    key_files: list[str]
    capabilities: list[str]


# Agent registry
AGENTS = {
    "cuda": AgentSpec(
        name="CUDA Specialist",
        domain=Domain.CUDA_GPU,
        description="GPU kernel optimization, PTX, Tensor Cores, sm_120 Blackwell",
        key_files=["crates/prism-gpu/src/kernels/", "crates/prism-gpu/src/*.rs"],
        capabilities=["kernel_analysis", "memory_optimization", "tensor_core", "ptx_debugging"]
    ),
    "md": AgentSpec(
        name="Molecular Dynamics",
        domain=Domain.MD_PHYSICS,
        description="AMBER force fields, integrators, thermodynamics",
        key_files=["crates/prism-physics/", "crates/prism-phases/"],
        capabilities=["force_calculation", "integrator_design", "constraint_algorithms"]
    ),
    "ml": AgentSpec(
        name="ML/AI Pipeline",
        domain=Domain.ML_NEURAL,
        description="RL, GNNs, DendriticAgent, training pipelines",
        key_files=["crates/prism-learning/", "crates/prism-gnn/", "crates/prism-fluxnet/"],
        capabilities=["model_training", "architecture_design", "hyperparameter_tuning"]
    ),
    "neuro": AgentSpec(
        name="Neuromorphic Signal",
        domain=Domain.NEUROMORPHIC,
        description="NHS engine, UV-aromatic coupling, spike analysis",
        key_files=["crates/prism-nhs/"],
        capabilities=["spike_detection", "uv_coupling", "transfer_entropy"]
    ),
    "bio": AgentSpec(
        name="Bioinformatics",
        domain=Domain.BIOINFORMATICS,
        description="PDB handling, topology, glycans, structure validation",
        key_files=["crates/prism-amber-prep/", "crates/prism-lbs/", "scripts/stage*.py"],
        capabilities=["structure_prep", "topology_gen", "validation"]
    ),
    "io": AgentSpec(
        name="I/O Systems",
        domain=Domain.IO_SYSTEMS,
        description="rkyv, io_uring, async streaming, LanceDB",
        key_files=["crates/prism-io/", "crates/prism-pipeline/"],
        capabilities=["serialization", "async_io", "memory_mapping"]
    ),
    "bench": AgentSpec(
        name="Validation & Benchmarks",
        domain=Domain.VALIDATION,
        description="ATLAS, Apo-Holo, metrics, QA",
        key_files=["crates/prism-validation/"],
        capabilities=["benchmark_design", "metrics_analysis", "statistical_testing"]
    ),
    "python": AgentSpec(
        name="Python Pipeline",
        domain=Domain.PYTHON_SCRIPTS,
        description="Scripts, visualization, data processing",
        key_files=["scripts/"],
        capabilities=["visualization", "data_prep", "workflow_automation"]
    ),
    "devops": AgentSpec(
        name="DevOps/Systems",
        domain=Domain.DEVOPS,
        description="Cargo, builds, profiling, deployment",
        key_files=["Cargo.toml", "build.rs", "*.sh"],
        capabilities=["build_system", "profiling", "deployment"]
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_task",
            description="Analyze a task and recommend which agents to use, in what order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task to accomplish"
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="list_agents",
            description="List all available specialist agents and their capabilities.",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_agent_prompt",
            description="Get a detailed prompt for invoking a specific agent via the Task tool.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Agent identifier (cuda, md, ml, neuro, bio, io, bench, python, devops)"
                    },
                    "task": {
                        "type": "string",
                        "description": "Specific task for this agent"
                    }
                },
                "required": ["agent", "task"]
            }
        ),
        Tool(
            name="plan_parallel_execution",
            description="Create a parallel execution plan for a complex task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": "Complex task requiring multiple agents"
                    },
                    "max_parallel": {
                        "type": "integer",
                        "description": "Maximum agents to run in parallel (default: 3)",
                        "default": 3
                    }
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="get_domain_files",
            description="Get the key files for a specific domain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain name"
                    }
                },
                "required": ["domain"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "analyze_task":
        return await analyze_task(arguments["task_description"])
    elif name == "list_agents":
        return await list_agents()
    elif name == "get_agent_prompt":
        return await get_agent_prompt(arguments["agent"], arguments["task"])
    elif name == "plan_parallel_execution":
        return await plan_parallel_execution(
            arguments["task_description"],
            arguments.get("max_parallel", 3)
        )
    elif name == "get_domain_files":
        return await get_domain_files(arguments["domain"])
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def analyze_task(task_description: str) -> list[TextContent]:
    """Analyze task and recommend agents."""
    output = f"## Task Analysis\n\n**Task:** {task_description}\n\n"

    # Keyword-based domain detection
    keywords = {
        "cuda": ["cuda", "kernel", "gpu", "ptx", "tensor core", "wmma", "shared memory", "occupancy"],
        "md": ["amber", "force field", "integrator", "langevin", "settle", "constraint", "md", "dynamics", "physics"],
        "ml": ["train", "rl", "reinforcement", "gnn", "neural", "model", "learning", "agent", "dqn", "dendritic"],
        "neuro": ["nhs", "neuromorphic", "spike", "uv", "aromatic", "coupling", "transfer entropy"],
        "bio": ["pdb", "protein", "structure", "topology", "glycan", "residue", "chain", "atom"],
        "io": ["io", "serialization", "rkyv", "async", "stream", "memory map", "lance"],
        "bench": ["benchmark", "validation", "test", "metric", "hit@", "atlas", "accuracy"],
        "python": ["script", "python", "visualization", "plot", "figure", "pipeline"],
        "devops": ["build", "cargo", "deploy", "profile", "ci", "release"],
    }

    task_lower = task_description.lower()
    matched_agents = []

    for agent_id, kws in keywords.items():
        score = sum(1 for kw in kws if kw in task_lower)
        if score > 0:
            matched_agents.append((agent_id, score, AGENTS[agent_id]))

    # Sort by relevance
    matched_agents.sort(key=lambda x: -x[1])

    if not matched_agents:
        output += "⚠️ Could not automatically determine relevant agents.\n"
        output += "Consider using `list_agents` to see available specialists.\n"
    else:
        output += "### Recommended Agents (by relevance)\n\n"

        primary = matched_agents[0] if matched_agents else None
        secondary = matched_agents[1:3]

        if primary:
            output += f"**Primary:** `{primary[0]}` - {primary[2].name}\n"
            output += f"  _{primary[2].description}_\n\n"

        if secondary:
            output += "**Supporting:**\n"
            for agent_id, score, agent in secondary:
                output += f"- `{agent_id}` - {agent.name}\n"

        # Determine execution pattern
        output += "\n### Execution Pattern\n"

        if len(matched_agents) == 1:
            output += "**Single Agent** - Direct execution with primary agent.\n"
        elif len(matched_agents) <= 3:
            # Check for dependencies
            has_bench = any(a[0] == "bench" for a in matched_agents)
            has_impl = any(a[0] in ("cuda", "md", "ml") for a in matched_agents)

            if has_bench and has_impl:
                output += "**Sequential** - Implement first, then validate.\n"
                output += "1. Implementation agents work\n"
                output += "2. Benchmark agent validates\n"
            else:
                output += "**Parallel** - Agents can work independently.\n"
        else:
            output += "**Hybrid** - Some parallel, some sequential.\n"

    return [TextContent(type="text", text=output)]


async def list_agents() -> list[TextContent]:
    """List all agents."""
    output = "## Prism4D Specialist Agents\n\n"

    for agent_id, agent in AGENTS.items():
        output += f"### `{agent_id}` - {agent.name}\n"
        output += f"**Domain:** {agent.domain.value}\n"
        output += f"**Description:** {agent.description}\n"
        output += f"**Capabilities:** {', '.join(agent.capabilities)}\n"
        output += f"**Key files:** `{', '.join(agent.key_files)}`\n\n"

    return [TextContent(type="text", text=output)]


async def get_agent_prompt(agent_id: str, task: str) -> list[TextContent]:
    """Generate a detailed prompt for invoking an agent."""
    if agent_id not in AGENTS:
        return [TextContent(type="text", text=f"Unknown agent: {agent_id}. Use list_agents to see available agents.")]

    agent = AGENTS[agent_id]

    prompt = f"""You are acting as the **{agent.name}** specialist for Prism4D.

## Your Domain
{agent.description}

## Your Expertise
{', '.join(agent.capabilities)}

## Key Files to Focus On
{chr(10).join(f'- `{f}`' for f in agent.key_files)}

## Task
{task}

## Instructions
1. Focus ONLY on your domain - delegate other aspects to appropriate specialists
2. Read relevant files before making changes
3. Provide specific file paths and line numbers in your analysis
4. If you need information outside your domain, note it for another agent

## Output Format
Provide:
1. Analysis of the current state
2. Specific recommendations or implementations
3. Any cross-domain dependencies (for other agents)
"""

    output = f"## Agent Prompt: {agent.name}\n\n"
    output += "Use this prompt with the Task tool:\n\n"
    output += f"```\n{prompt}\n```\n"

    return [TextContent(type="text", text=output)]


async def plan_parallel_execution(task_description: str, max_parallel: int = 3) -> list[TextContent]:
    """Create a parallel execution plan."""
    output = f"## Parallel Execution Plan\n\n**Task:** {task_description}\n\n"

    # Analyze task first
    analysis = await analyze_task(task_description)

    # Extract matched agents from analysis
    # (In a real implementation, this would be more sophisticated)

    output += "### Execution Waves\n\n"

    output += "**Wave 1 (Parallel Investigation):**\n"
    output += "- Launch exploration agents to gather information\n"
    output += "- Each agent analyzes their domain independently\n\n"

    output += "**Wave 2 (Synthesis):**\n"
    output += "- Combine findings from Wave 1\n"
    output += "- Identify cross-domain dependencies\n\n"

    output += "**Wave 3 (Implementation):**\n"
    output += "- Execute changes in dependency order\n"
    output += "- Primary domain first, then supporting\n\n"

    output += "**Wave 4 (Validation):**\n"
    output += "- Run benchmarks and tests\n"
    output += "- Verify no regressions\n\n"

    output += "### Task Tool Pattern\n"
    output += "To run agents in parallel, invoke multiple Task tools in a SINGLE message:\n\n"
    output += "```\n"
    output += "Task 1: {agent_1 prompt}\n"
    output += "Task 2: {agent_2 prompt}\n"
    output += "Task 3: {agent_3 prompt}\n"
    output += "```\n"

    return [TextContent(type="text", text=output)]


async def get_domain_files(domain: str) -> list[TextContent]:
    """Get files for a domain."""
    output = f"## Files for Domain: {domain}\n\n"

    # Find matching agent
    matching = [(k, v) for k, v in AGENTS.items() if domain.lower() in k or domain.lower() in v.domain.value]

    if not matching:
        output += f"Unknown domain. Available: {', '.join(AGENTS.keys())}"
        return [TextContent(type="text", text=output)]

    agent_id, agent = matching[0]

    output += f"**Agent:** {agent.name}\n\n"

    for pattern in agent.key_files:
        output += f"### {pattern}\n"

        if pattern.endswith("/"):
            # Directory
            dir_path = PRISM_ROOT / pattern.rstrip("/")
            if dir_path.exists():
                files = list(dir_path.glob("*.rs")) + list(dir_path.glob("*.cu")) + list(dir_path.glob("*.py"))
                for f in sorted(files)[:20]:
                    rel = f.relative_to(PRISM_ROOT)
                    output += f"- `{rel}`\n"
                if len(files) > 20:
                    output += f"- ... and {len(files) - 20} more\n"
        else:
            # Glob pattern
            files = list(PRISM_ROOT.glob(pattern))
            for f in sorted(files)[:10]:
                rel = f.relative_to(PRISM_ROOT)
                output += f"- `{rel}`\n"

        output += "\n"

    return [TextContent(type="text", text=output)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
