# IMMUTABLE_CONSTRAINTS.md

Scope: crates/prism-nhs production engine and its execution path.
Last updated: 2026-04-21
Authority: Sole inventor + maintainer Ididia Serfaty. Supersedes all prior versions.

## Purpose

This file constrains what Claude Code (or any agent) is allowed to claim about the PRISM-4D production codebase. It exists because fabricated claims about CUDA wiring, kernel activation, or pipeline completeness have historically caused real debugging time loss. Every constraint here traces to a specific failure mode.

## Scope boundaries

This document applies to:
- crates/prism-nhs/** (production neuromorphic + simulation engine)
- scripts/prism-validate-and-run.sh (canonical invocation)
- scripts/prism-prep (topology preparation)
- target/release/nhs_rt_full (production binary)

Does NOT apply to:
- crates/prism-validation (correctly creates own CudaContext/AmberSimdBatch per binary, by design)
- Research/experimental crates outside prism-nhs
- Test code under #[cfg(test)] including rescue_controller.rs:1553 thread::spawn (unit test context, not a violation)

## Evidence discipline (4-proof requirement)

Before claiming any of the following, provide all four proofs:

For "function exists":
1. File read showing the definition (view or cat with line numbers)
2. grep -n or rg -n locating the exact line
3. Type signature and return type visible in the read
4. Cargo workspace member confirmation (Cargo.toml or cargo metadata)

For "code path is active in production":
1. File read of the callsite
2. grep -n confirming callsite is NOT under #[cfg(test)] / #[cfg(feature = "...")] / #[cfg(not(...))]
3. Runtime log line from actual nhs_rt_full invocation showing the path executed
4. RUST_LOG=debug output or explicit printf proving call order

For "CUDA kernel is wired":
1. File read of the .cu showing the kernel definition
2. Rust FFI declaration in the corresponding .rs file
3. grep showing the Rust side actually calls the FFI function in a non-test path
4. nvprof, nsys, or RUST_LOG showing the kernel launched during actual runtime

For "campaign/target completed":
1. Output file exists on disk (ls with size)
2. File size is non-zero and matches expected schema (JSON parses, parquet has rows, etc.)
3. Content check: first N lines / parquet schema / JSON keys match expected
4. No error log entries for that target in the campaign log

If any proof is missing, the claim is: "I don't know" or "unverified."

## Banned claim patterns

These phrases or their equivalents indicate drift and must not appear:

- "This should work"
- "I believe the X kernel is active" (without the 4-proof above)
- "The pipeline is complete end-to-end" (without log evidence from real run)
- "This optimization provides Nx speedup" (without measured benchmark)
- "The bug is fixed" (without reproduction test passing)
- "All 372 targets succeeded" (without per-target verification via grep/count)
- "L4 CUDA is integrated" (it compiles but is unwired — see below)
- "TWIN runs on GPU" (production TWIN runs CPU coupling — see below)
- "Single-pass dual-pass handled correctly" (the dual-pass bug was resolved but the claim needs per-invocation proof)

When a result must be presented without complete evidence, precede it with "UNVERIFIED:" as an explicit prefix.

## Current architecture reality (2026-04-21 ground truth)

### Production path: CPU coupling
The nhs_rt_full binary runs CPU coupling for the TWIN architecture in production. This is the expected state, not a bug.

### L4 CUDA kernels: compiled but unwired
- twin_persistent.cu
- twin_persistent_physics.cu
- ring_buffer.cu
- tensor_ccf.cu

These compile successfully to sm_120 (Blackwell RTX 5080) and are present as compiled artifacts. They are NOT wired to the production dispatch path. Claude Code must not claim any of these kernels execute in a production run unless the 4-proof for "CUDA kernel is wired" has been satisfied with actual runtime evidence.

If a planned refactor wires one of these kernels into production, the wiring commit must include:
1. The FFI declaration in the corresponding Rust file
2. The non-test callsite
3. A feature flag or gate clearly documenting it
4. A run log showing the kernel launched

Until then, L4 kernels are considered dormant.

### Canonical run command
RUST_LOG=info ./target/release/nhs_rt_full <topology> <output>
--multi-stream 20 --multi-scale --rt-clustering
--lining-cutoff 8.0 --fast -v

￼

Invoked via scripts/prism-validate-and-run.sh wrapper. The binary refuses direct invocation as a deliberate gate.

### Prep command
scripts/prism-prep input.pdb output.topology.json

￼

Only prep tool. Requires OpenMM for --use-amber.

### Flag audit (from memory, verify before relying)
12 of 14 campaign flags are no-ops in twin mode. --ladd specifically degrades detection. Before claiming a flag does X, grep -n "args.<flag>" in nhs_rt_full.rs and confirm the code path.

## Realistic log-line expectations

In a clean production run you should see:
- NHS engine initialization log
- Topology parse confirmation with residue count
- Per-target CCNS protocol phase transitions (5 phases)
- Multi-stream scheduling logs with stream IDs
- Per-target spike count summary
- Output file writes with path + byte count

You should NOT see (unless the 4-proof is satisfied):
- L4 CUDA kernel launch logs (unwired)
- Tensor core WMMA cross-correlation logs (L4, unwired)
- Ring buffer exchange logs (L4, unwired)
- Differential NMA perturbation CUDA kernel logs (unwired)

If a log line claims any of the above, either:
1. The 4-proof is satisfied and Claude Code can reference it
2. The log line is from an experimental branch not in production
3. The claim is false

## Non-negotiable rules

1. No claim about code behavior without corresponding file read or runtime evidence.
2. No "UNVERIFIED:" prefix omitted when evidence is incomplete.
3. No fabrication of log output, stats, numbers, file sizes, or benchmark results.
4. No assumption that compiled code is wired into a runtime path.
5. No speculation about what nhs_rt_full does that is phrased as statement of fact.
6. No repeating historical claims (e.g., SR@1=53.1% from February 2026 build) without re-verifying against current build.
7. No test-code paths presented as production evidence.
8. "I don't know" is always a valid answer.

## When constraints conflict with operator request

If the operator asks for output that would require violating a constraint here (e.g., "just tell me the TWIN pipeline is complete"), the response is:
1. State that the 4-proof is not satisfied
2. Show what would satisfy it (specific grep commands, log searches, file reads)
3. Ask whether to proceed with verification or whether operator wants the claim under explicit "UNVERIFIED:" prefix

The operator may override with "UNVERIFIED:" acknowledgment. They may not override into false statements.

## Constraint updates

This file may be updated when architecture changes. Update procedure:
1. Verify the change is real via 4-proof on the thing being enabled/disabled
2. Update the relevant section
3. Update "Last updated" stamp
4. Preserve prior version as IMMUTABLE_CONSTRAINTS.legacy-<date>.md

Never delete constraints without replacement. Never weaken evidence requirements.
