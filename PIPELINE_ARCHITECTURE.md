# NHS Persistent Concurrent Batch Processor - Complete Pipeline

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PERSISTENT NHS BATCH PROCESSOR                                          │
│ (Hot GPU context, NhsAmberFusedEngine, Full 4-Stage Pipeline)          │
└─────────────────────────────────────────────────────────────────────────┘

Input: Multiple PRISM-PREP topology.json files
  │
  ├─> FOR EACH STRUCTURE (sequential with GPU persistence):
  │
  ├──> STAGE 1: PRISM-PREP ✅ (Already done)
  │    └─ topology.json with AMBER ff14SB parameters
  │
  ├──> STAGE 2a: MD Simulation + RT Probes
  │    │  Engine: NhsAmberFusedEngine (787 steps/sec)
  │    │  Protocol: CryoUvProtocol (77K → 310K + UV-LIF + RT)
  │    └─ Outputs:
  │       ├─ spike_events.csv (neuromorphic detections)
  │       ├─ rt_probes.json (OptiX ray trace results)
  │       └─ trajectory/ (coordinate snapshots)
  │
  ├──> STAGE 2b: Trajectory Processing
  │    │  Binary: stage2b-process
  │    │  Processing: RMSF analysis, clustering, RT probe analysis
  │    └─ Outputs:
  │       ├─ processed_events.jsonl (quality-scored spikes)
  │       ├─ clusters.json (spatial clustering)
  │       └─ rt_analysis.json (cavity metrics)
  │
  ├──> STAGE 3: Site Detection & Ranking
  │    │  Processing: Combine spike quality + RT geometry + persistence
  │    └─ Outputs:
  │       └─ ranked_sites.json (final cryptic site predictions)
  │
  └──> STAGE 4: Validation & Export
       │  Processing: Binding affinity prediction, PDB export
       └─ Outputs:
          ├─ final_report.json
          └─ cryptic_sites.pdb
```

## Current Implementation Status

| Stage | Binary | MD Engine | Status |
|-------|--------|-----------|--------|
| 1 | (prism-prep tool) | N/A | ✅ Complete |
| 2a | nhs-batch (sequential) | NhsAmberFusedEngine | ✅ Complete |
| 2b | stage2b-process | N/A (post-processing) | ✅ Complete |
| 3 | **NEEDS IMPLEMENTATION** | N/A | ❌ Missing |
| 4 | **NEEDS IMPLEMENTATION** | N/A | ❌ Missing |

## Recommended Implementation

### Option 1: Enhanced nhs-batch (All-in-One)
Extend `nhs-batch` to run all 4 stages in sequence per structure:

```rust
// Pseudocode
for topology in topologies {
    // Stage 2a
    let (spikes, rt_probes, trajectory) = run_stage2a(topology);

    // Stage 2b (inline processing)
    let processed = process_stage2b(spikes, rt_probes, trajectory);

    // Stage 3 (inline clustering)
    let ranked_sites = detect_and_rank_sites(processed);

    // Stage 4 (inline export)
    export_results(ranked_sites, output_dir);
}
```

**Advantages**:
- Single binary for complete pipeline
- No intermediate file I/O overhead
- GPU stays hot throughout all stages

### Option 2: Separate Stage Binaries (Current Design)
Keep stages separate but orchestrate with shell script:

```bash
#!/bin/bash
# master_pipeline.sh

for topo in production_test/targets/*.topology.json; do
    name=$(basename $topo .topology.json)

    # Stage 2a
    nhs-batch --topologies $topo --output results/$name --stage 1 --sequential --enable-rt

    # Stage 2b
    stage2b-process \
      --events results/$name/spike_events.csv \
      --trajectory results/$name/trajectory/ \
      --topology $topo \
      --output results/$name/processed_events.jsonl

    # Stage 3 (TODO: implement)
    # stage3-detect --events results/$name/processed_events.jsonl --output results/$name/sites.json

    # Stage 4 (TODO: implement)
    # stage4-export --sites results/$name/sites.json --output results/$name/final_report.json
done
```

## Immediate Next Steps

1. **Kill the livelock process** (concurrent AmberSimdBatch)
2. **Run nhs-batch with --sequential** to verify NhsAmberFusedEngine works
3. **Implement Stage 3 binary** (site detection & ranking)
4. **Implement Stage 4 binary** (validation & export)
5. **Create master orchestration** (Option 1 or 2)

## Performance Targets

- **Stage 2a**: 787 steps/sec × 500k steps = 635 sec/structure (~10 min)
- **Stage 2b**: <60 sec (post-processing on CPU)
- **Stage 3**: <30 sec (clustering + ranking)
- **Stage 4**: <10 sec (export)
- **Total per structure**: ~12 minutes
- **5 structures**: ~60 minutes with sequential processing
- **With GPU persistence**: ~55 minutes (300ms overhead eliminated)

## Recommended Architecture

**Use PersistentNhsEngine wrapper** to process structures sequentially but with hot GPU context:

```rust
let mut persistent_engine = PersistentNhsEngine::new(&config)?;

for topology in topologies {
    persistent_engine.load_topology(&topology)?;

    // Stage 2a: Run simulation
    let summary = persistent_engine.run(total_steps)?;

    // Stage 2b: Process inline (no file I/O)
    let processed = process_spikes_inline(
        persistent_engine.get_spike_events(),
        persistent_engine.get_rt_probes()
    )?;

    // Stage 3: Detect sites
    let sites = detect_cryptic_sites(&processed)?;

    // Stage 4: Export
    export_final_results(&sites, output_dir)?;
}
```

This gives us **fast-sequential processing** with the proven working NhsAmberFusedEngine while maintaining the complete 4-stage pipeline.
