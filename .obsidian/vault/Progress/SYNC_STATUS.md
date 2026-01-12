# Vault Synchronization Status

> This file provides quick verification that Claude's state matches the vault.
> Updated automatically at each session boundary.

---

## Quick Sync Check

| Component | Vault State | Expected Claude State |
|-----------|-------------|----------------------|
| **Active Todo** | `tests/gpu_scorer_tests.rs` (Unit 1.3) | TodoWrite: "Unit 1.3: tests/gpu_scorer_tests.rs" pending |
| **Current Phase** | 6 | CLAUDE.md active phase |
| **Current Week** | 1-2 | Plan section being implemented |
| **Next Action** | Create GPU scorer tests | Claude's stated next task |
| **Total Files** | 20 | 2 completed, 18 pending |

---

## Claude TodoWrite ↔ Vault Sync

Claude's TodoWrite MUST match these exact items:

```
[x] Unit 1.1: cryptic_features.rs - 16-dim feature vector (completed)
[x] Unit 1.2: gpu_zro_cryptic_scorer.rs - 512-neuron reservoir (completed)
[ ] Unit 1.3: tests/gpu_scorer_tests.rs - Zero fallback tests (pending) <- ACTIVE
```

---

## Sync Verification Commands

Run these to verify alignment:

```bash
# 1. Check vault JSON is valid
jq '.' .obsidian/vault/Progress/implementation_status.json > /dev/null && echo "JSON valid"

# 2. Get active todo from vault
jq -r '.active_todo.file' .obsidian/vault/Progress/implementation_status.json

# 3. Get next action
jq -r '.next_action.file' .obsidian/vault/Progress/implementation_status.json

# 4. Check sync state
jq '.sync_state' .obsidian/vault/Progress/implementation_status.json

# 5. Get file counts
jq '.file_counts' .obsidian/vault/Progress/implementation_status.json

# 6. Verify all links exist
ls ".obsidian/vault/Plans/Phase 6 Plan.md" && echo "Phase 6 Plan OK"
ls ".obsidian/vault/Plans/Full/PRISM_PHASE6_PLAN_PART1.md" && echo "Full Plan Part 1 OK"
```

---

## Last Sync Details

| Field | Value |
|-------|-------|
| **Timestamp** | 2026-01-12T03:30:00Z |
| **Session ID** | claude_final_alignment_session |
| **Todos Match** | Yes |
| **Plan Aligned** | Yes |
| **Links Verified** | Yes |
| **File List Verified** | Yes |

---

## Files Status Summary (20 total)

### Completed (2)
- [x] `cryptic_features.rs` - Unit 1.1 - commit c4d88c2
- [x] `gpu_zro_cryptic_scorer.rs` - Unit 1.2 - commit 5e55a7a

### Next Up (1)
- [ ] `tests/gpu_scorer_tests.rs` - Unit 1.3 - **ACTIVE TARGET**

### Pending by Week
| Week | Files | Status |
|------|-------|--------|
| Week 3 | 8 files | sampling/mod.rs, contract.rs, result.rs, paths/*, router/ |
| Week 4 | 5 files | shadow/*, migration/*, apo_holo_benchmark.rs |
| Week 5-6 | 3 files | cryptobench_dataset.rs, ablation.rs, failure_analysis.rs |
| Week 7-8 | 2 files | publication_outputs.rs, generate_figures.py |

---

## Checkpoint Gates

| Checkpoint | Gate Command | Status |
|------------|--------------|--------|
| Week 2 | `cargo test --release -p prism-validation --features cuda gpu_scorer` | Pending |
| Week 4 | `cargo run --bin apo-holo-single -- --apo 1AKE --holo 4AKE` | Pending |
| Week 6 | `cargo run --bin cryptobench -- --manifest manifest.json` | Pending |
| Week 8 | All metrics pass, publication ready | Pending |

---

## How to Use This File

### For Claude (at session start)
```
1. Read implementation_status.json
2. Set TodoWrite to match files.week_X items
3. Verify active_todo matches this file's "Active Todo"
4. If mismatch: STOP and resync
5. If match: Proceed with implementation
```

### For User (to verify Claude)
```
1. Ask Claude: "What is your current active todo?"
2. Compare answer to this file
3. Ask Claude: "Show me your TodoWrite items"
4. Verify they match the sync check above
5. If mismatch: Use recovery command
```

### Recovery Command
If Claude is out of sync, use this prompt:
```
SYNC RECOVERY: Read .obsidian/vault/Progress/implementation_status.json
and reset your internal state to match.

Set your TodoWrite to:
[x] Unit 1.1: cryptic_features.rs (completed)
[x] Unit 1.2: gpu_zro_cryptic_scorer.rs (completed)
[ ] Unit 1.3: tests/gpu_scorer_tests.rs (pending)

Report the active_todo and next_action fields.
```

---

## Context Preservation

### For New Sessions
Use this continuation prompt:
```
Continue Phase 6 implementation.

SYNC CHECK:
1. Read .obsidian/vault/Progress/implementation_status.json
2. Read .obsidian/vault/Sessions/Current Session.md
3. Set TodoWrite to match vault files status
4. Verify active_todo matches your target

Current target: tests/gpu_scorer_tests.rs (Unit 1.3)
Plan reference: docs/plans/PRISM_PHASE6_PLAN_PART1.md, Section 4.3

Confirm alignment and proceed.
```

### For Existing Sessions
At each file completion:
1. Mark todo as completed in TodoWrite
2. Update implementation_status.json file status to "completed"
3. Add commit hash to file entry
4. Update next_action to next pending file
5. Update Current Session.md completed list

---

## Links to Key Files

- [[Progress/implementation_status.json|Machine State (JSON)]]
- [[Sessions/Current Session|Current Session Context]]
- [[PRISM Dashboard|Dashboard Overview]]
- [[Plans/Phase 6 Plan|Phase 6 Plan]]
- [[Plans/Full/PRISM_PHASE6_PLAN_PART1|Full Plan Part 1]]

---

*This file is the quick-reference for vault synchronization.*
