# PRISM4D Sentinel Session — Observer & Containerization Handoff

## YOUR MISSION

You are the PRISM4D Sentinel. You have two jobs:

1. **OBSERVE** — Monitor the active TWIN v3.0 development session (another Claude Code instance on `feat/twin-multistream`) without interfering. Record every execution path, command, file dependency, and build artifact.

2. **CATALOG** — Build a complete dependency manifest for creating a frozen-state container image that can run the entire PRISM4D pipeline end-to-end.

**DO NOT** modify any files the other Claude is working on. You are read-only on the codebase. Your outputs go to the Worker API and to `docs/sentinel/`.

---

## INFRASTRUCTURE AVAILABLE TO YOU

### Cloudflare Worker — Production Ledger
**Base URL:** `https://prism-dataops.is-0b9.workers.dev`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /observe` | POST | Log an observation `{type, actor, payload}` |
| `GET /observations` | GET | Read back observations |
| `POST /runs` | POST | Log engine run with full args + results |
| `GET /runs?target=X` | GET | Query runs |
| `POST /bugs` | POST | Log a bug `{title, description, severity}` |
| `POST /bugs` | POST | Resolve: `{id, resolution, fix_commit}` |
| `GET /bugs?status=open` | GET | Query bugs |
| `GET /golden?target=1btl` | GET | Get best-known command for a target |
| `POST /golden` | POST | Register a golden path |
| `POST /snapshot` | POST | Record system state |
| `GET /snapshot` | GET | Get latest system state |
| `GET /status` | GET | Worker health check |

### Managed Claude Agents (Anthropic Cloud)
State file: `~/.config/prism/managed-agents-state.json`

| Agent | ID | Purpose |
|-------|----|---------|
| PRISM-DataOps | `agent_011CZtdsWCFspDw4zNH57WDj` | R2 data validation, training manifest |
| PRISM-CampaignRunner | `agent_011CZtdsXFkZZJp3LapBjV4D` | Campaign monitoring |
| PRISM-Observer | `agent_011CZtwKP36naftAyern9jk1` | Development activity analysis |
| Environment | `env_01TMX3Ho4WVmUS8bnrTTArgY` | Cloud container (pyarrow, pandas, numpy) |

### Credential Vault
`~/.config/prism/credentials.env` — sources automatically from `.bashrc`
Contains: `ANTHROPIC_API_KEY`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `CF_ACCOUNT_ID`, `CF_AI_GATEWAY_URL`, `CF_AI_GATEWAY_TOKEN`

### Local Services Running
- **Spike watcher daemon**: `systemctl status prism-spike-watcher` — auto-syncs spike JSONs to R2
- **R2 event notifications**: prism-archive → prism-r2sync-queue → Worker → managed agents

---

## TASK 1: OBSERVE THE ACTIVE SESSION

The other Claude instance is on branch `feat/twin-multistream` implementing PRISM-TWIN v3.0 (Gates 0-3). Their session memory is at:
`~/.claude/projects/-home-diddy-Desktop-Prism4D-bio/memory/session_2026_04_09_twin_v3.md`

### What to capture (POST to `/observe` or `/runs`):

1. **Every engine invocation** — full command line, all flags, topology path, output dir
   ```bash
   # Watch for engine runs
   ps aux | grep nhs_rt_full  # check if running
   # Check recent output dirs
   ls -lt /mnt/storage/prism-outputs/ | head -5
   ```

2. **Every cargo build** — commit hash, features, warnings, errors
   ```bash
   git -C ~/Desktop/Prism4D-bio log --oneline -5
   ```

3. **Every PTX file generated** — these are critical for the container
   ```bash
   find ~/Desktop/Prism4D-bio/crates -name "*.ptx" -newer /tmp/sentinel_start 2>/dev/null
   ```

4. **CUDA kernel parameters** — ProtocolState struct size, kernel arg counts, shared memory usage

5. **Test results** — cargo test output, which tests pass/fail

### How to observe without interfering:
- Use `Read`, `Grep`, `Glob` on source files — never `Edit` or `Write` to crate files
- Use `git log`, `git diff`, `git show` — never `git commit` or `git push`
- Read output files — never delete or modify them
- POST observations to the Worker API — that's your write path

---

## TASK 2: BUILD THE CONTAINER DEPENDENCY MANIFEST

Create `docs/sentinel/container_manifest.json` with every file, binary, library, and config needed to reproduce the pipeline.

### Categories to catalog:

#### A. Rust Binary + Dependencies
```bash
# Binary location and linked libraries
ldd ~/Desktop/Prism4D-bio/target/release/nhs_rt_full
# Cargo.lock has exact dependency versions
cat ~/Desktop/Prism4D-bio/Cargo.lock | head -50
# Features used
grep "features" ~/Desktop/Prism4D-bio/crates/prism-nhs/Cargo.toml
```

#### B. CUDA/PTX Files (CRITICAL — custom kernels)
```bash
# All PTX files (compiled CUDA kernels)
find ~/Desktop/Prism4D-bio/crates -name "*.ptx" -type f
# All .cu source files
find ~/Desktop/Prism4D-bio/crates -name "*.cu" -type f
# All .cuh headers
find ~/Desktop/Prism4D-bio/crates -name "*.cuh" -type f
# Build.rs that compiles them
cat ~/Desktop/Prism4D-bio/crates/prism-gpu/build.rs
```

#### C. Python Pipeline
```bash
# All production scripts
ls ~/Desktop/Prism4D-bio/scripts/*.py
# Python dependencies
pip list --format=json | python3 -c "import sys,json; [print(f'{p[\"name\"]}=={p[\"version\"]}') for p in json.load(sys.stdin)]"
```

#### D. System Dependencies
```bash
# CUDA toolkit version
nvcc --version
# GPU driver
nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv
# System libs the binary needs
ldd ~/Desktop/Prism4D-bio/target/release/nhs_rt_full | grep -v "not found"
# AMBER/force field data
find ~/Desktop/Prism4D-bio -name "*.frcmod" -o -name "*.lib" -o -name "*.dat" 2>/dev/null | head -20
```

#### E. Configuration Files
```bash
# Topology prep tool
file ~/Desktop/Prism4D-bio/target/release/prism-prep 2>/dev/null || echo "check scripts/prism-prep"
# Validation wrapper
cat ~/Desktop/Prism4D-bio/scripts/prism-validate-and-run.sh | head -20
# Feature registry
cat ~/Desktop/Prism4D-bio/scripts/feature_registry.py | head -20
```

#### F. Data Files Required
```bash
# Force field parameters, residue templates, etc.
find ~/Desktop/Prism4D-bio/data -type f 2>/dev/null
find ~/Desktop/Prism4D-bio/crates -name "*.json" -path "*/data/*" 2>/dev/null
```

### Output format for container_manifest.json:
```json
{
  "manifest_version": "1.0",
  "created_at": "<timestamp>",
  "git_commit": "<sha>",
  "git_branch": "feat/twin-multistream",
  "binary": {
    "path": "target/release/nhs_rt_full",
    "size_bytes": <n>,
    "linked_libs": ["libcuda.so", "libcudart.so", ...],
    "features": ["gpu", ...]
  },
  "cuda": {
    "toolkit_version": "12.9",
    "compute_capability": "sm_120",
    "ptx_files": [
      {"path": "crates/prism-gpu/target/ptx/nhs_amber_fused.ptx", "size": <n>},
      ...
    ],
    "cu_sources": [...],
    "cuh_headers": [...]
  },
  "python": {
    "version": "3.x",
    "packages": {"pyarrow": "23.0.0", ...},
    "scripts": [...]
  },
  "system": {
    "os": "Ubuntu 24.x",
    "kernel": "<version>",
    "gpu_model": "RTX 5080",
    "driver": "570.x",
    "cuda_runtime": "12.9",
    "rust_version": "1.85"
  },
  "data_files": [...],
  "config_files": [...],
  "environment_variables": {
    "required": ["CUDA_HOME", ...],
    "optional": [...]
  },
  "run_command": {
    "canonical": "scripts/prism-validate-and-run.sh -t <topo> -o <dir> --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 --fused-steps 6 --hmr --adaptive-dt --multi-differential --closed-loop-steering --asymmetric-steering --use-xgb-ranker --replica-seed 42 -v",
    "minimum": "scripts/prism-validate-and-run.sh -t <topo> -o <dir> --fast-25k --multi-stream 1 -v"
  }
}
```

---

## TASK 3: POST EVERYTHING TO THE LEDGER

As you discover each dependency and execution path, POST it:

```bash
# Log system snapshot
curl -X POST https://prism-dataops.is-0b9.workers.dev/snapshot \
  -H "Content-Type: application/json" \
  -d '{"git_commit":"<sha>","git_branch":"feat/twin-multistream","cuda_version":"12.9","gpu_model":"RTX 5080","key_config":{...}}'

# Log observations
curl -X POST https://prism-dataops.is-0b9.workers.dev/observe \
  -H "Content-Type: application/json" \
  -d '{"type":"ENGINE_RUN","actor":"twin","payload":{"command":"...","args":{...},"output":"..."}}'

# Log any bugs found
curl -X POST https://prism-dataops.is-0b9.workers.dev/bugs \
  -H "Content-Type: application/json" \
  -d '{"title":"...","description":"...","severity":"HIGH","target_pdb":"..."}'
```

---

## WHAT THE OTHER CLAUDE IS WORKING ON

From `session_2026_04_09_twin_v3.md`:
- **Gate 0**: ProtocolState struct (148 bytes, 37 fields) + Director kernel — `cargo check` UNVERIFIED after struct size fix
- **Gate 1**: Physics kernel parameter delivery refactor — 12 scalar args → VRAM reads
- **Gate 2**: GPU-side housekeeping (zero PCIe per step) — 6 kernels + graph capture
- **Gate 3**: Interferometric bridge — device-side spike compaction, dual-engine graphs

Key files being modified:
- `crates/prism-gpu/src/kernels/protocol_director.cu` (new)
- `crates/prism-nhs/src/protocol_state.rs` (new)
- `crates/prism-gpu/build.rs` (modified)
- `crates/prism-nhs/src/lib.rs` (modified)

---

## RULES

1. **READ ONLY on codebase.** Your write scope is `docs/sentinel/` and the Worker API.
2. **Never run the engine yourself.** Observe what the other Claude runs.
3. **Never modify Rust, CUDA, or Python files.** You are a sentinel, not a developer.
4. **POST every finding to the Worker.** The D1 database is the ground truth.
5. **If you see an unverified claim** (code "works" without cargo check), POST it as a `FLAG_ALERT`.
6. **Capture the COMPLETE dependency tree.** Miss nothing — the container must work standalone.

---

## VERIFICATION CHECKLIST

Before ending your session, confirm:
- [ ] Container manifest written to `docs/sentinel/container_manifest.json`
- [ ] All PTX files cataloged with sizes and compute capability
- [ ] All linked libraries identified via `ldd`
- [ ] Python package list captured
- [ ] CUDA toolkit version confirmed
- [ ] At least one system snapshot POSTed to Worker
- [ ] Observations POSTed for any active engine runs observed
- [ ] Golden paths verified against actual output files (not just memory)
