# PRISM GLP1R Tier 3 PoV — Cloud-First Runbook

Last updated: 2026-05-29  
Status: staged only, not executed

## Best operational choice

Yes: forcing this campaign through a containerized cloud workflow is the right completeness test.

Reason:

- it exposes undeclared local dependencies
- it exposes path assumptions tied to the workstation
- it exposes missing runtime assets, CUDA/PTX gaps, and wrapper issues
- it forces explicit input/output manifests
- it makes multi-pod scale-out possible without changing scientific logic

Do not start with 10 pods blind. Use a staged scale-up:

1. single-pod image validation
2. single-pod end-to-end smoke
3. two-pod dispatch smoke
4. ten-pod Loop 2 production dispatch

## Phase map

| Phase | Name | Purpose | Primary output |
|-------|------|---------|----------------|
| 00 | Image contract + authority promotion | prove the system is runnable and inputs are honest | Loop 0 authority manifest |
| 01 | Multi-head generation | produce dynamic candidate slots | Loop 1 candidate refresh package |
| 02 | Crucible manifest + dispatch | build cloud-executable proof matrix | Loop 2 execution manifest |
| 03 | Crucible execution | run incumbent vs competitor vs refreshed candidates | Loop 2 MD + Path-B outputs |
| 04 | Falsification manifest + dispatch | build finalist translation/failure matrix | Loop 3 execution manifest |
| 05 | Falsification execution | run failure-mode and translation lanes | Loop 3 MD + Path-B outputs |
| 06 | Dossier assembly | convert physics into decision package | Tier 3 translational dossier |

## Runtime principles

- local machine is control plane only
- cloud pods are stateless execution lanes
- all loop outputs are manifest-driven
- every row gets a unique task ID
- every task ID maps to exact input files, exact pod assignment, exact output prefix, exact checksum receipts
- no worker success claim counts unless the output receipt exists and verifies

## Required tracking surfaces

Every staged run must include:

- `RUNBOOK_STATUS.json`
- `tracking/pid_registry.template.json`
- `tracking/runtime_events.schema.json`
- `tracking/worker_heartbeats.jsonl`
- `tracking/file_registry.jsonl`
- `tracking/checksum_policy.json`
- `tracking/artifact_receipts.jsonl`
- `tracking/filetag_manifest.seed.json`
- `verification/verification_gates.json`
- `cloud/container_contract.json`
- `cloud/r2_keyspace_plan.json`
- `cloud/pod_assignment_plan.csv`

## PID policy

Track PIDs at three levels:

1. control-plane PIDs
   - image build
   - manifest build
   - dispatch submission
2. pod-level worker PIDs
   - entrypoint process
   - PRISM wrapper process
   - post-processing process
3. task-level row tracking
   - task claimed
   - task started
   - task heartbeat
   - task finished
   - task verified

## File tracking policy

Track three file classes:

1. source-of-truth inputs
   - configs
   - signal grids
   - topologies
   - SDFs
   - candidate dossiers
   - motif registries
2. runtime outputs
   - PRISM output directories
   - Path-B outputs
   - chronology tensors
   - motif attribution outputs
3. control artifacts
   - manifests
   - pod logs
   - checksums
   - receipts
   - verification reports

Every tracked file needs:

- logical ID
- relative path
- phase
- role
- existence
- size
- digest
- source authority
- expected producer

## File tagging policy

Use `scripts/prism_filetag.py` on the portable execution bundle, not on the whole repo.

Tag at minimum:

- loop manifests
- target authority manifests
- dynamic slot decisions
- cloud worker entrypoints
- per-task receipts
- chronology outputs
- final dossier inputs

The tag set is the portability contract. If a required file is missing after bundle export, the runbook fails before dispatch.

## Recommended pod topology

### Phase 00

- `1` pod
- purpose: image validation only
- run one known-good smoke lane

### Phase 01

- `1-2` pods
- purpose: multi-head generation and ranking
- cheaper to keep serial or lightly parallel here because the heavy cost is still later

### Phase 02 / 03

- `10` pods recommended
- target: `Loop 2`
- expected shard policy: `48` rows per pod if full `480`-row matrix

### Phase 04 / 05

- `5` pods recommended by default
- can scale to `10`
- target: `Loop 3`

## Worker contract

Each worker must receive only:

- image reference
- shard manifest path
- pod ID
- run ID
- object storage prefix
- non-secret environment contract

Each worker must write back:

- claim receipt
- heartbeat records
- stdout/stderr log paths
- output artifact receipt
- checksum receipt
- terminal status

## Silent-drop prevention

The dispatcher must not infer success from exit code alone.

Every row requires:

- `CLAIMED`
- `STARTED`
- `HEARTBEAT`
- `OUTPUT_WRITTEN`
- `CHECKSUM_VERIFIED`
- `VERIFIED`

Missing any stage means the row remains incomplete and is automatically requeueable.

## Container completeness gate

Before any real campaign dispatch:

1. build the image from the repo commit
2. run the image on one pod with one audited topology and one audited ligand
3. prove:
   - wrapper executes
   - topology and ligand resolve from the manifest
   - outputs stream to object storage
   - receipts are written
   - no absolute workstation path is required inside the container

If this gate fails, do not scale out.

## Recommended first smoke lane

Use a lane that already has the strongest prior support:

- molecule: `cand_015_bccda098`
- target: `glp1r_6XOX_A316T`
- replicas: `1` first, then `10`

This is the best first cloud lane because it is closest to an already-proven anchor and will expose container/runtime gaps fastest.

## End-state success criteria

The cloud program is operational only when all are true:

- Loop 0 authority manifest passes
- container completeness gate passes
- Loop 1 dynamic slots are nominated with provenance
- Loop 2 dispatch manifest is complete and fully receipted
- Loop 3 falsification manifest marks expected-fail lanes explicitly
- chronology and motif outputs are bound back to exact row IDs
- final dossier references only verified artifacts

That is the level of control required if this is going to be sold as a pharmaceutical decision system rather than a one-off computation campaign.
