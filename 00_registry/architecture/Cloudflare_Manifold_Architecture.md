# Cloudflare Manifold Architecture

## Scope

This document defines the production Persistent Manifold Store for concurrent Chem-BALD and GFlowNet loops. The design keeps epistemic classes separate, preserves cryptographic lineage for tensor artifacts, and prevents multiple optimization loops from racing on the same scaffold.

The local PRISM-4D GPU cluster remains the physics authority. Cloudflare stores, indexes, secures, and orchestrates the resulting artifacts; it does not fabricate receptor physics.

## Source Anchors

- Cloudflare Vectorize: vector indexes are bound directly to Workers, support metadata on vectors, and allow metadata indexes for filtered vector queries. Source: https://developers.cloudflare.com/vectorize/get-started/intro/
- Cloudflare D1: managed serverless SQLite with Worker and HTTP API access. Source: https://developers.cloudflare.com/d1/
- Cloudflare R2: S3-compatible object storage with no egress bandwidth fees. Source: https://developers.cloudflare.com/r2/how-r2-works/
- R2 Event Notifications: object-create and object-delete events can publish to Cloudflare Queues with prefix and suffix filtering. Source: https://developers.cloudflare.com/r2/buckets/event-notifications/
- Cloudflare Queues: Workers-integrated queues provide delivery, batching, retries, delays, dead-letter queues, and pull consumers. Source: https://developers.cloudflare.com/queues/
- Durable Objects: SQLite-backed Durable Object storage is strongly consistent and its storage methods are atomic and isolated. Source: https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/
- Workers KV: cacheTtl is intended for write-once or write-rarely data, not frequently changing coordination state. Source: https://developers.cloudflare.com/kv/api/read-key-value-pairs/
- Cloudflare Tunnel and Access: Tunnel connects origin services to Cloudflare using outbound-only encrypted connections, while Access policies protect self-hosted applications and Workers. Sources: https://developers.cloudflare.com/tunnel/ and https://developers.cloudflare.com/cloudflare-one/access-controls/applications/choose-application-type/

## Epistemic Storage Contract

Every object, row, and vector record must carry `epistemic_class`.

Allowed classes:

- `OBSERVED`: raw engine outputs, including spike streams, force grids, trajectory captures, and raw frame ledgers.
- `INFERRED`: receptor-side derived tensors, including coherence, shear stress, KCC, occupancy fatigue, and reward boundaries.
- `PROJECTED`: known ligand and scaffold interference projected onto receptor tensors.
- `HYPOTHESIZED`: generated poses, unsimulated fragments, GFlowNet trajectories, and RDKit-only conformers.
- `REPRESENTATIVE_CAPTURE`: de novo Cartesian reference frames used for spatial anchoring when deterministic reintegration is impossible.

The Worker rejects any upload whose key prefix, Arrow metadata, D1 row, and JSONL ledger disagree on epistemic class.

## Tripartite Store

### R2: Immutable Tensor And Ledger Store

R2 is the canonical blob layer for Parquet tensors, SDF/PDB/DCD/CIF frames, JSONL propagation ledgers, and model calibration artifacts. Object keys are content-addressed enough to make replay cheap:

```text
r2://prism-manifold/
  campaigns/{campaign_id}/observed/{sha256}/{artifact_name}
  campaigns/{campaign_id}/inferred/{sha256}/{artifact_name}
  campaigns/{campaign_id}/projected/{sha256}/{artifact_name}
  campaigns/{campaign_id}/hypothesized/{sha256}/{artifact_name}
  campaigns/{campaign_id}/ledgers/{artifact_name}.propagation.jsonl
```

Required R2 custom metadata:

```text
campaign_id
artifact_type
epistemic_class
schema_hash
content_sha256
parent_merkle_root
producer_git_sha
row_count
created_at
```

R2 is not a lock manager. It stores immutable artifacts and emits object events. Rewrites are only allowed for explicitly versioned staging keys.

### D1: Relational DAG And Metadata Store

D1 stores normalized state that must be queried with SQL:

- campaign registry and target conditions.
- tensor catalog: object key, content hash, row count, schema hash, epistemic class.
- DAG nodes, dependencies, run status, retry counters, and terminal failure states.
- Track 0 CSV-derived assay metadata after normalization.
- scaffold registry, fragment registry, candidate batches, and Chem-BALD acquisition results.
- Vectorize mutation state, including pending mutation IDs and last confirmed index version.

Core tables:

```sql
CREATE TABLE tensor_artifact (
  artifact_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL,
  r2_key TEXT NOT NULL UNIQUE,
  content_sha256 TEXT NOT NULL,
  schema_hash TEXT NOT NULL,
  epistemic_class TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE dag_run (
  run_id TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  input_merkle_root TEXT NOT NULL,
  output_artifact_id TEXT,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

CREATE TABLE scaffold_state (
  scaffold_id TEXT PRIMARY KEY,
  active_version INTEGER NOT NULL,
  best_reward REAL NOT NULL,
  best_artifact_id TEXT,
  updated_at TEXT NOT NULL
);
```

D1 is the queryable history and dashboard substrate. It is not the concurrency boundary for scaffold mutation.

### Vectorize: DKL Latent Neighbor Index

Vectorize stores Chem-BALD latent vectors for approximate nearest-neighbor retrieval. The vector ID must be stable:

```text
{campaign_id}:{scaffold_id}:{scaffold_version}:{candidate_id}
```

Vector metadata:

```json
{
  "campaign_id": "glp1r_aleniglipron",
  "scaffold_id": "ALENI-PARENT",
  "scaffold_version": 17,
  "epistemic_class": "HYPOTHESIZED",
  "artifact_id": "sha256:...",
  "condition_id": "glp1r_6XOX_WT",
  "reward": 0.98355067,
  "u_pose": 0.05,
  "sa_score": 1.0
}
```

Create metadata indexes for fields used to constrain Chem-BALD priors: `campaign_id`, `scaffold_id`, `condition_id`, `epistemic_class`, and `scaffold_version`.

Vectorize mutations are treated as asynchronous. The Worker records the mutation ID in D1 and does not mark a candidate batch as index-ready until a confirmation probe shows the mutation has been processed.

## Enterprise Primitives

### Access, Zero Trust, And Tunnel

The on-prem GPU cluster exposes no public inbound port. A `cloudflared` tunnel creates outbound-only connectivity from the cluster to Cloudflare. Worker ingestion endpoints are protected by Cloudflare Access with:

- service-token authentication for the Python orchestrator.
- mTLS for machine identity where available.
- device posture and IdP policies for human dashboard access.
- separate Access apps for ingestion, replay, and administrative APIs.

The Worker validates:

1. Access identity.
2. request body hash.
3. schema hash.
4. epistemic class.
5. R2 object checksum after upload.

### KV For Read-Mostly Constants

KV caches small, write-rarely values:

- `physical_constants.yml` hash and parsed beta values.
- schema hashes and schema version identifiers.
- current campaign feature flags.
- immutable assay vocabulary.

KV is not used for queue offsets, scaffold versions, locks, or candidate ranking state because those values need immediate consistency.

### R2 Event Notifications

R2 object-create notifications dispatch DAG progression. Example:

```bash
npx wrangler r2 bucket notification create prism-manifold \
  --event-type object-create \
  --queue chem-dag-events \
  --prefix "campaigns/glp1r_aleniglipron/projected/" \
  --suffix "Chem_Perturbed_DTSG.parquet"
```

The consumer Worker verifies the uploaded object hash, records the artifact in D1, then sends a task message to the stage queue. Notification rules are split by high-volume prefixes if event rate approaches queue throughput limits.

### Queues For Asynchronous Dispatch

Queues decouple each stage:

- `tensor-ingest-events`: R2 object events after upload.
- `chem-bald-score-requests`: candidate batches needing DKL/qEI scoring.
- `gpu-cluster-work`: tasks pulled by the on-prem orchestrator for PRISM-4D execution.
- `vectorize-upsert-events`: latent vectors ready for index insertion.
- `dag-dead-letter`: terminal failures after retry exhaustion.

Every queue message carries an idempotency key:

```text
sha256(campaign_id || stage || input_merkle_root || scaffold_id || candidate_batch_id)
```

Consumers must acknowledge only after the Durable Object transaction and D1 metadata update both succeed.

## Concurrency Model: Durable Objects Own Scaffold Mutation

The race condition trap is simultaneous Chem-BALD loops trying to update the same scaffold or candidate frontier. The solution is one Durable Object instance per `(campaign_id, scaffold_id)`.

Durable Object responsibilities:

- serialize all writes for that scaffold.
- reject stale writes using `expected_scaffold_version`.
- maintain an idempotency table keyed by message id.
- write a local outbox row in the same transaction as the scaffold update.
- forward committed outbox events to Queues.

Durable Object SQLite schema:

```sql
CREATE TABLE IF NOT EXISTS processed_message (
  idempotency_key TEXT PRIMARY KEY,
  processed_at TEXT NOT NULL,
  result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scaffold_version (
  scaffold_id TEXT PRIMARY KEY,
  version INTEGER NOT NULL,
  best_reward REAL NOT NULL,
  best_candidate_id TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outbox_event (
  event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  dispatched INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
```

Mutation procedure:

1. Queue consumer routes message to Durable Object ID `campaign_id:scaffold_id`.
2. Durable Object starts a SQLite transaction.
3. If `processed_message.idempotency_key` exists, return the stored result and ack.
4. Read current `scaffold_version`.
5. If `expected_scaffold_version` is stale, reject with `conflict_retryable=false`.
6. Apply the candidate frontier update.
7. Insert D1 mirror-update intent and Vectorize upsert intent into `outbox_event`.
8. Insert `processed_message`.
9. Commit.
10. Dispatch outbox messages to Queues; if dispatch fails, the Durable Object alarm retries the outbox drain.

This makes Queues safe under at-least-once delivery and makes D1 safe as a mirror rather than the lock holder.

## End-To-End Data Flow

### 1. Rust Engine To Python Orchestrator

The Rust PRISM-4D engine emits:

- Parquet tensors.
- `.propagation.jsonl` ledgers.
- optional PDB/DCD/CIF frames.
- stdout execution metrics.

The Python orchestrator computes:

- SHA-256 for each artifact.
- row counts and Arrow schema fingerprints.
- campaign Merkle root.
- epistemic class from the producing stage.
- dependency list from upstream ledgers.

### 2. Python Orchestrator To Zero Trust Tunnel

The orchestrator posts a signed manifest and streams large objects through the `cloudflared` tunnel to an Access-protected Worker endpoint:

```text
POST /v1/artifacts/initiate
PUT  /v1/artifacts/{upload_id}/chunk
POST /v1/artifacts/{upload_id}/commit
```

The Worker refuses upload if the Access service token, content hash, schema hash, or epistemic class is invalid.

### 3. Worker To Tripartite Store

The ingest Worker writes the artifact to R2, inserts the artifact catalog row in D1, and conditionally upserts a latent vector into Vectorize if the artifact includes DKL embeddings.

Artifact examples:

- `shear_stress_field.parquet`: R2 blob, D1 tensor row, no Vectorize row.
- `fragment_interference_attribution.parquet`: R2 blob, D1 tensor row, optional projected latent rows.
- `candidate_batch_latents.parquet`: R2 blob, D1 candidate rows, Vectorize upsert rows.

### 4. R2 Events To DAG Execution

R2 sends an `object-create` event when `Chem_Perturbed_DTSG.parquet` lands. The queue consumer:

1. validates object hash against D1.
2. verifies all required parent artifacts are present.
3. routes scaffold-specific mutation to the Durable Object.
4. enqueues the next DAG stage after the Durable Object commits.

### 5. Cloud To On-Prem GPU Cluster

The GPU orchestrator consumes `gpu-cluster-work` through a pull consumer or a Worker-mediated private endpoint. It receives only signed, content-addressed work descriptors. The descriptor names R2 input keys and expected SHA-256 hashes; the cluster downloads inputs, executes PRISM-4D, writes outputs locally, then uploads the next artifact bundle through the same Access-protected ingest path.

## Operational Rules

- Use R2 for large immutable artifacts and ledgers.
- Use D1 for queryable metadata, DAG state, and audit trails.
- Use Vectorize only for latent-neighbor retrieval, never as the source of truth.
- Use KV only for read-mostly constants and schema cache.
- Use Queues for asynchronous stage dispatch and R2 event handling.
- Use Durable Objects for per-scaffold serialization, idempotency, and conflict rejection.
- Use Access and Tunnel for all on-prem cluster ingress and egress API paths.

## Failure Handling

- Duplicate R2 event: ignored through the artifact `content_sha256` unique key and Durable Object idempotency table.
- Queue retry after Worker crash: safe because Durable Object checks idempotency before mutation.
- Vectorize delayed insert: D1 records `vectorize_status=pending`; downstream qEI queries are blocked until confirmed.
- D1 write failure after Durable Object commit: outbox retry handles mirror reconciliation.
- R2 object hash mismatch: object is quarantined under `rejected/` and never indexed.
- GPU task crash: queue retry runs up to policy limit, then sends the descriptor to `dag-dead-letter`.

## Minimal Production Deployment

1. Create R2 bucket `prism-manifold`.
2. Create D1 database `prism-manifold-meta`.
3. Create Vectorize index `prism-dkl-latents` with the DKL latent dimension.
4. Create Queues and a dead-letter queue.
5. Create Durable Object namespace `ScaffoldCoordinator`.
6. Deploy `manifold-ingest-worker` behind Cloudflare Access.
7. Install `cloudflared` on the on-prem PRISM-4D cluster.
8. Configure R2 object-create notifications for DAG trigger prefixes.
9. Run a replay test with a small Parquet and JSONL ledger.
10. Verify D1 artifact row, R2 object metadata, Vectorize mutation state, and Durable Object idempotency state.
