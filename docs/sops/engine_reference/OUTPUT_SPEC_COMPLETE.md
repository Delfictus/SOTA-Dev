---
name: Engine Output Spec (Complete)
description: Complete output specification — Sections 0-7, per-site 5.1-5.13, per-protein global, competitive scorecard
type: engine_reference
category: engine_reference
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 0.1-STUB
status: CONTENT_PENDING
---

# Engine Output Spec (Complete)

## ⚠️ STATUS: CONTENT PENDING

This SOP is a **placeholder**. The canonical output specification
(Sections 0-7, per-site subsections 5.1-5.13, per-protein global,
competitive scorecard, every field listed) has not yet been pasted
into the repository.

**Do not treat this file as authoritative until the `status:` field
in frontmatter reads `FROZEN` and a version >= 1.0 is assigned.**

## What this SOP WILL contain (once populated)

- **Section 0** — file naming + versioning scheme for engine output
- **Section 1** — top-level JSON envelope (metadata, engine version, flags, seed, timing)
- **Section 2** — per-protein global features (network statistics, coupling maps, NMA mode summary)
- **Section 3** — site list structure + lexicographic ranking keys
- **Section 4** — voxel grid reference frame + coordinate conventions
- **Section 5** — per-site object schema, subsections 5.1–5.13:
  - 5.1  centroid, volume, sphericity
  - 5.2  lining residues, burial, druggability
  - 5.3  persistence / pass_fraction / stability / quality (ranking keys)
  - 5.4  therm_class + hysteresis_asymmetry (PRISM-THERM)
  - 5.5  spike statistics (count, intensity, phase distribution)
  - 5.6  consensus features (12 per-residue, aggregated)
  - 5.7  cross-correlation features (12 per-residue, aggregated)
  - 5.8  differential features (18 per-residue, aggregated)
  - 5.9  scout/propagation features (8 per-residue, aggregated)
  - 5.10 anchor points + growth vectors
  - 5.11 pocket chemistry profile
  - 5.12 allosteric fingerprint (CCF summary stats, Layer 5)
  - 5.13 barrier classification + NMA mode annotation (Layer 4)
- **Section 6** — per-residue feature arrays (~48 features × N residues)
- **Section 7** — competitive scorecard (P2Rank, fpocket, PocketMiner, DCC metrics)

## How to populate this file

The user (Ididia Serfaty) will paste the full spec content. When that
happens:

1. Replace the entirety of this file with the pasted content, preserving the
   YAML frontmatter but updating `status: FROZEN`, `version: 1.0`,
   and `last_verified` to today's date.
2. Add an entry to the **History** table below.
3. Update `docs/sops/SOP_INDEX.md` (when created) with the new version.
4. Update any cross-references from `PRISM_TWIN_ARCHITECTURE.md` Layer 7.

## Related SOPs

- `PRISM_TWIN_ARCHITECTURE.md` — Layer 7 references this file
- `SPIKE_RECORD_FORMAT.md` — 48-byte spike record (upstream of per-site aggregation)
- `CLI_REFERENCE.md` — engine flags that affect output content

## History

| Date       | Change                              | By              |
|------------|-------------------------------------|-----------------|
| 2026-04-05 | Stub created — content pending.     | Ididia Serfaty  |
