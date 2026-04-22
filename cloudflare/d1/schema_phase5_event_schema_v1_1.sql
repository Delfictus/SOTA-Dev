-- Phase 5 migration: event_schema_v1 → event_schema_v1.1
-- Widens the `type` enum with the 3 aromatic-residue classes
-- (PHE, TYR, TRP) that are emitted by the live engine at material
-- frequency (≥2.9% each; 18.6% combined) and therefore qualify as
-- real contract values per the v1.0 YAML's own open-enum widening rule.
--
-- Prerequisites:
--   - W1 isolate commit            af991b7d
--   - W4 isolate commit            1f2690a2
--   - stale-row cleanup            closed
--   - W2 /dcc rollout              closed (corrected_dcc = 372/372)
--   - event_schema_v1.yaml         schema_version bumped to "1.1.0"
--
-- Applies 6 ALTER TABLE ADD COLUMN statements. No new tables, no new
-- indexes. D1 column caps NOT at risk (both tables well below 100).
--
-- Idempotent on retry via apply_phase4_migration.py (tolerates
-- "duplicate column name" and D1's "too many columns on
-- sqlite_altertab_…" when columns already exist).

ALTER TABLE site_event_aggregates       ADD COLUMN count_type_phe INTEGER;
ALTER TABLE site_event_aggregates       ADD COLUMN count_type_tyr INTEGER;
ALTER TABLE site_event_aggregates       ADD COLUMN count_type_trp INTEGER;

ALTER TABLE quarantined_event_aggregates ADD COLUMN count_type_phe INTEGER;
ALTER TABLE quarantined_event_aggregates ADD COLUMN count_type_tyr INTEGER;
ALTER TABLE quarantined_event_aggregates ADD COLUMN count_type_trp INTEGER;
