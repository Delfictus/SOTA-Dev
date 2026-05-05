//! Control-plane trace schema for audit records emitted outside the
//! runtime hot path.
//!
//! This module is deliberately data-only. It defines the typed wire
//! records and a small NDJSON writer helper; it does not read device
//! state, alter graph topology, or install runtime hooks.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::io::{self, Write};

/// Chronometric gearbox gear identifier.
pub type GearId = u8;

/// Current control trace schema version emitted by this module.
pub const CONTROL_TRACE_SCHEMA_VERSION: u32 = 1;

/// Required per-record control trace envelope.
///
/// These fields identify the trace schema, the logical run, the captured graph
/// launch/control epoch, and the per-run record order without requiring any
/// runtime wiring in this data-only module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlTraceEnvelope {
    /// Version of the serialized control trace schema.
    pub schema_version: u32,
    /// Stable identifier for the simulation or analysis run.
    pub run_id: String,
    /// Stable identifier for the graph launch/control epoch being audited.
    pub graph_launch_id: String,
    /// Monotonic record index within `run_id`.
    pub record_idx: u64,
}

impl ControlTraceEnvelope {
    /// Construct an envelope using the current schema version.
    pub fn current(
        run_id: impl Into<String>,
        graph_launch_id: impl Into<String>,
        record_idx: u64,
    ) -> Self {
        Self {
            schema_version: CONTROL_TRACE_SCHEMA_VERSION,
            run_id: run_id.into(),
            graph_launch_id: graph_launch_id.into(),
            record_idx,
        }
    }
}

/// Raw energy/control state cited by control-plane decisions.
///
/// The matching hash is stored beside this struct on each record that cites
/// energy state. Keeping both preserves replay diagnostics while allowing
/// downstream tamper checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceEnergyState {
    /// Potential energy scalar observed by the decision source.
    pub potential_energy: f64,
    /// External work scalar observed by the decision source.
    pub external_work: f64,
    /// Stable label naming the source reducer, ring, or control surface.
    pub source: String,
}

/// Stable route names for the F1 adjudicator SWITCH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F1Route {
    /// Noise cluster discarded before construction.
    Prune,
    /// Candidate cluster admitted to the construction branch.
    Construct,
    /// Invariant violation branch.
    Violation,
}

/// Host-visible WHILE loop exit state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WhileExitReason {
    /// Loop has not exited at the point this record was emitted.
    NotExited,
    /// Loop exited because the evaluated condition became false.
    ConditionFalse,
    /// Loop exited because the configured iteration bound was reached.
    MaxIterations,
    /// Loop exited because the watchdog tripped.
    WatchdogExpired,
    /// Loop exited due to an external host-side stop/cancel request.
    ExternalStop,
}

/// Top-level control-plane audit event.
///
/// The enum uses an externally visible `kind` tag so NDJSON consumers can
/// branch on record type without probing nullable fields. Every variant
/// carries only fields required for that control-plane event class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum ControlPlaneRecord {
    /// G26 chronometric gearbox state transition or steady-state observation.
    G26 {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic host-side frame index associated with the observation.
        frame_idx: u64,
        /// Runtime stream identifier that owns the observed control decision.
        stream_id: u32,
        /// Gear before the decision was applied.
        previous_gear: GearId,
        /// Gear selected for the next integration step.
        current_gear: GearId,
        /// Effective timestep in picoseconds after the gearbox decision.
        dt_ps: f64,
        /// Gear selection predicate value forwarded to the G26 switch.
        switch_selector: u32,
        /// Raw source energy state used or cited by the G26 decision.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// F1 adjudicator SWITCH routing decision.
    F1 {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic host-side frame index associated with the decision.
        frame_idx: u64,
        /// Runtime stream identifier that owns the adjudicator instance.
        stream_id: u32,
        /// Cluster identifier consumed by the F1 adjudicator.
        cluster_id: u32,
        /// Raw adjudication code written by the upstream classifier.
        adjudication_code: u32,
        /// Semantic branch selected by the F1 SWITCH.
        route: F1Route,
        /// Bitmask of violated invariants observed by the adjudicator.
        violation_bits: u32,
        /// Raw source energy state used or cited by the F1 decision.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// Host-visible WHILE control record for bounded loop progress.
    While {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Stable loop name chosen by the caller.
        loop_name: String,
        /// Monotonic iteration index inside the named loop.
        iteration: u64,
        /// Maximum iteration count allowed for this loop instance.
        max_iterations: u64,
        /// Boolean condition value evaluated for this iteration.
        condition_value: bool,
        /// Caller-supplied action label taken after condition evaluation.
        action: String,
        /// Exit state observed after this condition/action checkpoint.
        exit_reason: WhileExitReason,
        /// Maximum host watchdog duration allowed for this loop instance.
        watchdog_limit_ms: u64,
        /// Host watchdog duration consumed by this loop instance.
        watchdog_elapsed_ms: u64,
        /// True when the watchdog forced or will force loop termination.
        watchdog_tripped: bool,
        /// Raw source energy state used or cited by the WHILE predicate or exit.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// Adaptive sensing controller decision record.
    Asc {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic ASC chunk index.
        chunk_idx: u64,
        /// Runtime stream identifier for the ASC observation source.
        stream_id: u32,
        /// Residues selected as ASC steering focus targets.
        focus_residues: Vec<u32>,
        /// Steering gain applied to the focus set.
        alpha_gain: f64,
        /// Number of independent observer groups contributing evidence.
        observer_group_count: u32,
        /// Scalar surprise score used by the controller for this decision.
        surprise_score: f64,
        /// Raw source energy state used or cited by the ASC decision.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },

    /// Manual or supervisory gearbox override.
    GearOverride {
        /// Required per-record trace envelope.
        envelope: ControlTraceEnvelope,
        /// Monotonic host-side frame index associated with the override.
        frame_idx: u64,
        /// Runtime stream identifier affected by the override.
        stream_id: u32,
        /// Gear active before the override request.
        previous_gear: GearId,
        /// Gear requested by the override.
        requested_gear: GearId,
        /// Stable source label, such as a CLI, test harness, or operator.
        source: String,
        /// Human-readable reason supplied by the override source.
        reason: String,
        /// Gear overrides are host mutations by definition and must serialize as `true`.
        #[serde(
            serialize_with = "serialize_true_bool",
            deserialize_with = "deserialize_true_bool"
        )]
        host_mutation: bool,
        /// Raw source energy state cited by the current gear/dt context.
        source_energy_state: SourceEnergyState,
        /// Hash of `source_energy_state` for tamper detection.
        source_energy_state_hash: String,
    },
}

fn serialize_true_bool<S>(value: &bool, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if *value {
        serializer.serialize_bool(true)
    } else {
        Err(serde::ser::Error::custom(
            "GearOverride.host_mutation must be true",
        ))
    }
}

fn deserialize_true_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = bool::deserialize(deserializer)?;
    if value {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(
            "GearOverride.host_mutation must be true",
        ))
    }
}

/// Write control-plane records as newline-delimited JSON.
///
/// The helper accepts any writer so callers can use a buffered file,
/// memory buffer, or test sink without this module owning filesystem policy.
pub fn write_control_plane_ndjson<W, I>(writer: &mut W, records: I) -> io::Result<()>
where
    W: Write,
    I: IntoIterator<Item = ControlPlaneRecord>,
{
    for record in records {
        serde_json::to_writer(&mut *writer, &record).map_err(io::Error::other)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(record_idx: u64) -> ControlTraceEnvelope {
        ControlTraceEnvelope::current("run-001", "graph-launch-001", record_idx)
    }

    fn source_energy_state() -> SourceEnergyState {
        SourceEnergyState {
            potential_energy: -12345.625,
            external_work: 0.03125,
            source: "adj_pe_wext_reducer".to_string(),
        }
    }

    fn source_energy_state_hash() -> String {
        "sha256:source-energy-state-v1".to_string()
    }

    fn source_energy_state_json() -> serde_json::Value {
        serde_json::json!({
            "potential_energy": -12345.625,
            "external_work": 0.03125,
            "source": "adj_pe_wext_reducer"
        })
    }

    fn record_energy_fields(record: &ControlPlaneRecord) -> (&SourceEnergyState, &str) {
        match record {
            ControlPlaneRecord::G26 {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::F1 {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::While {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::Asc {
                source_energy_state,
                source_energy_state_hash,
                ..
            }
            | ControlPlaneRecord::GearOverride {
                source_energy_state,
                source_energy_state_hash,
                ..
            } => (source_energy_state, source_energy_state_hash.as_str()),
        }
    }

    fn sample_records() -> Vec<ControlPlaneRecord> {
        vec![
            ControlPlaneRecord::G26 {
                envelope: envelope(0),
                frame_idx: 42,
                stream_id: 2,
                previous_gear: 1,
                current_gear: 2,
                dt_ps: 0.001,
                switch_selector: 2,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::F1 {
                envelope: envelope(1),
                frame_idx: 43,
                stream_id: 2,
                cluster_id: 7,
                adjudication_code: 1,
                route: F1Route::Construct,
                violation_bits: 0,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::While {
                envelope: envelope(2),
                loop_name: "chunk_window".to_string(),
                iteration: 3,
                max_iterations: 16,
                condition_value: true,
                action: "continue".to_string(),
                exit_reason: WhileExitReason::NotExited,
                watchdog_limit_ms: 250,
                watchdog_elapsed_ms: 37,
                watchdog_tripped: false,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::Asc {
                envelope: envelope(3),
                chunk_idx: 9,
                stream_id: 0,
                focus_residues: vec![12, 47],
                alpha_gain: 0.01,
                observer_group_count: 4,
                surprise_score: 2.5,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
            ControlPlaneRecord::GearOverride {
                envelope: envelope(4),
                frame_idx: 44,
                stream_id: 2,
                previous_gear: 2,
                requested_gear: 0,
                source: "test_harness".to_string(),
                reason: "deterministic coverage".to_string(),
                host_mutation: true,
                source_energy_state: source_energy_state(),
                source_energy_state_hash: source_energy_state_hash(),
            },
        ]
    }

    #[test]
    fn records_round_trip_with_kind_tags() {
        let expected_kinds = ["G26", "F1", "While", "Asc", "GearOverride"];

        for (record, expected_kind) in sample_records().into_iter().zip(expected_kinds) {
            let encoded = serde_json::to_string(&record).unwrap();
            let json: serde_json::Value = serde_json::from_str(&encoded).unwrap();
            assert_eq!(json["kind"], serde_json::json!(expected_kind));
            assert!(encoded.contains("\"schema_version\":"));
            assert!(encoded.contains("\"run_id\":"));
            assert!(encoded.contains("\"graph_launch_id\":"));

            let decoded: ControlPlaneRecord = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, record);
        }
    }

    #[test]
    fn ndjson_writer_emits_one_record_per_line() {
        let records = sample_records();
        let mut buf = Vec::new();

        write_control_plane_ndjson(&mut buf, records.clone()).unwrap();

        let text = String::from_utf8(buf).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(
            text.bytes().filter(|byte| *byte == b'\n').count(),
            records.len()
        );
        assert!(!text.contains("\n\n"));

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), records.len());

        for (line, expected) in lines.into_iter().zip(records) {
            let decoded: ControlPlaneRecord = serde_json::from_str(line).unwrap();
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn serialization_includes_raw_source_energy_state_and_hash() {
        for record in sample_records() {
            let json = serde_json::to_value(&record).unwrap();
            assert_eq!(json["source_energy_state"], source_energy_state_json());
            assert_eq!(
                json["source_energy_state_hash"],
                serde_json::json!("sha256:source-energy-state-v1")
            );
        }
    }

    #[test]
    fn source_energy_state_round_trip_preserves_raw_values_and_hash() {
        for record in sample_records() {
            let encoded = serde_json::to_string(&record).unwrap();
            let decoded: ControlPlaneRecord = serde_json::from_str(&encoded).unwrap();

            assert_eq!(decoded, record);
            let (source_energy_state, source_energy_state_hash) = record_energy_fields(&decoded);
            assert_eq!(source_energy_state.potential_energy, -12345.625);
            assert_eq!(source_energy_state.external_work, 0.03125);
            assert_eq!(source_energy_state.source, "adj_pe_wext_reducer");
            assert_eq!(source_energy_state_hash, "sha256:source-energy-state-v1");
        }
    }

    #[test]
    fn missing_required_variant_fields_are_rejected() {
        let json = serde_json::json!({
            "kind": "G26",
            "envelope": {
                "schema_version": CONTROL_TRACE_SCHEMA_VERSION,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 0
            },
            "frame_idx": 42,
            "stream_id": 2,
            "previous_gear": 1,
            "current_gear": 2,
            "dt_ps": 0.001,
            "switch_selector": 2,
            "source_energy_state": source_energy_state_json()
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        assert!(err
            .to_string()
            .contains("missing field `source_energy_state_hash`"));
    }

    #[test]
    fn flat_option_bag_fields_are_rejected() {
        let json = serde_json::json!({
            "kind": "F1",
            "envelope": {
                "schema_version": CONTROL_TRACE_SCHEMA_VERSION,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 1
            },
            "frame_idx": 43,
            "stream_id": 2,
            "cluster_id": 7,
            "adjudication_code": 1,
            "route": "construct",
            "violation_bits": 0,
            "source_energy_state": source_energy_state_json(),
            "source_energy_state_hash": source_energy_state_hash(),
            "g26_branch": null
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        assert!(err.to_string().contains("unknown field `g26_branch`"));
    }

    #[test]
    fn gear_override_rejects_non_host_mutation() {
        let json = serde_json::json!({
            "kind": "GearOverride",
            "envelope": {
                "schema_version": CONTROL_TRACE_SCHEMA_VERSION,
                "run_id": "run-001",
                "graph_launch_id": "graph-launch-001",
                "record_idx": 0
            },
            "frame_idx": 44,
            "stream_id": 2,
            "previous_gear": 2,
            "requested_gear": 0,
            "source": "test_harness",
            "reason": "must be rejected",
            "host_mutation": false,
            "source_energy_state": source_energy_state_json(),
            "source_energy_state_hash": source_energy_state_hash()
        });

        let err = serde_json::from_value::<ControlPlaneRecord>(json).unwrap_err();
        assert!(err.to_string().contains("host_mutation must be true"));
    }

    #[test]
    fn gear_override_rejects_false_host_mutation_on_serialize() {
        let record = ControlPlaneRecord::GearOverride {
            envelope: envelope(4),
            frame_idx: 44,
            stream_id: 2,
            previous_gear: 2,
            requested_gear: 0,
            source: "test_harness".to_string(),
            reason: "must be rejected".to_string(),
            host_mutation: false,
            source_energy_state: source_energy_state(),
            source_energy_state_hash: source_energy_state_hash(),
        };

        let err = serde_json::to_string(&record).unwrap_err();
        assert!(err.to_string().contains("host_mutation must be true"));
    }
}
