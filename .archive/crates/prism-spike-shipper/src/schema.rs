use serde::Deserialize;
use std::fmt;

// ── CcnsPhase ─────────────────────────────────────────────────────────────────
// Deserializes from either the canonical string names (Ghost Lattice v5+)
// or legacy integer encoding (0-4).

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CcnsPhase {
    Heating = 0,
    WarmHold = 1,
    Cooling = 2,
    ColdReturn = 3,
    ColdHold = 4,
}

impl CcnsPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Heating => "heating",
            Self::WarmHold => "warm_hold",
            Self::Cooling => "cooling",
            Self::ColdReturn => "cold_return",
            Self::ColdHold => "cold_hold",
        }
    }
    pub fn from_int(v: i64) -> Option<Self> {
        match v {
            0 => Some(Self::Heating),
            1 => Some(Self::WarmHold),
            2 => Some(Self::Cooling),
            3 => Some(Self::ColdReturn),
            4 => Some(Self::ColdHold),
            _ => None,
        }
    }
}

impl fmt::Display for CcnsPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for CcnsPhase {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{Error, Unexpected, Visitor};
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = CcnsPhase;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("ccns_phase string (e.g. \"heating\") or integer 0–4")
            }
            fn visit_str<E: Error>(self, v: &str) -> Result<CcnsPhase, E> {
                match v {
                    "heating" => Ok(CcnsPhase::Heating),
                    "warm_hold" => Ok(CcnsPhase::WarmHold),
                    "cooling" => Ok(CcnsPhase::Cooling),
                    "cold_return" => Ok(CcnsPhase::ColdReturn),
                    "cold_hold" => Ok(CcnsPhase::ColdHold),
                    _ => Err(E::invalid_value(Unexpected::Str(v), &self)),
                }
            }
            fn visit_i64<E: Error>(self, v: i64) -> Result<CcnsPhase, E> {
                CcnsPhase::from_int(v).ok_or_else(|| E::invalid_value(Unexpected::Signed(v), &self))
            }
            fn visit_u64<E: Error>(self, v: u64) -> Result<CcnsPhase, E> {
                self.visit_i64(v as i64)
            }
        }
        d.deserialize_any(V)
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone, Debug)]
pub struct SpikeFile {
    pub site_id: u32,
    pub n_spikes: usize,
    pub centroid: Option<[f32; 3]>,
    pub lining_cutoff: Option<f32>,
    pub open_frequency: Option<f32>,
    pub spikes: Vec<SpikeEvent>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct SpikeEvent {
    // Block-6 critical (required for feature_extractor.py compute_phase_features)
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub ccns_phase: CcnsPhase,
    pub timestep: i32,

    // Strongly preferred (absent → zero-padded in Block 6)
    pub vibrational_energy: Option<f32>,
    pub water_density: Option<f32>,
    pub n_nearby_excited: Option<i32>,
    pub spike_source: Option<String>,
    #[serde(rename = "type")]
    pub spike_type: Option<String>,
    pub frame_index: Option<i32>,
    pub aromatic_residue_id: Option<i32>,
    pub wavelength_nm: Option<f32>,
    pub stream_id: Option<i32>,
    pub mechanism_tag: Option<String>,
    pub wd_change: Option<f32>,
    pub phase_bits: Option<u32>,
}

pub const MECHANISM_LIF_THERMAL_SHAPE: &str = "LIF_THERMAL_SHAPE";

pub fn derive_mechanism_tag(event: &SpikeEvent) -> &'static str {
    let source = event
        .spike_source
        .as_deref()
        .unwrap_or("LIF")
        .trim()
        .to_ascii_uppercase();
    let spike_type = event
        .spike_type
        .as_deref()
        .unwrap_or("UNK")
        .trim()
        .to_ascii_uppercase();

    match source.as_str() {
        "UV" => "UV_AROMATIC_PERTURBATION",
        "EFP" => "EFP_ELECTROSTATIC_FIELD",
        "LADD" => "LADD_ATOM_DEPARTURE",
        "COFIRE" => "COFIRE_COHERENCE",
        "LIF" | "" if spike_type == "UNK" => MECHANISM_LIF_THERMAL_SHAPE,
        "LIF" | "" => "LIF_LOCAL_INTENSITY",
        _ if spike_type == "UNK" => MECHANISM_LIF_THERMAL_SHAPE,
        _ => "UNKNOWN_MECHANISM",
    }
}

// ── Validation ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub n_spikes: usize,
    pub site_id: u32,
    pub field_coverage_pct: f32,
    pub preferred_missing: Vec<&'static str>,
    pub optional_missing: Vec<&'static str>,
}

#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("JSON parse: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("empty spikes array")]
    EmptySpikes,
    #[error("coordinate {axis} non-finite at spike[{idx}]: {val}")]
    NonFiniteCoord {
        idx: usize,
        axis: &'static str,
        val: f32,
    },
    #[error("n_spikes field ({declared}) ≠ spikes array length ({actual})")]
    CountMismatch { declared: usize, actual: usize },
}

pub fn validate(raw: &[u8]) -> Result<(SpikeFile, ValidationStats), ValidationError> {
    let file: SpikeFile = serde_json::from_slice(raw)?;
    // Deserializing already enforced all CcnsPhase values are valid.

    if file.spikes.is_empty() {
        return Err(ValidationError::EmptySpikes);
    }

    // n_spikes field must agree with actual array length (catches truncation).
    if file.n_spikes != file.spikes.len() {
        return Err(ValidationError::CountMismatch {
            declared: file.n_spikes,
            actual: file.spikes.len(),
        });
    }

    // Coordinate sanity on all spikes.
    for (idx, s) in file.spikes.iter().enumerate() {
        for (axis, val) in [("x", s.x), ("y", s.y), ("z", s.z)] {
            if !val.is_finite() {
                return Err(ValidationError::NonFiniteCoord { idx, axis, val });
            }
        }
    }

    // Coverage reporting (Block-6-critical fields are always present after
    // successful deserialization; optional ones may be absent).
    let first = &file.spikes[0];
    let mut preferred_missing = Vec::new();
    let mut optional_missing = Vec::new();
    let mut optional_present = 0usize;
    const OPT_TOTAL: usize = 12; // optional per-spike fields

    if file.centroid.is_none() {
        preferred_missing.push("centroid");
    }
    if file.lining_cutoff.is_none() {
        preferred_missing.push("lining_cutoff");
    }
    if file.open_frequency.is_none() {
        preferred_missing.push("open_frequency");
    }

    macro_rules! track_opt {
        ($field:expr, $name:literal) => {
            if $field.is_some() {
                optional_present += 1;
            } else {
                optional_missing.push($name);
            }
        };
    }
    track_opt!(first.vibrational_energy, "vibrational_energy");
    track_opt!(first.water_density, "water_density");
    track_opt!(first.n_nearby_excited, "n_nearby_excited");
    track_opt!(first.spike_source, "spike_source");
    track_opt!(first.spike_type, "type");
    track_opt!(first.frame_index, "frame_index");
    track_opt!(first.aromatic_residue_id, "aromatic_residue_id");
    track_opt!(first.wavelength_nm, "wavelength_nm");
    track_opt!(first.stream_id, "stream_id");
    track_opt!(first.mechanism_tag, "mechanism_tag");
    track_opt!(first.wd_change, "wd_change");
    track_opt!(first.phase_bits, "phase_bits");

    // 6 required + N optional present, out of 6 + OPT_TOTAL
    let coverage = (6 + optional_present) as f32 / (6 + OPT_TOTAL) as f32 * 100.0;

    let stats = ValidationStats {
        n_spikes: file.spikes.len(),
        site_id: file.site_id,
        field_coverage_pct: coverage,
        preferred_missing,
        optional_missing,
    };

    Ok((file, stats))
}
