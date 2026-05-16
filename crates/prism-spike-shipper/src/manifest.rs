use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use std::{fs::OpenOptions, io::Write, path::Path};

#[derive(Serialize)]
pub struct Entry<'a> {
    pub ts: String,
    pub action: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_format: Option<&'a str>,
    pub json_path: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arrow_path: Option<&'a str>,
    pub parquet_path: Option<&'a str>,
    pub json_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arrow_bytes: Option<u64>,
    pub parquet_bytes: u64,
    pub r2_json: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r2_arrow: Option<&'a str>,
    pub r2_parquet: Option<&'a str>,
    pub r2_archive_json: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r2_archive_arrow: Option<&'a str>,
    pub json_verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arrow_verified: Option<bool>,
    pub parquet_verified: bool,
    pub json_deleted: bool,
    pub disk_free_gib: f64,
    pub dry_run: bool,
    /// Validation stats (coverage %, site_id, n_spikes)
    pub n_spikes: usize,
    pub site_id: u32,
    pub coverage_pct: f32,
    pub validation_note: &'a str,
}

pub fn append(manifest_path: &Path, entry: &Entry<'_>) -> Result<()> {
    let mut record = serde_json::to_string(entry)?;
    // Insert timestamp at deserialization time — override what caller set.
    // Actually caller already sets ts; just write.
    record.push('\n');
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(manifest_path)?;
    f.write_all(record.as_bytes())?;
    Ok(())
}

pub fn now_ts() -> String {
    Utc::now().to_rfc3339()
}
