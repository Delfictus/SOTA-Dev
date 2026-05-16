use std::path::{Component, Path, PathBuf};

// ── Bucket names ─────────────────────────────────────────────────────────────

/// Training bucket — feature_extractor.py must set R2_BUCKET=prism-spikes-20260512
pub const SPIKE_BUCKET: &str = "prism-spikes-20260512";
/// Permanent archive of raw JSON — never pruned, separate from training bucket
pub const ARCHIVE_BUCKET: &str = "prism-archive";
pub const ARCHIVE_PREFIX: &str = "raw-spike-archive";

// ── S3 key construction ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct KeyPlan {
    pub bucket: &'static str,
    pub training_key: String,
    pub archive_bucket: &'static str,
    pub archive_key: String,
}

/// S3 key plan for both training and archive buckets.
///
/// The relative path is preserved from the most specific known route anchor
/// (`/cryptobench199/`, `/runs/`, `/tmp/`, etc.) or, for added watch roots, from
/// the longest matching configured root. This prevents deep output trees such as
/// `hect-family/itch/5_engine/...` from collapsing into a single `5_engine/`
/// namespace on R2.
pub fn key_plan(path: &Path, roots: &[PathBuf]) -> KeyPlan {
    let (prefix, relative_key, archive_keeps_prefix) = routed_relative_key(path, roots);
    let training_key = format!("{}/{}", prefix, relative_key);
    let archive_key = if archive_keeps_prefix {
        format!("{}/{}", ARCHIVE_PREFIX, training_key)
    } else {
        format!("{}/{}", ARCHIVE_PREFIX, relative_key)
    };
    KeyPlan {
        bucket: SPIKE_BUCKET,
        training_key,
        archive_bucket: ARCHIVE_BUCKET,
        archive_key,
    }
}

/// S3 key in the training bucket: "<prefix>/<preserved-relative-path>"
pub fn spike_key(path: &Path, roots: &[PathBuf]) -> (&'static str, String) {
    let plan = key_plan(path, roots);
    (plan.bucket, plan.training_key)
}

/// S3 key in the archive bucket.
///
/// Known historical routes keep their existing archive layout, while configured
/// broad/fallback roots include the training prefix to avoid namespace collapse.
pub fn archive_key(path: &Path, roots: &[PathBuf]) -> (&'static str, String) {
    let plan = key_plan(path, roots);
    (plan.archive_bucket, plan.archive_key)
}

// ── Routing ───────────────────────────────────────────────────────────────────

const ROUTE_ANCHORS: &[(&str, &str)] = &[
    ("/twin-runs/", "twin-runs"),
    ("/cryptobench199/", "cryptobench199"),
    ("/v1.1-physics/", "v1.1-physics"),
    ("/10k-runs/", "10k-runs"),
    ("/runs/", "runs"),
];

fn routed_relative_key(path: &Path, roots: &[PathBuf]) -> (&'static str, String, bool) {
    let s = path.to_string_lossy();

    if let Some(rel) = s.strip_prefix("/tmp/") {
        return ("dev-runs", normalize_key_path(Path::new(rel)), true);
    }

    for (anchor, prefix) in ROUTE_ANCHORS {
        if let Some(idx) = s.find(anchor) {
            let rel = &s[(idx + anchor.len())..];
            return (prefix, normalize_key_path(Path::new(rel)), false);
        }
    }

    if let Some(rel) = relative_to_best_root(path, roots) {
        return ("runs", rel, true);
    }

    let fallback = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|dir| path.file_name().map(|file| PathBuf::from(dir).join(file)))
        .or_else(|| path.file_name().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("unknown"));
    ("runs", normalize_key_path(&fallback), false)
}

fn relative_to_best_root(path: &Path, roots: &[PathBuf]) -> Option<String> {
    roots
        .iter()
        .filter_map(|root| {
            path.strip_prefix(root)
                .ok()
                .map(|rel| (root.components().count(), normalize_key_path(rel)))
        })
        .max_by_key(|(depth, _)| *depth)
        .map(|(_, rel)| rel)
        .filter(|rel| !rel.is_empty())
}

fn normalize_key_path(path: &Path) -> String {
    let parts: Vec<String> = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        "unknown".to_string()
    } else {
        parts.join("/")
    }
}

// ── Daemon constants ──────────────────────────────────────────────────────────

pub const STABILITY_WAIT_SECS: u64 = 10;
pub const DISK_CRITICAL_GB: f64 = 100.0;
pub const UPLOAD_RETRY_MAX: u32 = 6;
pub const UPLOAD_RETRY_BACKOFF: u64 = 5; // seconds, multiplied by attempt number
/// Max spike files processed concurrently. Upload PUTs remain globally bounded
/// by `--upload-concurrency`; this default keeps production backfill at the
/// requested high-throughput floor while validation remains separately capped.
pub const MAX_CONCURRENT_FILES: usize = 64;

// ── Multipart upload ──────────────────────────────────────────────────────────
// Match rclone: chunk_size=64M, upload_concurrency=4 — 4 parallel TCP streams
// per file saturates the 134 MiB/s measured link ceiling.
/// Files above this threshold use multipart upload.
pub const MULTIPART_THRESHOLD: usize = 32 * 1024 * 1024; // 32 MB
/// Size of each part (= rclone chunk_size).
pub const MULTIPART_PART_SIZE: usize = 64 * 1024 * 1024; // 64 MB
/// Concurrent part uploads per object. Keep this at 1 for high-volume
/// retroactive Arrow backfills: the outer upload semaphore already allows 64
/// active object uploads, so higher per-object fanout overloads the R2 endpoint
/// and triggers repeated UploadPart dispatch failures.
pub const MULTIPART_CONCURRENCY: usize = 1;

pub const MANIFEST_PATH: &str = "/mnt/storage/prism-outputs/.r2-sync-manifest.jsonl";
pub const REJECT_CACHE_PATH: &str = "/mnt/storage/prism-outputs/.r2-reject-cache.jsonl";
pub const LOG_DIR: &str = "/var/log/prism";
pub const LOG_FILE: &str = "spike-shipper.log";

pub const WATCH_DIRS: &[&str] = &[
    "/mnt/storage/prism-outputs/runs",
    "/mnt/storage/prism-outputs/twin-runs",
    "/mnt/storage/prism-outputs/10k-runs",
    "/tmp",
];

pub const JSON_SPIKE_SUFFIX: &str = ".spike_events.json";
pub const ARROW_SPIKE_SUFFIX: &str = ".spike_events.arrow";
pub const SPIKE_SUFFIXES: &[&str] = &[JSON_SPIKE_SUFFIX, ARROW_SPIKE_SUFFIX];

pub fn default_watch_dirs() -> Vec<PathBuf> {
    WATCH_DIRS.iter().map(PathBuf::from).collect()
}
