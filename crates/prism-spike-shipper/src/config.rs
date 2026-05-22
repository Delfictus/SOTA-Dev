use std::path::{Component, Path, PathBuf};

// ── Bucket names ─────────────────────────────────────────────────────────────

/// Training bucket — feature_extractor.py must set R2_BUCKET=prism-spikes-20260512
pub const DEFAULT_SPIKE_BUCKET: &str = "prism-spikes-20260512";
/// Permanent archive of raw JSON — never pruned, separate from training bucket
pub const DEFAULT_ARCHIVE_BUCKET: &str = "prism-archive";
pub const DEFAULT_ARCHIVE_PREFIX: &str = "raw-spike-archive";

#[derive(Clone, Debug)]
pub struct RootRoute {
    pub root: PathBuf,
    pub prefix: String,
}

#[derive(Clone, Debug)]
pub struct UploadConfig {
    pub spike_bucket: String,
    pub archive_bucket: String,
    pub archive_prefix: String,
    pub fallback_prefix: String,
    pub root_routes: Vec<RootRoute>,
}

impl UploadConfig {
    pub fn new(
        spike_bucket: String,
        archive_bucket: String,
        archive_prefix: String,
        fallback_prefix: String,
        root_routes: Vec<RootRoute>,
    ) -> Self {
        Self {
            spike_bucket,
            archive_bucket,
            archive_prefix: clean_prefix(&archive_prefix),
            fallback_prefix: clean_prefix(&fallback_prefix),
            root_routes: root_routes
                .into_iter()
                .map(|route| RootRoute {
                    root: route.root,
                    prefix: clean_prefix(&route.prefix),
                })
                .collect(),
        }
    }
}

// ── S3 key construction ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct KeyPlan {
    pub bucket: String,
    pub training_key: String,
    pub archive_bucket: String,
    pub archive_key: String,
}

/// S3 key plan for both training and archive buckets.
///
/// The relative path is preserved from the most specific known route anchor
/// (`/cryptobench199/`, `/runs/`, `/tmp/`, etc.) or, for added watch roots, from
/// the longest matching configured root. This prevents deep output trees such as
/// `hect-family/itch/5_engine/...` from collapsing into a single `5_engine/`
/// namespace on R2.
pub fn key_plan(path: &Path, roots: &[PathBuf], upload: &UploadConfig) -> KeyPlan {
    let (prefix, relative_key, archive_keeps_prefix) = routed_relative_key(path, roots, upload);
    let training_key = format!("{}/{}", prefix, relative_key);
    let archive_key = if archive_keeps_prefix {
        format!("{}/{}", upload.archive_prefix, training_key)
    } else {
        format!("{}/{}", upload.archive_prefix, relative_key)
    };
    KeyPlan {
        bucket: upload.spike_bucket.clone(),
        training_key,
        archive_bucket: upload.archive_bucket.clone(),
        archive_key,
    }
}

/// S3 key in the training bucket: "<prefix>/<preserved-relative-path>"
pub fn spike_key(path: &Path, roots: &[PathBuf], upload: &UploadConfig) -> (String, String) {
    let plan = key_plan(path, roots, upload);
    (plan.bucket, plan.training_key)
}

/// S3 key in the archive bucket.
///
/// Known historical routes keep their existing archive layout, while configured
/// broad/fallback roots include the training prefix to avoid namespace collapse.
pub fn archive_key(path: &Path, roots: &[PathBuf], upload: &UploadConfig) -> (String, String) {
    let plan = key_plan(path, roots, upload);
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

fn routed_relative_key(
    path: &Path,
    roots: &[PathBuf],
    upload: &UploadConfig,
) -> (String, String, bool) {
    let s = path.to_string_lossy();

    if let Some((prefix, rel)) = relative_to_best_route(path, &upload.root_routes) {
        return (prefix, rel, true);
    }

    if let Some(rel) = s.strip_prefix("/tmp/") {
        return (
            "dev-runs".to_string(),
            normalize_key_path(Path::new(rel)),
            true,
        );
    }

    for (anchor, prefix) in ROUTE_ANCHORS {
        if let Some(idx) = s.find(anchor) {
            let rel = &s[(idx + anchor.len())..];
            return (
                (*prefix).to_string(),
                normalize_key_path(Path::new(rel)),
                false,
            );
        }
    }

    if let Some(rel) = relative_to_best_root(path, roots) {
        return (upload.fallback_prefix.clone(), rel, true);
    }

    let fallback = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|dir| path.file_name().map(|file| PathBuf::from(dir).join(file)))
        .or_else(|| path.file_name().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("unknown"));
    (
        upload.fallback_prefix.clone(),
        normalize_key_path(&fallback),
        false,
    )
}

fn relative_to_best_route(path: &Path, routes: &[RootRoute]) -> Option<(String, String)> {
    routes
        .iter()
        .filter_map(|route| {
            path.strip_prefix(&route.root).ok().map(|rel| {
                (
                    route.root.components().count(),
                    route.prefix.clone(),
                    normalize_key_path(rel),
                )
            })
        })
        .max_by_key(|(depth, _, _)| *depth)
        .map(|(_, prefix, rel)| (prefix, rel))
        .filter(|(_, rel)| !rel.is_empty())
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

fn clean_prefix(prefix: &str) -> String {
    let cleaned = prefix
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("/");
    if cleaned.is_empty() {
        "runs".to_string()
    } else {
        cleaned
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
pub const PARQUET_SPIKE_SUFFIX: &str = ".spike_events.parquet";
pub const SPIKE_SUFFIXES: &[&str] = &[JSON_SPIKE_SUFFIX, ARROW_SPIKE_SUFFIX, PARQUET_SPIKE_SUFFIX];

pub fn default_watch_dirs() -> Vec<PathBuf> {
    WATCH_DIRS.iter().map(PathBuf::from).collect()
}
