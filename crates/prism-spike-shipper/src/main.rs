// prism-spike-shipper — kernel inotify daemon, native S3 upload
//
// Throughput design (replaces rclone subprocess approach):
//   • One persistent aws-sdk-s3 Client — HTTP/2 connection pool, no fork/TLS
//     re-handshake per call.
//   • 3 parallel PUTs per file via tokio::try_join!:
//       JSON  → prism-spikes-20260512  (training)
//       Parquet → prism-spikes-20260512  (training)
//       JSON  → prism-archive/raw-spike-archive  (permanent safety copy)
//   • json_data: Bytes cloned O(1) to both training and archive PUT; no memcpy.
//   • 3 parallel HEADs for verification.
//   • Semaphore(MAX_CONCURRENT_FILES=4) bounds concurrent file processing;
//     stability wait (10 s) happens outside the semaphore so the slot is only
//     held during active I/O.

mod config;
mod manifest;
mod parquet;
mod reject_cache;
mod s3;
mod schema;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use clap::Parser;
use inotify::{EventMask, Inotify, WatchDescriptor, WatchMask};
use tokio::sync::{mpsc, Semaphore};
use tracing::{error, info, warn};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser, Clone)]
#[command(name = "prism-spike-shipper")]
struct Cli {
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    retroactive: bool,
    #[arg(long)]
    retroactive_only: bool,
    #[arg(long, default_value = config::LOG_DIR)]
    log_dir: String,
    #[arg(long = "watch-dir")]
    extra_dirs: Vec<String>,
    /// Print local spike inventory and planned R2 keys; no S3, uploads, deletes, or manifest writes.
    #[arg(long)]
    inventory_only: bool,
    /// HEAD-check local spike artifacts against R2 and print missing/uploaded counts; no uploads, deletes, or manifest writes.
    #[arg(long)]
    backfill_report: bool,
    /// Keep local JSON files after successful upload/verification.
    #[arg(long)]
    keep_local_json: bool,
    /// Write generated Parquet sidecars under this cache root instead of beside source JSON.
    #[arg(long)]
    parquet_cache_dir: Option<PathBuf>,
    /// Override concurrent file limit (default: MAX_CONCURRENT_FILES)
    #[arg(long, default_value_t = config::MAX_CONCURRENT_FILES)]
    concurrency: usize,
    /// Bound simultaneous JSON parse + Parquet conversion work to protect RAM.
    #[arg(long, default_value_t = 16)]
    validation_concurrency: usize,
    /// Bound simultaneous object PUT requests across JSON, Parquet, and archive uploads.
    #[arg(long)]
    upload_concurrency: Option<usize>,
}

// ── Entrypoint ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    fs::create_dir_all(&cli.log_dir)?;
    let file_appender = tracing_appender::rolling::never(&cli.log_dir, config::LOG_FILE);
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("prism_spike_shipper=info".parse().unwrap()),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    // ── Watch directories ─────────────────────────────────────────────────────
    let mut watch_dirs: Vec<PathBuf> = config::default_watch_dirs();
    for d in &cli.extra_dirs {
        watch_dirs.push(PathBuf::from(d));
    }
    let reject_cache = Arc::new(
        reject_cache::RejectCache::load(config::REJECT_CACHE_PATH).context("load reject cache")?,
    );

    if cli.inventory_only {
        inventory_report(&watch_dirs, &reject_cache)?;
        return Ok(());
    }

    if cli.backfill_report {
        let s3 = s3::S3Client::from_env().context("S3 client init — check R2_* env vars")?;
        s3.probe(config::SPIKE_BUCKET)
            .await
            .context("R2 connectivity probe")?;
        backfill_report(&watch_dirs, &reject_cache, &s3).await?;
        return Ok(());
    }

    for d in &watch_dirs {
        if !d.exists() {
            fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
        }
    }
    let watch_roots = Arc::new(watch_dirs);

    // ── S3 client (shared across all tasks) ──────────────────────────────────
    let s3 = Arc::new(s3::S3Client::from_env().context("S3 client init — check R2_* env vars")?);
    s3.probe(config::SPIKE_BUCKET)
        .await
        .context("R2 connectivity probe")?;
    info!(
        spike_bucket = config::SPIKE_BUCKET,
        archive_bucket = config::ARCHIVE_BUCKET,
        concurrency = cli.concurrency,
        validation_concurrency = cli.validation_concurrency,
        upload_concurrency = cli.upload_concurrency.unwrap_or(cli.concurrency),
        dry_run = cli.dry_run,
        keep_local_json = cli.keep_local_json,
        parquet_cache_dir = ?cli.parquet_cache_dir,
        reject_cache_entries = reject_cache.len(),
        "prism-spike-shipper starting — S3 ok"
    );

    // ── Concurrency semaphore ─────────────────────────────────────────────────
    // Acquired AFTER the stability wait so the slot is only held during I/O.
    let sem = Arc::new(Semaphore::new(cli.concurrency));
    let validation_sem = Arc::new(Semaphore::new(cli.validation_concurrency));
    let upload_sem = Arc::new(Semaphore::new(
        cli.upload_concurrency.unwrap_or(cli.concurrency),
    ));

    // ── inotify watcher thread → async channel ────────────────────────────────
    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();
    let shutdown = Arc::new(AtomicBool::new(false));
    let sd_watcher = shutdown.clone();
    let dirs_clone = watch_roots.as_ref().clone();
    std::thread::Builder::new()
        .name("inotify-watcher".into())
        .spawn(move || {
            if let Err(e) = watcher_thread(tx, sd_watcher, &dirs_clone) {
                error!(error = %e, "inotify watcher thread crashed");
            }
        })?;

    // ── Retroactive scan ──────────────────────────────────────────────────────
    if cli.retroactive || cli.retroactive_only {
        info!("retroactive scan starting");
        let files = collect_spike_files(watch_roots.as_ref().as_slice());
        info!(files = files.len(), "retroactive scan queued");
        let mut handles = Vec::with_capacity(files.len());
        for path in files {
            let sem = Arc::clone(&sem);
            let validation_sem = Arc::clone(&validation_sem);
            let upload_sem = Arc::clone(&upload_sem);
            let s3 = Arc::clone(&s3);
            let roots = Arc::clone(&watch_roots);
            let reject_cache = Arc::clone(&reject_cache);
            let parquet_cache_dir = cli.parquet_cache_dir.clone();
            let dry_run = cli.dry_run;
            let keep_local_json = cli.keep_local_json;
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore closed");
                if let Err(e) = retroactive_process_file(
                    &path,
                    dry_run,
                    keep_local_json,
                    parquet_cache_dir.as_deref(),
                    &s3,
                    roots.as_ref().as_slice(),
                    &reject_cache,
                    &validation_sem,
                    &upload_sem,
                )
                .await
                {
                    error!(path = %path.display(), error = %e, "retroactive processing failed");
                }
            }));
        }
        for handle in handles {
            if let Err(e) = handle.await {
                error!(error = %e, "retroactive worker panicked");
            }
        }
        info!("retroactive scan complete");
        if cli.retroactive_only {
            shutdown.store(true, Ordering::Relaxed);
            return Ok(());
        }
    }

    // ── Signal handlers ───────────────────────────────────────────────────────
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;

    // ── Event loop ────────────────────────────────────────────────────────────
    // Receive events non-blockingly; spawn a task per file immediately.
    // Stability wait + semaphore acquisition happen inside the task.
    info!(concurrency = cli.concurrency, "entering watch loop");
    loop {
        tokio::select! {
            biased;
            _ = sigterm.recv() => {
                info!("SIGTERM — shutting down");
                shutdown.store(true, Ordering::Relaxed);
                break;
            }
            _ = sigint.recv() => {
                info!("SIGINT — shutting down");
                shutdown.store(true, Ordering::Relaxed);
                break;
            }
            Some(path) = rx.recv() => {
                let sem          = Arc::clone(&sem);
                let s3           = Arc::clone(&s3);
                let roots        = Arc::clone(&watch_roots);
                let reject_cache = Arc::clone(&reject_cache);
                let validation_sem = Arc::clone(&validation_sem);
                let upload_sem = Arc::clone(&upload_sem);
                let dry          = cli.dry_run;
                let keep_local_json = cli.keep_local_json;
                let parquet_cache_dir = cli.parquet_cache_dir.clone();
                tokio::spawn(async move {
                    // Stability wait outside semaphore — many files can wait in
                    // parallel without consuming concurrency slots.
                    tokio::time::sleep(Duration::from_secs(config::STABILITY_WAIT_SECS)).await;
                    if !is_stable(&path).await { return; }

                    // Acquire slot — limits active I/O to `concurrency` files.
                    let _permit = sem.acquire_owned().await.expect("semaphore closed");
                    if let Err(e) = process_file(
                        &path,
                        dry,
                        keep_local_json,
                        parquet_cache_dir.as_deref(),
                        &s3,
                        roots.as_ref().as_slice(),
                        &reject_cache,
                        &validation_sem,
                        &upload_sem,
                    ).await {
                        error!(path = %path.display(), error = %e, "processing failed");
                    }
                });
            }
        }
    }
    Ok(())
}

// ── Stability check ───────────────────────────────────────────────────────────

async fn is_stable(path: &Path) -> bool {
    let sz1 = tokio::fs::metadata(path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if sz1 == 0 {
        return false;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;
    let sz2 = tokio::fs::metadata(path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    sz1 == sz2 && sz2 > 0
}

// ── Core processing pipeline ─────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpikeArtifactKind {
    Json,
    Arrow,
}

impl SpikeArtifactKind {
    fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_string_lossy();
        if name.ends_with(config::JSON_SPIKE_SUFFIX) {
            Some(Self::Json)
        } else if name.ends_with(config::ARROW_SPIKE_SUFFIX) {
            Some(Self::Arrow)
        } else {
            None
        }
    }
}

async fn process_file(
    path: &Path,
    dry_run: bool,
    keep_local_json: bool,
    parquet_cache_dir: Option<&Path>,
    s3: &s3::S3Client,
    roots: &[PathBuf],
    reject_cache: &reject_cache::RejectCache,
    validation_sem: &Semaphore,
    upload_sem: &Semaphore,
) -> Result<()> {
    match SpikeArtifactKind::from_path(path) {
        Some(SpikeArtifactKind::Json) => {
            process_json_file(
                path,
                dry_run,
                keep_local_json,
                parquet_cache_dir,
                s3,
                roots,
                reject_cache,
                validation_sem,
                upload_sem,
            )
            .await
        }
        Some(SpikeArtifactKind::Arrow) => {
            process_arrow_file(
                path,
                dry_run,
                s3,
                roots,
                reject_cache,
                validation_sem,
                upload_sem,
            )
            .await
        }
        None => Ok(()),
    }
}

async fn process_json_file(
    json_path: &Path,
    dry_run: bool,
    keep_local_json: bool,
    parquet_cache_dir: Option<&Path>,
    s3: &s3::S3Client,
    roots: &[PathBuf],
    reject_cache: &reject_cache::RejectCache,
    validation_sem: &Semaphore,
    upload_sem: &Semaphore,
) -> Result<()> {
    let manifest_path = Path::new(config::MANIFEST_PATH);
    if let Some(p) = manifest_path.parent() {
        fs::create_dir_all(p).ok();
    }

    let identity = reject_cache::FileIdentity::from_path(json_path)
        .with_context(|| format!("stat {}", json_path.display()))?;
    if !dry_run {
        if let Some(reason) = reject_cache.unchanged_reason(json_path, &identity) {
            info!(
                path = %json_path.display(),
                reason,
                "cached schema reject unchanged — skipping"
            );
            return Ok(());
        }
    }

    // ── 1. Read JSON only for validation. Uploads stream from disk later. ────
    let validation_permit = validation_sem
        .acquire()
        .await
        .expect("validation semaphore closed");
    let json_bytes: Bytes = tokio::fs::read(json_path)
        .await
        .with_context(|| format!("read {}", json_path.display()))?
        .into();
    let json_size = identity.size;

    // ── 2. Schema validation ──────────────────────────────────────────────────
    let (spike_file, vstats) = match schema::validate(&json_bytes) {
        Ok(pair) => pair,
        Err(e) => {
            let reason = e.to_string();
            error!(path = %json_path.display(), error = %reason, "SCHEMA REJECT");
            if !dry_run {
                reject_cache.record(json_path, &identity, &reason).ok();
                manifest::append(
                    manifest_path,
                    &manifest::Entry {
                        ts: manifest::now_ts(),
                        action: "rejected",
                        source_format: Some("json"),
                        json_path: &json_path.to_string_lossy(),
                        arrow_path: None,
                        parquet_path: None,
                        json_bytes: json_size,
                        arrow_bytes: None,
                        parquet_bytes: 0,
                        r2_json: "",
                        r2_arrow: None,
                        r2_parquet: None,
                        r2_archive_json: None,
                        r2_archive_arrow: None,
                        json_verified: false,
                        arrow_verified: None,
                        parquet_verified: false,
                        json_deleted: false,
                        disk_free_gib: s3::disk_free_gib("/mnt/storage"),
                        dry_run,
                        n_spikes: 0,
                        site_id: 0,
                        coverage_pct: 0.0,
                        validation_note: "schema_reject",
                    },
                )
                .ok();
            }
            return Ok(());
        }
    };
    if !vstats.preferred_missing.is_empty() {
        warn!(path = %json_path.display(), missing = ?vstats.preferred_missing);
    }
    if !vstats.optional_missing.is_empty() {
        warn!(path = %json_path.display(), missing = ?vstats.optional_missing);
    }
    info!(
        path         = %json_path.display(),
        n_spikes     = vstats.n_spikes,
        site_id      = vstats.site_id,
        coverage_pct = vstats.field_coverage_pct,
        "SCHEMA OK"
    );

    // ── 3. Parquet conversion (CPU work — spawn_blocking) ─────────────────────
    let parquet_upload_path = parquet_upload_path_for(json_path);
    let parquet_path = if let Some(cache_dir) = parquet_cache_dir {
        let (_, parquet_key) = config::spike_key(&parquet_upload_path, roots);
        cache_dir.join(parquet_key)
    } else {
        parquet_upload_path.clone()
    };

    let parquet_bytes: Bytes = if parquet_path.exists() {
        info!(path = %parquet_path.display(), "Parquet already exists");
        drop(spike_file);
        tokio::fs::read(&parquet_path).await?.into()
    } else if dry_run {
        info!(path = %parquet_path.display(), "[dry-run] would write Parquet");
        drop(spike_file);
        Bytes::new()
    } else {
        let sf = spike_file;
        let pp = parquet_path.clone();
        if let Some(parent) = pp.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create parquet cache dir {}", parent.display()))?;
        }
        let rows = tokio::task::spawn_blocking(move || parquet::write(&sf, &pp)).await??;
        let data: Bytes = tokio::fs::read(&parquet_path).await?.into();
        info!(path = %parquet_path.display(), rows, bytes = data.len(), "Parquet written");
        data
    };
    let parquet_size = parquet_bytes.len() as u64;
    drop(validation_permit);

    // ── 4. S3 key construction ────────────────────────────────────────────────
    let (spike_b, spike_json_key) = config::spike_key(json_path, roots);
    let (spike_b2, spike_parquet_key) = config::spike_key(&parquet_upload_path, roots);
    let (arch_b, arch_json_key) = config::archive_key(json_path, roots);
    debug_assert_eq!(spike_b, spike_b2);

    // ── 5. 3 parallel PUTs ────────────────────────────────────────────────────
    // Use the proven multipart uploader. File concurrency is tuned so the two
    // JSON copies create 64 large transfer streams without the SDK file-stream
    // dispatch failures seen at 192 object PUTs.
    if dry_run {
        info!(
            training = %format!("s3://{}/{}", spike_b, spike_json_key),
            archive  = %format!("s3://{}/{}", arch_b, arch_json_key),
            "[dry-run] would upload 3 objects in parallel"
        );
    } else {
        // Skip objects already on R2 (idempotent restart-safety).
        let (j_exists, p_exists, a_exists) = tokio::join!(
            s3.exists(spike_b, &spike_json_key),
            s3.exists(spike_b, &spike_parquet_key),
            s3.exists(arch_b, &arch_json_key),
        );

        match (j_exists, p_exists, a_exists) {
            (true, true, true) => {
                info!(key = %spike_json_key, "all 3 objects already on R2 — skipping");
            }
            _ => {
                // Issue only the missing PUTs, still in parallel.
                let (r_json, r_parquet, r_archive) = tokio::join!(
                    async {
                        if j_exists {
                            Ok(())
                        } else {
                            let _permit =
                                upload_sem.acquire().await.expect("upload semaphore closed");
                            s3.put_with_retry(spike_b, &spike_json_key, json_bytes.clone())
                                .await
                        }
                    },
                    async {
                        if p_exists {
                            Ok(())
                        } else {
                            let _permit =
                                upload_sem.acquire().await.expect("upload semaphore closed");
                            s3.put_with_retry(spike_b, &spike_parquet_key, parquet_bytes.clone())
                                .await
                        }
                    },
                    async {
                        if a_exists {
                            Ok(())
                        } else {
                            let _permit =
                                upload_sem.acquire().await.expect("upload semaphore closed");
                            s3.put_with_retry(arch_b, &arch_json_key, json_bytes.clone())
                                .await
                        }
                    },
                );

                // All three must succeed; surface the first error.
                r_json.context("primary JSON upload")?;
                r_parquet.context("primary Parquet upload")?;
                r_archive.context("archive JSON upload")?;

                info!(
                    training_json    = %spike_json_key,
                    training_parquet = %spike_parquet_key,
                    archive          = %arch_json_key,
                    "3 parallel PUTs complete"
                );
            }
        }
    }

    // ── 6. 3 parallel HEADs (verify) ─────────────────────────────────────────
    let (json_ok, parquet_ok, archive_ok) = if dry_run {
        (true, true, true)
    } else {
        tokio::join!(
            s3.verify(spike_b, &spike_json_key, json_size),
            s3.verify(spike_b, &spike_parquet_key, parquet_size),
            s3.verify(arch_b, &arch_json_key, json_size),
        )
    };

    if !json_ok {
        error!(key = %spike_json_key,    "verify FAILED (training JSON)");
    }
    if !parquet_ok {
        error!(key = %spike_parquet_key, "verify FAILED (Parquet)");
    }
    if !archive_ok {
        error!(key = %arch_json_key,     "verify FAILED (archive JSON)");
    }

    // ── 7. Delete local JSON (primary + archive both must verify) ─────────────
    let safe = json_ok && archive_ok;
    let json_deleted = if safe && keep_local_json && !dry_run {
        info!(path = %json_path.display(), "keep-local-json enabled — local JSON retained");
        false
    } else if safe && !dry_run {
        fs::remove_file(json_path)
            .map(|_| {
                info!(path = %json_path.display(), "local JSON deleted");
                true
            })
            .unwrap_or_else(|e| {
                warn!(path = %json_path.display(), error = %e, "delete failed");
                false
            })
    } else if safe && dry_run {
        info!(path = %json_path.display(), "[dry-run] would delete local JSON");
        false
    } else {
        warn!(path = %json_path.display(), "NOT deleting — R2 verify incomplete");
        false
    };

    // ── 8. Disk pressure ──────────────────────────────────────────────────────
    let disk_free = s3::disk_free_gib("/mnt/storage");
    if parquet_ok && disk_free < config::DISK_CRITICAL_GB && !dry_run {
        warn!(
            free_gib = disk_free,
            "disk critical — evicting local Parquet"
        );
        fs::remove_file(&parquet_path).ok();
    }

    // ── 9. Manifest ───────────────────────────────────────────────────────────
    let completed = json_ok && parquet_ok && archive_ok;
    let ppath = parquet_path.to_string_lossy().into_owned();
    let sjk = format!("s3://{}/{}", spike_b, spike_json_key);
    let spk = format!("s3://{}/{}", spike_b, spike_parquet_key);
    let ajk = format!("s3://{}/{}", arch_b, arch_json_key);
    if !dry_run {
        manifest::append(
            manifest_path,
            &manifest::Entry {
                ts: manifest::now_ts(),
                action: if completed { "completed" } else { "partial" },
                source_format: Some("json"),
                json_path: &json_path.to_string_lossy(),
                arrow_path: None,
                parquet_path: Some(&ppath),
                json_bytes: json_size,
                arrow_bytes: None,
                parquet_bytes: parquet_size,
                r2_json: &sjk,
                r2_arrow: None,
                r2_parquet: Some(&spk),
                r2_archive_json: Some(&ajk),
                r2_archive_arrow: None,
                json_verified: json_ok,
                arrow_verified: None,
                parquet_verified: parquet_ok,
                json_deleted,
                disk_free_gib: disk_free,
                dry_run,
                n_spikes: vstats.n_spikes,
                site_id: vstats.site_id,
                coverage_pct: vstats.field_coverage_pct,
                validation_note: "ok",
            },
        )
        .ok();
    }

    info!(
        path         = %json_path.display(),
        json_ok, parquet_ok, archive_ok, json_deleted,
        disk_free_gib = disk_free,
        "DONE"
    );
    Ok(())
}

#[derive(Debug)]
struct ArrowValidation {
    n_batches: usize,
    n_columns: usize,
    has_mechanism_tag: bool,
}

fn validate_arrow_file(path: &Path) -> Result<ArrowValidation> {
    let file = fs::File::open(path).with_context(|| format!("open Arrow {}", path.display()))?;
    let reader =
        arrow_ipc::reader::FileReader::try_new(file, None).context("Arrow IPC file reader")?;
    let schema = reader.schema();
    let required = [
        "spike_id",
        "timestep",
        "x",
        "y",
        "z",
        "intensity",
        "spike_source",
        "phase_bits",
        "wd_change",
        "site_id",
    ];
    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|name| schema.field_with_name(name).is_err())
        .collect();
    if !missing.is_empty() {
        bail!(
            "Arrow schema missing required spike columns: {}",
            missing.join(",")
        );
    }
    Ok(ArrowValidation {
        n_batches: reader.num_batches(),
        n_columns: schema.fields().len(),
        has_mechanism_tag: schema.field_with_name("mechanism_tag").is_ok(),
    })
}

async fn process_arrow_file(
    arrow_path: &Path,
    dry_run: bool,
    s3: &s3::S3Client,
    roots: &[PathBuf],
    reject_cache: &reject_cache::RejectCache,
    validation_sem: &Semaphore,
    upload_sem: &Semaphore,
) -> Result<()> {
    let manifest_path = Path::new(config::MANIFEST_PATH);
    if let Some(p) = manifest_path.parent() {
        fs::create_dir_all(p).ok();
    }

    let identity = reject_cache::FileIdentity::from_path(arrow_path)
        .with_context(|| format!("stat {}", arrow_path.display()))?;
    if !dry_run {
        if let Some(reason) = reject_cache.unchanged_reason(arrow_path, &identity) {
            info!(
                path = %arrow_path.display(),
                reason,
                "cached Arrow reject unchanged — skipping"
            );
            return Ok(());
        }
    }

    let validation_permit = validation_sem
        .acquire()
        .await
        .expect("validation semaphore closed");
    let ap = arrow_path.to_path_buf();
    let stats = match tokio::task::spawn_blocking(move || validate_arrow_file(&ap)).await? {
        Ok(stats) => stats,
        Err(e) => {
            let reason = e.to_string();
            error!(path = %arrow_path.display(), error = %reason, "ARROW REJECT");
            if !dry_run {
                reject_cache.record(arrow_path, &identity, &reason).ok();
                manifest::append(
                    manifest_path,
                    &manifest::Entry {
                        ts: manifest::now_ts(),
                        action: "rejected",
                        source_format: Some("arrow"),
                        json_path: &arrow_path.to_string_lossy(),
                        arrow_path: Some(&arrow_path.to_string_lossy()),
                        parquet_path: None,
                        json_bytes: 0,
                        arrow_bytes: Some(identity.size),
                        parquet_bytes: 0,
                        r2_json: "",
                        r2_arrow: None,
                        r2_parquet: None,
                        r2_archive_json: None,
                        r2_archive_arrow: None,
                        json_verified: false,
                        arrow_verified: Some(false),
                        parquet_verified: false,
                        json_deleted: false,
                        disk_free_gib: s3::disk_free_gib("/mnt/storage"),
                        dry_run,
                        n_spikes: 0,
                        site_id: 0,
                        coverage_pct: 0.0,
                        validation_note: "arrow_schema_reject",
                    },
                )
                .ok();
            }
            return Ok(());
        }
    };
    drop(validation_permit);

    info!(
        path = %arrow_path.display(),
        batches = stats.n_batches,
        columns = stats.n_columns,
        has_mechanism_tag = stats.has_mechanism_tag,
        "ARROW SCHEMA OK"
    );

    let (spike_b, spike_arrow_key) = config::spike_key(arrow_path, roots);
    let (arch_b, arch_arrow_key) = config::archive_key(arrow_path, roots);

    if dry_run {
        info!(
            training = %format!("s3://{}/{}", spike_b, spike_arrow_key),
            archive = %format!("s3://{}/{}", arch_b, arch_arrow_key),
            "[dry-run] would stream-upload Arrow to training and archive"
        );
    } else {
        let (t_exists, a_exists) = tokio::join!(
            s3.exists(spike_b, &spike_arrow_key),
            s3.exists(arch_b, &arch_arrow_key),
        );
        let (r_training, r_archive) = tokio::join!(
            async {
                if t_exists {
                    Ok(())
                } else {
                    let _permit = upload_sem.acquire().await.expect("upload semaphore closed");
                    s3.put_file_with_retry(spike_b, &spike_arrow_key, arrow_path)
                        .await
                }
            },
            async {
                if a_exists {
                    Ok(())
                } else {
                    let _permit = upload_sem.acquire().await.expect("upload semaphore closed");
                    s3.put_file_with_retry(arch_b, &arch_arrow_key, arrow_path)
                        .await
                }
            },
        );
        r_training.context("primary Arrow upload")?;
        r_archive.context("archive Arrow upload")?;
        info!(
            training_arrow = %spike_arrow_key,
            archive_arrow = %arch_arrow_key,
            "Arrow PUTs complete"
        );
    }

    let (arrow_ok, archive_ok) = if dry_run {
        (true, true)
    } else {
        tokio::join!(
            s3.verify(spike_b, &spike_arrow_key, identity.size),
            s3.verify(arch_b, &arch_arrow_key, identity.size),
        )
    };

    let disk_free = s3::disk_free_gib("/mnt/storage");
    let completed = arrow_ok && archive_ok;
    let sak = format!("s3://{}/{}", spike_b, spike_arrow_key);
    let aak = format!("s3://{}/{}", arch_b, arch_arrow_key);
    if !dry_run {
        manifest::append(
            manifest_path,
            &manifest::Entry {
                ts: manifest::now_ts(),
                action: if completed { "completed" } else { "partial" },
                source_format: Some("arrow"),
                json_path: &arrow_path.to_string_lossy(),
                arrow_path: Some(&arrow_path.to_string_lossy()),
                parquet_path: None,
                json_bytes: 0,
                arrow_bytes: Some(identity.size),
                parquet_bytes: 0,
                r2_json: "",
                r2_arrow: Some(&sak),
                r2_parquet: None,
                r2_archive_json: None,
                r2_archive_arrow: Some(&aak),
                json_verified: false,
                arrow_verified: Some(arrow_ok && archive_ok),
                parquet_verified: false,
                json_deleted: false,
                disk_free_gib: disk_free,
                dry_run,
                n_spikes: 0,
                site_id: 0,
                coverage_pct: if stats.has_mechanism_tag { 100.0 } else { 96.7 },
                validation_note: if stats.has_mechanism_tag {
                    "ok"
                } else {
                    "ok_missing_mechanism_tag"
                },
            },
        )
        .ok();
    }

    info!(
        path = %arrow_path.display(),
        arrow_ok,
        archive_ok,
        disk_free_gib = disk_free,
        "DONE"
    );
    Ok(())
}

// ── inotify watcher thread ────────────────────────────────────────────────────

fn watcher_thread(
    tx: mpsc::UnboundedSender<PathBuf>,
    shutdown: Arc<AtomicBool>,
    watch_dirs: &[PathBuf],
) -> Result<()> {
    let mut inotify = Inotify::init().context("Inotify::init")?;
    let mut wd_map: HashMap<WatchDescriptor, PathBuf> = HashMap::new();
    let mask = WatchMask::CLOSE_WRITE | WatchMask::MOVED_TO | WatchMask::CREATE;

    for dir in watch_dirs {
        add_watches_recursive(&mut inotify, dir, mask, &mut wd_map);
    }

    let mut buf = vec![0u8; 65536];

    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        match inotify.read_events(&mut buf) {
            Ok(events) => {
                struct Ev {
                    wd: WatchDescriptor,
                    mask: EventMask,
                    name: Option<OsString>,
                }
                let owned: Vec<Ev> = events
                    .map(|e| Ev {
                        wd: e.wd,
                        mask: e.mask,
                        name: e.name.map(OsString::from),
                    })
                    .collect();

                let mut new_dirs = Vec::new();
                let mut spike_files = Vec::new();

                for ev in &owned {
                    let parent = match wd_map.get(&ev.wd) {
                        Some(p) => p.clone(),
                        None => continue,
                    };
                    let name = match &ev.name {
                        Some(n) => n.clone(),
                        None => continue,
                    };
                    let path = parent.join(&name);

                    if ev.mask.contains(EventMask::CREATE) && ev.mask.contains(EventMask::ISDIR) {
                        new_dirs.push(path);
                    } else if ev
                        .mask
                        .intersects(EventMask::CLOSE_WRITE | EventMask::MOVED_TO)
                    {
                        if is_spike_filename(&name.to_string_lossy()) {
                            info!(path = %path.display(), "spike file detected");
                            spike_files.push(path);
                        }
                    }
                }

                for dir in new_dirs {
                    add_watches_recursive(&mut inotify, &dir, mask, &mut wd_map);
                }
                for p in spike_files {
                    tx.send(p).ok();
                }
            }
            Err(ref e) if e.raw_os_error() == Some(libc::EAGAIN) => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return Err(e.into()),
        }

        // Periodic re-walk to catch any dirs missed during rapid creation bursts.
        for dir in watch_dirs {
            add_watches_recursive(&mut inotify, dir, mask, &mut wd_map);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Ok(())
}

fn add_watches_recursive(
    inotify: &mut Inotify,
    dir: &Path,
    mask: WatchMask,
    wd_map: &mut HashMap<WatchDescriptor, PathBuf>,
) {
    if !dir.is_dir() {
        return;
    }
    if let Ok(wd) = inotify.watches().add(dir, mask) {
        wd_map.entry(wd).or_insert_with(|| dir.to_path_buf());
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                add_watches_recursive(inotify, &entry.path(), mask, wd_map);
            }
        }
    }
}

fn is_spike_filename(name: &str) -> bool {
    config::SPIKE_SUFFIXES.iter().any(|s| name.ends_with(s))
}

// ── Retroactive scan ──────────────────────────────────────────────────────────

fn collect_spike_files(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();

    for root in roots {
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(meta) = fs::symlink_metadata(&path) else {
                    continue;
                };
                if meta.file_type().is_symlink() {
                    continue;
                }
                if meta.is_dir() {
                    stack.push(path);
                    continue;
                }
                let fname = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if is_spike_filename(&fname) && seen.insert(path.clone()) {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    files
}

fn parquet_upload_path_for(json_path: &Path) -> PathBuf {
    let stem = json_path.file_name().unwrap().to_string_lossy();
    let stem = stem.strip_suffix(".json").unwrap_or(&stem);
    json_path
        .parent()
        .unwrap()
        .join(format!("{}.parquet", stem))
}

async fn retroactive_process_file(
    path: &Path,
    dry_run: bool,
    keep_local_json: bool,
    parquet_cache_dir: Option<&Path>,
    s3: &s3::S3Client,
    roots: &[PathBuf],
    reject_cache: &reject_cache::RejectCache,
    validation_sem: &Semaphore,
    upload_sem: &Semaphore,
) -> Result<()> {
    let (sb, sk) = config::spike_key(path, roots);
    let (ab, ak) = config::archive_key(path, roots);
    match SpikeArtifactKind::from_path(path) {
        Some(SpikeArtifactKind::Json) => {
            let parquet_upload_path = parquet_upload_path_for(path);
            let (sbp, spk) = config::spike_key(&parquet_upload_path, roots);
            if s3.exists(sb, &sk).await && s3.exists(sbp, &spk).await && s3.exists(ab, &ak).await {
                info!(path = %path.display(), "retroactive: JSON already on R2");
                return Ok(());
            }
        }
        Some(SpikeArtifactKind::Arrow) => {
            if s3.exists(sb, &sk).await && s3.exists(ab, &ak).await {
                info!(path = %path.display(), "retroactive: Arrow already on R2");
                return Ok(());
            }
        }
        None => return Ok(()),
    }

    info!(path = %path.display(), "retroactive: processing");
    process_file(
        path,
        dry_run,
        keep_local_json,
        parquet_cache_dir,
        s3,
        roots,
        reject_cache,
        validation_sem,
        upload_sem,
    )
    .await
}

// ── Local inventory report ────────────────────────────────────────────────────

#[derive(Default)]
struct InventoryGroup {
    files: u64,
    json_files: u64,
    arrow_files: u64,
    bytes: u64,
    cached_rejects: u64,
    rejected_empty: u64,
}

fn inventory_report(roots: &[PathBuf], reject_cache: &reject_cache::RejectCache) -> Result<()> {
    let mut groups: BTreeMap<String, InventoryGroup> = BTreeMap::new();
    let mut total = InventoryGroup::default();
    let mut planned_keys: HashMap<String, PathBuf> = HashMap::new();
    let mut planned_archive_keys: HashMap<String, PathBuf> = HashMap::new();
    let mut seen_files: HashSet<PathBuf> = HashSet::new();
    let mut collisions: Vec<(String, PathBuf, PathBuf)> = Vec::new();
    let mut archive_collisions: Vec<(String, PathBuf, PathBuf)> = Vec::new();

    for root in roots {
        let root_label = root.display().to_string();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(meta) = fs::symlink_metadata(&path) else {
                    continue;
                };
                if meta.file_type().is_symlink() {
                    continue;
                }
                if meta.is_dir() {
                    stack.push(path);
                    continue;
                }

                let Some(kind) = SpikeArtifactKind::from_path(&path) else {
                    continue;
                };
                if !seen_files.insert(path.clone()) {
                    continue;
                }

                let identity = reject_cache::FileIdentity::from_path(&path).ok();
                let cached_reject = identity
                    .as_ref()
                    .and_then(|id| reject_cache.unchanged_reason(&path, id))
                    .unwrap_or_default();
                let is_cached_reject = !cached_reject.is_empty();
                let is_empty_reject = identity
                    .as_ref()
                    .map(|id| {
                        reject_cache.unchanged_reason_contains(&path, id, "empty spikes array")
                    })
                    .unwrap_or(false);
                let size = meta.len();

                let group = groups.entry(root_label.clone()).or_default();
                group.files += 1;
                match kind {
                    SpikeArtifactKind::Json => group.json_files += 1,
                    SpikeArtifactKind::Arrow => group.arrow_files += 1,
                }
                group.bytes += size;
                if is_cached_reject {
                    group.cached_rejects += 1;
                }
                if is_empty_reject {
                    group.rejected_empty += 1;
                }

                total.files += 1;
                match kind {
                    SpikeArtifactKind::Json => total.json_files += 1,
                    SpikeArtifactKind::Arrow => total.arrow_files += 1,
                }
                total.bytes += size;
                if is_cached_reject {
                    total.cached_rejects += 1;
                }
                if is_empty_reject {
                    total.rejected_empty += 1;
                }

                let plan = config::key_plan(&path, roots);
                let key_id = format!("s3://{}/{}", plan.bucket, plan.training_key);
                if let Some(first) = planned_keys.insert(key_id.clone(), path.clone()) {
                    collisions.push((key_id, first, path.clone()));
                }
                let archive_key_id = format!("s3://{}/{}", plan.archive_bucket, plan.archive_key);
                if let Some(first) =
                    planned_archive_keys.insert(archive_key_id.clone(), path.clone())
                {
                    archive_collisions.push((archive_key_id, first, path));
                }
            }
        }
    }

    println!("PRISM spike shipper inventory");
    println!("mode\tmetadata_only_no_uploads_no_deletes_no_manifest");
    println!("reject_cache_entries\t{}", reject_cache.len());
    println!("root\tfiles\tjson\tarrow\tbytes_gib\tcached_rejects\trejected_empty");
    for (root, group) in &groups {
        println!(
            "{}\t{}\t{}\t{}\t{:.3}\t{}\t{}",
            root,
            group.files,
            group.json_files,
            group.arrow_files,
            group.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
            group.cached_rejects,
            group.rejected_empty
        );
    }
    println!(
        "TOTAL\t{}\t{}\t{}\t{:.3}\t{}\t{}",
        total.files,
        total.json_files,
        total.arrow_files,
        total.bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        total.cached_rejects,
        total.rejected_empty
    );
    println!("planned_training_key_collisions\t{}", collisions.len());
    for (key, first, second) in collisions.iter().take(20) {
        println!(
            "collision\t{}\t{}\t{}",
            key,
            first.display(),
            second.display()
        );
    }
    println!(
        "planned_archive_key_collisions\t{}",
        archive_collisions.len()
    );
    for (key, first, second) in archive_collisions.iter().take(20) {
        println!(
            "archive_collision\t{}\t{}\t{}",
            key,
            first.display(),
            second.display()
        );
    }

    Ok(())
}

#[derive(Default)]
struct BackfillReport {
    files: u64,
    json_files: u64,
    arrow_files: u64,
    json_bytes: u64,
    arrow_bytes: u64,
    cached_rejects: u64,
    rejected_empty: u64,
    expected_objects: u64,
    uploaded_objects: u64,
    missing_objects: u64,
    fully_uploaded_files: u64,
    missing_any_files: u64,
    missing_training_objects: u64,
    missing_parquet_objects: u64,
    missing_archive_objects: u64,
    size_mismatch_objects: u64,
}

async fn check_object(
    s3: &s3::S3Client,
    bucket: &str,
    key: &str,
    expected_size: Option<u64>,
    report: &mut BackfillReport,
    missing_kind: MissingKind,
) -> bool {
    report.expected_objects += 1;
    match s3.head(bucket, key).await {
        None => {
            report.missing_objects += 1;
            match missing_kind {
                MissingKind::Training => report.missing_training_objects += 1,
                MissingKind::Parquet => report.missing_parquet_objects += 1,
                MissingKind::Archive => report.missing_archive_objects += 1,
            }
            false
        }
        Some(len) => {
            if expected_size.is_some_and(|expected| expected != 0 && len.abs_diff(expected) > 0) {
                report.size_mismatch_objects += 1;
            }
            report.uploaded_objects += 1;
            true
        }
    }
}

#[derive(Clone, Copy)]
enum MissingKind {
    Training,
    Parquet,
    Archive,
}

async fn backfill_report(
    roots: &[PathBuf],
    reject_cache: &reject_cache::RejectCache,
    s3: &s3::S3Client,
) -> Result<()> {
    let files = collect_spike_files(roots);
    let mut report = BackfillReport::default();

    for path in files {
        let Some(kind) = SpikeArtifactKind::from_path(&path) else {
            continue;
        };
        let identity = match reject_cache::FileIdentity::from_path(&path) {
            Ok(id) => id,
            Err(_) => continue,
        };
        report.files += 1;
        match kind {
            SpikeArtifactKind::Json => {
                report.json_files += 1;
                report.json_bytes += identity.size;
            }
            SpikeArtifactKind::Arrow => {
                report.arrow_files += 1;
                report.arrow_bytes += identity.size;
            }
        }

        if let Some(reason) = reject_cache.unchanged_reason(&path, &identity) {
            report.cached_rejects += 1;
            if reason.contains("empty spikes array") {
                report.rejected_empty += 1;
            }
            continue;
        }

        let mut file_ok = true;
        match kind {
            SpikeArtifactKind::Json => {
                let (sb, sk) = config::spike_key(&path, roots);
                let parquet_upload_path = parquet_upload_path_for(&path);
                let (spb, spk) = config::spike_key(&parquet_upload_path, roots);
                let (ab, ak) = config::archive_key(&path, roots);
                file_ok &= check_object(
                    s3,
                    sb,
                    &sk,
                    Some(identity.size),
                    &mut report,
                    MissingKind::Training,
                )
                .await;
                file_ok &=
                    check_object(s3, spb, &spk, None, &mut report, MissingKind::Parquet).await;
                file_ok &= check_object(
                    s3,
                    ab,
                    &ak,
                    Some(identity.size),
                    &mut report,
                    MissingKind::Archive,
                )
                .await;
            }
            SpikeArtifactKind::Arrow => {
                let (sb, sk) = config::spike_key(&path, roots);
                let (ab, ak) = config::archive_key(&path, roots);
                file_ok &= check_object(
                    s3,
                    sb,
                    &sk,
                    Some(identity.size),
                    &mut report,
                    MissingKind::Training,
                )
                .await;
                file_ok &= check_object(
                    s3,
                    ab,
                    &ak,
                    Some(identity.size),
                    &mut report,
                    MissingKind::Archive,
                )
                .await;
            }
        }

        if file_ok {
            report.fully_uploaded_files += 1;
        } else {
            report.missing_any_files += 1;
        }
    }

    println!("PRISM spike shipper backfill report");
    println!("mode\tdry_run_head_only_no_uploads_no_deletes_no_manifest");
    println!("reject_cache_entries\t{}", reject_cache.len());
    println!("files\t{}", report.files);
    println!("json_files\t{}", report.json_files);
    println!("arrow_files\t{}", report.arrow_files);
    println!(
        "json_bytes_gib\t{:.3}",
        report.json_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "arrow_bytes_gib\t{:.3}",
        report.arrow_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!("cached_rejects\t{}", report.cached_rejects);
    println!("rejected_empty\t{}", report.rejected_empty);
    println!("expected_objects\t{}", report.expected_objects);
    println!("uploaded_objects\t{}", report.uploaded_objects);
    println!("missing_objects\t{}", report.missing_objects);
    println!("fully_uploaded_files\t{}", report.fully_uploaded_files);
    println!("missing_any_files\t{}", report.missing_any_files);
    println!(
        "missing_training_objects\t{}",
        report.missing_training_objects
    );
    println!(
        "missing_parquet_objects\t{}",
        report.missing_parquet_objects
    );
    println!(
        "missing_archive_objects\t{}",
        report.missing_archive_objects
    );
    println!("size_mismatch_objects\t{}", report.size_mismatch_objects);
    Ok(())
}
