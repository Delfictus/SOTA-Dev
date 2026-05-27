/// Native S3 client for Cloudflare R2.
///
/// Upload strategy:
///   ≤ 32 MB  → single-part PUT (small files, Parquet, tiny JSON)
///   > 32 MB  → multipart upload: 64 MB parts, with per-object part fanout
///              kept below the outer upload semaphore so large Arrow backfills
///              can run many objects concurrently without overloading R2.
use crate::config;
use anyhow::{bail, Result};
use aws_sdk_s3::{
    config::{BehaviorVersion, Credentials, Region},
    primitives::{ByteStream, Length},
    types::{CompletedMultipartUpload, CompletedPart},
    Client, Config,
};
use bytes::Bytes;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tracing::{debug, info, warn};

pub struct S3Client {
    inner: Client,
}

impl S3Client {
    /// Build from `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`
    /// environment variables (loaded by systemd from credentials.env).
    pub fn from_env() -> Result<Self> {
        let ak = std::env::var("R2_ACCESS_KEY_ID")
            .map_err(|_| anyhow::anyhow!("R2_ACCESS_KEY_ID not set"))?;
        let sk = std::env::var("R2_SECRET_ACCESS_KEY")
            .map_err(|_| anyhow::anyhow!("R2_SECRET_ACCESS_KEY not set"))?;
        let ep =
            std::env::var("R2_ENDPOINT").map_err(|_| anyhow::anyhow!("R2_ENDPOINT not set"))?;
        Ok(Self::new(&ak, &sk, &ep))
    }

    pub fn new(access_key: &str, secret_key: &str, endpoint: &str) -> Self {
        let creds = Credentials::new(access_key, secret_key, None, None, "r2-static");
        let conf = Config::builder()
            .credentials_provider(creds)
            .region(Region::new("auto"))
            .endpoint_url(endpoint)
            .force_path_style(true)
            .behavior_version(BehaviorVersion::latest())
            .build();
        Self {
            inner: Client::from_conf(conf),
        }
    }

    /// Single-part PUT.  `data` is a `Bytes` handle — clone() is O(1).
    async fn put(&self, bucket: &str, key: &str, data: Bytes) -> Result<()> {
        let len = data.len() as i64;
        self.inner
            .put_object()
            .bucket(bucket)
            .key(key)
            .content_length(len)
            .body(ByteStream::from(data))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("PUT s3://{}/{}: {}", bucket, key, e))?;
        debug!(bucket, key, bytes = len, "single-part PUT ok");
        Ok(())
    }

    /// Single-part PUT streamed from disk. The AWS SDK file ByteStream is
    /// retryable and avoids retaining multi-GB JSON buffers during backfill.
    async fn put_file(&self, bucket: &str, key: &str, path: &Path, len: u64) -> Result<()> {
        let body = ByteStream::read_from()
            .path(path)
            .length(Length::Exact(len))
            .buffer_size(1024 * 1024)
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("open {} for PUT: {}", path.display(), e))?;
        self.inner
            .put_object()
            .bucket(bucket)
            .key(key)
            .content_length(len as i64)
            .body(body)
            .send()
            .await
            .map_err(|e| {
                anyhow::anyhow!("PUT s3://{}/{} from {}: {}", bucket, key, path.display(), e)
            })?;
        debug!(bucket, key, path = %path.display(), bytes = len, "file PUT ok");
        Ok(())
    }

    async fn put_file_multipart(
        &self,
        bucket: &str,
        key: &str,
        path: &Path,
        len: u64,
    ) -> Result<()> {
        let part_size = config::MULTIPART_PART_SIZE as u64;
        let n_parts = len.div_ceil(part_size) as usize;
        let path_buf = path.to_path_buf();

        let upload_id = self
            .inner
            .create_multipart_upload()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("CreateMultipartUpload s3://{}/{}: {}", bucket, key, e))?
            .upload_id()
            .ok_or_else(|| anyhow::anyhow!("CreateMultipartUpload: no upload_id returned"))?
            .to_string();

        info!(
            bucket,
            key,
            path = %path.display(),
            bytes = len,
            parts = n_parts,
            concurrency = config::MULTIPART_CONCURRENCY,
            "file multipart upload starting"
        );

        let sem = Arc::new(tokio::sync::Semaphore::new(config::MULTIPART_CONCURRENCY));
        let mut handles = Vec::with_capacity(n_parts);

        for i in 0..n_parts {
            let start = i as u64 * part_size;
            let part_len = (len - start).min(part_size);
            let part_number = (i + 1) as i32;
            let b = bucket.to_string();
            let k = key.to_string();
            let uid = upload_id.clone();
            let p: PathBuf = path_buf.clone();
            let sem2 = Arc::clone(&sem);
            let client = self.inner.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem2.acquire_owned().await.expect("semaphore closed");
                let body = ByteStream::read_from()
                    .path(&p)
                    .offset(start)
                    .length(Length::Exact(part_len))
                    .buffer_size(1024 * 1024)
                    .build()
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "open {} part {} offset {} len {} for PUT: {}",
                            p.display(),
                            part_number,
                            start,
                            part_len,
                            e
                        )
                    })?;
                let resp = client
                    .upload_part()
                    .bucket(&b)
                    .key(&k)
                    .upload_id(&uid)
                    .part_number(part_number)
                    .content_length(part_len as i64)
                    .body(body)
                    .send()
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!("UploadPart {} s3://{}/{}: {}", part_number, b, k, e)
                    })?;
                let etag = resp
                    .e_tag()
                    .ok_or_else(|| {
                        anyhow::anyhow!("UploadPart {}: no ETag in response", part_number)
                    })?
                    .to_string();
                debug!(part = part_number, bytes = part_len, "file part ok");
                Ok::<_, anyhow::Error>((part_number, etag))
            }));
        }

        let mut completed: Vec<CompletedPart> = Vec::with_capacity(n_parts);
        let mut failed: Option<anyhow::Error> = None;
        for handle in handles {
            match handle
                .await
                .map_err(|e| anyhow::anyhow!("file part task panicked: {}", e))
            {
                Err(e) | Ok(Err(e)) => {
                    failed.get_or_insert(e);
                }
                Ok(Ok((pn, etag))) => {
                    completed.push(CompletedPart::builder().part_number(pn).e_tag(etag).build());
                }
            }
        }

        if let Some(err) = failed {
            let _ = self
                .inner
                .abort_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(&upload_id)
                .send()
                .await;
            return Err(err);
        }

        completed.sort_by_key(|p| p.part_number().unwrap_or(0));
        self.inner
            .complete_multipart_upload()
            .bucket(bucket)
            .key(key)
            .upload_id(&upload_id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(completed))
                    .build(),
            )
            .send()
            .await
            .map_err(|e| {
                anyhow::anyhow!("CompleteMultipartUpload s3://{}/{}: {}", bucket, key, e)
            })?;

        info!(
            bucket,
            key,
            path = %path.display(),
            bytes = len,
            parts = n_parts,
            "file multipart upload complete"
        );
        Ok(())
    }

    /// Multipart upload: 64 MB parts, MULTIPART_CONCURRENCY parts in parallel.
    /// Bytes::slice() is O(1) — no memcpy per part.
    async fn put_multipart(&self, bucket: &str, key: &str, data: Bytes) -> Result<()> {
        let total = data.len();
        let part_size = config::MULTIPART_PART_SIZE;
        let n_parts = (total + part_size - 1) / part_size;

        // ── 1. Initiate ───────────────────────────────────────────────────────
        let upload_id = self
            .inner
            .create_multipart_upload()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("CreateMultipartUpload s3://{}/{}: {}", bucket, key, e))?
            .upload_id()
            .ok_or_else(|| anyhow::anyhow!("CreateMultipartUpload: no upload_id returned"))?
            .to_string();

        info!(
            bucket,
            key,
            bytes = total,
            parts = n_parts,
            concurrency = config::MULTIPART_CONCURRENCY,
            "multipart upload starting"
        );

        // ── 2. Upload parts (semaphore-bounded concurrency) ───────────────────
        let sem = Arc::new(tokio::sync::Semaphore::new(config::MULTIPART_CONCURRENCY));
        let mut handles = Vec::with_capacity(n_parts);

        for i in 0..n_parts {
            let start = i * part_size;
            let end = ((i + 1) * part_size).min(total);
            let part_data = data.slice(start..end); // O(1)
            let part_number = (i + 1) as i32;
            let b = bucket.to_string();
            let k = key.to_string();
            let uid = upload_id.clone();
            let sem2 = Arc::clone(&sem);
            let client = self.inner.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem2.acquire_owned().await.expect("semaphore closed");
                let part_len = part_data.len() as i64;
                let resp = client
                    .upload_part()
                    .bucket(&b)
                    .key(&k)
                    .upload_id(&uid)
                    .part_number(part_number)
                    .content_length(part_len)
                    .body(ByteStream::from(part_data))
                    .send()
                    .await
                    .map_err(|e| {
                        anyhow::anyhow!("UploadPart {} s3://{}/{}: {}", part_number, b, k, e)
                    })?;
                let etag = resp
                    .e_tag()
                    .ok_or_else(|| {
                        anyhow::anyhow!("UploadPart {}: no ETag in response", part_number)
                    })?
                    .to_string();
                debug!(part = part_number, bytes = part_len, "part ok");
                Ok::<_, anyhow::Error>((part_number, etag))
            }));
        }

        // ── 3. Collect or abort ───────────────────────────────────────────────
        let mut completed: Vec<CompletedPart> = Vec::with_capacity(n_parts);
        let mut failed: Option<anyhow::Error> = None;

        for handle in handles {
            match handle
                .await
                .map_err(|e| anyhow::anyhow!("part task panicked: {}", e))
            {
                Err(e) | Ok(Err(e)) => {
                    failed.get_or_insert(e);
                }
                Ok(Ok((pn, etag))) => {
                    completed.push(CompletedPart::builder().part_number(pn).e_tag(etag).build());
                }
            }
        }

        if let Some(err) = failed {
            let _ = self
                .inner
                .abort_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(&upload_id)
                .send()
                .await;
            return Err(err);
        }

        // ── 4. Complete ───────────────────────────────────────────────────────
        completed.sort_by_key(|p| p.part_number().unwrap_or(0));

        self.inner
            .complete_multipart_upload()
            .bucket(bucket)
            .key(key)
            .upload_id(&upload_id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(completed))
                    .build(),
            )
            .send()
            .await
            .map_err(|e| {
                anyhow::anyhow!("CompleteMultipartUpload s3://{}/{}: {}", bucket, key, e)
            })?;

        info!(
            bucket,
            key,
            bytes = total,
            parts = n_parts,
            "multipart upload complete"
        );
        Ok(())
    }

    /// PUT with retry.  Routes to multipart for files > MULTIPART_THRESHOLD.
    pub async fn put_with_retry(&self, bucket: &str, key: &str, data: Bytes) -> Result<()> {
        let use_multipart = data.len() > config::MULTIPART_THRESHOLD;
        let mut last_err = anyhow::anyhow!("no attempts made");

        for attempt in 1..=config::UPLOAD_RETRY_MAX {
            let result = if use_multipart {
                self.put_multipart(bucket, key, data.clone()).await
            } else {
                self.put(bucket, key, data.clone()).await
            };
            match result {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_err = e;
                    let wait = config::UPLOAD_RETRY_BACKOFF * attempt as u64;
                    warn!(
                        bucket,
                        key,
                        attempt,
                        retry_in = wait,
                        use_multipart,
                        "PUT failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_secs(wait)).await;
                }
            }
        }
        bail!(
            "PUT s3://{}/{} failed after {} attempts: {}",
            bucket,
            key,
            config::UPLOAD_RETRY_MAX,
            last_err
        )
    }

    /// PUT a local file with retry, streaming from disk instead of holding the
    /// entire object in memory. This is the preferred path for high-concurrency
    /// retroactive backfills.
    pub async fn put_file_with_retry(&self, bucket: &str, key: &str, path: &Path) -> Result<()> {
        let len = std::fs::metadata(path)
            .map_err(|e| anyhow::anyhow!("stat {} for PUT: {}", path.display(), e))?
            .len();
        let use_multipart = len as usize > config::MULTIPART_THRESHOLD;
        let mut last_err = anyhow::anyhow!("no attempts made");

        for attempt in 1..=config::UPLOAD_RETRY_MAX {
            let result = if use_multipart {
                self.put_file_multipart(bucket, key, path, len).await
            } else {
                self.put_file(bucket, key, path, len).await
            };
            match result {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_err = e;
                    let wait = config::UPLOAD_RETRY_BACKOFF * attempt as u64;
                    warn!(
                        bucket,
                        key,
                        path = %path.display(),
                        attempt,
                        retry_in = wait,
                        use_multipart,
                        error = %last_err,
                        "file PUT failed, retrying"
                    );
                    tokio::time::sleep(Duration::from_secs(wait)).await;
                }
            }
        }
        bail!(
            "PUT s3://{}/{} from {} failed after {} attempts: {}",
            bucket,
            key,
            path.display(),
            config::UPLOAD_RETRY_MAX,
            last_err
        )
    }

    /// HEAD → `Some(content_length)`, or `None` if object doesn't exist.
    pub async fn head(&self, bucket: &str, key: &str) -> Option<u64> {
        self.inner
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .ok()
            .and_then(|r| r.content_length().map(|v| v as u64))
    }

    pub async fn exists(&self, bucket: &str, key: &str) -> bool {
        self.head(bucket, key).await.is_some()
    }

    /// Verify object is present and content-length exactly matches `expected`.
    /// Disk eviction depends on this gate, so approximate size checks are not
    /// acceptable for the production corpus lane.
    pub async fn verify(&self, bucket: &str, key: &str, expected: u64) -> bool {
        match self.head(bucket, key).await {
            None => {
                warn!(bucket, key, "verify: object not found on R2");
                false
            }
            Some(r2_len) => {
                if r2_len != expected {
                    warn!(bucket, key, r2_len, expected, "verify: size mismatch");
                    false
                } else {
                    debug!(bucket, key, r2_len, expected, "verify ok");
                    true
                }
            }
        }
    }

    /// Connectivity probe: list ≤1 key from the spike bucket.
    pub async fn probe(&self, bucket: &str) -> Result<()> {
        self.inner
            .list_objects_v2()
            .bucket(bucket)
            .max_keys(1)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("S3 probe (bucket={}): {}", bucket, e))?;
        Ok(())
    }
}

/// Free disk space in GiB at `path` (e.g. "/mnt/storage").
pub fn disk_free_gib(path: &str) -> f64 {
    let cpath = std::ffi::CString::new(path).unwrap();
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(cpath.as_ptr(), &mut st) } == 0 {
        (st.f_bavail as f64 * st.f_frsize as f64) / (1u64 << 30) as f64
    } else {
        f64::MAX
    }
}
