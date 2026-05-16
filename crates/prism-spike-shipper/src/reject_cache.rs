use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::UNIX_EPOCH,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileIdentity {
    pub size: u64,
    pub mtime_ns: u128,
}

impl FileIdentity {
    pub fn from_path(path: &Path) -> Result<Self> {
        let meta = fs::metadata(path)?;
        let modified = meta.modified()?.duration_since(UNIX_EPOCH)?;
        Ok(Self {
            size: meta.len(),
            mtime_ns: modified.as_nanos(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RejectRecord {
    ts: String,
    path: String,
    size: u64,
    mtime_ns: u128,
    reason: String,
}

pub struct RejectCache {
    path: PathBuf,
    entries: Mutex<HashMap<String, RejectRecord>>,
}

impl RejectCache {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let mut entries = HashMap::new();
        if let Ok(file) = fs::File::open(&path) {
            for line in BufReader::new(file).lines().map_while(|line| line.ok()) {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(record) = serde_json::from_str::<RejectRecord>(&line) {
                    entries.insert(record.path.clone(), record);
                }
            }
        }
        Ok(Self {
            path,
            entries: Mutex::new(entries),
        })
    }

    pub fn unchanged_reason(&self, path: &Path, identity: &FileIdentity) -> Option<String> {
        let key = path.to_string_lossy();
        let entries = self.entries.lock().ok()?;
        let record = entries.get(key.as_ref())?;
        if record.size == identity.size && record.mtime_ns == identity.mtime_ns {
            Some(record.reason.clone())
        } else {
            None
        }
    }

    pub fn record(&self, path: &Path, identity: &FileIdentity, reason: &str) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let record = RejectRecord {
            ts: Utc::now().to_rfc3339(),
            path: path.to_string_lossy().into_owned(),
            size: identity.size,
            mtime_ns: identity.mtime_ns,
            reason: reason.to_string(),
        };

        let mut line = serde_json::to_string(&record)?;
        line.push('\n');

        let mut entries = self.entries.lock().expect("reject cache mutex poisoned");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())?;
        entries.insert(record.path.clone(), record);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries
            .lock()
            .map(|entries| entries.len())
            .unwrap_or(0)
    }

    pub fn unchanged_reason_contains(
        &self,
        path: &Path,
        identity: &FileIdentity,
        needle: &str,
    ) -> bool {
        self.unchanged_reason(path, identity)
            .map(|reason| reason.contains(needle))
            .unwrap_or(false)
    }
}
