//! Compatibility data-acquisition API for benchmark CLIs.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSource {
    Atlas,
    Nmr,
    Misato,
}

#[derive(Debug, Clone)]
pub struct NmrAcquisitionConfig {
    pub min_models: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DataAvailability {
    pub atlas_targets: usize,
    pub atlas_complete: bool,
    pub nmr_ensembles: usize,
    pub nmr_complete: bool,
    pub misato_complexes: usize,
    pub misato_complete: bool,
}

pub struct DataAcquisition {
    data_dir: PathBuf,
    pub nmr_config: NmrAcquisitionConfig,
}

impl DataAcquisition {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            nmr_config: NmrAcquisitionConfig { min_models: 5 },
        }
    }

    pub fn check_available_data(&self) -> DataAvailability {
        DataAvailability {
            atlas_targets: count_files(self.data_dir.join("atlas"), "pdb"),
            atlas_complete: self.data_dir.join("atlas").exists(),
            nmr_ensembles: count_files(self.data_dir.join("nmr"), "pdb"),
            nmr_complete: self.data_dir.join("nmr").exists(),
            misato_complexes: count_files(self.data_dir.join("misato"), "pdb"),
            misato_complete: self.data_dir.join("misato").exists(),
        }
    }

    pub fn atlas_dir(&self) -> PathBuf {
        self.data_dir.join("atlas")
    }

    pub fn nmr_dir(&self) -> PathBuf {
        self.data_dir.join("nmr")
    }

    pub fn misato_dir(&self) -> PathBuf {
        self.data_dir.join("misato")
    }

    pub async fn download_all(&mut self) -> Result<()> {
        self.download_atlas().await?;
        self.download_nmr_ensembles().await?;
        self.download_misato().await
    }

    pub async fn download_atlas(&mut self) -> Result<()> {
        self.prepare_source_dir(DataSource::Atlas)
    }

    pub async fn download_nmr_ensembles(&mut self) -> Result<()> {
        self.prepare_source_dir(DataSource::Nmr)
    }

    pub async fn download_misato(&mut self) -> Result<()> {
        self.prepare_source_dir(DataSource::Misato)
    }

    fn prepare_source_dir(&self, source: DataSource) -> Result<()> {
        fs::create_dir_all(self.data_dir.join(match source {
            DataSource::Atlas => "atlas",
            DataSource::Nmr => "nmr",
            DataSource::Misato => "misato",
        }))?;
        Ok(())
    }
}

fn count_files(dir: PathBuf, extension: &str) -> usize {
    fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case(extension))
                .unwrap_or(false)
        })
        .count()
}
