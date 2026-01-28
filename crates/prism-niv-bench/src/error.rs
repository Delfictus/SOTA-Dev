//! Error types for NiV-Bench

use std::fmt;

#[derive(Debug)]
pub enum NivBenchError {
    Io(std::io::Error),
    Reqwest(reqwest::Error),
    Serde(serde_json::Error),
    Parse(String),
    Gpu(String),
    InvalidStructure(String),
    InvalidEpitope(String),
    Environment(String),
}

impl fmt::Display for NivBenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Reqwest(e) => write!(f, "HTTP request error: {}", e),
            Self::Serde(e) => write!(f, "JSON serialization error: {}", e),
            Self::Parse(msg) => write!(f, "Parse error: {}", msg),
            Self::Gpu(msg) => write!(f, "GPU error: {}", msg),
            Self::InvalidStructure(msg) => write!(f, "Invalid structure: {}", msg),
            Self::InvalidEpitope(msg) => write!(f, "Invalid epitope: {}", msg),
            Self::Environment(msg) => write!(f, "Environment error: {}", msg),
        }
    }
}

impl std::error::Error for NivBenchError {}

impl From<std::io::Error> for NivBenchError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<reqwest::Error> for NivBenchError {
    fn from(err: reqwest::Error) -> Self {
        Self::Reqwest(err)
    }
}

impl From<serde_json::Error> for NivBenchError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serde(err)
    }
}

impl From<anyhow::Error> for NivBenchError {
    fn from(err: anyhow::Error) -> Self {
        Self::Parse(err.to_string())
    }
}