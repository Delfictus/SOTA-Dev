//! Data loading and processing for PRISM-VE
//!
//! Handles:
//! - DMS escape data loading
//! - GISAID frequency data
//! - Variant mutation data
//! - Population immunity landscapes

pub mod loaders;

pub use loaders::{
    load_dms_escape_data,
    load_gisaid_frequencies,
    load_variant_mutations,
};
