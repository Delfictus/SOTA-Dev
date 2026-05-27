//! Chemistry Module - Molecular Structure Processing
//!
//! This module provides chemistry-aware processing of molecular structures,
//! including protonation (hydrogen addition) for all-atom force fields.
//!
//! # Zero Fallback Policy
//!
//! All chemistry operations produce deterministic, reproducible results.
//! No external dependencies (Python, pdb2pqr, etc.) are used.

pub mod protonation;

pub use protonation::Protonator;
