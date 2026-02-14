//! Output Generators for Cryptic Site Pilot
//!
//! Produces all deliverables required for pharmaceutical pilots:
//! - Multi-MODEL PDB trajectories
//! - HTML executive reports
//! - CSV data files
//! - Contact residue lists

pub mod pdb_writer;
pub mod html_report;
pub mod csv_outputs;

pub use pdb_writer::MultiModelPdbWriter;
pub use html_report::ReportGenerator;
pub use csv_outputs::{write_rmsf_csv, write_contacts_csv};
