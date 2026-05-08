//! Output Generators for Cryptic Site Pilot
//!
//! Produces all deliverables required for pharmaceutical pilots:
//! - Multi-MODEL PDB trajectories
//! - HTML executive reports
//! - CSV data files
//! - Contact residue lists

pub mod csv_outputs;
pub mod html_report;
pub mod pdb_writer;

pub use csv_outputs::{write_contacts_csv, write_rmsf_csv};
pub use html_report::ReportGenerator;
pub use pdb_writer::MultiModelPdbWriter;
