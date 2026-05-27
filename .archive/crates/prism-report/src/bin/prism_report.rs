//! PRISM4D Report Generator CLI (DEPRECATED)
//!
//! This binary is DEPRECATED. Use `prism4d` instead:
//!
//!   # Run complete pipeline (engine + finalize):
//!   prism4d run --topology prep.json --pdb input.pdb --out results/
//!
//!   # Run finalize stage only (from existing events.jsonl):
//!   prism4d finalize --events events.jsonl --pdb input.pdb --out results/
//!
//! The `prism-report` binary was the old standalone report generator.
//! It has been superseded by `prism4d` which provides:
//!   - Unified engine + report pipeline
//!   - Proper ablation analysis (baseline/cryo/cryo+UV)
//!   - Mandatory events.jsonl validation (no synthetic fallback)
//!   - Automatic FinalizeStage execution after engine

use anyhow::{bail, Result};

fn main() -> Result<()> {
    eprintln!("╔══════════════════════════════════════════════════════════════════════════════╗");
    eprintln!("║                                                                              ║");
    eprintln!("║  DEPRECATED: prism-report has been replaced by prism4d                       ║");
    eprintln!("║                                                                              ║");
    eprintln!("║  Use instead:                                                                ║");
    eprintln!("║                                                                              ║");
    eprintln!("║    # Complete pipeline (engine + report):                                    ║");
    eprintln!("║    prism4d run --topology prep.json --pdb input.pdb --out results/           ║");
    eprintln!("║                                                                              ║");
    eprintln!("║    # Report only (from existing events.jsonl):                               ║");
    eprintln!("║    prism4d finalize --events events.jsonl --pdb input.pdb --out results/     ║");
    eprintln!("║                                                                              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════════════════╝");
    eprintln!();

    bail!(
        "prism-report is deprecated.\n\n\
         The prism-report binary has been replaced by prism4d.\n\
         prism4d provides a unified pipeline with:\n\
         - Real GPU-accelerated MD engine (no fabricated data)\n\
         - Mandatory ablation analysis (baseline/cryo/cryo+UV)\n\
         - Automatic FinalizeStage after engine completion\n\
         - Hard-fail on missing events.jsonl (no fallback)\n\n\
         Run 'prism4d --help' for usage information."
    );
}
